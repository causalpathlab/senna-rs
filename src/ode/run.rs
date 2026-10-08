//! `senna ode`: read a finished base fit and a two-track count backend,
//! group the cells into multilevel pseudobulks by their base states, and fit
//! the module kinetics and a time per pseudobulk coarse to fine.
//!
//! The coarsest level is fitted from the diffusion order of its pseudobulks,
//! both orientations; each finer level starts from its parents' times and the
//! coarser kinetics, keeps the orientation, and is pulled toward its parents
//! ([`TauPrior::anchor`]). Every level shares evidence between neighbouring
//! pseudobulks ([`TauPrior::graph`]).

use super::args::OdeArgs;
use super::counts::{pb_tracks, select_genes, PbTracks};
use super::fit::{fit, refine, FitConfig, FitResult, PbCounts, TauPrior};
use super::init::{diffusion_order, knn_edges};
use super::kinetics::ModuleKinetics;
use super::levels::{parents, pb_levels};
use super::modules::gene_modules;
use data_beans::sparse_io::open_sparse_matrix_by_path;
use legume_numeric::candle::candle_core::Device;
use legume_numeric::matrix::parquet::{write_named_table, Column};
use legume_numeric::matrix::traits::{ConvertMatOps, IoOps};
use log::info;
use nalgebra::DMatrix;
use std::collections::HashMap;

/// One fitted level: its pseudobulks' names, reads, starting times and fit.
struct Level {
    pbs: Vec<Box<str>>,
    tracks: PbTracks,
    start: Vec<f32>,
    out: FitResult,
}

pub fn run_ode(args: &OdeArgs) -> anyhow::Result<()> {
    let dev = args.device.to_device(args.device_no)?;

    // Cells' base states in the backend's column order, and their levels.
    let cells = open_sparse_matrix_by_path(&args.tracks)?.column_names()?;
    let embedding = DMatrix::<f32>::from_parquet_with_row_names(
        &format!("{}.cell_embedding.parquet", args.base),
        Some(0),
    )?;
    let row_of: HashMap<&str, usize> = embedding
        .rows
        .iter()
        .enumerate()
        .map(|(i, r)| (r.as_ref(), i))
        .collect();
    let state: Vec<Option<usize>> = cells
        .iter()
        .map(|c| row_of.get(c.as_ref()).copied())
        .collect();
    let has_state: Vec<bool> = state.iter().map(Option::is_some).collect();
    let h = embedding.mat.ncols();
    let states = DMatrix::<f32>::from_fn(h, cells.len(), |k, c| {
        state[c].map_or(0.0, |r| embedding.mat[(r, k)])
    });
    let levels = pb_levels(&args.tracks, &states, &has_state, args.levels)?;

    // Every level's reads, and its pseudobulks' mean states.
    let mut per_level: Vec<(Vec<Box<str>>, PbTracks, DMatrix<f32>)> = Vec::new();
    for (l, member) in levels.iter().enumerate() {
        let n_pb = member.iter().flatten().max().map_or(0, |&m| m + 1);
        let pbs: Vec<Box<str>> = (0..n_pb).map(|i| format!("l{l}:pb{i}").into()).collect();
        let membership: HashMap<Box<str>, Box<str>> = cells
            .iter()
            .zip(member)
            .filter_map(|(c, p)| p.map(|p| (c.clone(), pbs[p].clone())))
            .collect();
        let tracks = pb_tracks(&args.tracks, &membership, &pbs)?;
        let mut theta = DMatrix::<f32>::zeros(n_pb, h);
        for (c, p) in member.iter().enumerate() {
            if let (Some(p), Some(r)) = (p, state[c]) {
                for k in 0..h {
                    theta[(*p, k)] += embedding.mat[(r, k)] / tracks.cells[*p];
                }
            }
        }
        per_level.push((pbs, tracks, theta));
    }

    // The genes to fit, chosen on the finest level: enough reads, most
    // variable, in a module of at least `min_module_genes`.
    let (_, finest, _) = per_level.last().expect("at least one level");
    let chosen = select_genes(finest, args.min_reads, args.max_genes);
    let totals = DMatrix::<f32>::from_fn(chosen.len(), finest.cells.len(), |i, p| {
        finest.unspliced[(p, chosen[i])] + finest.spliced[(p, chosen[i])]
    });
    let (module_of, n_modules) = gene_modules(
        &totals,
        &finest.cells,
        args.modules,
        args.min_module_genes,
        args.seed,
    )?;
    let fitted: Vec<usize> = (0..chosen.len())
        .filter(|&i| module_of[i].is_some())
        .collect();
    anyhow::ensure!(!fitted.is_empty() && n_modules > 0, "no gene left to fit");
    let genes: Vec<usize> = fitted.iter().map(|&i| chosen[i]).collect();
    let module_of_gene: Vec<u32> = fitted
        .iter()
        .map(|&i| module_of[i].expect("fitted"))
        .collect();
    info!(
        "ode: {} levels of {:?} pseudobulks; {} genes with both tracks, {} selected, {} in \
         {n_modules} modules",
        per_level.len(),
        per_level.iter().map(|l| l.0.len()).collect::<Vec<_>>(),
        finest.genes.len(),
        chosen.len(),
        genes.len(),
    );

    let cfg = FitConfig {
        rounds: args.rounds,
        adam_steps: args.adam_steps,
        learning_rate: args.learning_rate,
        grid: args.grid,
    };
    let mut fitted_levels: Vec<Level> = Vec::with_capacity(per_level.len());
    for (l, (pbs, tracks, theta)) in per_level.into_iter().enumerate() {
        let counts = level_counts(&tracks, &genes, &module_of_gene, n_modules, &dev)?;
        let graph = (args.smooth > 0.0).then(|| (args.smooth, knn_edges(&theta, args.knn)));
        let (start, out) = match fitted_levels.last() {
            None => {
                let start = diffusion_order(&theta, args.knn);
                let prior = TauPrior {
                    anchor: None,
                    graph,
                };
                let out = fit(&counts, &start, &prior, &cfg)?;
                info!(
                    "ode level {l}: loss per read {:.5} (starting order) vs {:.5} \
                     (reversed); kept the {}",
                    out.orientation_loss[0],
                    out.orientation_loss[1],
                    if out.reversed {
                        "reverse"
                    } else {
                        "starting order"
                    },
                );
                (start, out)
            }
            Some(coarse) => {
                let parent = parents(&levels[l], &levels[l - 1]);
                let start: Vec<f32> = parent.iter().map(|&q| coarse.out.tau[q]).collect();
                let prior = TauPrior {
                    anchor: (args.parent_pull > 0.0).then(|| (args.parent_pull, start.clone())),
                    graph,
                };
                let out = refine(
                    &counts,
                    &start,
                    &coarse.out.modules,
                    &coarse.out.rates,
                    &prior,
                    &cfg,
                )?;
                info!(
                    "ode level {l}: loss per read {:.5} after {} rounds",
                    out.loss.last().copied().unwrap_or(f64::NAN),
                    cfg.rounds
                );
                (start, out)
            }
        };
        fitted_levels.push(Level {
            pbs,
            tracks,
            start,
            out,
        });
    }
    write_outputs(
        &args.out,
        &cells,
        &levels,
        &fitted_levels,
        &genes,
        &module_of_gene,
    )
}

/// A level's reads on the fitted genes, on `dev`.
fn level_counts(
    tracks: &PbTracks,
    genes: &[usize],
    module_of_gene: &[u32],
    n_modules: usize,
    dev: &Device,
) -> anyhow::Result<PbCounts> {
    let p = tracks.cells.len();
    let column =
        |m: &DMatrix<f32>| DMatrix::<f32>::from_fn(p, genes.len(), |i, c| m[(i, genes[c])]);
    Ok(PbCounts {
        unspliced: column(&tracks.unspliced).to_tensor(dev)?.contiguous()?,
        spliced: column(&tracks.spliced).to_tensor(dev)?.contiguous()?,
        module_of_gene: module_of_gene.to_vec(),
        n_modules,
    })
}

/// `{out}.pb_ode.parquet` (every level's pseudobulks: level, τ, starting τ,
/// cells), `{out}.cell_pb.parquet` (each cell's pseudobulk per level), and
/// the finest level's `{out}.module_kinetics.parquet` and
/// `{out}.gene_ode.parquet` (module, rates, offset, loading).
fn write_outputs(
    prefix: &str,
    cells: &[Box<str>],
    levels: &[Vec<Option<usize>>],
    fitted: &[Level],
    genes: &[usize],
    module_of_gene: &[u32],
) -> anyhow::Result<()> {
    let mut rows: Vec<Box<str>> = Vec::new();
    let mut level_col: Vec<i32> = Vec::new();
    let mut cols: Vec<Vec<f32>> = vec![Vec::new(); 3];
    for (l, lv) in fitted.iter().enumerate() {
        rows.extend(lv.pbs.iter().cloned());
        level_col.extend(std::iter::repeat_n(l as i32, lv.pbs.len()));
        cols[0].extend(&lv.out.tau);
        cols[1].extend(&lv.start);
        cols[2].extend(&lv.tracks.cells);
    }
    let names = ["tau", "tau_start", "cells"];
    let mut table: Vec<(Box<str>, Column)> = vec![("level".into(), Column::I32(&level_col))];
    table.extend(
        names
            .iter()
            .map(|&n| Box::from(n))
            .zip(cols.iter().map(|v| Column::F32(v))),
    );
    write_named_table(&format!("{prefix}.pb_ode.parquet"), "pb", &rows, &table)?;

    let cell_pb: Vec<Vec<Box<str>>> = levels
        .iter()
        .zip(fitted)
        .map(|(member, lv)| {
            member
                .iter()
                .map(|p| p.map_or_else(|| Box::from(""), |p| lv.pbs[p].clone()))
                .collect()
        })
        .collect();
    let level_names: Vec<Box<str>> = (0..levels.len()).map(|l| format!("l{l}").into()).collect();
    let cell_table: Vec<(Box<str>, Column)> = level_names
        .iter()
        .cloned()
        .zip(cell_pb.iter().map(|v| Column::Str(v)))
        .collect();
    write_named_table(
        &format!("{prefix}.cell_pb.parquet"),
        "cell",
        cells,
        &cell_table,
    )?;

    let finest = fitted.last().expect("a level");
    write_kinetics(prefix, &finest.out.modules, module_of_gene)?;
    let gene_names: Vec<Box<str>> = genes
        .iter()
        .map(|&j| finest.tracks.genes[j].clone())
        .collect();
    let module: Vec<i32> = module_of_gene.iter().map(|&m| m as i32).collect();
    let beta: Vec<f32> = finest.out.rates.iter().map(|r| r.beta).collect();
    let gamma: Vec<f32> = finest.out.rates.iter().map(|r| r.gamma).collect();
    write_named_table(
        &format!("{prefix}.gene_ode.parquet"),
        "gene",
        &gene_names,
        &[
            ("module".into(), Column::I32(&module)),
            ("beta".into(), Column::F32(&beta)),
            ("gamma".into(), Column::F32(&gamma)),
            ("offset".into(), Column::F32(&finest.out.offset)),
            ("log_load".into(), Column::F32(&finest.out.loading)),
        ],
    )?;
    info!("Wrote {prefix}.{{pb_ode,cell_pb,module_kinetics,gene_ode}}.parquet");
    Ok(())
}

/// `{prefix}.module_kinetics.parquet`: every module's transcription, its
/// genes' geometric-mean rates and its number of genes.
fn write_kinetics(
    prefix: &str,
    modules: &[ModuleKinetics],
    module_of_gene: &[u32],
) -> anyhow::Result<()> {
    let mut n_genes = vec![0i32; modules.len()];
    for &m in module_of_gene {
        n_genes[m as usize] += 1;
    }
    let col = |f: fn(&ModuleKinetics) -> f32| modules.iter().map(f).collect::<Vec<f32>>();
    let t_off: Vec<f32> = modules.iter().map(|k| k.t_on + k.duration).collect();
    let names: Vec<Box<str>> = (0..modules.len()).map(|i| format!("m{i}").into()).collect();
    write_named_table(
        &format!("{prefix}.module_kinetics.parquet"),
        "module",
        &names,
        &[
            ("t_on".into(), Column::F32(&col(|k| k.t_on))),
            ("t_off".into(), Column::F32(&t_off)),
            ("lambda".into(), Column::F32(&col(|k| k.lambda))),
            ("basal".into(), Column::F32(&col(|k| k.basal))),
            ("beta".into(), Column::F32(&col(|k| k.beta))),
            ("gamma".into(), Column::F32(&col(|k| k.gamma))),
            ("genes".into(), Column::I32(&n_genes)),
        ],
    )
}
