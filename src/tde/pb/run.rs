//! `senna tde-pb`: read a finished base fit and a two-track count backend,
//! group the cells into multilevel pseudobulks by their base states, and fit
//! the module kinetics and a time per pseudobulk coarse to fine; write them
//! with each pseudobulk's velocity in the base space.
//!
//! Every level is fitted with the joint model ([`super::joint`]): a unit's
//! state is a function of its time, `θ(τ) = W φ(τ)`, and the gene rows `ρ`
//! share its space. The coarsest level is fitted from the diffusion order of
//! its pseudobulks, both orientations, the joint model deciding; each finer
//! level starts from its parents' times and the coarser fit (kinetics, `ρ`,
//! biases), keeps the orientation, and is pulled toward its parents
//! ([`TauPrior::anchor`]). Every level shares evidence between neighbouring
//! pseudobulks ([`TauPrior::graph`]).

use super::args::TdePbArgs;
use super::counts::{pb_tracks, select_genes, PbTracks};
use super::fit::{fit, refine, FitConfig, FitResult, PbCounts, TauPrior};
use super::init::{diffusion_order, knn_edges};
use super::joint::{fit_dynamic, fit_joint, JointConfig, JointInit, JointResult};
use super::kinetics::{curves, Kinetics, ModuleKinetics};
use super::levels::{parents, pb_levels};
use super::modules::gene_modules;
use data_beans::sparse_io::open_sparse_matrix_by_path;
use legume_numeric::candle::candle_core::{DType, Device, Tensor};
use legume_numeric::matrix::parquet::{write_named_table, Column};
use legume_numeric::matrix::traits::{ConvertMatOps, IoOps};
use log::info;
use nalgebra::DMatrix;
use std::collections::HashMap;

/// A parquet table's rows by name.
fn named_rows(path: &str) -> anyhow::Result<(Vec<Box<str>>, DMatrix<f32>)> {
    let t = DMatrix::<f32>::from_parquet_with_row_names(path, Some(0))?;
    Ok((t.rows, t.mat))
}

/// Rows `names` of `(rows, mat)`, `None` for a name it does not have.
fn pick(rows: &[Box<str>], mat: &DMatrix<f32>, names: &[Box<str>]) -> Vec<Option<Vec<f32>>> {
    let index: HashMap<&str, usize> = rows
        .iter()
        .enumerate()
        .map(|(i, r)| (r.as_ref(), i))
        .collect();
    names
        .iter()
        .map(|n| {
            index
                .get(n.as_ref())
                .map(|&i| mat.row(i).iter().copied().collect())
        })
        .collect()
}

/// One level: its pseudobulks' names, reads, starting times and joint fit.
struct Level {
    pbs: Vec<Box<str>>,
    tracks: PbTracks,
    start: Vec<f32>,
    out: JointResult,
}

pub fn run_tde_pb(args: &TdePbArgs) -> anyhow::Result<()> {
    let base = &args.base;
    let dev = args.device.to_device(args.device_no)?;

    // Cells' base states in the backend's column order, and their levels.
    let cells = open_sparse_matrix_by_path(&args.tracks)?.column_names()?;
    let (cell_rows, cell_mat) = named_rows(&format!("{base}.cell_embedding.parquet"))?;
    let h = cell_mat.ncols();
    let state = pick(&cell_rows, &cell_mat, &cells);
    let has_state: Vec<bool> = state.iter().map(Option::is_some).collect();
    let states = DMatrix::<f32>::from_fn(h, cells.len(), |k, c| {
        state[c].as_ref().map_or(0.0, |s| s[k])
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
            if let (Some(p), Some(s)) = (p, &state[c]) {
                for k in 0..h {
                    theta[(*p, k)] += s[k] / tracks.cells[*p];
                }
            }
        }
        per_level.push((pbs, tracks, theta));
    }

    // The genes to fit, chosen on the finest level: enough reads, known to
    // the base fit, most variable, in a module.
    let (_, finest, _) = per_level.last().expect("at least one level");
    let (rho_rows, rho_mat) = named_rows(&format!("{base}.feature_embedding.parquet"))?;
    let (bias_rows, bias_mat) = named_rows(&format!("{base}.feature_bias.parquet"))?;
    let chosen = select_genes(finest, args.min_reads, args.max_genes);
    let names: Vec<Box<str>> = chosen.iter().map(|&j| finest.genes[j].clone()).collect();
    let rho = pick(&rho_rows, &rho_mat, &names);
    let bias = pick(&bias_rows, &bias_mat, &names);
    let known: Vec<usize> = (0..chosen.len())
        .filter(|&i| rho[i].is_some() && bias[i].is_some())
        .collect();
    let totals = DMatrix::<f32>::from_fn(known.len(), finest.cells.len(), |i, p| {
        let j = chosen[known[i]];
        finest.unspliced[(p, j)] + finest.spliced[(p, j)]
    });
    let (module_of, n_modules) = gene_modules(&totals, &finest.cells, args.modules, args.seed)?;
    let fitted: Vec<usize> = (0..known.len())
        .filter(|&i| module_of[i].is_some())
        .collect();
    anyhow::ensure!(!fitted.is_empty() && n_modules > 0, "no gene left to fit");
    let genes: Vec<usize> = fitted.iter().map(|&i| chosen[known[i]]).collect();
    let module_of_gene: Vec<u32> = fitted
        .iter()
        .map(|&i| module_of[i].expect("fitted"))
        .collect();
    let g = genes.len();
    info!(
        "tde-pb: {} levels of {:?} pseudobulks; {} genes with both tracks, {} selected, {g} in \
         {n_modules} modules",
        per_level.len(),
        per_level.iter().map(|l| l.0.len()).collect::<Vec<_>>(),
        finest.genes.len(),
        chosen.len(),
    );
    let rho_fit = DMatrix::<f32>::from_fn(g, h, |i, c| {
        rho[known[fitted[i]]].as_ref().expect("known")[c]
    });
    let bias_fit: Vec<f32> = (0..g)
        .map(|i| bias[known[fitted[i]]].as_ref().expect("known")[0])
        .collect();

    let stage1 = FitConfig {
        rounds: args.rounds,
        adam_steps: args.adam_steps,
        learning_rate: args.learning_rate,
        grid: args.grid,
    };
    if args.no_embedding {
        return stage1_only(
            args,
            &cells,
            &levels,
            per_level,
            &genes,
            &module_of_gene,
            n_modules,
            &stage1,
            &dev,
        );
    }
    let joint = JointConfig {
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
                let (out, losses, reversed) = fit_dynamic(
                    &counts, &start, &rho_fit, &bias_fit, &theta, &prior, &stage1, &joint,
                )?;
                info!(
                    "tde-pb level {l}: loss per read {:.5} (starting order) vs {:.5} \
                     (reversed); kept the {}",
                    losses[0],
                    losses[1],
                    if reversed {
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
                let s1 = refine(&counts, &start, &coarse.out.modules, &prior, &stage1)?;
                let init = JointInit {
                    tau: &s1.tau,
                    modules: &s1.modules,
                    rho: &coarse.out.rho,
                    bias: &coarse.out.bias,
                    theta: &theta,
                };
                let out = fit_joint(&counts, &init, &prior, &joint)?;
                info!(
                    "tde-pb level {l}: loss per read {:.5} after {} rounds",
                    out.loss.last().copied().unwrap_or(f64::NAN),
                    joint.rounds
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

    let finest = fitted_levels.last().expect("a level");
    let gene_names: Vec<Box<str>> = genes
        .iter()
        .map(|&j| finest.tracks.genes[j].clone())
        .collect();
    let peak = peak_times(&finest.out, &dev)?;
    write_outputs(
        &args.out,
        &cells,
        &levels,
        &fitted_levels,
        &gene_names,
        &module_of_gene,
        &peak,
    )
}

/// Points of the grid each gene's peak time is read on.
const PEAK_GRID: usize = 101;

/// Every gene's peak time: the `τ` at which its spliced score
/// `⟨θ^s(τ), ρ_g⟩` is highest.
fn peak_times(out: &JointResult, dev: &Device) -> anyhow::Result<Vec<f32>> {
    let grid: Vec<f32> = (0..PEAK_GRID)
        .map(|i| i as f32 / (PEAK_GRID - 1) as f32)
        .collect();
    let k = Kinetics::from_modules(&out.modules, DType::F32, dev)?;
    let (_, s) = curves(&k, &Tensor::from_slice(&grid, PEAK_GRID, dev)?)?;
    let follow = out.directions.clone() * out.rho.transpose(); // [M × G]
    let score = DMatrix::<f32>::from_tensor(&s.log()?)? * follow; // [K × G]
    Ok((0..score.ncols())
        .map(|j| grid[score.column(j).imax()])
        .collect())
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

/// `{out}.pb_ode.parquet` (every level's pseudobulks: time, dynamic state
/// `h*` and velocity `v*`), `{out}.cell_pb.parquet` (each cell's pseudobulk
/// per level), and the finest level's `{out}.feature_ode.parquet` (gene rows
/// `h*`, module, peak time), `{out}.module_kinetics.parquet` and
/// `{out}.module_directions.parquet`.
fn write_outputs(
    prefix: &str,
    cells: &[Box<str>],
    levels: &[Vec<Option<usize>>],
    fitted: &[Level],
    gene_names: &[Box<str>],
    module_of_gene: &[u32],
    peak: &[f32],
) -> anyhow::Result<()> {
    let h = fitted[0].out.theta.ncols();
    let mut rows: Vec<Box<str>> = Vec::new();
    let mut level_col: Vec<i32> = Vec::new();
    let mut cols: Vec<Vec<f32>> = vec![Vec::new(); 4 + 2 * h];
    for (l, lv) in fitted.iter().enumerate() {
        rows.extend(lv.pbs.iter().cloned());
        level_col.extend(std::iter::repeat_n(l as i32, lv.pbs.len()));
        cols[0].extend(&lv.out.tau);
        cols[1].extend(&lv.start);
        cols[2].extend(&lv.out.kappa);
        cols[3].extend(&lv.tracks.cells);
        for c in 0..h {
            cols[4 + c].extend(lv.out.theta.column(c).iter());
            cols[4 + h + c].extend(lv.out.velocity.column(c).iter());
        }
    }
    let mut names: Vec<Box<str>> = ["tau", "tau_start", "kappa", "cells"]
        .into_iter()
        .map(Box::from)
        .collect();
    names.extend((0..h).map(|c| format!("h{c}").into()));
    names.extend((0..h).map(|c| format!("v{c}").into()));
    let mut table: Vec<(Box<str>, Column)> = vec![("level".into(), Column::I32(&level_col))];
    table.extend(
        names
            .iter()
            .cloned()
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

    let finest = &fitted.last().expect("a level").out;
    let module: Vec<i32> = module_of_gene.iter().map(|&m| m as i32).collect();
    let rho_cols: Vec<Vec<f32>> = (0..h)
        .map(|c| finest.rho.column(c).iter().copied().collect())
        .collect();
    let mut feature: Vec<(Box<str>, Column)> = vec![
        ("module".into(), Column::I32(&module)),
        ("peak_tau".into(), Column::F32(peak)),
        ("bias".into(), Column::F32(&finest.bias)),
        ("bias_u".into(), Column::F32(&finest.bias_u)),
    ];
    let rho_names: Vec<Box<str>> = (0..h).map(|c| format!("h{c}").into()).collect();
    feature.extend(
        rho_names
            .iter()
            .cloned()
            .zip(rho_cols.iter().map(|v| Column::F32(v))),
    );
    write_named_table(
        &format!("{prefix}.feature_ode.parquet"),
        "gene",
        gene_names,
        &feature,
    )?;

    write_kinetics(prefix, &finest.modules, None, module_of_gene)?;
    let m = finest.directions.nrows();
    let module_names: Vec<Box<str>> = (0..m).map(|i| format!("m{i}").into()).collect();
    let dir_cols: Vec<Vec<f32>> = (0..h)
        .map(|c| finest.directions.column(c).iter().copied().collect())
        .collect();
    let dir_table: Vec<(Box<str>, Column)> = rho_names
        .iter()
        .cloned()
        .zip(dir_cols.iter().map(|v| Column::F32(v)))
        .collect();
    write_named_table(
        &format!("{prefix}.module_directions.parquet"),
        "module",
        &module_names,
        &dir_table,
    )?;
    info!(
        "Wrote {prefix}.{{pb_ode,cell_pb,feature_ode,module_kinetics,module_directions}}.parquet"
    );
    Ok(())
}

/// `{prefix}.module_kinetics.parquet`; stage 1's which-module log loadings
/// when given.
fn write_kinetics(
    prefix: &str,
    modules: &[ModuleKinetics],
    loading: Option<&[f32]>,
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
        ]
        .into_iter()
        .chain(loading.map(|l| ("log_load".into(), Column::F32(l))))
        .collect::<Vec<_>>(),
    )
}

/// `--no-embedding`: stage 1 alone at every level, coarse to fine, its
/// direction decided by stage 1; writes `{out}.pb_ode.parquet` (level, τ,
/// starting τ, κ, cells), `{out}.cell_pb.parquet`,
/// `{out}.module_kinetics.parquet` and `{out}.gene_ode.parquet` (module,
/// offset).
#[allow(clippy::too_many_arguments)]
fn stage1_only(
    args: &TdePbArgs,
    cells: &[Box<str>],
    levels: &[Vec<Option<usize>>],
    per_level: Vec<(Vec<Box<str>>, PbTracks, DMatrix<f32>)>,
    genes: &[usize],
    module_of_gene: &[u32],
    n_modules: usize,
    cfg: &FitConfig,
    dev: &Device,
) -> anyhow::Result<()> {
    let mut fits: Vec<(Vec<Box<str>>, PbTracks, Vec<f32>, FitResult)> = Vec::new();
    for (l, (pbs, tracks, theta)) in per_level.into_iter().enumerate() {
        let counts = level_counts(&tracks, genes, module_of_gene, n_modules, dev)?;
        let graph = (args.smooth > 0.0).then(|| (args.smooth, knn_edges(&theta, args.knn)));
        let (start, out) = match fits.last() {
            None => {
                let start = diffusion_order(&theta, args.knn);
                let prior = TauPrior {
                    anchor: None,
                    graph,
                };
                let out = fit(&counts, &start, &prior, cfg)?;
                info!(
                    "tde-pb level {l}: loss per read {:.5} (starting order) vs {:.5} \
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
            Some((_, _, _, coarse)) => {
                let parent = parents(&levels[l], &levels[l - 1]);
                let start: Vec<f32> = parent.iter().map(|&q| coarse.tau[q]).collect();
                let prior = TauPrior {
                    anchor: (args.parent_pull > 0.0).then(|| (args.parent_pull, start.clone())),
                    graph,
                };
                let out = refine(&counts, &start, &coarse.modules, &prior, cfg)?;
                info!(
                    "tde-pb level {l}: loss per read {:.5} after {} rounds",
                    out.loss.last().copied().unwrap_or(f64::NAN),
                    cfg.rounds
                );
                (start, out)
            }
        };
        fits.push((pbs, tracks, start, out));
    }
    let prefix = &args.out;
    let mut rows: Vec<Box<str>> = Vec::new();
    let mut level_col: Vec<i32> = Vec::new();
    let mut cols: Vec<Vec<f32>> = vec![Vec::new(); 4];
    for (l, (pbs, tracks, start, out)) in fits.iter().enumerate() {
        rows.extend(pbs.iter().cloned());
        level_col.extend(std::iter::repeat_n(l as i32, pbs.len()));
        cols[0].extend(&out.tau);
        cols[1].extend(start);
        cols[2].extend(&out.kappa);
        cols[3].extend(&tracks.cells);
    }
    let names = ["tau", "tau_start", "kappa", "cells"];
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
        .zip(&fits)
        .map(|(member, (pbs, ..))| {
            member
                .iter()
                .map(|p| p.map_or_else(|| Box::from(""), |p| pbs[p].clone()))
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
    let (_, tracks, _, finest) = fits.last().expect("a level");
    write_kinetics(
        prefix,
        &finest.modules,
        Some(&finest.loading),
        module_of_gene,
    )?;
    let gene_names: Vec<Box<str>> = genes.iter().map(|&j| tracks.genes[j].clone()).collect();
    let module: Vec<i32> = module_of_gene.iter().map(|&m| m as i32).collect();
    write_named_table(
        &format!("{prefix}.gene_ode.parquet"),
        "gene",
        &gene_names,
        &[
            ("module".into(), Column::I32(&module)),
            ("offset".into(), Column::F32(&finest.offset)),
        ],
    )?;
    info!("Wrote {prefix}.{{pb_ode,cell_pb,module_kinetics,gene_ode}}.parquet");
    Ok(())
}
