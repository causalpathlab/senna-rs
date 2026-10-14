//! `senna td-rbm`: read two-track count backends, collapse their cells into
//! multilevel pseudobulks (the collapse's own rates, batch-adjusted when
//! there are batches), start the fit on the most variable genes (the
//! forest, the curves, the genes' embedding), fit the machine on samples of
//! every level over every gene, with gene modules ([`super::train`]); the
//! roots and directions are called on the most variable genes alone. Every
//! cell is then read once, a block at a time.

use super::args::TdRbmArgs;
use super::counts::{
    for_cell_blocks, gene_columns, pb_tracks, select_genes, two_track_genes, PbTracks,
};
use super::curve::{grid_times, Spline};
use super::model::{GivenRho, Init0, ModulesInit};
use super::start::{base_start, rho_start, row_states, start_along, STATE_DIMS};
use super::train::{fit, Level, Summary, TrainConfig, BETA0, GAMMA0};
use super::tree::Forest;
use data_beans::sparse_io_vector::SparseIoVec;
use legume_numeric::candle::candle_core::{Device, Tensor};
use legume_numeric::matrix::parquet::{write_named_table, Column};
use legume_numeric::param::traits::Inference;
use log::info;
use rand::rngs::StdRng;
use rand::SeedableRng;
use rand_distr::{Binomial, Distribution};
use senna::embed_common::Mat;

/// Columns `0..n` of `rows` (`[N × n]` row-major) as named columns.
fn columns(prefix: &str, rows: &[Vec<f32>], n: usize) -> Vec<(Box<str>, Vec<f32>)> {
    (0..n)
        .map(|j| {
            (
                format!("{prefix}{}", j + 1).into(),
                rows.iter().map(|r| r[j]).collect(),
            )
        })
        .collect()
}

fn write(
    path: &str,
    key: &str,
    names: &[Box<str>],
    cols: &[(Box<str>, Vec<f32>)],
) -> anyhow::Result<()> {
    let cols: Vec<(Box<str>, Column)> = cols
        .iter()
        .map(|(n, v)| (n.clone(), Column::F32(v)))
        .collect();
    write_named_table(path, key, names, &cols)
}

/// Gene modules when neither `--feature-modules` nor `--no-feature-modules`
/// is given (as rbm).
const DEFAULT_GENE_MODULES: usize = 50;

/// The cells as the shared loader reads them, and their pseudobulks.
struct Loaded {
    data: SparseIoVec,
    /// The cells written out (QC's near-empty cells left out), when QC ran.
    keep: Option<Vec<usize>>,
    inputs: Inputs,
    proj_kn: Mat,
    /// Every cell's pseudobulk at every level, coarse to fine.
    cell_to_pb: Vec<Vec<usize>>,
    /// The collapse's levels, coarse to fine.
    collapsed: Vec<data_beans::alg::collapse_data::CollapsedOut>,
    pb_tree: Option<data_beans::alg::collapse_data::PbTree>,
}

/// The files a run read, as its manifest records them.
struct Inputs {
    data_files: Vec<Box<str>>,
    batch_files: Option<Vec<Box<str>>>,
}

/// Read the data as every senna fit does (files from the command line or
/// `--from`, batches, QC, the multilevel collapse or a given partition) on
/// the genes every file has.
fn load(args: &TdRbmArgs) -> anyhow::Result<Loaded> {
    use senna::run_manifest::InheritedFromManifest;
    let inherited = args
        .from
        .as_deref()
        .map(senna::run_manifest::inherit_data_from)
        .transpose()?;
    InheritedFromManifest::ensure_replayable(inherited.as_ref(), "td-rbm")?;
    let data_files = InheritedFromManifest::resolve_data(inherited.as_ref(), &args.data_files)?;
    let batch_files =
        InheritedFromManifest::resolve_batch(inherited.as_ref(), args.batch_files.as_deref());
    // Before the collapse, as a file's header says it: every file has genes
    // with both tracks.
    for file in &data_files {
        let rows = data_beans::sparse_io::open_sparse_matrix_by_path(file)?.row_names()?;
        anyhow::ensure!(
            two_track_genes(&rows).0 > 0,
            "{file}: no gene has both `/count/spliced` and `/count/unspliced` rows"
        );
    }
    let prebuilt_partition = args.pb_from.partition(
        inherited.as_ref(),
        &crate::refine_weighting::partition_settings(args),
    )?;
    let hvg = crate::hvg::resolve_multiome_with_hvg(false, data_files.len(), &args.hvg);
    let crate::topic::common::PreparedData {
        data_vec,
        collapsed_levels,
        proj_kn,
        cell_to_pb_per_level,
        pb_tree,
        output_keep_idx,
    } = crate::topic::common::load_and_collapse(&crate::topic::common::LoadCollapseArgs {
        data_files: &data_files,
        batch_files: &batch_files,
        preload: args.preload_data,
        proj_dim: args.collapse.proj_dim,
        sort_dim: args.collapse.sort_dim,
        knn_cells: args.collapse.knn_cells,
        num_levels: args.collapse.num_levels,
        iter_opt: args.collapse.iter_opt,
        block_size: Some(args.block_size),
        out: &args.out,
        max_features: hvg.n_hvg,
        feature_list_file: hvg.feature_list_file,
        must_train_file: hvg.must_train_file,
        refine: args.collapse.pb_refine.to_params(),
        pb_tree: args.collapse.pb_tree_params(),
        ignore_batch: args.collapse.ignore_batch,
        qc: args.qc.to_config(),
        qc_block_size: Some(args.block_size),
        qc_report_out: args.qc.qc_report.as_deref(),
        feature_mask_fn: None,
        pb_reference: None,
        mixture_batches: args.collapse.mixture_batch.clone(),
        observe_panels: true,
        // A gene's tracks are read only where every file has them.
        row_alignment: data_beans::sparse_io_vector::RowAlignment::Intersect,
        column_alignment: data_beans::sparse_io_vector::ColumnAlignment::Disjoint,
        // Track rows are matched by their full names.
        feature_kind: Some(data_beans::aux::feature_names::FeatureNameKind::Exact),
        want_hierarchy: true,
        prebuilt_partition,
        cnv_clones: args.collapse.cnv_clones.as_deref(),
        multiome: None,
        reuse: args.pb_from.pb_from.as_deref(),
    })?;
    let cell_to_pb = cell_to_pb_per_level.expect("the collapse keeps its hierarchy");
    Ok(Loaded {
        cell_to_pb,
        collapsed: collapsed_levels,
        data: data_vec,
        keep: output_keep_idx,
        inputs: Inputs {
            data_files,
            batch_files,
        },
        proj_kn,
        pb_tree,
    })
}

pub fn run_td_rbm(args: &TdRbmArgs) -> anyhow::Result<()> {
    senna::embed_common::mkdir_parent(&args.out)?;
    // Ctrl+C ends training; the cells are then read and everything written.
    let _ = legume_numeric::matrix::stop::stop_flag();
    anyhow::ensure!(
        args.minibatch_size > 0 && args.chains > 0 && args.block_size > 0,
        "--minibatch-size, --chains and --block-size must be positive"
    );
    anyhow::ensure!(
        args.depth > 0 && args.state_depth > 0 && args.assign_depth > 0.0,
        "--depth, --state-depth and --assign-depth must be positive"
    );
    anyhow::ensure!(
        args.states > 0 && args.epochs > args.time_epochs,
        "--states must be positive and --epochs exceed --time-epochs, so the states are fitted"
    );
    let dev = args.device.to_device(args.device_no)?;
    let Loaded {
        data,
        keep,
        inputs,
        proj_kn,
        cell_to_pb,
        collapsed,
        pb_tree,
    } = load(args)?;
    let all_cells = data.column_names()?;
    let member = cell_to_pb.last().expect("at least one level");
    // Cells per pseudobulk at every level.
    let level_cells: Vec<Vec<f32>> = cell_to_pb
        .iter()
        .map(|member| {
            let mut n = vec![0f32; member.iter().max().map_or(0, |&m| m + 1)];
            member.iter().for_each(|&p| n[p] += 1.0);
            n
        })
        .collect();
    // Every level's pseudobulks on both tracks of every gene, as the
    // collapse has them (batch-adjusted when there are batches): its rates
    // as the data alone give them (no prior), times the cells.
    let row_names = data.row_names()?;
    let tracks: Vec<PbTracks> = collapsed
        .iter()
        .zip(&level_cells)
        .map(|(c, n)| {
            let rates = c.mu_adjusted.as_ref().unwrap_or(&c.mu_observed);
            let mean = rates.posterior_mean();
            anyhow::ensure!(mean.ncols() == n.len(), "a pseudobulk per rate column");
            let counts = nalgebra::DMatrix::<f32>::from_fn(mean.nrows(), n.len(), |r, j| {
                let rate = if c.stats_kept {
                    rates.evidence_mean(r, j)
                } else {
                    mean[(r, j)]
                };
                rate * n[j]
            });
            pb_tracks(&row_names, &counts)
        })
        .collect::<anyhow::Result<_>>()?;
    // The finest level's batch effects `[D × batches]`, when there are
    // batches: what the cells are adjusted by as they are read.
    let delta = collapsed
        .last()
        .and_then(|c| c.delta.as_ref())
        .map(|d| d.posterior_mean().clone());
    drop(collapsed);
    let finest = tracks.last().expect("at least one level");
    let names: Vec<Box<str>> = finest.genes.clone();
    let (g, n_pb) = (names.len(), finest.spliced.nrows());
    let f = 2 * g;
    // The most variable genes start the fit and call the roots.
    let top = select_genes(finest, args.min_reads, args.max_genes);
    anyhow::ensure!(!top.is_empty(), "no gene to start from");
    let t = top.len();
    info!(
        "td-rbm: {} cells from {} files; {} levels, {n_pb} finest pseudobulks; {g} genes with both tracks in the model, the {t} most variable start it and call the roots",
        all_cells.len(),
        inputs.data_files.len(),
        cell_to_pb.len(),
    );
    // Every level's pseudobulks, `[P × F]` (unspliced genes first), coarse
    // to fine; with `--steady-unspliced`, every gene's reads split anew
    // between the tracks at its overall unspliced share (the negative
    // control: no splicing dynamics, the same either way in time).
    let mut rows: Vec<Vec<f32>> = tracks
        .iter()
        .map(|tr| {
            (0..tr.spliced.nrows())
                .flat_map(|p| {
                    (0..g)
                        .map(move |j| tr.unspliced[(p, j)])
                        .chain((0..g).map(move |j| tr.spliced[(p, j)]))
                })
                .collect()
        })
        .collect();
    // The genes' overall unspliced shares the control splits at, and its
    // draws: the pseudobulks here, the cells when they are read.
    let mut steady: Option<(Vec<f64>, StdRng)> = args.steady_unspliced.map(|seed| {
        let total = |m: &nalgebra::DMatrix<f32>, j: usize| -> f64 {
            m.column(j).iter().map(|&v| f64::from(v)).sum()
        };
        let share = (0..g)
            .map(|j| {
                let (u, s) = (total(&finest.unspliced, j), total(&finest.spliced, j));
                u / (u + s).max(1.0)
            })
            .collect();
        (share, StdRng::seed_from_u64(seed))
    });
    if let Some((share, rng)) = steady.as_mut() {
        rows.iter_mut()
            .for_each(|x| x.chunks_mut(f).for_each(|row| resplit(row, share, rng)));
        info!("td-rbm: every gene's reads split at its overall unspliced share");
    }
    drop(tracks);
    // An earlier run's gene table (`--{init,freeze,lora}-feature-embedding`):
    // its rows of the model's genes, matched by name.
    let (preset, _) = crate::feature_preset::resolve_preset(
        args.feature_embedding.resolve()?,
        &names,
        &graph_embedding_util::FeatureNameKind::Gene { delim: '_' },
    )?;

    // Where the fit starts, on the most variable genes: a forest over
    // clusters of the finest pseudobulks, every tree provisionally rooted at
    // one end of its longest path (or where `--root` says), the lineages its
    // root-to-leaf paths; the genes' embedding from every finest pseudobulk,
    // every lineage's curve through its pseudobulks in their order from the
    // root.
    let fin = rows.last().expect("a level");
    let ft = 2 * t;
    let fin_top: Vec<f32> = fin.chunks(f).flat_map(|r| gene_columns(r, &top)).collect();
    let states = row_states(&fin_top, ft, STATE_DIMS);
    let forest = Forest::new(&states, args.clusters, args.knn, args.seed);
    let mut roots: Vec<usize> = (0..forest.trees.len()).map(|t| forest.far_end(t)).collect();
    let mut given = vec![false; forest.trees.len()];
    if let Some(path) = &args.root {
        let wanted: Vec<Box<str>> = std::fs::read_to_string(path)?
            .lines()
            .map(str::trim)
            .filter(|l| !l.is_empty())
            .map(Box::from)
            .collect();
        // Named as written out, or by a barcode that names one cell.
        let mut votes = vec![0usize; forest.clusters];
        for &i in crate::view::match_cells(&wanted, &all_cells).values() {
            votes[forest.cluster[member[i as usize]]] += 1;
        }
        let top_cluster = (0..forest.clusters)
            .max_by_key(|&c| votes[c])
            .expect("a cluster");
        anyhow::ensure!(
            votes[top_cluster] > 0,
            "none of the --root cells is in the data"
        );
        roots[forest.tree[top_cluster]] = top_cluster;
        given[forest.tree[top_cluster]] = true;
    }
    info!(
        "td-rbm: {} clusters in {} trees ({} with one cluster)",
        forest.clusters,
        forest.trees.len(),
        forest.trees.iter().filter(|t| t.len() == 1).count()
    );
    let ps: Vec<f32> = fin.chunks(f).flat_map(|r| r[g..].to_vec()).collect();
    let (pu_top, ps_top): (Vec<f32>, Vec<f32>) = (
        fin_top.chunks(ft).flat_map(|r| r[..t].to_vec()).collect(),
        fin_top.chunks(ft).flat_map(|r| r[t..].to_vec()).collect(),
    );
    // The embedding's width: the given table's, or `--embedding-dim`.
    let h = match preset.as_ref() {
        Some(p) => p.width(),
        None => args.embedding_dim.min(n_pb.saturating_sub(1)).max(1),
    };
    // The most variable genes' start; every other gene's comes from its
    // module (or a given table), else starts at zero.
    let rho_top = rho_start(&pu_top, &ps_top, n_pb, t, h);
    let mut rho = vec![0f32; g * h];
    for (i, &j) in top.iter().enumerate() {
        rho[j * h..(j + 1) * h].copy_from_slice(&rho_top[i * h..(i + 1) * h]);
    }
    let mut mask = vec![0f32; g];
    if let Some(p) = preset.as_ref() {
        given_rows(&mut rho, &mut mask, p, h);
    }
    // The curves start in the coordinates ρ starts at: the most variable
    // genes' rows as the fit takes them (a given table's, when there is one).
    let rho_start_top: Vec<f32> = top
        .iter()
        .flat_map(|&j| rho[j * h..(j + 1) * h].to_vec())
        .collect();
    // Gene modules, as bge and rbm have them, unless a table is given.
    anyhow::ensure!(
        preset.is_none() || args.modules.feature_modules.is_none(),
        "--feature-modules builds every gene's row from its module, so no row is a given \
         table's: drop --feature-modules or the --{{init,freeze,lora}}-feature-embedding"
    );
    let module_cfg = if args.no_feature_modules {
        None
    } else if preset.is_some() {
        info!("td-rbm: a given feature embedding has per-gene rows; gene modules are off");
        None
    } else {
        args.modules.resolve(Some(DEFAULT_GENE_MODULES))?
    };
    let modules = match module_cfg.as_ref() {
        None => None,
        Some(mc) => {
            let totals = nalgebra::DMatrix::<f32>::from_fn(g, n_pb, |j, p| {
                fin[p * f + j] + fin[p * f + g + j]
            });
            let sizes: Vec<f32> = level_cells.last().expect("a level").clone();
            let labels =
                graph_embedding_util::partition_modules(&totals, &sizes, mc.n_modules, args.seed)?;
            let n = labels.iter().max().map_or(1, |&m| m as usize + 1);
            // A module starts at the mean of its most variable genes'
            // starts; those genes keep theirs as residuals, every other
            // gene starts at its module.
            let mut mu = vec![0f32; n * h];
            let mut size = vec![0f32; n];
            for &j in &top {
                let m = labels[j] as usize;
                size[m] += 1.0;
                for d in 0..h {
                    mu[m * h + d] += rho[j * h + d];
                }
            }
            for m in 0..n {
                for d in 0..h {
                    mu[m * h + d] /= size[m].max(1.0);
                }
            }
            for j in 0..g {
                let m = labels[j] as usize;
                for d in 0..h {
                    rho[j * h + d] -= mu[m * h + d];
                }
            }
            for j in (0..g).filter(|j| top.binary_search(j).is_err()) {
                rho[j * h..(j + 1) * h].iter_mut().for_each(|v| *v = 0.0);
            }
            info!("td-rbm: {n} gene module(s) over {g} genes");
            Some((labels, mu))
        }
    };
    let base = base_start(&ps, n_pb, g);
    let (k, m) = (args.grid as usize, args.knots as usize);
    let values = Spline::new(m, &grid_times(k), &Device::Cpu)?
        .values
        .flatten_all()?
        .to_vec1::<f32>()?;
    let begin = start_along(
        &fin_top,
        ft,
        &states,
        &forest,
        &roots,
        &rho_start_top,
        &values,
        (t, h, k, m),
        args.knn,
    );
    let l = begin.paths.len();

    let lora = args.feature_embedding.lora.spec();
    let mut root_genes = vec![false; g];
    top.iter().for_each(|&j| root_genes[j] = true);
    let cfg = TrainConfig {
        grid: k,
        knots: m,
        states: args.states,
        embedding: h,
        lineages: l,
        concentration: args.lineage_concentration,
        epochs: args.epochs,
        time_epochs: args.time_epochs,
        steady_epochs: args.steady_epochs,
        direction_draws: args.direction_draws,
        knn: args.knn,
        smooth: args.smooth,
        assign_depth: args.assign_depth,
        depth: args.depth,
        state_depth: args.state_depth,
        batch: args.minibatch_size,
        chains: args.chains,
        sweeps: args.gibbs_sweeps,
        lr: args.learning_rate,
        seed: args.seed,
        lora_lr_ratio: f64::from(lora.lr_ratio),
        lora_ridge: f64::from(lora.ridge),
        module_ridge: module_cfg
            .as_ref()
            .map_or(0.0, |m| f64::from(m.residual_l2)),
        root_genes,
    };
    // The map's space: the leading principal components of the finest
    // pseudobulks' expression on the most variable genes.
    let basis = MapBasis::new(&fin_top, ft)?;
    let levels_in: Vec<Level> = rows
        .into_iter()
        .zip(level_cells)
        .map(|(x, cells)| Level { x, cells })
        .collect();
    let mut begin = begin;
    let curve = std::mem::take(&mut begin.curve);
    let model = fit(
        &levels_in,
        f,
        &Init0 {
            rho: &rho,
            base: &base,
            curve: &curve,
            beta: BETA0,
            gamma: GAMMA0,
            given: preset.as_ref().map(|p| GivenRho {
                mask: &mask,
                pinned: p.mode.pins(),
                lora_rank: p.mode.lora().map(|s| s.rank),
            }),
            modules: modules
                .as_ref()
                .map(|(labels, mu)| ModulesInit { labels, mu }),
        },
        begin,
        &forest,
        &states,
        &roots,
        &given,
        &cfg,
        &dev,
    )?;
    drop(levels_in);

    // Every cell read once, a block at a time, sparse until a block is read
    // (batch-adjusted when there are batches); QC's near-empty cells stay out.
    let rho_rows = model.energy.rho()?.to_vec2::<f32>()?;
    let rho_top_fit: Vec<Vec<f32>> = top.iter().map(|&j| rho_rows[j].clone()).collect();
    let kept: Option<Vec<bool>> = keep.as_ref().map(|k| {
        let mut m = vec![false; all_cells.len()];
        k.iter().for_each(|&i| m[i] = true);
        m
    });
    let tables = model.read_tables()?;
    let mut out = Summary::default();
    let (mut cells, mut of_cell, mut reads, mut map) =
        (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    for_cell_blocks(
        &data,
        &names,
        args.block_size,
        delta.as_ref(),
        |start, u, s| {
            let rows: Vec<usize> = (0..u.nrows())
                .filter(|i| kept.as_ref().is_none_or(|k| k[start + i]))
                .collect();
            let mut x = Vec::with_capacity(rows.len() * f);
            for &i in &rows {
                x.extend(u.row(i).iter().chain(s.row(i).iter()));
                cells.push(all_cells[start + i].clone());
                of_cell.push(member[start + i]);
            }
            if let Some((share, rng)) = steady.as_mut() {
                x.chunks_mut(f).for_each(|row| resplit(row, share, rng));
            }
            for row in x.chunks(f) {
                reads.push(row.iter().sum::<f32>().clamp(1.0, args.depth as f32));
                map.push(basis.place(&gene_columns(row, &top)));
            }
            out.extend(model.summarise(&tables, x, f, args.depth as f32, &dev)?);
            Ok(())
        },
    )?;
    let map_velocity: Vec<Vec<f32>> = out
        .velocity
        .iter()
        .map(|v| basis.velocity(&rho_top_fit, v))
        .collect();

    // The start as the chosen roots have it, so its lineages are the
    // fitted ones' paths.
    let start: Vec<f32> = of_cell.iter().map(|&p| model.start_order[p]).collect();
    let pb_names: Vec<Box<str>> = of_cell.iter().map(|p| format!("pb{p}").into()).collect();
    let lineage_names: Vec<Box<str>> = out
        .lineage
        .iter()
        .map(|&i| format!("L{}", i + 1).into())
        .collect();
    let start_names: Vec<Box<str>> = of_cell
        .iter()
        .map(|&p| format!("L{}", model.start_lineage[p] + 1).into())
        .collect();
    let cluster_names: Vec<Box<str>> = of_cell
        .iter()
        .map(|&p| format!("C{}", forest.cluster[p]).into())
        .collect();
    let p_lineage = columns("p_L", &out.p_lineage, l);
    let mut cols: Vec<(Box<str>, Column)> = vec![
        ("lineage".into(), Column::Str(&lineage_names)),
        ("tau".into(), Column::F32(&out.tau)),
        ("tau_sd".into(), Column::F32(&out.tau_sd)),
        ("tau_start".into(), Column::F32(&start)),
        ("lineage_start".into(), Column::Str(&start_names)),
        ("pb".into(), Column::Str(&pb_names)),
        ("cluster".into(), Column::Str(&cluster_names)),
    ];
    cols.extend(p_lineage.iter().map(|(n, v)| (n.clone(), Column::F32(v))));
    write_named_table(
        &format!("{}.cell_time.parquet", args.out),
        "cell",
        &cells,
        &cols,
    )?;
    write(
        &format!("{}.cell_state.parquet", args.out),
        "cell",
        &cells,
        &columns("h", &out.state, args.states),
    )?;
    write(
        &format!("{}.cell_velocity_rho.parquet", args.out),
        "cell",
        &cells,
        &columns("v", &out.velocity, h),
    )?;
    // The map's space: every cell's expression on the leading principal
    // components (layouts, clusters), and its velocity there.
    let pcs = map.first().map_or(0, Vec::len);
    write(
        &format!("{}.cell_embedding.parquet", args.out),
        "cell",
        &cells,
        &columns("pc", &map, pcs),
    )?;
    write(
        &format!("{}.cell_velocity.parquet", args.out),
        "cell",
        &cells,
        &columns("pc", &map_velocity, pcs),
    )?;

    // Every used lineage's direction and what it rests on.
    {
        use std::io::Write;
        let mut file = std::fs::File::create(format!("{}.direction.tsv", args.out))?;
        let speed = model
            .energy
            .speed(Some(&model.log_weight))?
            .flatten_all()?
            .to_vec1::<f32>()?;
        writeln!(file, "lineage\ttree\tclusters\tweight\tspeed\tdetermined\tp_forward\tlog_bf\tlog_bf_sd\tlog_abf\tgenes_for\tgenes_against")?;
        for d in &model.directions {
            let path = &model.paths[d.lineage];
            let along: Vec<String> = path.clusters.iter().map(|c| format!("C{c}")).collect();
            writeln!(
                file,
                "L{}\tT{}\t{}\t{:.4}\t{:.4}\t{}\t{:.6}\t{:.3}\t{:.3}\t{:.3}\t{}\t{}",
                d.lineage + 1,
                path.tree + 1,
                along.join(">"),
                model.log_weight[d.lineage].exp(),
                speed[d.lineage],
                model.roots[path.tree].determined,
                d.p_forward,
                d.log_bf,
                d.log_bf_sd,
                d.log_abf,
                d.gene_votes.iter().filter(|&&v| v > 0.0).count(),
                d.gene_votes.iter().filter(|&&v| v < 0.0).count(),
            )?;
        }
    }
    // Every tree's root: the candidates' posterior.
    {
        use std::io::Write;
        let mut file = std::fs::File::create(format!("{}.roots.tsv", args.out))?;
        writeln!(
            file,
            "tree\tcandidate\tp\tchosen\tprovisional\tdetermined\tgiven\tmirrored"
        )?;
        for c in &model.roots {
            for (&a, &p) in c.candidates.iter().zip(&c.p) {
                writeln!(
                    file,
                    "T{}\tC{}\t{:.6}\t{}\tC{}\t{}\t{}\t{}",
                    c.tree + 1,
                    a,
                    p,
                    a == c.chosen,
                    c.provisional,
                    c.determined,
                    c.given,
                    c.mirrored
                )?;
            }
        }
    }
    // Every tree's edges and the splicing along them.
    {
        use std::io::Write;
        let mut file = std::fs::File::create(format!("{}.edges.tsv", args.out))?;
        writeln!(
            file,
            "tree\tfrom\tto\tlog_bf\tgenes_for\tgenes_against\tunder_chosen_root"
        )?;
        for c in &model.roots {
            let (parent, _) = forest.rooted(c.chosen);
            for &(a, b, v, up, down) in &c.edges {
                let kept = parent[b] == Some(a);
                writeln!(
                    file,
                    "T{}\tC{}\tC{}\t{:.3}\t{}\t{}\t{}",
                    c.tree + 1,
                    a,
                    b,
                    v,
                    up,
                    down,
                    if kept { "same" } else { "reversed" }
                )?;
            }
        }
    }
    let vote_cols: Vec<(Box<str>, Vec<f32>)> = model
        .directions
        .iter()
        .flat_map(|d| {
            [
                (
                    format!("log_bf_L{}", d.lineage + 1).into(),
                    d.gene_votes.clone(),
                ),
                (format!("z_L{}", d.lineage + 1).into(), d.gene_z.clone()),
            ]
        })
        .collect();
    if !vote_cols.is_empty() {
        // The genes that called the directions.
        let top_names: Vec<Box<str>> = top.iter().map(|&j| names[j].clone()).collect();
        let top_cols: Vec<(Box<str>, Vec<f32>)> = vote_cols
            .into_iter()
            .map(|(n, v)| {
                (
                    n,
                    top.iter()
                        .map(|&j| v.get(j).copied().unwrap_or(0.0))
                        .collect(),
                )
            })
            .collect();
        write(
            &format!("{}.gene_direction.parquet", args.out),
            "gene",
            &top_names,
            &top_cols,
        )?;
    }

    let e = &model.energy;
    let row = |t: &Tensor| -> anyhow::Result<Vec<f32>> { Ok(t.flatten_all()?.to_vec1::<f32>()?) };
    write(
        &format!("{}.gene_rates.parquet", args.out),
        "gene",
        &names,
        &[
            ("beta".into(), row(&e.beta()?)?),
            ("gamma".into(), row(&e.gamma()?)?),
            ("base".into(), row(e.base())?),
        ],
    )?;
    write(
        &format!("{}.feature_embedding.parquet", args.out),
        "gene",
        &names,
        &columns("rho", &e.rho()?.to_vec2::<f32>()?, h),
    )?;
    let t = model.tables(true)?;
    let w = t.w.narrow(0, 0, g)?.to_vec2::<f32>()?;
    write(
        &format!("{}.gene_states.parquet", args.out),
        "gene",
        &names,
        &columns("w", &w, args.states),
    )?;
    // The model of expression `senna view` reads as latent × dictionary: a
    // gene's spliced logit in a cell is `ρ_g · z + w_g · h / N + base_g`, its
    // place on the curves `z` and its state `h` averaged over its posterior,
    // `N` its reads.
    let base = row(e.base())?;
    let rho = e.rho()?.to_vec2::<f32>()?;
    let cell_model: Vec<Vec<f32>> = out
        .position
        .iter()
        .zip(&out.state)
        .zip(&reads)
        .map(|((z, s), n)| {
            z.iter()
                .copied()
                .chain(s.iter().map(|v| v / n))
                .chain([1.0])
                .collect()
        })
        .collect();
    let gene_model: Vec<Vec<f32>> = (0..g)
        .map(|j| {
            rho[j]
                .iter()
                .chain(&w[j])
                .copied()
                .chain([base[j]])
                .collect()
        })
        .collect();
    let model_names: Vec<Box<str>> = (0..h)
        .map(|i| format!("z{i}"))
        .chain((0..args.states).map(|i| format!("h{i}")))
        .chain(["one".to_string()])
        .map(Into::into)
        .collect();
    let named = |rows: &[Vec<f32>]| -> Vec<(Box<str>, Vec<f32>)> {
        model_names
            .iter()
            .enumerate()
            .map(|(k, n)| (n.clone(), rows.iter().map(|r| r[k]).collect()))
            .collect()
    };
    write(
        &format!("{}.latent.parquet", args.out),
        "cell",
        &cells,
        &named(&cell_model),
    )?;
    write(
        &format!("{}.dictionary.parquet", args.out),
        "gene",
        &names,
        &named(&gene_model),
    )?;
    let times: Vec<Box<str>> = (0..l)
        .flat_map(|i| {
            grid_times(k)
                .into_iter()
                .map(move |t| format!("L{}:{t:.4}", i + 1).into())
        })
        .collect();
    let mut curve = columns("z", &t.curve.to_vec2::<f32>()?, h);
    curve.extend(columns("dz", &t.velocity.to_vec2::<f32>()?, h));
    curve.push(("nu".into(), row(&t.size)?));
    write(
        &format!("{}.curve.parquet", args.out),
        "tau",
        &times,
        &curve,
    )?;
    // What `senna view` and a later `--from` / `--pb-from` read.
    let keep = keep.as_deref();
    crate::postprocess::viz_prep::write_cell_proj(&args.out, &proj_kn, &all_cells, keep)?;
    crate::postprocess::viz_prep::write_cell_to_pb(&args.out, &cell_to_pb, &all_cells, keep)?;
    if let Some(tree) = &pb_tree {
        crate::postprocess::viz_prep::write_pb_tree(&args.out, tree, &data.row_names()?)?;
    }
    write_manifest(args, &inputs, pb_tree.is_some())?;
    info!(
        "Wrote {0}.{{cell_time,cell_state,cell_velocity,cell_velocity_rho,cell_embedding,gene_direction,gene_rates,feature_embedding,gene_states,curve}}.parquet, {0}.{{direction,roots,edges}}.tsv and {0}.senna.json",
        args.out
    );
    Ok(())
}

/// `{out}.senna.json`: the run as `senna view` and the other readers open
/// it. The cells' positions on their curves are the cell embedding and the
/// genes' ρ the feature embedding; the time, velocity and state tables are
/// recorded beside them, with the input and its partition.
fn write_manifest(args: &TdRbmArgs, inputs: &Inputs, has_pb_tree: bool) -> anyhow::Result<()> {
    use senna::run_manifest::{
        record_train_args, write_run_manifest, RunDescription, RunKind, RunManifest,
    };
    let prefix: &str = &args.out;
    let input: Vec<String> = inputs.data_files.iter().map(ToString::to_string).collect();
    let batch: Vec<String> = inputs
        .batch_files
        .as_deref()
        .map(|v| v.iter().map(ToString::to_string).collect())
        .unwrap_or_default();
    write_run_manifest(&RunDescription {
        train_args: Some(record_train_args(args)?),
        kind: RunKind::TdRbm,
        prefix,
        data_input: &input,
        data_multiome: None,
        data_batch: &batch,
        data_input_null: &[],
        dictionary_suffix: Some("dictionary.parquet"),
        has_model: false,
        has_cell_proj: true,
        pb_gene_suffix: None,
        pb_latent_suffix: None,
        pb_reference_suffix: None,
        dictionary_empirical_suffix: None,
        feature_embedding_suffix: Some("feature_embedding.parquet"),
        feature_coembedding_suffix: None,
        carried: None,
        module_membership_suffix: None,
        module_dictionary_suffix: None,
        softmax_dictionary_suffix: None,
        cell_embedding_suffix: Some("cell_embedding.parquet"),
        cell_encoder_suffix: None,
        default_colour_by: "cluster",
        has_latent: true,
        has_cell_to_pb: true,
        has_pb_tree,
    })?;
    let path = std::path::PathBuf::from(format!("{prefix}.senna.json"));
    let (mut m, _) = RunManifest::load(&path)?;
    let base = std::path::Path::new(prefix)
        .file_name()
        .map_or_else(|| prefix.to_string(), |n| n.to_string_lossy().into_owned());
    m.outputs.cell_time = Some(format!("{base}.cell_time.parquet"));
    m.outputs.cell_velocity = Some(format!("{base}.cell_velocity.parquet"));
    m.outputs.cell_state = Some(format!("{base}.cell_state.parquet"));
    m.save(&path)
}

/// Put a given table's rows into `rho` (`[G × H]` row-major) and mark them
/// in `mask`. The free rows started from the expression (the non-zero ones)
/// are scaled per column to the given rows' spread, so the two start alike.
fn given_rows(rho: &mut [f32], mask: &mut [f32], p: &graph_embedding_util::PresetRows, h: usize) {
    let rms = |rows: &mut dyn Iterator<Item = f32>| {
        let (s, n) = rows.fold((0f64, 0usize), |(s, n), v| (s + f64::from(v * v), n + 1));
        (s / n.max(1) as f64).sqrt()
    };
    for d in 0..h {
        let given = rms(&mut (0..p.ids.len()).map(|i| p.rows[i * h + d]));
        let started: Vec<usize> = (0..mask.len())
            .filter(|&j| rho[j * h..(j + 1) * h].iter().any(|&v| v != 0.0))
            .collect();
        let free = rms(&mut started.iter().map(|&j| rho[j * h + d]));
        let scale = if free > 1e-12 {
            (given / free) as f32
        } else {
            0.0
        };
        (0..mask.len()).for_each(|j| rho[j * h + d] *= scale);
    }
    for (i, &id) in p.ids.iter().enumerate() {
        let j = id as usize;
        rho[j * h..(j + 1) * h].copy_from_slice(&p.rows[i * h..(i + 1) * h]);
        mask[j] = 1.0;
    }
}

/// A row's reads of every gene (`[2G]`, unspliced genes first) split anew
/// between the tracks at the gene's unspliced `share`: the negative control
/// with no splicing dynamics.
fn resplit(row: &mut [f32], share: &[f64], rng: &mut StdRng) {
    let g = share.len();
    for (j, &p) in share.iter().enumerate() {
        let n = (row[j] + row[g + j]).round() as u64;
        let a = Binomial::new(n, p).map_or(0, |b| b.sample(rng));
        row[j] = a as f32;
        row[g + j] = (n - a) as f32;
    }
}

/// Principal components of the map space.
const MAP_PCS: usize = 30;

/// The map space, as scVelo has it: `log1p` reads per 10⁴ on the most
/// variable genes (both tracks summed), centred, on the leading principal
/// components of the finest pseudobulks; a cell is placed on them as it is
/// read, its velocity read out per gene (`ρ_g · v`, a `d log s_g/dt`,
/// centred over the genes) through the same components.
struct MapBasis {
    /// `[T]` the pseudobulks' mean expression per gene.
    mean: Vec<f32>,
    /// `[T × k]` the components.
    v: nalgebra::DMatrix<f32>,
}

impl MapBasis {
    /// From rows `x` (`[P × F]`, unspliced genes first).
    fn new(x: &[f32], f: usize) -> anyhow::Result<Self> {
        use legume_numeric::matrix::traits::RandomizedAlgs;
        let g = f / 2;
        let p = x.len() / f;
        let mut m = nalgebra::DMatrix::<f32>::zeros(p, g);
        for (i, row) in x.chunks(f).enumerate() {
            let e = expression(row);
            m.row_mut(i).copy_from_slice(&e);
        }
        let mean: Vec<f32> = (0..g).map(|j| m.column(j).mean()).collect();
        for (j, &mu) in mean.iter().enumerate() {
            m.column_mut(j).add_scalar_mut(-mu);
        }
        let k = MAP_PCS.min(p.saturating_sub(1)).min(g).max(1);
        let (_, _, v) = m.rsvd(k)?;
        Ok(Self { mean, v })
    }

    /// A row's (`[F]`) place on the components.
    fn place(&self, row: &[f32]) -> Vec<f32> {
        let e = expression(row);
        (0..self.v.ncols())
            .map(|c| {
                e.iter()
                    .zip(&self.mean)
                    .enumerate()
                    .map(|(j, (a, b))| (a - b) * self.v[(j, c)])
                    .sum()
            })
            .collect()
    }

    /// A velocity `v` (`[H]`) through the genes' `rho` (`[T][H]`) on the
    /// components.
    fn velocity(&self, rho: &[Vec<f32>], v: &[f32]) -> Vec<f32> {
        let per_gene: Vec<f32> = rho
            .iter()
            .map(|r| r.iter().zip(v).map(|(a, b)| a * b).sum())
            .collect();
        let mean = per_gene.iter().sum::<f32>() / per_gene.len().max(1) as f32;
        (0..self.v.ncols())
            .map(|c| {
                per_gene
                    .iter()
                    .enumerate()
                    .map(|(j, x)| (x - mean) * self.v[(j, c)])
                    .sum()
            })
            .collect()
    }
}

/// A row's (`[F]`, unspliced genes first) `log1p` reads per 10⁴ per gene,
/// both tracks summed.
fn expression(row: &[f32]) -> Vec<f32> {
    let g = row.len() / 2;
    let total: f32 = row.iter().sum::<f32>().max(1.0);
    (0..g)
        .map(|j| ((row[j] + row[g + j]) / total * 1e4).ln_1p())
        .collect()
}
