//! `senna td-rbm`: read two-track count backends, collapse their cells into
//! multilevel pseudobulks, pick the genes, start the genes' embedding and the
//! curve from the finest pseudobulks in their expression order (as `senna
//! time` starts), fit the machine on samples of every level
//! ([`super::train`]) and read every cell.

use super::args::TdRbmArgs;
use super::counts::{for_cell_blocks, pb_tracks, select_genes, two_track_genes};
use super::curve::{grid_times, Spline};
use super::model::{GivenRho, Init0};
use super::start::{base_start, rho_start, row_states, start_along, STATE_DIMS};
use super::train::{fit, Level, TrainConfig, BETA0, GAMMA0};
use super::tree::Forest;
use data_beans::sparse_io_vector::SparseIoVec;
use legume_numeric::candle::candle_core::{Device, Tensor};
use legume_numeric::matrix::parquet::{write_named_table, Column};
use log::info;
use rand::rngs::StdRng;
use rand::SeedableRng;
use rand_distr::{Binomial, Distribution};
use senna::embed_common::Mat;
use std::borrow::Cow;

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

/// The cells as the shared loader reads them, and their pseudobulks.
struct Loaded {
    data: SparseIoVec,
    /// The cells written out (QC's near-empty cells left out), when QC ran.
    keep: Option<Vec<usize>>,
    inputs: Inputs,
    proj_kn: Mat,
    /// Every cell's pseudobulk at every level, coarse to fine.
    cell_to_pb: Vec<Vec<usize>>,
    /// The finest pseudobulks' reads `[D × P]` on every row of `data`.
    finest_counts: nalgebra::DMatrix<f32>,
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
    })?;
    let cell_to_pb = cell_to_pb_per_level.expect("the collapse keeps its hierarchy");
    let (finest_counts, _) = collapsed_levels
        .last()
        .expect("at least one level")
        .observed_counts(cell_to_pb.last().expect("at least one level"))?;
    Ok(Loaded {
        cell_to_pb,
        finest_counts,
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
        finest_counts,
        pb_tree,
    } = load(args)?;
    let all_cells = data.column_names()?;
    let member = cell_to_pb.last().expect("at least one level");
    let finest = pb_tracks(&data.row_names()?, &finest_counts)?;
    drop(finest_counts);
    let genes = select_genes(&finest, args.min_reads, args.max_genes);
    anyhow::ensure!(!genes.is_empty(), "no gene left to fit");
    let (g, n_pb) = (genes.len(), finest.spliced.nrows());
    // Cells per pseudobulk at every level.
    let level_cells: Vec<Vec<f32>> = cell_to_pb
        .iter()
        .map(|member| {
            let mut n = vec![0f32; member.iter().max().map_or(0, |&m| m + 1)];
            member.iter().for_each(|&p| n[p] += 1.0);
            n
        })
        .collect();
    let f = 2 * g;
    info!(
        "td-rbm: {} cells from {} files; {} levels, {n_pb} finest pseudobulks; {} genes with both tracks, {g} fitted",
        all_cells.len(),
        inputs.data_files.len(),
        cell_to_pb.len(),
        finest.genes.len(),
    );
    // Every cell's reads on the fitted genes, `[C × F]` (unspliced genes
    // first); with `--steady-unspliced`, every cell's reads of a gene split
    // anew between the tracks at the gene's overall unspliced share (the
    // negative control: no splicing dynamics, the same either way in time).
    let names: Vec<Box<str>> = genes.iter().map(|&j| finest.genes[j].clone()).collect();
    // An earlier run's gene table (`--{init,freeze,lora}-feature-embedding`):
    // its rows of the fitted genes, matched by name.
    let (preset, _) = crate::feature_preset::resolve_preset(
        args.feature_embedding.resolve()?,
        &names,
        &graph_embedding_util::FeatureNameKind::Gene { delim: '_' },
    )?;
    let mut x = Vec::with_capacity(all_cells.len() * f);
    for_cell_blocks(&data, &names, args.block_size, |_, u, s| {
        for i in 0..u.nrows() {
            x.extend((0..g).map(|j| u[(i, j)]));
            x.extend((0..g).map(|j| s[(i, j)]));
        }
        Ok(())
    })?;
    let c = all_cells.len();
    if let Some(seed) = args.steady_unspliced {
        let mut rng = StdRng::seed_from_u64(seed);
        for j in 0..g {
            let (u, t) = (0..c).fold((0f64, 0f64), |(u, t), i| {
                let (a, b) = (f64::from(x[i * f + j]), f64::from(x[i * f + g + j]));
                (u + a, t + a + b)
            });
            let share = u / t.max(1.0);
            for i in 0..c {
                let n = (x[i * f + j] + x[i * f + g + j]).round() as u64;
                let a = Binomial::new(n, share).map_or(0, |b| b.sample(&mut rng));
                x[i * f + j] = a as f32;
                x[i * f + g + j] = (n - a) as f32;
            }
        }
        info!("td-rbm: every gene's reads split at its overall unspliced share (seed {seed})");
    }
    // Every level's pseudobulks, `[P × F]`, coarse to fine: sums of their
    // cells' reads.
    let rows: Vec<Vec<f32>> = cell_to_pb
        .iter()
        .zip(&level_cells)
        .map(|(member, n)| {
            let mut r = vec![0f32; n.len() * f];
            for (i, &p) in member.iter().enumerate() {
                r[p * f..(p + 1) * f]
                    .iter_mut()
                    .zip(&x[i * f..(i + 1) * f])
                    .for_each(|(a, b)| *a += b);
            }
            r
        })
        .collect();
    // The cells read out: QC keeps near-empty cells in the pseudobulks but
    // not in the outputs.
    let (cells, of_cell): (Cow<[Box<str>]>, Cow<[usize]>) = match &keep {
        Some(keep) => {
            // In place, every kept row moved forward.
            for (to, &from) in keep.iter().enumerate() {
                x.copy_within(from * f..(from + 1) * f, to * f);
            }
            x.truncate(keep.len() * f);
            let (cells, of_cell): (Vec<_>, Vec<_>) = keep
                .iter()
                .map(|&i| (all_cells[i].clone(), member[i]))
                .unzip();
            (cells.into(), of_cell.into())
        }
        None => (all_cells.as_slice().into(), member.as_slice().into()),
    };

    // Where the fit starts: a forest over clusters of the finest
    // pseudobulks, every tree provisionally rooted at one end of its longest
    // path (or where `--root` says), the lineages its root-to-leaf paths; the
    // genes' embedding and baselines from every finest pseudobulk, every
    // lineage's curve through its pseudobulks in their order from the root.
    let fin = rows.last().expect("a level");
    let states = row_states(fin, f, STATE_DIMS);
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
        for &i in crate::view::match_cells(&wanted, &cells).values() {
            votes[forest.cluster[of_cell[i as usize]]] += 1;
        }
        let top = (0..forest.clusters)
            .max_by_key(|&c| votes[c])
            .expect("a cluster");
        anyhow::ensure!(votes[top] > 0, "none of the --root cells is in the data");
        roots[forest.tree[top]] = top;
        given[forest.tree[top]] = true;
    }
    info!(
        "td-rbm: {} clusters in {} trees ({} with one cluster)",
        forest.clusters,
        forest.trees.len(),
        forest.trees.iter().filter(|t| t.len() == 1).count()
    );
    let (pu, ps): (Vec<f32>, Vec<f32>) = (
        fin.chunks(f).flat_map(|r| r[..g].to_vec()).collect(),
        fin.chunks(f).flat_map(|r| r[g..].to_vec()).collect(),
    );
    // The embedding's width: the given table's, or `--embedding-dim`.
    let h = match preset.as_ref() {
        Some(p) => p.width(),
        None => args.embedding_dim.min(n_pb.saturating_sub(1)).max(1),
    };
    let mut rho = rho_start(&pu, &ps, n_pb, g, h);
    let mut mask = vec![0f32; g];
    if let Some(p) = preset.as_ref() {
        given_rows(&mut rho, &mut mask, p, h);
    }
    let base = base_start(&ps, n_pb, g);
    let (k, m) = (args.grid as usize, args.knots as usize);
    let values = Spline::new(m, &grid_times(k), &Device::Cpu)?
        .values
        .flatten_all()?
        .to_vec1::<f32>()?;
    let begin = start_along(
        fin,
        f,
        &states,
        &forest,
        &roots,
        &rho,
        &values,
        (g, h, k, m),
        args.knn,
    );
    let l = begin.paths.len();

    let lora = args.feature_embedding.lora.spec();
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
    };
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
        },
        begin,
        &forest,
        &states,
        &roots,
        &given,
        &cfg,
        &dev,
    )?;

    let out = model.summarise(&x, f, args.depth as f32, &dev)?;
    let rho_rows = model.energy.rho()?.to_vec2::<f32>()?;
    let (map, map_velocity) = map_space(&x, f, &rho_rows, &out.velocity)?;
    drop(x);

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
        write(
            &format!("{}.gene_direction.parquet", args.out),
            "gene",
            &names,
            &vote_cols,
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
    write(
        &format!("{}.gene_states.parquet", args.out),
        "gene",
        &names,
        &columns("w", &t.w.narrow(0, 0, g)?.to_vec2::<f32>()?, args.states),
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
        dictionary_suffix: None,
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
        has_latent: false,
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
/// in `mask`. The free rows, started from the expression, are scaled per
/// column to the given rows' spread, so the two start alike.
fn given_rows(rho: &mut [f32], mask: &mut [f32], p: &graph_embedding_util::PresetRows, h: usize) {
    let rms = |rows: &mut dyn Iterator<Item = f32>| {
        let (s, n) = rows.fold((0f64, 0usize), |(s, n), v| (s + f64::from(v * v), n + 1));
        (s / n.max(1) as f64).sqrt()
    };
    for d in 0..h {
        let given = rms(&mut (0..p.ids.len()).map(|i| p.rows[i * h + d]));
        let free = rms(&mut (0..mask.len()).map(|j| rho[j * h + d]));
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

/// Principal components of the map space.
const MAP_PCS: usize = 30;

/// Every cell's place in the map space, and its velocity there.
type MapSpace = (Vec<Vec<f32>>, Vec<Vec<f32>>);

/// The cells' map space and their velocity in it, as scVelo has them:
/// every cell's `log1p` reads per 10⁴ (both tracks summed) on the fitted
/// genes, centred, on its leading principal components; and every cell's
/// velocity read out per gene (`ρ_g · v`, a `d log s_g/dt`, centred over
/// the genes) taken through the same components. `x` is `[C × F]`, `rho`
/// `[G × H]`, `velocity` `[C × H]`.
fn map_space(
    x: &[f32],
    f: usize,
    rho: &[Vec<f32>],
    velocity: &[Vec<f32>],
) -> anyhow::Result<MapSpace> {
    use legume_numeric::matrix::traits::RandomizedAlgs;
    let g = f / 2;
    let c = x.len() / f;
    let mut m = nalgebra::DMatrix::<f32>::zeros(c, g);
    for (i, row) in x.chunks(f).enumerate() {
        let total: f32 = row.iter().sum::<f32>().max(1.0);
        for j in 0..g {
            m[(i, j)] = ((row[j] + row[g + j]) / total * 1e4).ln_1p();
        }
    }
    for j in 0..g {
        let mean = m.column(j).mean();
        m.column_mut(j).add_scalar_mut(-mean);
    }
    let k = MAP_PCS.min(c.saturating_sub(1)).min(g).max(1);
    let (u, s, v) = m.rsvd(k)?; // [C × k], [k], [G × k]
    let k = s.len();
    let scores: Vec<Vec<f32>> = (0..c)
        .map(|i| (0..k).map(|j| u[(i, j)] * s[j]).collect())
        .collect();
    let moved: Vec<Vec<f32>> = velocity
        .iter()
        .map(|vc| {
            let per_gene: Vec<f32> = rho
                .iter()
                .map(|r| r.iter().zip(vc).map(|(a, b)| a * b).sum())
                .collect();
            let mean = per_gene.iter().sum::<f32>() / g.max(1) as f32;
            (0..k)
                .map(|j| (0..g).map(|q| (per_gene[q] - mean) * v[(q, j)]).sum())
                .collect()
        })
        .collect();
    Ok((scores, moved))
}
