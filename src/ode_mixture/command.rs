//! `senna ode`: read two-track count backends, collapse their cells into
//! pseudobulks (as every senna fit does), fit the mixture of splicing ODEs
//! on the finest pseudobulks and the most variable genes (the climb), and
//! read every cell out at the climb's mode, a block at a time.

use super::coarsen::{coarse_reads, coarsen_genes};
use super::counts::{for_cell_blocks, pb_tracks, select_genes, two_track_genes};
use super::init::{climb, root_pseudobulks, Climb};
use super::likelihood::Reads;
use super::model::lineage_label;
use super::output::MapBasis;
use super::readout::{grid_posterior, place, read_cells, GridPosterior, Mode};
use super::refine::genes_at_mode;
use anyhow::Context;
use data_beans::alg::feature_coarsening::FeatureCoarsening;
use legume_numeric::matrix::parquet::{write_named_table, Column};
use log::info;
use nalgebra::DMatrix;
use senna::embed_common::*;

/// The long help of `--max-coarse-features` as ode reads it (data-beans'
/// flag; ode sets its default and help in the `mut_arg` below).
const GROUPS_HELP: &str = "Fit on at most N groups of genes that share kinetics, when the\n\
     genes kept by --max-genes outnumber N. N is a ceiling: the groups\n\
     are built once before the climb (by timing, within bins of\n\
     log(S/U)) and may number fewer. The climb runs on each group's\n\
     summed reads; at its mode every gene's rates are then fitted on\n\
     its own reads, starting from its group's, so {out}.gene_rates\n\
     has every gene. The groups are written as {out}.coarsening.json.\n\
     \n\
     --no-feature-coarsening fits every gene on its own instead; so\n\
     does 0 here.";

#[derive(Args, Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(default = "senna::embed_common::clap_defaults")]
#[command(mut_arg("max_coarse_features", |a| a
    .default_value("250")
    .help("Fit on at most N groups of genes sharing kinetics; 0 = every gene")
    .long_help(GROUPS_HELP)))]
pub struct OdeArgs {
    #[arg(
        value_delimiter = ',',
        help = "Two-track count backends (optional when --from is given)",
        long_help = "Count backends whose rows are `{gene}/count/spliced` and\n\
                     `{gene}/count/unspliced`; optional when --from is given. Several\n\
                     files are read together on the genes they share, every file a\n\
                     batch unless --batch-files says otherwise."
    )]
    pub(crate) data_files: Vec<Box<str>>,

    #[arg(
        long,
        help = "Chain data, batch and cell→pb partition from a prior run"
    )]
    pub(crate) from: Option<Box<str>>,

    #[arg(short, long, required = true, help = "Output prefix")]
    pub(crate) out: Box<str>,

    #[arg(
        long,
        short,
        value_delimiter(','),
        help = "Batch membership files, one per data file"
    )]
    pub(crate) batch_files: Option<Vec<Box<str>>>,

    #[command(flatten)]
    pub(crate) collapse: crate::refine_weighting::CollapseArgs,

    #[command(flatten)]
    #[serde(flatten)]
    pub(crate) pb_from: crate::refine_weighting::PbFromArgs,

    #[arg(
        long,
        default_value_t = 1000,
        help = "Most variable genes the model is fitted on"
    )]
    pub max_genes: usize,

    #[arg(
        long,
        default_value_t = 20.0,
        help = "Fewest reads a fitted gene has on each track, over all cells"
    )]
    pub min_reads: f32,

    #[arg(
        long,
        default_value_t = 2,
        help = "Lineages, each with its own path and its own time (unless --trunk)"
    )]
    pub lineages: usize,

    #[arg(
        long,
        default_value_t = false,
        help = "Every other lineage on lineage 0's trunk until a branch time it fits, one clock for all"
    )]
    pub trunk: bool,

    #[arg(
        long,
        default_value_t = false,
        help = "No jitter of every pseudobulk's unspliced share"
    )]
    pub no_jitter: bool,

    #[arg(long, default_value_t = 45, help = "Grid points per lineage's path")]
    pub points: usize,

    #[arg(long, default_value_t = 4, help = "Time positions per grid step")]
    pub sub: usize,

    #[arg(long, default_value_t = 3, help = "Dimension of the paths")]
    pub dim: usize,

    #[arg(
        long,
        default_value_t = 0.25,
        help = "The paths' prior length scale over the duration"
    )]
    pub kappa: f64,

    #[arg(
        long,
        default_value_t = 2500,
        help = "Adam steps of the climb, per start"
    )]
    pub steps: usize,

    #[arg(
        long,
        default_value_t = 0.03,
        help = "Adam's learning rate in the climb"
    )]
    pub lr: f64,

    #[arg(
        long,
        help = "Cells at the root of the trajectory, one name a line: a tree start grows from the pseudobulks mostly of them"
    )]
    pub root_cells: Option<std::path::PathBuf>,

    #[arg(
        long,
        default_value_t = 30,
        help = "Principal components of the map space, fitted on the trajectories"
    )]
    pub map_pcs: usize,

    #[arg(long, default_value_t = 1, help = "Random seed")]
    pub seed: u64,

    #[arg(long, default_value_t = ComputeDevice::Cpu, value_enum, help = "Compute device of the climb")]
    pub device: ComputeDevice,

    #[arg(long, default_value_t = 0, help = "CUDA/Metal device index")]
    pub device_no: usize,

    #[arg(long, default_value_t = 1024, help = "Cells per readout block")]
    pub(crate) block_size: usize,

    #[arg(
        long,
        default_value_t = false,
        help = "Load all columns into memory first",
        hide = true
    )]
    pub(crate) preload_data: bool,

    #[command(flatten)]
    pub(crate) hvg: crate::hvg::HvgCliArgs,

    #[command(flatten)]
    #[serde(flatten)]
    pub(crate) coarsening: data_beans::alg::feature_coarsening::FeatureCoarseningArgs,

    #[command(flatten)]
    pub(crate) qc: QcArgs,
}

/// The files a run read, as its manifest records them.
struct Inputs {
    data_files: Vec<Box<str>>,
    batch_files: Option<Vec<Box<str>>>,
}

/// `cols` as f32 parquet columns.
fn f32_columns(cols: &[(Box<str>, Vec<f32>)]) -> Vec<(Box<str>, Column<'_>)> {
    cols.iter()
        .map(|(n, v)| (n.clone(), Column::F32(v)))
        .collect()
}

/// `[N × k]` rows, written with columns `{prefix}0..`, rows named `cell`.
fn write_rows(
    path: &str,
    cells: &[Box<str>],
    prefix: &str,
    rows: &[f32],
    k: usize,
) -> anyhow::Result<()> {
    let m = Mat::from_row_slice(cells.len(), k, rows);
    m.to_parquet_with_names(
        path,
        (Some(cells), Some("cell")),
        Some(&axis_id_names(prefix, k)),
    )
}

/// The likeliest lineage, `None` when nothing is known.
fn lineage_index(probs: &[f64]) -> Option<usize> {
    probs
        .iter()
        .enumerate()
        .filter(|(_, x)| x.is_finite())
        .max_by(|a, b| a.1.total_cmp(b.1))
        .map(|(l, _)| l)
}

/// What the fit is run on, from the collapse.
struct Finest {
    /// The pseudobulks' reads on the genes fitted.
    reads: Reads,
    /// The genes' names.
    names: Vec<Box<str>>,
    /// Every row's pseudobulk in the collapse's finest level, and its cells.
    pbs: Vec<usize>,
    sizes: Vec<f32>,
    /// `[D × batches]` the batch effects the cells are read with.
    delta: Option<DMatrix<f32>>,
}

/// The finest pseudobulks' reads on the genes fitted, the genes' names and
/// the batch effects the cells are read with. The reads are the
/// collapse's own rates (batch-adjusted when there are batches; the data
/// alone, no prior's mass) times the cells; the pseudobulks with none are
/// left out, of the gene selection too.
fn finest_reads(
    collapsed: Vec<data_beans::alg::collapse_data::CollapsedOut>,
    member: &[usize],
    row_names: &[Box<str>],
    args: &OdeArgs,
) -> anyhow::Result<Finest> {
    use legume_numeric::param::traits::Inference;
    let finest = collapsed.last().expect("at least one level");
    anyhow::ensure!(
        finest.stats_kept,
        "the collapse kept no statistics of its finest pseudobulks"
    );
    let rates = finest.mu_adjusted.as_ref().unwrap_or(&finest.mu_observed);
    let mut n_cells = vec![0f32; rates.ncols()];
    for &p in member {
        anyhow::ensure!(p < n_cells.len(), "a cell's pseudobulk {p} out of range");
        n_cells[p] += 1.0;
    }
    let pbs: Vec<usize> = (0..n_cells.len()).filter(|&p| n_cells[p] > 0.0).collect();
    let counts = DMatrix::<f32>::from_fn(rates.nrows(), pbs.len(), |r, c| {
        rates.evidence_mean(r, pbs[c]) * n_cells[pbs[c]]
    });
    let delta = finest.delta.as_ref().map(|d| d.posterior_mean().clone());
    drop(collapsed);
    let tracks = pb_tracks(row_names, &counts)?;
    drop(counts);
    let genes = select_genes(&tracks, args.min_reads, args.max_genes);
    anyhow::ensure!(!genes.is_empty(), "no gene with --min-reads on both tracks");
    let names = genes.iter().map(|&j| tracks.genes[j].clone()).collect();
    let reads = Reads::from_matrices(
        tracks.unspliced.select_columns(&genes).cast(),
        tracks.spliced.select_columns(&genes).cast(),
    );
    let sizes = pbs.iter().map(|&p| n_cells[p]).collect();
    Ok(Finest {
        reads,
        names,
        pbs,
        sizes,
        delta,
    })
}

pub fn run_ode(args: &OdeArgs) -> anyhow::Result<()> {
    senna::embed_common::mkdir_parent(&args.out)?;
    anyhow::ensure!(
        args.lineages > 0 && args.block_size > 0,
        "--lineages and --block-size must be positive"
    );
    let dev = args.device.to_device(args.device_no)?;

    // The data as every senna fit reads it, its finest pseudobulks.
    let inherited = args
        .from
        .as_deref()
        .map(senna::run_manifest::inherit_from)
        .transpose()?;
    use senna::run_manifest::InheritedFromManifest;
    InheritedFromManifest::ensure_replayable(inherited.as_ref(), "ode")?;
    let inputs = Inputs {
        data_files: InheritedFromManifest::resolve_data(inherited.as_ref(), &args.data_files)?,
        batch_files: InheritedFromManifest::resolve_batch(
            inherited.as_ref(),
            args.batch_files.as_deref(),
        ),
    };
    for file in &inputs.data_files {
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
    let hvg = crate::hvg::resolve_multiome_with_hvg(false, inputs.data_files.len(), &args.hvg);
    let crate::topic::common::PreparedData {
        data_vec: data,
        collapsed_levels,
        proj_kn,
        cell_to_pb_per_level,
        pb_tree,
        output_keep_idx: keep,
    } = crate::topic::common::load_and_collapse(&crate::topic::common::LoadCollapseArgs {
        data_files: &inputs.data_files,
        batch_files: &inputs.batch_files,
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
    let all_cells = data.column_names()?;
    let row_names = data.row_names()?;

    let Finest {
        reads,
        names,
        pbs,
        sizes,
        delta,
    } = finest_reads(
        collapsed_levels,
        cell_to_pb.last().expect("at least one level"),
        &row_names,
        args,
    )?;
    let (p, g) = (reads.u.nrows(), reads.u.ncols());
    info!(
        "ode: {} cells from {} files; {p} pseudobulks, {g} genes",
        all_cells.len(),
        inputs.data_files.len()
    );
    // The root pseudobulks, from the cells the user names.
    let roots = match &args.root_cells {
        None => Vec::new(),
        Some(path) => {
            let path = path.to_string_lossy();
            let named: std::collections::HashSet<Box<str>> =
                legume_numeric::matrix::common_io::read_name_list(&path)
                    .with_context(|| format!("reading --root-cells {path}"))?
                    .into_iter()
                    .collect();
            let member = cell_to_pb.last().expect("at least one level");
            let (roots, mostly) = root_pseudobulks(&named, &all_cells, member, &pbs, &sizes);
            anyhow::ensure!(
                !roots.is_empty(),
                "no pseudobulk holds a cell of --root-cells"
            );
            if mostly {
                info!(
                    "ode: {} of {} pseudobulks are mostly --root-cells ({} named)",
                    roots.len(),
                    pbs.len(),
                    named.len()
                );
            } else {
                log::warn!(
                    "ode: no pseudobulk is mostly --root-cells ({} named); the root is the one richest in them",
                    named.len()
                );
            }
            roots
        }
    };

    // The genes in groups, when they outnumber --max-coarse-features: the
    // climb on the groups' reads, every gene on its own after.
    let cap = args.coarsening.cap().map(std::num::NonZeroUsize::get);
    let fc = match cap.filter(|&c| g > c) {
        Some(cap) => {
            let fc = coarsen_genes(&reads, &sizes, cap, args.seed)?;
            info!("ode: {g} genes in {} groups", fc.num_coarse);
            Some(fc)
        }
        None => {
            info!("ode: {g} genes, not grouped");
            None
        }
    };
    let grouped = fc.as_ref().map(|fc| coarse_reads(fc, &reads));
    let fit_reads = grouped.as_ref().unwrap_or(&reads);

    // The climb, then everything at its mode.
    let opts = Climb {
        lineages: args.lineages,
        points: args.points,
        dim: args.dim,
        kappa: args.kappa,
        steps: args.steps,
        lr: args.lr,
        sub: args.sub,
        jitter: !args.no_jitter,
        trunk: args.trunk,
        roots,
    };
    let (fit, objective) = climb(fit_reads, &opts, &dev)?;
    info!("ode: the climb's objective {objective:.4}");
    let start_at = fit.params()?;
    let share = fit.share_sd()?;
    let log_w = fit.log_weights()?;
    drop(fit);
    info!(
        "ode: at the mode, the unspliced reads' log capture {:.3}, jitter sd {:?}, lineage durations {:?}, branch times {:?}",
        start_at.log_capture, share, start_at.t_len, start_at.branch
    );
    // Every other lineage's branch time off the trunk; none with them apart.
    let branch: Vec<f64> = start_at.branch.clone().unwrap_or_default();
    let mode = Mode {
        params: start_at,
        log_w,
        share_sd: share,
    };
    // Every pseudobulk's posterior on the grid and its place at the mode:
    // what the genes' pass needs.
    let grid = grid_posterior(fit_reads, &mode);
    let placed = place(fit_reads, &mode, &grid);
    let unplaced = placed.rows.iter().filter(|r| r.is_none()).count();
    if unplaced > 0 {
        log::warn!(
            "ode: {unplaced} pseudobulks the mode cannot place are left out of the genes' pass"
        );
    }

    // Every gene's rates at the mode with their Laplace sds (on groups,
    // every gene from its group's start).
    let genes = genes_at_mode(&reads, fc.as_ref(), &mode, &placed);
    write_gene_rates(&args.out, &names, &genes.log_gamma, &genes.log_ratio)?;
    if fc.is_some() {
        crate::topic::model_metadata::save_coarsening_levels(std::slice::from_ref(&fc), &args.out)?;
    }

    // The map on the trajectories' own axes, every cell placed by its reads.
    let map = MapBasis::of_trajectories(&mode, args.map_pcs)?;
    // Every cell read once, a block at a time; QC's near-empty cells out.
    let table = read_out_cells(
        &data,
        &names,
        fc.as_ref(),
        delta.as_ref(),
        keep.as_deref(),
        &mode,
        &map,
        args.block_size,
    )?;
    table.write(&args.out, map.dim())?;
    let keep = keep.as_deref();
    crate::postprocess::viz_prep::write_cell_proj(&args.out, &proj_kn, &all_cells, keep)?;
    crate::postprocess::viz_prep::write_cell_to_pb(&args.out, &cell_to_pb, &all_cells, keep)?;
    if let Some(tree) = &pb_tree {
        crate::postprocess::viz_prep::write_pb_tree(&args.out, tree, &row_names)?;
    }
    write_pb_time(&args.out, &pbs, &grid)?;
    if !branch.is_empty() {
        write_branch_times(&args.out, &branch)?;
    }
    write_manifest(args, &inputs, pb_tree.is_some(), !branch.is_empty())?;
    info!(
        "Wrote {0}.{{cell_time,cell_embedding,cell_velocity,gene_rates,pb_time}}.parquet and {0}.senna.json (and {0}.branch_time.tsv on a trunk)",
        args.out
    );
    Ok(())
}

/// Every cell's readout, as the tables have it.
struct CellTable {
    cells: Vec<Box<str>>,
    lineage: Vec<Box<str>>,
    /// `[L][C]` the lineages' posteriors.
    p_lineage: Vec<Vec<f32>>,
    tau: Vec<f32>,
    tau_sd: Vec<f32>,
    tau_lo: Vec<f32>,
    tau_hi: Vec<f32>,
    /// `[C × k]` row-major, every cell's place on the map and its velocity.
    place: Vec<f32>,
    velocity: Vec<f32>,
}

/// Every cell of `data` read out at `mode` a block at a time (QC's
/// near-empty cells, those not in `keep`, left out), placed on `map` with
/// its velocity there.
#[allow(clippy::too_many_arguments)]
fn read_out_cells(
    data: &data_beans::sparse_io_vector::SparseIoVec,
    names: &[Box<str>],
    fc: Option<&FeatureCoarsening>,
    delta: Option<&DMatrix<f32>>,
    keep: Option<&[usize]>,
    mode: &Mode,
    map: &MapBasis,
    block_size: usize,
) -> anyhow::Result<CellTable> {
    let lineages = mode.params.path.len();
    let all_cells = data.column_names()?;
    let kept_cell: Option<Vec<bool>> = keep.map(|k| {
        let mut m = vec![false; all_cells.len()];
        k.iter().for_each(|&i| m[i] = true);
        m
    });
    let mut t = CellTable {
        cells: Vec::new(),
        lineage: Vec::new(),
        p_lineage: vec![Vec::new(); lineages],
        tau: Vec::new(),
        tau_sd: Vec::new(),
        tau_lo: Vec::new(),
        tau_hi: Vec::new(),
        place: Vec::new(),
        velocity: Vec::new(),
    };
    let f32s = |v: &[f64]| v.iter().map(|&x| x as f32).collect::<Vec<_>>();
    for_cell_blocks(data, names, block_size, delta, |start, u, s| {
        let rows: Vec<usize> = (0..u.nrows())
            .filter(|i| kept_cell.as_ref().is_none_or(|k| k[start + i]))
            .collect();
        let block = Reads::from_matrices(u.select_rows(&rows).cast(), s.select_rows(&rows).cast());
        let block = match fc {
            Some(fc) => coarse_reads(fc, &block),
            None => block,
        };
        let out = read_cells(&block, mode);
        t.cells
            .extend(rows.iter().map(|&i| all_cells[start + i].clone()));
        for p in &out.lineage {
            let best = lineage_index(p);
            t.lineage
                .push(best.map_or_else(|| "NA".into(), lineage_label));
        }
        for (l, col) in t.p_lineage.iter_mut().enumerate() {
            col.extend(
                out.lineage
                    .iter()
                    .map(|p| p.get(l).map_or(f32::NAN, |&x| x as f32)),
            );
        }
        t.tau.extend(f32s(&out.time_mean));
        t.tau_sd.extend(f32s(&out.time_sd));
        t.tau_lo.extend(f32s(&out.time_lo));
        t.tau_hi.extend(f32s(&out.time_hi));
        let total = &block.u + &block.s;
        for c in 0..rows.len() {
            let x: Vec<f64> = total.row(c).iter().copied().collect();
            let dlog: Vec<f64> = out.velocity_log.row(c).iter().copied().collect();
            t.place.extend(f32s(&map.place(&x)));
            t.velocity.extend(f32s(&map.velocity(&x, &dlog)));
        }
        Ok(())
    })?;
    Ok(t)
}

impl CellTable {
    /// `{out}.cell_time.parquet`, and the map's `cell_embedding` and
    /// `cell_velocity` (`k` components).
    fn write(&self, out: &str, k: usize) -> anyhow::Result<()> {
        let p_cols: Vec<(Box<str>, Vec<f32>)> = self
            .p_lineage
            .iter()
            .enumerate()
            .map(|(l, v)| (format!("p_L{}", l + 1).into(), v.clone()))
            .collect();
        let mut cols: Vec<(Box<str>, Column)> = vec![
            ("lineage".into(), Column::Str(&self.lineage)),
            ("tau".into(), Column::F32(&self.tau)),
            ("tau_sd".into(), Column::F32(&self.tau_sd)),
            ("tau_lo".into(), Column::F32(&self.tau_lo)),
            ("tau_hi".into(), Column::F32(&self.tau_hi)),
        ];
        cols.extend(f32_columns(&p_cols));
        write_named_table(
            &format!("{out}.cell_time.parquet"),
            "cell",
            &self.cells,
            &cols,
        )?;
        write_rows(
            &format!("{out}.cell_embedding.parquet"),
            &self.cells,
            "pc",
            &self.place,
            k,
        )?;
        write_rows(
            &format!("{out}.cell_velocity.parquet"),
            &self.cells,
            "pc",
            &self.velocity,
            k,
        )
    }
}

/// `{out}.gene_rates.parquet`: every gene's `log γ` and `log γ/β` at the
/// mode with the sd of its Laplace approximation there.
fn write_gene_rates(
    out: &str,
    names: &[Box<str>],
    log_gamma: &[(f64, f64)],
    log_ratio: &[(f64, f64)],
) -> anyhow::Result<()> {
    let mut cols: Vec<(Box<str>, Vec<f32>)> = Vec::new();
    for (what, stats) in [("log_gamma", log_gamma), ("log_ratio", log_ratio)] {
        cols.push((what.into(), stats.iter().map(|s| s.0 as f32).collect()));
        cols.push((
            format!("{what}_sd").into(),
            stats.iter().map(|s| s.1 as f32).collect(),
        ));
    }
    write_named_table(
        &format!("{out}.gene_rates.parquet"),
        "gene",
        names,
        &f32_columns(&cols),
    )
}

/// Below this a pseudobulk's mass at a grid row is left out of `pb_time`.
const PB_TIME_FLOOR: f64 = 1e-6;

/// `{out}.pb_time.parquet`: every pseudobulk's posterior on the grid, a
/// row per `(pb, lineage, position)` it has mass at, `pb` its index in the
/// collapse's finest level (as `cell_to_pb` writes it), `phase` the
/// position's share of its lineage's grid (its lineage's own time, the
/// branch time's scale too). A pseudobulk the mode cannot place has no row.
fn write_pb_time(out: &str, pbs: &[usize], grid: &GridPosterior) -> anyhow::Result<()> {
    let n = grid.positions;
    let labels: Vec<Box<str>> = (0..grid.phase.len() / n).map(lineage_label).collect();
    let (mut key, mut lineage) = (Vec::<Box<str>>::new(), Vec::<Box<str>>::new());
    let (mut position, mut phase) = (Vec::<i32>::new(), Vec::<f32>::new());
    let mut q = Vec::<f32>::new();
    for (row, &pb) in grid.post.row_iter().zip(pbs) {
        let pb: Box<str> = pb.to_string().into();
        for (r, &mass) in row.iter().enumerate().filter(|(_, m)| **m >= PB_TIME_FLOOR) {
            key.push(pb.clone());
            lineage.push(labels[r / n].clone());
            position.push((r % n) as i32);
            phase.push(grid.phase[r] as f32);
            q.push(mass as f32);
        }
    }
    write_named_table(
        &format!("{out}.pb_time.parquet"),
        "pb",
        &key,
        &[
            ("lineage".into(), Column::Str(&lineage)),
            ("position".into(), Column::I32(&position)),
            ("phase".into(), Column::F32(&phase)),
            ("q".into(), Column::F32(&q)),
        ],
    )
}

/// `{out}.branch_time.tsv`: every other lineage's branch time off the
/// trunk at the mode, on the phase scale (a share of the grid, as
/// `pb_time`'s `phase`).
fn write_branch_times(out: &str, branch: &[f64]) -> anyhow::Result<()> {
    use std::fmt::Write;
    let mut tsv = String::from("lineage\tbranch_time\n");
    for (i, b) in branch.iter().enumerate() {
        writeln!(tsv, "{}\t{}", lineage_label(i + 1), b)?;
    }
    std::fs::write(format!("{out}.branch_time.tsv"), tsv)?;
    Ok(())
}

/// `{out}.senna.json`: the run as `senna view` and the layouts open it.
fn write_manifest(
    args: &OdeArgs,
    inputs: &Inputs,
    has_pb_tree: bool,
    has_branch: bool,
) -> anyhow::Result<()> {
    use senna::run_manifest::{
        record_train_args, write_run_manifest_with, RunDescription, RunKind,
    };
    let input: Vec<String> = inputs.data_files.iter().map(ToString::to_string).collect();
    let batch: Vec<String> = inputs
        .batch_files
        .as_deref()
        .map(|v| v.iter().map(ToString::to_string).collect())
        .unwrap_or_default();
    let record_time = |m: &mut senna::run_manifest::RunManifest, base: &str| {
        m.outputs.cell_time = Some(format!("{base}.cell_time.parquet"));
        m.outputs.cell_velocity = Some(format!("{base}.cell_velocity.parquet"));
        m.outputs.pb_time = Some(format!("{base}.pb_time.parquet"));
        m.outputs.branch_time = has_branch.then(|| format!("{base}.branch_time.tsv"));
    };
    write_run_manifest_with(
        &RunDescription {
            train_args: Some(record_train_args(args)?),
            kind: RunKind::Ode,
            prefix: &args.out,
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
            feature_embedding_suffix: None,
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
        },
        record_time,
    )
}
