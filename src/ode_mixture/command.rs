//! `senna ode`: read two-track count backends, collapse their cells into
//! pseudobulks (as every senna fit does), fit the mixture of splicing ODEs
//! on the finest pseudobulks and the most variable genes (the climb), and
//! read every cell out at the climb's mode, a block at a time.

use super::coarsen::{coarse_reads, coarsen_genes};
use super::counts::{pb_tracks, select_genes, two_track_genes};
use super::init::{climb, root_gap, root_pseudobulks, Climb, Climbed, TriedRoot, ROOT_GAP};
use super::likelihood::Reads;
use super::model::lineage_label;
use super::output::MapBasis;
use super::readout::{grid_posterior, place, GridPosterior, Mode};
use super::refine::genes_at_mode;
use anyhow::Context;
use legume_numeric::matrix::parquet::{write_named_table, Column};
use log::info;
use nalgebra::DMatrix;
use senna::embed_common::*;

mod cell_table;

use cell_table::{read_out_cells, CellSource};

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

/// How `senna ode` groups cells into pseudobulks.
#[derive(
    Clone, Copy, Debug, PartialEq, Eq, clap::ValueEnum, serde::Serialize, serde::Deserialize,
)]
#[serde(rename_all = "lowercase")]
pub enum PbGraph {
    /// The collapse every senna fit makes.
    Collapse,
    /// The kNN graph of the cells' splicing residuals, coarsened.
    Ratio,
}

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
        value_enum,
        default_value_t = PbGraph::Collapse,
        help = "How cells are grouped into the pseudobulks fitted",
        long_help = "How cells are grouped into the pseudobulks fitted.\n\
                     collapse: the collapse every senna fit makes (cells alike in counts).\n\
                     ratio:    cells alike in splicing and levels: each gene's unspliced share\n\
                               against its own (the cell's overall share taken out) sketched\n\
                               next to the cell's levels (the two tracks summed), the sketch's\n\
                               kNN graph coarsened into as many pseudobulks."
    )]
    pub pb_graph: PbGraph,

    #[arg(
        long,
        default_value_t = 2,
        value_parser = clap::builder::RangedU64ValueParser::<usize>::new().range(1..),
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

    #[arg(
        long,
        default_value_t = 45,
        value_parser = clap::builder::RangedU64ValueParser::<usize>::new().range(2..),
        help = "Grid points per lineage's path (at least 2)"
    )]
    pub points: usize,

    #[arg(long, default_value_t = 4, help = "Time positions per grid step")]
    pub sub: usize,

    #[arg(
        long,
        default_value_t = 3,
        value_parser = clap::builder::RangedU64ValueParser::<usize>::new().range(1..),
        help = "Dimension of the paths"
    )]
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

    #[command(flatten)]
    #[serde(flatten)]
    pub compute: senna::embed_common::DeviceArgs,

    #[arg(
        long,
        default_value_t = 1024,
        value_parser = clap::builder::RangedU64ValueParser::<usize>::new().range(1..),
        help = "Cells per readout block"
    )]
    pub(crate) block_size: usize,

    #[arg(
        long,
        default_value_t = false,
        help = "Load all columns into memory first",
        hide = true
    )]
    pub(crate) preload_data: bool,

    #[command(flatten)]
    pub(crate) hvg: data_beans::alg::hvg::HvgCliArgs,

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

/// Every one of `n` cells: is it among the cells QC `keep`s (all, with no
/// QC)?
fn keep_mask(keep: Option<&[usize]>, n: usize) -> Vec<bool> {
    let mut m = vec![keep.is_none(); n];
    keep.into_iter().flatten().for_each(|&c| m[c] = true);
    m
}

/// What the fit is run on.
struct Finest {
    /// The pseudobulks' reads on the genes fitted.
    reads: Reads,
    /// The genes' names.
    names: Vec<Box<str>>,
    /// Every row's pseudobulk (its id in the partition), and its cells.
    pbs: Vec<usize>,
    sizes: Vec<f32>,
    /// `[D × batches]` the batch effects the cells are read with.
    delta: Option<DMatrix<f32>>,
}

/// The collapse's finest pseudobulks: every row's reads `[D × pbs]`, the
/// pseudobulks with cells `pbs`, cells per pseudobulk (by id) and the batch
/// folds `[D × batches]`. The reads are the collapse's own rates
/// (batch-adjusted when there are batches; the data alone, no prior's
/// mass) times the cells.
type CollapseCounts = (DMatrix<f32>, Vec<usize>, Vec<f32>, Option<DMatrix<f32>>);

fn collapse_counts(
    collapsed: Vec<data_beans::alg::collapse_data::CollapsedOut>,
    member: &[usize],
) -> anyhow::Result<CollapseCounts> {
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
    Ok((counts, pbs, n_cells, delta))
}

/// [`Finest`] from every row's reads `counts` `[D × pbs]` in the pseudobulks
/// `pbs` (of `n_cells` cells each, by their ids): the genes fitted, and
/// their names.
fn finest_of_counts(
    counts: &DMatrix<f32>,
    pbs: Vec<usize>,
    n_cells: &[f32],
    delta: Option<DMatrix<f32>>,
    row_names: &[Box<str>],
    args: &OdeArgs,
) -> anyhow::Result<Finest> {
    let tracks = pb_tracks(row_names, counts)?;
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
    // Also when the arguments come from a manifest or MCP, not clap.
    anyhow::ensure!(
        args.lineages > 0 && args.block_size > 0 && args.dim > 0,
        "--lineages, --block-size and --dim must be positive"
    );
    anyhow::ensure!(
        args.points >= 2,
        "--points must be at least 2: a trajectory runs from one point to another"
    );
    let dev = args.compute.to_device()?;

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
    anyhow::ensure!(
        prebuilt_partition.is_none() || args.pb_graph == PbGraph::Collapse,
        "--pb-graph ratio groups the cells by their splicing, so it takes no partition \
         from another run (--pb-from, or the one --from inherits)"
    );
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
        out: Some(&args.out),
        max_features: hvg.n_hvg,
        feature_list_file: hvg.feature_list_file,
        must_train_file: hvg.must_train_file,
        refine: args.collapse.pb_refine.to_params(),
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

    let collapsed = cell_to_pb.last().expect("at least one level");
    let (counts, pbs, n_cells, delta) = collapse_counts(collapsed_levels, collapsed)?;
    let (cell_to_pb, pb_tree, finest) = match args.pb_graph {
        PbGraph::Collapse => {
            let finest = finest_of_counts(&counts, pbs, &n_cells, delta, &row_names, args)?;
            drop(counts);
            (cell_to_pb, pb_tree, finest)
        }
        PbGraph::Ratio => {
            let included = keep_mask(keep.as_deref(), all_cells.len());
            // As many pseudobulks as the collapse made of the grouped cells;
            // every row's reads over the cells, from the collapse's.
            let n_pbs = collapsed
                .iter()
                .zip(&included)
                .filter(|(_, &i)| i)
                .map(|(&p, _)| p)
                .collect::<std::collections::HashSet<_>>()
                .len();
            let row_totals: Vec<f64> = counts
                .row_iter()
                .map(|r| r.iter().map(|&x| f64::from(x)).sum())
                .collect();
            drop(counts);
            let ratio = super::ratio_pbs::ratio_pbs(
                &data,
                delta.as_ref(),
                &included,
                collapsed,
                &row_totals,
                super::ratio_pbs::RatioArgs {
                    n_pbs,
                    block_size: args.block_size,
                    seed: args.seed,
                },
            )?;
            let pbs = (0..ratio.n_cells.len()).collect();
            let finest =
                finest_of_counts(&ratio.counts, pbs, &ratio.n_cells, delta, &row_names, args)?;
            // One level, the ratio's; the collapse's tree does not hold.
            (vec![ratio.member], None, finest)
        }
    };
    let Finest {
        reads,
        names,
        pbs,
        sizes,
        delta,
    } = finest;
    let (p, g) = (reads.u.nrows(), reads.u.ncols());
    info!(
        "ode: {} cells from {} files; {p} pseudobulks, {g} genes",
        all_cells.len(),
        inputs.data_files.len()
    );
    // The root pseudobulks, from the cells the user names, and whether
    // they are mostly of them.
    let (roots, roots_mostly) = match &args.root_cells {
        None => (Vec::new(), false),
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
            (roots, mostly)
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
    let climbed = climb(fit_reads, &opts, &dev)?;
    let member = cell_to_pb.last().expect("at least one level");
    let named = args.root_cells.as_deref();
    clear_root_lists(&args.out, named)?;
    let decision = root_decision(&args.out, &climbed.roots, member, &pbs, &all_cells)?;
    let unused = climbed
        .unrooted
        .map(|why| why.note(opts.roots.len(), pbs.len(), roots_mostly, args.lineages));
    let root = root_list(&args.out, named, unused, &climbed, member, &pbs, &all_cells)?;
    let (fit, objective) = (climbed.fit, climbed.objective);
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
    let source = CellSource {
        data: &data,
        names: &names,
        fc: fc.as_ref(),
        delta: delta.as_ref(),
        keep: keep.as_deref(),
    };
    let table = read_out_cells(source, &mode, &map, args.block_size)?;
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
    write_manifest(
        args,
        &inputs,
        pb_tree.is_some(),
        !branch.is_empty(),
        Settled {
            decision,
            root,
            objective: f64::from(objective),
            lineages_held: table.lineages_held(),
        },
    )?;
    info!(
        "Wrote {0}.{{cell_time,cell_embedding,cell_velocity,gene_rates,pb_time}}.parquet and {0}.senna.json (and {0}.branch_time.tsv on a trunk)",
        args.out
    );
    Ok(())
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

/// With trees from other roots finishing within [`ROOT_GAP`] of the best
/// (no root named), the root left to the user: every root's cells, best
/// first, in `{out}.root_{i}.txt` (`member` every cell's pseudobulk, `pbs`
/// the pseudobulk of every row the climb read), and the decision whose
/// answers rerun from each.
pub(crate) fn root_decision(
    out: &str,
    roots: &[TriedRoot],
    member: &[usize],
    pbs: &[usize],
    cells: &[Box<str>],
) -> anyhow::Result<Option<senna::run_manifest::Decision>> {
    use senna::run_manifest::{root_list_path, Decision, DecisionOption};
    let finished: Vec<f32> = roots.iter().map(|r| r.objective).collect();
    let Some(gap) = root_gap(&finished).filter(|&g| g < ROOT_GAP) else {
        return Ok(None);
    };
    log::warn!(
        "ode: trees from different roots finish within {:.3}% of one another, so the data barely tell where the trajectory starts (or which way it runs); the root is left to you ({out}.root_*.txt, rerun with --root-cells)",
        100.0 * gap
    );
    let mut ranked: Vec<&TriedRoot> = roots.iter().filter(|r| r.objective.is_finite()).collect();
    ranked.sort_by(|a, b| b.objective.total_cmp(&a.objective));
    let best = ranked[0].objective;
    let mut options = Vec::with_capacity(ranked.len());
    for (i, root) in ranked.iter().enumerate() {
        let named = root_cells_of(root, member, pbs, cells);
        let path = root_list_path(out, Some(i));
        legume_numeric::matrix::common_io::write_lines(&named, &path)?;
        let path = std::path::absolute(&path)?.to_string_lossy().into_owned();
        let behind = 100.0 * (best - root.objective) / best.abs().max(f32::MIN_POSITIVE);
        options.push(DecisionOption {
            label: format!(
                "root {i}: {} cells in {} pseudobulks, its fit {behind:.4}% behind the best",
                named.len(),
                root.pbs.len()
            ),
            rerun_args: vec!["--root-cells".into(), path.clone()],
            objective: Some(f64::from(root.objective)),
            cells: Some(path),
        });
    }
    Ok(Some(Decision {
        topic: "root".into(),
        question: format!(
            "Trees grown from different roots fit within {:.3}% of one another, so the data barely tell where the trajectory starts (or which way it runs). Which root's cells start it?",
            100.0 * gap
        ),
        hint: "a trajectory's root is its least differentiated cells (stem or progenitor \
               types and markers: view_markers), and its time runs from them to the known \
               mature types (view_map with time)"
            .into(),
        options,
    }))
}

/// The cells of `root`'s pseudobulks (`member` every cell's pseudobulk,
/// `pbs` the pseudobulk of every row the climb read).
fn root_cells_of(
    root: &TriedRoot,
    member: &[usize],
    pbs: &[usize],
    cells: &[Box<str>],
) -> Vec<Box<str>> {
    let at: std::collections::HashSet<usize> = root.pbs.iter().map(|&r| pbs[r]).collect();
    cells
        .iter()
        .zip(member)
        .filter(|(_, p)| at.contains(p))
        .map(|(c, _)| c.clone())
        .collect()
}

/// The root a fit was given or took, as its manifest records it.
enum RootList {
    /// The list of the root it grew from (`root_cells`).
    Grew(String),
    /// The `--root-cells` list it was given, no tree start growing from
    /// it (`root_cells_unused`), and why ([`NoTree::note`]).
    ///
    /// [`NoTree::note`]: super::init::NoTree::note
    Unused(String, Option<String>),
}

/// The root the fit took: the list the user named (`named`, made
/// absolute) when the fit grew from it, else the cells of the tree start's
/// root it grew from, written to `{out}.root.txt` (its name beside the
/// manifest); a named list no tree start grew from, as unused (`unused`
/// why); `None` when a k-means start won with no root named.
fn root_list(
    out: &str,
    named: Option<&std::path::Path>,
    unused: Option<String>,
    climbed: &Climbed,
    member: &[usize],
    pbs: &[usize],
    cells: &[Box<str>],
) -> anyhow::Result<Option<RootList>> {
    let named = named
        .map(|n| std::path::absolute(n).map(|n| n.to_string_lossy().into_owned()))
        .transpose()?;
    let Some(root) = climbed.root.map(|r| &climbed.roots[r]) else {
        return Ok(named.map(|list| RootList::Unused(list, unused)));
    };
    if let Some(named) = named {
        return Ok(Some(RootList::Grew(named)));
    }
    let path = senna::run_manifest::root_list_path(out, None);
    legume_numeric::matrix::common_io::write_lines(
        &root_cells_of(root, member, pbs, cells),
        &path,
    )?;
    let name = std::path::Path::new(&path).file_name().unwrap_or_default();
    Ok(Some(RootList::Grew(name.to_string_lossy().into_owned())))
}

/// Every root list an earlier run into `out` left
/// ([`root_list_path`](senna::run_manifest::root_list_path)), removed: what
/// is there after a run is this run's own, but for `keep` (the user's
/// `--root-cells`, which may be one of them).
pub(crate) fn clear_root_lists(out: &str, keep: Option<&std::path::Path>) -> anyhow::Result<()> {
    let path = std::path::Path::new(out);
    let Some(name) = path.file_name() else {
        return Ok(());
    };
    let dir = path
        .parent()
        .filter(|d| !d.as_os_str().is_empty())
        .unwrap_or(std::path::Path::new("."));
    let Ok(entries) = std::fs::read_dir(dir) else {
        return Ok(());
    };
    for entry in entries.flatten() {
        let file = entry.file_name().to_string_lossy().into_owned();
        let is_list = senna::run_manifest::root_list_of(&file)
            .is_some_and(|(of, _)| std::ffi::OsStr::new(of) == name);
        let kept = keep.is_some_and(|k| senna::run_manifest::same_file(k, &entry.path()));
        if is_list && !kept {
            std::fs::remove_file(entry.path())?;
        }
    }
    Ok(())
}

/// What the fit settled, as its manifest records it.
struct Settled {
    /// What it left to decide.
    decision: Option<senna::run_manifest::Decision>,
    root: Option<RootList>,
    objective: f64,
    /// The lineages some cell is most on ([`cell_table::CellTable::lineages_held`]).
    lineages_held: usize,
}

/// `{out}.senna.json`: the run as `senna view` and the layouts open it,
/// with what the fit settled.
fn write_manifest(
    args: &OdeArgs,
    inputs: &Inputs,
    has_pb_tree: bool,
    has_branch: bool,
    Settled {
        decision,
        root,
        objective,
        lineages_held,
    }: Settled,
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
        m.outputs.cell_time_lineages = Some(lineages_held);
        match root {
            Some(RootList::Grew(list)) => m.outputs.root_cells = Some(list),
            Some(RootList::Unused(list, why)) => {
                m.outputs.root_cells_unused = Some(list);
                m.outputs.root_cells_unused_why = why;
            }
            None => {}
        }
        m.decisions.extend(decision);
        m.objective = Some(objective);
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
