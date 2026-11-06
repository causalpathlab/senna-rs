//! `senna ode`: read two-track count backends, collapse their cells into
//! pseudobulks (as every senna fit does), fit the mixture of splicing ODEs
//! on the finest pseudobulks and the most variable genes (the climb, then
//! the chains), and read every cell out from the chains' draws, a block at
//! a time.

use super::counts::{for_cell_blocks, pb_tracks, select_genes, two_track_genes};
use super::direction::{direction_of, Call};
use super::init::{climb, Climb};
use super::likelihood::Reads;
use super::output::{chain_stats, ChainStat, MapBasis};
use super::readout::{read_cells, Snapshot};
use super::sampler::{self, Draws};
use legume_numeric::matrix::parquet::{write_named_table, Column};
use log::info;
use nalgebra::DMatrix;
use senna::embed_common::*;

#[derive(Args, Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(default = "senna::embed_common::clap_defaults")]
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
        help = "Lineages (with more than one, every other on lineage 0's trunk until its branch time unless --no-trunk)"
    )]
    pub lineages: usize,

    #[arg(
        long,
        default_value_t = false,
        help = "Lineages apart, not on a shared trunk"
    )]
    pub no_trunk: bool,

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

    #[arg(long, default_value_t = 4, help = "Chains")]
    pub chains: usize,

    #[arg(
        long,
        default_value_t = 1500,
        help = "Sweeps per chain, burn-in included"
    )]
    pub sweeps: usize,

    #[arg(long, default_value_t = 250, help = "Burn-in sweeps (not kept)")]
    pub burn: usize,

    #[arg(
        long,
        default_value_t = 5,
        help = "During burn-in, refit the moves' proposals every this many sweeps (frozen after)"
    )]
    pub laplace_every: usize,

    #[arg(
        long,
        default_value_t = 0,
        help = "Keep every this many kept sweeps' parameters for the cells' readout (0: 25 a chain)"
    )]
    pub snapshot_every: usize,

    #[arg(
        long,
        default_value_t = false,
        help = "Skip every lineage's direction evidence (two held refits a lineage)"
    )]
    pub no_direction: bool,

    #[arg(
        long,
        default_value_t = 30,
        help = "Principal components of the map space"
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
    pub(crate) qc: QcArgs,
}

/// Snapshots a chain keeps for the readout when `--snapshot-every` is 0.
const SNAPSHOTS_A_CHAIN: usize = 25;

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
    Ok(Finest {
        reads,
        names,
        delta,
    })
}

/// Every chain's draws of one per-gene quantity, `[chain][draw][gene]`.
fn per_chain(draws: &[Draws], of: impl Fn(&Draws) -> &Vec<Vec<f64>>) -> Vec<&Vec<Vec<f64>>> {
    draws.iter().map(of).collect()
}

pub fn run_ode(args: &OdeArgs) -> anyhow::Result<()> {
    senna::embed_common::mkdir_parent(&args.out)?;
    anyhow::ensure!(
        args.lineages > 0 && args.chains > 0 && args.block_size > 0,
        "--lineages, --chains and --block-size must be positive"
    );
    anyhow::ensure!(
        args.sweeps > args.burn,
        "--sweeps must exceed --burn, so some sweeps are kept"
    );
    let stop = legume_numeric::matrix::stop::stop_flag();
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

    // The climb, then the chains from its mode.
    let opts = Climb {
        lineages: args.lineages,
        points: args.points,
        dim: args.dim,
        kappa: args.kappa,
        steps: args.steps,
        lr: args.lr,
        sub: args.sub,
        jitter: !args.no_jitter,
        trunk: !args.no_trunk,
    };
    let (fit, objective) = climb(&reads, &opts, &dev)?;
    info!("ode: the climb's objective {objective:.4}");
    // Every lineage's state in the output, as each of its cells reports it.
    let (fit, runs) = if args.no_direction {
        (fit, vec![None; args.lineages])
    } else {
        lineage_directions(&args.out, fit, objective, &reads, &opts, &dev)?
    };
    let start_at = fit.params()?;
    let share = fit.share_sd()?;
    drop(fit);
    let kept = args.sweeps - args.burn;
    let snapshot_every = if args.snapshot_every > 0 {
        args.snapshot_every
    } else {
        kept.div_ceil(SNAPSHOTS_A_CHAIN).max(1)
    };
    let mut draws = sampler::run(
        &reads,
        &start_at,
        &sampler::Options {
            kappa: args.kappa,
            chains: args.chains,
            sweeps: args.sweeps,
            burn: args.burn,
            seed: args.seed,
            stop: Some(stop),
            share,
            snapshot_every,
            laplace_every: args.laplace_every,
            ..sampler::Options::default()
        },
    )?;
    report_chains(&draws);

    // Nothing to report or read out from chains stopped in burn-in.
    let snaps: Vec<Snapshot> = draws
        .iter_mut()
        .flat_map(|d| std::mem::take(&mut d.snapshots))
        .collect();
    anyhow::ensure!(
        !snaps.is_empty(),
        "no snapshot kept: the chains stopped too early"
    );

    // Every gene's rates over the chains.
    let log_gamma = chain_stats(&per_chain(&draws, |d| &d.log_gamma));
    let log_ratio = chain_stats(&per_chain(&draws, |d| &d.log_ratio));
    write_gene_rates(&args.out, &names, &log_gamma, &log_ratio)?;

    // Every cell read once, a block at a time; QC's near-empty cells out.
    let map = {
        let x = &reads.u + &reads.s;
        MapBasis::new(&x, args.map_pcs)?
    };
    let table = read_out_cells(
        &data,
        &names,
        delta.as_ref(),
        keep.as_deref(),
        &snaps,
        &map,
        &runs,
        args.block_size,
    )?;
    table.write(&args.out, map.dim())?;
    let keep = keep.as_deref();
    crate::postprocess::viz_prep::write_cell_proj(&args.out, &proj_kn, &all_cells, keep)?;
    crate::postprocess::viz_prep::write_cell_to_pb(&args.out, &cell_to_pb, &all_cells, keep)?;
    if let Some(tree) = &pb_tree {
        crate::postprocess::viz_prep::write_pb_tree(&args.out, tree, &row_names)?;
    }
    write_manifest(args, &inputs, pb_tree.is_some())?;
    info!(
        "Wrote {0}.{{cell_time,cell_embedding,cell_velocity,gene_rates}}.parquet and {0}.senna.json (and {0}.lineages.tsv unless --no-direction)",
        args.out
    );
    Ok(())
}

/// How every lineage runs in the output, `None` where its direction was
/// not called (`--no-direction`).
type Runs = Vec<Option<Call>>;

/// A lineage's state as its cells report it.
fn run_label(run: Option<Call>) -> &'static str {
    run.map_or("not called", Call::as_str)
}

/// Every cell's readout, as the tables have it.
struct CellTable {
    cells: Vec<Box<str>>,
    lineage: Vec<Box<str>>,
    direction: Vec<Box<str>>,
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

/// Every cell of `data` read out from `snaps` a block at a time (QC's
/// near-empty cells, those not in `keep`, left out), placed on `map` with
/// its velocity there, and its lineage's state from `runs`.
#[allow(clippy::too_many_arguments)]
fn read_out_cells(
    data: &data_beans::sparse_io_vector::SparseIoVec,
    names: &[Box<str>],
    delta: Option<&DMatrix<f32>>,
    keep: Option<&[usize]>,
    snaps: &[Snapshot],
    map: &MapBasis,
    runs: &Runs,
    block_size: usize,
) -> anyhow::Result<CellTable> {
    let all_cells = data.column_names()?;
    let kept_cell: Option<Vec<bool>> = keep.map(|k| {
        let mut m = vec![false; all_cells.len()];
        k.iter().for_each(|&i| m[i] = true);
        m
    });
    let mut t = CellTable {
        cells: Vec::new(),
        lineage: Vec::new(),
        direction: Vec::new(),
        p_lineage: vec![Vec::new(); runs.len()],
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
        let out = read_cells(&block, snaps);
        t.cells
            .extend(rows.iter().map(|&i| all_cells[start + i].clone()));
        for p in &out.lineage {
            let best = lineage_index(p);
            t.lineage
                .push(best.map_or_else(|| "NA".into(), |l| format!("L{}", l + 1).into()));
            t.direction
                .push(best.map_or("NA", |l| run_label(runs[l])).into());
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
            ("direction".into(), Column::Str(&self.direction)),
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

/// `{out}.lineages.tsv`: every lineage's direction evidence from `fit`
/// ([`direction_of`]: the climbed order held, refitted both ways round)
/// and its [`Call`]. A lineage the evidence reverses is turned round (the
/// fit carried on is the held refit turned round) only with the lineages
/// apart: on a shared trunk every lineage starts at the trunk's root, so
/// one cannot run the other way alone, and it is reported, not turned.
/// The fit carried on, and how every lineage runs in it.
fn lineage_directions(
    out: &str,
    mut fit: super::fit::Fit,
    mut objective: f32,
    reads: &Reads,
    opts: &Climb,
    dev: &legume_numeric::candle::candle_core::Device,
) -> anyhow::Result<(super::fit::Fit, Runs)> {
    let mut runs: Runs = Vec::with_capacity(opts.lineages);
    let mut tsv = String::from("lineage\tevidence\tper_pseudobulk\tdirection\tturned\n");
    for l in 0..opts.lineages {
        let on_trunk = fit.branch_at()?.is_some();
        let dir = direction_of(fit, objective, reads, opts, l, dev)?;
        let (evidence, call) = (dir.evidence(), dir.call());
        let per = evidence / dir.rows.len().max(1) as f64;
        let turned = call == Call::Reversed && !on_trunk;
        info!(
            "ode: lineage {}'s direction {}: evidence {evidence:.1} ({per:.2} a pseudobulk){}",
            l + 1,
            call.as_str(),
            if turned {
                "; turned round"
            } else if call == Call::Reversed {
                "; on the trunk, not turned"
            } else {
                ""
            }
        );
        tsv.push_str(&format!(
            "L{}\t{evidence:.3}\t{per:.4}\t{}\t{}\n",
            l + 1,
            call.as_str(),
            if turned { "yes" } else { "no" }
        ));
        // How the lineage runs in the output: a reversal turned round now
        // runs forward; one left (on the trunk) still runs backwards.
        runs.push(Some(if turned { Call::Forward } else { call }));
        (fit, objective) = if turned {
            (dir.reversed, dir.reversed_objective)
        } else {
            (dir.fit, dir.objective)
        };
    }
    std::fs::write(format!("{out}.lineages.tsv"), tsv)?;
    Ok((fit, runs))
}

/// The chains' lineage durations and branch times, and how well they mix.
fn report_chains(draws: &[Draws]) {
    let say = |what: &str, stats: &[ChainStat]| {
        for (i, s) in stats.iter().enumerate() {
            info!(
                "ode: {what} {}: {:.3} ± {:.3} (R̂ {:.3}, ESS {:.0}){}",
                i + 1,
                s.mean,
                s.sd,
                s.rhat,
                s.ess,
                if s.flagged() { " — not mixed" } else { "" }
            );
        }
    };
    say(
        "log duration of lineage",
        &chain_stats(&per_chain(draws, |d| &d.log_t)),
    );
    say(
        "branch time of lineage",
        &chain_stats(&per_chain(draws, |d| &d.branch)),
    );
}

/// `{out}.gene_rates.parquet`: every gene's `log γ` and `log γ/β` over the
/// chains, with R̂ and ESS, and whether either is flagged as not mixed.
fn write_gene_rates(
    out: &str,
    names: &[Box<str>],
    log_gamma: &[ChainStat],
    log_ratio: &[ChainStat],
) -> anyhow::Result<()> {
    let mut cols: Vec<(Box<str>, Vec<f32>)> = Vec::new();
    for (what, stats) in [("log_gamma", log_gamma), ("log_ratio", log_ratio)] {
        let col = |f: fn(&ChainStat) -> f64| stats.iter().map(|s| f(s) as f32).collect();
        cols.push((what.into(), col(|s| s.mean)));
        cols.push((format!("{what}_sd").into(), col(|s| s.sd)));
        cols.push((format!("{what}_rhat").into(), col(|s| s.rhat)));
        cols.push((format!("{what}_ess").into(), col(|s| s.ess)));
    }
    let flagged: Vec<f32> = log_gamma
        .iter()
        .zip(log_ratio)
        .map(|(a, b)| f32::from(u8::from(a.flagged() || b.flagged())))
        .collect();
    info!(
        "ode: {} of {} genes' rates flagged as not mixed (R̂ > {})",
        flagged.iter().filter(|&&f| f > 0.0).count(),
        names.len(),
        super::output::RHAT_FLAG
    );
    cols.push(("flagged".into(), flagged));
    write_named_table(
        &format!("{out}.gene_rates.parquet"),
        "gene",
        names,
        &f32_columns(&cols),
    )
}

/// `{out}.senna.json`: the run as `senna view` and the layouts open it.
fn write_manifest(args: &OdeArgs, inputs: &Inputs, has_pb_tree: bool) -> anyhow::Result<()> {
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
