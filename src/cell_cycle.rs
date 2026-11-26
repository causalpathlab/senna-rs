//! `senna cell-cycle`: each cell's S and G2/M scores, phase and cycle angle,
//! and optionally the genes that move with them.
//!
//! The genes are Tirosh et al. 2016's (Science 352:189), bundled in
//! legume-genomic-types ([`genomic_data::cell_cycle`]), or a table of the
//! user's own in the same form (`--cycle-genes`). When too few of them
//! match the features (another species, other names), nothing is scored:
//! the command says why and writes nothing, rather than failing.
//!
//! Writes `{out}.cell_cycle.parquet` (cell, `s_score`, `g2m_score`,
//! `phase`, `angle`), a table `senna view`'s `values` colours by, and with
//! `--learn` `{out}.cell_cycle.genes.parquet`: every gene's correlation
//! with the two scores and whether it is in the sets, the genes past
//! `--learn-min-r` the data's own cycle programme, to audit.
//!
//! With `--from RUN`, also what the scores say about the run ([`check`]):
//! each latent dimension's R² on the scores, and each cluster's phase
//! make-up, a cluster at least 95% one phase marked.

mod check;
pub(crate) mod covariate;
mod score;
#[cfg(test)]
mod tests;

pub use score::{Phase, Scores};

use clap::Args;
use data_beans::aux::data_loading::{read_data_on_shared_rows, ReadSharedRowsArgs};
use genomic_data::cell_cycle::CycleGenes;
use legume_numeric::matrix::parquet::{write_named_table, Column};
use log::{info, warn};

#[derive(Args, Debug)]
pub struct CellCycleArgs {
    #[arg(
        value_delimiter = ',',
        required_unless_present = "from",
        help = "Data files to score (.zarr, .zarr.zip or .h5)"
    )]
    data_files: Vec<Box<str>>,

    #[arg(
        long,
        conflicts_with = "data_files",
        help = "Score a run's own cells: its manifest ({out}.senna.json)"
    )]
    from: Option<Box<str>>,

    #[arg(short, long, required = true, help = "Output file prefix")]
    out: Box<str>,

    #[arg(
        long,
        value_name = "FILE",
        help = "Cell-cycle genes of your own: tab-separated phase (S or G2M), symbol and optionally a current symbol [default: Tirosh et al. 2016]"
    )]
    cycle_genes: Option<Box<str>>,

    #[arg(
        long,
        help = "Also write every gene's correlation with the S and G2/M scores ({out}.cell_cycle.genes.parquet)"
    )]
    learn: bool,

    #[arg(
        long,
        default_value_t = 0.3,
        help = "With --learn: genes positively correlated past this with either score count as the data's cycle programme"
    )]
    learn_min_r: f32,

    #[arg(
        long,
        default_value_t = 1024,
        help = "Cells read at a time (blocks are read in parallel, so memory grows with it)"
    )]
    block_size: usize,

    #[arg(long, default_value_t = 42, help = "Seed for drawing control genes")]
    seed: u64,

    #[arg(long, help = "Read the data into memory first")]
    preload_data: bool,
}

/// The gene sets asked for: the user's file, else Tirosh et al. 2016.
fn cycle_genes(path: Option<&str>) -> anyhow::Result<CycleGenes> {
    match path {
        Some(p) => CycleGenes::read(std::path::Path::new(p)).map_err(anyhow::Error::msg),
        None => Ok(CycleGenes::tirosh_2016()),
    }
}

pub fn run_cell_cycle(args: &CellCycleArgs) -> anyhow::Result<()> {
    let genes = cycle_genes(args.cycle_genes.as_deref())?;
    let (data_files, batch_files) = match &args.from {
        Some(m) => {
            let inh = senna::run_manifest::inherit_from(m)?;
            senna::run_manifest::InheritedFromManifest::ensure_replayable(
                Some(&inh),
                "cell-cycle",
            )?;
            (
                inh.data_files,
                (!inh.batch_files.is_empty()).then_some(inh.batch_files),
            )
        }
        None => (args.data_files.clone(), None),
    };
    let loaded = read_data_on_shared_rows(ReadSharedRowsArgs {
        data_files,
        batch_files,
        preload: args.preload_data,
        // Every cell the run has, not a QC'd subset.
        keep_empty_barcodes: true,
        ..Default::default()
    })?;
    let data = loaded.data;
    info!(
        "cell cycle: {} genes × {} cells, genes from {}",
        data.num_rows(),
        data.num_columns(),
        genes.source
    );
    let scores = match score::score(&data, &genes, args.block_size, args.learn, args.seed)? {
        Ok(s) => s,
        Err(why) => {
            warn!("cell cycle not scored: {why}");
            return Ok(());
        }
    };
    info!(
        "matched {} S and {} G2/M genes",
        scores.s_rows.len(),
        scores.g2m_rows.len()
    );
    senna::embed_common::mkdir_parent(&args.out)?;
    let cells = data.column_names()?;
    let path = format!("{}.cell_cycle.parquet", args.out);
    write_scores(&path, &cells, &scores)?;
    info!("wrote {path}");
    log_phases(&scores.phase);

    if let Some((r_s, r_g2m)) = &scores.learned {
        let path = format!("{}.cell_cycle.genes.parquet", args.out);
        let names = data.row_names()?;
        let n_prog = write_genes(&path, &names, &scores, r_s, r_g2m, args.learn_min_r)?;
        info!(
            "wrote {path}: {n_prog} genes correlated past r = {} with S or G2/M",
            args.learn_min_r
        );
    }
    // The run's diagnostics come last: one that cannot be read leaves the
    // scores and genes written.
    if let Some(m) = &args.from {
        if let Err(e) = check::check(m, &cells, &scores, &args.out) {
            warn!("cell cycle diagnostics of {m} skipped: {e:#}");
        }
    }
    Ok(())
}

fn write_scores(path: &str, cells: &[Box<str>], s: &Scores) -> anyhow::Result<()> {
    let phase: Vec<Box<str>> = s.phase.iter().map(|p| p.name().into()).collect();
    write_named_table(
        path,
        "cell",
        cells,
        &[
            ("s_score".into(), Column::F32(&s.s)),
            ("g2m_score".into(), Column::F32(&s.g2m)),
            ("phase".into(), Column::Str(&phase)),
            ("angle".into(), Column::F32(&s.angle)),
        ],
    )
}

/// Every gene's correlations and set membership; returns how many pass
/// `min_r` with either score.
fn write_genes(
    path: &str,
    names: &[Box<str>],
    s: &Scores,
    r_s: &[f32],
    r_g2m: &[f32],
    min_r: f32,
) -> anyhow::Result<usize> {
    let mut set: Vec<&str> = vec![""; names.len()];
    for &g in &s.s_rows {
        set[g] = "S";
    }
    for &g in &s.g2m_rows {
        set[g] = if set[g] == "S" { "S,G2M" } else { "G2M" };
    }
    let set: Vec<Box<str>> = set.into_iter().map(Box::from).collect();
    let programme: Vec<i32> = r_s
        .iter()
        .zip(r_g2m)
        .map(|(&a, &b)| i32::from(a.max(b) >= min_r))
        .collect();
    write_named_table(
        path,
        "gene",
        names,
        &[
            ("r_s".into(), Column::F32(r_s)),
            ("r_g2m".into(), Column::F32(r_g2m)),
            ("in_set".into(), Column::Str(&set)),
            ("programme".into(), Column::I32(&programme)),
        ],
    )?;
    Ok(programme.iter().filter(|&&p| p == 1).count())
}

fn log_phases(phase: &[Phase]) {
    let n = phase.len().max(1) as f32;
    let share = |p: Phase| 100.0 * phase.iter().filter(|&&q| q == p).count() as f32 / n;
    info!(
        "phases: G1 {:.1}%, S {:.1}%, G2/M {:.1}%, undecided {:.1}%",
        share(Phase::G1),
        share(Phase::S),
        share(Phase::G2M),
        share(Phase::Undecided)
    );
}
