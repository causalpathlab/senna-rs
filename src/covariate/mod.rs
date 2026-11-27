//! senna's wiring of the shared per-cell covariates
//! ([`genomic_data::covariate`], one implementation for senna, pinto and
//! chickpea): the loaded cells scored and binned, their batch labels made
//! `batch|cvK` so the collapse's per-gene δ (and bge's gene fold) is fitted
//! per batch and bin, and, beside the run, the model, every cell's scores,
//! one batch file per data file (which the manifest records) and each gene's
//! covariate effect. `predict` bins held-out cells with the recorded model.

pub mod outputs;
#[cfg(test)]
mod tests;

use genomic_data::covariate::files;
pub use genomic_data::covariate::files::model as model_path;
pub use genomic_data::covariate::{CovariateArgs, CovariateModel};

use data_beans::sparse_io_vector::SparseIoVec;
use genomic_data::covariate::ColumnBlocks;
use nalgebra::DMatrix;
use nalgebra_sparse::CscMatrix;

/// A loaded `SparseIoVec` as the shared module reads it.
pub struct Blocks<'a>(pub &'a SparseIoVec);

impl ColumnBlocks for Blocks<'_> {
    fn n_cols(&self) -> usize {
        self.0.num_columns()
    }
    fn row_names(&self) -> anyhow::Result<Vec<Box<str>>> {
        self.0.row_names()
    }
    fn block(&self, lb: usize, ub: usize) -> anyhow::Result<CscMatrix<f32>> {
        self.0.read_columns_csc(lb..ub)
    }
}

/// Whether this load can model covariates, checked before anything is read:
/// not multiome (senna writes its batch files per data file, one location
/// per column) and not a carried pseudobulk reference (columns that are not
/// cells). The default `all` says so in the log; a list is refused.
///
/// # Errors
/// Covariates named for such a load, or a malformed `--covariate`.
pub fn loadable(args: &CovariateArgs, multiome: bool, pb_reference: bool) -> anyhow::Result<bool> {
    args.enabled(&[
        (
            multiome,
            "multiome loads, where cells are glued across files",
        ),
        (
            pb_reference,
            "a carried pseudobulk reference (--pb-reference)",
        ),
    ])
}

/// The covariates as modelled for a fit: each loaded cell's `batch|cvK`
/// label and, with an output prefix, the batch files written for the run
/// to record.
pub struct Modelled {
    pub labels: Vec<Box<str>>,
    pub batch_files: Option<Vec<Box<str>>>,
}

/// Where a fit writes its covariate model and batch files, and what the
/// plain labels of columns QC set aside come from
/// ([`outputs::column_layout`]).
pub struct CovariateOutput<'a> {
    pub out: &'a str,
    pub batch_files: Option<&'a [Box<str>]>,
    pub ignore_batch: bool,
}

/// Each loaded cell's `batch|cvK` label ([`genomic_data::covariate::model_covariates`]
/// over every row) and, with `output`, the model, the cells' scores and the
/// batch files written; `None` when nothing is modelled.
///
/// # Errors
/// As the shared module, or a file that cannot be written.
pub fn model_covariates(
    data: &SparseIoVec,
    batch: &[Box<str>],
    proj_kn: &DMatrix<f32>,
    args: &CovariateArgs,
    block: usize,
    output: Option<CovariateOutput<'_>>,
    reread: impl FnOnce() -> anyhow::Result<data_beans::aux::data_loading::SparseDataWithBatch>,
) -> anyhow::Result<Option<Modelled>> {
    let Some(m) = genomic_data::covariate::model_covariates(
        &Blocks(data),
        batch,
        proj_kn,
        args,
        None,
        block,
    )?
    else {
        return Ok(None);
    };
    let batch_files = output
        .map(|o| {
            senna::embed_common::mkdir_parent(o.out)?;
            m.model.write(std::path::Path::new(&model_path(o.out)))?;
            genomic_data::covariate::write_cells(
                std::path::Path::new(&files::cells(o.out)),
                &m.model,
                &data.column_names()?,
                &m.scores,
                &m.labels,
            )?;
            let layout = outputs::column_layout(data, o.batch_files, o.ignore_batch, reread)?;
            let written = outputs::write_batch_files(o.out, &m.labels, &layout)?;
            log::info!(
                "--covariate: wrote {}, {} and {} batch file(s) {}",
                model_path(o.out),
                files::cells(o.out),
                written.len(),
                files::batch(o.out, 0).replace(".0.", ".*."),
            );
            anyhow::Ok(written)
        })
        .transpose()?;
    Ok(Some(Modelled {
        labels: m.labels,
        batch_files,
    }))
}

/// The batch files of covariate labels a run recorded, for a fit reusing its
/// preprocessing (`--pb-from`, which has already refused a run of other
/// collapse settings, covariates and `--ignore-batch` among them): `Some`
/// when it recorded a covariate model; `None`, and the fit models them
/// afresh, otherwise.
#[must_use]
pub fn recorded_batch_files(
    m: &senna::run_manifest::RunManifest,
    dir: &std::path::Path,
    prefix: &str,
) -> Option<Vec<Box<str>>> {
    (!m.data.batch.is_empty() && std::path::Path::new(&model_path(prefix)).is_file()).then(|| {
        m.data
            .batch
            .iter()
            .map(|b| {
                senna::run_manifest::resolve(dir, b)
                    .to_string_lossy()
                    .into()
            })
            .collect()
    })
}

/// A reused run's covariate model and cell scores, copied beside `out` for
/// predict to read (nothing to copy when `out` is that run's prefix).
///
/// # Errors
/// A file that cannot be copied.
pub fn copy_recorded(prefix: &str, out: &str) -> anyhow::Result<()> {
    senna::embed_common::mkdir_parent(out)?;
    for (from, to) in [
        (model_path(prefix), model_path(out)),
        (files::cells(prefix), files::cells(out)),
    ] {
        if std::path::Path::new(&from).is_file()
            && !senna::run_manifest::same_file(
                std::path::Path::new(&from),
                std::path::Path::new(&to),
            )
        {
            std::fs::copy(&from, &to)?;
        }
    }
    Ok(())
}

/// Held-out cells' batch labels as the run at `model` binned its own: their
/// plain `batch` made `batch|cvK` by its recorded model, or kept as they are
/// when it recorded none.
///
/// # Errors
/// A recorded model that cannot be read, cells that cannot be scored, or a
/// run fitted with the former `--cell-cycle`, whose `batch|ccK` δ held-out
/// cells cannot be binned for.
pub fn query_batches(
    model: &str,
    data: &SparseIoVec,
    batch: Vec<Box<str>>,
) -> anyhow::Result<Vec<Box<str>>> {
    let prefix = senna::run_manifest::derive_out_prefix(model);
    let path = model_path(&prefix);
    Ok(match CovariateModel::read(std::path::Path::new(&path))? {
        Some(cm) => {
            log::info!(
                "Query batches binned by {path} ({} bins)",
                cm.bins.mixture.k()
            );
            cm.labels(&Blocks(data), &batch, None, 1024)?
        }
        None => {
            let cycle = format!("{prefix}.cell_cycle_model.json");
            anyhow::ensure!(
                !std::path::Path::new(&cycle).is_file(),
                "{model} was fitted with --cell-cycle ({cycle}), whose batch|ccK δ held-out \
                 cells cannot be binned for: refit it with --covariate"
            );
            batch
        }
    })
}
