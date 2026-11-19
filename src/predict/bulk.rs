//! Dense bulk tables as `predict` inputs.
//!
//! Everything downstream of `predict_model` streams a `SparseIoVec` in blocks:
//! `read_columns_csc`, the δ estimate, the minibatch loops, bge's projection.
//! Rather than fork those five paths for a dense matrix, a `--bulk` table is
//! materialized ONCE into a throwaway sparse backend and handed down as a
//! path, so no scoring code knows which door the data came in through. The
//! equivalence with a backend built the long way (`data-beans from-mtx`) is
//! pinned in `bulk_tests.rs`.
//!
//! The table is kept on its OWN gene axis. Alignment to the model — and with
//! it `--feature-name-kind`, `--ablate-features` and the coverage gate —
//! happens in `build_remap`, as for every other input. The model's genes are
//! consulted here only to decide which axis of the table is the gene axis.

use data_beans::sparse_io::remove_backend_path;
use legume_numeric::matrix::common_io::create_temp_dir_file;
use log::{info, warn};
use senna::embed_common::{oriented, read_labeled_mat, resolve_orientation, BulkTableOpts};
use senna::run_manifest::RunKind;
use senna::sparse_output::{write_backend, Entries};
use std::path::Path;

/// The model's gene axis, whichever family wrote it. Used only to decide the
/// orientation of a bulk table; alignment happens downstream.
///
/// A full match on purpose, no `_` arm: a family without a dictionary file
/// (`fne`, `joint-svd`, ...) must fail here with a clear message, not three
/// calls deep with a parquet path the user never typed. The depth note rides
/// on the same match so a new family is classified once, at compile time.
pub(crate) fn model_gene_axis(kind: RunKind, model: &str) -> anyhow::Result<Vec<Box<str>>> {
    if matches!(
        kind,
        RunKind::Topic | RunKind::Itopic | RunKind::MaskedVae | RunKind::JointTopic | RunKind::Vae
    ) {
        info!(
            "bulk input into a {kind} model: its encoder was trained at single-cell depth. \
             The latent is a per-sample mixture, not a composition; `senna bge` / `senna simba` / `senna svd` \
             project without an encoder, and `senna deconvolve` estimates fractions."
        );
    }
    model_gene_names(kind, model)
}

/// The model's gene axis, read from whichever file the family keeps it in.
/// Quiet: says nothing about depth, so the early gene-list gate can call it
/// for every query, bulk or not.
pub(crate) fn model_gene_names(kind: RunKind, model: &str) -> anyhow::Result<Vec<Box<str>>> {
    match kind {
        // bge and simba write no dictionary; their gene axis is the row axis
        // of the gene table. They and svd project each column against a
        // frozen table and do not care what depth it came at.
        RunKind::Bge | RunKind::Simba => {
            Ok(crate::bge::score::BgeEmbedding::open(model)?.gene_names)
        }
        RunKind::Svd
        | RunKind::Topic
        | RunKind::Itopic
        | RunKind::MaskedVae
        | RunKind::JointTopic
        | RunKind::Vae => Ok(crate::topic::model_metadata::load_dictionary(model)?.0),
        RunKind::JointSvd
        | RunKind::Fne
        | RunKind::ResolveEmbeddingSpace
        | RunKind::Rbm
        | RunKind::Ode => {
            anyhow::bail!(
                "predict: a {kind} run has no gene dictionary to align a query against, and \
             predict does not score this family"
            )
        }
    }
}

/// Temp backends standing in for dense bulk inputs. Removed on drop, so the
/// guard has to outlive the run that reads them.
#[derive(Debug)]
pub(crate) struct BulkBackends {
    paths: Vec<Box<str>>,
    /// Samples per file, for the minibatch check.
    n_samples: Vec<usize>,
}

impl BulkBackends {
    /// One backend path per `--bulk` file, in the order given.
    pub(crate) fn paths(&self) -> &[Box<str>] {
        &self.paths
    }

    /// The encoder standardizes each gene WITHIN a scored block
    /// (`anscombe_residual`), so splitting a bulk cohort across blocks changes
    /// its latent. The default block already exceeds any realistic cohort;
    /// warn when it does not.
    pub(crate) fn warn_if_split(&self, minibatch_size: usize) {
        let n: usize = self.n_samples.iter().sum();
        if n > minibatch_size {
            warn!(
                "{n} bulk samples exceed --minibatch-size {minibatch_size}; the encoder \
                 standardizes genes within a block, so the split moves the latent. Pass \
                 --minibatch-size {n} to score the cohort as one block."
            );
        }
    }
}

impl Drop for BulkBackends {
    fn drop(&mut self) {
        for p in &self.paths {
            let path = Path::new(p.as_ref());
            if let Err(e) = remove_backend_path(p) {
                warn!("could not remove temp bulk backend {p}: {e}");
                continue;
            }
            // The factory puts each temp backend in a directory of its own;
            // `remove_dir` is non-recursive, so it only succeeds when that
            // directory holds nothing else — never delete what we did not make.
            if let Some(parent) = path.parent() {
                let _ = std::fs::remove_dir(parent);
            }
        }
    }
}

/// Read each dense table, put genes on the rows, and write it as a temp
/// sparse backend with its row and column names.
///
/// Guards: negative values are refused (the backend is non-negative by
/// contract, and log-space input is the usual cause); non-integer values are
/// allowed with a warning, because the decoders are count models but
/// estimated fractional counts are legitimate input.
pub(crate) fn materialize(
    bulk_files: &[Box<str>],
    model_genes: &[Box<str>],
    opts: &BulkTableOpts,
) -> anyhow::Result<BulkBackends> {
    // Built up in place, so a failure on a later file drops it and removes
    // the backends already written.
    let mut out = BulkBackends {
        paths: Vec::with_capacity(bulk_files.len()),
        n_samples: Vec::with_capacity(bulk_files.len()),
    };
    for f in bulk_files {
        let m = read_labeled_mat(f, opts.header)?;
        let o = resolve_orientation(&m.rows, &m.cols, model_genes, opts.orientation)
            .map_err(|e| anyhow::anyhow!("{f}: {e}"))?;
        let m = oriented(m, o);

        let (any_negative, any_fractional) = m.mat.iter().fold((false, false), |(neg, frac), v| {
            (neg || *v < 0.0, frac || v.fract() != 0.0)
        });
        anyhow::ensure!(
            !any_negative,
            "{f}: the bulk table has negative values; counts cannot be negative. \
             Log-transformed input is the usual cause — pass counts."
        );
        if any_fractional {
            warn!(
                "{f}: the bulk table has non-integer values. The model scores counts; \
                 estimated fractional counts are fine, but TPM / FPKM / log-CPM are not, and \
                 the run will not notice the difference."
            );
        }
        info!(
            "{f}: bulk table {} genes × {} samples (first gene `{}`, first sample `{}`)",
            m.rows.len(),
            m.cols.len(),
            m.rows.first().map_or("", AsRef::as_ref),
            m.cols.first().map_or("", AsRef::as_ref),
        );

        // A temp `.zarr.zip` archive in a fresh directory of its own. A failed
        // write leaves that directory empty, and it goes the way `Drop` takes
        // the others: non-recursively.
        let target = create_temp_dir_file(".zarr.zip")?;
        let eff = write_backend(
            &target.to_string_lossy(),
            Entries::Dense(&m.mat),
            &m.rows,
            &m.cols,
        )
        .inspect_err(|_| {
            if let Some(dir) = target.parent() {
                let _ = std::fs::remove_dir(dir);
            }
        })?;
        info!("{f}: materialized as {eff}");
        out.paths.push(eff);
        out.n_samples.push(m.cols.len());
    }
    Ok(out)
}

#[cfg(test)]
#[path = "bulk_tests.rs"]
mod bulk_tests;
