//! Multi-file input resolution and loading for `senna tde`.
//!
//! Every gene-count file gets a sample id, the key that tags its barcodes
//! (`{barcode}@{sample}`) so files of different samples stay apart when the
//! files are merged by barcode. [`sample_ids`] derives the ids;
//! [`load_tde_data`] does the load (`Union` column alignment, exact row-name
//! matching: the row itself is the join key) and hands the result to
//! [`crate::tde::tracks::assign_tracks`].

use graph_embedding_util as ge;
use legume_numeric::matrix::common_io::basename;
use log::info;

use crate::tde::tracks::{assign_tracks, TrackPlan};

/// The suffix `faba count` gives its gene-count matrices (`{sample}_count`).
const COUNT_SUFFIX: &str = "_count";

/// Each gene file's sample id: `strip` stripped from its basename when
/// non-empty; else the `sample` its producer wrote into its metadata; else,
/// for a file written before that, `_count` stripped from its basename. A
/// basename that does not end with the suffix keeps its full name, and is
/// warned.
pub(crate) fn sample_ids(genes: &[Box<str>], strip: &str) -> anyhow::Result<Vec<Box<str>>> {
    anyhow::ensure!(
        !genes.is_empty(),
        "no gene matrices given: pass them positionally \
         (`senna tde out/*_count.zarr.zip -o out/tde`)"
    );
    genes
        .iter()
        .map(|file| sample_id_for(file, strip))
        .collect()
}

fn sample_id_for(file: &str, strip: &str) -> anyhow::Result<Box<str>> {
    if strip.is_empty() {
        if let Some(sample) = recorded_sample(file) {
            return Ok(sample.into());
        }
        log::info!("{file}: no sample in its metadata; reading it from the file name");
    }
    let base = basename(file)?;
    let suffix = if strip.is_empty() {
        COUNT_SUFFIX
    } else {
        strip
    };
    match base.strip_suffix(suffix) {
        Some(sid) => Ok(sid.into()),
        None => {
            log::warn!(
                "{file}: basename {base:?} does not end with {suffix:?}; using the full \
                 basename as its sample id"
            );
            Ok(base)
        }
    }
}

/// The sample the file's producer recorded in its metadata, if any.
fn recorded_sample(file: &str) -> Option<String> {
    use data_beans::sparse_io::{meta, open_sparse_matrix_by_path};
    let m = open_sparse_matrix_by_path(file).ok()?;
    m.meta(meta::SAMPLE).filter(|s| !s.trim().is_empty())
}

/// Load the gene files into one [`ge::UnifiedData`] and assign its
/// [`TrackPlan`].
///
/// `Union` column alignment (cells merge by barcode), with each file's
/// barcodes tagged `@{sample}` whenever more than one file is given, including
/// with `--batch-files`, which then names the batch of each already-tagged
/// barcode. Row names are matched exactly across files: the row itself
/// (`{gene}/count/{channel}`) is the join key.
pub(crate) fn load_tde_data(
    files: &[Box<str>],
    sample_ids: &[Box<str>],
    batch_files: Option<&[Box<str>]>,
    preload: bool,
) -> anyhow::Result<(ge::UnifiedData, TrackPlan)> {
    let per_file_barcode_suffix: Option<Vec<Option<Box<str>>>> =
        (files.len() > 1).then(|| sample_ids.iter().cloned().map(Some).collect());

    let unified = ge::load_unified_data(ge::LoadUnifiedArgs {
        data_files: files.to_vec(),
        batch_files: batch_files.map(<[Box<str>]>::to_vec),
        feature_kind: Some(ge::FeatureNameKind::Exact),
        preload,
        column_alignment: data_beans::sparse_io_vector::ColumnAlignment::Union,
        per_file_feature_suffix: None,
        per_file_barcode_suffix,
        ..Default::default()
    })?;
    info!(
        "tde: loaded {} feature row(s) x {} cell(s) from {} file(s), {} batch(es)",
        unified.n_features(),
        unified.n_cells(),
        files.len(),
        unified.n_batches()
    );

    let plan = assign_tracks(&unified.feature_names)?;
    log_cells_per_track(&unified, &plan);

    Ok((unified, plan))
}

/// Log, per track, how many cells carry any mass on that track's rows.
fn log_cells_per_track(unified: &ge::UnifiedData, plan: &TrackPlan) {
    let backend = unified.count_backend();
    for (unspliced, name) in [(false, "spliced"), (true, "unspliced")] {
        let rows = plan.rows(unspliced);
        match backend.read_rows_csc(rows.iter().copied()) {
            Ok(csc) => {
                let n_cells = (0..csc.ncols()).filter(|&j| csc.col(j).nnz() > 0).count();
                info!(
                    "tde count/{name}: {} row(s), {n_cells} cell(s) with mass",
                    rows.len()
                );
            }
            Err(e) => log::warn!("tde count/{name}: could not count cells with mass: {e}"),
        }
    }
}

#[cfg(test)]
#[path = "load/tests.rs"]
mod tests;
