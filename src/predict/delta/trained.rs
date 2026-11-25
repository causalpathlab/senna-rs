//! The δ a model was trained with: reading it per batch and onto an axis,
//! and the per-batch gene fold a projection divides counts by.

use super::TRAINED_DELTA_FLOOR;
use data_beans::sparse_io_vector::SparseIoVec;
use log::info;
use senna::embed_common::*;

/// The trained δ of `model` on the axis `training_genes`: its batch names
/// and `[D_train, B_train]`, with 1 for a gene the file does not carry.
/// `None` when the model was trained without batches.
pub(super) fn load_trained_delta(
    model: &str,
    training_genes: &[Box<str>],
) -> anyhow::Result<Option<(Vec<Box<str>>, Mat)>> {
    if !crate::predict::null::model_has_null(model) {
        return load_batch_gene_fold(model, training_genes);
    }
    let path = format!("{model}.delta.parquet");
    let (strs, nums) =
        legume_numeric::matrix::parquet::read_table_columns(&path, &["gene", "batch"], &["mean"])?;
    let [genes, batches] = <[Vec<Box<str>>; 2]>::try_from(strs)
        .map_err(|_| anyhow::anyhow!("{path}: expected gene and batch columns"))?;
    let mean = nums.into_iter().next().unwrap_or_default();
    anyhow::ensure!(
        genes.len() == batches.len() && genes.len() == mean.len(),
        "{path}: columns of different lengths"
    );
    let gene_at: rustc_hash::FxHashMap<&str, usize> = training_genes
        .iter()
        .enumerate()
        .map(|(i, g)| (&**g, i))
        .collect();
    let mut names: Vec<Box<str>> = Vec::new();
    let mut batch_at: rustc_hash::FxHashMap<Box<str>, usize> = rustc_hash::FxHashMap::default();
    for b in &batches {
        if !batch_at.contains_key(b) {
            batch_at.insert(b.clone(), names.len());
            names.push(b.clone());
        }
    }
    let mut delta = Mat::from_element(training_genes.len(), names.len(), 1.0);
    let mut seen = vec![false; training_genes.len()];
    for ((g, b), &v) in genes.iter().zip(&batches).zip(&mean) {
        if let Some(&d) = gene_at.get(&**g) {
            let v = v as f32;
            if v.is_finite() && v > 0.0 {
                delta[(d, batch_at[b])] = v.max(TRAINED_DELTA_FLOOR);
            }
            seen[d] = true;
        }
    }
    let missing = seen.iter().filter(|s| !**s).count();
    anyhow::ensure!(
        missing < seen.len(),
        "{path}: none of its genes is named like a gene of the model's dictionary, so the \
         trained δ cannot be read; use --scoring-delta crossfit or none"
    );
    if 2 * missing > seen.len() {
        log::warn!(
            "{path}: no trained δ for {missing} of the model's {} genes; they take δ = 1",
            seen.len()
        );
    } else if missing > 0 {
        info!("{path}: no trained δ for {missing} of the model's genes; they take δ = 1");
    }
    Ok(Some((names, delta)))
}

/// The trained δ of a `senna bge` run, from `{model}.batch_gene_fold.parquet`
/// (`log δ`, features × batches), on the axis `training_genes`; `None` when
/// the run wrote none.
fn load_batch_gene_fold(
    model: &str,
    training_genes: &[Box<str>],
) -> anyhow::Result<Option<(Vec<Box<str>>, Mat)>> {
    use legume_numeric::matrix::traits::IoOps;
    let path = format!("{model}.batch_gene_fold.parquet");
    if !std::path::Path::new(&path).is_file() {
        return Ok(None);
    }
    let table = Mat::from_parquet_with_row_names(&path, Some(0))?;
    let row_of: rustc_hash::FxHashMap<&str, usize> = table
        .rows
        .iter()
        .enumerate()
        .map(|(i, g)| (&**g, i))
        .collect();
    let mut delta = Mat::from_element(training_genes.len(), table.cols.len(), 1.0);
    let mut missing = 0usize;
    for (d, g) in training_genes.iter().enumerate() {
        match row_of.get(&**g) {
            Some(&i) => {
                for b in 0..table.cols.len() {
                    let v = table.mat[(i, b)].exp();
                    if v.is_finite() {
                        delta[(d, b)] = v.max(TRAINED_DELTA_FLOOR);
                    }
                }
            }
            None => missing += 1,
        }
    }
    anyhow::ensure!(
        missing < training_genes.len(),
        "{path}: none of its features is named like a feature of the model, so the trained δ \
         cannot be read; use --scoring-delta crossfit or none"
    );
    if 2 * missing > training_genes.len() {
        log::warn!(
            "{path}: no trained δ for {missing} of the model's {} features; they take δ = 1",
            training_genes.len()
        );
    } else if missing > 0 {
        info!("{path}: no trained δ for {missing} of the model's features; they take δ = 1");
    }
    Ok(Some((table.cols, delta)))
}

/// Whether `model` (a run's prefix) saved a trained per-batch δ of either
/// form: a collapse's `delta.parquet` or a bge run's gene fold.
#[must_use]
pub(crate) fn has_trained_delta(model: &str) -> bool {
    crate::predict::null::model_has_null(model)
        || std::path::Path::new(&format!("{model}.batch_gene_fold.parquet")).is_file()
}

/// The trained δ of each held-out batch, `[D_train, held-out batches]` in the
/// query's own batch order, with a column of ones for a batch the model was
/// not trained on, and which batches those are (`known`). `None` when the
/// model has no trained δ or the query no batches.
pub(crate) struct TrainedByBatch {
    pub delta: Mat,
    pub known: Vec<bool>,
    names: Vec<Box<str>>,
}

impl TrainedByBatch {
    pub(crate) fn unknown(&self) -> Vec<&str> {
        let named = self.names.iter().zip(&self.known);
        named.filter(|(_, k)| !**k).map(|(n, _)| &**n).collect()
    }
}

pub(crate) fn trained_delta_by_batch(
    model: &str,
    training_genes: &[Box<str>],
    data_vec: &SparseIoVec,
) -> anyhow::Result<Option<TrainedByBatch>> {
    let names = data_vec.batch_names().unwrap_or_default();
    trained_delta_for_names(model, training_genes, names)
}

/// [`trained_delta_by_batch`] for batches given by name.
fn trained_delta_for_names(
    model: &str,
    training_genes: &[Box<str>],
    names: Vec<Box<str>>,
) -> anyhow::Result<Option<TrainedByBatch>> {
    if names.is_empty() {
        return Ok(None);
    }
    let Some((trained_names, trained_db)) = load_trained_delta(model, training_genes)? else {
        return Ok(None);
    };
    let mut delta = Mat::from_element(trained_db.nrows(), names.len(), 1.0);
    let mut known = vec![false; names.len()];
    for (b, name) in names.iter().enumerate() {
        if let Some(c) = trained_names.iter().position(|t| t == name) {
            delta.set_column(b, &trained_db.column(c));
            known[b] = true;
        }
    }
    Ok(Some(TrainedByBatch {
        delta,
        known,
        names,
    }))
}

/// [`trained_delta_by_batch`] when EVERY held-out batch is a training batch,
/// `None` otherwise (the unknown ones are logged). Nothing is fitted on the
/// held-out cells, so one batch is enough.
///
/// What the encoder's null reads: a null has to be on one footing for every
/// cell of a block, so a query with an unknown batch gets the null rebuilt
/// from its cells instead.
pub(crate) fn trained_delta_of_batches(
    model: &str,
    training_genes: &[Box<str>],
    data_vec: &SparseIoVec,
) -> anyhow::Result<Option<Mat>> {
    let Some(by_batch) = trained_delta_by_batch(model, training_genes, data_vec)? else {
        return Ok(None);
    };
    let unknown = by_batch.unknown();
    if !unknown.is_empty() {
        info!(
            "{model}: no trained δ for held-out batches {}",
            unknown.join(", ")
        );
        return Ok(None);
    }
    info!(
        "{model}: the encoder reads the trained δ of {} held-out batches",
        by_batch.known.len()
    );
    Ok(Some(by_batch.delta))
}

/// The fold a projection divides each held-out cell's counts by, per batch:
/// the trained δ for a training batch, ones for any other (those cells are
/// placed unadjusted, and scored under a δ fitted for their batch). `None`
/// when no held-out batch is a training batch.
pub(crate) fn projection_fold(
    model: &str,
    training_genes: &[Box<str>],
    data_vec: &SparseIoVec,
) -> anyhow::Result<Option<Mat>> {
    let Some(by_batch) = trained_delta_by_batch(model, training_genes, data_vec)? else {
        return Ok(None);
    };
    let n_known = by_batch.known.iter().filter(|k| **k).count();
    if n_known == 0 {
        return Ok(None);
    }
    let unknown = by_batch.unknown();
    info!(
        "{model}: counts of {n_known} held-out batches are divided by their trained δ before \
         projection{}",
        if unknown.is_empty() {
            String::new()
        } else {
            format!(
                "; not trained on, so left as they are: {}",
                unknown.join(", ")
            )
        }
    );
    Ok(Some(by_batch.delta))
}

/// The trained fold of cells known only by their batch labels (one per
/// cell): `[genes, distinct labels]` and each cell's column, with ones for a
/// label the model was not trained on. `None` when no label is a training
/// batch. What a reference's own cells are adjusted by, where no loaded data
/// set carries their batches.
pub(crate) fn fold_by_labels(
    model: &str,
    genes: &[Box<str>],
    labels: &[Box<str>],
) -> anyhow::Result<Option<(Mat, Vec<u32>)>> {
    if labels.is_empty() {
        return Ok(None);
    }
    let mut names: Vec<Box<str>> = Vec::new();
    let mut at: rustc_hash::FxHashMap<&str, u32> = rustc_hash::FxHashMap::default();
    let cols: Vec<u32> = labels
        .iter()
        .map(|l| {
            *at.entry(&**l).or_insert_with(|| {
                names.push(l.clone());
                names.len() as u32 - 1
            })
        })
        .collect();
    let Some(by_batch) = trained_delta_for_names(model, genes, names)? else {
        return Ok(None);
    };
    let n_known = by_batch.known.iter().filter(|k| **k).count();
    if n_known == 0 {
        return Ok(None);
    }
    info!(
        "{model}: the cells of {n_known} of {} batches are divided by their trained δ",
        by_batch.known.len()
    );
    Ok(Some((by_batch.delta, cols)))
}
