//! The encoder's null for held-out cells, built the way training built it.
//!
//! Training collapses cells into pseudobulks `p` across batches and fits
//! each a batch-free mean `μ_adj[g,p]` and each batch a ratio `δ[g,b]`. The
//! encoder reads `y / (x0 · μ_g)`, with `x0` the pseudobulk's own fold
//! `μ_residual[g,p] = Σ_{c∈p} y_cg / (n_p · μ_adj[g,p])` under
//! `--adj-method residual`, or the batch's `δ[g,b]` under `batch`.
//!
//! Predict replays that collapse on its own held-out cells and their
//! batches, with the model's recorded settings. The collapse leaves the
//! finest pseudobulk membership on the data set, as at training, so each
//! cell reads its null off its own pseudobulk or batch. Nothing is written.
//!
//! This is the encoder's input only. Scoring the held-out counts needs the
//! per-batch rate multiplier, which `predict::delta::scoring_delta` gives
//! (the trained one, or one cross-fitted from the encoded cells); the two
//! are kept apart.

use crate::predict::delta::ScoringSpec;
use crate::refine_weighting::CollapseArgs;
use crate::topic::common::{collapse_loaded_finest, expand_delta_for_block, ProjectSettings};
use crate::topic::eval::GeneRemap;
use crate::topic::model_metadata::TopicModelMetadata;
use candle_core::{Device, Tensor};
use clap::Parser;
use data_beans::alg::collapse_data::MultilevelParams;
use data_beans::alg::feature_coarsening::FeatureCoarsening;
use data_beans::alg::hvg::HvgCliArgs;
use data_beans::sparse_io_vector::SparseIoVec;
use log::{info, warn};
use senna::embed_common::*;

/// The fit settings predict replays, as the run recorded them.
pub(crate) struct RecordedFit {
    /// The block size the encoder wrote its latent at; `None` when the
    /// record could not be read.
    minibatch_size: Option<usize>,
    block_size: Option<usize>,
    collapse: CollapseArgs,
    /// Absent for the masked families, which select no HVGs.
    hvg: Option<HvgCliArgs>,
}

impl RecordedFit {
    /// The settings recorded in `model`'s run manifest. A manifest that
    /// cannot be read, or predates the record, gives the training defaults
    /// (and the caller's block size), with a warning.
    pub(crate) fn read(model: &str) -> Self {
        /// Every family's own default when `--minibatch-size` was left unset.
        const TRAIN_MINIBATCH: usize = 100;
        #[derive(serde::Deserialize)]
        struct Raw {
            #[serde(default)]
            minibatch_size: Option<usize>,
            #[serde(default)]
            block_size: Option<usize>,
            collapse: CollapseArgs,
            #[serde(default)]
            hvg: Option<HvgCliArgs>,
        }
        let raw =
            senna::run_manifest::load_for(model).and_then(|(m, _)| m.train_args_as::<Raw>(model));
        match raw {
            Ok(r) => Self {
                minibatch_size: Some(r.minibatch_size.unwrap_or(TRAIN_MINIBATCH)),
                block_size: r.block_size,
                collapse: r.collapse,
                hvg: r.hvg,
            },
            Err(e) => {
                warn!(
                    "{model}: cannot read the recorded fit settings ({e}); using the \
                     defaults for the null and the requested block size"
                );
                #[derive(Parser)]
                struct Defaults {
                    #[clap(flatten)]
                    collapse: CollapseArgs,
                    #[clap(flatten)]
                    hvg: HvgCliArgs,
                }
                let d = Defaults::parse_from(["predict"]);
                Self {
                    minibatch_size: None,
                    block_size: None,
                    collapse: d.collapse,
                    hvg: Some(d.hvg),
                }
            }
        }
    }

    /// The block size to score at: the one the run was TRAINED with.
    ///
    /// Every encoder's input depends on the block a cell is scored in: the
    /// value transform standardises each gene over the block's cells, and
    /// the vae's `GaussianEncoder` adds a batch norm. Training wrote its
    /// latent in blocks of its `--minibatch-size`; scoring in blocks of
    /// another size puts the encoder on statistics it never saw, and the
    /// latent moves.
    ///
    /// It is the DECLARED value: training may shrink the block further to fit
    /// device memory, and that effective size is not recorded, so this narrows
    /// the gap rather than closing it. Without a record it is `requested`.
    pub(crate) fn minibatch(&self, requested: usize) -> usize {
        let Some(trained_at) = self.minibatch_size else {
            return requested;
        };
        if trained_at != requested {
            info!(
                "Scoring at the training block size ({trained_at}, not {requested}): the \
                 encoder's input depends on the block a cell is scored in"
            );
        }
        trained_at
    }
}

/// What a scorer takes from the held-out cells' batches.
#[derive(Clone, Copy)]
pub(crate) enum HeldOut<'a> {
    /// Neither the encoder's null nor the scoring δ. `probe` wants this:
    /// fitted from the query, they would absorb the novelty it looks for.
    Ignore,
    /// The scoring δ only (`--bulk`: whole samples are not cells to
    /// collapse into pseudobulks).
    Delta(ScoringSpec),
    /// The encoder's null, built with the fit's recorded settings, and the
    /// scoring δ.
    NullAndDelta(&'a RecordedFit, ScoringSpec),
}

impl HeldOut<'_> {
    /// How to get the scoring δ, or `None` for no δ.
    pub(crate) fn scoring(self) -> Option<ScoringSpec> {
        match self {
            HeldOut::Ignore => None,
            HeldOut::Delta(spec) | HeldOut::NullAndDelta(_, spec) => Some(spec),
        }
    }
}

/// The encoder's null for the held-out cells: `null` is `[columns, D_enc]`,
/// and each cell reads the column of its pseudobulk (`Residual`) or its
/// batch (`Batch`).
pub(crate) struct EncoderNull {
    pub null: Tensor,
    pub adj_method: AdjMethod,
}

/// The encoder's `x0` for the cells `lb..ub`, or `None` without a null.
pub(crate) fn block_x0(
    null: Option<&EncoderNull>,
    data_vec: &SparseIoVec,
    lb: usize,
    ub: usize,
    dev: &Device,
) -> anyhow::Result<Option<Tensor>> {
    null.map(|n| expand_delta_for_block(data_vec, &n.null, &n.adj_method, lb, ub, dev))
        .transpose()
}

/// Whether the model was trained with a null at all: its collapse found
/// batches and wrote their δ. A fit with one batch (or `--ignore-batch`)
/// had none, and gets none here, whatever batches the query brings.
#[must_use]
pub(crate) fn model_has_null(model: &str) -> bool {
    std::path::Path::new(&format!("{model}.delta.parquet")).exists()
}

/// The model a null is built for.
#[derive(Clone, Copy)]
pub(crate) struct NullModel<'a> {
    pub prefix: &'a str,
    pub metadata: &'a TopicModelMetadata,
    /// The model's gene axis, which its trained δ is read onto.
    pub training_genes: &'a [Box<str>],
}

/// The encoder's null for the held-out cells in `data_vec`, on the
/// encoder's axis (`gene_remap`, then `coarsening`) and on `dev`, or `None`
/// when `held_out` asks for none or the model was trained without one.
/// Collapses `data_vec` with the fit's recorded settings, which leaves its
/// finest pseudobulk membership on it.
pub(crate) fn encoder_null(
    held_out: HeldOut<'_>,
    model: NullModel<'_>,
    data_vec: &mut SparseIoVec,
    gene_remap: Option<&GeneRemap>,
    coarsening: Option<&FeatureCoarsening>,
    dev: &Device,
) -> anyhow::Result<Option<EncoderNull>> {
    let NullModel {
        prefix: model,
        metadata,
        training_genes,
    } = model;
    let HeldOut::NullAndDelta(fit, _) = held_out else {
        return Ok(None);
    };
    if !model_has_null(model) {
        info!("{model}: trained without a batch null; none for the held-out cells");
        return Ok(None);
    }
    if let Some(null) = trained_null(model, metadata, training_genes, data_vec, coarsening, dev)? {
        return Ok(Some(null));
    }
    let Some(names) = data_vec
        .batch_names()
        .filter(|_| data_vec.num_batches() > 1)
    else {
        // One held-out batch: no null can be rebuilt from the cells. A batch
        // the model was trained on still has its trained δ, the batch-level
        // part of what a residual fit's encoder trained under.
        if let Some(null) = trained_batch_null(model, training_genes, data_vec, coarsening, dev)? {
            info!("{model}: one held-out batch; the encoder reads its trained δ");
            return Ok(Some(null));
        }
        warn!(
            "{model}: trained with batches, but the held-out cells come in one batch with no \
             trained δ: no null. Pass one batch file per data file (-b) to say which training \
             batch the cells belong to."
        );
        return Ok(None);
    };
    let batch_membership: Vec<Box<str>> = data_vec
        .get_batch_membership(0..data_vec.num_columns())
        .into_iter()
        .map(|b| names[b].clone())
        .collect();

    let c = &fit.collapse;
    let hvg = fit.hvg.as_ref();
    let proj_dim = c.proj_dim.max(metadata.n_topics);
    let params = MultilevelParams {
        knn_pb_samples: c.knn_cells,
        num_levels: c.num_levels,
        sort_dim: c.sort_dim,
        num_opt_iter: c.iter_opt,
        refine: c.pb_refine.to_params(),
        bulk_batches: c.mixture_batch.clone(),
        pb_tree: c.pb_tree_params(),
        ..MultilevelParams::new(proj_dim)
    };
    info!("Collapsing held-out cells for the encoder's null ...");
    let finest = collapse_loaded_finest(
        data_vec,
        &batch_membership,
        &ProjectSettings {
            proj_dim,
            block_size: fit.block_size,
            max_features: hvg.map_or(0, |h| h.n_hvg),
            feature_list_file: hvg.and_then(|h| h.feature_list_file.as_deref()),
            must_train_file: hvg.and_then(|h| h.must_train_features.as_deref()),
        },
        &params,
    )?;
    let (adj_method, null) = if metadata.adj_method.as_ref() == "batch" {
        (AdjMethod::Batch, finest.delta.as_ref())
    } else {
        (AdjMethod::Residual, finest.mu_residual.as_ref())
    };
    let Some(null) = null else {
        warn!("the held-out cells give the collapse one batch: no null");
        return Ok(None);
    };
    let mut null_dp = match gene_remap {
        Some(remap) => to_training_axis(null.posterior_mean(), remap),
        None => null.posterior_mean().clone(),
    };
    if let Some(fc) = coarsening {
        null_dp = fc.aggregate_rows_ds(&null_dp);
    }
    Ok(Some(EncoderNull {
        null: null_dp.to_tensor(dev)?.transpose(0, 1)?.contiguous()?,
        adj_method,
    }))
}

/// The encoder's null from the TRAINED δ, `[held-out batches, D_enc]`, when
/// the encoder trained on a per-batch δ and every held-out batch is one the
/// model was trained on; `None` otherwise, and the caller estimates a null
/// from the held-out cells.
///
/// Nothing is fitted on the held-out cells here, so one batch is enough.
fn trained_null(
    model: &str,
    metadata: &TopicModelMetadata,
    training_genes: &[Box<str>],
    data_vec: &SparseIoVec,
    coarsening: Option<&FeatureCoarsening>,
    dev: &Device,
) -> anyhow::Result<Option<EncoderNull>> {
    if metadata.adj_method.as_ref() != "batch" || !metadata.batch_null_in_training {
        return Ok(None);
    }
    trained_batch_null(model, training_genes, data_vec, coarsening, dev)
}

/// The trained δ of the held-out batches as an encoder null, `[held-out
/// batches, D_enc]`, when every one of them is a training batch.
fn trained_batch_null(
    model: &str,
    training_genes: &[Box<str>],
    data_vec: &SparseIoVec,
    coarsening: Option<&FeatureCoarsening>,
    dev: &Device,
) -> anyhow::Result<Option<EncoderNull>> {
    let Some(mut null_db) =
        crate::predict::delta::trained_delta_of_batches(model, training_genes, data_vec)?
    else {
        return Ok(None);
    };
    if let Some(fc) = coarsening {
        null_db = fc.aggregate_rows_ds(&null_db);
    }
    Ok(Some(EncoderNull {
        null: null_db.to_tensor(dev)?.transpose(0, 1)?.contiguous()?,
        adj_method: AdjMethod::Batch,
    }))
}

/// `null`'s rows on the training axis by `remap`; a training gene no query
/// row maps to takes 1 (no correction), one several map to their mean.
fn to_training_axis(null: &Mat, remap: &GeneRemap) -> Mat {
    let mut sum = Mat::zeros(remap.d_train, null.ncols());
    let mut count = vec![0_u32; remap.d_train];
    for (q, t) in remap.new_to_train.iter().enumerate() {
        if let Some(&t) = t.as_ref() {
            let mut row = sum.row_mut(t);
            row += &null.row(q);
            count[t] += 1;
        }
    }
    for (t, &n) in count.iter().enumerate() {
        if n == 0 {
            sum.row_mut(t).fill(1.0);
        } else {
            sum.row_mut(t).unscale_mut(n as f32);
        }
    }
    sum
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::topic::eval::{build_gene_remap_with, QueryNameOpts};

    #[test]
    fn rows_move_to_the_training_axis_by_name() {
        let null = Mat::from_row_slice(2, 2, &[2.0, 3.0, 5.0, 7.0]);
        let rows: Vec<Box<str>> = vec!["gene2".into(), "GENE1".into()];
        let train: Vec<Box<str>> = vec!["GENE1".into(), "GENE3".into(), "GENE2".into()];
        let remap = build_gene_remap_with(&train, &rows, &QueryNameOpts::default());
        let out = to_training_axis(&null, &remap);
        assert_eq!(
            out,
            Mat::from_row_slice(3, 2, &[5.0, 7.0, 1.0, 1.0, 2.0, 3.0])
        );
    }
}
