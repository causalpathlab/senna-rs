//! Scoring the masked topic families (`masked-topic`).

use crate::topic::eval::{GeneRemap, QueryNameOpts};
use crate::topic::model_metadata::{load_dictionary, TopicModelMetadata};
use senna::embed_common::*;

use super::{
    build_remap, delta, feature_mean_on, finalize::topic_agreement, finalize::write_outputs,
    finalize::FinalizePredict, held_out, null, read_query, scoring_remap, PredictArgs,
};
use crate::logging::new_progress_bar;
use candle_core::Device;
use data_beans::sparse_io_vector::SparseIoVec;
use indicatif::ParallelProgressIterator;
use legume_numeric::candle::encoder::{IndexedEmbeddingEncoder, IndexedEmbeddingEncoderArgs};
use log::info;
use rayon::prelude::*;

/// Encoder-only prediction for the masked-topic
/// ([`MODEL_TYPE_INDEXED_MASKED`]). Rebuilds the indexed symbol-embedding
/// encoder, runs the deterministic masked-encoder forward (all genes visible)
/// on the held-out cells, and writes the latent plus the per-cell full-cell
/// predictive log-likelihood (`{out}.predictive.parquet`, via
/// [`predictive_llik_masked`]) — the reconstruction-residual fit score. Encoder
/// forward only (no decoder refinement), under the null training gave the
/// encoder (see `null`).
pub(super) fn predict_masked(
    args: &PredictArgs,
    metadata: &TopicModelMetadata,
    head: legume_numeric::candle::vae::masked_topic::LatentHead,
) -> anyhow::Result<()> {
    use crate::topic::model_metadata::masked_head_label;

    args.refuse_latent_refinement(&format!(
        "a `senna {}` model, whose inference is encoder-only",
        masked_head_label(head)
    ))?;
    let fit = null::RecordedFit::read(&args.model);
    let scored = score_masked_backend(MaskedScoreArgs {
        model: &args.model,
        data_files: &args.data_files,
        batch_files: args.batch_files.as_deref(),
        preload: args.preload_data,
        minibatch_size: fit.minibatch(args.minibatch_size),
        query_name_opts: &args.query_name_opts()?,
        metadata,
        head,
        need_llik: true,
        block_size: args.block_size,
        held_out: held_out(args, &fit),
        dev: &args.resolve_device()?,
    })?;
    let (masked_training_genes, masked_beta_dk) = load_dictionary(&args.model)?;
    let agreement = topic_agreement(&FinalizePredict {
        args,
        training_genes: &masked_training_genes,
        data_vec: &scored.data_vec,
        z_nk: &scored.z_nk,
        llik: &scored.llik,
        total: &scored.total,
        beta_dk: &masked_beta_dk,
        delta_db: scored.delta_db.as_ref(),
        gene_remap: scored.gene_remap.as_ref(),
    })?;
    write_outputs(
        args,
        &scored.data_vec,
        &scored.z_nk,
        &scored.llik,
        &scored.total,
        agreement.as_ref(),
    )?;
    info!(
        "predict complete ({}, encoder-only latent + full-cell predictive llik)",
        masked_head_label(head)
    );
    Ok(())
}

/// Encoder + full-cell predictive scores for a masked / indexed model on one
/// backend. Shared by `predict` and `probe` (the latter reuses only the scores,
/// not the file output).
pub(crate) struct MaskedScored {
    pub data_vec: SparseIoVec,
    pub z_nk: Mat,
    /// Empty when the caller passed `need_llik: false` (see `MaskedScoreArgs`).
    pub llik: Vec<f32>,
    /// Empty when the caller passed `need_llik: false`.
    pub total: Vec<f32>,
    /// Query→training gene mapping used for the scores; `probe` reuses it for
    /// the influence/gradient pass.
    pub gene_remap: Option<GeneRemap>,
    /// The δ the score was taken under; `None` when it took none.
    pub delta_db: Option<delta::ScoringDelta>,
}

pub(crate) struct MaskedScoreArgs<'a> {
    pub model: &'a str,
    pub data_files: &'a [Box<str>],
    pub batch_files: Option<&'a [Box<str>]>,
    pub preload: bool,
    pub minibatch_size: usize,
    pub query_name_opts: &'a QueryNameOpts,
    pub metadata: &'a TopicModelMetadata,
    pub head: legume_numeric::candle::vae::masked_topic::LatentHead,
    /// Compute the per-cell predictive log-likelihood. `false` skips it entirely,
    /// leaving `llik`/`total` empty — worth it for callers that want only the latent,
    /// since the score costs a second full pass over every column plus a dense
    /// `[D, minibatch]` reconstruction per block.
    pub need_llik: bool,
    /// Cells per block of the scoring δ pass (`None` = auto).
    pub block_size: Option<usize>,
    /// The encoder's null and the scoring δ from the held-out batches; the
    /// δ also scores the predictive likelihood.
    ///
    /// **Explicit, because both are fitted from the query itself.** That is
    /// what `predict` wants (it is the held-out batch effect) and exactly what
    /// `probe` must not have: they absorb compositional novelty, the signal
    /// probe exists to detect, and a query's batch count depends on how many
    /// files the user happened to pass, so leaving it implicit made probe's
    /// calibration and query arms score under different models.
    pub held_out: null::HeldOut<'a>,
    /// Where the encoder and decoder run.
    pub dev: &'a Device,
}

/// Rebuild the indexed encoder from a trained masked model, run encoder-only
/// latent inference on `data_files`, and compute the per-cell full-cell
/// predictive log-likelihood (the probe fit score). Writes nothing.
pub(crate) fn score_masked_backend(a: MaskedScoreArgs<'_>) -> anyhow::Result<MaskedScored> {
    use crate::topic::eval_indexed::{evaluate_latent_masked, EvaluateLatentMaskedConfig};

    let embedding_dim = a
        .metadata
        .embedding_dim
        .ok_or_else(|| anyhow::anyhow!("masked-topic metadata missing embedding_dim"))?;

    // Refused, not warned: with the head unwired nothing registers its weights,
    // so the model would be scored at a rate it was never trained at.
    crate::topic::model_metadata::ensure_query_head_not_wired(a.metadata.query_rank)?;
    let (training_genes, beta_dk) = load_dictionary(a.model)?;
    let feature_mean = feature_mean_on(a.model, &training_genes)?;

    let mut data_vec = read_query(
        a.data_files,
        a.batch_files,
        a.preload,
        a.query_name_opts,
        &training_genes,
    )?;
    let new_genes = data_vec.row_names()?;
    let gene_remap = build_remap(&training_genes, &new_genes, a.query_name_opts)?;

    let dev = a.dev;
    let mut parameters = candle_nn::VarMap::new();
    let vb = candle_nn::VarBuilder::from_varmap(&parameters, candle_core::DType::F32, dev);
    let encoder = IndexedEmbeddingEncoder::new(
        IndexedEmbeddingEncoderArgs {
            n_features: a.metadata.n_features_full,
            n_topics: a.metadata.n_topics,
            embedding_dim,
            layers: &a.metadata.encoder_hidden,
            attn_pool: true,
            // Must match the checkpoint: M widens the first FC layer, and `VarMap::load`
            // errors on a shape mismatch.
            n_gene_modules: a.metadata.gene_modules(),
            lora_rank: 0,
        },
        &parameters,
        vb.pp("enc"),
    )?;
    let safetensors_path = format!("{}.safetensors", a.model);
    info!("Loading weights from {safetensors_path}");
    parameters.load(&safetensors_path)?;

    // The encoder's null, built the way training built it (the masked encoder
    // reads the full gene axis, so no coarsening).
    let null = null::encoder_null(
        a.held_out,
        null::NullModel {
            prefix: a.model,
            metadata: a.metadata,
            training_genes: &training_genes,
        },
        &mut data_vec,
        gene_remap.as_ref(),
        None,
        dev,
    )?;
    let adj_method = null
        .as_ref()
        .map_or(AdjMethod::Residual, |n| n.adj_method.clone());
    let eval_config = EvaluateLatentMaskedConfig {
        dev,
        adj_method: &adj_method,
        minibatch_size: a.minibatch_size,
        feature_mean: &feature_mean,
        head: a.head,
    };

    let z_nk = evaluate_latent_masked(
        &data_vec,
        &encoder,
        &eval_config,
        null.as_ref().map(|n| &n.null),
        gene_remap.as_ref().map(|r| r.new_to_train.as_slice()),
    )?;

    {
        let (eff, mx) = latent_sharpness(&crate::topic::model_metadata::latent_to_theta(
            &z_nk, a.head,
        ));
        info!(
            "Latent sharpness on the query: {eff:.2} effective topics of {}, mean max θ {mx:.3}",
            z_nk.ncols()
        );
    }

    // The stored latent stays raw (`z` for the Gaussian head); scoring needs
    // proportions, so convert per the head rather than assuming `exp`.
    // The scoring δ, as `a.scoring` says.
    let theta_nk = crate::topic::model_metadata::latent_to_theta(&z_nk, a.head);
    let delta_db = if let Some(spec) = a.held_out.scoring() {
        let score_remap = scoring_remap(&training_genes, &new_genes, a.query_name_opts)?;
        delta::scoring_delta(&delta::ScoringDeltaArgs {
            spec,
            model: a.model,
            training_genes: &training_genes,
            data_vec: &data_vec,
            theta_nk: &theta_nk,
            gene_remap: score_remap.as_ref(),
            rate: delta::RateModel::Topic {
                exp_beta_dk: &beta_dk.map(f32::exp),
            },
            phi: delta::nb_dispersion(a.model, a.metadata)?.as_deref(),
            block_size: a.block_size,
        })?
    } else {
        None
    };
    let (llik, total) = if a.need_llik {
        predictive_llik_masked(
            &data_vec,
            &theta_nk,
            &beta_dk,
            delta_db.as_ref(),
            gene_remap.as_ref(),
            a.minibatch_size,
        )?
    } else {
        (Vec::new(), Vec::new())
    };

    Ok(MaskedScored {
        data_vec,
        z_nk,
        llik,
        total,
        gene_remap,
        delta_db,
    })
}

/// Per-cell predictive log-likelihood for the masked / indexed topic model.
///
/// The masked ETM decoder's dictionary `β` is a gene-simplex per topic
/// (`exp(β)` columns sum to 1), so `recon = exp(β)·θ` is the reconstructed
/// composition over genes (columns sum to 1). Scoring the observed counts under
/// it gives the multinomial predictive log-likelihood
///   `llik(cell) = Σ_g x_gj · log recon_gj`,   `total(cell) = Σ_g x_gj`,
/// and `write_outputs` derives `llik_per_count = llik / total`. Genes absent
/// from the reference model are skipped (μ = 0), mirroring
/// [`write_residual_backend`].
///
/// Takes `theta_nk` — proportions, rows summing to 1 — **not** the raw latent.
/// The score is head-agnostic only once that conversion has happened: the
/// simplex heads reach it with `exp(log θ)` and the Gaussian masked-VAE head
/// with `softmax(z)`. See `crate::topic::model_metadata::latent_to_theta`.
///
/// With a scoring δ the composition is `δ_{d,g(j)} · Σ_k θ_k·exp(β_dk)`,
/// renormalized over genes per cell — the multinomial analogue of the NB head's
/// `μ = θβ · residual · lib`. Which δ a cell reads is `delta::ScoringDelta`'s
/// business: the trained one, or one fitted on other cells of its batch.
fn predictive_llik_masked(
    data_vec: &SparseIoVec,
    theta_nk: &Mat,
    beta_dk: &Mat,
    delta_db: Option<&delta::ScoringDelta>,
    gene_remap: Option<&GeneRemap>,
    minibatch_size: usize,
) -> anyhow::Result<(Vec<f32>, Vec<f32>)> {
    let exp_beta_dk = beta_dk.map(f32::exp);
    let ntot = data_vec.num_columns();

    let jobs = create_jobs(ntot, 0, Some(minibatch_size));
    let njobs = jobs.len() as u64;
    let blocks: Vec<(usize, Vec<f32>, Vec<f32>)> = jobs
        .par_iter()
        .progress_with(new_progress_bar(njobs))
        .map(|&(lb, ub)| -> anyhow::Result<(usize, Vec<f32>, Vec<f32>)> {
            let csc = data_vec.read_columns_csc(lb..ub)?;
            let n_block = csc.ncols();
            let theta_kn = theta_nk.rows(lb, n_block).transpose();
            let mut recon_dn = &exp_beta_dk * theta_kn; // [D_train, n_block]
            if let Some(delta) = delta_db {
                delta.apply(&mut recon_dn, lb);
            }

            let mut llik = vec![0f32; n_block];
            let mut total = vec![0f32; n_block];
            for jloc in 0..n_block {
                let col = csc.col(jloc);
                for (&row_new, &val) in col.row_indices().iter().zip(col.values().iter()) {
                    let row_train = match gene_remap {
                        Some(rm) => rm.new_to_train[row_new],
                        None => Some(row_new),
                    };
                    let Some(row_train) = row_train else {
                        continue; // gene absent from the reference model
                    };
                    let recon = recon_dn[(row_train, jloc)].max(1e-12);
                    llik[jloc] += val * recon.ln();
                    total[jloc] += val;
                }
            }
            Ok((lb, llik, total))
        })
        .collect::<anyhow::Result<Vec<_>>>()?;

    let mut llik = vec![0f32; ntot];
    let mut total = vec![0f32; ntot];
    for (lb, ll, tot) in blocks {
        for (k, (&l, &t)) in ll.iter().zip(tot.iter()).enumerate() {
            llik[lb + k] = l;
            total[lb + k] = t;
        }
    }
    Ok((llik, total))
}
