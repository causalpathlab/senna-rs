//! Scoring the VAE family.

use crate::topic::eval::QueryNameOpts;
use crate::topic::model_metadata::{load_dictionary, TopicModelMetadata};
use crate::topic::predict_eval::Reconstruction;
use senna::embed_common::*;

use super::{
    aggregate_feature_mean_to_coarse, build_remap, dense::remap_and_coarsen_dense, dense_bytes,
    embedding_scoring_delta, feature_mean_on, finalize::evaluate_agreement,
    finalize::write_outputs, finalize::AgreementInputs, held_out, null, read_query,
    run_predict_blocks, EmbeddingScore, PredictArgs, NB_CHAIN_TENSORS,
};
use candle_core::Device;
use data_beans::sparse_io_vector::SparseIoVec;
use legume_numeric::candle::encoder::{GaussianEncoder, GaussianEncoderArgs};
use legume_numeric::candle::traits::{DecoderModuleT, EncoderModuleT};
use log::info;

/// Held-out latent inference for the Gaussian VAE ([`MODEL_TYPE_VAE`]).
/// Rebuilds the [`GaussianEncoder`] and runs it (encoder-only, eval mode →
/// posterior mean `z`) over the held-out cells, under the null training gave
/// the encoder (see `null`), like `predict_masked`. The latent is continuous
/// factors, so there is no decoder refinement to do.
pub(super) fn predict_vae(args: &PredictArgs, metadata: &TopicModelMetadata) -> anyhow::Result<()> {
    args.refuse_latent_refinement("a vae model, whose inference is encoder-only")?;

    // The decoder is rebuilt so a vae run is scorable like every other family.
    // It used to be skipped (latent only, no `.predictive.parquet`), which left
    // vae the one family a benchmark could not grade on the shared columns.
    let fit = null::RecordedFit::read(&args.model);
    let s = score_vae_backend(VaeScoreArgs {
        model: &args.model,
        data_files: &args.data_files,
        batch_files: args.batch_files.as_deref(),
        preload: args.preload_data,
        minibatch_size: fit.minibatch(args.minibatch_size),
        query_name_opts: &args.query_name_opts()?,
        metadata,
        need_llik: true,
        held_out: held_out(args, &fit),
        dev: &args.resolve_device()?,
    })?;

    // `π = softmax_d(z·W + b)` is `exp(b + ρ·θ)` normalised over genes — the
    // same arithmetic `Reconstruction::Embedding` already grades bge by, so
    // the two families land on one comparable axis. The bare `llik` column
    // stays the backend's own NB density and is NOT comparable across
    // families; `eval_llik_per_count` is.
    let (training_genes, _) = load_dictionary(&args.model)?;
    let scoring = match s.recon.as_ref() {
        Some((rho_dh, b_feat)) => embedding_scoring_delta(EmbeddingScore {
            args,
            model: &args.model,
            training_genes: &training_genes,
            data_vec: &s.data_vec,
            rho_dh,
            b_feat,
            theta_nh: &s.z_nk,
        })?,
        None => None,
    };
    let agreement = match s.recon.as_ref() {
        Some((rho_dh, b_feat)) => evaluate_agreement(AgreementInputs {
            args,
            training_genes: &training_genes,
            data_vec: &s.data_vec,
            recon: Reconstruction::Embedding {
                rho_dh: rho_dh.clone(),
                b_feat,
                theta_nh: &s.z_nk,
                delta_db: scoring.as_ref(),
            },
        })?,
        None => {
            log::warn!(
                "no decoder bias recovered from {}.safetensors; writing the latent and \
                 the backend likelihood without the agreement metrics",
                args.model
            );
            None
        }
    };

    write_outputs(
        args,
        &s.data_vec,
        &s.z_nk,
        &s.llik,
        &s.total,
        agreement.as_ref(),
    )?;
    Ok(())
}

/// Inputs for [`score_vae_backend`].
pub(crate) struct VaeScoreArgs<'a> {
    pub model: &'a str,
    pub data_files: &'a [Box<str>],
    pub batch_files: Option<&'a [Box<str>]>,
    pub preload: bool,
    pub minibatch_size: usize,
    pub query_name_opts: &'a QueryNameOpts,
    pub metadata: &'a TopicModelMetadata,
    /// `predict` wants the latent only and passes `false`, which also skips rebuilding the
    /// decoder entirely. `probe` needs the per-cell score and passes `true`.
    pub need_llik: bool,
    /// The encoder's null and the scoring δ from the held-out batches.
    pub held_out: null::HeldOut<'a>,
    /// Where the encoder and decoder run.
    pub dev: &'a Device,
}

pub(crate) struct VaeScored {
    pub data_vec: SparseIoVec,
    pub z_nk: Mat,
    /// Empty when `need_llik` was `false`.
    pub llik: Vec<f32>,
    /// Empty when `need_llik` was `false`.
    pub total: Vec<f32>,
    /// The gene side of the rate, `(loadings [D,K], feature bias [D])`, on the
    /// dictionary's gene axis. `None` when `need_llik` was `false` (no decoder
    /// was rebuilt) or when the checkpoint carries no decoder bias.
    pub recon: Option<(Mat, Vec<f32>)>,
}

/// Encoder pass for a `senna vae` model, optionally with the NB predictive score.
///
/// **This is new capability, not a refactor.** `predict_vae` never built a decoder, so the
/// vae family had no fit score at all while every other family did — which is exactly what
/// `probe` needs. The decoder is rebuilt here only when asked for.
///
/// ⚠️ `dictionary.parquet` for a vae holds **raw `[D,K]` factor loadings**, not log-β: the
/// model is `π = softmax_d(z·W + b)`, a per-cell softmax over the gene axis, not a mixture of
/// per-topic simplices. So the parquet shortcut the masked path uses is invalid here and the
/// score must go through the rebuilt decoder.
pub(crate) fn score_vae_backend(a: VaeScoreArgs<'_>) -> anyhow::Result<VaeScored> {
    let metadata = a.metadata;

    // The coarsening this model trained at, if any: the encoder reads coarse
    // features, so the query has to be aggregated the same way before it is
    // scored, and `μ_d` re-aggregated to match.
    let coarsening = if metadata.has_coarsening {
        crate::topic::model_metadata::load_coarsening(a.model)?
    } else {
        None
    };
    let (training_genes, loadings_dk) = load_dictionary(a.model)?;
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
    let feature_mean = aggregate_feature_mean_to_coarse(&feature_mean, coarsening.as_ref());
    let encoder = GaussianEncoder::new(
        GaussianEncoderArgs {
            n_features: metadata.n_features_encoder,
            n_latent: metadata.n_topics,
            layers: &metadata.encoder_hidden,
            feature_mean: Some(&feature_mean),
        },
        &parameters,
        vb.clone(),
    )?;

    // Only the finest level is ever scored, and only it needs registering: `VarMap::load`
    // iterates the varmap's OWN entries and looks each up in the file, so extra tensors in the
    // checkpoint are simply never visited.
    let finest = metadata.num_levels.saturating_sub(1);
    let decoder = if a.need_llik {
        Some(legume_numeric::candle::decoder::GaussianNbDecoder::new(
            *metadata
                .level_decoder_dims
                .last()
                .unwrap_or(&metadata.n_features_full),
            metadata.n_topics,
            vb.pp(format!("dec_{finest}")),
        )?)
    } else {
        None
    };

    let safetensors_path = format!("{}.safetensors", a.model);
    info!("Loading weights from {safetensors_path}");
    parameters.load(&safetensors_path)?;
    let decoder = decoder.as_ref();

    // Read the gene side AFTER the load, or it is the initialization rather
    // than the fit. The loadings come from `dictionary.parquet` rather than
    // the checkpoint's weight so they are guaranteed to sit on the same gene
    // axis as `training_genes`; only `b` has no parquet of its own.
    let recon = decoder.and_then(|dec| {
        let bias = dec.feature_bias()?;
        let bias: Vec<f32> = bias.flatten_all().ok()?.to_vec1().ok()?;
        // Under coarsening the offset is per coarse feature, while the
        // loadings were expanded to every feature on the way out. Split each
        // group's offset evenly across its members, which is what makes the
        // per-feature softmax reproduce the group's rate exactly: the shares
        // inside a group sum to one, so the normalizer is unchanged.
        let bias = match coarsening.as_ref() {
            None => bias,
            Some(fc) => {
                let mut per_gene = vec![0f32; loadings_dk.nrows()];
                for (m, fines) in fc.coarse_to_fine.iter().enumerate() {
                    let share = (fines.len().max(1) as f32).ln();
                    for &g in fines {
                        per_gene[g] = bias.get(m).copied().unwrap_or(0.0) - share;
                    }
                }
                per_gene
            }
        };
        if bias.len() != loadings_dk.nrows() {
            log::warn!(
                "vae decoder bias has {} entries but the dictionary has {} genes; \
                 skipping the agreement metrics for this run",
                bias.len(),
                loadings_dk.nrows()
            );
            return None;
        }
        Some((loadings_dk.clone(), bias))
    });

    // The encoder's null, built the way training built it.
    let null = null::encoder_null(
        a.held_out,
        null::NullModel {
            prefix: a.model,
            metadata,
            training_genes: &training_genes,
        },
        &mut data_vec,
        gene_remap.as_ref(),
        coarsening.as_ref(),
        dev,
    )?;
    let ntot = data_vec.num_columns();
    let (z_nk, llik, total) = run_predict_blocks(
        ntot,
        metadata.n_topics,
        a.minibatch_size,
        // The encoder input is the only full-width tensor left; the likelihood
        // works one `crate::topic::predict_common::SCORE_GENE_CHUNK` slice at a time, so the chain's
        // temporaries are counted at slice width, not at D.
        dense_bytes(a.minibatch_size, training_genes.len(), 1)
            + dense_bytes(
                a.minibatch_size,
                crate::topic::predict_common::SCORE_GENE_CHUNK,
                NB_CHAIN_TENSORS,
            ),
        dev,
        |(lb, ub)| {
            // The divisive μ_d correction is baked into the encoder via
            // `feature_mean`; `x0` is the held-out null, when there is one.
            let csc = data_vec.read_columns_csc(lb..ub)?;
            let x_nd =
                remap_and_coarsen_dense(&csc, gene_remap.as_ref(), coarsening.as_ref(), dev)?;
            let x0 = null::block_x0(null.as_ref(), &data_vec, lb, ub, dev)?;
            let (z, _) = encoder.forward_t(&x_nd, x0.as_ref(), false)?;
            let z_mat = Mat::from_tensor(&z.to_device(&Device::Cpu)?)?;
            let Some(dec) = decoder else {
                return Ok((lb, z_mat, Vec::new(), Vec::new()));
            };
            // `GaussianNbDecoder` takes the **raw** `z` — it applies its own softmax over the
            // gene axis — unlike the topic decoders, which are handed log θ.
            //
            // Gene-chunked, not `forward_with_llik`: that returns `(π, llik)`
            // and π is `[N, D]` by construction — training needs it for the
            // gradient, inference throws it away. Paying for a full dense
            // matrix (and the dozen more inside the NB chain) to get one
            // scalar per cell is what made this path an OOM kill at
            // whole-transcriptome width.
            let llik: Vec<f32> = dec
                .llik_gene_chunked(&z, &x_nd, crate::topic::predict_common::SCORE_GENE_CHUNK)?
                .to_device(&Device::Cpu)?
                .to_vec1()?;
            let total: Vec<f32> = x_nd.sum(1)?.to_device(&Device::Cpu)?.to_vec1()?;
            Ok((lb, z_mat, llik, total))
        },
    )?;

    Ok(VaeScored {
        data_vec,
        z_nk,
        llik,
        total,
        recon,
    })
}
