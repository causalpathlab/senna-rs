//! Scoring the dense topic model (`senna topic`): encoder, optional refinement, decoder llik.

use crate::topic::eval::{GeneRemap, QueryNameOpts};
use crate::topic::model_metadata::{load_coarsening, load_dictionary, TopicModelMetadata};
use crate::topic::predict_common::{
    decoder_only_inference_dense, predictive_llik_dense, LatentMode,
};
use senna::embed_common::*;

use super::{
    aggregate_feature_mean_to_coarse, build_remap, delta, dense_bytes, finalize::finalize_predict,
    finalize::FinalizePredict, held_out, null, read_query, resolve_mode, run_predict_blocks,
    scoring_remap, PredictArgs, QueryBatches, NB_CHAIN_TENSORS,
};
use candle_core::{Device, Tensor};
use data_beans::alg::feature_coarsening::FeatureCoarsening;
use data_beans::sparse_io_vector::SparseIoVec;
use legume_numeric::candle::decoder::nb_mixture::DECODER_NAME as NBMIXTURE_NAME;
use legume_numeric::candle::decoder::{
    MultinomTopicDecoder, NbMixtureTopicDecoder, NbTopicDecoder,
};
use legume_numeric::candle::encoder::{LogSoftmaxEncoder, LogSoftmaxEncoderArgs};
use legume_numeric::candle::topic_refinement::{refine_topic_proportions, TopicRefinementConfig};
use legume_numeric::candle::traits::{DecoderModuleT, EncoderModuleT, NewDecoder};
use log::info;

pub(super) fn predict_dense(
    args: &PredictArgs,
    metadata: &TopicModelMetadata,
) -> anyhow::Result<()> {
    let mode = resolve_mode(args);
    let refine_config = TopicRefinementConfig {
        num_steps: if args.decoder_only && args.refine_steps == 0 {
            100
        } else {
            args.refine_steps
        },
        learning_rate: if args.decoder_only && args.refine_lr <= 0.01 {
            0.05
        } else {
            args.refine_lr
        },
        regularization: args.refine_reg,
    };

    let fit = null::RecordedFit::read(&args.model);
    let s = score_dense_backend(DenseScoreArgs {
        model: &args.model,
        data_files: &args.data_files,
        batch_files: args.batch_files.as_deref(),
        preload: args.preload_data,
        minibatch_size: fit.minibatch(args.minibatch_size),
        block_size: args.block_size,
        held_out: held_out(args, &fit),
        query_name_opts: &args.query_name_opts()?,
        metadata,
        mode,
        refine_config: &refine_config,
        dev: &args.resolve_device()?,
    })?;

    finalize_predict(FinalizePredict {
        args,
        training_genes: &s.training_genes,
        data_vec: &s.data_vec,
        z_nk: &s.z_nk,
        llik: &s.llik,
        total: &s.total,
        beta_dk: &s.beta_dk,
        delta_db: s.delta_db.as_ref(),
        gene_remap: s.gene_remap.as_ref(),
    })
}

/// Inputs for [`score_dense_backend`] — the `predict`-independent subset, so `probe` can
/// score a dense model without constructing a whole `PredictArgs`.
pub(crate) struct DenseScoreArgs<'a> {
    pub model: &'a str,
    pub data_files: &'a [Box<str>],
    pub batch_files: Option<&'a [Box<str>]>,
    pub preload: bool,
    pub minibatch_size: usize,
    /// Cells per block of the scoring δ pass (`None` = auto).
    pub block_size: Option<usize>,
    /// The encoder's null and the scoring δ from the held-out batches.
    pub held_out: null::HeldOut<'a>,
    pub query_name_opts: &'a QueryNameOpts,
    pub metadata: &'a TopicModelMetadata,
    pub mode: LatentMode,
    pub refine_config: &'a TopicRefinementConfig,
    /// Where the encoder and decoder run.
    pub dev: &'a Device,
}

/// What the dense scoring pass produces.
///
/// `predict` consumes every field; `probe` needs only `data_vec` / `llik` / `total`. The
/// struct exists because the dense path used to return a bare 4-tuple from inside a driver
/// that also wrote files, so there was no way to ask it for a score and nothing else.
///
/// ⚠️ `llik`'s **scale** is decoder-dependent: `multinom` gives `Σ_d w_d·x_d·log p_d` (NB-Fisher
/// weighted, against an unweighted `total`), while `nb`/`nbmixture` give an NB log-density sum.
/// All are monotone in fit, and `probe` calibrates against its own null, so verdicts are
/// unaffected — but these numbers are not comparable to the masked path's nats/count.
pub(crate) struct DenseScored {
    pub data_vec: SparseIoVec,
    /// The model's feature axis — the space `beta_dk` and the agreement pass index in.
    pub training_genes: Vec<Box<str>>,
    pub z_nk: Mat,
    pub llik: Vec<f32>,
    pub total: Vec<f32>,
    pub gene_remap: Option<GeneRemap>,
    pub beta_dk: Mat,
    /// The δ the cells are scored under, so the caller can regress it out
    /// when writing a residual backend.
    pub delta_db: Option<delta::ScoringDelta>,
}

pub(crate) fn score_dense_backend(a: DenseScoreArgs<'_>) -> anyhow::Result<DenseScored> {
    let metadata = a.metadata;
    let (training_genes, beta_dk) = load_dictionary(a.model)?;
    let coarsening = if metadata.has_coarsening {
        load_coarsening(a.model)?
    } else {
        None
    };

    // Reload `μ_d` at D_full and aggregate via the (optional) coarsening
    // matrix to the encoder's D_coarse. Saved by `senna topic` at
    // training time; absent for older models, where the encoder falls
    // back to live per-feature batch centering inside `anscombe_residual`.
    let feature_mean_enc: Option<Vec<f32>> =
        match crate::topic::model_metadata::load_feature_mean(a.model) {
            Ok((_, full)) => Some(aggregate_feature_mean_to_coarse(&full, coarsening.as_ref())),
            Err(_) => None,
        };

    let mut data_vec = read_query(
        a.data_files,
        QueryBatches::Register {
            files: a.batch_files,
            model: a.model,
        },
        a.preload,
        a.query_name_opts,
        &training_genes,
    )?;
    let new_genes = data_vec.row_names()?;
    let gene_remap = build_remap(&training_genes, &new_genes, a.query_name_opts)?;

    // The encoder's null, built the way training built it; the scoring δ
    // is got afterwards, as `a.scoring` says.
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
        a.dev,
    )?;

    let dev = a.dev;
    let mut parameters = candle_nn::VarMap::new();
    let vb = candle_nn::VarBuilder::from_varmap(&parameters, candle_core::DType::F32, dev);

    let encoder = LogSoftmaxEncoder::new(
        LogSoftmaxEncoderArgs {
            n_features: metadata.n_features_encoder,
            n_topics: metadata.n_topics,
            layers: &metadata.encoder_hidden,
            feature_mean: feature_mean_enc.as_deref(),
        },
        &parameters,
        vb.clone(),
    )?;

    info!("Latent inference mode: {:?}", a.mode);

    let decoder_name = metadata
        .decoder_types
        .first()
        .map_or("multinom", std::convert::AsRef::as_ref);
    info!("Predicting with decoder: {decoder_name}");

    let inputs = DensePredictInputs {
        metadata,
        parameters: &mut parameters,
        vb: &vb,
        model_prefix: a.model,
        encoder: &encoder,
        data_vec: &data_vec,
        null: null.as_ref(),
        gene_remap: gene_remap.as_ref(),
        coarsening: coarsening.as_ref(),
        dev,
        minibatch_size: a.minibatch_size,
        mode: a.mode,
        refine_config: a.refine_config,
    };

    let (z_nk, llik, total) = match decoder_name {
        "multinom" => predict_dense_with_decoder::<MultinomTopicDecoder>(inputs)?,
        "nb" => predict_dense_with_decoder::<NbTopicDecoder>(inputs)?,
        name if name == NBMIXTURE_NAME => {
            predict_dense_with_decoder::<NbMixtureTopicDecoder>(inputs)?
        }
        other => anyhow::bail!("unsupported decoder type in metadata: {other}"),
    };

    // The scoring δ: the caller reconstructs with it and regresses it out of
    // the residual backend.
    let delta_final = if let Some(spec) = a.held_out.scoring() {
        let score_remap = scoring_remap(&training_genes, &new_genes, a.query_name_opts)?;
        delta::scoring_delta(&delta::ScoringDeltaArgs {
            spec,
            model: a.model,
            training_genes: &training_genes,
            data_vec: &data_vec,
            theta_nk: &z_nk.map(f32::exp),
            gene_remap: score_remap.as_ref(),
            rate: delta::RateModel::Topic {
                exp_beta_dk: &beta_dk.map(f32::exp),
            },
            phi: delta::nb_dispersion(a.model, metadata)?.as_deref(),
            block_size: a.block_size,
        })?
    } else {
        None
    };

    Ok(DenseScored {
        training_genes,
        data_vec,
        z_nk,
        llik,
        total,
        gene_remap,
        beta_dk,
        delta_db: delta_final,
    })
}

struct DensePredictInputs<'a> {
    metadata: &'a TopicModelMetadata,
    parameters: &'a mut candle_nn::VarMap,
    vb: &'a candle_nn::VarBuilder<'a>,
    model_prefix: &'a str,
    encoder: &'a LogSoftmaxEncoder,
    data_vec: &'a SparseIoVec,
    null: Option<&'a null::EncoderNull>,
    gene_remap: Option<&'a GeneRemap>,
    coarsening: Option<&'a FeatureCoarsening>,
    dev: &'a Device,
    minibatch_size: usize,
    mode: LatentMode,
    refine_config: &'a TopicRefinementConfig,
}

/// `(log θ [N×K], per-cell llik, per-cell total)`.
type DensePredictOut = (Mat, Vec<f32>, Vec<f32>);

fn predict_dense_with_decoder<Dec>(
    inputs: DensePredictInputs<'_>,
) -> anyhow::Result<DensePredictOut>
where
    Dec: DecoderModuleT + NewDecoder + Send + Sync,
{
    let DensePredictInputs {
        metadata,
        parameters,
        vb,
        model_prefix,
        encoder,
        data_vec,
        null,
        gene_remap,
        coarsening,
        dev,
        minibatch_size,
        mode,
        refine_config,
    } = inputs;

    // Register decoders at every level so safetensors keys match training.
    // Predict only uses the finest-level decoder.
    let mut decoders: Vec<Dec> = Vec::with_capacity(metadata.level_decoder_dims.len());
    for (i, &d_l) in metadata.level_decoder_dims.iter().enumerate() {
        decoders.push(Dec::new(d_l, metadata.n_topics, vb.pp(format!("dec_{i}")))?);
    }

    let safetensors_path = format!("{model_prefix}.safetensors");
    info!("Loading weights from {safetensors_path}");
    parameters.load(&safetensors_path)?;

    // Attach finest-level NB-Fisher weights to the finest decoder so
    // predictive llik uses the same loss as training. Older models
    // without saved coarse weights fall back to the unweighted form.
    if let Some((_, coarse_w)) =
        data_beans::alg::gene_weighting::load_fisher_weights_coarse(model_prefix)?
    {
        if let Some(finest) = decoders.last_mut() {
            finest.attach_feature_weights(&coarse_w, dev)?;
        }
    }
    let decoder = decoders.last().expect("at least one decoder level");

    let ntot = data_vec.num_columns();
    let kk = metadata.n_topics;
    // The finest DECODER's width, not `n_features_full`: the dozen-odd dense
    // temporaries live in the likelihood chain, which runs on the decoder's
    // axis. A coarsened dense model densifies briefly at full width but scores
    // at the coarsened one, so sizing the cap off the full axis would throttle
    // the topic paths that have never had a memory problem.
    let d_dense = metadata
        .level_decoder_dims
        .last()
        .copied()
        .unwrap_or(metadata.n_features_full);
    // Ask the decoder rather than assume: one that slices holds the block's
    // input plus a chunk's temporaries, one that does not still materialises
    // the reconstruction and the whole chain at full decoder width.
    let bytes_per_block = if decoder.llik_is_gene_chunked() {
        dense_bytes(minibatch_size, d_dense, 1)
            + dense_bytes(
                minibatch_size,
                crate::topic::predict_common::SCORE_GENE_CHUNK,
                NB_CHAIN_TENSORS,
            )
    } else {
        dense_bytes(minibatch_size, d_dense, NB_CHAIN_TENSORS)
    };

    let (z_nk, llik, total) = run_predict_blocks(
        ntot,
        kk,
        minibatch_size,
        bytes_per_block,
        dev,
        |(lb, ub)| {
            predict_block_dense::<Dec>(PredictBlockDenseArgs {
                lb,
                ub,
                data_vec,
                encoder,
                decoder,
                null,
                gene_remap,
                coarsening,
                dev,
                mode,
                refine_config,
            })
        },
    )?;
    Ok((z_nk, llik, total))
}

struct PredictBlockDenseArgs<'a, Dec> {
    lb: usize,
    ub: usize,
    data_vec: &'a SparseIoVec,
    encoder: &'a LogSoftmaxEncoder,
    decoder: &'a Dec,
    null: Option<&'a null::EncoderNull>,
    gene_remap: Option<&'a GeneRemap>,
    coarsening: Option<&'a FeatureCoarsening>,
    dev: &'a Device,
    mode: LatentMode,
    refine_config: &'a TopicRefinementConfig,
}

fn predict_block_dense<Dec>(
    a: PredictBlockDenseArgs<'_, Dec>,
) -> anyhow::Result<(usize, Mat, Vec<f32>, Vec<f32>)>
where
    Dec: DecoderModuleT,
{
    let PredictBlockDenseArgs {
        lb,
        ub,
        data_vec,
        encoder,
        decoder,
        null,
        gene_remap,
        coarsening,
        dev,
        mode,
        refine_config,
    } = a;

    let x0_nd = null::block_x0(null, data_vec, lb, ub, dev)?;

    let csc = data_vec.read_columns_csc(lb..ub)?;
    let x_at_dec = remap_and_coarsen_dense(&csc, gene_remap, coarsening, dev)?;

    let log_z_nk = match mode {
        LatentMode::Encoder => {
            let (log_z, _) = encoder.forward_t(&x_at_dec, x0_nd.as_ref(), false)?;
            log_z
        }
        LatentMode::EncoderRefine => {
            let (log_z, _) = encoder.forward_t(&x_at_dec, x0_nd.as_ref(), false)?;
            refine_topic_proportions(&log_z, &x_at_dec, decoder, refine_config)?
        }
        LatentMode::DecoderOnly => decoder_only_inference_dense(
            decoder,
            &x_at_dec,
            decoder.dim_latent(),
            refine_config.learning_rate,
            refine_config.num_steps,
            dev,
        )?,
    };

    let llik_t = predictive_llik_dense(decoder, &log_z_nk, &x_at_dec)?;
    let llik: Vec<f32> = llik_t.to_device(&Device::Cpu)?.to_vec1()?;

    let total: Vec<f32> = {
        let summed = x_at_dec.sum(1)?.to_device(&Device::Cpu)?;
        summed.to_vec1()?
    };

    let z_cpu = log_z_nk.to_device(&Device::Cpu)?;
    let z_mat = Mat::from_tensor(&z_cpu)?;
    Ok((lb, z_mat, llik, total))
}

/// Scatter CSC rows from new-data order to training gene order, then optionally
/// coarsen, returning a `[N, D_dec]` tensor on the requested device.
pub(super) fn remap_and_coarsen_dense(
    csc: &nalgebra_sparse::CscMatrix<f32>,
    gene_remap: Option<&GeneRemap>,
    coarsening: Option<&FeatureCoarsening>,
    dev: &Device,
) -> anyhow::Result<Tensor> {
    let nd = if let Some(remap) = gene_remap {
        let out = crate::topic::eval::remap_csc_to_dense(csc, remap);
        if let Some(fc) = coarsening {
            fc.aggregate_rows_ds(&out)
                .to_tensor(dev)?
                .transpose(0, 1)?
                .contiguous()?
        } else {
            out.to_tensor(dev)?.transpose(0, 1)?.contiguous()?
        }
    } else if let Some(fc) = coarsening {
        fc.aggregate_sparse_csc(csc)
            .to_tensor(dev)?
            .transpose(0, 1)?
            .contiguous()?
    } else {
        csc.to_tensor(dev)?.transpose(0, 1)?.contiguous()?
    };
    Ok(nd)
}

////////////////////////
// Indexed prediction //
////////////////////////
