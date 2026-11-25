//! The generic dense topic pipeline: per-decoder configuration, training
//! over the collapse levels, metadata and evaluation, and the multi-decoder fit.

use crate::topic::common::move_varmap_to_cpu;
use crate::topic::eval::{evaluate_latent_by_encoder, EvaluateLatentConfig};
use crate::topic::train::{train_mixed, TrainConfig};
use senna::embed_common::*;

use legume_numeric::candle::decoder::*;
use legume_numeric::candle::encoder::*;
use legume_numeric::candle::traits::*;
use log::warn;

use super::{DecoderType, TopicArgs};
use crate::topic::decoder_output::{write_dictionary_tensor, DecoderExtras};

// Decoder-specific post-training I/O moved to `crate::topic::decoder_output`.
// This file keeps only the per-decoder hyperparameter `ConfigureDecoder`
// trait, since that surface depends on the local `TopicArgs`.

/// Trait for optional per-run hyperparameter configuration from CLI args.
/// Default is no-op; specific decoders override to set their own knobs.
/// Per-feature Fisher weights flow through `DecoderModuleT::attach_feature_weights`
/// (declared in `legume_numeric::candle`) rather than this trait, so callers outside
/// senna (e.g. `predict`) can attach weights without importing this.
pub(super) trait ConfigureDecoder {
    fn configure(&mut self, _args: &TopicArgs) {}
}

impl ConfigureDecoder for MultinomTopicDecoder {}
impl ConfigureDecoder for NbTopicDecoder {}
impl ConfigureDecoder for NbMixtureTopicDecoder {
    fn configure(&mut self, args: &TopicArgs) {
        self.set_rho_prior(
            args.rho_prior_weight,
            args.rho_prior_alpha,
            args.rho_prior_beta,
        );
    }
}

////////////////////////////
// Generic topic pipeline //
////////////////////////////

pub(super) struct PipelineCtx<'a> {
    pub(super) level_decoder_dims: &'a [usize],
    pub(super) n_topics: usize,
    pub(super) param_builder: &'a candle_nn::VarBuilder<'a>,
    pub(super) collapsed_levels: &'a [CollapsedOut],
    pub(super) level_coarsenings: &'a [Option<FeatureCoarsening>],
    pub(super) finest_coarsening: Option<&'a FeatureCoarsening>,
    pub(super) finest_collapsed: &'a CollapsedOut,
    /// Finest-level cell → pb membership, for anything that needs to know how
    /// many cells a pseudobulk stands for. `None` when the run built no hierarchy.
    pub(super) cell_to_pb_finest: Option<&'a [usize]>,
    pub(super) n_features_full: usize,
    pub(super) gene_names: &'a [Box<str>],
    /// How this run's gene axis lines up with the `--init-from` source run's,
    /// when the two differ. `None` for a fresh fit or an identical axis.
    pub(super) gene_axis: Option<&'a crate::topic::eval::GeneRemap>,
    pub(super) data_vec: &'a SparseIoVec,
    pub(super) parameters: &'a candle_nn::VarMap,
    pub(super) dev: &'a candle_core::Device,
    pub(super) args: &'a TopicArgs,
    pub(super) stop: &'a std::sync::atomic::AtomicBool,
    /// Data-driven β prior built from finest-level pseudobulks. Used for
    /// β init and (when `--anchor-penalty > 0`) as a training-time
    /// cross-entropy penalty.
    pub(super) anchor_prior: Option<&'a crate::topic::anchor_prior::AnchorPrior>,
    /// Per-level `[D_l, K]` anchor tensors pre-built on `dev`.
    pub(super) anchor_prior_per_level: Option<&'a [candle_core::Tensor]>,
    /// The encoder's training null per level under `--adj-method batch`.
    pub(super) batch_nulls: Option<&'a [Mat]>,
    /// Per-feature statistics computed once, threaded through every
    /// stage that needs them (encoder construction, decoder Fisher
    /// weighting, model save).
    pub(super) feature_stats: FeatureStats<'a>,
}

/// Per-feature statistics derived from the finest-level pseudobulk
/// posterior + a streaming pass over coarsened cells. Bundled together
/// because all three flow as a unit through `PipelineCtx` and the
/// save/predict path.
pub(super) struct FeatureStats<'a> {
    /// Per-gene mean rate `μ_d` at the encoder's `D_coarse` (or
    /// `D_full` when no coarsening) — what the encoder actually divides
    /// by during forward.
    pub(super) mean: &'a [f32],
    /// Same `μ_d` at full `D_full`, gene-name aligned. Saved to disk
    /// so predict can re-aggregate via the loaded coarsening matrix.
    pub(super) mean_full: &'a [f32],
    /// Per-level NB-Fisher weights `w_d ∈ (0, 1]` at each decoder's `D_l`,
    /// computed *after* feature coarsening (so the dispersion trend is
    /// fit at the resolution the decoder actually sees). Multiplicative
    /// per-gene weight on the multinomial loss term — housekeeping
    /// observations contribute less to β's gradient. Per-level vectors
    /// align with `level_decoder_dims`.
    pub(super) fisher_per_level: &'a [Vec<f32>],
}

/// Draw this run's weights from `--seed`, then load an `--init-from`
/// checkpoint over them, if one was named (a var it does not cover, such as
/// an added topic, keeps its seeded draw).
///
/// Both dense pipelines initialize identically — same architecture
/// invariants, same growth surface — so the check lives here rather than
/// being written out at each of them.
fn initialize_dense(ctx: &PipelineCtx<'_>) -> anyhow::Result<()> {
    use crate::topic::warm_start::{warm_start_load, GeneAxisGrowth, WarmStartCheck};
    legume_numeric::candle::nn::seed_declared_vars(ctx.parameters, ctx.args.seed, |_| false)?;
    let Some(prefix) = ctx.args.init_from.as_deref() else {
        return Ok(());
    };
    let n_features_encoder = *ctx
        .level_decoder_dims
        .last()
        .unwrap_or(&ctx.n_features_full);
    warm_start_load(
        ctx.parameters,
        prefix,
        &WarmStartCheck {
            model_type_expected: crate::topic::model_metadata::MODEL_TYPE_TOPIC,
            n_topics: ctx.n_topics,
            n_features_full: ctx.n_features_full,
            n_features_encoder,
            encoder_hidden: &ctx.args.encoder_layers,
            level_decoder_dims: ctx.level_decoder_dims,
            embedding_dim: None,
            growth: crate::topic::warm_start::Growth {
                add_topics: ctx.args.add_topics,
                add_embedding_dim: 0,
            },
            // This family has no per-gene embedding: its gene-keyed state is
            // the decoder dictionary, which the loader gathers by itself.
            n_gene_modules: 0,
            gene_axis: ctx.gene_axis.map(|remap| GeneAxisGrowth {
                remap,
                coarsening: None,
            }),
        },
    )
}

pub(super) fn run_topic_pipeline<Enc, Dec>(
    ctx: &PipelineCtx,
    encoder: &mut Enc,
) -> anyhow::Result<(TrainScores, Mat)>
where
    Enc: EncoderModuleT + Send + Sync,
    Dec: DecoderModuleT + DecoderExtras + NewDecoder + ConfigureDecoder + Send + Sync,
{
    let mut decoders: Vec<Dec> = ctx
        .level_decoder_dims
        .iter()
        .enumerate()
        .map(|(i, &d_l)| {
            Dec::new(d_l, ctx.n_topics, ctx.param_builder.pp(format!("dec_{i}")))
                .expect("decoder creation")
        })
        .collect();
    for (level, dec) in decoders.iter_mut().enumerate() {
        dec.configure(ctx.args);
        // Attach per-level NB-Fisher weights (coarse-resolution). No-op
        // for decoders that don't use them.
        dec.attach_feature_weights(&ctx.feature_stats.fisher_per_level[level], ctx.dev)?;
    }

    // β init from anchor prior is disabled — random (Kaiming) initialisation
    // works well when the anchor penalty (default 1.0) pulls β toward the
    // prior during training. Warm-starting logits with log(anchor) can lock
    // the dictionary too early.

    // Seeded weights, then an optional model-checkpoint warm-start: encoder +
    // decoder weights from a previously trained run (same architecture).
    initialize_dense(ctx)?;

    let train_config = TrainConfig {
        parameters: ctx.parameters,
        dev: ctx.dev,
        args: ctx.args,
        stop: ctx.stop,
        anchor_prior_per_level: ctx.anchor_prior_per_level,
        anchor_penalty: ctx.args.anchor_penalty,
        batch_nulls: ctx.batch_nulls,
    };
    let scores = train_mixed(
        ctx.collapsed_levels,
        encoder,
        &decoders,
        ctx.level_coarsenings,
        &train_config,
    )?;

    // PB-level topic usage + persistence of pb_gene / pb_latent for
    // downstream `lupin annotate` (enrichment) (enrichment-based annotation works from
    // PB aggregates, no zarr reopen needed).
    {
        let enc_fc = ctx.level_coarsenings.last().and_then(|c| c.as_ref());
        // The finest level's draw, the same one it trained on.
        let finest = ctx.collapsed_levels.len() - 1;
        let (mixed, batch, _) = crate::topic::common::sample_collapsed_data(
            ctx.finest_collapsed,
            crate::topic::common::posterior_seed(ctx.args.seed, finest),
            ctx.batch_nulls.map(|n| &n[finest]),
        )?;
        let enc_nd = if let Some(fc) = enc_fc {
            fc.aggregate_columns_nd(&mixed)
        } else {
            mixed
        };
        let batch_nd = batch.map(|b| {
            if let Some(fc) = enc_fc {
                fc.aggregate_columns_nd(&b)
            } else {
                b
            }
        });
        let enc_t = enc_nd.to_tensor(ctx.dev)?;
        let batch_t = batch_nd.map(|b| b.to_tensor(ctx.dev)).transpose()?;
        let (log_z, _) = encoder.forward_t(&enc_t, batch_t.as_ref(), false)?;
        let theta: Vec<Vec<f32>> = log_z.exp()?.to_vec2()?;
        let n_pb = theta.len();
        let k = theta[0].len();
        let mean_t: Vec<f32> = (0..k)
            .map(|ki| theta.iter().map(|r| r[ki]).sum::<f32>() / n_pb as f32)
            .collect();
        let active = mean_t.iter().filter(|&&v| v > 0.01).count();
        info!(
            "PB topic usage ({} PBs): mean_θ={:?}, active={}/{}",
            n_pb,
            mean_t.iter().map(|v| format!("{v:.3}")).collect::<Vec<_>>(),
            active,
            k,
        );

        let mut pb_latent_pk = Mat::zeros(n_pb, k);
        for (pi, row) in theta.iter().enumerate() {
            for (kj, v) in row.iter().enumerate() {
                pb_latent_pk[(pi, kj)] = *v;
            }
        }
        let pb_names = axis_id_names("PB_", n_pb);
        let topic_names = axis_id_names("T", k);
        pb_latent_pk.to_parquet_with_names(
            &format!("{}.pb_latent.parquet", ctx.args.out),
            (Some(&pb_names), Some("pb")),
            Some(&topic_names),
        )?;

        let pb_gene_gp: Mat = ctx.finest_collapsed.mu_observed.posterior_mean().clone();
        senna::output_helpers::save_pb_gene(ctx.args.out.as_ref(), &pb_gene_gp, ctx.gene_names)?;

        // Empirical NB-Fisher-weighted gene × topic dictionary at full gene
        // resolution. Avoids the lossy expand-from-coarse approximation in
        // `dictionary.parquet` so rare informative genes survive into the
        // annotate-side enrichment ranking.
        info!("Computing NB Fisher gene weights for empirical dictionary");
        // Full gene resolution — the point of the empirical dictionary is to
        // escape the coarsening, so no `FeatureCoarsening` here.
        let fisher_w = crate::refine_weighting::fit_fisher_weights(
            ctx.finest_collapsed,
            ctx.cell_to_pb_finest,
            None,
            ctx.data_vec,
            ctx.args.block_size,
        )?;
        senna::output_helpers::save_fisher_weights(
            ctx.args.out.as_ref(),
            &fisher_w,
            ctx.gene_names,
        )?;
        info!(
            "Wrote {}.fisher_weights.parquet ({} genes)",
            ctx.args.out,
            fisher_w.len()
        );

        let beta_emp = crate::empirical_dict::build_empirical_dictionary(
            &pb_gene_gp,
            &pb_latent_pk,
            &fisher_w,
        );
        beta_emp.to_parquet_with_names(
            &format!("{}.dictionary_empirical.parquet", ctx.args.out),
            (Some(ctx.gene_names), Some("gene")),
            Some(&topic_names),
        )?;
        info!(
            "Wrote empirical dictionary {}×{} (NB-Fisher-weighted, column-simplex)",
            beta_emp.nrows(),
            beta_emp.ncols()
        );
    }

    info!("Writing down the model parameters");

    let finest_decoder = decoders.last().unwrap();
    finest_decoder.write_dictionary(
        ctx.finest_coarsening,
        ctx.n_features_full,
        ctx.gene_names,
        &ctx.args.out,
    )?;
    finest_decoder.write_extras(
        ctx.finest_coarsening,
        ctx.n_features_full,
        ctx.gene_names,
        &ctx.args.out,
    )?;

    let weights = compute_decoder_weights(&ctx.args.decoder, ctx.args.decoder_weights.as_ref());
    let z_nk = save_metadata_and_evaluate::<Dec>(ctx, &weights)?;
    Ok((scores, z_nk))
}

/// Compute normalized decoder weights. If user provided weights, normalize
/// them to sum to 1. Otherwise, use equal weights.
fn compute_decoder_weights(decoders: &[DecoderType], user_weights: Option<&Vec<f64>>) -> Vec<f64> {
    if let Some(w) = user_weights {
        let sum: f64 = w.iter().sum();
        w.iter().map(|x| x / sum).collect()
    } else {
        let n = decoders.len() as f64;
        vec![1.0 / n; decoders.len()]
    }
}

/// Save model metadata/weights, move parameters to CPU, and evaluate latent states.
///
/// When `--amort-refine-steps > 0`, rebuilds the finest-level decoder on CPU
/// and uses it for per-cell likelihood refinement during evaluation.
fn save_metadata_and_evaluate<Dec>(
    ctx: &PipelineCtx,
    decoder_weights: &[f64],
) -> anyhow::Result<Mat>
where
    Dec: DecoderModuleT + NewDecoder + ConfigureDecoder + Send + Sync,
{
    use crate::topic::model_metadata::{
        save_coarsening_levels, save_feature_mean, save_parameters, TopicModelMetadata,
    };
    save_parameters(ctx.parameters, &ctx.args.out)?;
    // Persist `μ_d` at `D_full` (gene-name aligned). At predict time we
    // reload it and aggregate through the saved coarsening matrix to
    // get the encoder's `D_coarse` view.
    save_feature_mean(ctx.feature_stats.mean_full, ctx.gene_names, &ctx.args.out)?;
    // Persist NB-Fisher weights at the *finest* level's `D_l` (= what
    // `MultinomTopicDecoder` actually applies). Predict reloads it and
    // attaches to the rebuilt decoder. We only save the finest level
    // because predict only uses the finest decoder.
    {
        let finest_idx = ctx.feature_stats.fisher_per_level.len().saturating_sub(1);
        let finest_w = &ctx.feature_stats.fisher_per_level[finest_idx];
        data_beans::alg::gene_weighting::save_fisher_weights_coarse(&ctx.args.out, finest_w)?;
    }

    let mut metadata = TopicModelMetadata {
        model_type: crate::topic::model_metadata::MODEL_TYPE_TOPIC.into(),
        decoder_types: ctx.args.decoder.iter().map(|d| d.as_str().into()).collect(),
        decoder_weights: decoder_weights.to_vec(),
        n_features_encoder: *ctx
            .level_decoder_dims
            .last()
            .unwrap_or(&ctx.n_features_full),
        n_features_full: ctx.n_features_full,
        n_topics: ctx.n_topics,
        encoder_hidden: ctx.args.encoder_layers.clone(),
        num_levels: ctx.level_decoder_dims.len(),
        level_decoder_dims: ctx.level_decoder_dims.to_vec(),
        adj_method: ctx.args.adj_method.as_str().into(),
        batch_null_in_training: ctx.batch_nulls.is_some(),
        has_coarsening: ctx.finest_coarsening.is_some(),
        embedding_dim: None,
        theta_mean: None,
        n_train_cells: Some(ctx.data_vec.num_columns()),
        // Dense `topic` uses no indexed encoder, so the module branch does not apply.
        n_gene_modules: None,
        query_rank: None,
    };
    metadata.save(&ctx.args.out)?;

    // Every level, not just the finest: `predict` reads only the finest, but a
    // `--init-from` child has to reproduce each decoder level's own grouping.
    if ctx.level_coarsenings.iter().any(Option::is_some) {
        save_coarsening_levels(ctx.level_coarsenings, &ctx.args.out)?;
    }

    // Move VarMap to CPU, then rebuild encoder from CPU Vars.
    // The old encoder still holds Metal/CUDA Vars and must not be reused.
    info!("Moving parameters to CPU for multi-threaded inference");
    let cpu_dev = candle_core::Device::Cpu;
    move_varmap_to_cpu(ctx.parameters)?;

    let n_features_encoder = ctx
        .finest_coarsening
        .map_or(ctx.n_features_full, |c| c.num_coarse);
    let cpu_vb =
        candle_nn::VarBuilder::from_varmap(ctx.parameters, candle_core::DType::F32, &cpu_dev);
    let cpu_encoder = LogSoftmaxEncoder::new(
        LogSoftmaxEncoderArgs {
            n_features: n_features_encoder,
            n_topics: ctx.n_topics,
            layers: &ctx.args.encoder_layers,
            feature_mean: Some(ctx.feature_stats.mean),
        },
        ctx.parameters,
        cpu_vb.clone(),
    )?;

    let refine_config = ctx.args.amort_refine.to_config();

    let finest_dec_dim = *ctx
        .level_decoder_dims
        .last()
        .unwrap_or(&ctx.n_features_full);
    let finest_dec_idx = ctx.level_decoder_dims.len().saturating_sub(1);
    let refine_decoder = if refine_config.is_some() {
        let cpu_vb =
            candle_nn::VarBuilder::from_varmap(ctx.parameters, candle_core::DType::F32, &cpu_dev);
        let mut d = Dec::new(
            finest_dec_dim,
            ctx.n_topics,
            cpu_vb.pp(format!("dec_{finest_dec_idx}")),
        )?;
        // Re-attach the finest level's NB-Fisher weights so predictive llik
        // uses the same loss as training.
        d.attach_feature_weights(
            &ctx.feature_stats.fisher_per_level[finest_dec_idx],
            &cpu_dev,
        )?;
        Some(d)
    } else {
        None
    };

    let eval_config = EvaluateLatentConfig {
        dev: &cpu_dev,
        adj_method: &ctx.args.adj_method,
        minibatch_size: ctx.args.minibatch_size,
        feature_coarsening: ctx.finest_coarsening,
        decoder: refine_decoder.as_ref(),
        refine_config: refine_config.as_ref(),
    };
    let z_nk = evaluate_latent_by_encoder(
        ctx.data_vec,
        &cpu_encoder,
        ctx.finest_collapsed,
        &eval_config,
    )?;

    // Re-save metadata with θ̄_train populated — initial δ guess at predict
    // time, better than uniform 1/K when training is composition-imbalanced.
    // `senna topic` stores log θ, so proportions are just `exp`.
    metadata.populate_theta_mean_and_save(&z_nk.map(f32::exp), &ctx.args.out)?;
    Ok(z_nk)
}

/// Multi-decoder pipeline: builds multiple decoder types per level,
/// trains with weighted multi-decoder loss, saves per-decoder dictionaries.
pub(super) fn run_multi_decoder_pipeline<Enc: EncoderModuleT + Send + Sync>(
    ctx: &PipelineCtx,
    encoder: &mut Enc,
) -> anyhow::Result<(TrainScores, Mat)> {
    use crate::topic::train::train_mixed_multi_decoder;
    use legume_numeric::candle::decoder::{create_dyn_decoder, DynDecoderModuleT};

    let decoder_weights =
        compute_decoder_weights(&ctx.args.decoder, ctx.args.decoder_weights.as_ref());

    // Build per-level × per-decoder-type grid
    let decoders_per_level: Vec<Vec<Box<dyn DynDecoderModuleT>>> = ctx
        .level_decoder_dims
        .iter()
        .enumerate()
        .map(|(level_i, &d_l)| {
            ctx.args
                .decoder
                .iter()
                .map(|dec_type| {
                    let name = dec_type.as_str();
                    let prefix = format!("dec_{level_i}.{name}");
                    create_dyn_decoder(name, d_l, ctx.n_topics, ctx.param_builder.pp(prefix))
                        .expect("decoder creation")
                })
                .collect()
        })
        .collect();

    // Seeded weights, then an optional warm-start (multi-decoder variant).
    initialize_dense(ctx)?;

    let train_config = TrainConfig {
        parameters: ctx.parameters,
        dev: ctx.dev,
        args: ctx.args,
        stop: ctx.stop,
        anchor_prior_per_level: None,
        anchor_penalty: 0.0,
        batch_nulls: ctx.batch_nulls,
    };

    if ctx.anchor_prior.is_some() {
        warn!("anchor prior is not applied in multi-decoder mode; β init + penalty skipped");
    }

    let scores = train_mixed_multi_decoder(
        ctx.collapsed_levels,
        encoder,
        &decoders_per_level,
        ctx.level_coarsenings,
        &decoder_weights,
        &train_config,
    )?;

    // Write per-decoder dictionaries at finest level
    info!("Writing down the model parameters");
    for dec in decoders_per_level.last().unwrap() {
        let name = dec.decoder_name();
        let out_prefix = format!("{}.{}", ctx.args.out, name);
        let dict_tensor = dec.get_dictionary()?;
        write_dictionary_tensor(
            &dict_tensor,
            ctx.finest_coarsening,
            ctx.n_features_full,
            ctx.gene_names,
            &out_prefix,
        )?;
    }

    let z_nk = save_metadata_and_evaluate::<MultinomTopicDecoder>(ctx, &decoder_weights)?;
    Ok((scores, z_nk))
}
