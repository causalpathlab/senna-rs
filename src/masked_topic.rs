use crate::topic::common::{
    load_and_collapse, move_varmap_to_cpu, setup_stop_handler, LoadCollapseArgs, PreparedData,
};
use crate::topic::eval_indexed::{evaluate_latent_masked, EvaluateLatentMaskedConfig};
use crate::topic::train_masked::{train_masked, write_feature_embedding, write_masked_dictionary};
use senna::embed_common::*;

use legume_numeric::candle::decoder::masked_etm::{log_background_from_mean, pin_background};
use legume_numeric::candle::decoder::EmbeddedNbTopicDecoder;
use legume_numeric::candle::encoder::*;
use legume_numeric::candle::vae::masked_topic::{IndexedTrainConfig, LatentHead};
use log::warn;

mod args;

pub use args::{MaskScheduleArg, MaskedTopicArgs};
pub use data_beans::aux::feature_names::FeatureNameKindArg;

/// `senna masked-topic` — softmax simplex-`θ`, deterministic (no-KL) masked ETM.
pub fn fit_masked_topic_model(args: &MaskedTopicArgs) -> anyhow::Result<()> {
    fit_masked_model(args, LatentHead::Softmax)
}

/// `senna masked-sbp` — **stick-breaking process** simplex-`θ`, deterministic
/// (no-KL) masked ETM. Same pipeline as `masked-topic`; the encoder maps its
/// logits through a stick-breaking simplex instead of softmax, giving ordered,
/// exchangeability-broken topics with a self-pruning tail.
pub fn fit_masked_sbp_model(args: &MaskedTopicArgs) -> anyhow::Result<()> {
    fit_masked_model(args, LatentHead::StickBreaking)
}

/// `senna masked-vae` — unconstrained-latent masked ETM. Shares the
/// masked-topic pipeline (PB training, encoder-only cell eval, ETM ρ/α decoder);
/// the encoder emits a raw `z` (no softmax) that the decoder reads through
/// `log_softmax`, and the latent written out is that raw `z`. Deterministic and
/// KL-free like its siblings: the masked objective is the regularizer.
pub fn fit_masked_vae_model(args: &MaskedTopicArgs) -> anyhow::Result<()> {
    fit_masked_model(args, LatentHead::Gaussian)
}

/// The manifest kind for a masked run. `masked-vae` gets its own kind because
/// its latent is a Gaussian `z`, not `log θ` — see [`RunKind::latent_is_log_simplex`].
pub(crate) fn masked_run_kind(head: LatentHead) -> senna::run_manifest::RunKind {
    match head {
        LatentHead::Gaussian => senna::run_manifest::RunKind::MaskedVae,
        LatentHead::Softmax | LatentHead::StickBreaking => senna::run_manifest::RunKind::Itopic,
    }
}

pub(crate) fn fit_masked_model(args: &MaskedTopicArgs, head: LatentHead) -> anyhow::Result<()> {
    mkdir_parent(&args.out)?;
    args.validate()?;
    let recorded_query = args.recorded_query_flags();
    if !recorded_query.is_empty() {
        info!(
            "this run was recorded with {}; the query decoder is no longer wired into the \
             masked family, so these no longer apply",
            recorded_query.join(", ")
        );
    }

    let k = args.n_latent_topics;

    // --from chain-inheritance: explicit CLI flags win per-field; with no
    // explicit init/freeze flag, default to freezing the inherited ρ.
    let inherited = args
        .from
        .as_deref()
        .map(senna::run_manifest::inherit_from)
        .transpose()?;
    if let Some(inh) = inherited.as_ref() {
        info!(
            "--from: inheriting data + batch + ρ-prefix from a '{}' manifest",
            inh.source_kind
        );
    }
    senna::run_manifest::InheritedFromManifest::ensure_replayable(
        inherited.as_ref(),
        "masked-topic",
    )?;
    let data_files = senna::run_manifest::InheritedFromManifest::resolve_data(
        inherited.as_ref(),
        &args.data_files,
    )?;
    let batch_files = senna::run_manifest::InheritedFromManifest::resolve_batch(
        inherited.as_ref(),
        args.batch_files.as_deref(),
    );

    // The flag given, else the parent manifest's table, pinned.
    let preset_mode: Option<(Box<str>, graph_embedding_util::PresetMode)> =
        match args.feature_embedding.resolve()? {
            Some((p, m)) => Some((Box::from(p), m)),
            None => inherited.as_ref().map(|inh| {
                (
                    inh.feature_embedding_prefix.clone(),
                    graph_embedding_util::PresetMode::Freeze,
                )
            }),
        };
    let pretrained_prefix = preset_mode.as_ref().map(|(p, _)| p.as_ref());
    let pinned_rho = preset_mode.as_ref().is_some_and(|(_, m)| m.pins());
    let lora = preset_mode.as_ref().and_then(|(_, m)| m.lora());
    let preset_flag = preset_mode
        .as_ref()
        .map(|(_, m)| crate::feature_embedding_args::flag_name(*m))
        .unwrap_or_default();
    // Whether this run's feature side is the checkpoint's is decidable from the
    // checkpoint's metadata, so ask now rather than after the import and the
    // collapse. Re-checked inside the warm start, which every entry point takes.
    if let Some(prefix) = args.init_from.as_deref() {
        crate::topic::warm_start::check_feature_side(prefix, 0, args.add_embedding_dim)?;
    }

    // Resolved before H so the pre-trained dictionary's column count
    // pins the encoder dim.
    let pretrained_spec: Option<crate::topic::freeze::FrozenFeatureSpec> = match pretrained_prefix {
        None => None,
        Some(prefix) => {
            if pinned_rho {
                anyhow::ensure!(
                    args.feature_network.is_none(),
                    "{preset_flag} is incompatible with --feature-network \
                     (network restriction would change the gene axis that the pinned ρ holds)"
                );
            }
            // Pre-train inputs are gene-keyed; row names aren't available yet.
            let kind = args.feature_name_kind.resolve_or_gene();
            Some(
                crate::topic::freeze::FrozenFeatureSpec::resolve_from_prefix(
                    prefix,
                    preset_flag,
                    kind,
                )?,
            )
        }
    };

    let pretrained_h: Option<usize> = pretrained_spec
        .as_ref()
        .map(super::topic::freeze::FrozenFeatureSpec::dictionary_h)
        .transpose()?;
    let h = crate::topic::common::resolve_embedding_dim(args.embedding_dim, pretrained_h, k)?;
    // The LoRA settings against the width the table fixes, as every engine does.
    if let Some((_, mode)) = preset_mode.as_ref() {
        mode.validate(h)?;
    }

    let prebuilt_partition = args.pb_from.partition(
        inherited.as_ref(),
        &crate::refine_weighting::partition_settings(args),
    )?;

    let effective_hvg =
        crate::hvg::resolve_multiome_with_hvg(args.multiome, data_files.len(), &args.hvg);
    let effective_multiome = effective_hvg.multiome;

    let net_opts = crate::topic::common::FeatureNetworkOpts {
        prefix_match: args.feature_network_prefix_match,
        delim: args.feature_network_delim,
        min_shared_neighbors: args.feature_network_min_shared_neighbors,
        max_degree: args.feature_network_max_degree,
        min_degree: args.feature_network_min_degree,
    };
    let restrict_path = if args.no_feature_network_restrict {
        None
    } else {
        args.feature_network.as_deref()
    };

    let feature_network = crate::topic::common::setup_feature_network(restrict_path, net_opts);
    let pretrained_mask_holder = pretrained_spec.as_ref().map(|s| s.mask_fn());
    let feature_mask_fn: Option<&crate::topic::common::FeatureMaskFn> = match (
        pretrained_mask_holder.as_deref(),
        feature_network.mask_fn.as_deref(),
    ) {
        (Some(fm), _) => Some(fm),
        (None, Some(nm)) => Some(nm),
        (None, None) => None,
    };

    let PreparedData {
        data_vec,
        collapsed_levels,
        proj_kn,
        cell_to_pb_per_level,
        pb_tree,
        output_keep_idx,
    } = load_and_collapse(&LoadCollapseArgs {
        data_files: &data_files,
        batch_files: &batch_files,
        preload: args.preload_data,
        proj_dim: args.collapse.proj_dim.max(args.n_latent_topics),
        sort_dim: args.collapse.sort_dim,
        knn_cells: args.collapse.knn_cells,
        num_levels: args.collapse.num_levels,
        iter_opt: args.collapse.iter_opt,
        block_size: args.block_size,
        out: Some(&args.out),
        max_features: effective_hvg.n_hvg,
        feature_list_file: effective_hvg.feature_list_file,
        must_train_file: effective_hvg.must_train_file,
        qc: args.qc.to_config(),
        qc_block_size: args.block_size,
        qc_report_out: args.qc.qc_report.as_deref(),
        feature_mask_fn,
        pb_reference: args.pb_reference.as_ref(),
        mixture_batches: args.collapse.mixture_batch.clone(),
        observe_panels: !effective_multiome,
        row_alignment: data_beans::sparse_io_vector::RowAlignment::default(),
        column_alignment: if effective_multiome {
            data_beans::sparse_io_vector::ColumnAlignment::Union
        } else {
            data_beans::sparse_io_vector::ColumnAlignment::Disjoint
        },
        feature_kind: args.feature_name_kind.clone().into(),
        refine: args.collapse.pb_refine.to_params(),
        pb_tree: args.collapse.pb_tree_params(),
        ignore_batch: args.collapse.ignore_batch,
        want_hierarchy: true,
        prebuilt_partition,
        cnv_clones: args.collapse.cnv_clones.as_deref(),
        multiome: None,
        reuse: args.pb_from.pb_from.as_deref(),
    })?;

    let finest_collapsed: &CollapsedOut = collapsed_levels.last().unwrap();

    // 4. No feature coarsening for indexed model — both encoder and decoder
    //    use indexed top-K lookup, so D_full is efficient.
    //    Levels differ only in sample coarsening (N).
    let n_features_full = data_vec.num_rows();
    let num_levels = collapsed_levels.len();

    // 5. Train masked topic model on collapsed data
    let n_topics = args.n_latent_topics;

    let dev = args.compute.to_device()?;

    let parameters = candle_nn::VarMap::new();
    let param_builder =
        candle_nn::VarBuilder::from_varmap(&parameters, candle_core::DType::F32, &dev);

    // Gene names — used for output artifacts further down.
    let gene_names = data_vec.row_names()?;

    // `--feature-network` on the masked path is used for feature *restriction*
    // only (applied above when building `feature_network`). Graph diffusion is
    // NOT supported by the masked encoder: the encoder takes no edges at all
    // any more, and the encoder-only eval path reads CSC directly with no edge
    // cache, so there is nothing for a network to diffuse over here.
    if args.feature_network.is_some() && args.no_feature_network_restrict {
        warn!(
            "--feature-network with --no-feature-network-restrict has no effect on \
             masked-topic: graph diffusion is not supported by the masked encoder, \
             and restriction is disabled."
        );
    }

    let base_encoder = IndexedEmbeddingEncoder::new(
        IndexedEmbeddingEncoderArgs {
            n_features: n_features_full,
            n_topics,
            embedding_dim: h,
            layers: &args.encoder_layers,
            attn_pool: true,
            n_gene_modules: 0,
            lora_rank: lora.map_or(0, |l| l.rank),
        },
        &parameters,
        param_builder.pp(crate::topic::gene_axis::ENCODER_PREFIX),
    )?;

    // Per-level decoders: all at D_full, levels differ in N (sample coarsening).
    // ETM-factorized — each decoder shares the encoder's feature embeddings ρ,
    // and learns only its own topic embeddings α_{level} [K, H].
    // Per-gene mean expression rate `μ_d` from the finest-level pseudobulk
    // posterior. The encoder composes it with the per-cell batch null as a
    // multiplicative count-rate divisor before Anscombe — joint correction
    // for batch effect × gene-typical-rate, leaving the cell's biological
    // deviation. Stored as the raw mean; Anscombe is applied inside the
    // encoder via `anscombe_lite`.
    let feature_mean: Vec<f32> = {
        let mu = finest_collapsed.mu_observed.posterior_mean();
        let n_pb = mu.ncols().max(1) as f32;
        (0..n_features_full)
            .map(|d| mu.row(d).iter().sum::<f32>() / n_pb)
            .collect()
    };

    // Per-level gene → module maps for the decoder targets (identity when
    // `--max-coarse-features` is off), keyed like `senna topic`'s coarsening
    // and persisted with the model.
    // Where this run's gene axis differs from an `--init-from` source run's,
    // that run's gene-keyed state is continued by NAME: its modules are grown
    // onto this axis below, and its ρ at the warm start. A cohort measuring
    // genes the model has never seen is then absorbed rather than refused.
    let gene_axis = crate::topic::gene_axis::remap_for_init_from(
        args.init_from.as_deref(),
        &args.feature_name_kind,
        &gene_names,
    )?;
    let cell_to_pb_finest: &[usize] = cell_to_pb_per_level
        .as_deref()
        .and_then(<[Vec<usize>]>::last)
        .map(Vec::as_slice)
        .ok_or_else(|| {
            anyhow::anyhow!("feature coarsening needs the finest cell → pseudobulk membership")
        })?;
    let level_coarsenings = crate::topic::common::resolve_level_coarsenings(
        args.coarsening.cap(),
        args.init_from.as_deref(),
        &crate::topic::common::FinestPseudobulks {
            collapsed: finest_collapsed,
            cell_to_pb: cell_to_pb_finest,
        },
        num_levels,
        n_features_full,
        args.seed,
        gene_axis.as_ref(),
    )?;
    // The decoders share the encoder's feature side by handle, not by value:
    // with modules the rows are computed, so a copy would freeze them.
    let shared_features = base_encoder.features_shared();
    let mut decoders: Vec<EmbeddedNbTopicDecoder> = Vec::with_capacity(num_levels);
    let mut backgrounds = Vec::with_capacity(num_levels);
    for (i, fc) in level_coarsenings.iter().enumerate() {
        let (map, coarse_mass) =
            crate::topic::train_masked::coarsening_map_for(fc.as_ref(), &feature_mean, &dev)?;
        decoders.push(EmbeddedNbTopicDecoder::new_with_coarsening(
            n_topics,
            std::sync::Arc::clone(&shared_features),
            map,
            param_builder.pp(format!("dec_{i}")),
        )?);
        backgrounds.push(log_background_from_mean(&coarse_mass, &dev)?);
    }
    // Every var is declared: draw them from `--seed`, before anything below
    // (a background, a pre-trained ρ, a checkpoint) sets one from data.
    legume_numeric::candle::nn::seed_declared_vars(&parameters, args.seed, |_| false)?;
    // Pin each level's background at the data's marginal over its output
    // axis: the home for shared abundance that centering α removes from the
    // topics.
    for (i, log_pi) in backgrounds.iter().enumerate() {
        pin_background(&parameters, &format!("dec_{i}"), log_pi)?;
    }
    let level_decoder_dims: Vec<usize> = decoders.iter().map(|d| d.dim_obs()).collect();
    let has_coarsening = level_coarsenings.iter().any(Option::is_some);

    // Overwrite ρ in place with the pre-trained values BEFORE warm-start
    // from a prior topic checkpoint. The encoder/decoder both hold a
    // Tensor reference into the same Var, so a single `var.set(...)`
    // updates everywhere. The Var stays in the VarMap (round-trips
    // through safetensors); for freeze mode the optimizer excludes it
    // via `trainable_vars` (see `train_masked.rs`), for init mode it
    // keeps updating.
    let mut carried: Option<senna::carried_rows::CarriedRows> = None;
    if let Some(spec) = pretrained_spec.as_ref() {
        anyhow::ensure!(
            args.init_from.is_none(),
            "ρ pre-training is incompatible with --init-from \
             (warm-start would overwrite the pre-trained ρ from a different checkpoint)"
        );
        let host = spec.materialize(&gene_names)?;
        carried = senna::carried_rows::CarriedRows::from_unmatched(
            pinned_rho,
            preset_flag,
            &host,
            &gene_names,
            &spec.name_kind,
            spec.carried_types(&host),
            &spec.dictionary_path,
        )?;
        anyhow::ensure!(
            host.h == h,
            "pre-trained feature embedding has H={} but --embedding-dim={}",
            host.h,
            h
        );
        anyhow::ensure!(
            host.e_feat.nrows() == n_features_full,
            "pre-trained ρ has {} rows but the post-mask gene axis has {} — \
             loader and feature-mask disagree (this is a bug, please report)",
            host.e_feat.nrows(),
            n_features_full
        );
        legume_numeric::candle::frozen_features::overwrite_var_2d(
            &parameters,
            crate::topic::gene_axis::RHO_TENSOR,
            &host.e_feat,
            &dev,
        )?;
        if let Some(l) = lora {
            let (rank, ratio) = (l.rank, l.lr_ratio);
            info!(
                "LoRA: ρ anchored to {} (D={}, H={}); a rank-{rank} residual trains on top, \
                 V at {ratio}× the rate, with α + FC + BN",
                spec.dictionary_path, n_features_full, h
            );
        } else if pinned_rho {
            info!(
                "Freeze mode: ρ seeded from {} (D={}, H={}); encoder/decoders share frozen ρ, \
                 only α + FC + BN train",
                spec.dictionary_path, n_features_full, h
            );
        } else {
            info!(
                "Warm-start: ρ initialised from {} (D={}, H={}); AdamW continues to update it \
                 alongside α + FC + BN",
                spec.dictionary_path, n_features_full, h
            );
        }
    }

    if let Some(prefix) = args.init_from.as_deref() {
        use crate::topic::warm_start::{warm_start_load, GeneAxisGrowth, WarmStartCheck};
        warm_start_load(
            &parameters,
            prefix,
            &WarmStartCheck {
                // Warm-start from a checkpoint of the same head — the z_mean
                // semantics differ per head, so `--init-from` must match.
                model_type_expected: crate::topic::model_metadata::masked_model_type(head),
                n_topics,
                n_features_full,
                n_features_encoder: n_features_full,
                encoder_hidden: &args.encoder_layers,
                level_decoder_dims: &level_decoder_dims,
                embedding_dim: Some(h),
                growth: crate::topic::warm_start::Growth {
                    add_topics: args.add_topics,
                    add_embedding_dim: args.add_embedding_dim,
                },
                n_gene_modules: 0,
                // A free ρ restarts an unseen gene at the mean of its finest
                // coarse group, or at the global mean when the source run
                // trained at full resolution. A composed feature side ignores
                // this: its membership starts flat.
                gene_axis: gene_axis.as_ref().map(|remap| GeneAxisGrowth {
                    remap,
                    coarsening: level_coarsenings.last().and_then(Option::as_ref),
                }),
            },
        )?;
    }

    info!(
        "input: {} genes -> window-free masked encoder (emb={}, every gene read, zeros \
         included) -> {} decoders over {:?} (the genes the encoder did not see are scored)",
        n_features_full, h, num_levels, level_decoder_dims,
    );

    // Bulk deconvolution is not supported on the masked-imputation path; the
    // `-x/--bulk-data-files` flag that used to be accepted here (and silently
    // ignored) is gone. `senna deconvolve --bulk` is the supported route.

    let stop = setup_stop_handler();

    let lora_v_name =
        legume_numeric::candle::lora::factor_names(&legume_numeric::candle::lora::join(
            crate::topic::gene_axis::ENCODER_PREFIX,
            legume_numeric::candle::feature_embedding::LORA_PREFIX,
        ))
        .1;
    let train_config = IndexedTrainConfig {
        parameters: &parameters,
        dev: &dev,
        epochs: args.epochs,
        gpu_mem_fraction: args
            .minibatch_size
            .is_none()
            .then_some(args.gpu_mem_fraction),
        minibatch_size: args.minibatch_size.unwrap_or(100),
        learning_rate: args.learning_rate,
        topic_smoothing: args.topic_smoothing,
        stop: &stop,
        feature_mean: &feature_mean,
        grad_clip: args.grad_clip,
        feature_embedding_l2: args.feature_embedding_l2,
        weight_decay: args.weight_decay,
        feature_anchor: pinned_rho.then_some(
            legume_numeric::candle::vae::masked_topic::FeatureAnchor {
                base_var: crate::topic::gene_axis::RHO_TENSOR,
                lora: lora.map(|l| legume_numeric::candle::lora::LoraPlus {
                    v_var: &lora_v_name,
                    lr_ratio: l.lr_ratio,
                    ridge: l.ridge,
                }),
            },
        ),
    };

    use legume_numeric::candle::vae::masked_topic::{MaskSchedule, MaskedTrainOpts};
    let masked_opts = MaskedTrainOpts {
        mask_schedule: match args.mask_schedule {
            MaskScheduleArg::Fixed => MaskSchedule::Fixed,
            MaskScheduleArg::Uniform => MaskSchedule::Uniform {
                lo: args.mask_rate_lo,
                hi: args.mask_rate_hi,
            },
        },
        likelihood: args.masked_likelihood.to_lib(),
        latent: head,
        poisson_thin: args.poisson_thin,
        seed: args.seed,
    };

    // Under `--adj-method batch` the encoder trains on the δ it reads at
    // inference, mixed by each pseudobulk's batch shares.
    let batch_nulls = crate::topic::common::batch_level_nulls(
        &args.adj_method,
        &collapsed_levels,
        cell_to_pb_per_level.as_deref(),
        &data_vec,
    )?;
    let scores = train_masked(
        &collapsed_levels,
        &base_encoder,
        &decoders,
        &train_config,
        args.mask_fraction,
        &masked_opts,
        batch_nulls.as_deref(),
    )?;

    info!("Writing down the model parameters");

    // Use finest-level decoder for output
    let finest_decoder = decoders.last().unwrap();
    write_masked_dictionary(finest_decoder, &gene_names, &args.out)?;
    write_feature_embedding(&base_encoder.feature_embeddings()?, &gene_names, &args.out)?;

    // Optional held-out masked-imputation evaluation — the un-optimized
    // generalization metric (see `--eval-mask-fraction`). Runs on the training
    // device with the in-memory encoder + finest decoder, before the CPU move.
    if args.eval_mask_fraction > 0.0 {
        use crate::topic::eval_indexed::{evaluate_holdout_imputation, HoldoutEvalConfig};
        let delta_train = match args.adj_method {
            AdjMethod::Batch => finest_collapsed.delta.as_ref(),
            AdjMethod::Residual => finest_collapsed.mu_residual.as_ref(),
        }
        .map(|x| {
            x.posterior_mean()
                .to_tensor(&dev)
                .expect("delta to tensor")
                .transpose(0, 1)
                .expect("transpose")
                .contiguous()
                .expect("contiguous")
        });
        let holdout_cfg = HoldoutEvalConfig {
            dev: &dev,
            adj_method: &args.adj_method,
            minibatch_size: args.minibatch_size.unwrap_or(100),
            feature_mean: &feature_mean,
            head,
            likelihood: args.masked_likelihood.to_lib(),
            topic_smoothing: args.topic_smoothing,
            mask_fraction: args.eval_mask_fraction,
            seed: args.eval_seed,
        };
        let holdout_llik = evaluate_holdout_imputation(
            &data_vec,
            &base_encoder,
            finest_decoder,
            &holdout_cfg,
            delta_train.as_ref(),
        )?;
        info!(
            "Held-out imputation: mean log-likelihood/unit = {holdout_llik:.4} \
             (mask={}, seed={})",
            args.eval_mask_fraction, args.eval_seed
        );
    }

    // Persist trainable weights and model architecture
    // so `senna predict` (and `--init-from` re-runs) can rebuild this model.
    use crate::topic::model_metadata::{
        masked_decoder_type, masked_model_type, save_feature_mean, save_parameters,
        TopicModelMetadata,
    };

    // No feature graph is persisted on the masked path (GCN diffusion is not
    // wired into the masked encoder — see the encoder construction above).
    save_parameters(&parameters, &args.out)?;
    let mut metadata = TopicModelMetadata {
        model_type: masked_model_type(head).into(),
        decoder_types: vec![masked_decoder_type(head).into()],
        decoder_weights: vec![1.0],
        n_features_encoder: n_features_full,
        n_features_full,
        n_topics,
        encoder_hidden: args.encoder_layers.clone(),
        num_levels,
        level_decoder_dims: level_decoder_dims.clone(),
        adj_method: args.adj_method.as_str().into(),
        batch_null_in_training: batch_nulls.is_some(),
        has_coarsening,
        embedding_dim: Some(h),
        theta_mean: None,
        n_train_cells: Some(data_vec.num_columns()),
        // Round-trips the encoder's FC input width; without it every rebuild site
        // would construct `[L, H]` and `VarMap::load` would reject the checkpoint.
        n_gene_modules: Some(0),
        enc_context_size: None,
        query_rank: None,
    };
    metadata.save(&args.out)?;
    if has_coarsening {
        crate::topic::model_metadata::save_coarsening_levels(&level_coarsenings, &args.out)?;
    }
    save_feature_mean(&feature_mean, &gene_names, &args.out)?;

    // Move VarMap to CPU, then rebuild the encoder from the CPU Vars. The
    // masked model's inference is encoder-only (no decoder / no refinement).
    info!("Moving parameters to CPU for multi-threaded inference");
    let cpu_dev = candle_core::Device::Cpu;
    move_varmap_to_cpu(&parameters)?;

    let cpu_vb = candle_nn::VarBuilder::from_varmap(&parameters, candle_core::DType::F32, &cpu_dev);
    let cpu_encoder = IndexedEmbeddingEncoder::new(
        IndexedEmbeddingEncoderArgs {
            n_features: n_features_full,
            n_topics,
            embedding_dim: h,
            layers: &args.encoder_layers,
            attn_pool: true,
            n_gene_modules: 0,
            lora_rank: 0,
        },
        &parameters,
        cpu_vb.pp(crate::topic::gene_axis::ENCODER_PREFIX),
    )?;

    info!("Writing down the latent states");
    // Residual/batch correction tensor at full D (encoder operates at D_full).
    let delta = match args.adj_method {
        AdjMethod::Batch => finest_collapsed.delta.as_ref(),
        AdjMethod::Residual => finest_collapsed.mu_residual.as_ref(),
    }
    .map(|x| {
        x.posterior_mean()
            .to_tensor(&cpu_dev)
            .expect("delta to tensor")
            .transpose(0, 1)
            .expect("transpose")
            .contiguous()
            .expect("contiguous")
    });
    let eval_config = EvaluateLatentMaskedConfig {
        dev: &cpu_dev,
        adj_method: &args.adj_method,
        minibatch_size: args.minibatch_size.unwrap_or(100),
        feature_mean: &feature_mean,
        head,
    };
    let z_nk = evaluate_latent_masked(&data_vec, &cpu_encoder, &eval_config, delta.as_ref(), None)?;

    // Per-cell topic proportions. Every θ consumer below reads this rather
    // than re-deriving it, so the head's latent semantics are applied once.
    let theta_nk = crate::topic::model_metadata::latent_to_theta(&z_nk, head);

    // Sharpness on the cells vs. on the pseudobulk rows the encoder was trained on.
    // Same encoder, same weights; only the input distribution differs. In an A/B
    // the encoder came out near one-hot on the rows it TRAINED on and softer on
    // cells, and under `--poisson-thin` the pattern flipped with the training
    // distribution. So a one-hot latent is a property of the trained model, not
    // of feeding it single-cell counts — this line is what showed that, and it
    // stays so the next hypothesis is tested the same way.
    {
        use crate::topic::eval_indexed::evaluate_latent_masked_rows;
        let (eff_cells, max_cells) = latent_sharpness(&theta_nk);
        // The null those rows trained under: the batch δ mixed by shares
        // under `--adj-method batch`, the pseudobulk's own fold otherwise.
        let null_pd = match batch_nulls.as_ref().and_then(|n| n.last()) {
            Some(null) => Some(null.transpose()),
            None => finest_collapsed
                .mu_residual
                .as_ref()
                .map(|r| r.posterior_mean().transpose()),
        };
        let z_pk = evaluate_latent_masked_rows(
            finest_collapsed.mu_observed.posterior_mean(),
            null_pd.as_ref(),
            &cpu_encoder,
            &eval_config,
        )?;
        let theta_pk = crate::topic::model_metadata::latent_to_theta(&z_pk, head);
        let (eff_pb, max_pb) = latent_sharpness(&theta_pk);
        info!(
            "Latent sharpness — cells: {eff_cells:.2} effective topics, mean max θ {max_cells:.3}; \
             pseudobulk rows: {eff_pb:.2} effective topics, mean max θ {max_pb:.3} (K = {})",
            theta_nk.ncols()
        );
        if (eff_pb - eff_cells).abs() > 0.5 * eff_pb.max(eff_cells) {
            info!(
                "the encoder's sharpness differs between the rows it trained on and single \
                 cells; it is one-hot on whichever distribution it was trained on and softer \
                 off it. A near one-hot latent on BOTH is a training-side property of the \
                 masked objective, not an input mismatch — see K and --mask-fraction."
            );
        }
    }

    metadata.populate_theta_mean_and_save(&theta_nk, &args.out)?;

    scores.to_parquet(&format!("{}.log_likelihood.parquet", args.out))?;

    let cell_names = data_vec.column_names()?;

    senna::output_helpers::save_latent(&args.out, &z_nk, &cell_names, output_keep_idx.as_deref())?;

    // PB aggregates + the empirical dictionary, exactly as `senna topic` writes them.
    //
    // `topic` gets θ_PB by running the encoder's forward pass on the collapsed pseudobulk. The
    // indexed encoder's PB-level pass is not wired, and it does not need to be: θ is already known
    // per cell, and the finest collapsing already says which pseudobulk each cell belongs to — so
    // θ_PB is just the within-PB mean of it. That is the same quantity, without a second encoder
    // pass on data the encoder never saw in that shape.
    //
    // The empirical dictionary is the **model-free** counterpart to `dictionary.parquet`. The
    // latter is factorized — β = log_softmax(ρ·αᵀ) — so it can only say what the H-dimensional
    // embedding can express. This one is the plain NB-Fisher-weighted average of each topic's
    // pseudobulk expression, and depends on the factorization only through θ.
    //
    // Neither dominates, and this is not the "sharper" of the two — the softmax over ρ·αᵀ makes the
    // factorized β the more peaked one. It is an *independent* estimate, so having both lets a
    // caller cross-check a topic against something the model did not itself construct. It is also
    // the same object `senna topic` writes, so consumers that already prefer it (`lupin plot-topic` does
    // `dictionary_empirical.or(dictionary)`) behave the same across every topic-family run instead
    // of silently falling back here.
    {
        let pb_gene_gp: Mat = finest_collapsed.mu_observed.posterior_mean().clone();
        senna::output_helpers::save_pb_gene(&args.out, &pb_gene_gp, &gene_names)?;

        if let Some(c2p) = cell_to_pb_per_level.as_ref().and_then(|l| l.last()) {
            let n_pb = pb_gene_gp.ncols();
            let k = theta_nk.ncols();

            // θ_PB[p, ·] = mean over the cells collapsed into pseudobulk p.
            // `theta_nk` is already on the simplex for every head.
            let mut pb_latent_pk = Mat::zeros(n_pb, k);
            let mut n_in_pb = vec![0f32; n_pb];
            for (n, &p) in c2p.iter().enumerate() {
                if p < n_pb && n < theta_nk.nrows() {
                    n_in_pb[p] += 1.0;
                    for kk in 0..k {
                        pb_latent_pk[(p, kk)] += theta_nk[(n, kk)];
                    }
                }
            }
            for (p, &cnt) in n_in_pb.iter().enumerate() {
                if cnt > 0.0 {
                    for kk in 0..k {
                        pb_latent_pk[(p, kk)] /= cnt;
                    }
                }
            }

            let pb_names = axis_id_names("PB_", n_pb);
            let topic_names = axis_id_names("T", k);
            pb_latent_pk.to_parquet_with_names(
                &format!("{}.pb_latent.parquet", args.out),
                (Some(&pb_names), Some("pb")),
                Some(&topic_names),
            )?;

            // NB-Fisher weights. These used to double as the encoder's top-K
            // shortlist score; with no window that role is gone, but the
            // REPORTING weight is a different thing that happens to share a
            // formula, and dropping it would quietly flatten rare informative
            // genes in the empirical dictionary. Computed here, beside its only
            // remaining consumer, and no longer persisted.
            let fisher_weights: Vec<f32> = crate::refine_weighting::fit_fisher_weights(
                finest_collapsed,
                cell_to_pb_per_level
                    .as_deref()
                    .and_then(<[Vec<usize>]>::last)
                    .map(Vec::as_slice),
                None,
                &data_vec,
                args.block_size,
            )?;
            let beta_emp = crate::empirical_dict::build_empirical_dictionary(
                &pb_gene_gp,
                &pb_latent_pk,
                &fisher_weights,
            );
            beta_emp.to_parquet_with_names(
                &format!("{}.dictionary_empirical.parquet", args.out),
                (Some(&gene_names), Some("gene")),
                Some(&topic_names),
            )?;
            info!(
                "Wrote empirical dictionary {}×{} (NB-Fisher-weighted, column-simplex) + \
                 pb_latent {}×{}",
                beta_emp.nrows(),
                beta_emp.ncols(),
                n_pb,
                k,
            );
        } else {
            warn!(
                "no cell→pseudobulk map, so no θ_PB and no empirical dictionary — \
                 {}.dictionary_empirical.parquet is not written and downstream falls back to the \
                 factorized dictionary.parquet",
                args.out
            );
        }
    }

    let column_weight = data_vec.column_multiplicities();

    crate::postprocess::viz_prep::write_cell_proj(
        &args.out,
        &proj_kn,
        &cell_names,
        output_keep_idx.as_deref(),
    )?;
    let has_cell_to_pb = if let Some(ref c2p) = cell_to_pb_per_level {
        crate::postprocess::viz_prep::write_cell_to_pb(
            &args.out,
            c2p,
            &cell_names,
            output_keep_idx.as_deref(),
        )?;
        true
    } else {
        false
    };
    let has_pb_tree = if let Some(ref tree) = pb_tree {
        crate::postprocess::viz_prep::write_pb_tree(&args.out, tree, &gene_names)?;
        true
    } else {
        false
    };

    let pb_reference_suffix = senna::pb_reference::emit_if_requested(
        args.collapse.emits_pb_reference(),
        &args.out,
        finest_collapsed,
        cell_to_pb_per_level.as_deref(),
        column_weight,
        &gene_names,
        args.init_from.as_deref(),
        args.pb_reference.as_ref(),
    )?;

    let input: Vec<String> = data_files
        .iter()
        .map(std::string::ToString::to_string)
        .collect();
    let batch: Vec<String> = batch_files
        .as_ref()
        .map(|v| v.iter().map(std::string::ToString::to_string).collect())
        .unwrap_or_default();
    senna::run_manifest::write_run_manifest(&senna::run_manifest::RunDescription {
        train_args: Some(senna::run_manifest::record_train_args(args)?),
        kind: masked_run_kind(head),
        prefix: &args.out,
        data_input: &input,
        data_multiome: None,
        data_batch: &batch,
        data_input_null: &[],
        dictionary_suffix: Some("dictionary.parquet"),
        has_model: true,
        has_cell_proj: true,
        pb_gene_suffix: Some("pb_gene.parquet"),
        pb_reference_suffix,
        pb_latent_suffix: Some("pb_latent.parquet"),
        // Full-gene-resolution β̂, the same object `senna topic` writes. Preferred over the
        // factorized `dictionary.parquet` by anything that ranks genes (plot-topic already does
        // `dictionary_empirical.or(dictionary)`), because the factorization flattens rare genes.
        dictionary_empirical_suffix: Some("dictionary_empirical.parquet"),
        feature_embedding_suffix: Some("feature_embedding.parquet"),
        feature_coembedding_suffix: None,
        carried: carried.as_ref(),
        module_membership_suffix: None,
        module_dictionary_suffix: None,
        softmax_dictionary_suffix: Some("dictionary.parquet"),
        cell_embedding_suffix: None,
        cell_encoder_suffix: None,
        default_colour_by: "cluster",
        has_latent: true,
        has_cell_to_pb,
        has_pb_tree,
    })?;

    info!("Done");
    Ok(())
}

#[cfg(test)]
#[path = "masked_topic_tests.rs"]
mod masked_topic_tests;
