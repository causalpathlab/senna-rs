use crate::topic::common::{load_and_collapse, setup_stop_handler, LoadCollapseArgs, PreparedData};
use senna::embed_common::*;

use legume_numeric::candle::decoder::*;
use legume_numeric::candle::encoder::*;

mod args;
mod pipeline;

use pipeline::{run_multi_decoder_pipeline, run_topic_pipeline, FeatureStats, PipelineCtx};

pub(crate) use args::DecoderType;
pub use args::TopicArgs;

pub fn fit_topic_model(args: &TopicArgs) -> anyhow::Result<()> {
    mkdir_parent(&args.out)?;

    let inherited = args
        .from
        .as_deref()
        .map(senna::run_manifest::inherit_from)
        .transpose()?;
    if let Some(inh) = inherited.as_ref() {
        info!(
            "--from: inheriting data + batch{} from a '{}' manifest",
            if inh.cell_to_pb_path.is_some() {
                " + cell→pb partition"
            } else {
                ""
            },
            inh.source_kind
        );
    }
    senna::run_manifest::InheritedFromManifest::ensure_replayable(inherited.as_ref(), "topic")?;
    let data_files = senna::run_manifest::InheritedFromManifest::resolve_data(
        inherited.as_ref(),
        &args.data_files,
    )?;
    let batch_files = senna::run_manifest::InheritedFromManifest::resolve_batch(
        inherited.as_ref(),
        args.batch_files.as_deref(),
    );
    let prebuilt_partition = args.pb_from.partition(
        inherited.as_ref(),
        &crate::refine_weighting::partition_settings(args),
    )?;

    let PreparedData {
        data_vec,
        collapsed_levels,
        proj_kn,
        cell_to_pb_per_level,
        pb_tree,
        output_keep_idx,
        batch_files,
    } = load_and_collapse(&LoadCollapseArgs {
        covariates: &args.collapse.covariates,
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
        max_features: args.hvg.n_hvg,
        feature_list_file: args.hvg.feature_list_file.as_deref(),
        must_train_file: args.hvg.must_train_features.as_deref(),
        refine: args.collapse.pb_refine.to_params(),
        ignore_batch: args.collapse.ignore_batch,
        qc: args.qc.to_config(),
        qc_block_size: args.block_size,
        qc_report_out: args.qc.qc_report.as_deref(),
        feature_mask_fn: None,
        pb_reference: args.pb_reference.as_ref(),
        mixture_batches: args.collapse.mixture_batch.clone(),
        observe_panels: true,
        row_alignment: data_beans::sparse_io_vector::RowAlignment::default(),
        column_alignment: data_beans::sparse_io_vector::ColumnAlignment::default(),
        feature_kind: args.feature_name_kind.clone().into(),
        want_hierarchy: true,
        prebuilt_partition,
        cnv_clones: args.collapse.cnv_clones.as_deref(),
        multiome: None,
        reuse: args.pb_from.pb_from.as_deref(),
    })?;

    let finest_collapsed: &CollapsedOut = collapsed_levels.last().unwrap();

    // 4. Per-level feature coarsenings for decoders.
    //    Dense encoder operates at D_coarse (finest level's coarsening).
    //    Decoders at coarser levels use fewer feature groups.
    let n_features_full = data_vec.num_rows();
    let num_levels = collapsed_levels.len();

    // Where this run's gene axis differs from an `--init-from` source run's,
    // that run's gene-keyed state is continued by NAME: its modules are grown
    // onto this axis here, and its checkpoint is gathered onto it at the warm
    // start. What growth does move for this family is the per-module mean the
    // encoder divides by: a module that absorbed unseen genes has more members
    // than the one the checkpoint was fitted against. It is a divisive null
    // before the variance-stabilizing transform, so the effect is a level
    // shift rather than a change of shape.
    let gene_names = data_vec.row_names()?;
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

    // Finest-level coarsening (used for encoder, evaluation, dictionary output)
    let finest_coarsening: Option<&FeatureCoarsening> =
        level_coarsenings.last().and_then(|c| c.as_ref());

    // 5. Train a topic model on the collapsed data
    let n_topics = args.n_latent_topics;

    // Encoder at finest level's D_coarse (dense ops are O(D))
    let n_features_encoder = finest_coarsening.map_or(n_features_full, |c| c.num_coarse);

    // Per-feature mean rate `μ_d` from the finest-level pseudobulk
    // posterior. We save it at `D_full` (gene-name aligned) and
    // aggregate through the coarsening matrix here for the encoder's
    // own `D_coarse` view. Same correction the indexed encoder gets
    // via gather — gives the dense encoder a stable per-gene null
    // beside the per-cell batch null.
    let feature_mean_full: Vec<f32> = crate::topic::common::pseudobulk_feature_mean(
        finest_collapsed.mu_observed.posterior_mean(),
    );
    // `μ_d` at coarse level = **mean** per fine gene within each
    // group. The data and batch null are sum-coarsened
    // (`aggregate_columns_nd` on counts/ratios), so `y_coarse ≈
    // G·batch·<μ>·bio` and `x0_coarse ≈ G·batch`. For the divisive
    // correction `clean = y / (x0 · μ_coarse)` to recover `bio`
    // cleanly, `μ_coarse` must be on a per-fine-gene rate scale —
    // i.e. `<μ>` (mean), not `Σμ` (sum). The asymmetry comes from
    // `x0` being a ratio (~1) summed to ≈G while `μ` is a count
    // rate; mean-coarsening `μ` makes the units match.
    let feature_mean: Vec<f32> = {
        let mu_1d =
            nalgebra::DMatrix::<f32>::from_row_slice(1, n_features_full, &feature_mean_full);
        let mu_summed = match finest_coarsening {
            Some(fc) => fc.aggregate_columns_nd(&mu_1d),
            None => mu_1d,
        };
        match finest_coarsening {
            Some(fc) => mu_summed
                .row(0)
                .iter()
                .zip(fc.coarse_to_fine.iter())
                .map(|(&s, fines)| s / fines.len().max(1) as f32)
                .collect(),
            None => mu_summed.row(0).iter().copied().collect(),
        }
    };

    let dev = args.compute.to_device()?;

    let parameters = candle_nn::VarMap::new();
    let param_builder =
        candle_nn::VarBuilder::from_varmap(&parameters, candle_core::DType::F32, &dev);

    let level_decoder_dims: Vec<usize> = level_coarsenings
        .iter()
        .map(|fc| fc.as_ref().map_or(n_features_full, |c| c.num_coarse))
        .collect();

    info!(
        "input: {} -> encoder -> {:?} decoder(s) (dims {:?}) -> finest: {}",
        n_features_encoder, args.decoder, level_decoder_dims, n_features_encoder,
    );

    // Per-level NB-Fisher weights at the *coarse* feature resolution each
    // decoder operates on, so the dispersion trend matches the data the
    // decoder actually sees. Used by `MultinomTopicDecoder` to multiplicatively
    // weight the per-gene log-likelihood term.
    //
    // `collapsed_levels` and `level_coarsenings` are indexed alike (both
    // finest-last, and `build_level_data` zips them), so a level's own
    // pseudobulks are its trend's population.
    let feature_fisher_per_level: Vec<Vec<f32>> = {
        info!(
            "Computing per-level NB-Fisher weights at coarse resolution from {} ({} levels)",
            if data_vec.has_column_multiplicity() {
                "pseudobulks"
            } else {
                "cells"
            },
            level_coarsenings.len(),
        );
        // Every construction branch above sizes `level_coarsenings` to
        // `num_levels = collapsed_levels.len()`; debug-checked because a zip
        // would otherwise truncate without a word.
        debug_assert_eq!(collapsed_levels.len(), level_coarsenings.len());
        let membership_per_level = cell_to_pb_per_level.as_deref().unwrap_or(&[]);
        collapsed_levels
            .iter()
            .enumerate()
            .zip(level_coarsenings.iter())
            .map(|((level, collapsed), fc)| {
                crate::refine_weighting::fit_fisher_weights(
                    collapsed,
                    membership_per_level.get(level).map(Vec::as_slice),
                    fc.as_ref(),
                    &data_vec,
                    args.block_size,
                )
            })
            .collect::<anyhow::Result<Vec<_>>>()?
    };

    let stop = setup_stop_handler();

    let anchor_prior = crate::topic::anchor_prior::AnchorPrior::from_pseudobulk(
        finest_collapsed,
        n_topics,
        finest_coarsening,
    )?;

    // Per-level [K, D_l] anchor tensors on the training device. Built once
    // here, held alive for the entire fit via the outer scope.
    let anchor_tensors = anchor_prior.per_level_device_tensors(&level_coarsenings, &dev)?;

    // Under `--adj-method batch` the encoder trains on the δ it reads at
    // inference, mixed by each pseudobulk's batch shares.
    let batch_nulls = crate::topic::common::batch_level_nulls(
        &args.adj_method,
        &collapsed_levels,
        cell_to_pb_per_level.as_deref(),
        &data_vec,
    )?;

    let ctx = PipelineCtx {
        level_decoder_dims: &level_decoder_dims,
        n_topics,
        param_builder: &param_builder,
        collapsed_levels: &collapsed_levels,
        level_coarsenings: &level_coarsenings,
        finest_coarsening,
        finest_collapsed,
        cell_to_pb_finest: cell_to_pb_per_level
            .as_deref()
            .and_then(<[Vec<usize>]>::last)
            .map(Vec::as_slice),
        n_features_full,
        gene_names: &gene_names,
        gene_axis: gene_axis.as_ref(),
        data_vec: &data_vec,
        parameters: &parameters,
        dev: &dev,
        args,
        stop: &stop,
        anchor_prior: Some(&anchor_prior),
        anchor_prior_per_level: Some(&anchor_tensors),
        batch_nulls: batch_nulls.as_deref(),
        feature_stats: FeatureStats {
            mean: &feature_mean,
            mean_full: &feature_mean_full,
            fisher_per_level: &feature_fisher_per_level,
        },
    };

    let mut encoder = LogSoftmaxEncoder::new(
        LogSoftmaxEncoderArgs {
            n_features: n_features_encoder,
            n_topics,
            layers: &args.encoder_layers,
            feature_mean: Some(&feature_mean),
        },
        &parameters,
        param_builder.clone(),
    )?;

    let (scores, z_nk) = if args.decoder.len() == 1 {
        match args.decoder[0] {
            DecoderType::Nb => run_topic_pipeline::<_, NbTopicDecoder>(&ctx, &mut encoder)?,
            DecoderType::Multinom => {
                run_topic_pipeline::<_, MultinomTopicDecoder>(&ctx, &mut encoder)?
            }
            DecoderType::NbMixture => {
                run_topic_pipeline::<_, NbMixtureTopicDecoder>(&ctx, &mut encoder)?
            }
        }
    } else {
        run_multi_decoder_pipeline(&ctx, &mut encoder)?
    };

    scores.to_parquet(&format!("{}.log_likelihood.parquet", args.out))?;

    let cell_names = data_vec.column_names()?;

    senna::output_helpers::save_latent(&args.out, &z_nk, &cell_names, output_keep_idx.as_deref())?;

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

    write_topic_manifest(
        &args.out,
        &data_files,
        batch_files.as_deref(),
        has_cell_to_pb,
        has_pb_tree,
        pb_reference_suffix,
        senna::run_manifest::record_train_args(args)?,
    )?;

    info!("Done");
    Ok(())
}

/// Assemble + save the `{prefix}.senna.json` manifest for a `senna topic` run.
/// The other families build their `RunDescription` inline; this one is split
/// out only because `fit_topic_model` is already long.
fn write_topic_manifest(
    prefix: &str,
    data_files: &[Box<str>],
    batch_files: Option<&[Box<str>]>,
    has_cell_to_pb: bool,
    has_pb_tree: bool,
    pb_reference_suffix: Option<&'static str>,
    train_args: senna::run_manifest::TrainArgsRecord,
) -> anyhow::Result<()> {
    let input: Vec<String> = data_files
        .iter()
        .map(std::string::ToString::to_string)
        .collect();
    let batch: Vec<String> = batch_files
        .map(|v| v.iter().map(std::string::ToString::to_string).collect())
        .unwrap_or_default();
    senna::run_manifest::write_run_manifest(&senna::run_manifest::RunDescription {
        train_args: Some(train_args),
        kind: senna::run_manifest::RunKind::Topic,
        prefix,
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
        dictionary_empirical_suffix: Some("dictionary_empirical.parquet"),
        feature_embedding_suffix: None,
        feature_coembedding_suffix: None,
        carried: None,
        module_membership_suffix: None,
        module_dictionary_suffix: None,
        softmax_dictionary_suffix: Some("dictionary.parquet"),
        cell_embedding_suffix: None,
        cell_encoder_suffix: None,
        default_colour_by: "cluster",
        has_latent: true,
        has_cell_to_pb,
        has_pb_tree,
    })
}
