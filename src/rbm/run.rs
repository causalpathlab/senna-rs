//! `senna rbm`: load (multiome detected as `senna bge` detects it), collapse
//! into pseudobulk levels, fit the machine on them, read out every cell.

use super::args::RbmArgs;
use super::blocks::Blocks;
use super::eval::{score_cells, score_rows};
use super::hidden::Hidden;
use super::train::{fit, Level, TrainConfig};
use super::weights::{Given, Weights};
use crate::topic::common::{
    create_device, load_and_collapse, setup_stop_handler, LoadCollapseArgs,
};
use graph_embedding_util as ge;
use legume_numeric::candle::candle_core::{Device, Tensor};
use legume_numeric::matrix::parquet::{write_named_table, Column};
use senna::embed_common::*;

/// Feature modules unless told otherwise: with free rows, units drift out of
/// use over a long fit.
const DEFAULT_FEATURE_MODULES: usize = 50;

/// What `{out}.model.json` records of the machine.
#[derive(serde::Serialize, serde::Deserialize)]
pub struct RbmModel {
    pub hidden: super::hidden::HiddenArg,
    pub trials: u32,
    pub max_count: u32,
    pub n_latent: usize,
    pub embedding_dim: usize,
    pub tie_symbols: bool,
    pub blocks: Vec<Box<str>>,
    pub block_sizes: Vec<usize>,
    pub depth: Vec<f32>,
}

pub fn fit_rbm(args: &RbmArgs) -> anyhow::Result<()> {
    mkdir_parent(&args.out)?;
    anyhow::ensure!(
        args.n_latent > 0 && args.chains != Some(0) && args.minibatch_size > 0,
        "--n-latent, --chains and --minibatch-size must be positive"
    );

    let inherited = args
        .from
        .as_deref()
        .map(senna::run_manifest::inherit_from)
        .transpose()?;
    senna::run_manifest::InheritedFromManifest::ensure_replayable(inherited.as_ref(), "rbm")?;
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

    // Multiome: detected off the inputs, or declared, as `senna bge` does.
    let plan: Option<ge::MultiomePlan> = if args.multiome.is_empty() {
        crate::bge::multiome::auto_plan(
            &data_files,
            batch_files.as_ref().map_or(0, Vec::len),
            false,
        )?
    } else {
        crate::bge::multiome::declared_plan(&args.multiome)?
    };
    if let Some(p) = plan.as_ref() {
        crate::bge::multiome::log_plan(p, !args.multiome.is_empty());
    }
    let is_multiome = plan.is_some();
    let data_files: Vec<Box<str>> = plan.as_ref().map_or(data_files, |p| p.files.clone());
    let hvg = crate::hvg::resolve_multiome_with_hvg(is_multiome, data_files.len(), &args.hvg);
    let feature_kind = if is_multiome {
        Some(ge::FeatureNameKind::Mixed)
    } else {
        args.feature_name_kind.clone().into()
    };

    let crate::topic::common::PreparedData {
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
        proj_dim: args.collapse.proj_dim.max(args.n_latent),
        sort_dim: args.collapse.sort_dim,
        knn_cells: args.collapse.knn_cells,
        num_levels: args.collapse.num_levels,
        iter_opt: args.collapse.iter_opt,
        block_size: Some(args.block_size),
        out: &args.out,
        max_features: hvg.n_hvg,
        feature_list_file: hvg.feature_list_file,
        must_train_file: hvg.must_train_file,
        refine: args.collapse.pb_refine.to_params(),
        pb_tree: args.collapse.pb_tree_params(),
        ignore_batch: args.collapse.ignore_batch,
        qc: args.qc.to_config(),
        qc_block_size: Some(args.block_size),
        qc_report_out: args.qc.qc_report.as_deref(),
        feature_mask_fn: None,
        pb_reference: None,
        mixture_batches: args.collapse.mixture_batch.clone(),
        observe_panels: !is_multiome,
        row_alignment: data_beans::sparse_io_vector::RowAlignment::default(),
        column_alignment: if is_multiome {
            data_beans::sparse_io_vector::ColumnAlignment::Union
        } else {
            data_beans::sparse_io_vector::ColumnAlignment::Disjoint
        },
        feature_kind: feature_kind.clone(),
        want_hierarchy: true,
        prebuilt_partition,
        cnv_clones: args.collapse.cnv_clones.as_deref(),
        per_file_feature_suffix: plan.as_ref().map(|p| p.modality.clone()),
        per_file_barcode_suffix: plan.as_ref().and_then(ge::MultiomePlan::barcode_suffix),
    })?;

    let row_names = data_vec.row_names()?;
    let blocks = Blocks::from_names(&row_names);
    anyhow::ensure!(
        !is_multiome || blocks.len() > 1,
        "the inputs load as a multiome layout, but their features do not split into \
         modality blocks: every row must be named `{{name}}/{{modality}}` with exactly one '/'. \
         A feature name of the input already holds a '/'; rename it, or fit each modality \
         on its own."
    );
    let feature_names = blocks.feature_names(&row_names);
    info!(
        "rbm: {} feature(s) in {} block(s) [{}]",
        blocks.features(),
        blocks.len(),
        blocks
            .names
            .iter()
            .zip(&blocks.ranges)
            .map(|(n, r)| format!("{n}:{}", r.len()))
            .collect::<Vec<_>>()
            .join(", ")
    );

    // The rows of ρ: one per feature, or one per name with tied symbols.
    let (rho_row, rho_names): (Vec<u32>, Vec<Box<str>>) = if args.tie_symbols {
        (blocks.symbol.clone(), blocks.symbols.clone())
    } else {
        (
            (0..blocks.features() as u32).collect(),
            feature_names.clone(),
        )
    };
    let preset_kind = if is_multiome && !args.tie_symbols {
        ge::FeatureNameKind::Mixed
    } else {
        args.feature_name_kind.resolve_or_gene()
    };
    let (preset, _carried) = crate::feature_preset::resolve_preset(
        args.feature_embedding.resolve()?,
        &rho_names,
        &preset_kind,
    )?;
    let h = match preset.as_ref() {
        Some(p) => {
            anyhow::ensure!(
                !p.ids.is_empty(),
                "the given feature embedding matches no feature of this run"
            );
            let h = p.rows.len() / p.ids.len();
            p.mode.validate(h)?;
            if h != args.embedding_dim {
                info!(
                    "rbm: H = {h}, the given table's width (not --embedding-dim {})",
                    args.embedding_dim
                );
            }
            h
        }
        None => args.embedding_dim,
    };
    anyhow::ensure!(h > 0, "--embedding-dim must be positive");
    let lora = preset
        .as_ref()
        .and_then(|p| p.mode.lora())
        .unwrap_or_default();

    let levels: Vec<Level> = collapsed_levels
        .iter()
        .map(|c| {
            let rates = c
                .mu_adjusted
                .as_ref()
                .unwrap_or(&c.mu_observed)
                .posterior_mean();
            Level::from_rates(rates, &blocks)
        })
        .collect();
    let depth: Vec<f32> = match args.depth.as_slice() {
        [d] => vec![*d; blocks.len()],
        d => {
            anyhow::ensure!(
                d.len() == blocks.len(),
                "--depth gives {} value(s) for {} block(s)",
                d.len(),
                blocks.len()
            );
            d.to_vec()
        }
    };
    anyhow::ensure!(
        depth.iter().all(|&d| d >= 1.0),
        "--depth must be at least 1"
    );
    info!("rbm: training depth per block {depth:?}");

    let dev = create_device(&args.device, args.device_no)?;
    let weights = Weights::new(
        &rho_row,
        args.n_latent,
        h,
        blocks.len(),
        preset.as_ref().map(|p| Given {
            ids: &p.ids,
            rows: &p.rows,
            mode: p.mode,
        }),
        args.seed,
        &dev,
    )?;
    // Feature modules, as `senna bge` builds them: a partition of the rows of
    // `ρ` from the finest pseudobulks' counts (per modality when untied).
    let modules = if args.no_feature_modules {
        None
    } else if preset.is_some() && args.modules.feature_modules.is_none() {
        info!("rbm: a given feature embedding has per-feature rows; feature modules are off");
        None
    } else {
        args.modules.resolve(Some(DEFAULT_FEATURE_MODULES))?
    };
    let weights = match modules.as_ref() {
        None => weights,
        Some(mc) => {
            let finest = collapsed_levels.last().expect("a level");
            let c2p = cell_to_pb_per_level
                .as_deref()
                .and_then(<[Vec<usize>]>::last)
                .ok_or_else(|| {
                    anyhow::anyhow!("feature modules need the finest cell → pseudobulk membership")
                })?;
            let (counts, sizes) = finest.observed_counts(c2p)?;
            let n_rows = rho_names.len();
            let mut by_row = Mat::zeros(n_rows, counts.ncols());
            for (i, &r) in rho_row.iter().enumerate() {
                let mut dst = by_row.row_mut(r as usize);
                dst += counts.row(blocks.perm[i]);
            }
            let (labels, n_modules) = if blocks.len() > 1 && !args.tie_symbols {
                let mut group = vec![0u32; n_rows];
                for (b, rg) in blocks.ranges.iter().enumerate() {
                    rg.clone()
                        .for_each(|i| group[rho_row[i] as usize] = b as u32);
                }
                let part = graph_embedding_util::fit::module_partition::partition_modules_by_group(
                    &by_row,
                    &sizes,
                    &group,
                    &vec![mc.n_modules; blocks.len()],
                    &vec![None; blocks.len()],
                    None,
                    args.seed,
                )?;
                (part.labels, part.n_modules)
            } else {
                (
                    graph_embedding_util::partition_modules(
                        &by_row,
                        &sizes,
                        mc.n_modules,
                        args.seed,
                    )?,
                    mc.n_modules,
                )
            };
            info!("rbm: {n_modules} feature module(s) over {n_rows} rows of ρ");
            weights.with_modules(labels, n_modules, mc.init_own_mass, args.seed)?
        }
    };
    let hidden = Hidden::new(args.hidden, args.trials, args.max_count);
    let stop = setup_stop_handler();
    let cfg = TrainConfig {
        epochs: args.epochs,
        batch: args.minibatch_size,
        chains: args.chains.unwrap_or(args.minibatch_size),
        sweeps: args.gibbs_sweeps,
        lr: f64::from(args.learning_rate),
        lora_lr_ratio: f64::from(lora.lr_ratio),
        lora_ridge: f64::from(lora.ridge),
        embedding_l2: f64::from(args.embedding_l2),
        draw_data: args.draw_data,
        sparsity_target: f64::from(args.sparsity_target),
        sparsity_cost: f64::from(args.sparsity_cost),
        module_balance: modules
            .as_ref()
            .map_or(0.0, |m| f64::from(m.lambda_balance)),
        module_ridge: modules.as_ref().map_or(0.0, |m| f64::from(m.residual_l2)),
        module_warmup: modules
            .as_ref()
            .map_or(0, |m| m.warmup_epochs_for(args.epochs)),
        holdout: f64::from(args.holdout),
        seed: args.seed,
        stop: &stop,
    };
    let fitted = fit(&levels, &blocks, weights, hidden, depth, &cfg, &dev)?;
    let machine = fitted.machine(&blocks)?.detach();
    let k = args.n_latent;

    // Cells, batch-adjusted by the finest level's δ.
    let finest = collapsed_levels.last().expect("a level");
    let delta = finest
        .delta
        .as_ref()
        .map(|d| -> anyhow::Result<Tensor> {
            let perm = Tensor::from_iter(blocks.perm.iter().map(|&i| i as u32), &dev)?;
            Ok(d.posterior_mean()
                .to_tensor(&dev)?
                .contiguous()?
                .index_select(&perm, 0)?
                .t()?
                .contiguous()?)
        })
        .transpose()?;
    let cells = score_cells(
        &data_vec,
        &machine,
        &blocks,
        delta.as_ref(),
        args.block_size,
    )?;
    let cell_names = data_vec.column_names()?;
    let keep = output_keep_idx.as_deref();
    senna::output_helpers::save_latent(
        &args.out,
        &cells.columns(0, k).into_owned(),
        &cell_names,
        keep,
    )?;
    let scores = cells.columns(k, 2).into_owned();
    let (scores, score_cells) = senna::output_helpers::cell_subset(&scores, &cell_names, keep)
        .unwrap_or_else(|| (scores, cell_names.clone()));
    let fe: Vec<f32> = scores.column(0).iter().copied().collect();
    let recon: Vec<f32> = scores.column(1).iter().copied().collect();
    write_named_table(
        &format!("{}.cell_score.parquet", args.out),
        "cell",
        &score_cells,
        &[
            ("free_energy".into(), Column::F32(&fe)),
            ("reconstruction".into(), Column::F32(&recon)),
        ],
    )?;

    // Pseudobulks of the finest level, at the training depth.
    let finest_level = levels.last().expect("a level");
    let pbs = score_rows(
        &machine,
        &finest_level.expected(&blocks, &fitted.depth),
        args.block_size,
    )?;
    let pb_names = axis_id_names("PB_", pbs.nrows());
    pbs.columns(0, k).into_owned().to_parquet_with_names(
        &format!("{}.pb_latent.parquet", args.out),
        (Some(&pb_names), Some("pb")),
        Some(&axis_id_names("T", k)),
    )?;

    // Tables.
    let w = Mat::from_tensor(&machine.w.to_device(&Device::Cpu)?)?;
    senna::output_helpers::save_dictionary(&args.out, &w, &feature_names)?;
    let rho = fitted.weights.rho()?;
    crate::topic::train_masked::write_feature_embedding(&rho, &rho_names, &args.out)?;
    let has_modules = if let Some(m) = fitted.weights.modules.as_ref() {
        use graph_embedding_util::transfer::{
            MODULE_DICTIONARY_SUFFIX, MODULE_MEMBERSHIP_SUFFIX, MODULE_RESIDUAL_SUFFIX,
        };
        let pi = Mat::from_tensor(&m.membership()?.to_device(&Device::Cpu)?)?;
        let module_names: Vec<Box<str>> = (0..pi.ncols()).map(|i| format!("M{i}").into()).collect();
        pi.to_parquet_with_names(
            &format!("{}.{MODULE_MEMBERSHIP_SUFFIX}", args.out),
            (Some(&rho_names), Some("feature")),
            Some(&module_names),
        )?;
        Mat::from_tensor(&m.mu.as_tensor().to_device(&Device::Cpu)?)?.to_parquet_with_names(
            &format!("{}.{MODULE_DICTIONARY_SUFFIX}", args.out),
            (Some(&module_names), Some("module")),
            Some(&axis_id_names("H", h)),
        )?;
        Mat::from_tensor(&m.residual.as_tensor().to_device(&Device::Cpu)?)?.to_parquet_with_names(
            &format!("{}.{MODULE_RESIDUAL_SUFFIX}", args.out),
            (Some(&rho_names), Some("feature")),
            Some(&axis_id_names("H", h)),
        )?;
        true
    } else {
        false
    };
    let alpha = Tensor::cat(
        &fitted
            .weights
            .alpha
            .iter()
            .map(|a| a.as_tensor().clone())
            .collect::<Vec<_>>(),
        0,
    )?;
    let alpha_names: Vec<Box<str>> = blocks
        .names
        .iter()
        .flat_map(|b| {
            (0..k).map(move |t| {
                if b.is_empty() {
                    format!("T{t}").into()
                } else {
                    format!("{b}/T{t}").into()
                }
            })
        })
        .collect();
    Mat::from_tensor(&alpha.to_device(&Device::Cpu)?)?.to_parquet_with_names(
        &format!("{}.hidden_embedding.parquet", args.out),
        (Some(&alpha_names), Some("unit")),
        Some(&axis_id_names("H", h)),
    )?;
    let b: Vec<f32> = fitted.b.as_tensor().flatten_all()?.to_vec1()?;
    write_named_table(
        &format!("{}.feature_bias.parquet", args.out),
        "feature",
        &feature_names,
        &[("bias".into(), Column::F32(&b))],
    )?;
    let epochs: Vec<Box<str>> = (1..=fitted.trace.gap.len())
        .map(|e| e.to_string().into())
        .collect();
    write_named_table(
        &format!("{}.log_likelihood.parquet", args.out),
        "epoch",
        &epochs,
        &[
            ("free_energy_gap".into(), Column::F32(&fitted.trace.gap)),
            (
                "reconstruction".into(),
                Column::F32(&fitted.trace.reconstruction),
            ),
            ("heldout_gap".into(), Column::F32(&fitted.trace.heldout)),
            ("live_units".into(), Column::F32(&fitted.trace.live_units)),
        ],
    )?;

    // The machine.
    let mut tensors = std::collections::HashMap::new();
    tensors.insert("rho".to_string(), rho.to_device(&Device::Cpu)?);
    // The centres: the latent and the free energy read the reads and the
    // hidden units against them, so the machine is incomplete without them.
    tensors.insert("mu".to_string(), fitted.mu.to_device(&Device::Cpu)?);
    tensors.insert("lambda".to_string(), fitted.lambda.to_device(&Device::Cpu)?);
    tensors.insert("alpha".to_string(), alpha.to_device(&Device::Cpu)?);
    tensors.insert(
        "b".to_string(),
        fitted.b.as_tensor().to_device(&Device::Cpu)?,
    );
    tensors.insert(
        "c".to_string(),
        fitted.c.as_tensor().to_device(&Device::Cpu)?,
    );
    legume_numeric::candle::candle_core::safetensors::save(
        &tensors,
        format!("{}.safetensors", args.out),
    )?;
    let model = RbmModel {
        hidden: args.hidden,
        trials: args.trials,
        max_count: args.max_count,
        n_latent: k,
        embedding_dim: h,
        tie_symbols: args.tie_symbols,
        blocks: blocks.names.clone(),
        block_sizes: blocks.ranges.iter().map(|r| r.len()).collect(),
        depth: fitted.depth.clone(),
    };
    std::fs::write(
        format!("{}.model.json", args.out),
        serde_json::to_string_pretty(&model)?,
    )?;

    // What `senna view` reads.
    crate::postprocess::viz_prep::write_cell_proj(&args.out, &proj_kn, &cell_names, keep)?;
    let has_cell_to_pb = if let Some(ref c2p) = cell_to_pb_per_level {
        crate::postprocess::viz_prep::write_cell_to_pb(&args.out, c2p, &cell_names, keep)?;
        true
    } else {
        false
    };
    let has_pb_tree = if let Some(ref tree) = pb_tree {
        crate::postprocess::viz_prep::write_pb_tree(&args.out, tree, &row_names)?;
        true
    } else {
        false
    };
    let input: Vec<String> = data_files.iter().map(ToString::to_string).collect();
    let batch: Vec<String> = batch_files
        .as_deref()
        .map(|v| v.iter().map(ToString::to_string).collect())
        .unwrap_or_default();
    senna::run_manifest::write_run_manifest(&senna::run_manifest::RunDescription {
        train_args: Some(senna::run_manifest::record_train_args(args)?),
        kind: senna::run_manifest::RunKind::Rbm,
        prefix: &args.out,
        data_input: &input,
        data_multiome: plan
            .as_ref()
            .map(senna::multiome_layout::RunMultiome::from_plan),
        data_batch: &batch,
        data_input_null: &[],
        dictionary_suffix: Some("dictionary.parquet"),
        has_model: false,
        has_cell_proj: true,
        pb_gene_suffix: None,
        pb_reference_suffix: None,
        pb_latent_suffix: Some("pb_latent.parquet"),
        dictionary_empirical_suffix: None,
        feature_embedding_suffix: Some("feature_embedding.parquet"),
        feature_coembedding_suffix: None,
        carried: None,
        module_membership_suffix: has_modules
            .then_some(graph_embedding_util::transfer::MODULE_MEMBERSHIP_SUFFIX),
        module_dictionary_suffix: has_modules
            .then_some(graph_embedding_util::transfer::MODULE_DICTIONARY_SUFFIX),
        softmax_dictionary_suffix: None,
        cell_embedding_suffix: None,
        cell_encoder_suffix: None,
        feature_contrast_suffix: None,
        feature_contrast_bias_suffix: None,
        track_encoder_suffixes: vec![],
        default_colour_by: "cluster",
        has_latent: true,
        has_cell_to_pb,
        has_pb_tree,
    })?;
    info!("Done");
    Ok(())
}

#[cfg(test)]
#[path = "tests/run.rs"]
mod tests;
