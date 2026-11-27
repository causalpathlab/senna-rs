//! Unified prediction subcommand for the dense and masked topic models.
//!
//! Loads a model trained by `senna topic` / `masked-topic`, applies it
//! to a held-out backend file, and writes:
//!   - `{out}.latent.parquet`     [N × K] log θ (per-cell topic proportions)
//!   - `{out}.predictive.parquet` [N × 3] per-cell `[llik, total, llik_per_count]`
//!
//! Three latent-inference modes (mutually exclusive):
//!   - `Encoder` (default): forward pass through the trained encoder only.
//!   - `EncoderRefine` (`--refine-steps > 0`): encoder warm-start, then
//!     decoder gradient on θ anchored to encoder by L2.
//!   - `DecoderOnly` (`--decoder-only`): skip encoder; init θ uniform
//!     `log(1/K)`; optimize purely against the frozen decoder. Useful when
//!     the test feature set is too divergent for the encoder.

use crate::topic::eval::{build_gene_remap_with, GeneRemap, QueryNameOpts};
use crate::topic::model_metadata::TopicModelMetadata;
use crate::topic::predict_common::LatentMode;
use senna::embed_common::*;

use candle_core::Device;
use data_beans::alg::feature_coarsening::FeatureCoarsening;
use data_beans::aux::data_loading::{read_data_on_shared_rows, ReadSharedRowsArgs};
use data_beans::sparse_io_vector::SparseIoVec;
use dense::predict_dense;
use log::info;
use masked::predict_masked;
use projection::{predict_bge, predict_svd};
use vae::predict_vae;

pub fn predict_model(args: &PredictArgs) -> anyhow::Result<()> {
    mkdir_parent(&args.out)?;

    // Resolve the family BEFORE reaching for `{model}.model.json`. A `senna bge` run never
    // writes one (`has_model: false`), so opening it first turned "predict on a bge run"
    // into a bare file-not-found naming a path the user never typed — the same confusion
    // `BgeEmbedding::open` exists to prevent, reintroduced one level up.
    let kind = crate::topic::model_metadata::resolve_run_kind(&args.model)?;
    // Before any data is touched: the latent refit only exists for the dense
    // topic models, and anywhere else the flags would be dropped silently.
    // (`--refine-lr` / `--refine-reg` only tune that refit.)
    use senna::run_manifest::RunKind;
    // `itopic` is the manifest's name for both masked simplex heads, not a
    // command anyone types.
    let family = match kind {
        RunKind::Itopic => "masked-topic / masked-sbp",
        k => k.as_str(),
    };
    // No scoring path: say so by name, before anything is read, not as a
    // field missing from a model.json the run never wrote in that shape.
    let scored = match kind {
        RunKind::Topic
        | RunKind::Itopic
        | RunKind::MaskedVae
        | RunKind::Vae
        | RunKind::JointTopic
        | RunKind::Svd
        | RunKind::Bge
        | RunKind::Simba => true,
        RunKind::Rbm
        | RunKind::Ode
        | RunKind::JointSvd
        | RunKind::Fne
        | RunKind::ResolveEmbeddingSpace => false,
    };
    anyhow::ensure!(
        scored,
        "predict does not score a {family} run ({}): it predicts topic, masked, vae, svd, bge \
         and simba runs",
        args.model
    );
    anyhow::ensure!(
        kind.has_decoder_latent_fit() || (!args.decoder_only && args.refine_steps == 0),
        "--decoder-only / --refine-steps re-fit the latent against a topic decoder; {} is a \
         {family} run, which is predicted from its encoder or projection only",
        args.model
    );

    // `--bulk`: dense tables become temp backends here, and every family path
    // below reads them as it would any data file. The guard removes them when
    // this function returns, so it has to live to the end of it.
    let mut args = args.clone();
    let _bulk_guard = if args.bulk.is_empty() {
        None
    } else {
        let model_genes = bulk::model_gene_axis(kind, &args.model)?;
        let b = bulk::materialize(&args.bulk, &model_genes, &args.bulk_table.opts())?;
        b.warn_if_split(args.minibatch_size);
        args.data_files = b.paths().to_vec();
        Some(b)
    };
    let args = &args;

    // Pre-flight the gene lists against the MODEL's axis, before any backend
    // is imported: a list on a foreign axis used to surface only after the
    // whole query had been loaded, as "matched no feature". This is the same
    // resolution the scoring pass runs later, brought forward; the query-side
    // ablation check in `hide_features` still runs on the query's own axis.
    let listed_paths: Vec<(&str, &str)> = [
        ("--eval-features", args.eval_features.as_deref()),
        ("--ablate-features", args.ablate_features.as_deref()),
    ]
    .into_iter()
    .filter_map(|(flag, path)| path.map(|p| (flag, p)))
    .collect();
    if !listed_paths.is_empty() {
        // Loaded once: `model_gene_names` decodes the whole dictionary parquet.
        let axis = bulk::model_gene_names(kind, &args.model)?;
        for (flag, path) in listed_paths {
            crate::topic::predict_eval::resolve_eval_genes(Some(path), &axis, flag)?;
        }
    }

    // Without -b each data file is a batch named by the file, and a name
    // that matches a training batch takes that batch's trained δ. Right for a
    // training file predicted back; wrong for an unrelated file of the same
    // name, which nothing here can tell apart.
    if args.batch_files.is_none()
        && args.scoring_delta != delta::ScoringDeltaMode::None
        && delta::has_trained_delta(&args.model)
    {
        log::warn!(
            "no --batch-files: each data file is its own batch, named by the file. A file \
             named like a training batch is adjusted by that batch's trained δ; pass -b to \
             say which batch the cells belong to"
        );
    }

    match kind {
        // bge and simba: a frozen gene table, projected onto per cell (see
        // `BgeEmbedding`).
        k if k.has_frozen_gene_table() => return predict_bge(args, kind),
        // svd writes no checkpoint either; its query side is the Nyström
        // projection onto the frozen dictionary.
        RunKind::Svd => return predict_svd(args),
        _ => {}
    }

    let metadata = TopicModelMetadata::load(&args.model)?;
    info!(
        "Loaded model metadata: type={}, K={}, D_full={}, D_enc={}",
        metadata.model_type,
        metadata.n_topics,
        metadata.n_features_full,
        metadata.n_features_encoder,
    );

    use crate::topic::model_metadata::{
        masked_head_from_model_type, MODEL_TYPE_INDEXED_MASKED, MODEL_TYPE_MASKED_SBP,
        MODEL_TYPE_MASKED_VAE, MODEL_TYPE_TOPIC, MODEL_TYPE_VAE,
    };
    // All masked heads (softmax / stick-breaking / Gaussian) share the
    // encoder-only path; `masked_head_from_model_type` recovers which one.
    if let Some(head) = masked_head_from_model_type(&metadata.model_type) {
        return predict_masked(args, &metadata, head);
    }
    match metadata.model_type.as_ref() {
        MODEL_TYPE_TOPIC => predict_dense(args, &metadata),
        MODEL_TYPE_VAE => predict_vae(args, &metadata),
        other => anyhow::bail!(
            "predict: unsupported model_type '{other}' (expected '{MODEL_TYPE_TOPIC}', \
             '{MODEL_TYPE_INDEXED_MASKED}', '{MODEL_TYPE_MASKED_SBP}', \
             '{MODEL_TYPE_MASKED_VAE}', or '{MODEL_TYPE_VAE}')",
        ),
    }
}

fn resolve_mode(args: &PredictArgs) -> LatentMode {
    if args.decoder_only {
        LatentMode::DecoderOnly
    } else if args.refine_steps > 0 {
        LatentMode::EncoderRefine
    } else {
        LatentMode::Encoder
    }
}

/// What predict takes from the held-out batches: the encoder's null and the
/// scoring δ, or only the δ for `--bulk`, whose whole samples are not cells
/// to collapse into pseudobulks.
fn held_out<'a>(args: &PredictArgs, fit: &'a null::RecordedFit) -> null::HeldOut<'a> {
    if args.bulk.is_empty() {
        null::HeldOut::NullAndDelta(fit, args.scoring_spec())
    } else {
        null::HeldOut::Delta(args.scoring_spec())
    }
}

mod args;
pub(crate) mod bulk;
pub(crate) mod delta;
mod dense;
mod finalize;
mod masked;
pub(crate) mod null;
mod projection;
mod vae;

pub(crate) use args::parse_model_prefix;
pub use args::PredictArgs;
pub(crate) use dense::{score_dense_backend, DenseScoreArgs, DenseScored};
pub(crate) use masked::{score_masked_backend, MaskedScoreArgs, MaskedScored};
pub(crate) use vae::{score_vae_backend, VaeScoreArgs, VaeScored};

#[cfg(test)]
mod tests;

/// Which flag named the scored gene set, and its path. `--ablate-features`
/// implies `--eval-features` on the same file unless that is given separately,
/// so the two are resolved in one place rather than re-`or`-ed at each site.
fn eval_gene_source(args: &PredictArgs) -> (Option<&str>, &'static str) {
    match args.eval_features.as_deref() {
        Some(p) => (Some(p), "--eval-features"),
        None => (args.ablate_features.as_deref(), "--ablate-features"),
    }
}

/// The model's saved `μ_d`, refused unless it has one entry per dictionary gene.
fn feature_mean_on(model: &str, training_genes: &[Box<str>]) -> anyhow::Result<Vec<f32>> {
    let (_, feature_mean) = crate::topic::model_metadata::load_feature_mean(model)?;
    anyhow::ensure!(
        feature_mean.len() == training_genes.len(),
        "feature_mean gene count ({}) != dictionary gene count ({})",
        feature_mean.len(),
        training_genes.len()
    );
    Ok(feature_mean)
}

/// How [`read_query`] treats the cells' batches.
#[derive(Clone, Copy)]
pub(crate) enum QueryBatches<'a> {
    /// Register each cell's batch: from `files`, or one batch per data file
    /// when `None`; made `batch|cvK` by the covariate model the run `model`
    /// recorded, when it has one, so trained batches find their δ.
    Register {
        files: Option<&'a [Box<str>]>,
        model: &'a str,
    },
    /// The caller never reads batches; skip the registration pass.
    Ignore,
}

/// Count data read onto the model's gene axis the way every family reads a
/// query: shared rows under the caller's name rules, batch membership
/// registered as `batches` says.
pub(crate) fn read_query(
    data_files: &[Box<str>],
    batches: QueryBatches<'_>,
    preload: bool,
    opts: &QueryNameOpts,
    training_genes: &[Box<str>],
) -> anyhow::Result<SparseIoVec> {
    let batch_files = match batches {
        QueryBatches::Register { files, .. } => files,
        QueryBatches::Ignore => None,
    };
    let loaded = read_data_on_shared_rows(senna::multiome_layout::query_load(
        ReadSharedRowsArgs {
            data_files: data_files.to_vec(),
            batch_files: batch_files.map(<[_]>::to_vec),
            preload,
            feature_kind: opts.loader_kind.clone(),
            ..Default::default()
        },
        training_genes,
    )?)?;
    let mut data_vec = loaded.data;
    if let QueryBatches::Register { model, .. } = batches {
        let batch = crate::covariate::query_batches(model, &data_vec, loaded.batch)?;
        data_vec.register_batch_membership(&batch);
    }
    info!(
        "Query data: {} features × {} cells",
        data_vec.num_rows(),
        data_vec.num_columns()
    );
    Ok(data_vec)
}

/// Align query genes onto the model's axis, or `None` when the axes already
/// match. Coverage is logged and gated by
/// [`crate::topic::eval::ensure_gene_coverage`].
fn build_remap(
    training_genes: &[Box<str>],
    new_genes: &[Box<str>],
    opts: &QueryNameOpts,
) -> anyhow::Result<Option<GeneRemap>> {
    let mut gene_remap = build_gene_remap_with(training_genes, new_genes, opts);
    crate::topic::eval::ensure_gene_coverage(&gene_remap, opts.min_overlap, "--feature-name-kind")?;
    // After the coverage gate, never before: hidden features are withheld on
    // purpose, so counting them as missing would refuse every ablated run.
    if let Some(hide) = opts.hide.as_deref() {
        crate::topic::eval::hide_features(&mut gene_remap, new_genes, hide)?;
        return Ok(Some(gene_remap));
    }

    let needs_remap = gene_remap
        .new_to_train
        .iter()
        .enumerate()
        .any(|(i, opt)| *opt != Some(i))
        || new_genes.len() != training_genes.len();

    if needs_remap {
        info!(
            "Gene remapping enabled ({} → {} features)",
            new_genes.len(),
            training_genes.len()
        );
        Ok(Some(gene_remap))
    } else {
        info!("Genes match training — no remapping needed");
        Ok(None)
    }
}

/// What an embedding family (vae, bge, simba) hands the scoring δ.
struct EmbeddingScore<'a> {
    args: &'a PredictArgs,
    /// The run's prefix, where its trained δ is.
    model: &'a str,
    /// The model's feature axis, which `rho_dh` and `b_feat` live on.
    training_genes: &'a [Box<str>],
    data_vec: &'a SparseIoVec,
    rho_dh: &'a Mat,
    b_feat: &'a [f32],
    theta_nh: &'a Mat,
}

/// The scoring δ of an embedding family, under the same `--scoring-delta`
/// rule as the topic families: its rate `exp(b + ρ·θ)`, normalised over
/// genes, takes δ as theirs does.
fn embedding_scoring_delta(e: EmbeddingScore<'_>) -> anyhow::Result<Option<delta::ScoringDelta>> {
    let score_remap = scoring_remap(
        e.training_genes,
        &e.data_vec.row_names()?,
        &e.args.query_name_opts()?,
    )?;
    delta::scoring_delta(&delta::ScoringDeltaArgs {
        spec: e.args.scoring_spec(),
        model: e.model,
        training_genes: e.training_genes,
        data_vec: e.data_vec,
        theta_nk: e.theta_nh,
        gene_remap: score_remap.as_ref(),
        rate: delta::RateModel::Embedding {
            rho_dh: e.rho_dh,
            b_feat: e.b_feat,
        },
        phi: None,
        block_size: e.args.block_size,
    })
}

/// The remap the observed counts are read through to score them: the query's
/// rows on the model's axis WITHOUT the ablation. The encoder's remap has the
/// hidden genes pointing at `None`, which is right for its input and wrong
/// for anything that reads the scored counts (the scoring δ, the agreement
/// pass), since the scored genes are the hidden ones.
fn scoring_remap(
    training_genes: &[Box<str>],
    new_genes: &[Box<str>],
    opts: &QueryNameOpts,
) -> anyhow::Result<Option<GeneRemap>> {
    let unhidden = QueryNameOpts {
        hide: None,
        ..opts.clone()
    };
    build_remap(training_genes, new_genes, &unhidden)
}

/// Aggregate a `[D_full]` per-gene mean to `[D_coarse]` so the divisor
/// is on the same per-fine-gene rate scale as the encoder expects.
///
/// The encoder's input `y_coarse` and batch null `x0_coarse` are both
/// **sum**-coarsened (`aggregate_columns_nd`), giving `y ≈ G·batch·<μ>·bio`
/// and `x0 ≈ G·batch`. For `clean = y / (x0·μ_coarse)` to recover `bio`,
/// `μ_coarse` must be at the per-fine-gene rate `<μ>`, not the summed
/// `Σμ`. So we sum-coarsen first (matching the call sites used for
/// data and batch null) then divide each coarse cell by its group size.
pub(crate) fn aggregate_feature_mean_to_coarse(
    full: &[f32],
    coarsening: Option<&FeatureCoarsening>,
) -> Vec<f32> {
    let n_full = full.len();
    let mu_1d = nalgebra::DMatrix::<f32>::from_row_slice(1, n_full, full);
    match coarsening {
        Some(fc) => {
            let mu_summed = fc.aggregate_columns_nd(&mu_1d);
            mu_summed
                .row(0)
                .iter()
                .zip(fc.coarse_to_fine.iter())
                .map(|(&s, fines)| s / fines.len().max(1) as f32)
                .collect()
        }
        None => mu_1d.row(0).iter().copied().collect(),
    }
}

//////////////////////
// Dense prediction //
//////////////////////

/// Dense `[rows × width]` f32 tensors the NB likelihood chain holds at peak —
/// counted off `legume_numeric::candle::loss::nb_log_likelihood_elem` (phi, mu, phi+mu,
/// three logs, two terms, x+phi, the lgammas) plus the decoder's logits,
/// softmax and mu. Rounded up: under-counting is an OOM, over-counting is
/// only slower.
pub(crate) const NB_CHAIN_TENSORS: usize = 16;

/// Bytes of dense f32 one block holds, as a shape rather than a number the
/// caller multiplied out — the callers and the tests all go through this, so
/// there is one formula instead of three that can drift apart. Mirrors
/// `data_beans::sparse_io::helpers::preload_within_budget`, which likewise
/// takes the raw shape and computes the cost itself.
pub(crate) fn dense_bytes(rows: usize, width: usize, tensors: usize) -> usize {
    rows.saturating_mul(width)
        .saturating_mul(std::mem::size_of::<f32>())
        .saturating_mul(tensors)
}

/// How many dense blocks may be in flight at once, given the working set one
/// block costs.
///
/// The dense drivers densify each block to `[minibatch × D]` and then run a
/// decoder over it; the NB likelihood chain alone allocates a dozen-odd
/// tensors of that shape. Left to `par_iter`, EVERY block runs at once, so the
/// peak is `threads × per-block` — which at whole-transcriptome D is tens of
/// gigabytes and was an OOM kill, silently, with only a SIGKILL to show for
/// it. The work per block is unchanged; only how many run together is capped.
///
/// `LEGUME_PREDICT_BUDGET_BYTES` overrides the default, following the
/// `LEGUME_PRELOAD_BUDGET_BYTES` precedent for memory knobs.
pub(crate) fn dense_block_concurrency(dev: &Device, bytes_per_block: usize) -> usize {
    let budget = std::env::var("LEGUME_PREDICT_BUDGET_BYTES")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(DEFAULT_PREDICT_BUDGET_BYTES);
    // Two ceilings, composed: the device's (one block off the CPU — see
    // `device_concurrency`) and the memory budget's. `block_concurrency`
    // clamps to the first, so off the CPU the budget cannot raise it.
    let threads =
        crate::topic::common::device_concurrency(dev.is_cpu(), rayon::current_num_threads());
    block_concurrency(bytes_per_block, budget, threads)
}

/// The memory budget the cap works from when `LEGUME_PREDICT_BUDGET_BYTES` is unset.
pub(crate) const DEFAULT_PREDICT_BUDGET_BYTES: usize = 8 << 30;

/// The cap as a pure function of the three things that set it. Kept apart from
/// the env/rayon lookup so a test can state the machine it means instead of
/// inheriting the one it happens to run on: the same block that runs wide
/// open on a 32-thread box is rightly held at ~34 on a 64-thread one.
pub(crate) fn block_concurrency(bytes_per_block: usize, budget: usize, threads: usize) -> usize {
    (budget / bytes_per_block.max(1)).clamp(1, threads.max(1))
}

/// Run `block_fn` over `[0, ntot)` in `minibatch_size` blocks, concatenating
/// results into `(z_nk [ntot, kk], llik [ntot], total [ntot])`. Shared by the
/// dense predict drivers.
fn run_predict_blocks<F>(
    ntot: usize,
    kk: usize,
    minibatch_size: usize,
    // Peak dense bytes ONE block holds, for the concurrency cap.
    bytes_per_block: usize,
    // Where the blocks run: off the CPU they run one at a time, on the
    // calling thread (see `blocks_in_flight`).
    dev: &Device,
    block_fn: F,
) -> anyhow::Result<(Mat, Vec<f32>, Vec<f32>)>
where
    F: Fn((usize, usize)) -> anyhow::Result<(usize, Mat, Vec<f32>, Vec<f32>)> + Send + Sync,
{
    let jobs = create_jobs(ntot, 0, Some(minibatch_size));
    let max_conc = dense_block_concurrency(dev, bytes_per_block);
    if dev.is_cpu() && max_conc < rayon::current_num_threads() {
        info!(
            "Scoring {} blocks at most {max_conc} at a time: one block's ~{} MB dense \
             working set would otherwise be multiplied by every thread \
             (LEGUME_PREDICT_BUDGET_BYTES to raise)",
            jobs.len(),
            bytes_per_block >> 20
        );
    }
    let mut chunks: Vec<(usize, Mat, Vec<f32>, Vec<f32>)> =
        crate::topic::common::map_blocks(&jobs, max_conc, block_fn)?;
    chunks.sort_by_key(|c| c.0);

    let mut z_nk = Mat::zeros(ntot, kk);
    let mut llik = Vec::with_capacity(ntot);
    let mut total = Vec::with_capacity(ntot);
    let mut row = 0;
    for (_, z_block, lk, tot) in chunks {
        let n = z_block.nrows();
        z_nk.rows_range_mut(row..row + n).copy_from(&z_block);
        llik.extend(lk);
        total.extend(tot);
        row += n;
    }
    Ok((z_nk, llik, total))
}

////////////////////
// Output writers //
////////////////////
