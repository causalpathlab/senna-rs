//! `senna predict` command-line arguments.

use crate::masked_topic::FeatureNameKindArg;
use crate::topic::eval::QueryNameOpts;
use senna::embed_common::*;

use super::delta;
use candle_core::Device;
use senna::sparse_output::check_target;

#[derive(Args, Clone, Debug)]
#[command(group(
    clap::ArgGroup::new("input")
        .required(true)
        .multiple(false)
        .args(["data_files", "bulk"])
))]
pub struct PredictArgs {
    #[arg(
        value_delimiter = ',',
        help = "Held-out data files (.zarr or .h5); or --bulk for a dense table",
        long_help = "Sparse backends to score with the pre-trained model.\n\
                     Gene sets may differ from training. Missing genes are padded.\n\
                     With -b, the encoder's null is built as training built it.\n\
                     Give these, or --bulk, not both."
    )]
    pub(crate) data_files: Vec<Box<str>>,

    #[arg(
        long,
        value_name = "FILE",
        num_args = 1..,
        help = "Dense bulk count matrices (parquet, or tab/comma text), genes × samples",
        long_help = "Dense bulk count matrices instead of sparse backends.\n\
                     Parquet, or tab/comma text; column 0 holds the row names\n\
                     and a header line, when present, names the samples.\n\
                     A samples × genes table is turned on read: the axis whose\n\
                     names match the model's genes is the gene axis\n\
                     (see --bulk-orientation).\n\
                     \n\
                     Each table becomes a temp sparse backend and is then scored\n\
                     exactly as a data file would be, by every model family.\n\
                     Several tables behave as several data files: each is its\n\
                     own batch, and sample names get an @<file> suffix.\n\
                     --batch-files is one label per sample.\n\
                     \n\
                     Values must be counts. Negative values are refused;\n\
                     non-integer values are accepted with a warning.\n\
                     --null-from still takes a backend: the null is the\n\
                     composition of the TRAINING half, which the model was fitted on."
    )]
    pub(crate) bulk: Vec<Box<str>>,

    #[command(flatten)]
    pub(crate) bulk_table: BulkTableArgs,

    #[arg(
        long,
        required = true,
        value_parser = parse_model_prefix,
        help = "Trained model prefix (`-o` of the training run), or its `.senna.json` manifest",
        long_help = "The training run's `-o` prefix, or its `{model}.senna.json` /\n\
                     `{model}.model.json` file, which reads as that prefix. Loads:\n  \
                     {model}.dictionary.parquet      gene × topic dictionary\n  \
                     {model}.model.json              model architecture metadata\n  \
                     {model}.safetensors             encoder + decoder weights\n  \
                     {model}.coarsening.json         (dense only) feature coarsening"
    )]
    pub(crate) model: Box<str>,

    #[arg(
        short,
        long,
        required = true,
        help = "Output file prefix",
        long_help = "Writes:\n  \
                     {out}.latent.parquet      [N × K] per-cell latent — log θ for the\n                               \
                     topic family, raw Gaussian z for vae /\n                               \
                     Gaussian-head masked models\n  \
                     {out}.predictive.parquet  per-cell [llik, total, llik_per_count]\n\
                     \n\
                     vae inference is encoder-only, so --decoder-only and\n\
                     --refine-* do not apply to it. It is still scored:\n\
                     its decoder is rebuilt to grade the same columns\n\
                     every other family is graded on."
    )]
    pub(crate) out: Box<str>,

    #[arg(
        short,
        long,
        value_delimiter = ',',
        help = "Batch membership files, one per data file"
    )]
    pub(crate) batch_files: Option<Vec<Box<str>>>,

    #[arg(
        long,
        default_value_t = 500,
        help = "Evaluation minibatch size",
        long_help = "Cells per scored block.\n\
                     \n\
                     IGNORED for a model whose run manifest records its training\n\
                     block size: it is scored at that size, because every encoder's\n\
                     input is standardised over the block a cell is scored in (and\n\
                     the vae's carries a batch norm), so another size would move the\n\
                     latent. The substitution is logged."
    )]
    pub(crate) minibatch_size: usize,

    #[arg(
        long,
        help = "Cells per block of the scoring delta pass (auto by default)",
        hide = true
    )]
    pub(crate) block_size: Option<usize>,

    #[arg(
        long,
        value_enum,
        default_value_t = delta::ScoringDeltaMode::Auto,
        value_name = "MODE",
        help = "Where the per-batch δ the held-out score is taken under comes from",
        long_help = "The topic families score held-out counts against δ_{g,b}·Σ_k θ_k·exp(β_gk),\n\
                     renormalised per cell. This picks δ.\n\
                     \n\
                     auto      the trained δ ({model}.delta.parquet) for a held-out batch named\n\
                     \x20         like a training batch; a cross-fitted δ for any other batch\n\
                     \x20         when there are two or more held-out batches; no δ at all\n\
                     \x20         for a model trained without batches\n\
                     train     the trained δ only; every held-out batch must be a training batch\n\
                     crossfit  each batch's cells are dealt to --scoring-folds folds, and a\n\
                     \x20         fold is scored under the δ fitted on the batch's other folds\n\
                     none      no δ: the batch-free rate θβ (and a bge run projects the\n\
                     \x20         counts as they are, not divided by its trained fold)\n\
                     heldout   δ fitted on the very cells it scores. In-sample (genes × batches\n\
                     \x20         free parameters), so the score is optimistic; kept for comparison\n\
                     \n\
                     No count is scored under a δ it helped fit, except with `heldout`.\n\
                     The null gets the same batch term: δ·null under a trained δ (with\n\
                     --null-from), and the gene composition of the cells δ was fitted on\n\
                     otherwise. δ reads every mapped gene, the ones --ablate-features hides\n\
                     included.\n\
                     \n\
                     A fitted δ is a batch's whole observed-to-predicted ratio per gene, so\n\
                     it also takes up what the dictionary misses in that batch; the trained\n\
                     δ is a contrast between batches. Compare scores taken under one mode.\n\
                     --residual-include-delta divides by one δ per batch: the trained one,\n\
                     or the batch's own fit."
    )]
    pub(crate) scoring_delta: delta::ScoringDeltaMode,

    #[arg(
        long,
        default_value_t = delta::DEFAULT_SCORING_FOLDS,
        value_parser = clap::builder::RangedU64ValueParser::<usize>::new().range(2..),
        value_name = "K",
        help = "Folds per batch of the cross-fitted scoring δ (at least 2)"
    )]
    pub(crate) scoring_folds: usize,

    #[arg(
        long,
        help = "Load all columns into memory before evaluation",
        hide = true
    )]
    pub(crate) preload_data: bool,

    #[arg(
        long,
        default_value_t = ComputeDevice::Cpu,
        value_enum,
        help = "Compute device",
        long_help = "Compute device. `cuda` / `metal` require the matching cargo feature.\n\
                     \n\
                     Matters most for a bge or simba model. Every other family infers with\n\
                     one encoder forward pass, but these have none — prediction re-runs\n\
                     the same per-cell Poisson SGD the training did, so it belongs on\n\
                     the device that trained it."
    )]
    pub(crate) device: ComputeDevice,

    #[arg(long, default_value_t = 0, help = "Device ordinal (for cuda/metal)")]
    pub(crate) device_no: usize,

    #[arg(
        long,
        default_value_t = 0,
        help = "Decoder-side gradient steps on θ at inference (0 = encoder forward only)",
        long_help = "If --decoder-only is set,\n\
                     this controls iterations of uniform-init optimization. Otherwise,\n\
                     controls per-cell refinement steps anchored to the encoder output."
    )]
    pub(crate) refine_steps: usize,

    #[arg(
        long,
        default_value_t = 0.01,
        help = "Learning rate for refinement / decoder-only"
    )]
    pub(crate) refine_lr: f64,

    #[arg(
        long,
        default_value_t = 1.0,
        help = "L2 anchor strength for refinement"
    )]
    pub(crate) refine_reg: f64,

    #[arg(
        long,
        help = "Skip the encoder;\n\
                init θ uniform and optimize purely against the frozen decoder",
        long_help = "Useful when the held-out feature set is too divergent.\n\
                     The trained encoder cannot handle it.\n\
                     Uses --refine-steps and --refine-lr.\n\
                     Those default to 100 / 0.05 if --refine-steps was left at 0."
    )]
    pub(crate) decoder_only: bool,

    #[arg(
        long,
        default_value_t = 0.0,
        value_name = "FRACTION",
        help = "Refuse to score below this share of the model's genes (0 = no gate)",
        long_help = "Gene coverage is always reported. This turns it into a hard floor.\n\
                     \n\
                     Off by default, because low coverage is not by itself wrong:\n\
                     a targeted panel legitimately measures a small slice of a\n\
                     whole-transcriptome model, and a narrow panel that spans the\n\
                     latent directions can still identify it. Zero mapped genes is\n\
                     always refused — that is a naming failure, not thin coverage."
    )]
    pub(crate) min_gene_overlap: f32,

    #[arg(
        long,
        value_parser = check_target,
        help = "Also write residual expression to a sparse backend ({out}.residual.zarr.zip / .h5)",
        long_help = "Regress the reference reconstruction out of the held-out counts.\n\
                     The reference is μ ∝ δ?·Σ_k θ_k·exp(β_dk). Regression is by DIVISION.\n\
                     The leftover is written as a NEW sparse backend, gene × cell.\n\
                     \n\
                     It reuses the shared `adjust_by_division_inplace`. Per cell,\n\
                     x_d /= μ_d·λ. λ = Σ_d x / Σ_d μ is the self-normalizing column scale.\n\
                     So the residual is a per-cell relative fold-change.\n\
                     `senna svd` uses the same division semantics for batches.\n\
                     \n\
                     Only entries above --residual-threshold are kept. All are ≥ 0,\n\
                     so the file stays sparse. The backend follows the extension: .zarr.zip,\n\
                     or .h5 with the `hdf5` feature."
    )]
    pub(crate) residual_out: Option<Box<str>>,

    #[arg(
        long,
        help = "Fold per-batch δ into μ (removes topics AND batch effect)",
        long_help = "When set, the per-gene denominator is δ_{d,b}·Σ_k θ_k·exp(β_dk).\n\
                     The residual is then harmonized, with the batch effect divided out.\n\
                     When unset, μ comes from topics only.\n\
                     The residual then still carries batch effects."
    )]
    pub(crate) residual_include_delta: bool,

    #[arg(
        long,
        default_value_t = 0.0,
        help = "Drop residual entries ≤ this value (default 0 = keep all nonzeros)"
    )]
    pub(crate) residual_threshold: f32,

    /// Restrict the agreement metrics to these features (one name per line).
    ///
    /// Without it every gene the model knows is scored, which makes two models
    /// trained on different gene axes incomparable — each is graded on its own
    /// curriculum. Pass the same file to every arm of a benchmark and the
    /// correlations become comparable numbers. It is also what makes the
    /// per-gene table affordable: that table has to hold every scored gene's
    /// values across all cells, so it is written only when this is given.
    #[arg(
        long,
        value_name = "FILE",
        help = "Score the agreement metrics on these features only (one name per line)"
    )]
    pub(crate) eval_features: Option<Box<str>>,

    /// Hide these features from the encoder, then score on exactly them.
    ///
    /// Without it the test cell's latent is fitted from that same cell's counts,
    /// so the score measures reconstruction with K free parameters per cell —
    /// and a model with a larger latent wins on capacity rather than on the
    /// quality of its dictionary. Hiding a slice of the features breaks that:
    /// the scored genes never enter the fit, so extra latent dimensions stop
    /// buying accuracy and the number becomes a genuine prediction. Implies
    /// `--eval-features` on the same file unless that is given separately.
    #[arg(
        long,
        value_name = "FILE",
        help = "Hide these features from the encoder and score on exactly them (one name per line)"
    )]
    pub(crate) ablate_features: Option<Box<str>>,

    /// bge only. Genes in the query that the model never saw are placed through the
    /// learned modules (their membership is the similarity-weighted mean of the k
    /// most similar matched genes' memberships, their bias is moment-matched to the
    /// pass-1 latents) instead of being dropped. `--no-init-genes` restores the drop.
    #[arg(
        long,
        help = "bge: drop genes the model never saw instead of initializing them through the modules"
    )]
    pub(crate) no_init_genes: bool,

    #[arg(
        long,
        default_value_t = graph_embedding_util::transfer::DEFAULT_INIT_NEIGHBOURS,
        value_name = "K",
        help = "bge: matched genes whose memberships are averaged to initialize an unseen gene"
    )]
    pub(crate) init_neighbours: usize,

    #[arg(
        long,
        default_value_t = graph_embedding_util::transfer::DEFAULT_SIMILARITY_FLOOR,
        value_name = "S",
        help = "bge: below this best profile similarity an unseen gene takes the diffuse prior"
    )]
    pub(crate) init_similarity_floor: f32,

    #[arg(
        long,
        help = "bge: re-project every cell with the initialized genes as observations (pass 2)",
        long_help = "bge: after initialization, run a second projection in which the initialized\n\
                     genes' counts are observed. Off, they are scored from the pass-1 latent and\n\
                     never move it. The comparable score still normalizes over the model's genes."
    )]
    pub(crate) init_genes_in_fit: bool,

    #[arg(
        long,
        help = "bge: write {out}.gene_rates.parquet, per-cell predicted rates of the missing and initialized genes"
    )]
    pub(crate) emit_gene_rates: bool,

    /// Training data, read once to build the null every arm is scored against.
    ///
    /// The floor is the count-weighted gene composition of this data — what a
    /// predictor with no cell-specific information would say. It has to come from
    /// the TRAINING half: a null built from the test half knows that half's exact
    /// marginal and is an oracle, not a floor. Without this flag the test half is
    /// used and a warning says so, because a silently different floor makes two
    /// runs quietly incomparable.
    #[arg(
        long,
        value_name = "FILE",
        num_args = 1..,
        help = "Training data whose gene composition is the null (pass the train half)"
    )]
    pub(crate) null_from: Option<Vec<Box<str>>>,

    #[arg(
        long,
        value_enum,
        help = "Row-name rule for the query: auto|exact|gene|locus|locus-overlap|mixed",
        long_help = "Mirrors the training-side flag, and drives both halves of the query side:\n\
                     how the query files' rows are aligned when they are read,\n\
                     and how each row name is matched to the model's gene axis.\n\
                     \n\
                     Unset: rows are aligned by auto-detection,\n\
                     then matched to the model exactly, with a flexible fallback.\n\
                     `exact` keeps the rows exactly as the file spells them\n\
                     (the choice for a model trained with `--feature-name-kind exact`).\n\
                     `gene` resolves `ENSG..._GENE1` to `GENE1`, rsplit on '_',\n\
                     so a symbol-keyed dictionary matches a query keyed by `<ensembl>_<symbol>`.\n\
                     `auto` aligns the rows by auto-detection and matches under `gene`.\n\
                     Applied AFTER the suffix trim (see --feature-name-suffix-delim)."
    )]
    pub(crate) feature_name_kind: Option<FeatureNameKindArg>,

    #[arg(
        long,
        help = "Split query row names on this char; keep prefix as base key",
        long_help = "Split query row names on this character.\n\
                     With '/', `ENSG00000000010_GENE1/count/spliced` splits in two.\n\
                     The base is `ENSG00000000010_GENE1`.\n\
                     The suffix is `count/spliced`.\n\
                     The suffix is then available to --keep-feature-suffix.\n\
                     The base is handed to --feature-name-kind."
    )]
    pub(crate) feature_name_suffix_delim: Option<char>,

    #[arg(
        long,
        help = "Keep only rows whose suffix equals this value",
        long_help = "e.g. `count/spliced` drops the `count/unspliced` rows of a `faba count` backend,\n\
                     collapsing the {spliced,unspliced} doubling to one row per gene.\n\
                     Requires --feature-name-suffix-delim.\n\
                     Rows lacking the delimiter are dropped when this is set."
    )]
    pub(crate) keep_feature_suffix: Option<Box<str>>,
}

impl PredictArgs {
    /// How this run gets its scoring δ.
    pub(super) fn scoring_spec(&self) -> delta::ScoringSpec {
        delta::ScoringSpec {
            mode: self.scoring_delta,
            folds: self.scoring_folds,
        }
    }

    /// The compute device this run asks for, resolved once.
    pub(super) fn resolve_device(&self) -> anyhow::Result<Device> {
        self.device.to_device(self.device_no)
    }

    /// Resolve every query-axis rule, including `--ablate-features`.
    ///
    /// Fallible now because the ablation list is read here rather than at the
    /// bottom of the score path — which is the point: a mistyped path fails
    /// immediately instead of after the whole backend has been loaded.
    pub(super) fn query_name_opts(&self) -> anyhow::Result<QueryNameOpts> {
        let hide = match self.ablate_features.as_deref() {
            // The SAME reader `--eval-features` uses. These two flags name the
            // same file by contract, so parsing them differently would hide a
            // different gene set than the one being scored.
            Some(path) => Some(std::sync::Arc::new(
                legume_numeric::matrix::common_io::read_name_list(path)
                    .map_err(|e| anyhow::anyhow!("reading --ablate-features {path}: {e}"))?
                    .into_iter()
                    .collect::<std::collections::HashSet<Box<str>>>(),
            )),
            None => None,
        };
        // Both halves of the flag in one expression, so the pairing cannot
        // drift: unset is the legacy behaviour (loader auto-detects, the remap
        // matches exactly then falls back to the flexible matcher).
        let (loader_kind, kind) = match &self.feature_name_kind {
            None => (None, data_beans::aux::feature_names::FeatureNameKind::Exact),
            Some(arg) => (arg.clone().into(), arg.resolve_or_gene()),
        };
        Ok(QueryNameOpts {
            loader_kind,
            kind,
            suffix_delim: self.feature_name_suffix_delim,
            keep_suffix: self.keep_feature_suffix.clone(),
            min_overlap: self.min_gene_overlap,
            hide,
        })
    }
}

/// `--model` names the training run's `-o` prefix. Its manifest
/// (`{prefix}.senna.json`) or metadata (`{prefix}.model.json`) is the file a
/// user tab-completes to, so either reads as the prefix it sits under.
pub(crate) fn parse_model_prefix(s: &str) -> Result<Box<str>, String> {
    let prefix = s
        .strip_suffix(".senna.json")
        .or_else(|| s.strip_suffix(".model.json"))
        .unwrap_or(s);
    if prefix.is_empty() {
        return Err(format!("{s}: no model prefix before the suffix"));
    }
    Ok(prefix.into())
}
