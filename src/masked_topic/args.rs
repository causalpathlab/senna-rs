//! `senna masked-topic` / `masked-sbp` / `masked-vae` command-line arguments,
//! and how a manifest replays and updates them.

use super::FeatureNameKindArg;
use senna::embed_common::*;

/// Mask-rate schedule (CLI surface for `MaskSchedule`).
#[derive(
    clap::ValueEnum,
    Clone,
    Copy,
    Debug,
    Default,
    PartialEq,
    Eq,
    serde::Serialize,
    serde::Deserialize,
)]
#[serde(rename_all = "kebab-case")]
pub enum MaskScheduleArg {
    /// Constant mask fraction (`--mask-fraction`).
    #[default]
    Fixed,
    /// Sample the rate per minibatch in `[--mask-rate-lo, --mask-rate-hi]`.
    Uniform,
}

/// Per-gene likelihood for the masked imputation loss (CLI surface for
/// [`legume_numeric::candle::vae::masked_topic::MaskedLikelihood`]).
#[derive(
    clap::ValueEnum,
    Clone,
    Copy,
    Debug,
    Default,
    PartialEq,
    Eq,
    serde::Serialize,
    serde::Deserialize,
)]
#[serde(rename_all = "kebab-case")]
pub enum MaskedLikelihoodArg {
    /// Negative binomial — overdispersed counts (library-scaled, learnable φ).
    #[default]
    Nb,
    /// Multinomial / categorical — depth-invariant composition; the same
    /// likelihood the generative ELBO path uses.
    Multinomial,
}

impl MaskedLikelihoodArg {
    /// Map to the legume_numeric::candle training enum.
    pub fn to_lib(self) -> legume_numeric::candle::vae::masked_topic::MaskedLikelihood {
        use legume_numeric::candle::vae::masked_topic::MaskedLikelihood as L;
        match self {
            MaskedLikelihoodArg::Nb => L::Nb,
            MaskedLikelihoodArg::Multinomial => L::Multinomial,
        }
    }
}

#[derive(Args, Debug, serde::Serialize, serde::Deserialize)]
#[serde(default = "senna::embed_common::clap_defaults")]
pub struct MaskedTopicArgs {
    #[arg(
        value_delimiter = ',',
        help = "Input data files (.zarr or .h5; optional when --from is given)",
        long_help = "Sparse backends produced by `data-beans from-mtx`.\n\
                     Multiple files may be passed, comma- or space-separated.\n\
                     They are concatenated column-wise on a shared feature set.\n\
                     When `--from <run.senna.json>` is provided and this list is empty,\n\
                     the data paths come from the source manifest."
    )]
    pub(super) data_files: Vec<Box<str>>,

    #[arg(
        long,
        help = "Chain inputs + warm-start from a prior `senna bge / fne / topic` manifest",
        long_help = "Read a `{run}.senna.json` manifest and pre-fill `data_files`,\n\
                     `--batch-files`, and `--freeze-feature-embedding` from it.\n\
                     Explicit CLI flags override the manifest values.\n\
                     SVD-family manifests are rejected (no feature embedding to inherit).\n\
                     Typical use: bge → masked-topic warm-start.\n\
                     \n\
                     senna bge   data.zarr.zip -b batch.gz -o run-bge ...\n\
                     senna masked-topic --from run-bge.senna.json -o run-topic ..."
    )]
    pub(super) from: Option<Box<str>>,

    #[arg(
        long,
        short,
        required = true,
        help = "Output file prefix",
        long_help = "Prefix for generated files:\n  \
                     {out}.dictionary.parquet       gene × topic loadings (log-prob)\n  \
                     {out}.latent.parquet           cell × topic log-softmax proportions\n  \
                     {out}.delta.parquet            per-batch effects (if --batch-files)\n  \
                     {out}.log_likelihood.parquet   training loss trace\n  \
                     {out}.safetensors              encoder+decoder weights\n  \
                     {out}.cell_proj.parquet        cached random projection (for `senna layout`)\n  \
                     {out}.senna.json               run manifest for `senna layout/plot --from`"
    )]
    pub(super) out: Box<str>,

    #[arg(
        long,
        short,
        value_delimiter(','),
        help = "Batch membership files, one per data file",
        long_help = "Each file lists a batch label per cell.\n\
                     The cells come in the same order as its matching data file.\n\
                     Example: batch1.tsv,batch2.tsv"
    )]
    pub(super) batch_files: Option<Vec<Box<str>>>,

    #[command(flatten)]
    pub(super) collapse: crate::refine_weighting::CollapseArgs,

    #[command(flatten)]
    #[serde(flatten)]
    pub(super) pb_from: crate::refine_weighting::PbFromArgs,

    #[arg(
        long = "init-from",
        help = "Initialize encoder + decoder weights from a previously trained model",
        long_help = "Path prefix of a model saved by `senna masked-topic`.\n\
                     It matches {prefix}.model.json + {prefix}.safetensors.\n\
                     Architecture must match: same K, encoder layers, embedding_dim,\n\
                     and n_features_full.\n\
                     Cross-gene-set warm-start is not supported.\n\
                     Train on the same gene set."
    )]
    pub(super) init_from: Option<Box<str>>,

    #[arg(
        long,
        default_value_t = 0,
        help = "Add N topics on top of the warm-started model (needs --init-from)",
        long_help = "Grow K when continuing a trained run, so a cohort carrying biology the\n\
                     parent has no topic for can acquire one instead of distorting an\n\
                     existing topic.\n\
                     \n\
                     Added topics start switched off: their encoder rows get zero weights\n\
                     and a strongly negative bias, so they hold ~0 mass at step 0 and have\n\
                     to earn their way in. The parent's topics keep their indices, so\n\
                     annotations keyed to them stay valid.\n\
                     \n\
                     Off by default — K is part of every downstream artifact's identity."
    )]
    pub(super) add_topics: usize,

    #[arg(
        long,
        default_value_t = 0,
        help = "Add N dimensions to the gene embedding ρ (needs --init-from)",
        long_help = "Widen H when continuing a trained run, giving the dictionary room for\n\
                     structure the parent's embedding cannot represent.\n\
                     \n\
                     Exactly function-preserving at step 0: the new α columns are zero, so\n\
                     β = softmax(α·ρᵀ) is unchanged bit for bit, while the new ρ columns\n\
                     stay random so the added subspace still receives gradient."
    )]
    pub(super) add_embedding_dim: usize,

    /// The parent's carried pseudobulks, when `senna update` chose to reuse
    /// them instead of re-reading its cells. Derived per invocation, so it is
    /// neither a CLI flag nor part of the recorded configuration.
    #[arg(skip)]
    #[serde(skip)]
    pub(super) pb_reference: Option<senna::pb_reference::ReferenceInput>,

    #[arg(
        long,
        help = "Cells per rayon job (omit for auto-scaling by feature count)",
        hide = true
    )]
    pub(super) block_size: Option<usize>,

    #[arg(
        short = 't',
        long,
        default_value_t = 10,
        help = "Number of latent topics (K)"
    )]
    pub(super) n_latent_topics: usize,

    #[arg(
        long,
        short = 'e',
        value_delimiter(','),
        default_values_t = vec![128, 1024, 128],
        help = "Encoder hidden layer sizes (comma-separated)",
        long_help = "Example: 128,1024,128 (input → 128 → 1024 → 128 → topics)."
    )]
    pub(super) encoder_layers: Vec<usize>,

    #[arg(long, short = 'i', default_value_t = 1000, help = "Training epochs")]
    pub(super) epochs: usize,

    #[arg(
        long,
        help = "Training minibatch size (unset: 100, shrunk to fit GPU memory on CUDA)",
        long_help = "Cells per training minibatch.\n\
                     Unset, the default is 100 on CPU.\n\
                     On CUDA the size is chosen automatically:\n\
                     a short probe measures the memory one step retains,\n\
                     and shrinks the minibatch from 100\n\
                     when --gpu-mem-fraction of free device memory\n\
                     cannot hold it (it never grows past 100:\n\
                     minibatch size is not fit-neutral).\n\
                     Passing a value disables the probe and always wins."
    )]
    pub(super) minibatch_size: Option<usize>,

    #[arg(
        long,
        default_value_t = 0.6,
        help = "Fraction of free GPU memory the training minibatch may target",
        long_help = "Ceiling for the automatic minibatch sizing on CUDA.\n\
                     The probe grows the minibatch\n\
                     while one step's retained memory,\n\
                     with half reserved for the backward pass,\n\
                     fits this fraction of the device memory free at start.\n\
                     Fractions outside 0.05 to 0.95 are clamped to that range.\n\
                     Ignored on CPU and when --minibatch-size is set."
    )]
    pub(super) gpu_mem_fraction: f32,

    #[arg(
        long,
        alias = "lr",
        default_value_t = 0.01,
        help = "Adam learning rate"
    )]
    pub(super) learning_rate: f32,

    #[arg(
        long,
        default_value_t = 1.0,
        help = "Global L2 gradient norm clip per minibatch (0 = off; typical 0.5–5.0)"
    )]
    pub(super) grad_clip: f32,

    #[arg(
        long,
        default_value_t = 1.0,
        help = "L2 penalty λ on the feature embedding matrix ρ (default 1.0; 0 = off)",
        long_help = "L2 penalty λ on the feature embedding matrix ρ ∈ ℝ^{D×H}.\n\
                     It adds λ · mean(ρ²) to the per-minibatch loss.\n\
                     That mean-normalization keeps λ scale-invariant across D·H.\n\
                     It shrinks the β dynamic range, where β = log_softmax(α·ρᵀ).\n\
                     That can speed ETM convergence on high-D gene sets.\n\
                     The default of 1.0 is mild shrinkage; 0.0 disables it.\n\
                     Typical values run 0.1 to 10.0."
    )]
    pub(super) feature_embedding_l2: f32,

    #[command(flatten)]
    #[serde(flatten)]
    pub(super) feature_embedding: crate::feature_embedding_args::FeatureEmbeddingArgs,

    #[arg(
        long,
        default_value_t = 0.0,
        help = "AdamW decoupled weight decay for all parameters (default 0.0 = off)",
        long_help = "AdamW decoupled weight decay, applied uniformly to every parameter.\n\
                     That covers the encoder ρ, α, FC and BN.\n\
                     Per-step post-update shrinkage; doesn't enter the backward graph.\n\
                     Default 0.0 (off, i.e. plain Adam despite the name).\n\
                     Typical values run 1e-5 to 1e-4."
    )]
    pub(super) weight_decay: f32,

    #[command(flatten)]
    #[serde(flatten)]
    pub(super) compute: senna::embed_common::DeviceArgs,

    #[arg(
        long,
        value_enum,
        default_value = "residual",
        help = "Batch adjustment (batch|residual)",
        long_help = "batch    — divide by the per-batch delta: one fold per gene per batch.\n\
                     \x20          The encoder trains on it and reads it at inference, and\n\
                     \x20          `senna predict` reads the trained delta for a batch the\n\
                     \x20          model was trained on, one held-out batch included.\n\
                     residual — divide by each pseudobulk group's own fold (finer than a\n\
                     \x20          batch; `senna predict` re-estimates it from the held-out\n\
                     \x20          cells, which takes two or more held-out batches)."
    )]
    pub(super) adj_method: AdjMethod,

    #[arg(
        long,
        default_value_t = false,
        help = "Load all columns into memory before training",
        hide = true
    )]
    pub(super) preload_data: bool,

    #[arg(
        long,
        default_value_t = false,
        help = "Treat input files as modalities of the same cells, glued by raw barcode.",
        long_help = "Patchy multi-modal (multiome) load.\n\
                     Each file keeps its own feature space.\n\
                     There is no cross-file barcode suffixing.\n\
                     Cells are unioned across files by raw barcode.\n\
                     A cell seen only in RNA contributes triplets to the RNA block.\n\
                     ATAC-only cells contribute just to the ATAC block.\n\
                     Shared cells get both.\n\
                     Disables `@<basename>` suffixing on cell names.\n\
                     Maps to `ColumnAlignment::Union` in the loader.\n\
                     \n\
                     With --multiome, batch resolution is constrained:\n\
                     a single --batch-files file is allowed (one label per unified cell),\n\
                     or embedded `@batch` tags in raw column names that AGREE across modalities.\n\
                     The default `@<filename>` fallback is disabled.\n\
                     A cell can come from multiple files and cannot carry two labels."
    )]
    pub(super) multiome: bool,

    #[command(flatten)]
    pub(super) hvg: data_beans::alg::hvg::HvgCliArgs,

    #[arg(
        long,
        default_value_t = 0.01,
        help = "Uniform smoothing of topic proportions during training",
        long_help = "z_smooth = (1-α) z + α/K. It keeps every topic on the gradient path.\n\
                     Dead topics are thereby prevented. Typical values run 0.01 to 0.2;\n\
                     0 disables it."
    )]
    pub(super) topic_smoothing: f64,

    #[arg(
        long,
        default_value_t = 0.3,
        help = "Masked-imputation fraction:\n\
                held-out top-K genes per cell (typical 0.2–0.5)",
        long_help = "Masked-imputation fraction. Per cell,\n\
                     this share of its top-K genes is held out.\n\
                     The NB embedded-topic head predicts them.\n\
                     The rest are the encoder's visible input. Typical values run 0.2 to 0.5.\n\
                     \n\
                     This masking is the regularizer.\n\
                     It replaces the collapse-prone ELBO and KL."
    )]
    pub(super) mask_fraction: f64,

    #[arg(
        long,
        default_value_t = 0.0,
        help = "Held-out imputation eval: fraction to hold out after training (0 = off)",
        long_help = "Run a held-out masked-imputation evaluation after training.\n\
                     It logs the mean log-likelihood per held-out gene.\n\
                     \n\
                     For each cell, this fraction of its observed genes is hidden.\n\
                     The rest encode a latent.\n\
                     The trained decoder then imputes the hidden genes.\n\
                     \n\
                     Unlike the per-epoch training likelihood, this is not optimized.\n\
                     So it distinguishes real structure from overfitting.\n\
                     Use a fixed --eval-seed to compare heads on the same positions.\n\
                     0 disables it, and is the default.",
        hide = true
    )]
    pub(super) eval_mask_fraction: f64,

    #[arg(
        long,
        default_value_t = 42,
        help = "Seed for the held-out imputation mask (see --eval-mask-fraction)",
        hide = true
    )]
    pub(super) eval_seed: u64,

    #[arg(
        long,
        value_enum,
        default_value_t = MaskScheduleArg::Fixed,
        help = "Mask-rate schedule: fixed or uniform per-minibatch sampling",
        long_help = "Mask-rate schedule. fixed uses --mask-fraction.\n\
                     uniform samples the rate per row and epoch,\n\
                     within [--mask-rate-lo, --mask-rate-hi]. That is the any-order,\n\
                     absorbing-diffusion style."
    )]
    pub(super) mask_schedule: MaskScheduleArg,

    #[arg(
        long,
        value_enum,
        default_value_t = MaskedLikelihoodArg::Nb,
        help = "Masked-loss likelihood:\n\
                nb (overdispersed counts) or multinomial (depth-invariant)."
    )]
    pub(super) masked_likelihood: MaskedLikelihoodArg,

    /// The query-decoder settings an OLD run recorded, kept only so `senna
    /// update` can replay such a manifest.
    ///
    /// Not flags: the query head is no longer wired into the masked family (the
    /// library module stays, unreachable, for re-wiring later). Recorded values
    /// are reported once and ignored — see [`MaskedTopicArgs::recorded_query_flags`].
    #[arg(skip)]
    #[serde(default, skip_serializing_if = "Option::is_none")]
    query_decoder: Option<bool>,

    #[arg(skip)]
    #[serde(default, skip_serializing_if = "Option::is_none")]
    query_rank: Option<usize>,

    #[arg(skip)]
    #[serde(default, skip_serializing_if = "Option::is_none")]
    query_extra: Option<usize>,

    #[arg(skip)]
    #[serde(default, skip_serializing_if = "Option::is_none")]
    query_penalty: Option<f64>,

    #[arg(
        long,
        default_value_t = 0.1,
        help = "Lower bound of the per-minibatch mask rate when --mask-schedule=uniform.",
        hide = true
    )]
    pub(super) mask_rate_lo: f64,

    #[arg(
        long,
        default_value_t = 0.6,
        help = "Upper bound of the per-minibatch mask rate when --mask-schedule=uniform.",
        hide = true
    )]
    pub(super) mask_rate_hi: f64,

    #[arg(
        long,
        help = "Train on Poisson draws from the pseudobulk rates, redrawn each epoch",
        long_help = "Train on Poisson draws from the pseudobulk rates, not the rates themselves.\n\
                     A fresh draw is taken every epoch.\n\
                     The encoder then trains on rows shaped like the counts it sees at inference.\n\
                     \n\
                     In an A/B on a targeted panel the held-out imputation\n\
                     likelihood improved, while the cell latent became SHARPER,\n\
                     not softer.\n\
                     \n\
                     So this is a likelihood lever, not a fix for a one-hot latent.\n\
                     The encoder is one-hot on whichever distribution it trained on.\n\
                     Read the \"latent sharpness\" line after training either way.\n\
                     \n\
                     Off by default. Use --seed to reproduce a draw."
    )]
    pub(super) poisson_thin: bool,

    #[command(flatten)]
    #[serde(flatten)]
    pub(super) coarsening: data_beans::alg::feature_coarsening::FeatureCoarseningArgs,

    #[arg(
        long,
        default_value_t = 42,
        value_name = "N",
        help = "Seed for every random choice the fit makes",
        long_help = "Seed for the stochastic training choices this subcommand owns:\n\
                     the hidden set (and its rate under --mask-schedule uniform),\n\
                     and --poisson-thin's per-epoch draw.\n\
                     \n\
                     Each is keyed on its own sub-stream — the hidden set on\n\
                     (seed, epoch, level, row), drawn once per epoch, the thinning\n\
                     draw on (seed, epoch, level, column) — so all are reproducible\n\
                     whatever the thread count, the batch size or the shuffle.\n\
                     \n\
                     The pseudobulk posterior draw, the feature grouping, the\n\
                     initial weights and the minibatch order follow it too, so\n\
                     the same seed replays the fit (a GPU run may still differ\n\
                     in its last bits)."
    )]
    pub(super) seed: u64,

    #[arg(
        long,
        default_value_t = graph_embedding_util::EmbeddingDim::Fixed(128),
        value_name = "H|auto",
        help = "Per-feature embedding dimension H (default 128; auto = a given feature embedding's width, else 2 × n-latent-topics)",
        long_help = "Dimension H of the per-gene embedding ρ ∈ ℝ^{D×H}.\n\
                     ρ is shared between the encoder and the decoder.\n\
                     The encoder value-weights a pool over each cell's top-K.\n\
                     The decoder computes β_kd = log_softmax_d(α_k · ρ_dᵀ),\n\
                     with α ∈ ℝ^{K×H} as the topic embeddings.\n\
                     \n\
                     β has rank ≤ H, so H must be at least K. K is --n-latent-topics.\n\
                     Otherwise K independent topics are not representable. Default 128;\n\
                     `auto` takes the width of a given feature embedding, else 2K.\n\
                     H < K errors at startup."
    )]
    pub(super) embedding_dim: graph_embedding_util::EmbeddingDim,

    #[arg(
        long,
        help = "Feature-feature edge list (TSV/CSV) to restrict the feature axis",
        long_help = "Optional feature-feature edge list (TSV/CSV).\n\
                     It RESTRICTS the feature axis to graph-connected genes.\n\
                     See --no-feature-network-restrict to turn that off.\n\
                     Graph *diffusion* (GCN) is not supported by the masked encoder,\n\
                     so the edges only drive feature selection here.\n\
                     Edges may be intra- or cross-modal.\n\
                     Examples: gene-gene PPI, peak-gene ABC, ATAC regulatory links.\n\
                     Edge names are resolved against the loaded gene axis."
    )]
    pub(super) feature_network: Option<Box<str>>,

    #[arg(
        long,
        default_value_t = false,
        help = "Also try forward/reverse prefix matching for feature-network name resolution",
        hide = true
    )]
    pub(super) feature_network_prefix_match: bool,

    #[arg(
        long,
        help = "Alias-splitting delimiter for feature-network name resolution (e.g. '_')",
        long_help = "Alias-splitting delimiter for feature-network name resolution.\n\
                     When set, such as '_', each row name is registered twice.\n\
                     It registers under its full form AND every split component.\n\
                     So `ENSG00000000130_GENE1` matches edges naming either part.\n\
                     That is `ENSG00000000130` or `GENE1`.\n\
                     The resolver points both aliases at one row.",
        hide = true
    )]
    pub(super) feature_network_delim: Option<char>,

    #[arg(
        long,
        default_value_t = 1,
        help = "Shared-neighbor edge QC threshold (default 1; 0 = keep all edges)",
        long_help = "Shared-neighbor edge QC.\n\
                     It drops an edge (u,v) whose endpoints share fewer than N neighbours.\n\
                     Neighbours are counted in the feature network.\n\
                     The default of 1 drops edges with zero corroboration.\n\
                     That is standard PPI topological-overlap denoising.\n\
                     Set 0 to keep every parsed edge.",
        hide = true
    )]
    pub(super) feature_network_min_shared_neighbors: usize,

    #[arg(
        long,
        default_value_t = 0,
        help = "Per-node degree cap on the feature network (0 = off)",
        long_help = "Per-node degree cap on the feature network (0 = off).\n\
                     After shared-neighbor QC, for each feature with degree > N,\n\
                     rank its neighbors by shared-neighbor count and keep the top N. Union-symmetric:\n\
                     an edge survives iff either endpoint kept it.\n\
                     Caps PPI hubs whose degree would otherwise blow up per-cell sub-adjacency.",
        hide = true
    )]
    pub(super) feature_network_max_degree: usize,

    #[arg(
        long,
        default_value_t = 0,
        help = "Iterative k-core pruning threshold on the feature network (0 = off)",
        long_help = "Iterative k-core pruning threshold on the feature network.\n\
                     The default of 0 turns it off.\n\
                     It drops every feature whose degree falls below N.\n\
                     Pruning repeats until the subgraph is N-degenerate.",
        hide = true
    )]
    pub(super) feature_network_min_degree: usize,

    #[arg(
        long,
        default_value_t = false,
        help = "Disable feature-network feature restriction (keep full feature axis)",
        long_help = "Disable feature-network feature restriction. By default,\n\
                     when --feature-network is supplied,\n\
                     features with zero edges after QC are dropped.\n\
                     The QC pipeline is shared-neighbor prune, hub cap, k-core.\n\
                     Dropping happens before projection, collapse and training.\n\
                     \n\
                     Pass this to keep the full feature axis.\n\
                     Note that with restriction off the graph has no effect:\n\
                     the masked encoder does not diffuse."
    )]
    pub(super) no_feature_network_restrict: bool,

    #[arg(
        long,
        value_enum,
        default_value = "auto",
        help = "Per-name canonicalization across input backends",
        long_help = "How row names align across `--data-files`.\n\
                     \n\
                     `auto` — sniff EACH file's row names, pick a rule per file,\n\
                     and reconcile them into one rule for the load.\n\
                     It picks locus-overlap if ≥50% parse as `chr:start-end`.\n\
                     It picks gene if ≥50% contain `_`, and exact otherwise (default).\n\
                     \n\
                     `exact` — strict string match.\n\
                     \n\
                     `gene` — also register each `_`-split component as an alias.\n\
                     So `ENSG000_GENE1` and `GENE1` resolve to the same row.\n\
                     \n\
                     `locus` — normalize `chr1:1000-2000`, `1:1000-2000`, and so on.\n\
                     They all reach a canonical form.\n\
                     \n\
                     `locus-overlap` — same as `locus`, plus overlap clustering.\n\
                     Intervals that overlap on the same chromosome are grouped.\n\
                     Useful for cross-dataset ATAC peak sets called independently."
    )]
    pub(super) feature_name_kind: FeatureNameKindArg,

    #[command(flatten)]
    pub(super) qc: QcArgs,
}

impl MaskedTopicArgs {
    /// Which query-decoder flags a replayed manifest recorded, by their old
    /// CLI names. Empty for anything this build wrote.
    pub(crate) fn recorded_query_flags(&self) -> Vec<&'static str> {
        [
            self.query_decoder.is_some().then_some("--query-decoder"),
            self.query_rank.is_some().then_some("--query-rank"),
            self.query_extra.is_some().then_some("--query-extra"),
            self.query_penalty.is_some().then_some("--query-penalty"),
        ]
        .into_iter()
        .flatten()
        .collect()
    }

    /// Refuse the combinations the window-free encoder cannot express, before
    /// anything is read.
    ///
    /// The reason modules cannot compose is the encoder's to give, and it gives
    /// it in full at
    /// [`legume_numeric::candle::encoder::IndexedEmbeddingEncoder::forward_dense_masked`].
    /// What belongs here is only the early refusal, so a run that cannot work
    /// stops before it reads a single file.
    pub(crate) fn validate(&self) -> anyhow::Result<()> {
        // The rate is the model, not a knob with a safe default: 0 hides
        // nothing and 1 hides everything, and either leaves one side of the
        // masked objective with no work. The loader used to clamp both back to
        // a one-gene draw, answering a question nobody asked. This refuses the
        // flag by name instead, in the open interval the draw actually needs.
        let rate_in_unit_interval = |flag: &str, x: f64| -> anyhow::Result<()> {
            anyhow::ensure!(
                x > 0.0 && x < 1.0,
                "{flag} must be in the open interval (0, 1), got {x}: a rate of 0 hides \
                 nothing for the decoder to impute and a rate of 1 leaves the encoder \
                 nothing to read."
            );
            Ok(())
        };
        rate_in_unit_interval("--mask-fraction", self.mask_fraction)?;
        // Under the uniform schedule it is these bounds, not --mask-fraction,
        // that become the per-row rate.
        if matches!(self.mask_schedule, MaskScheduleArg::Uniform) {
            rate_in_unit_interval("--mask-rate-lo", self.mask_rate_lo)?;
            rate_in_unit_interval("--mask-rate-hi", self.mask_rate_hi)?;
        }
        Ok(())
    }
}

impl crate::update::Updatable for MaskedTopicArgs {
    fn rebase(&mut self, r: crate::update::Rebase) {
        self.data_files = r.data_files;
        self.batch_files = r.batch_files;
        self.out = r.out;
        self.init_from = Some(r.init_from);
        self.pb_reference = r.reference;
        // Only when growth was asked for; otherwise the recorded sizes replay
        // verbatim. Both axes are pinned together even if only one grows: a
        // recorded `--embedding-dim auto` means "2K", which would otherwise
        // track the grown K and silently resize ρ.
        if !r.growth.is_none() {
            self.n_latent_topics = r.parent_topics + r.growth.add_topics;
            self.add_topics = r.growth.add_topics;
            self.embedding_dim = match r.parent_embedding_dim {
                Some(h) => {
                    graph_embedding_util::EmbeddingDim::Fixed(h + r.growth.add_embedding_dim)
                }
                None => self.embedding_dim,
            };
            self.add_embedding_dim = r.growth.add_embedding_dim;
        }
        // See `TopicArgs::rebase` — the inherited partition cannot cover new cells.
        // (Here `--from` also carries --freeze-feature-embedding, which a warm
        // start supersedes: the weights already contain that ρ.)
        self.from = None;
        self.pb_from = r.pb_from;
        if let Some(e) = r.epochs {
            self.epochs = e;
        }
    }
}
