//! `senna topic` command-line arguments: the decoder choice and [`TopicArgs`],
//! and how `senna update` rebases them.

use senna::embed_common::*;

use legume_numeric::candle::decoder::nb_mixture::DECODER_NAME as NBMIXTURE_NAME;

#[derive(ValueEnum, Clone, Copy, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
#[clap(rename_all = "lowercase")]
#[serde(rename_all = "lowercase")]
pub(crate) enum DecoderType {
    /// Softmax dictionary with multinomial likelihood
    Multinom,
    /// Negative binomial with per-gene dispersion and library size
    Nb,
    /// NB with an ambient-RNA mixture (`α_g`) and per-sample ρ from library size
    NbMixture,
}

impl DecoderType {
    pub fn as_str(self) -> &'static str {
        match self {
            DecoderType::Multinom => "multinom",
            DecoderType::Nb => "nb",
            DecoderType::NbMixture => NBMIXTURE_NAME,
        }
    }
}

#[derive(Args, Debug, serde::Serialize, serde::Deserialize)]
#[serde(default = "senna::embed_common::clap_defaults")]
pub struct TopicArgs {
    #[arg(
        value_delimiter = ',',
        help = "Input data files (.zarr or .h5; optional when --from is given)",
        long_help = "Sparse backends produced by `data-beans from-mtx`.\n\
                     Multiple files may be passed, comma- or space-separated.\n\
                     They are concatenated column-wise on a shared feature set.\n\
                     When `--from <run.senna.json>` is provided and this list is empty,\n\
                     the data paths come from the source manifest."
    )]
    pub(crate) data_files: Vec<Box<str>>,

    #[arg(
        long,
        help = "Chain data, batch and cell→pb partition from a prior run",
        long_help = "Read a `{run}.senna.json` manifest and pre-fill `data_files`,\n\
                     `--batch-files`,\n\
                     and (when present) the cell→pb partition from the source run.\n\
                     Inheriting the partition skips a costly refinement step.\n\
                     That step is BBKNN plus Poisson DC-SBM.\n\
                     Explicit CLI flags override the manifest.\n\
                     SVD-family sources are rejected."
    )]
    pub(crate) from: Option<Box<str>>,

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
                     {out}.model.json               model metadata (for `senna predict`)\n  \
                     {out}.dispersion.parquet       NB dispersion (nb / nbmixture)\n  \
                     {out}.alpha.parquet            ambient gene profile (nbmixture)\n  \
                     {out}.rho.parquet              ρ sigmoid coefficients (nbmixture)\n  \
                     {out}.cell_proj.parquet        cached random projection (for `senna layout`)\n  \
                     {out}.senna.json               run manifest for `senna layout/plot --from`\n\
                     \n\
                     With --decoder a,b,c:\n\
                     per-decoder dictionaries written as {out}.{name}.dictionary.parquet."
    )]
    pub(crate) out: Box<str>,

    #[arg(
        long,
        short,
        value_delimiter(','),
        help = "Batch membership files, one per data file",
        long_help = "Each file lists a batch label per cell.\n\
                     The cells come in the same order as its matching data file.\n\
                     Example: batch1.tsv,batch2.tsv"
    )]
    pub(crate) batch_files: Option<Vec<Box<str>>>,

    #[command(flatten)]
    pub(crate) collapse: crate::refine_weighting::CollapseArgs,

    #[command(flatten)]
    #[serde(flatten)]
    pub(crate) pb_from: crate::refine_weighting::PbFromArgs,

    #[arg(
        long = "init-from",
        help = "Initialize encoder + decoder weights from a previously trained model",
        long_help = "Path prefix of a model saved by `senna topic`.\n\
                     It matches {prefix}.model.json + {prefix}.safetensors.\n\
                     Architecture must match: same K, encoder layers, level_decoder_dims,\n\
                     and n_features_full / n_features_encoder.\n\
                     Cross-gene-set warm-start is not supported.\n\
                     Train on the same gene set."
    )]
    pub(crate) init_from: Option<Box<str>>,

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
    pub(crate) add_topics: usize,

    /// The parent's carried pseudobulks, when `senna update` chose to reuse
    /// them instead of re-reading its cells. Derived per invocation, so it is
    /// neither a CLI flag nor part of the recorded configuration.
    #[arg(skip)]
    #[serde(skip)]
    pub(crate) pb_reference: Option<senna::pb_reference::ReferenceInput>,

    #[arg(
        long,
        help = "Cells per rayon job (omit for auto-scaling by feature count)",
        hide = true
    )]
    pub(crate) block_size: Option<usize>,

    #[arg(
        short = 't',
        long,
        default_value_t = 10,
        help = "Number of latent topics (K)"
    )]
    pub(crate) n_latent_topics: usize,

    #[arg(
        long,
        short = 'e',
        value_delimiter(','),
        default_values_t = vec![128, 1024, 128],
        help = "Encoder hidden layer sizes (comma-separated)",
        long_help = "Example: 128,1024,128 (input → 128 → 1024 → 128 → topics)."
    )]
    pub(crate) encoder_layers: Vec<usize>,

    #[arg(long, short = 'i', default_value_t = 1000, help = "Training epochs")]
    pub(crate) epochs: usize,

    #[arg(
        long,
        default_value_t = 42,
        value_name = "N",
        help = "Seed for every random choice the fit makes",
        long_help = "The pseudobulk posterior draw it trains on, the feature grouping,\n\
                     the initial weights, the minibatch order and the\n\
                     reparameterization noise are each drawn from their own\n\
                     sub-stream of it. A GPU run may still differ in its last bits."
    )]
    pub(crate) seed: u64,

    #[arg(long, default_value_t = 100, help = "Training minibatch size")]
    pub(crate) minibatch_size: usize,

    #[arg(
        long,
        default_value_t = 0.01,
        alias = "lr",
        help = "Adam learning rate"
    )]
    pub(crate) learning_rate: f32,

    #[arg(
        long,
        default_value_t = 1.0,
        help = "Global L2 gradient norm clip per minibatch (0 = off; typical 0.5–5.0)"
    )]
    pub(crate) grad_clip: f32,

    #[command(flatten)]
    #[serde(flatten)]
    pub(crate) compute: senna::embed_common::DeviceArgs,

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
    pub(crate) adj_method: AdjMethod,

    #[arg(
        long,
        default_value_t = false,
        help = "Load all columns into memory before training",
        hide = true
    )]
    pub(crate) preload_data: bool,

    #[arg(
        long,
        value_enum,
        default_value = "auto",
        help = "Canonicalize row names across backends: auto|exact|gene|locus|locus-overlap|mixed",
        long_help = "How to decide that two backends mean the same feature.\n\
                     \n\
                     Rows are unioned across input files, so a name the rule fails\n\
                     to reconcile becomes a SECOND row for the same gene — which\n\
                     enlarges the feature axis and, under `senna update`, no longer\n\
                     matches the parent's.\n\
                     \n\
                     gene   — `ENSG00000000130_GENE1` and `GENE1` are one feature\n\
                     locus  — `chr1:1000-2000`, `1:1000-2000`, `chr1_1000_2000` agree\n\
                     mixed  — per-row dispatch for a heterogeneous axis\n\
                     auto   — detect from the names present\n\
                     \n\
                     Defaults to `auto`, which detects the rule from the names\n\
                     present — the behaviour this command already had before the\n\
                     flag was exposed. Set it explicitly when auto-detection\n\
                     picks wrong for a cohort."
    )]
    pub(crate) feature_name_kind: crate::masked_topic::FeatureNameKindArg,

    #[command(flatten)]
    pub(crate) hvg: data_beans::alg::hvg::HvgCliArgs,

    #[command(flatten)]
    #[serde(flatten)]
    pub(crate) coarsening: data_beans::alg::feature_coarsening::FeatureCoarseningArgs,

    #[arg(
        long,
        value_enum,
        value_delimiter = ',',
        default_value = "nbmixture",
        help = "Decoder type(s) [multinom|nb|nbmixture], comma-separated",
        long_help = "multinom  — NB-Fisher-weighted multinomial.\n\
                     nb        — negative binomial with per-gene dispersion.\n\
                     nbmixture — NB with ambient-RNA mixture α and per-sample ρ (default).\n\
                     \n\
                     Multiple types (e.g. --decoder multinom,nb) train jointly with a shared encoder;\n\
                     see --decoder-weights for loss weighting."
    )]
    pub(crate) decoder: Vec<DecoderType>,

    #[arg(
        long,
        value_delimiter = ',',
        help = "Per-decoder loss weights (default: equal)"
    )]
    pub(crate) decoder_weights: Option<Vec<f64>>,

    #[arg(
        long,
        default_value_t = 1e-4,
        help = "Uniform smoothing α for topic proportions (0 = off)",
        long_help = "θ = (1-α) softmax(z) + α/K. It prevents dead topics,\n\
                     by keeping every topic on the gradient path. Set 0 to disable."
    )]
    pub(crate) topic_smoothing: f64,

    #[command(flatten)]
    pub(crate) amort_refine: crate::refine_weighting::AmortRefineArgs,

    #[arg(
        long,
        default_value_t = 1.0,
        help = "Cross-entropy penalty λ on β toward anchor prior (0 = off)",
        long_help = "Pulls the decoder dictionary toward anchor PB profiles.\n\
                     This happens during training. β starts from a random init,\n\
                     and the penalty guides it."
    )]
    pub(crate) anchor_penalty: f32,

    #[arg(
        long,
        default_value_t = 0.0,
        help = "Beta(α,β) prior weight on ρ (0 = off; nbmixture only, rarely used)"
    )]
    pub(crate) rho_prior_weight: f32,

    #[arg(
        long,
        default_value_t = 2.0,
        hide = true,
        help = "Beta(α,·) shape on ρ prior"
    )]
    pub(crate) rho_prior_alpha: f32,

    #[arg(
        long,
        default_value_t = 18.0,
        hide = true,
        help = "Beta(·,β) shape on ρ prior"
    )]
    pub(crate) rho_prior_beta: f32,

    #[command(flatten)]
    pub(crate) qc: QcArgs,
}

impl crate::update::Updatable for TopicArgs {
    fn rebase(&mut self, r: crate::update::Rebase) {
        self.data_files = r.data_files;
        self.batch_files = r.batch_files;
        self.out = r.out;
        self.init_from = Some(r.init_from);
        self.pb_reference = r.reference;

        // Only when growth was asked for: otherwise the recorded K is replayed
        // verbatim, and `parent_topics` is not even looked up.
        if !r.growth.is_none() {
            self.n_latent_topics = r.parent_topics + r.growth.add_topics;
            self.add_topics = r.growth.add_topics;
        }
        // NOT inherited: `--from` would pull the parent's cell→pb partition,
        // and `align_cell_to_pb_to_cells` bails on any cell absent from the
        // source — which every newly absorbed cell is. A round's `--pb-from`
        // names the partition every model collapses on; with new data `update`
        // leaves it empty, since none could cover them.
        self.from = None;
        self.pb_from = r.pb_from;
        if let Some(e) = r.epochs {
            self.epochs = e;
        }
    }
}
