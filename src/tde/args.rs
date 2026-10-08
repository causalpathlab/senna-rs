//! `senna tde`'s command-line surface: the [`TdeArgs`] clap struct.
//!
//! Mirrors `senna/src/bge/args.rs` field for field: the same shared groups
//! (`HvgCliArgs`, `refine_weighting::CollapseArgs`, `QcArgs`,
//! `ge::FeatureModuleArgs`), the same top-level knobs, and the same help text
//! for every flag they share, including the
//! `--{freeze,init,lora}-feature-embedding` triple. tde adds its inputs
//! (`GENES...`, `--genes-sample-strip`) and the knobs of the unspliced
//! track's fit (`--divergence-l2`, `--divergence-epochs`). `TdeArgs` does not
//! implement `Updatable`; `senna update` does not (yet) continue a tde run.

use data_beans::alg::hvg::HvgCliArgs;
use graph_embedding_util as ge;
use senna::embed_common::*;

#[derive(Args, Debug, serde::Serialize, serde::Deserialize)]
#[serde(default = "senna::embed_common::clap_defaults")]
pub(crate) struct TdeArgs {
    #[arg(
        value_name = "GENES",
        value_delimiter = ',',
        required = true,
        help = "Gene-count matrices (zarr/h5) with spliced and unspliced rows",
        long_help = "Gene-count matrices to embed. Pass them all, in any order;\n\
                     space-separated so shell globs work (`senna tde out/*_count.zarr.zip`),\n\
                     commas also accepted.\n\
                     \x20\n\
                     Rows must follow `{gene}/count/{spliced|unspliced}`. The gene key is the first field,\n\
                     and the row itself is the join key across files. Every gene needs a\n\
                     spliced row; a gene without an unspliced row has no divergence.\n\
                     \x20\n\
                     Cells are matched across files by barcode within a sample;\n\
                     see --genes-sample-strip."
    )]
    pub(crate) genes: Vec<Box<str>>,

    #[arg(
        short = 'b',
        long,
        value_delimiter = ',',
        help = "Batch label files, one per data file"
    )]
    pub(crate) batch_files: Option<Vec<Box<str>>>,

    #[arg(
        long,
        default_value = "",
        help = "Suffix stripped from every input basename to form its sample id.",
        long_help = "Suffix stripped from every input basename to form its sample id.\n\
                     Empty (the default) takes each file's sample from its metadata (faba\n\
                     writes it); a file without one has _count stripped from its name.\n\
                     Files of one sample must land on the same id."
    )]
    pub(crate) genes_sample_strip: Box<str>,

    #[command(flatten)]
    pub(crate) hvg: HvgCliArgs,

    #[arg(
        long,
        default_value_t = graph_embedding_util::EmbeddingDim::Fixed(128),
        value_name = "H|auto",
        help = "Embedding dimension H (auto = the width of a given feature embedding)",
        alias = "dim-embedding"
    )]
    pub(crate) embedding_dim: graph_embedding_util::EmbeddingDim,

    #[command(flatten)]
    #[serde(flatten)]
    pub(crate) feature_embedding: crate::feature_embedding_args::FeatureEmbeddingArgs,

    #[command(flatten)]
    pub(crate) collapse: crate::refine_weighting::CollapseArgs,

    #[command(flatten)]
    pub(crate) qc: QcArgs,

    #[arg(
        long = "phase1-cells-per-pb",
        default_value_t = 16,
        help = "Cells injected per pseudobulk into phase-1 training (k); 0 = pure-pb.",
        long_help = "Phase-1 cell-axis mode (k): how many raw cells per pseudobulk-sample,\n\
                     at each collapse level, train alongside the pseudobulks.\n\
                     Phase 2 ALWAYS analytically projects every cell,\n\
                     so this shapes the feature dictionary, not the per-cell output.\n\
                     \n\
                     A pseudobulk averages many cells, and averaging smooths away the\n\
                     sharp present-versus-absent contrast that separates closely related\n\
                     states. A few raw cells restore it: measured, a moderate k raised\n\
                     purity on exactly the within-lineage classes pure-pb lost, and\n\
                     several-fold more of the embedding's dimensions carry variance.\n\
                     \n\
                     It has an interior optimum. Injecting EVERY cell is worse than\n\
                     injecting none, and costs the most; the default is the moderate\n\
                     setting that measured best, at a few times pure-pb's runtime.\n\
                     \n\
                     k = 0 → suppress the cell axis (pure-pb, fastest).\n\
                     1 ≤ k < n_cells → keep ≤ k cells per pb-sample at each level (union).\n\
                     k ≥ n_cells → every cell (slowest, and measured worse)."
    )]
    pub(crate) phase1_cells_per_pb: usize,

    #[arg(
        long = "modules-per-unit",
        default_value_t = 8,
        value_name = "K",
        help = "Modules scored at the gene level per unit per step (phase 1)",
        long_help = "Phase 1 scores every module exactly, every step.\n\
                     At the gene level it scores K modules per unit per step,\n\
                     drawn in proportion to the unit's share of counts in them.\n\
                     Higher K covers more of a unit's genes per epoch at linear cost."
    )]
    pub(crate) modules_per_unit: usize,

    #[arg(
        long = "skip-etm",
        default_value_t = false,
        help = "Skip ETM resolution; emit raw bge embeddings (Z and ρ) only.",
        long_help = "Skip the default ETM resolution.\n\
                     Only the raw bge embeddings are then emitted: cell_embedding = Z,\n\
                     dictionary = ρ, and no latent.\n\
                     \n\
                     By default tde resolves ETM topics from the cell embedding,\n\
                     by anchor analysis. It then ALSO writes the topic-model tables:\n\
                     latent = log θ, dictionary = β, topic_embedding = α.\n\
                     \n\
                     Either way, Z lands in {out}.cell_embedding.parquet."
    )]
    pub(crate) skip_etm: bool,

    #[arg(
        long = "num-topics",
        help = "ETM topics K (omit to take one topic per cell cluster)."
    )]
    pub(crate) num_topics: Option<usize>,

    #[arg(short = 'i', long, default_value_t = 1000, help = "Training epochs")]
    pub(crate) epochs: usize,

    #[arg(
        long,
        help = "Units (pseudobulks + phase-1 cells) per phase-1 step (unset: 256)",
        long_help = "Units per phase-1 step: the pseudobulks at every collapse level,\n\
                     plus each pseudobulk's phase-1 cell subsample.\n\
                     Unset, the default is 256."
    )]
    pub(crate) batch_size: Option<usize>,

    #[arg(
        long,
        default_value_t = 0.01,
        help = "Learning rate: the row-wise Adagrad step of phase 1.",
        alias = "lr"
    )]
    pub(crate) learning_rate: f64,

    #[arg(
        long,
        default_value_t = 0.0,
        help = "Weight decay: a per-row shrink 1 − lr·wd on every touched row.",
        long_help = "Weight decay: a per-row shrink 1 − lr·wd is applied to every row a step\n\
                     touches, right before that row's Adagrad update.\n\
                     Per-step post-update shrinkage; doesn't enter the backward graph.\n\
                     Default 0.0 (off)."
    )]
    pub(crate) weight_decay: f64,

    #[arg(
        long,
        help = "Cells per block for column I/O / streaming (omit for auto).",
        long_help = "Cells per parallel block for streaming column-block I/O.\n\
                     Omit for auto-scaling, which clamps to 100 for large feature counts.\n\
                     That is slow on rotational disks.\n\
                     Pass 1024+ when you have RAM, especially without --preload-data.",
        hide = true
    )]
    pub(crate) block_size: Option<usize>,

    #[arg(
        long,
        default_value_t = false,
        help = "Preload all sparse column data into memory. Faster when data fits in RAM;\n\
                required on slow disks.",
        hide = true
    )]
    pub(crate) preload_data: bool,

    #[arg(
        long = "divergence-l2",
        default_value_t = 30.0,
        value_name = "L2",
        help = "Ridge on each gene's unspliced direction; 0 = off",
        long_help = "Each gene's unspliced share in a cell is read as\n\
                     κ_c + δ_g + ⟨θ_c, η_g⟩, with θ_c the cell's spliced state and η_g the\n\
                     gene's direction in that space. This is the ridge λ/2 Σ_g ‖η_g‖²\n\
                     against the summed binomial log-likelihood of every cell's reads.\n\
                     Larger values pull every gene toward one ratio across cells."
    )]
    pub(crate) divergence_l2: f32,

    #[arg(
        long = "divergence-epochs",
        default_value_t = 50,
        help = "Passes over the cells fitting the unspliced ratios"
    )]
    pub(crate) divergence_epochs: usize,

    #[arg(
        long = "divergence-learning-rate",
        default_value_t = 0.01,
        help = "Adam step of the unspliced fit",
        hide = true
    )]
    pub(crate) divergence_learning_rate: f64,

    #[arg(
        long,
        default_value_t = 1,
        value_name = "N",
        help = "Seed for training (default 1).",
        long_help = "Seed for the fit's sampling RNG and parameter initialization.\n\
                     \n\
                     Changing it gives an INDEPENDENT fit.\n\
                     Initialization and minibatch order both differ.\n\
                     That is what an A/B across seeds needs.\n\
                     \n\
                     It does NOT make a run bit-reproducible.\n\
                     Two runs at the same seed still differ slightly."
    )]
    pub(crate) seed: u64,

    #[arg(long, default_value_t = ComputeDevice::Cpu, value_enum, help = "Compute device")]
    pub(crate) device: ComputeDevice,

    #[arg(long, default_value_t = 0, help = "Device ordinal (for cuda/metal)")]
    pub(crate) device_no: usize,

    #[command(flatten)]
    pub(crate) modules: ge::FeatureModuleArgs,

    #[arg(
        long,
        short,
        required = true,
        help = "Output prefix",
        long_help = "Output prefix. It produces {out}.cell_embedding.parquet, which is Z,\n\
                     {out}.feature_embedding.parquet, which is the raw gene table,\n\
                     {out}.feature_coembedding.parquet, the genes on the cell manifold,\n\
                     {out}.feature_bias.parquet, {out}.cell_bias.parquet, and {out}.senna.json,\n\
                     all from the spliced counts. The unspliced track adds\n\
                     {out}.cell_velocity.parquet (each cell's velocity θ̇ and κ),\n\
                     {out}.feature_divergence.parquet (each gene's ratio, steady state, log γ)\n\
                     and {out}.divergence_loading.parquet (each gene's direction η).\n\
                     Unless --skip-etm, it adds three more:\n\
                     {out}.latent.parquet, {out}.dictionary.parquet and {out}.topic_embedding.parquet."
    )]
    pub(crate) out: Box<str>,
}

impl TdeArgs {
    /// The divergent-track knobs on `axis`.
    pub(crate) fn divergence(&self, axis: ge::DivergenceAxis) -> ge::DivergenceConfig {
        ge::DivergenceConfig {
            axis,
            l2: self.divergence_l2,
            epochs: self.divergence_epochs,
            learning_rate: self.divergence_learning_rate,
        }
    }
}

#[cfg(test)]
#[path = "args/tests.rs"]
mod tests;
