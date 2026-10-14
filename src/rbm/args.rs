use super::hidden::HiddenArg;
use senna::embed_common::*;

#[derive(Args, Debug, serde::Serialize, serde::Deserialize)]
#[serde(default = "senna::embed_common::clap_defaults")]
pub struct RbmArgs {
    #[arg(
        value_delimiter = ',',
        help = "Input data files (.zarr or .h5; optional when --from is given)",
        long_help = "Input data files (.zarr or .h5; optional when --from is given).\n\
                     Several files that are the same cells in different assays\n\
                     (RNA and ATAC, say) are detected as a multiome layout,\n\
                     as `senna bge` detects it: every modality becomes its own\n\
                     block of visible units, and a cell missing one is read\n\
                     from the blocks it has."
    )]
    pub(crate) data_files: Vec<Box<str>>,

    #[arg(
        long,
        help = "Chain data, batch and cell→pb partition from a prior run"
    )]
    pub(crate) from: Option<Box<str>>,

    #[arg(
        long,
        short,
        required = true,
        help = "Output file prefix",
        long_help = "Prefix for generated files:\n  \
                     {out}.latent.parquet             cell × K posterior mean E[h | v]\n  \
                     {out}.pb_latent.parquet          pseudobulk × K posterior mean\n  \
                     {out}.cell_score.parquet         free energy and reconstruction per read\n  \
                     {out}.feature_embedding.parquet  feature embedding ρ\n  \
                     {out}.hidden_embedding.parquet   every block's hidden-unit embedding α\n  \
                     {out}.dictionary.parquet         W = ρ αᵀ, feature × K\n  \
                     {out}.feature_bias.parquet       visible bias b\n  \
                     {out}.log_likelihood.parquet     training trace\n  \
                     {out}.model.json, .safetensors   the fitted machine\n  \
                     {out}.senna.json                 run manifest"
    )]
    pub(crate) out: Box<str>,

    #[arg(
        long,
        short,
        value_delimiter(','),
        help = "Batch membership files, one per data file"
    )]
    pub(crate) batch_files: Option<Vec<Box<str>>>,

    #[arg(
        long,
        value_name = "FILES",
        help = "Declare the multiome layout by hand; one sample (group) per flag",
        long_help = "Declare the multiome layout instead of letting it be detected,\n\
                     as `senna bge --multiome`: the comma-separated files of one sample,\n\
                     each optionally `label=file`, one flag per sample.\n\
                     Needed when modalities share feature names\n\
                     (`--multiome spliced=s.zarr,unspliced=u.zarr`)."
    )]
    pub(crate) multiome: Vec<Box<str>>,

    #[command(flatten)]
    pub(crate) collapse: crate::refine_weighting::CollapseArgs,

    #[command(flatten)]
    #[serde(flatten)]
    pub(crate) pb_from: crate::refine_weighting::PbFromArgs,

    #[arg(short = 't', long, default_value_t = 64, help = "Hidden units (K)")]
    pub(crate) n_latent: usize,

    #[arg(
        long,
        value_enum,
        default_value = "bernoulli",
        help = "Hidden unit kind (bernoulli|binomial|poisson|gaussian)",
        long_help = "The hidden units' distribution given a cell's reads.\n\
                     bernoulli: on/off units.\n\
                     binomial: counts 0..--trials.\n\
                     poisson: counts truncated at --max-count.\n\
                     gaussian: unit-variance real units."
    )]
    pub(crate) hidden: HiddenArg,

    #[arg(long, default_value_t = 4, help = "Trials of a binomial hidden unit")]
    pub(crate) trials: u32,

    #[arg(
        long,
        default_value_t = 10,
        help = "Largest count of a Poisson hidden unit"
    )]
    pub(crate) max_count: u32,

    #[arg(
        long,
        default_value_t = 64,
        help = "Feature embedding width H (W = ρ αᵀ)",
        long_help = "Width H of the feature embedding ρ and the hidden-unit embedding α.\n\
                     The weights are W = ρ αᵀ: a cell reaches the hidden units only\n\
                     through Σ_f (D/N) v_f ρ_f, where its reads sit in ρ.\n\
                     A given table (--freeze/--init/--lora-feature-embedding) sets H."
    )]
    pub(crate) embedding_dim: usize,

    #[arg(
        long,
        default_value_t = false,
        help = "Share one row of ρ across modalities per feature name",
        long_help = "Under a multiome layout, features `{name}/{modality}` with the same\n\
                     name share one row of ρ; each modality keeps its own α and bias."
    )]
    pub(crate) tie_symbols: bool,

    #[command(flatten)]
    pub(crate) feature_embedding: crate::feature_embedding_args::FeatureEmbeddingArgs,

    #[arg(
        long,
        default_value_t = 0.01,
        help = "L2 penalty on ρ and α (mean squared row norm; 0 = off)"
    )]
    pub(crate) embedding_l2: f32,

    #[arg(
        long,
        default_value_t = false,
        hide = true,
        help = "Data term at a fresh draw of D reads instead of the expected reads"
    )]
    pub(crate) draw_data: bool,

    #[arg(
        long,
        default_value_t = 0.1,
        help = "Sparsity target: the mean activation every hidden unit is held near",
        long_help = "Hinton's sparsity target (A Practical Guide to Training RBMs, 2010).\n\
                     A running estimate q of every unit's mean activation (on the unit's\n\
                     0..1 scale) is held near this target by a cross-entropy penalty at\n\
                     --sparsity-cost, so no unit dies or saturates and a cell is read as a\n\
                     combination of units. Not applied to Gaussian units."
    )]
    pub(crate) sparsity_target: f32,

    #[arg(
        long,
        default_value_t = 0.01,
        help = "Weight of the sparsity penalty (0 = off)"
    )]
    pub(crate) sparsity_cost: f32,

    #[arg(
        long,
        default_value_t = 0.05,
        help = "Share of the finest pseudobulks held out to watch for overfitting",
        long_help = "These pseudobulks are left out of the finest level's training; every\n\
                     epoch logs their mean free energy per read minus that of as many\n\
                     training pseudobulks (log_likelihood.parquet, `heldout_gap`).\n\
                     A gap that keeps rising is overfitting. Coarser levels still contain\n\
                     their cells, so the check is conservative."
    )]
    pub(crate) holdout: f32,

    #[command(flatten)]
    pub(crate) modules: graph_embedding_util::FeatureModuleArgs,

    #[arg(
        long,
        default_value_t = false,
        conflicts_with = "feature_modules",
        help = "Free per-feature rows of ρ instead of feature modules",
        long_help = "Feature modules are on by default (M = 50): every row of ρ is a sparse\n\
                     mixture of shared module vectors plus a small residual, which keeps\n\
                     the hidden units in use. This trains one free row per feature instead.\n\
                     Modules are also off when a feature embedding is given\n\
                     (--freeze/--init/--lora-feature-embedding)."
    )]
    pub(crate) no_feature_modules: bool,

    #[arg(long, short = 'i', default_value_t = 200, help = "Training epochs")]
    pub(crate) epochs: usize,

    #[arg(long, default_value_t = 100, help = "Pseudobulks per minibatch")]
    pub(crate) minibatch_size: usize,

    #[arg(
        long,
        help = "Persistent Gibbs chains, i.e. fantasy particles (default: the minibatch size)"
    )]
    pub(crate) chains: Option<usize>,

    #[arg(long, default_value_t = 1, help = "Gibbs sweeps per training step")]
    pub(crate) gibbs_sweeps: usize,

    #[arg(
        long,
        alias = "lr",
        help = "Adam learning rate [default: 0.005; 0.0005 for Gaussian hidden units]",
        long_help = "Adam learning rate. Default 0.005, and ten times smaller (0.0005)\n\
                     for Gaussian hidden units: their activity has no bound, so a large\n\
                     step lets the weights run away (Hinton, A practical guide to\n\
                     training restricted Boltzmann machines (2010), §13.2-13.3)."
    )]
    pub(crate) learning_rate: Option<f32>,

    #[arg(
        long,
        value_delimiter = ',',
        default_values_t = vec![20.0],
        help = "Training depth D: reads a pseudobulk is drawn at (one value, or one per block)",
        long_help = "Training depth D, per block or one value for all.\n\
                     Pseudobulks are drawn at D reads every step, and a cell is read\n\
                     per unit depth against it (its reads scaled to D), so its latent\n\
                     does not depend on its depth. D tempers the likelihood: the energy\n\
                     barriers between modes grow with D, and the persistent chains\n\
                     mix only while they stay low. Larger D, sharper posteriors."
    )]
    pub(crate) depth: Vec<f32>,

    #[arg(
        long,
        default_value_t = 42,
        value_name = "N",
        help = "Seed for every random choice the fit makes"
    )]
    pub(crate) seed: u64,

    #[arg(
        long,
        value_enum,
        default_value = "cpu",
        help = "Compute device (cpu|cuda|metal)"
    )]
    pub(crate) device: ComputeDevice,

    #[arg(long, default_value_t = 0, help = "CUDA/Metal device index")]
    pub(crate) device_no: usize,

    #[arg(long, default_value_t = 1024, help = "Cells per evaluation block")]
    pub(crate) block_size: usize,

    #[arg(
        long,
        default_value_t = false,
        help = "Load all columns into memory before training",
        hide = true
    )]
    pub(crate) preload_data: bool,

    #[command(flatten)]
    pub(crate) hvg: crate::hvg::HvgCliArgs,

    #[arg(
        long,
        value_enum,
        default_value = "auto",
        help = "Per-name canonicalization across input backends (as `senna vae`)"
    )]
    pub(crate) feature_name_kind: crate::masked_topic::FeatureNameKindArg,

    #[command(flatten)]
    pub(crate) qc: QcArgs,
}
