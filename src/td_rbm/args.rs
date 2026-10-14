//! `senna td-rbm`'s command-line surface.

use senna::embed_common::*;

#[derive(Args, Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(default = "senna::embed_common::clap_defaults")]
pub struct TdRbmArgs {
    #[arg(
        value_delimiter = ',',
        help = "Two-track count backends (optional when --from is given)",
        long_help = "Count backends whose rows are `{gene}/count/spliced` and\n\
                     `{gene}/count/unspliced` (as `faba count` writes); optional when\n\
                     --from is given. Several files are read together on the genes\n\
                     they share, every file a batch unless --batch-files says\n\
                     otherwise (--ignore-batch when they are one tissue with no batch\n\
                     effect: the cross-batch collapse makes fewer, coarser pseudobulks).\n\
                     Their cells are grouped into multilevel pseudobulks;\n\
                     the machine is fitted on samples of them and then reads every\n\
                     cell."
    )]
    pub(crate) data_files: Vec<Box<str>>,

    #[arg(
        long,
        help = "Chain data, batch and cell→pb partition from a prior run"
    )]
    pub(crate) from: Option<Box<str>>,

    #[arg(short, long, required = true, help = "Output prefix")]
    pub(crate) out: Box<str>,

    #[arg(
        long,
        short,
        value_delimiter(','),
        help = "Batch membership files, one per data file"
    )]
    pub(crate) batch_files: Option<Vec<Box<str>>>,

    #[command(flatten)]
    pub(crate) collapse: crate::refine_weighting::CollapseArgs,

    #[command(flatten)]
    #[serde(flatten)]
    pub(crate) pb_from: crate::refine_weighting::PbFromArgs,

    #[arg(
        long,
        default_value_t = 2000,
        help = "Most variable genes: they start the fit and call the roots and directions (every gene is fitted)"
    )]
    pub max_genes: usize,

    #[arg(
        long,
        default_value_t = 20.0,
        help = "Fewest reads a starting gene (--max-genes) has on each track, over all cells"
    )]
    pub min_reads: f32,

    #[arg(
        long,
        default_value_t = 32,
        value_parser = clap::value_parser!(u64).range(2..),
        help = "Grid times over (0, 1): the time unit's states"
    )]
    pub grid: u64,

    #[arg(
        long,
        default_value_t = 8,
        value_parser = clap::value_parser!(u64).range(4..),
        help = "Basis functions of the trajectory curve (cubic B-splines)"
    )]
    pub knots: u64,

    #[arg(
        long,
        default_value_t = 12,
        help = "Clusters of the finest pseudobulks the lineage forest is built on (its root-to-leaf paths are the lineages)"
    )]
    pub clusters: usize,

    #[arg(
        long,
        value_name = "FILE",
        help = "Cells (one barcode a line) where time starts: their cluster roots its tree, instead of the splicing choosing",
        long_help = "Cells (one barcode a line) where time starts. The cluster holding\n\
                     most of them is its tree's root; the other trees' roots are chosen\n\
                     from the splicing. A barcode matches without its `@` tags when it names one cell."
    )]
    pub root: Option<String>,

    #[arg(
        long,
        default_value_t = 1.0,
        help = "Dirichlet-process concentration of the lineages (higher: more lineages)"
    )]
    pub lineage_concentration: f64,

    #[arg(
        long,
        default_value_t = 40,
        help = "Epochs first fitting every lineage at steady state (blind to direction)"
    )]
    pub steady_epochs: usize,

    #[arg(
        long,
        default_value_t = 16,
        help = "Draws of every pseudobulk's time behind each lineage's direction (its posterior averages over them)"
    )]
    pub direction_draws: usize,

    #[arg(
        long,
        default_value_t = 300.0,
        help = "Reads a pseudobulk sample counts as when its lineage is chosen (lower: lineages merge more)"
    )]
    pub assign_depth: f64,

    #[arg(
        long,
        default_value_t = 8,
        help = "Width of a cell's state beside time"
    )]
    pub states: usize,

    #[arg(
        long,
        default_value_t = 16,
        help = "Width of the gene embedding (a given feature embedding's own width instead)"
    )]
    pub embedding_dim: usize,

    #[arg(
        long,
        short = 'i',
        default_value_t = 80,
        help = "Epochs over every level"
    )]
    pub epochs: usize,

    #[arg(
        long,
        default_value_t = 40,
        help = "Epochs first fitting lineages and time alone, before the states join"
    )]
    pub time_epochs: usize,

    #[arg(
        long,
        default_value_t = 128,
        alias = "batch",
        help = "Pseudobulk samples per minibatch"
    )]
    pub minibatch_size: usize,

    #[arg(
        long,
        default_value_t = 128,
        help = "Persistent Gibbs chains (about a minibatch)"
    )]
    pub chains: usize,

    #[arg(
        long,
        default_value_t = 1,
        alias = "sweeps",
        help = "Gibbs sweeps per training step"
    )]
    pub gibbs_sweeps: usize,

    #[arg(
        long,
        default_value_t = 1e-2,
        alias = "lr",
        help = "Adam learning rate"
    )]
    pub learning_rate: f64,

    #[arg(
        long,
        default_value_t = 10,
        help = "Neighbours per pseudobulk for the starting order"
    )]
    pub knn: usize,

    #[arg(
        long,
        default_value_t = 1e4,
        help = "Prior weight on every curve's roughness (squared second differences of its coefficients), in nats"
    )]
    pub smooth: f64,

    #[arg(
        long,
        default_value_t = 1000,
        help = "Reads of every pseudobulk sample, chain and (scaled) cell"
    )]
    pub depth: u64,

    #[arg(
        long,
        default_value_t = 1000,
        help = "Reads of every sample and chain in the states phase (smaller tempers the time unit, so chains mix)"
    )]
    pub state_depth: u64,

    #[arg(
        long,
        value_name = "SEED",
        help = "Negative control: split every cell's reads of a gene at the gene's overall unspliced share",
        long_help = "Negative control: before the fit, every cell's reads of a gene are\n\
                     split anew (seeded) between unspliced and spliced at the gene's\n\
                     overall unspliced share: no splicing dynamics, the same either way\n\
                     in time. Every lineage's direction should come out undetermined.\n\
                     (Shuffling the unspliced reads across cells is not such a control:\n\
                     a flat unspliced level makes the spliced relax towards it, one way.)"
    )]
    pub steady_unspliced: Option<u64>,

    #[arg(
        long,
        default_value_t = 1,
        value_name = "N",
        help = "Seed for every random choice the fit makes"
    )]
    pub seed: u64,

    #[arg(long, default_value_t = ComputeDevice::Cpu, value_enum, help = "Compute device")]
    pub device: ComputeDevice,

    #[arg(long, default_value_t = 0, help = "CUDA/Metal device index")]
    pub device_no: usize,

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
    pub(crate) feature_embedding: crate::feature_embedding_args::FeatureEmbeddingArgs,

    #[command(flatten)]
    pub(crate) modules: graph_embedding_util::FeatureModuleArgs,

    #[arg(
        long,
        default_value_t = false,
        conflicts_with = "feature_modules",
        help = "Free per-gene rows of ρ instead of gene modules",
        long_help = "Gene modules are on by default (M = 50): every gene's row of ρ is\n\
                     its module's vector plus a small residual of its own, so a gene with\n\
                     few reads sits near its module. This trains one free row per gene\n\
                     instead. Modules are also off when a feature embedding is given\n\
                     (--freeze/--init/--lora-feature-embedding)."
    )]
    pub(crate) no_feature_modules: bool,

    #[command(flatten)]
    pub(crate) hvg: crate::hvg::HvgCliArgs,

    #[command(flatten)]
    pub(crate) qc: QcArgs,
}
