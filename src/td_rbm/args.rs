//! `senna td-rbm`'s command-line surface.

use senna::embed_common::*;

#[derive(Args, Debug, Clone)]
pub struct TdRbmArgs {
    #[arg(
        help = "Count backend with {gene}/count/spliced and {gene}/count/unspliced rows",
        long_help = "Count backend whose rows are `{gene}/count/spliced` and\n\
                     `{gene}/count/unspliced` (as `faba count` writes). Its cells are\n\
                     grouped into multilevel pseudobulks; the machine is fitted on\n\
                     samples of them and then reads every cell."
    )]
    pub tracks: Box<str>,

    #[arg(short, long, help = "Output prefix")]
    pub out: Box<str>,

    #[arg(
        long,
        default_value_t = 3,
        value_parser = clap::value_parser!(u64).range(1..),
        help = "Pseudobulk levels of the collapse, read coarse to fine every epoch"
    )]
    pub levels: u64,

    #[arg(long, default_value_t = 2000, help = "Most variable genes fitted")]
    pub max_genes: usize,

    #[arg(
        long,
        default_value_t = 20.0,
        help = "Fewest reads a fitted gene has on each track, over all cells"
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
        default_value_t = 4,
        help = "Most lineages (independent timelines); the data decide how many are used"
    )]
    pub max_lineages: usize,

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
        default_value_t = 20,
        help = "Epochs fitting each lineage both ways round, to call its direction"
    )]
    pub orient_epochs: usize,

    #[arg(
        long,
        default_value_t = 40,
        help = "Bootstrap resamples of each lineage's pseudobulks behind its direction call"
    )]
    pub orient_draws: usize,

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

    #[arg(long, default_value_t = 16, help = "Width of the gene embedding")]
    pub embedding_dim: usize,

    #[arg(long, default_value_t = 80, help = "Epochs over every level")]
    pub epochs: usize,

    #[arg(
        long,
        default_value_t = 40,
        help = "Epochs first fitting lineages and time alone, before the states join"
    )]
    pub time_epochs: usize,

    #[arg(long, default_value_t = 128, help = "Pseudobulk samples per minibatch")]
    pub batch: usize,

    #[arg(long, default_value_t = 256, help = "Persistent Gibbs chains")]
    pub chains: usize,

    #[arg(long, default_value_t = 1, help = "Gibbs sweeps per step")]
    pub sweeps: usize,

    #[arg(long, default_value_t = 1e-2, help = "Learning rate")]
    pub lr: f64,

    #[arg(
        long,
        default_value_t = 10,
        help = "Neighbours per pseudobulk for the starting order"
    )]
    pub knn: usize,

    #[arg(
        long,
        default_value_t = 1000,
        help = "Reads of every pseudobulk sample, chain and (scaled) cell"
    )]
    pub depth: u64,

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

    #[arg(long, default_value_t = 1, help = "Seed of the chains and the start")]
    pub seed: u64,

    #[arg(long, default_value_t = ComputeDevice::Cpu, value_enum, help = "Compute device")]
    pub device: ComputeDevice,

    #[arg(long, default_value_t = 0, help = "Device number")]
    pub device_no: usize,
}
