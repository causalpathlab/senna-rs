//! `senna td-rbm`'s command-line surface.

use senna::embed_common::*;

#[derive(Args, Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct TdRbmArgs {
    #[arg(
        required = true,
        num_args = 1..,
        help = "Count backends with {gene}/count/spliced and {gene}/count/unspliced rows",
        long_help = "Count backends whose rows are `{gene}/count/spliced` and\n\
                     `{gene}/count/unspliced` (as `faba count` writes). Their cells are\n\
                     read together, on the genes every file has, and grouped into\n\
                     multilevel pseudobulks (every file a batch); the machine is\n\
                     fitted on samples of them and then reads every cell. With\n\
                     several files, every barcode is tagged `@{file}`."
    )]
    #[serde(deserialize_with = "one_or_many")]
    pub tracks: Vec<Box<str>>,

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
                     from the splicing. A barcode matches with or without its `@` tag."
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

    #[arg(
        long,
        default_value_t = 128,
        help = "Persistent Gibbs chains (about a minibatch)"
    )]
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

    #[arg(long, default_value_t = 1, help = "Seed of the chains and the start")]
    pub seed: u64,

    #[arg(long, default_value_t = ComputeDevice::Cpu, value_enum, help = "Compute device")]
    pub device: ComputeDevice,

    #[arg(long, default_value_t = 0, help = "Device number")]
    pub device_no: usize,
}

/// A file list, or the one file a run recorded before td-rbm took several.
fn one_or_many<'de, D: serde::Deserializer<'de>>(d: D) -> Result<Vec<Box<str>>, D::Error> {
    #[derive(serde::Deserialize)]
    #[serde(untagged)]
    enum Files {
        One(Box<str>),
        Many(Vec<Box<str>>),
    }
    Ok(match serde::Deserialize::deserialize(d)? {
        Files::One(f) => vec![f],
        Files::Many(v) => v,
    })
}

#[cfg(test)]
#[path = "tests/args.rs"]
mod tests;
