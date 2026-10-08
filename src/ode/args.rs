//! `senna ode`'s command-line surface.

use senna::embed_common::*;

#[derive(Args, Debug, Clone)]
pub struct OdeArgs {
    #[arg(
        help = "Count backend with {gene}/count/spliced and {gene}/count/unspliced rows",
        long_help = "Count backend whose rows are `{gene}/count/spliced` and\n\
                     `{gene}/count/unspliced` (as `faba count` writes), over the cells\n\
                     of the base fit."
    )]
    pub tracks: Box<str>,

    #[arg(
        long,
        help = "Prefix of a finished `senna bge` run (the base fit)",
        long_help = "Prefix of a finished `senna bge` run. Read: cell_embedding, the\n\
                     cells' states, which group them into pseudobulks and give the\n\
                     starting order."
    )]
    pub base: Box<str>,

    #[arg(short, long, help = "Output prefix")]
    pub out: Box<str>,

    #[arg(
        long,
        default_value_t = 3,
        help = "Pseudobulk levels, fitted coarse to fine"
    )]
    pub levels: usize,

    #[arg(
        long,
        default_value_t = 10.0,
        help = "Pull λ_g between neighbouring pseudobulks' times (0 = off)",
        long_help = "λ_g of the prior λ_g/2 Σ w_pq (τ_p − τ_q)² over the pseudobulks'\n\
                     neighbour graph, in nats beside the reads' likelihood: neighbours\n\
                     share their evidence. 0 turns it off."
    )]
    pub smooth: f64,

    #[arg(
        long,
        default_value_t = 50.0,
        help = "Pull λ_a toward the parent pseudobulk's time (0 = off)",
        long_help = "λ_a of the prior λ_a/2 (τ_p − τ_parent)² on every level below the\n\
                     coarsest, in nats beside the reads' likelihood. 0 turns it off."
    )]
    pub parent_pull: f64,

    #[arg(
        long,
        default_value_t = 32,
        help = "Gene modules sharing one set of kinetics"
    )]
    pub modules: usize,

    #[arg(
        long,
        default_value_t = 20,
        help = "Fewest genes a module keeps; genes of smaller ones are left out"
    )]
    pub min_module_genes: usize,

    #[arg(long, default_value_t = 2000, help = "Most variable genes fitted")]
    pub max_genes: usize,

    #[arg(
        long,
        default_value_t = 20.0,
        help = "Fewest reads a fitted gene has on each track, over all pseudobulks"
    )]
    pub min_reads: f32,

    #[arg(
        long,
        default_value_t = 30,
        help = "Rounds of τ grid search and Adam steps"
    )]
    pub rounds: usize,

    #[arg(long, default_value_t = 50, help = "Adam steps per round")]
    pub adam_steps: usize,

    #[arg(long, default_value_t = 0.01)]
    pub learning_rate: f64,

    #[arg(long, default_value_t = 64, help = "Grid points over τ")]
    pub grid: usize,

    #[arg(
        long,
        default_value_t = 15,
        help = "Neighbours in the graph the starting order comes from"
    )]
    pub knn: usize,

    #[arg(long, default_value_t = 42)]
    pub seed: u64,

    #[arg(long, default_value_t = ComputeDevice::Cpu, value_enum, help = "Compute device")]
    pub device: ComputeDevice,

    #[arg(long, default_value_t = 0)]
    pub device_no: usize,
}
