//! Every gene on its own, given the paths, the latents and the depths
//! (the genes are then independent): its rates at the climb's mode with
//! the sds of the Laplace approximation of its conditional there
//! ([`genes_at_mode`]). A run fitted on groups of genes
//! ([`super::coarsen`]) is uncoarsened the same way: every gene starts at
//! its group's rates and embedding and its share of the group's baseline,
//! and its own mode is found from there ([`gene_mode`]). The paths and
//! latents stay the groups'.

use super::blocks::{gene_mode, to_theta, GenePrior};
use super::coarsen::gene_params;
use super::likelihood::{GeneRows, Reads, RowLayout};
use super::readout::Snapshot;
use data_beans::alg::feature_coarsening::FeatureCoarsening;
use rayon::prelude::*;

/// Every gene's `log γ` and `log γ/β` at a mode, `(mean, sd)` each.
pub struct GeneLaplace {
    pub log_gamma: Vec<(f64, f64)>,
    pub log_ratio: Vec<(f64, f64)>,
}

/// Every gene's rates at `snap` (the climb's mode, its rows and depths)
/// by the Laplace approximation of its conditional ([`gene_mode`]), on the
/// genes of a run on `fc`'s groups or on the run's own.
pub fn genes_at_mode(
    reads: &Reads,
    fc: Option<&FeatureCoarsening>,
    snap: &Snapshot,
) -> GeneLaplace {
    let params = match fc {
        Some(fc) => gene_params(fc, &snap.params, reads),
        None => snap.params.clone(),
    };
    let prior = GenePrior::bounded(&params);
    let z_bar = params.path_mean();
    let layout = RowLayout::new(&snap.rows, &snap.lambda);
    let (log_gamma, log_ratio) = (0..params.beta.len())
        .into_par_iter()
        .map(|i| {
            let by_row = GeneRows::new(&layout, reads, i);
            let start = to_theta(
                params.beta[i],
                params.gamma[i],
                &params.rho[i],
                params.base[i],
                &z_bar,
            );
            let prop = gene_mode(&params, &by_row, &z_bar, &prior, &start);
            let (sd_gamma, sd_ratio) = prop.rate_sds(&prior);
            ((prop.mode[0], sd_gamma), (prop.mode[1], sd_ratio))
        })
        .unzip();
    GeneLaplace {
        log_gamma,
        log_ratio,
    }
}
