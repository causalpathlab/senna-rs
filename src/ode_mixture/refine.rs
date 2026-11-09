//! Every gene on its own, given the paths, the latents and the depths
//! (the genes are then independent).
//!
//! A run fitted on groups of genes ([`super::coarsen`]) is uncoarsened one
//! draw at a time: at every snapshot every gene starts at its group's rates
//! and embedding and its share of the group's baseline, its proposal is
//! fitted from there ([`gene_mode`]) and it takes a few moves from that
//! mode ([`gene_step`]). The paths and latents stay the groups' draws.
//! With no chains, every gene's rates are read at the climb's mode with the
//! Laplace approximation's sds ([`genes_at_mode`]).

use super::blocks::{from_theta, gene_mode, gene_step, stream, to_theta, GenePrior};
use super::coarsen::gene_params;
use super::likelihood::{GeneRows, Reads, RowLayout};
use super::readout::Snapshot;
use data_beans::alg::feature_coarsening::FeatureCoarsening;
use rayon::prelude::*;

/// Every snapshot's draw of every gene's rates, `[S][G]`.
pub struct GeneDraws {
    pub log_gamma: Vec<Vec<f64>>,
    pub log_ratio: Vec<Vec<f64>>,
}

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

/// Every gene drawn at every snapshot of a run on `fc`'s groups, `moves`
/// moves from its own mode, every gene and snapshot its own stream of
/// `seed`.
pub fn refine_genes(
    reads: &Reads,
    fc: &FeatureCoarsening,
    snaps: &[Snapshot],
    moves: usize,
    seed: u64,
) -> GeneDraws {
    let g = fc.fine_to_coarse.len();
    let (mut log_gamma, mut log_ratio) = (Vec::new(), Vec::new());
    for (s, snap) in snaps.iter().enumerate() {
        let params = gene_params(fc, &snap.params, reads);
        let prior = GenePrior::bounded(&params);
        let z_bar = params.path_mean();
        let layout = RowLayout::new(&snap.rows, &snap.lambda);
        let rates: Vec<(f64, f64)> = (0..g)
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
                let mut rng = stream(seed, i as u64, s as u64);
                let (mut theta, mut ll) = (prop.mode.clone(), f64::NEG_INFINITY);
                for _ in 0..moves {
                    (theta, ll) = gene_step(
                        &params, &theta, ll, &by_row, &z_bar, &prior, &prop, &mut rng,
                    );
                }
                let (beta, gamma, _, _) = from_theta(&theta, &z_bar);
                (gamma.ln(), (gamma / beta).ln())
            })
            .collect();
        let (y, r) = rates.into_iter().unzip();
        log_gamma.push(y);
        log_ratio.push(r);
    }
    GeneDraws {
        log_gamma,
        log_ratio,
    }
}
