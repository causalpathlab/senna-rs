//! Every gene drawn on its own from a coarse run ([`super::coarsen`]).
//!
//! Given the paths, the latents and the depths, the genes are independent,
//! so a run fitted on groups of genes is uncoarsened one draw at a time:
//! at every snapshot every gene starts at its group's rates and embedding
//! and its share of the group's baseline, its proposal is fitted from
//! there ([`gene_mode`]) and it takes a few moves from that mode
//! ([`gene_step`]). The paths and latents stay the groups' draws.

use super::blocks::{from_theta, gene_mode, gene_step, stream, to_theta, GenePrior};
use super::coarsen::uncoarsen_params;
use super::likelihood::{GeneRows, Reads, RowLayout};
use super::readout::Snapshot;
use data_beans::alg::feature_coarsening::FeatureCoarsening;
use rayon::prelude::*;

/// Every snapshot's draw of every gene's rates, `[S][G]`.
pub struct GeneDraws {
    pub log_gamma: Vec<Vec<f64>>,
    pub log_ratio: Vec<Vec<f64>>,
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
    let each = FeatureCoarsening {
        fine_to_coarse: (0..g).collect(),
        coarse_to_fine: (0..g).map(|i| vec![i]).collect(),
        num_coarse: g,
    };
    let (mut log_gamma, mut log_ratio) = (Vec::new(), Vec::new());
    for (s, snap) in snaps.iter().enumerate() {
        let params = uncoarsen_params(fc, &each, &snap.params, reads);
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
