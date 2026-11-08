//! The ODE on a feature coarsening ([`FeatureCoarsening`]): every group of
//! genes as one gene.
//!
//! Genes that share rates and embedding `(β, γ, ρ)` and differ in baseline
//! alone have levels that are one curve scaled by `e^{b_g}`: the ODE is
//! linear in the transcription. Their reads summed are Poisson at the
//! summed levels, the group's at baseline `b_K = log Σ_{g∈K} e^{b_g}`, and
//! what is left of the genes' reads is a multinomial split by the shares
//! `e^{b_g − b_K}`, free of the kinetics, paths and times. Everything but
//! the shares is fitted on the groups' reads exactly.

use super::likelihood::Reads;
use super::model::Params;
use data_beans::alg::feature_coarsening::FeatureCoarsening;
use nalgebra::DMatrix;

/// Every group's reads, its genes' summed, `[P × K]` per track.
pub fn coarse_reads(fc: &FeatureCoarsening, reads: &Reads) -> Reads {
    let sum = |m: &DMatrix<f64>| {
        DMatrix::from_fn(m.nrows(), fc.num_coarse, |p, k| {
            fc.coarse_to_fine[k].iter().map(|&g| m[(p, g)]).sum()
        })
    };
    Reads::from_matrices(sum(&reads.u), sum(&reads.s))
}

/// Every group as one gene: its first gene's rates and embedding, its
/// baseline `log Σ_{g∈K} e^{b_g}`.
pub fn coarse_params(fc: &FeatureCoarsening, params: &Params) -> Params {
    let first: Vec<usize> = fc.coarse_to_fine.iter().map(|genes| genes[0]).collect();
    let base = fc
        .coarse_to_fine
        .iter()
        .map(|genes| {
            let top = genes
                .iter()
                .map(|&g| params.base[g])
                .fold(f64::NEG_INFINITY, f64::max);
            top + genes
                .iter()
                .map(|&g| (params.base[g] - top).exp())
                .sum::<f64>()
                .ln()
        })
        .collect();
    Params {
        beta: first.iter().map(|&g| params.beta[g]).collect(),
        gamma: first.iter().map(|&g| params.gamma[g]).collect(),
        rho: first.iter().map(|&g| params.rho[g].clone()).collect(),
        base,
        ..params.clone()
    }
}

/// Reads added to every fine group's total before its share is taken, so a
/// group with none keeps a finite baseline.
const SHARE_PSEUDO: f64 = 0.5;

/// A coarse state on the finer coarsening `fine` nested in `coarse` (every
/// fine group's genes in one coarse group): every fine group on its
/// parent's rates and embedding, the parent's baseline split by the fine
/// groups' reads, `b_j = b_k + log(R_j/R_k)` (`R` the reads on both
/// tracks, [`SHARE_PSEUDO`] added to each `R_j`). The children's levels sum
/// to the parent's.
pub fn uncoarsen_params(
    coarse: &FeatureCoarsening,
    fine: &FeatureCoarsening,
    params: &Params,
    reads: &Reads,
) -> Params {
    let parent: Vec<usize> = fine
        .coarse_to_fine
        .iter()
        .map(|genes| coarse.fine_to_coarse[genes[0]])
        .collect();
    let reads_of = |j: usize| -> f64 {
        let total = |g: usize| reads.u.column(g).sum() + reads.s.column(g).sum();
        SHARE_PSEUDO
            + fine.coarse_to_fine[j]
                .iter()
                .map(|&g| total(g))
                .sum::<f64>()
    };
    let r: Vec<f64> = (0..fine.num_coarse).map(reads_of).collect();
    let mut r_parent = vec![0.0; coarse.num_coarse];
    for (j, &k) in parent.iter().enumerate() {
        r_parent[k] += r[j];
    }
    Params {
        beta: parent.iter().map(|&k| params.beta[k]).collect(),
        gamma: parent.iter().map(|&k| params.gamma[k]).collect(),
        rho: parent.iter().map(|&k| params.rho[k].clone()).collect(),
        base: parent
            .iter()
            .enumerate()
            .map(|(j, &k)| params.base[k] + (r[j] / r_parent[k]).ln())
            .collect(),
        ..params.clone()
    }
}
