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
