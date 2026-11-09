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
use data_beans::alg::feature_coarsening::{
    partition_features, FeatureCoarsening, PartitionOptions,
};
use nalgebra::DMatrix;

/// Width (nats) of the bins of a gene's spliced-to-unspliced balance
/// `log(S/U)` no group crosses: at steady state it is `log(β/γ)`, so a
/// group's genes share their kinetics to about this.
const KINETIC_BIN: f64 = 0.5;

/// The genes in at most `cap` groups (at least one a kinetic bin;
/// [`partition_features`]), `sizes`
/// every pseudobulk's cells. A group's genes share what the coarse fit
/// takes them to share:
/// - their transcription's timing: k-means on their pseudobulk profiles,
///   both tracks side by side, each track scaled to the gene's mean total
///   (else a slow gene's spliced reads drown its unspliced);
/// - their kinetics: no group crosses a [`KINETIC_BIN`] of `log(S/U)`,
///   the flat genes' background split by bin as well.
pub fn coarsen_genes(
    reads: &Reads,
    sizes: &[f32],
    cap: usize,
    seed: u64,
) -> anyhow::Result<FeatureCoarsening> {
    let (p, g) = reads.u.shape();
    let totals: Vec<(f64, f64)> = (0..g)
        .map(|j| {
            (
                reads.u.column(j).sum().max(1.0),
                reads.s.column(j).sum().max(1.0),
            )
        })
        .collect();
    let counts = DMatrix::<f32>::from_fn(g, 2 * p, |j, c| {
        let (tu, ts) = totals[j];
        let mean = 0.5 * (tu + ts);
        let x = if c < p {
            reads.u[(c, j)] * mean / tu
        } else {
            reads.s[(c - p, j)] * mean / ts
        };
        x as f32
    });
    let block: Vec<u32> = totals
        .iter()
        .map(|&(tu, ts)| ((ts / tu).ln() / KINETIC_BIN).floor() as i64 as u32)
        .collect();
    let opts = PartitionOptions {
        block: Some(block.clone()),
        ..PartitionOptions::default()
    };
    // A slot kept for every bin the background may split into.
    let bins = block
        .iter()
        .collect::<std::collections::BTreeSet<_>>()
        .len();
    let slots = cap.saturating_sub(bins - 1).max(1);
    let part = partition_features(&counts, &[sizes, sizes].concat(), slots, seed, &opts)?;
    // The background is one group across the blocks: split it by block too.
    let mut labels = part.labels;
    if let Some(bg) = part.background {
        let mut by_block = std::collections::BTreeMap::new();
        for (j, label) in labels.iter_mut().enumerate() {
            if *label == bg {
                let next = part.num_groups + by_block.len();
                *label = *by_block.entry(block[j]).or_insert(next);
            }
        }
        // The background's own id is now empty: the last group takes it.
        let last = part.num_groups + by_block.len() - 1;
        for label in labels.iter_mut().filter(|l| **l == last) {
            *label = bg;
        }
        return FeatureCoarsening::from_fine_to_coarse(labels, last);
    }
    FeatureCoarsening::from_fine_to_coarse(labels, part.num_groups)
}

/// Every group's reads, its genes' summed, `[P × K]` per track.
pub fn coarse_reads(fc: &FeatureCoarsening, reads: &Reads) -> Reads {
    // Gene by gene into its group, every column contiguous (the same sum
    // in the same order as a group's genes summed).
    let sum = |m: &DMatrix<f64>| {
        let mut out = DMatrix::<f64>::zeros(m.nrows(), fc.num_coarse);
        for (g, &k) in fc.fine_to_coarse.iter().enumerate() {
            let mut col = out.column_mut(k);
            col += m.column(g);
        }
        out
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

/// Reads added to every gene's total before its share is taken, so a gene
/// with none keeps a finite baseline.
const SHARE_PSEUDO: f64 = 0.5;

/// A state on `fc`'s groups back on the genes: every gene on its group's
/// rates and embedding, the group's baseline split by the genes' reads,
/// `b_g = b_K + log(R_g/R_K)` (`R` the reads on both tracks,
/// [`SHARE_PSEUDO`] added to each `R_g`). A group's genes' levels sum to
/// its own.
pub fn gene_params(fc: &FeatureCoarsening, params: &Params, reads: &Reads) -> Params {
    let r: Vec<f64> = (0..fc.fine_to_coarse.len())
        .map(|g| SHARE_PSEUDO + reads.u.column(g).sum() + reads.s.column(g).sum())
        .collect();
    let mut r_group = vec![0.0; fc.num_coarse];
    for (g, &k) in fc.fine_to_coarse.iter().enumerate() {
        r_group[k] += r[g];
    }
    let k = &fc.fine_to_coarse;
    Params {
        beta: k.iter().map(|&k| params.beta[k]).collect(),
        gamma: k.iter().map(|&k| params.gamma[k]).collect(),
        rho: k.iter().map(|&k| params.rho[k].clone()).collect(),
        base: k
            .iter()
            .enumerate()
            .map(|(g, &k)| params.base[k] + (r[g] / r_group[k]).ln())
            .collect(),
        ..params.clone()
    }
}
