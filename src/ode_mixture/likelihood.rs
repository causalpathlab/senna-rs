//! The reads' log-likelihood on the grid.
//!
//! A pseudobulk's reads are Poisson at `λ_p ū`, `λ_p s̄`; with `λ_p` at its
//! best, `N_p / Z` (`N_p` its reads, `Z = Σ_g(ū + s̄)` at its grid point),
//! that is the multinomial over the `2G` features:
//!
//! ```text
//! log p(x_p | ℓ, k) = Σ_g u_pg log(ū_gℓk/Z_ℓk) + s_pg log(s̄_gℓk/Z_ℓk)
//! ```
//!
//! (up to the multinomial coefficient), every `(ℓ, k)` at once by two
//! matrix products.
//!
//! A pseudobulk's unspliced share may jitter on its own: its unspliced
//! levels scaled by `e^{ω_p}`, `ω_p ~ N(0, σ²)`. Integrated out, that adds
//! [`share_term`] to its log-likelihood at every point; `σ = 0` is the
//! shared model. Jitter from pseudobulk to pseudobulk is told apart from
//! kinetics because the levels are smooth in time and the jitter is not;
//! a drift smooth in time is not.

use super::model::{clamped_exp, Levels, LOG_CLAMP};
use nalgebra::DMatrix;
use rayon::prelude::*;

/// A pseudobulk's depth per track, `[λ^u, λ^s]` (the same with no jitter;
/// `λ^u = e^{ω_p} λ^s` with one).
pub type Depth = [f64; 2];

/// Pseudobulks' reads, `[P × G]` per track, their totals and unspliced totals.
pub struct Reads {
    pub u: DMatrix<f64>,
    pub s: DMatrix<f64>,
    total: Vec<f64>,
    unspliced: Vec<f64>,
}

impl Reads {
    /// From row-major `[P × G]` reads.
    pub fn new(u: &[f64], s: &[f64], p: usize, g: usize) -> Self {
        Self::from_matrices(
            DMatrix::from_row_slice(p, g, u),
            DMatrix::from_row_slice(p, g, s),
        )
    }

    /// From `[P × G]` reads per track.
    pub fn from_matrices(u: DMatrix<f64>, s: DMatrix<f64>) -> Self {
        assert_eq!(u.shape(), s.shape(), "both tracks the same shape");
        let p = u.nrows();
        let unspliced: Vec<f64> = (0..p).map(|i| u.row(i).sum()).collect();
        let total = (0..p).map(|i| unspliced[i] + s.row(i).sum()).collect();
        Self {
            u,
            s,
            total,
            unspliced,
        }
    }

    /// Pseudobulk `p`'s reads, `N_p`.
    pub fn total(&self, p: usize) -> f64 {
        self.total[p]
    }

    /// Every pseudobulk's best depth `λ_p = N_p / Z` at its row `rows[p]`,
    /// on both tracks.
    pub fn depths(&self, rows: &[usize], lv: &Levels) -> Vec<Depth> {
        self.jittered_depths(rows, lv, &vec![0.0; rows.len()])
    }

    /// Every pseudobulk's best depths at its row `rows[p]` with its jitter
    /// `omega[p]`: `λ_p = N_p / (e^{ω_p} Z_u + Z_s)`, `[e^{ω_p} λ_p, λ_p]`.
    pub fn jittered_depths(&self, rows: &[usize], lv: &Levels, omega: &[f64]) -> Vec<Depth> {
        rows.iter()
            .zip(omega)
            .enumerate()
            .map(|(p, (&r, &w))| {
                let e = clamped_exp(w);
                let lambda = self.total[p] / (e * lv.u.row(r).sum() + lv.s.row(r).sum());
                [e * lambda, lambda]
            })
            .collect()
    }
}

/// `[P × L·n]` every pseudobulk's log-likelihood at every grid point.
pub fn table(reads: &Reads, lv: &Levels) -> DMatrix<f64> {
    let z: Vec<f64> = lv.totals().iter().map(|(u, s)| u + s).collect();
    let log_share =
        |m: &DMatrix<f64>| DMatrix::from_fn(m.ncols(), m.nrows(), |g, r| (m[(r, g)] / z[r]).ln());
    &reads.u * log_share(&lv.u) + &reads.s * log_share(&lv.s)
}

/// [`table`] with every pseudobulk's unspliced levels scaled by its jitter
/// `e^{omega[p]}`: the shared table plus
/// `N^u_p ω_p − N_p log((e^{ω_p} Z_u + Z_s)/(Z_u + Z_s))` at every point.
pub fn jittered_table(reads: &Reads, lv: &Levels, omega: &[f64]) -> DMatrix<f64> {
    let mut tab = table(reads, lv);
    let z = lv.totals();
    for (p, &w) in omega.iter().enumerate() {
        let w = w.clamp(-LOG_CLAMP, LOG_CLAMP);
        let (n_u, n, e) = (reads.unspliced[p], reads.total[p], w.exp());
        for (r, &(z_u, z_s)) in z.iter().enumerate() {
            tab[(p, r)] += n_u * w - n * ((e * z_u + z_s) / (z_u + z_s)).ln();
        }
    }
    tab
}

/// [`table`] with every pseudobulk's jitter of sd `sd` integrated out:
/// [`share_term`] added at every point (the climb's marginal; a cell's).
pub fn integrated_table(reads: &Reads, lv: &Levels, sd: f64) -> DMatrix<f64> {
    let mut tab = table(reads, lv);
    let n = tab.nrows();
    tab.as_mut_slice()
        .par_chunks_mut(n)
        .zip(lv.totals())
        .for_each(|(col, (z_u, z_s))| {
            for (p, x) in col.iter_mut().enumerate() {
                *x += share_term(reads.unspliced[p], reads.total[p], z_u, z_s, sd);
            }
        });
    tab
}

/// The rows the pseudobulks take and their depths summed per row,
/// `Λ_r = Σ_{p at r} λ_p` per track: the same for every gene.
pub struct RowLayout {
    /// The rows taken, ascending, and every pseudobulk's place among them.
    rows: Vec<usize>,
    slot: Vec<usize>,
    depth: Vec<[f64; 2]>,
}

impl RowLayout {
    pub fn new(rows: &[usize], lambda: &[Depth]) -> Self {
        let n_rows = rows.iter().max().map_or(0, |r| r + 1);
        let mut at = vec![usize::MAX; n_rows];
        let taken: Vec<usize> = {
            let mut t: Vec<usize> = rows.to_vec();
            t.sort_unstable();
            t.dedup();
            t
        };
        for (i, &r) in taken.iter().enumerate() {
            at[r] = i;
        }
        let slot: Vec<usize> = rows.iter().map(|&r| at[r]).collect();
        let mut depth = vec![[0.0; 2]; taken.len()];
        for (&i, l) in slot.iter().zip(lambda) {
            depth[i][0] += l[0];
            depth[i][1] += l[1];
        }
        Self {
            rows: taken,
            slot,
            depth,
        }
    }
}

/// Gene `g`'s reads summed by row on a [`RowLayout`]: its likelihood at
/// the cost of the rows taken, not the pseudobulks. Pseudobulks at one row
/// share its levels, so with `X_r = Σ_{p at r} x_p` per track it is
/// `Σ_r X_r log ū_r − Λ_r ū_r` (and likewise `s̄`): [`gene_log_likelihood`]
/// up to `Σ_p x_p log λ_p`, constant in the gene's parameters.
pub struct GeneRows<'a> {
    layout: &'a RowLayout,
    reads: Vec<[f64; 2]>,
}

impl<'a> GeneRows<'a> {
    pub fn new(layout: &'a RowLayout, reads: &Reads, g: usize) -> Self {
        let mut x = vec![[0.0; 2]; layout.rows.len()];
        for (p, &i) in layout.slot.iter().enumerate() {
            x[i][0] += reads.u[(p, g)];
            x[i][1] += reads.s[(p, g)];
        }
        Self { layout, reads: x }
    }

    /// The log-likelihood at the gene's levels `(ū, s̄)` by row, up to the
    /// constant ([`GeneRows`]).
    pub fn log_likelihood(&self, (u, s): (&[f64], &[f64])) -> f64 {
        self.layout
            .rows
            .iter()
            .zip(self.reads.iter().zip(&self.layout.depth))
            .map(|(&r, ([xu, xs], [lu, ls]))| {
                xu * u[r].ln() - lu * u[r] + xs * s[r].ln() - ls * s[r]
            })
            .sum()
    }
}

/// Gene `g`'s Poisson log-likelihood (no `log x!`) with every pseudobulk at
/// its row `rows[p]` and depths `lambda[p]`, the gene's levels `(ū, s̄)` by
/// row; one term a pseudobulk, the definition [`GeneRows`] is checked against.
#[cfg(test)]
pub fn gene_log_likelihood(
    reads: &Reads,
    g: usize,
    rows: &[usize],
    lambda: &[Depth],
    (u, s): (&[f64], &[f64]),
) -> f64 {
    rows.iter()
        .enumerate()
        .map(|(p, &r)| {
            let (mu, ms) = (lambda[p][0] * u[r], lambda[p][1] * s[r]);
            reads.u[(p, g)] * mu.ln() - mu + reads.s[(p, g)] * ms.ln() - ms
        })
        .sum()
}

/// Newton steps to the jitter's mode, from its Gaussian guess.
const SHARE_NEWTON: usize = 8;

/// `log(1 + e^x)`, stable.
pub fn softplus(x: f64) -> f64 {
    x.max(0.0) + (-x.abs()).exp().ln_1p()
}

/// The jitter's mode `ω̂` given `N_u` unspliced reads of `N`, the levels'
/// log unspliced-to-spliced ratio `a = log(Z_u/Z_s)` and the jitter's
/// variance `v`: the maximum of
/// `f(ω) = N_u ω − N(softplus(a + ω) − softplus(a)) − ω²/(2v)` (concave).
pub fn share_mode(n_u: f64, n: f64, a: f64, v: f64) -> f64 {
    let q = (n_u + 0.5) / (n + 1.0);
    let h = n * q * (1.0 - q);
    let mut w = ((q / (1.0 - q)).ln() - a) * h * v / (1.0 + h * v);
    for _ in 0..SHARE_NEWTON {
        let q = 1.0 / (1.0 + (-(a + w)).exp());
        w += (n_u - n * q - w / v) / (n * q * (1.0 - q) + 1.0 / v);
    }
    w
}

/// A pseudobulk's log-likelihood gain from its share's jitter integrated
/// out, `log ∫ e^{Δ(ω)} N(ω; 0, σ²) dω` with
/// `Δ(ω) = N_u ω − N log((e^ω Z_u + Z_s)/(Z_u + Z_s))` (the multinomial with
/// the unspliced levels scaled by `e^ω`, minus without), by Laplace at the
/// mode: `f(ω̂) − ½ log(1 + σ² N q̂(1 − q̂))`, `q̂` the jittered unspliced
/// share. `N_u`, `N` its unspliced and all reads, `Z_u`, `Z_s` the levels'
/// totals at its point.
pub fn share_term(n_u: f64, n: f64, z_u: f64, z_s: f64, sd: f64) -> f64 {
    if sd <= 0.0 {
        return 0.0;
    }
    let (a, v) = ((z_u / z_s).ln(), sd * sd);
    let w = share_mode(n_u, n, a, v);
    let q = 1.0 / (1.0 + (-(a + w)).exp());
    n_u * w
        - n * (softplus(a + w) - softplus(a))
        - w * w / (2.0 * v)
        - 0.5 * (v * n * q * (1.0 - q)).ln_1p()
}
