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

use super::model::Levels;
use nalgebra::DMatrix;

/// Pseudobulks' reads, `[P × G]` per track, and their totals.
pub struct Reads {
    pub u: DMatrix<f64>,
    pub s: DMatrix<f64>,
    total: Vec<f64>,
}

impl Reads {
    /// From row-major `[P × G]` reads.
    pub fn new(u: &[f64], s: &[f64], p: usize, g: usize) -> Self {
        let (u, s) = (
            DMatrix::from_row_slice(p, g, u),
            DMatrix::from_row_slice(p, g, s),
        );
        let total = (0..p).map(|i| u.row(i).sum() + s.row(i).sum()).collect();
        Self { u, s, total }
    }

    /// Pseudobulk `p`'s reads, `N_p`.
    pub fn total(&self, p: usize) -> f64 {
        self.total[p]
    }

    /// Every pseudobulk's best depth `λ_p = N_p / Z` at its row `rows[p]`.
    pub fn depths(&self, rows: &[usize], lv: &Levels) -> Vec<f64> {
        rows.iter()
            .enumerate()
            .map(|(p, &r)| self.total[p] / (lv.u.row(r).sum() + lv.s.row(r).sum()))
            .collect()
    }
}

/// `[P × L·n]` every pseudobulk's log-likelihood at every grid point.
pub fn table(reads: &Reads, lv: &Levels) -> DMatrix<f64> {
    let z: Vec<f64> = (0..lv.u.nrows())
        .map(|r| lv.u.row(r).sum() + lv.s.row(r).sum())
        .collect();
    let log_share =
        |m: &DMatrix<f64>| DMatrix::from_fn(m.ncols(), m.nrows(), |g, r| (m[(r, g)] / z[r]).ln());
    &reads.u * log_share(&lv.u) + &reads.s * log_share(&lv.s)
}

/// Gene `g`'s Poisson log-likelihood (no `log x!`) with every pseudobulk at
/// its row `rows[p]` and depth `lambda[p]`.
pub fn gene_log_likelihood(
    reads: &Reads,
    g: usize,
    rows: &[usize],
    lambda: &[f64],
    lv: &Levels,
) -> f64 {
    rows.iter()
        .enumerate()
        .map(|(p, &r)| {
            let (mu, ms) = (lambda[p] * lv.u[(r, g)], lambda[p] * lv.s[(r, g)]);
            reads.u[(p, g)] * mu.ln() - mu + reads.s[(p, g)] * ms.ln() - ms
        })
        .sum()
}
