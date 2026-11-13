//! The map the cells and their velocities are drawn on.

use super::readout::Mode;
use legume_numeric::matrix::traits::RandomizedAlgs;
use nalgebra::{DMatrix, DVector};

/// The map space: `log1p` reads per 10⁴ (both tracks summed), centred, on
/// the leading principal components of the rows it is built from, the
/// fitted trajectories' in a run ([`MapBasis::of_trajectories`]).
pub struct MapBasis {
    /// `[G]` the rows' mean expression.
    mean: Vec<f64>,
    /// `[G × k]` the components.
    v: DMatrix<f64>,
}

/// A row's total to divide by: infinite for an empty row, so its shares
/// are zero, not NaN. Any other scale divides out (counts, shares, levels).
fn total_of(reads: &[f64]) -> f64 {
    match reads.iter().sum::<f64>() {
        t if t > 0.0 => t,
        _ => f64::INFINITY,
    }
}

/// `log1p` reads per 10⁴ of a row of reads.
fn expression(reads: &[f64]) -> Vec<f64> {
    let total = total_of(reads);
    reads.iter().map(|x| (x / total * 1e4).ln_1p()).collect()
}

impl MapBasis {
    /// On the leading `k` components of rows `x` (`[P × G]` reads).
    #[cfg(test)]
    pub fn new(x: &DMatrix<f64>, k: usize) -> anyhow::Result<Self> {
        Self::weighted(x, &vec![1.0; x.nrows()], k)
    }

    /// As [`MapBasis::new`], row `i` of weight `w[i]` in the mean and the
    /// components.
    fn weighted(x: &DMatrix<f64>, w: &[f64], k: usize) -> anyhow::Result<Self> {
        let (p, g) = x.shape();
        let total = w.iter().sum::<f64>();
        anyhow::ensure!(total > 0.0, "no row of any weight to build the map on");
        let mut m = DMatrix::<f64>::zeros(p, g);
        for (i, row) in x.row_iter().enumerate() {
            let e = expression(&row.iter().copied().collect::<Vec<_>>());
            m.row_mut(i).copy_from_slice(&e);
        }
        let mean = m.tr_mul(&DVector::from_column_slice(w)) / total;
        for (i, mut row) in m.row_iter_mut().enumerate() {
            row -= mean.transpose();
            row *= (w[i] / total).sqrt();
        }
        let k = k.min(p.saturating_sub(1)).min(g).max(1);
        let (_, _, v) = m.rsvd(k)?;
        Ok(Self {
            mean: mean.iter().copied().collect(),
            v,
        })
    }

    /// On the leading `k` components of the trajectories themselves: every
    /// lineage's levels (both tracks summed) at every position as shares at
    /// `mode`, a lineage weighted by its weight. The axes are the directions
    /// the dynamics move along, not whatever else varies most among the
    /// reads. A position whose levels are not finite has no weight.
    pub fn of_trajectories(mode: &Mode, k: usize) -> anyhow::Result<Self> {
        let p = &mode.params;
        let positions = p.positions();
        let rows = p.path.len() * positions;
        let g = p.beta.len();
        let lv = p.levels();
        let totals: Vec<f64> = lv.totals().iter().map(|(u, s)| u + s).collect();
        let fine: Vec<bool> = totals.iter().map(|t| t.is_finite() && *t > 0.0).collect();
        let shares = DMatrix::from_fn(rows, g, |r, j| {
            if fine[r] {
                (lv.u[(r, j)] + lv.s[(r, j)]) / totals[r]
            } else {
                0.0
            }
        });
        let weight: Vec<f64> = (0..rows)
            .map(|r| {
                if fine[r] {
                    mode.log_w[r / positions].exp()
                } else {
                    0.0
                }
            })
            .collect();
        Self::weighted(&shares, &weight, k)
    }

    pub fn dim(&self) -> usize {
        self.v.ncols()
    }

    /// A row of reads' (`[G]`) place on the components.
    pub fn place(&self, reads: &[f64]) -> Vec<f64> {
        let e: Vec<f64> = expression(reads)
            .iter()
            .zip(&self.mean)
            .map(|(a, b)| a - b)
            .collect();
        self.project(&e)
    }

    /// The rate the place of `reads` moves at when every gene's level
    /// changes at relative rate `dlog` (`d log x_g/dt`): the chain rule
    /// through `log1p(10⁴ x_g/N)`, `N` the row's total.
    pub fn velocity(&self, reads: &[f64], dlog: &[f64]) -> Vec<f64> {
        let total = total_of(reads);
        let dlog_n = reads.iter().zip(dlog).map(|(x, v)| x * v).sum::<f64>() / total;
        let de: Vec<f64> = reads
            .iter()
            .zip(dlog)
            .map(|(x, v)| {
                let y = x / total * 1e4;
                y / (1.0 + y) * (v - dlog_n)
            })
            .collect();
        self.project(&de)
    }

    fn project(&self, e: &[f64]) -> Vec<f64> {
        self.v
            .tr_mul(&nalgebra::DVector::from_column_slice(e))
            .iter()
            .copied()
            .collect()
    }
}
