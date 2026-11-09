//! What a run reports beyond the cells' readout: every coordinate's
//! summary over the chains, and the map the cells and their velocities
//! are drawn on.

use super::readout::Snapshot;
use anyhow::Context;
use legume_numeric::matrix::traits::RandomizedAlgs;
use legume_numeric::mcmc::engine::diagnostics::{ess, split_rhat_chains};
use nalgebra::{DMatrix, DVector};

/// A coordinate is flagged as not mixed above this split R̂.
pub const RHAT_FLAG: f64 = 1.05;

/// One coordinate over the chains' kept draws.
#[derive(Clone, Debug)]
pub struct ChainStat {
    pub mean: f64,
    pub sd: f64,
    /// Split R̂ over the chains (NaN with fewer than two draws a chain).
    pub rhat: f64,
    /// The chains' ESS summed.
    pub ess: f64,
}

impl ChainStat {
    /// Not mixed, or not known to be.
    pub fn flagged(&self) -> bool {
        self.rhat.is_nan() || self.rhat > RHAT_FLAG
    }

    /// A coordinate at a mode with its sd, no chains run: R̂ and ESS are
    /// not known.
    pub fn at_mode(mean: f64, sd: f64) -> Self {
        Self {
            mean,
            sd,
            rhat: f64::NAN,
            ess: f64::NAN,
        }
    }
}

/// Every coordinate's summary from `chains[c][s][j]` (chain, kept draw,
/// coordinate); the coordinates the first non-empty chain has.
pub fn chain_stats<C: AsRef<[Vec<f64>]>>(chains: &[C]) -> Vec<ChainStat> {
    let Some(width) = chains.iter().find_map(|c| c.as_ref().first().map(Vec::len)) else {
        return Vec::new();
    };
    let kept: Vec<&[Vec<f64>]> = chains
        .iter()
        .map(AsRef::as_ref)
        .filter(|c| !c.is_empty())
        .collect();
    (0..width)
        .map(|j| {
            let per: Vec<Vec<f32>> = kept
                .iter()
                .map(|c| c.iter().map(|d| d[j] as f32).collect())
                .collect();
            let all: Vec<f64> = kept.iter().flat_map(|c| c.iter().map(|d| d[j])).collect();
            let n = all.len() as f64;
            let mean = all.iter().sum::<f64>() / n;
            let sd = (all.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n).sqrt();
            let enough = per.len() >= 2 && per.iter().all(|c| c.len() >= 4);
            let rhat = if enough {
                f64::from(split_rhat_chains(&per))
            } else {
                f64::NAN
            };
            ChainStat {
                mean,
                sd,
                rhat,
                ess: per.iter().map(|c| f64::from(ess(c))).sum(),
            }
        })
        .collect()
}

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
    /// lineage's levels (both tracks summed) at every position as shares,
    /// averaged over the draws `snaps`, a lineage weighted by its mean
    /// weight. The axes are the directions the dynamics move along, not
    /// whatever else varies most among the reads. A draw whose levels at a
    /// position are not finite is left out of that position's mean.
    pub fn of_trajectories(snaps: &[Snapshot], k: usize) -> anyhow::Result<Self> {
        let first = snaps.first().context("no draw to build the map on")?;
        let positions = first.params.positions();
        let rows = first.params.path.len() * positions;
        let g = first.params.beta.len();
        let mut sum = DMatrix::<f64>::zeros(rows, g);
        let mut n = vec![0usize; rows];
        let mut weight = vec![0.0; rows];
        for snap in snaps {
            let lv = snap.params.levels();
            anyhow::ensure!(
                lv.u.shape() == (rows, g),
                "every draw on the same lineages, grid and genes"
            );
            let totals: Vec<f64> = lv.totals().iter().map(|(u, s)| u + s).collect();
            let fine: Vec<bool> = totals.iter().map(|t| t.is_finite() && *t > 0.0).collect();
            for j in 0..g {
                for r in (0..rows).filter(|&r| fine[r]) {
                    sum[(r, j)] += (lv.u[(r, j)] + lv.s[(r, j)]) / totals[r];
                }
            }
            for r in (0..rows).filter(|&r| fine[r]) {
                n[r] += 1;
                weight[r] += snap.log_w[r / positions].exp();
            }
        }
        for (r, &nr) in n.iter().enumerate().filter(|&(_, &nr)| nr > 0) {
            sum.row_mut(r).scale_mut(1.0 / nr as f64);
            weight[r] /= nr as f64;
        }
        Self::weighted(&sum, &weight, k)
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
