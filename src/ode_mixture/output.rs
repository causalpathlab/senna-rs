//! What a run reports beyond the cells' readout: every coordinate's
//! summary over the chains, and the map the cells and their velocities
//! are drawn on.

use legume_numeric::matrix::traits::RandomizedAlgs;
use legume_numeric::mcmc::engine::diagnostics::{ess, split_rhat_chains};
use nalgebra::DMatrix;

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
/// the leading principal components of the rows it is built from.
pub struct MapBasis {
    /// `[G]` the rows' mean expression.
    mean: Vec<f64>,
    /// `[G × k]` the components.
    v: DMatrix<f64>,
}

/// `log1p` reads per 10⁴ of a row of reads.
fn expression(reads: &[f64]) -> Vec<f64> {
    let total = reads.iter().sum::<f64>().max(1.0);
    reads.iter().map(|x| (x / total * 1e4).ln_1p()).collect()
}

impl MapBasis {
    /// On the leading `k` components of rows `x` (`[P × G]` reads).
    pub fn new(x: &DMatrix<f64>, k: usize) -> anyhow::Result<Self> {
        let (p, g) = x.shape();
        let mut m = DMatrix::<f64>::zeros(p, g);
        for (i, row) in x.row_iter().enumerate() {
            let e = expression(&row.iter().copied().collect::<Vec<_>>());
            m.row_mut(i).copy_from_slice(&e);
        }
        let mean: Vec<f64> = (0..g).map(|j| m.column(j).mean()).collect();
        for (j, &mu) in mean.iter().enumerate() {
            m.column_mut(j).add_scalar_mut(-mu);
        }
        let k = k.min(p.saturating_sub(1)).min(g).max(1);
        let (_, _, v) = m.rsvd(k)?;
        Ok(Self { mean, v })
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
        let total = reads.iter().sum::<f64>().max(1.0);
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
