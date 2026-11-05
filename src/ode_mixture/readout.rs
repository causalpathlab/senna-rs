//! Every cell read out from the chains' draws.
//!
//! A cell is a pseudobulk of one, so the model is the same: given a draw's
//! parameters, the cell's posterior over every row `(ℓ, k)` is its table
//! entry plus `log π_ℓ` normalised ([`log_posterior_row`]), the jitter of
//! its unspliced share integrated out ([`share_term`]) when the model has
//! one. Over the draws the posteriors are averaged (the parameters' draws
//! integrated out); from that mixture every cell's lineage probabilities
//! and the mean, sd and 90% interval of its time, and its mean velocities.

use super::blocks::log_posterior_row;
use super::likelihood::{integrated_table, table, Reads};
use super::model::Params;
use nalgebra::DMatrix;
use rayon::prelude::*;

/// One draw of what a cell's posterior depends on.
#[derive(Clone, Debug)]
pub struct Snapshot {
    pub params: Params,
    /// `[L]` the lineages' log weights.
    pub log_w: Vec<f64>,
    /// The jitter's sd, `None` with no jitter.
    pub share_sd: Option<f64>,
}

/// Every cell's readout, `[C]` each (`lineage` `[C][L]`).
pub struct CellReadout {
    pub lineage: Vec<Vec<f64>>,
    pub time_mean: Vec<f64>,
    pub time_sd: Vec<f64>,
    /// The 5% and 95% points of the time's posterior.
    pub time_lo: Vec<f64>,
    pub time_hi: Vec<f64>,
    /// `[C × G]` the posterior mean velocities, `dū/dt` and `ds̄/dt`
    /// ([`Params::velocities`]), in the levels' units.
    pub velocity_u: DMatrix<f64>,
    pub velocity_s: DMatrix<f64>,
    /// The time bins' width the intervals sit on: a position of the
    /// shortest lineage, wider only past [`MAX_BINS`] bins.
    pub time_bin: f64,
}

/// One cell's posterior so far, summed over the draws: every lineage's
/// mass, the time's first two moments and its mass on fixed time bins
/// (one a position of the unit duration wide, so memory is cells × bins,
/// not cells × rows × draws).
#[derive(Clone)]
struct Acc {
    /// Some draw's row was not finite throughout: nothing is known.
    unknown: bool,
    lineage: Vec<f64>,
    m1: f64,
    m2: f64,
    bins: Vec<f64>,
}

impl Acc {
    /// The `q` point of the time's posterior, to within half a bin.
    fn quantile(&self, q: f64, width: f64) -> f64 {
        let mut acc = 0.0;
        for (i, &m) in self.bins.iter().enumerate() {
            acc += m;
            if acc >= q {
                return i as f64 * width;
            }
        }
        (self.bins.len() - 1) as f64 * width
    }
}

impl CellReadout {
    /// No draws: nothing known of any cell.
    fn empty(n_cells: usize, genes: usize) -> Self {
        let nan = vec![f64::NAN; n_cells];
        Self {
            lineage: vec![Vec::new(); n_cells],
            time_mean: nan.clone(),
            time_sd: nan.clone(),
            time_lo: nan.clone(),
            time_hi: nan,
            velocity_u: DMatrix::from_element(n_cells, genes, f64::NAN),
            velocity_s: DMatrix::from_element(n_cells, genes, f64::NAN),
            time_bin: f64::NAN,
        }
    }
}

/// At most this many time bins a cell (wider bins past it).
const MAX_BINS: usize = 4096;

/// Every cell of `cells` read out over the draws `snaps` (the same model:
/// lineages and grid, checked). Time bins are one position of the
/// shortest lineage wide (at most [`MAX_BINS`]); a cell some draw cannot
/// place anywhere (its row not finite) is NaN throughout.
pub fn read_cells(cells: &Reads, snaps: &[Snapshot]) -> CellReadout {
    let (n_cells, genes) = (cells.u.nrows(), cells.u.ncols());
    let Some(first) = snaps.first() else {
        return CellReadout::empty(n_cells, genes);
    };
    let (lineages, positions) = (first.params.path.len(), first.params.positions());
    assert!(
        snaps
            .iter()
            .all(|s| s.params.path.len() == lineages && s.params.positions() == positions),
        "every snapshot on the same lineages and grid"
    );
    let weight = 1.0 / snaps.len() as f64;
    let durations = || snaps.iter().flat_map(|s| s.params.t_len.iter().copied());
    let t_max = durations().fold(0.0, f64::max);
    let finest = durations().fold(f64::INFINITY, f64::min) / (positions - 1) as f64;
    let n_bins = ((t_max / finest).ceil() as usize + 1).min(MAX_BINS);
    let width = finest.max(t_max / (n_bins - 1).max(1) as f64);
    let mut acc = vec![
        Acc {
            unknown: false,
            lineage: vec![0.0; lineages],
            m1: 0.0,
            m2: 0.0,
            bins: vec![0.0; n_bins],
        };
        n_cells
    ];
    // `[G × C]`, transposed at the end: `duᵀ · post` is a plain product
    // (threaded), where `postᵀ · du` would not be.
    let (mut vel_u, mut vel_s) = (
        DMatrix::zeros(genes, n_cells),
        DMatrix::zeros(genes, n_cells),
    );
    let rows = lineages * positions;
    // `[rows × cells]`, a cell's posterior one contiguous column, reused.
    let mut post = DMatrix::<f64>::zeros(rows, n_cells);
    for snap in snaps {
        let p = &snap.params;
        let lv = p.levels();
        let tab = match snap.share_sd {
            Some(sd) => integrated_table(cells, &lv, sd),
            None => table(cells, &lv),
        };
        let times: Vec<(usize, f64, usize)> = (0..rows)
            .map(|r| {
                let (l, k) = (r / positions, r % positions);
                let t = p.time_at(l, k);
                (l, t, ((t / width).round() as usize).min(n_bins - 1))
            })
            .collect();
        post.as_mut_slice()
            .par_chunks_mut(rows)
            .zip(acc.par_iter_mut())
            .enumerate()
            .for_each(|(c, (col, a))| {
                let (lp, norm) = log_posterior_row(&tab, &snap.log_w, positions, c);
                if !norm.is_finite() {
                    a.unknown = true;
                    col.fill(0.0);
                    return;
                }
                for ((x, &l), &(lin, t, bin)) in col.iter_mut().zip(&lp).zip(&times) {
                    *x = weight * l.exp();
                    a.lineage[lin] += *x;
                    a.m1 += *x * t;
                    a.m2 += *x * t * t;
                    a.bins[bin] += *x;
                }
            });
        let (du, ds) = p.velocities_of(&lv);
        vel_u.gemm(1.0, &du.transpose(), &post, 1.0);
        vel_s.gemm(1.0, &ds.transpose(), &post, 1.0);
    }
    let nan_rows = |mut m: DMatrix<f64>| {
        for c in (0..n_cells).filter(|&c| acc[c].unknown) {
            m.row_mut(c).fill(f64::NAN);
        }
        m
    };
    let (velocity_u, velocity_s) = (nan_rows(vel_u.transpose()), nan_rows(vel_s.transpose()));
    let pick = |f: &dyn Fn(&Acc) -> f64| -> Vec<f64> {
        acc.iter()
            .map(|a| if a.unknown { f64::NAN } else { f(a) })
            .collect()
    };
    CellReadout {
        time_mean: pick(&|a| a.m1),
        time_sd: pick(&|a| (a.m2 - a.m1 * a.m1).max(0.0).sqrt()),
        time_lo: pick(&|a| a.quantile(0.05, width)),
        time_hi: pick(&|a| a.quantile(0.95, width)),
        lineage: acc
            .iter()
            .map(|a| {
                if a.unknown {
                    vec![f64::NAN; lineages]
                } else {
                    a.lineage.clone()
                }
            })
            .collect(),
        velocity_u,
        velocity_s,
        time_bin: width,
    }
}
