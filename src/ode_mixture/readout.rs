//! Every cell read out at the climb's mode.
//!
//! A cell is a pseudobulk of one, so the model is the same: given the
//! mode's parameters, the cell's posterior over every row `(ℓ, k)` is its
//! table entry plus `log π_ℓ` normalised ([`log_posterior_row`]), the
//! jitter of its unspliced share integrated out ([`share_term`]) when the
//! model has one. From that posterior every cell's lineage probabilities,
//! the mean, sd and 90% interval of its time (its lineage's own phase on
//! `[0, 1]`), and its mean velocities, on each track and of its log level
//! ([`log_rate`]).

use super::blocks::{argmax, log_posterior_row};
use super::likelihood::{integrated_table, share_mode, table, Depth, Reads};
use super::model::{clamped_exp, Params};
use nalgebra::DMatrix;
use rayon::prelude::*;

/// The climb's mode: what every cell's posterior depends on.
#[derive(Clone, Debug)]
pub struct Mode {
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
    /// `[C × G]` the posterior mean of `d log(ū + s̄)/dt`, every gene's
    /// relative rate of change on both tracks.
    pub velocity_log: DMatrix<f64>,
}

/// `d log(ū + s̄)/dt` from the velocities and levels; `0` where the level
/// is not a normal positive number (underflowed: no transcription to speak
/// of), so it never turns the readout's products infinite.
pub fn log_rate(du: f64, ds: f64, u: f64, s: f64) -> f64 {
    let level = u + s;
    let rate = (du + ds) / level;
    if level.is_normal() && level > 0.0 && rate.is_finite() {
        rate
    } else {
        0.0
    }
}

/// Every cell of `cells` read out at `mode`. Time is each lineage's own
/// phase on `[0, 1]` ([`Params::phase_at`]; a lineage's duration is not
/// read into it, so lineages are not put on one clock), its quantiles on
/// the grid's positions; a cell the mode cannot place anywhere (its row
/// not finite) is NaN throughout.
pub fn read_cells(cells: &Reads, mode: &Mode) -> CellReadout {
    let (n_cells, genes) = (cells.u.nrows(), cells.u.ncols());
    let p = &mode.params;
    let (lineages, positions) = (p.path.len(), p.positions());
    let rows = lineages * positions;
    let lv = p.levels();
    // `[rows × cells]`, a cell's posterior one contiguous column.
    let mut post = DMatrix::<f64>::zeros(rows, n_cells);
    fill_posterior(cells, mode, &lv, &mut post);
    let phase: Vec<f64> = (0..positions).map(|k| p.phase_at(k)).collect();
    // Every cell's lineage masses, the time's first two moments and its
    // mass at every position (the quantiles' bins).
    struct Acc {
        unknown: bool,
        lineage: Vec<f64>,
        m1: f64,
        m2: f64,
        bins: Vec<f64>,
    }
    let acc: Vec<Acc> = post
        .as_slice()
        .par_chunks(rows)
        .map(|col| {
            let mut a = Acc {
                unknown: col[0].is_nan(),
                lineage: vec![0.0; lineages],
                m1: 0.0,
                m2: 0.0,
                bins: vec![0.0; positions],
            };
            if !a.unknown {
                for (r, &x) in col.iter().enumerate() {
                    let (l, k) = (r / positions, r % positions);
                    a.lineage[l] += x;
                    a.m1 += x * phase[k];
                    a.m2 += x * phase[k] * phase[k];
                    a.bins[k] += x;
                }
            }
            a
        })
        .collect();
    let quantile = |a: &Acc, q: f64| {
        let mut mass = 0.0;
        for (k, &m) in a.bins.iter().enumerate() {
            mass += m;
            if mass >= q {
                return phase[k];
            }
        }
        1.0
    };
    // `[G × C]`, transposed at the end: `duᵀ · post` is a plain product
    // (threaded), where `postᵀ · du` would not be.
    let (du, ds) = p.velocities_of(&lv);
    let dlog = DMatrix::from_fn(rows, genes, |r, g| {
        log_rate(du[(r, g)], ds[(r, g)], lv.u[(r, g)], lv.s[(r, g)])
    });
    let nan_rows = |m: DMatrix<f64>| {
        let mut m = m.transpose();
        for c in (0..n_cells).filter(|&c| acc[c].unknown) {
            m.row_mut(c).fill(f64::NAN);
        }
        m
    };
    let pick = |f: &dyn Fn(&Acc) -> f64| -> Vec<f64> {
        acc.iter()
            .map(|a| if a.unknown { f64::NAN } else { f(a) })
            .collect()
    };
    CellReadout {
        time_mean: pick(&|a| a.m1),
        time_sd: pick(&|a| (a.m2 - a.m1 * a.m1).max(0.0).sqrt()),
        time_lo: pick(&|a| quantile(a, 0.05)),
        time_hi: pick(&|a| quantile(a, 0.95)),
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
        velocity_u: nan_rows(du.transpose() * &post),
        velocity_s: nan_rows(ds.transpose() * &post),
        velocity_log: nan_rows(dlog.transpose() * &post),
    }
}

/// Every cell's posterior over every row `(ℓ, k)` at `mode` into `post`
/// (`[rows × cells]`, a cell a column); a cell the mode cannot place (its
/// row not finite) is NaN.
fn fill_posterior(cells: &Reads, mode: &Mode, lv: &super::model::Levels, post: &mut DMatrix<f64>) {
    let (rows, positions) = (post.nrows(), mode.params.positions());
    let tab = match mode.share_sd {
        Some(sd) => integrated_table(cells, lv, sd),
        None => table(cells, lv),
    };
    post.as_mut_slice()
        .par_chunks_mut(rows)
        .enumerate()
        .for_each(|(c, col)| {
            let (lp, norm) = log_posterior_row(&tab, &mode.log_w, positions, c);
            if !norm.is_finite() {
                col.fill(f64::NAN);
                return;
            }
            for (x, &l) in col.iter_mut().zip(&lp) {
                *x = l.exp();
            }
        });
}

/// Every cell's posterior over the grid at the mode.
pub struct GridPosterior {
    /// `[C × L positions]`, a cell's row summing to one (NaN where the
    /// mode cannot place it), columns `ℓ positions + k`.
    pub post: DMatrix<f64>,
    /// `[L positions]` every column's phase ([`Params::phase_at`]).
    pub phase: Vec<f64>,
    pub positions: usize,
}

/// Every cell of `cells` on the grid at `mode`.
pub fn grid_posterior(cells: &Reads, mode: &Mode) -> GridPosterior {
    let p = &mode.params;
    let positions = p.positions();
    let rows = p.path.len() * positions;
    let mut post = DMatrix::<f64>::zeros(rows, cells.u.nrows());
    fill_posterior(cells, mode, &p.levels(), &mut post);
    GridPosterior {
        post: post.transpose(),
        phase: (0..rows).map(|r| p.phase_at(r % positions)).collect(),
        positions,
    }
}

/// Every pseudobulk placed at the mode: its most probable row (`None`
/// where the mode cannot place it) and its depths there, the jitter of its
/// unspliced share at its own mode given that row.
pub struct Placed {
    pub rows: Vec<Option<usize>>,
    pub lambda: Vec<Depth>,
}

/// Every pseudobulk of `reads` placed from its posterior `grid` at `mode`
/// ([`grid_posterior`] of the same reads): the gene pass's rows and depths.
pub fn place(reads: &Reads, mode: &Mode, grid: &GridPosterior) -> Placed {
    let lv = mode.params.levels();
    let z = lv.totals();
    let (rows, lambda): (Vec<Option<usize>>, Vec<Depth>) = (0..reads.u.nrows())
        .map(|p| {
            let Some(r) = argmax(grid.post.row(p).iter().copied()) else {
                return (None, [0.0, 0.0]);
            };
            let (z_u, z_s) = z[r];
            let omega = mode.share_sd.map_or(0.0, |sd| {
                share_mode(
                    reads.unspliced(p),
                    reads.total(p),
                    (z_u / z_s).ln(),
                    sd * sd,
                )
            });
            let e = clamped_exp(omega);
            let depth = reads.total(p) / (e * z_u + z_s);
            (Some(r), [e * depth, depth])
        })
        .unzip();
    Placed { rows, lambda }
}
