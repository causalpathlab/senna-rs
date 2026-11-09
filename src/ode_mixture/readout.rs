//! Every cell read out at the climb's mode.
//!
//! A cell is a pseudobulk of one, so the model is the same: given a draw's
//! parameters, the cell's posterior over every row `(ℓ, k)` is its table
//! entry plus `log π_ℓ` normalised ([`log_posterior_row`]), the jitter of
//! its unspliced share integrated out ([`share_term`]) when the model has
//! one. Over the draws the posteriors are averaged (the parameters' draws
//! integrated out); from that mixture every cell's lineage probabilities,
//! the mean, sd and 90% interval of its time, and its mean velocities, on
//! each track and of its log level ([`log_rate`]).

use super::blocks::log_posterior_row;
use super::likelihood::{integrated_table, table, Depth, Reads};
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
    /// `[P]` every pseudobulk's row and depths at this draw (empty where
    /// the draw has none, as a truth made up for a test).
    pub rows: Vec<usize>,
    pub lambda: Vec<Depth>,
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
            velocity_log: DMatrix::from_element(n_cells, genes, f64::NAN),
            time_bin: f64::NAN,
        }
    }
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

/// At most this many time bins a cell (wider bins past it).
const MAX_BINS: usize = 4096;

/// Every cell of `cells` read out over the draws `snaps` (the same model:
/// lineages and grid, checked). Time bins are one position of the
/// shortest lineage wide (at most [`MAX_BINS`]); a cell some draw cannot
/// place anywhere (its row not finite) is NaN throughout.
pub fn read_cells(cells: &Reads, snaps: &[Snapshot]) -> CellReadout {
    let (n_cells, genes) = (cells.u.nrows(), cells.u.ncols());
    let Some((lineages, positions)) = grid_shape(snaps) else {
        return CellReadout::empty(n_cells, genes);
    };
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
    let (mut vel_u, mut vel_s, mut vel_log) = (
        DMatrix::zeros(genes, n_cells),
        DMatrix::zeros(genes, n_cells),
        DMatrix::zeros(genes, n_cells),
    );
    let rows = lineages * positions;
    // `[rows × cells]`, a cell's posterior one contiguous column, reused.
    let mut post = DMatrix::<f64>::zeros(rows, n_cells);
    for snap in snaps {
        let p = &snap.params;
        let lv = p.levels();
        fill_posterior(cells, snap, &lv, weight, &mut post);
        let times: Vec<(usize, f64, usize)> = (0..rows)
            .map(|r| {
                let (l, k) = (r / positions, r % positions);
                let t = p.time_at(l, k);
                (l, t, ((t / width).round() as usize).min(n_bins - 1))
            })
            .collect();
        post.as_slice()
            .par_chunks(rows)
            .zip(acc.par_iter_mut())
            .for_each(|(col, a)| {
                if col[0].is_nan() {
                    a.unknown = true;
                    return;
                }
                for (&x, &(lin, t, bin)) in col.iter().zip(&times) {
                    a.lineage[lin] += x;
                    a.m1 += x * t;
                    a.m2 += x * t * t;
                    a.bins[bin] += x;
                }
            });
        let (du, ds) = p.velocities_of(&lv);
        vel_u.gemm(1.0, &du.transpose(), &post, 1.0);
        vel_s.gemm(1.0, &ds.transpose(), &post, 1.0);
        // Filled `[rows × G]` (the inputs' own order), transposed once for
        // the product.
        let dlog = DMatrix::from_fn(rows, genes, |r, g| {
            log_rate(du[(r, g)], ds[(r, g)], lv.u[(r, g)], lv.s[(r, g)])
        });
        vel_log.gemm(1.0, &dlog.transpose(), &post, 1.0);
    }
    let nan_rows = |mut m: DMatrix<f64>| {
        for c in (0..n_cells).filter(|&c| acc[c].unknown) {
            m.row_mut(c).fill(f64::NAN);
        }
        m
    };
    let (velocity_u, velocity_s, velocity_log) = (
        nan_rows(vel_u.transpose()),
        nan_rows(vel_s.transpose()),
        nan_rows(vel_log.transpose()),
    );
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
        velocity_log,
        time_bin: width,
    }
}

/// The draws' lineages and positions, `None` with no draw; every draw on
/// the same (checked).
fn grid_shape(snaps: &[Snapshot]) -> Option<(usize, usize)> {
    let first = snaps.first()?;
    let (lineages, positions) = (first.params.path.len(), first.params.positions());
    assert!(
        snaps
            .iter()
            .all(|s| s.params.path.len() == lineages && s.params.positions() == positions),
        "every snapshot on the same lineages and grid"
    );
    Some((lineages, positions))
}

/// One draw's posterior of every cell over every row `(ℓ, k)`, times
/// `weight`, into `post` (`[rows × cells]`, a cell a column); a cell the
/// draw cannot place (its row not finite) is NaN.
fn fill_posterior(
    cells: &Reads,
    snap: &Snapshot,
    lv: &super::model::Levels,
    weight: f64,
    post: &mut DMatrix<f64>,
) {
    let (rows, positions) = (post.nrows(), snap.params.positions());
    let tab = match snap.share_sd {
        Some(sd) => integrated_table(cells, lv, sd),
        None => table(cells, lv),
    };
    post.as_mut_slice()
        .par_chunks_mut(rows)
        .enumerate()
        .for_each(|(c, col)| {
            let (lp, norm) = log_posterior_row(&tab, &snap.log_w, positions, c);
            if !norm.is_finite() {
                col.fill(f64::NAN);
                return;
            }
            for (x, &l) in col.iter_mut().zip(&lp) {
                *x = weight * l.exp();
            }
        });
}

/// Every cell's posterior over the grid, averaged over the draws.
pub struct GridPosterior {
    /// `[C × L positions]`, a cell's row summing to one (NaN where some
    /// draw cannot place it), columns `ℓ positions + k`.
    pub post: DMatrix<f64>,
    /// `[L positions]` every column's phase ([`Params::phase_at`]), the same
    /// in every draw, and its time averaged over the draws.
    pub phase: Vec<f64>,
    pub time: Vec<f64>,
    pub positions: usize,
}

/// Every cell of `cells` on the grid of the draws `snaps` (at least one; the
/// same lineages and grid, checked), its posterior averaged over them.
pub fn grid_posterior(cells: &Reads, snaps: &[Snapshot]) -> GridPosterior {
    let n_cells = cells.u.nrows();
    let (lineages, positions) = grid_shape(snaps).expect("at least one draw");
    let rows = lineages * positions;
    let weight = 1.0 / snaps.len() as f64;
    let (mut sum, mut post) = (
        DMatrix::<f64>::zeros(rows, n_cells),
        DMatrix::<f64>::zeros(rows, n_cells),
    );
    let mut time = vec![0.0; rows];
    for snap in snaps {
        let lv = snap.params.levels();
        fill_posterior(cells, snap, &lv, weight, &mut post);
        sum += &post;
        for (r, t) in time.iter_mut().enumerate() {
            *t += weight * snap.params.time_at(r / positions, r % positions);
        }
    }
    let phase = (0..rows)
        .map(|r| snaps[0].params.phase_at(r % positions))
        .collect();
    GridPosterior {
        post: sum.transpose(),
        phase,
        time,
        positions,
    }
}

/// What the readout needs of the climb's mode: every pseudobulk at its
/// most probable row under `params` and the lineages' `log_w` (the jitter
/// of its unspliced share integrated out at `share_sd`), and its depths
/// there.
pub fn mode_snapshot(
    reads: &Reads,
    params: &Params,
    log_w: &[f64],
    share_sd: Option<f64>,
) -> Snapshot {
    let lv = params.levels();
    let tab = match share_sd {
        Some(sd) => integrated_table(reads, &lv, sd),
        None => table(reads, &lv),
    };
    let n = params.positions();
    let rows: Vec<usize> = (0..tab.nrows())
        .into_par_iter()
        .map(|p| {
            let (lp, _) = log_posterior_row(&tab, log_w, n, p);
            (0..lp.len())
                .max_by(|&a, &b| lp[a].total_cmp(&lp[b]))
                .expect("a row")
        })
        .collect();
    let lambda = reads.depths(&rows, &lv);
    Snapshot {
        params: params.clone(),
        log_w: log_w.to_vec(),
        share_sd,
        rows,
        lambda,
    }
}
