//! A start for the climb, from the reads alone.
//!
//! 1. The pseudobulks' log spliced shares, centred per gene, on their
//!    leading `H` principal axes: the gene loadings start `ρ`, the genes'
//!    mean log shares `b`.
//! 2. `L` lineages by k-means on the scores; within each, the pseudobulks
//!    ordered along its own first principal direction, onto `[0, 1]`.
//! 3. Every lineage's path the scores smoothed along that order onto its
//!    grid; `γ` at the prior's mean, `γ/β` every gene's unspliced over
//!    spliced reads.
//!
//! [`climb`] then runs Adam from it, and again with each lineage turned
//! round, keeping a turn that raises the objective: the start fixes the
//! order, the splicing the direction.

use super::blocks::GenePrior;
use super::fit::Fit;
use super::kernel::GridPrior;
use super::likelihood::Reads;
use super::model::Params;
use legume_numeric::candle::candle_core::{Device, Result};
use nalgebra::DMatrix;

/// What [`climb`] fits.
#[derive(Clone, Debug)]
pub struct Climb {
    pub lineages: usize,
    /// Grid points per lineage.
    pub points: usize,
    /// The paths' dimension `H`.
    pub dim: usize,
    /// The prior's length scale over the duration.
    pub kappa: f64,
    pub steps: usize,
    pub lr: f64,
}

/// Bandwidth, on `[0, 1]`, of the smoothing of scores onto the grid.
const BANDWIDTH: f64 = 0.08;

/// `([P × G] centred log spliced shares, [G] their means)`.
fn log_shares(reads: &Reads) -> (DMatrix<f64>, Vec<f64>) {
    let (p, g) = (reads.s.nrows(), reads.s.ncols());
    let mut y = DMatrix::from_fn(p, g, |i, j| {
        let total = reads.s.row(i).sum() + 0.5 * g as f64;
        ((reads.s[(i, j)] + 0.5) / total).ln()
    });
    let mean: Vec<f64> = (0..g).map(|j| y.column(j).mean()).collect();
    for (j, m) in mean.iter().enumerate() {
        y.column_mut(j).add_scalar_mut(-m);
    }
    (y, mean)
}

/// `([P × H] scores, [G × H] loadings)` of `y` on its leading axes.
fn axes(y: &DMatrix<f64>, h: usize) -> (DMatrix<f64>, DMatrix<f64>) {
    let svd = y.clone().svd(true, true);
    let (u, v) = (
        svd.u.expect("asked for"),
        svd.v_t.expect("asked for").transpose(),
    );
    let h = h.min(svd.singular_values.len());
    let scores = DMatrix::from_fn(y.nrows(), h, |i, d| u[(i, d)] * svd.singular_values[d]);
    (scores, v.columns(0, h).into_owned())
}

/// `k` clusters of the rows by Lloyd's algorithm, started at far-apart rows
/// (the first row, then each next the farthest from those chosen).
fn kmeans(x: &DMatrix<f64>, k: usize) -> Vec<usize> {
    let n = x.nrows();
    let dist =
        |a: usize, c: &[f64]| -> f64 { x.row(a).iter().zip(c).map(|(p, q)| (p - q).powi(2)).sum() };
    let mut centres: Vec<Vec<f64>> = vec![x.row(0).iter().copied().collect()];
    while centres.len() < k.min(n) {
        let far = (0..n)
            .max_by(|&a, &b| {
                let da = centres
                    .iter()
                    .map(|c| dist(a, c))
                    .fold(f64::INFINITY, f64::min);
                let db = centres
                    .iter()
                    .map(|c| dist(b, c))
                    .fold(f64::INFINITY, f64::min);
                da.total_cmp(&db)
            })
            .expect("rows");
        centres.push(x.row(far).iter().copied().collect());
    }
    let mut label = vec![0; n];
    for _ in 0..100 {
        let next: Vec<usize> = (0..n)
            .map(|a| {
                (0..centres.len())
                    .min_by(|&i, &j| dist(a, &centres[i]).total_cmp(&dist(a, &centres[j])))
                    .expect("centres")
            })
            .collect();
        if next == label && centres.len() > 1 {
            break;
        }
        label = next;
        for (c, centre) in centres.iter_mut().enumerate() {
            let rows: Vec<usize> = (0..n).filter(|&a| label[a] == c).collect();
            if !rows.is_empty() {
                for (d, v) in centre.iter_mut().enumerate() {
                    *v = rows.iter().map(|&a| x[(a, d)]).sum::<f64>() / rows.len() as f64;
                }
            }
        }
    }
    label
}

/// Every pseudobulk's start lineage and its order within it on `[0, 1]`.
pub fn start_order(reads: &Reads, lineages: usize, h: usize) -> (Vec<usize>, Vec<f64>) {
    let (y, _) = log_shares(reads);
    let (scores, _) = axes(&y, h);
    order_of(&scores, lineages)
}

fn order_of(scores: &DMatrix<f64>, lineages: usize) -> (Vec<usize>, Vec<f64>) {
    let p = scores.nrows();
    let label = kmeans(scores, lineages);
    let mut order = vec![0.0; p];
    for l in 0..lineages {
        let rows: Vec<usize> = (0..p).filter(|&a| label[a] == l).collect();
        if rows.len() < 2 {
            continue;
        }
        let sub = DMatrix::from_fn(rows.len(), scores.ncols(), |i, d| scores[(rows[i], d)]);
        let mut centred = sub.clone();
        for d in 0..sub.ncols() {
            let m = sub.column(d).mean();
            centred.column_mut(d).add_scalar_mut(-m);
        }
        let (along, _) = axes(&centred, 1);
        let mut idx: Vec<usize> = (0..rows.len()).collect();
        idx.sort_by(|&i, &j| along[(i, 0)].total_cmp(&along[(j, 0)]));
        for (rank, &i) in idx.iter().enumerate() {
            order[rows[i]] = rank as f64 / (rows.len() - 1) as f64;
        }
    }
    (label, order)
}

/// The start, every lineage `l` with `reversed[l]` turned round.
pub fn initial_params(reads: &Reads, opts: &Climb, reversed: &[bool]) -> Params {
    let (p, g) = (reads.s.nrows(), reads.s.ncols());
    let (y, mean) = log_shares(reads);
    let (scores, loadings) = axes(&y, opts.dim);
    let h = scores.ncols();
    let (label, mut order) = order_of(&scores, opts.lineages);
    for a in 0..p {
        if reversed[label[a]] {
            order[a] = 1.0 - order[a];
        }
    }
    // Unit-variance coordinates for the paths, the scale in ρ.
    let sd: Vec<f64> = (0..h)
        .map(|d| {
            (scores.column(d).iter().map(|x| x * x).sum::<f64>() / p as f64)
                .sqrt()
                .max(1e-9)
        })
        .collect();
    let n = opts.points;
    let path: Vec<Vec<Vec<f64>>> = (0..opts.lineages)
        .map(|l| {
            (0..n)
                .map(|k| {
                    let tau = k as f64 / (n - 1) as f64;
                    let (mut acc, mut wsum) = (vec![0.0; h], 0.0);
                    for a in (0..p).filter(|&a| label[a] == l) {
                        let w = (-(order[a] - tau).powi(2) / (2.0 * BANDWIDTH * BANDWIDTH)).exp();
                        wsum += w;
                        for d in 0..h {
                            acc[d] += w * scores[(a, d)] / sd[d];
                        }
                    }
                    acc.iter().map(|v| v / wsum.max(1e-300)).collect()
                })
                .collect()
        })
        .collect();
    let gamma0 = GenePrior::default().log_gamma.0.exp();
    let (su, ss): (Vec<f64>, Vec<f64>) = (0..g)
        .map(|j| (reads.u.column(j).sum() + 0.5, reads.s.column(j).sum() + 0.5))
        .unzip();
    Params {
        beta: (0..g).map(|j| gamma0 * ss[j] / su[j]).collect(),
        gamma: vec![gamma0; g],
        rho: (0..g)
            .map(|j| (0..h).map(|d| loadings[(j, d)] * sd[d]).collect())
            .collect(),
        base: mean.iter().map(|m| m + gamma0.ln()).collect(),
        t_len: vec![1.0; opts.lineages],
        path,
    }
}

/// Adam from the start, then from the start with each lineage in turn
/// turned round, a turn kept when it raises the objective. The fit and its
/// objective.
pub fn climb(reads: &Reads, opts: &Climb, dev: &Device) -> Result<(Fit, f32)> {
    let prior = GridPrior::new(opts.points, 1.0, opts.kappa)
        .map_err(|e| legume_numeric::candle::candle_core::Error::Msg(e.to_string()))?;
    let run = |reversed: &[bool]| -> Result<(Fit, f32)> {
        let fit = Fit::from_params(&initial_params(reads, opts, reversed), &prior, dev)?;
        let objective = fit.adam(reads, opts.steps, opts.lr)?;
        Ok((fit, objective))
    };
    let mut reversed = vec![false; opts.lineages];
    let mut best = run(&reversed)?;
    for l in 0..opts.lineages {
        reversed[l] = true;
        let other = run(&reversed)?;
        if other.1 > best.1 {
            best = other;
        } else {
            reversed[l] = false;
        }
    }
    Ok(best)
}
