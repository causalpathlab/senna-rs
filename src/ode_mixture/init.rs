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
//! A tree (two lineages on a shared trunk) is started apart
//! ([`tree_starts`]): k-means would cut the trunk off as a cluster of its
//! own. Geodesic distances on the scores' nearest-neighbour graph find the
//! three ends of the tree; with one end as the root the other two are the
//! tips, and the Gromov products against them split the pseudobulks into
//! the trunk (on both lineages) and two branches, every part ordered by its
//! distance from the root onto one clock shared by both lineages.
//!
//! [`climb`] then runs Adam from the k-means start, and again with each
//! lineage turned round, keeping a turn that raises the objective (the
//! start fixes the order; the splicing says which way round fits); with
//! two lineages also from every tree start, and keeps the best of all.
//! This is a search for the climb's basin, not for a root: a lineage's
//! `t = 0` is a convention.

use super::blocks::GenePrior;
use super::fit::{Fit, SHARE_START};
use super::kernel::GridPrior;
use super::likelihood::Reads;
use super::model::Params;
use legume_numeric::candle::candle_core::{Device, Result};
use legume_numeric::matrix::traits::RandomizedAlgs;
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
    /// Positions per grid step a pseudobulk's time can take
    /// ([`Params::sub`]).
    pub sub: usize,
    /// Every pseudobulk's unspliced share jitters on its own, `σ` fitted
    /// ([`Fit::with_share`]).
    pub jitter: bool,
    /// Every other lineage on lineage 0's trunk until a branch time it
    /// fits, one clock for all ([`Fit::with_trunk`]).
    pub trunk: bool,
    /// The pseudobulk at the root of the trajectory, when the user names
    /// one: under a trunk the one tree start grows from it
    /// ([`tree_start_from_root`]); else every end of the tree is tried.
    pub root: Option<usize>,
}

/// Bandwidth, on `[0, 1]`, of the smoothing of scores onto the grid.
const BANDWIDTH: f64 = 0.08;

/// Neighbours per pseudobulk in the graph of a tree start.
const NEIGHBOURS: usize = 10;

/// Every pseudobulk's places in a start: `(lineage, time on [0, 1])`, one
/// per lineage it is on (a trunk pseudobulk is on both).
pub type Places = Vec<Vec<(usize, f64)>>;

/// `([P × G] centred log spliced shares, [G] their means)`.
fn log_shares(reads: &Reads) -> (DMatrix<f64>, Vec<f64>) {
    let (p, g) = (reads.s.nrows(), reads.s.ncols());
    let total: Vec<f64> = reads
        .s
        .row_iter()
        .map(|r| r.sum() + 0.5 * g as f64)
        .collect();
    let mut y = DMatrix::from_fn(p, g, |i, j| ((reads.s[(i, j)] + 0.5) / total[i]).ln());
    let mean: Vec<f64> = (0..g).map(|j| y.column(j).mean()).collect();
    for (j, m) in mean.iter().enumerate() {
        y.column_mut(j).add_scalar_mut(-m);
    }
    (y, mean)
}

/// `([P × H] scores, [G × H] loadings)` of `y` on its leading axes, by
/// randomized SVD (a fixed start, so reproducible): only `H` axes are read,
/// and a full SVD of thousands of pseudobulks by every gene takes minutes.
pub(crate) fn axes(y: &DMatrix<f64>, h: usize) -> (DMatrix<f64>, DMatrix<f64>) {
    let h = h.min(y.nrows()).min(y.ncols());
    let (u, d, v) = y.rsvd(h).expect("a non-empty matrix");
    let scores = DMatrix::from_fn(y.nrows(), h, |i, k| u[(i, k)] * d[k]);
    (scores, v)
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

/// The symmetrised `NEIGHBOURS`-nearest-neighbour graph of the rows,
/// `[P][(neighbour, distance)]`.
fn knn_graph(x: &DMatrix<f64>) -> Vec<Vec<(usize, f64)>> {
    let n = x.nrows();
    let dist = |a: usize, b: usize| (x.row(a) - x.row(b)).norm();
    let mut adj: Vec<Vec<(usize, f64)>> = vec![Vec::new(); n];
    for a in 0..n {
        let mut near: Vec<(usize, f64)> = (0..n)
            .filter(|&b| b != a)
            .map(|b| (b, dist(a, b)))
            .collect();
        near.sort_by(|i, j| i.1.total_cmp(&j.1));
        for &(b, d) in near.iter().take(NEIGHBOURS) {
            adj[a].push((b, d));
            adj[b].push((a, d));
        }
    }
    adj
}

/// Every node's geodesic distance from `from` (∞ where unreachable).
fn geodesic(adj: &[Vec<(usize, f64)>], from: usize) -> Vec<f64> {
    use std::cmp::Reverse;
    use std::collections::BinaryHeap;
    let mut dist = vec![f64::INFINITY; adj.len()];
    let mut heap = BinaryHeap::new();
    dist[from] = 0.0;
    // Non-negative floats order as their bits.
    heap.push(Reverse((0f64.to_bits(), from)));
    while let Some(Reverse((d, a))) = heap.pop() {
        let d = f64::from_bits(d);
        if d > dist[a] {
            continue;
        }
        for &(b, w) in &adj[a] {
            let next = d + w;
            if next < dist[b] {
                dist[b] = next;
                heap.push(Reverse((next.to_bits(), b)));
            }
        }
    }
    dist
}

fn farthest(d: &[f64]) -> usize {
    (0..d.len())
        .max_by(|&a, &b| d[a].total_cmp(&d[b]))
        .expect("rows")
}

/// Starts for `lineages` on a shared trunk, one per end of the tree taken
/// as the root (`lineages + 1` ends); none when the graph falls apart.
pub fn tree_starts(reads: &Reads, h: usize, lineages: usize) -> Vec<Places> {
    let (y, _) = log_shares(reads);
    let (scores, _) = axes(&y, h);
    tree_places(&scores, lineages)
}

/// The one tree start from a root the user names: the pseudobulk `root`,
/// its tips the `lineages` ends farthest from it (and from one another).
/// A stem in the middle of a tree is no end of it, so no end taken as the
/// root finds it; `None` when the graph falls apart or fewer branch.
pub fn tree_start_from_root(
    reads: &Reads,
    h: usize,
    lineages: usize,
    root: usize,
) -> Option<Places> {
    let (y, _) = log_shares(reads);
    let (scores, _) = axes(&y, h);
    tree_place_from_root(&scores, lineages, root)
}

fn tree_place_from_root(scores: &DMatrix<f64>, lineages: usize, root: usize) -> Option<Places> {
    let adj = knn_graph(scores);
    let dr = geodesic(&adj, root);
    if dr.iter().any(|d| !d.is_finite()) {
        return None;
    }
    let mut near = dr.clone();
    let mut tips: Vec<(usize, Vec<f64>)> = Vec::with_capacity(lineages);
    for _ in 0..lineages {
        let e = farthest(&near);
        let d = geodesic(&adj, e);
        for (n, &x) in near.iter_mut().zip(&d) {
            *n = n.min(x);
        }
        tips.push((e, d));
    }
    tree_from_root(&dr, &tips.iter().collect::<Vec<_>>())
}

fn tree_places(scores: &DMatrix<f64>, lineages: usize) -> Vec<Places> {
    let p = scores.nrows();
    if p <= lineages + 1 {
        return Vec::new();
    }
    let adj = knn_graph(scores);
    let d0 = geodesic(&adj, 0);
    if d0.iter().any(|d| !d.is_finite()) {
        return Vec::new();
    }
    // The ends, farthest first: the farthest from anywhere, then each the
    // farthest from the ends so far.
    let e1 = farthest(&d0);
    let mut near = geodesic(&adj, e1);
    let mut ends: Vec<(usize, Vec<f64>)> = vec![(e1, near.clone())];
    for _ in 0..lineages {
        let e = farthest(&near);
        let d = geodesic(&adj, e);
        for (n, &x) in near.iter_mut().zip(&d) {
            *n = n.min(x);
        }
        ends.push((e, d));
    }
    (0..ends.len())
        .filter_map(|r| {
            let tips: Vec<&(usize, Vec<f64>)> = ends
                .iter()
                .enumerate()
                .filter(|&(i, _)| i != r)
                .map(|(_, e)| e)
                .collect();
            tree_from_root(&ends[r].1, &tips)
        })
        .collect()
}

/// The tree from the root (`dr` its distances) to the `tips`, tip 0's way
/// the spine (lineage 0) every other lineage parts from: a pseudobulk's
/// Gromov product with a tip is how far it runs along the way to that
/// tip, so it is on the lineage whose way it shares longest, and lineage
/// `a` parts from the spine where its tip's way does; a pseudobulk before
/// two lineages' parting is on both.
fn tree_from_root(dr: &[f64], tips: &[&(usize, Vec<f64>)]) -> Option<Places> {
    let (p, l) = (dr.len(), tips.len());
    let along = |x: usize, a: usize| 0.5 * (dr[tips[a].0] + dr[x] - tips[a].1[x]);
    let on: Vec<usize> = (0..p)
        .map(|x| {
            (0..l)
                .max_by(|&a, &b| along(x, a).total_cmp(&along(x, b)))
                .expect("a tip")
        })
        .collect();
    if (0..l).any(|a| !on.contains(&a)) {
        return None;
    }
    // A pseudobulk's progress is how far it runs along its own lineage's
    // way; it is on another lineage's way too while short of their
    // parting (lineage 0, the spine, parts from lineage a at a's fork).
    let fork: Vec<f64> = (0..l)
        .map(|a| {
            if a == 0 {
                f64::INFINITY
            } else {
                along(tips[0].0, a)
            }
        })
        .collect();
    let progress: Vec<f64> = (0..p).map(|x| along(x, on[x])).collect();
    let shared = |x: usize, b: usize| b != on[x] && progress[x] <= fork[on[x]].min(fork[b]);
    // One clock: on every lineage, the pseudobulks it shares come first,
    // their share of all pseudobulks their share of time, its own after.
    let mut places: Places = vec![Vec::new(); p];
    for a in 0..l {
        let (mut with, mut own): (Vec<usize>, Vec<usize>) = (0..p)
            .filter(|&x| on[x] == a || shared(x, a))
            .partition(|&x| (0..l).any(|b| shared(x, b)));
        with.sort_by(|&i, &j| progress[i].total_cmp(&progress[j]));
        own.sort_by(|&i, &j| progress[i].total_cmp(&progress[j]));
        let tb = with.len() as f64 / p as f64;
        for (i, &x) in with.iter().enumerate() {
            places[x].push((a, tb * i as f64 / with.len() as f64));
        }
        for (i, &x) in own.iter().enumerate() {
            places[x].push((a, tb + (1.0 - tb) * (i + 1) as f64 / own.len() as f64));
        }
    }
    Some(places)
}

/// Where every lineage `1..lineages` parts from lineage 0: the latest time
/// of a pseudobulk on both, or (none shared) [`NO_TRUNK`].
fn branch_times(places: &Places, lineages: usize) -> Vec<f64> {
    (1..lineages)
        .map(|l| {
            places
                .iter()
                .filter(|at| at.iter().any(|&(m, _)| m == 0) && at.iter().any(|&(m, _)| m == l))
                .flat_map(|at| at.iter().map(|&(_, t)| t))
                .fold(None, |m: Option<f64>, t| Some(m.map_or(t, |m| m.max(t))))
                .unwrap_or(NO_TRUNK)
        })
        .collect()
}

/// The branch time of a start with no trunk: the lineages part at once.
const NO_TRUNK: f64 = 0.05;

/// The k-means start, every lineage `l` with `reversed[l]` turned round.
fn kmeans_places(scores: &DMatrix<f64>, lineages: usize, reversed: &[bool]) -> Places {
    let (label, order) = order_of(scores, lineages);
    label
        .iter()
        .zip(&order)
        .map(|(&l, &t)| vec![(l, if reversed[l] { 1.0 - t } else { t })])
        .collect()
}

/// The start from every pseudobulk's places.
pub fn params_from(reads: &Reads, opts: &Climb, places: &Places) -> Params {
    let (p, g) = (reads.s.nrows(), reads.s.ncols());
    let (y, mean) = log_shares(reads);
    let (scores, loadings) = axes(&y, opts.dim);
    let h = scores.ncols();
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
                    let on = places.iter().enumerate().flat_map(|(a, at)| {
                        at.iter()
                            .filter(|&&(m, _)| m == l)
                            .map(move |&(_, t)| (a, t))
                    });
                    for (a, t) in on {
                        let w = (-(t - tau).powi(2) / (2.0 * BANDWIDTH * BANDWIDTH)).exp();
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
    let prior = GenePrior::default();
    let gamma0 = prior.log_gamma.0.exp();
    let (su, ss): (Vec<f64>, Vec<f64>) = (0..g)
        .map(|j| (reads.u.column(j).sum() + 0.5, reads.s.column(j).sum() + 0.5))
        .unzip();
    // Every gene's reads' `U/S = e^c γ/β` at steady state: the median gene
    // on the prior's `γ/β`, the rest of its offset the capture's, each
    // gene's own in its `β` (the levels the same however it is split).
    let mut log_us: Vec<f64> = su.iter().zip(&ss).map(|(u, s)| (u / s).ln()).collect();
    log_us.sort_by(f64::total_cmp);
    let log_capture = log_us.get(g / 2).map_or(0.0, |m| m - prior.log_ratio.0);
    let mut start = Params {
        beta: (0..g)
            .map(|j| gamma0 * log_capture.exp() * ss[j] / su[j])
            .collect(),
        gamma: vec![gamma0; g],
        rho: (0..g)
            .map(|j| (0..h).map(|d| loadings[(j, d)] * sd[d]).collect())
            .collect(),
        base: mean.iter().map(|m| m + gamma0.ln()).collect(),
        t_len: vec![1.0; opts.lineages],
        path,
        sub: opts.sub,
        branch: None,
        log_capture,
    };
    // Inside the rates' bounds ([`GenePrior::bounded`]).
    if let Some((lo, hi)) = GenePrior::bounded(&start).log_rate {
        let inside = |r: f64| r.ln().clamp(lo, hi).exp();
        for r in start.beta.iter_mut().chain(start.gamma.iter_mut()) {
            *r = inside(*r);
        }
    }
    start
}

/// Whether a start's objective `new` beats the best so far: a NaN (a climb
/// that diverged) never does, and any finite one beats a NaN best.
pub fn beats(new: f32, best: f32) -> bool {
    new > best || (best.is_nan() && !new.is_nan())
}

/// Adam from the start, then from the start with each lineage in turn
/// turned round, a turn kept when it raises the objective; with two
/// lineages also from every tree start. The best fit and its objective.
pub fn climb(reads: &Reads, opts: &Climb, dev: &Device) -> Result<(Fit, f32)> {
    let prior = GridPrior::new(opts.points, 1.0, opts.kappa)
        .map_err(|e| legume_numeric::candle::candle_core::Error::Msg(e.to_string()))?;
    let (y, _) = log_shares(reads);
    let (scores, _) = axes(&y, opts.dim);
    let run = |places: &Places| -> Result<(Fit, f32)> {
        let start = params_from(reads, opts, places);
        let mut fit = Fit::from_params(&start, &prior, dev)?
            .with_share(opts.jitter.then_some(SHARE_START))?;
        if opts.trunk && opts.lineages > 1 {
            fit = fit.with_trunk(&start, &prior, &branch_times(places, opts.lineages))?;
        }
        let objective = fit.adam(reads, opts.steps, opts.lr)?;
        Ok((fit, objective))
    };
    // Under a trunk the starts are trees, one per end taken as the root
    // (a k-means start has no trunk, and its lineages turned round one at
    // a time cannot share one); apart, the k-means start and each lineage
    // turned round in turn, then the trees too.
    let trees = match (opts.lineages > 1, opts.root) {
        (false, _) => Vec::new(),
        (true, Some(root)) => tree_place_from_root(&scores, opts.lineages, root)
            .into_iter()
            .collect(),
        (true, None) => tree_places(&scores, opts.lineages),
    };
    let mut best: Option<(Fit, f32)> = None;
    let keep = |other: (Fit, f32), best: &mut Option<(Fit, f32)>| {
        if best.as_ref().is_none_or(|b| beats(other.1, b.1)) {
            *best = Some(other);
        }
    };
    if !(opts.trunk && opts.lineages > 1 && !trees.is_empty()) {
        let mut reversed = vec![false; opts.lineages];
        let mut best_kmeans = run(&kmeans_places(&scores, opts.lineages, &reversed))?;
        for l in 0..opts.lineages {
            reversed[l] = true;
            let other = run(&kmeans_places(&scores, opts.lineages, &reversed))?;
            if beats(other.1, best_kmeans.1) {
                best_kmeans = other;
            } else {
                reversed[l] = false;
            }
        }
        keep(best_kmeans, &mut best);
    }
    for places in &trees {
        let climbed = run(places)?;
        log::debug!("ode climb: a tree start's objective {}", climbed.1);
        keep(climbed, &mut best);
    }
    let best = best.expect("a start");
    if best.1.is_nan() {
        return Err(legume_numeric::candle::candle_core::Error::Msg(
            "ode-mixture climb: every start diverged (objective NaN)".into(),
        ));
    }
    Ok(best)
}
