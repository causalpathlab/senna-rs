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
//! start fixes the order, the splicing the direction); with two lineages
//! also from every tree start, and keeps the best of all.

use super::blocks::GenePrior;
use super::fit::{Fit, SHARE_START};
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
    /// Positions per grid step a pseudobulk's time can take
    /// ([`Params::sub`]).
    pub sub: usize,
    /// Every pseudobulk's unspliced share jitters on its own, `σ` fitted
    /// ([`Fit::with_share`]).
    pub jitter: bool,
    /// Every other lineage on lineage 0's trunk until a branch time it
    /// fits, one clock for all ([`Fit::with_trunk`]).
    pub trunk: bool,
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

/// Starts for two lineages on a shared trunk, one per end of the tree taken
/// as the root; none when the graph falls apart.
pub fn tree_starts(reads: &Reads, h: usize) -> Vec<Places> {
    let (y, _) = log_shares(reads);
    let (scores, _) = axes(&y, h);
    tree_places(&scores)
}

fn tree_places(scores: &DMatrix<f64>) -> Vec<Places> {
    let p = scores.nrows();
    if p < 3 {
        return Vec::new();
    }
    let adj = knn_graph(scores);
    let d0 = geodesic(&adj, 0);
    if d0.iter().any(|d| !d.is_finite()) {
        return Vec::new();
    }
    // The ends: the farthest from anywhere, the farthest from it, and the
    // farthest from both.
    let e1 = farthest(&d0);
    let d1 = geodesic(&adj, e1);
    let e2 = farthest(&d1);
    let d2 = geodesic(&adj, e2);
    let near: Vec<f64> = (0..p).map(|a| d1[a].min(d2[a])).collect();
    let e3 = farthest(&near);
    let d3 = geodesic(&adj, e3);
    let ends = [(e1, &d1), (e2, &d2), (e3, &d3)];
    (0..3)
        .filter_map(|r| {
            let (_, dr) = ends[r];
            tree_from_root(dr, ends[(r + 1) % 3], ends[(r + 2) % 3])
        })
        .collect()
}

/// The trunk and the branches to the tips `a`, `b` seen from the root
/// (`dr` its distances): a pseudobulk's Gromov product with a tip is how
/// far it runs along the way to that tip, the tips' with each other where
/// the branches part.
fn tree_from_root(
    dr: &[f64],
    (a, da): (usize, &Vec<f64>),
    (b, db): (usize, &Vec<f64>),
) -> Option<Places> {
    let p = dr.len();
    let fork = 0.5 * (dr[a] + dr[b] - da[b]);
    // The trunk, the branch to `a`, the branch to `b`.
    let mut part = vec![Vec::new(); 3];
    for x in 0..p {
        let ga = 0.5 * (dr[a] + dr[x] - da[x]);
        let gb = 0.5 * (dr[b] + dr[x] - db[x]);
        let which = if ga.max(gb) <= fork {
            0
        } else if ga > gb {
            1
        } else {
            2
        };
        part[which].push(x);
    }
    if part[1].is_empty() || part[2].is_empty() {
        return None;
    }
    for rows in &mut part {
        rows.sort_by(|&i, &j| dr[i].total_cmp(&dr[j]));
    }
    // One clock: the trunk's share of the pseudobulks is its share of time.
    let tb = part[0].len() as f64 / p as f64;
    let mut places: Places = vec![Vec::new(); p];
    for (i, &x) in part[0].iter().enumerate() {
        let t = tb * i as f64 / part[0].len() as f64;
        places[x] = vec![(0, t), (1, t)];
    }
    for (l, rows) in part[1..].iter().enumerate() {
        for (i, &x) in rows.iter().enumerate() {
            places[x] = vec![(l, tb + (1.0 - tb) * (i + 1) as f64 / rows.len() as f64)];
        }
    }
    Some(places)
}

/// Where a start's lineages part: the latest time of a pseudobulk on more
/// than one, or (none shared) [`NO_TRUNK`].
fn branch_of(places: &Places) -> f64 {
    places
        .iter()
        .filter(|at| at.len() > 1)
        .flat_map(|at| at.iter().map(|&(_, t)| t))
        .fold(None, |m: Option<f64>, t| Some(m.map_or(t, |m| m.max(t))))
        .unwrap_or(NO_TRUNK)
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
        sub: opts.sub,
        branch: None,
    }
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
            fit = fit.with_trunk(&start, &prior, branch_of(places))?;
        }
        let objective = fit.adam(reads, opts.steps, opts.lr)?;
        Ok((fit, objective))
    };
    let mut reversed = vec![false; opts.lineages];
    let mut best = run(&kmeans_places(&scores, opts.lineages, &reversed))?;
    for l in 0..opts.lineages {
        reversed[l] = true;
        let other = run(&kmeans_places(&scores, opts.lineages, &reversed))?;
        if beats(other.1, best.1) {
            best = other;
        } else {
            reversed[l] = false;
        }
    }
    if opts.lineages == 2 {
        for places in tree_places(&scores) {
            let other = run(&places)?;
            if beats(other.1, best.1) {
                best = other;
            }
        }
    }
    if best.1.is_nan() {
        return Err(legume_numeric::candle::candle_core::Error::Msg(
            "ode-mixture climb: every start diverged (objective NaN)".into(),
        ));
    }
    Ok(best)
}
