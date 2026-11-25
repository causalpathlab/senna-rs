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
//! A tree (`L` lineages on a shared trunk) is started apart
//! ([`tree_starts`]): k-means would cut the trunk off as a cluster of its
//! own. Geodesic distances on the scores' nearest-neighbour graph find the
//! `L + 1` ends of the tree; with one end as the root the others are the
//! tips, and the Gromov products against them put every pseudobulk on the
//! lineage whose way it shares longest, the trunk on several, every part
//! ordered by its distance from the root onto one clock shared by all. A
//! stem inside the tree is no end of it, so a root the user names (its
//! pseudobulks as one node) grows the one tree start from it instead
//! ([`tree_start_from_roots`]).
//!
//! [`climb`] runs Adam from every start and keeps the best: under a trunk
//! the tree starts alone (a k-means start has no trunk), each from its own
//! branch times and from each of [`BRANCH_STARTS`], all of them climbed
//! together as one batch ([`Fit::from_starts`]); with the lineages apart
//! the k-means start, then with each lineage turned round in turn, keeping
//! a turn that raises the objective (the start fixes the order; the
//! splicing says which way round fits), then the tree starts too. A root
//! the user names is theirs, not a candidate: its tree start alone is
//! climbed, and only when none grows from it ([`NoTree`]) the k-means
//! starts instead. Every tree's root is returned with where its climb
//! finished, so that roots the data cannot tell apart ([`ROOT_GAP`]) are
//! left to the user.

use super::blocks::GenePrior;
use super::fit::{starts_per_batch, DeviceReads, Fit, SHARE_START};
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
    /// fits, one clock for all ([`Fit::with_trunks`]).
    pub trunk: bool,
    /// The pseudobulks at the root of the trajectory, when the user names
    /// its cells: the one tree start grows from them as one root
    /// ([`tree_start_from_roots`]); empty, every end of the tree is tried.
    pub roots: Vec<usize>,
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

/// `[P × H]` the pseudobulks' scores on the leading axes of their log
/// spliced shares.
fn scores_of(reads: &Reads, h: usize) -> DMatrix<f64> {
    let (y, _) = log_shares(reads);
    axes(&y, h).0
}

/// Every pseudobulk's start lineage and its order within it on `[0, 1]`.
#[cfg(test)]
pub fn start_order(reads: &Reads, lineages: usize, h: usize) -> (Vec<usize>, Vec<f64>) {
    order_of(&scores_of(reads, h), lineages)
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
    geodesic_from(adj, &[from])
}

/// Every node's geodesic distance to the nearest of `sources` (∞ where
/// none is reachable): the sources as one node.
fn geodesic_from(adj: &[Vec<(usize, f64)>], sources: &[usize]) -> Vec<f64> {
    use std::cmp::Reverse;
    use std::collections::BinaryHeap;
    let mut dist = vec![f64::INFINITY; adj.len()];
    let mut heap = BinaryHeap::new();
    for &from in sources {
        dist[from] = 0.0;
        // Non-negative floats order as their bits.
        heap.push(Reverse((0f64.to_bits(), from)));
    }
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
/// as the root (`lineages + 1` ends); none with one lineage, too few
/// pseudobulks, or the graph falling apart.
#[cfg(test)]
pub fn tree_starts(reads: &Reads, h: usize, lineages: usize) -> Vec<Places> {
    tree_places(&scores_of(reads, h), lineages)
}

/// The one tree start from a root the user names: the pseudobulks `roots`
/// as one node (every distance from the root the distance to the nearest
/// of them, so no one of them decides the tree), its tips the `lineages`
/// ends farthest from it (and from one another). A stem in the middle of a
/// tree is no end of it, so no end taken as the root finds it; `None` as
/// [`tree_starts`] is empty, with no root, or when fewer branch.
#[cfg(test)]
pub fn tree_start_from_roots(
    reads: &Reads,
    h: usize,
    lineages: usize,
    roots: &[usize],
) -> Option<Places> {
    tree_place_from_roots(&scores_of(reads, h), lineages, roots).ok()
}

/// A tree needs two lineages and more pseudobulks than its ends.
fn tree_sized(p: usize, lineages: usize) -> bool {
    lineages > 1 && p > lineages + 1
}

/// Every node's geodesic distances to the nearest of `sources`, `None`
/// where the graph falls apart.
fn reaching(adj: &[Vec<(usize, f64)>], sources: &[usize]) -> Option<Vec<f64>> {
    let d = geodesic_from(adj, sources);
    d.iter().all(|d| d.is_finite()).then_some(d)
}

/// `count` ends, each the farthest from `near` (the distance to the ends
/// so far) and then folded into it, with its geodesic distances.
fn farthest_ends(
    adj: &[Vec<(usize, f64)>],
    mut near: Vec<f64>,
    count: usize,
) -> Vec<(usize, Vec<f64>)> {
    (0..count)
        .map(|_| {
            let e = farthest(&near);
            let d = geodesic(adj, e);
            for (n, &x) in near.iter_mut().zip(&d) {
                *n = n.min(x);
            }
            (e, d)
        })
        .collect()
}

/// Why no tree start grows from the root the user names.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NoTree {
    /// One lineage: a tree needs two.
    OneLineage,
    /// No more pseudobulks than the tree's ends.
    TooFew,
    /// The pseudobulks' neighbour graph falls apart, so the root does not
    /// reach them all.
    Apart,
    /// The tree from the root leaves a lineage without a pseudobulk.
    EmptyLineage,
}

impl NoTree {
    /// What a result says of a root named but not grown from: why, how
    /// many of the `p` pseudobulks are the root's (`roots`, `mostly`
    /// whether mostly of its cells, [`root_pseudobulks`]), and what to try
    /// (`lineages` asked).
    #[must_use]
    pub fn note(self, roots: usize, p: usize, mostly: bool, lineages: usize) -> String {
        let (why, fix) = match self {
            Self::OneLineage => (
                "with one lineage there is no tree to grow from a root".to_string(),
                "ask for two or more lineages",
            ),
            Self::TooFew => (
                format!("{p} pseudobulks are too few for a tree of {lineages} lineages"),
                "ask for fewer lineages, or fit more pseudobulks",
            ),
            Self::Apart => (
                "the pseudobulks' neighbour graph falls apart, so the root does not reach them all"
                    .to_string(),
                "fit on cells that form one trajectory (drop a separate type), or name a root \
                 inside the part that holds the trajectory",
            ),
            Self::EmptyLineage => (
                format!("a tree from the root leaves one of the {lineages} lineages empty"),
                "ask for fewer lineages, or name a root nearer an end of the trajectory",
            ),
        };
        let held = if mostly {
            format!("{roots} of {p} pseudobulks are mostly root cells")
        } else {
            format!(
                "no pseudobulk of {p} is mostly root cells, so the root was the one richest in them"
            )
        };
        format!(
            "no tree start grows from this root ({why}; {held}), so the k-means starts ran \
             instead and the trajectory does not grow from these cells, nor need its time run \
             from them: {fix}, and rerun with the same root-cells"
        )
    }
}

fn tree_place_from_roots(
    scores: &DMatrix<f64>,
    lineages: usize,
    roots: &[usize],
) -> std::result::Result<Places, NoTree> {
    if lineages < 2 {
        return Err(NoTree::OneLineage);
    }
    if !tree_sized(scores.nrows(), lineages) {
        return Err(NoTree::TooFew);
    }
    let adj = knn_graph(scores);
    let dr = reaching(&adj, roots).ok_or(NoTree::Apart)?;
    let tips = farthest_ends(&adj, dr.clone(), lineages);
    tree_from_root(&dr, &tips).ok_or(NoTree::EmptyLineage)
}

fn tree_places(scores: &DMatrix<f64>, lineages: usize) -> Vec<Places> {
    if !tree_sized(scores.nrows(), lineages) {
        return Vec::new();
    }
    let adj = knn_graph(scores);
    let Some(d0) = reaching(&adj, &[0]) else {
        return Vec::new();
    };
    // The ends, farthest first: the farthest from anywhere, then each the
    // farthest from the ends so far.
    let e1 = farthest(&d0);
    let d1 = geodesic(&adj, e1);
    let mut ends = vec![(e1, d1.clone())];
    ends.extend(farthest_ends(&adj, d1, lineages));
    (0..ends.len())
        .filter_map(|r| {
            let mut tips = ends.clone();
            let (_, dr) = tips.remove(r);
            tree_from_root(&dr, &tips)
        })
        .collect()
}

/// The tree from the root (`dr` its distances) to the `tips`, tip 0's way
/// the spine (lineage 0) every other lineage parts from: a pseudobulk's
/// Gromov product with a tip is how far it runs along the way to that
/// tip, so it is on the lineage whose way it shares longest, and lineage
/// `a` parts from the spine where its tip's way does; a pseudobulk before
/// two lineages' parting is on both.
fn tree_from_root(dr: &[f64], tips: &[(usize, Vec<f64>)]) -> Option<Places> {
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
    let mut by_progress: Vec<usize> = (0..p).collect();
    by_progress.sort_by(|&i, &j| progress[i].total_cmp(&progress[j]));
    // One clock: on every lineage, the pseudobulks it shares come first,
    // their share of all pseudobulks their share of time, its own after.
    let mut places: Places = vec![Vec::new(); p];
    for a in 0..l {
        let (with, own): (Vec<usize>, Vec<usize>) = by_progress
            .iter()
            .copied()
            .filter(|&x| on[x] == a || shared(x, a))
            .partition(|&x| (0..l).any(|b| shared(x, b)));
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
pub(crate) fn branch_times(places: &Places, lineages: usize) -> Vec<f64> {
    (1..lineages)
        .map(|l| {
            places
                .iter()
                .filter(|at| at.iter().any(|&(m, _)| m == 0) && at.iter().any(|&(m, _)| m == l))
                .flat_map(|at| at.iter().map(|&(_, t)| t))
                .reduce(f64::max)
                .unwrap_or(NO_TRUNK)
        })
        .collect()
}

/// The branch time of a start with no trunk: the lineages part at once.
const NO_TRUNK: f64 = 0.05;

/// A pseudobulk is at the root when at least this share of its cells are
/// the named root cells.
const ROOT_SHARE: f32 = 0.5;

/// The root pseudobulks from the cells the user names: of the pseudobulks
/// (`pbs` their indices in `member`, `sizes` their cells) those mostly of
/// named cells ([`ROOT_SHARE`]; `true` with them), or, none being, the one
/// richest in them (its named cells times its share of them, so a large
/// pseudobulk mostly of them beats a tiny pure one); empty when none holds
/// one.
pub fn root_pseudobulks(
    named: &std::collections::HashSet<Box<str>>,
    cells: &[Box<str>],
    member: &[usize],
    pbs: &[usize],
    sizes: &[f32],
) -> (Vec<usize>, bool) {
    let row_of: std::collections::HashMap<usize, usize> =
        pbs.iter().enumerate().map(|(c, &p)| (p, c)).collect();
    let mut hits = vec![0f32; pbs.len()];
    for (_, &p) in cells.iter().zip(member).filter(|(c, _)| named.contains(*c)) {
        if let Some(&c) = row_of.get(&p) {
            hits[c] += 1.0;
        }
    }
    let mostly: Vec<usize> = (0..pbs.len())
        .filter(|&c| hits[c] >= ROOT_SHARE * sizes[c] && hits[c] > 0.0)
        .collect();
    if !mostly.is_empty() {
        return (mostly, true);
    }
    let richest = (0..pbs.len())
        .filter(|&c| hits[c] > 0.0)
        .max_by(|&a, &b| (hits[a] * hits[a] / sizes[a]).total_cmp(&(hits[b] * hits[b] / sizes[b])));
    (richest.into_iter().collect(), false)
}

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

/// Under a trunk every tree start is also tried with every lineage parting
/// here: from its own branch times alone the climb can settle on a branch
/// far from the truth (on simulated trees, at twice its time).
const BRANCH_STARTS: [f64; 3] = [0.2, 0.35, 0.5];

/// With no root named, trees from other roots finishing within this share
/// of the best objective leave the direction to the data's whim: the run
/// warns, and leaves the root to the user as a decision in its manifest.
/// A share, not a count of nats: the objective treats a pseudobulk's reads
/// as independent draws, so its nats overstate the evidence, in proportion
/// to the reads.
pub const ROOT_GAP: f32 = 1e-3;

/// The share of a tree start's pseudobulks, earliest first, that stand for
/// its root ([`TriedRoot::pbs`]).
const ROOT_REGION: f64 = 0.05;

/// A tree start's root and where its climb finished.
pub struct TriedRoot {
    /// The pseudobulks at the root: the start's earliest
    /// [`ROOT_REGION`], at least one.
    pub pbs: Vec<usize>,
    pub objective: f32,
}

/// The climb's best fit and its objective, and, with no root named, every
/// tree start's root as it finished.
pub struct Climbed {
    pub fit: Fit,
    pub objective: f32,
    pub roots: Vec<TriedRoot>,
    /// The root of the fit, an index of `roots`: the tree start it grew
    /// from; `None` when a k-means start won.
    pub root: Option<usize>,
    /// Why no tree start grew from the root the user named, the k-means
    /// starts run instead; `None` with no root named or one grown from.
    pub unrooted: Option<NoTree>,
}

/// The pseudobulks at a tree start's root: the earliest [`ROOT_REGION`] of
/// them by their earliest place.
fn root_region(places: &Places) -> Vec<usize> {
    let first = |at: &Vec<(usize, f64)>| at.iter().map(|&(_, t)| t).fold(f64::INFINITY, f64::min);
    let mut order: Vec<usize> = (0..places.len()).collect();
    order.sort_by(|&a, &b| first(&places[a]).total_cmp(&first(&places[b])));
    order.truncate(((ROOT_REGION * places.len() as f64).ceil() as usize).max(1));
    order
}

/// Whether a start's objective `new` beats the best so far: a NaN (a climb
/// that diverged) never does, and any finite one beats a NaN best.
pub fn beats(new: f32, best: f32) -> bool {
    new > best || (best.is_nan() && !new.is_nan())
}

/// The gap, as a share of the best, from the best of the trees' finished
/// objectives to the runner-up (from another root); `None` with fewer than
/// two finite.
pub fn root_gap(finished: &[f32]) -> Option<f32> {
    let mut finite: Vec<f32> = finished.iter().copied().filter(|o| o.is_finite()).collect();
    finite.sort_by(|a, b| b.total_cmp(a));
    match finite[..] {
        [best, next, ..] => Some((best - next) / best.abs().max(f32::MIN_POSITIVE)),
        _ => None,
    }
}

/// Adam from every start, the best fit and its objective: under a trunk
/// the tree starts, one per end taken as the root or the one from the
/// named root (a k-means start has no trunk, and its lineages turned round
/// one at a time cannot share one), each from its own branch times and
/// from every one of [`BRANCH_STARTS`], all climbed to the end together
/// ([`climb_all`]) and each tree's best kept, with its root; apart, the k-means start
/// and each lineage turned round in turn (a turn kept when it raises the
/// objective), then the trees too. A root the user names is theirs: its
/// tree start alone climbs. With no tree start to be had (one lineage, the
/// graph falling apart) the k-means starts run either way.
pub fn climb(reads: &Reads, opts: &Climb, dev: &Device) -> Result<Climbed> {
    let prior = GridPrior::new(opts.points, 1.0, opts.kappa)
        .map_err(|e| legume_numeric::candle::candle_core::Error::Msg(e.to_string()))?;
    let scores = scores_of(reads, opts.dim);
    let trunk = opts.trunk && opts.lineages > 1;
    let on_dev = DeviceReads::new(reads, dev)?;
    let climbed = |starts: &[Params], at: Option<&[Vec<f64>]>| {
        climb_all(starts, at, &on_dev, &prior, opts, dev)
    };
    // A k-means start, on the trunk when one is asked for.
    let run = |places: &Places| -> Result<(Fit, f32)> {
        let at = trunk.then(|| vec![branch_times(places, opts.lineages)]);
        let mut one = climbed(&[params_from(reads, opts, places)], at.as_deref())?;
        Ok((one.fits.swap_remove(0), one.objectives[0]))
    };
    let named = !opts.roots.is_empty();
    let (trees, unrooted) = if named {
        match tree_place_from_roots(&scores, opts.lineages, &opts.roots) {
            Ok(tree) => (vec![tree], None),
            Err(why) => {
                log::warn!(
                    "ode climb: no tree start grows from the root named ({why:?}); the k-means starts run instead"
                );
                (Vec::new(), Some(why))
            }
        }
    } else {
        (tree_places(&scores, opts.lineages), None)
    };
    let mut tried: Vec<TriedRoot> = Vec::with_capacity(trees.len());
    let mut best: Option<(Fit, f32)> = None;
    // Whether `other` is the best so far, kept.
    let mut keep = |other: (Fit, f32)| {
        let better = best.as_ref().is_none_or(|b| beats(other.1, b.1));
        if better {
            best = Some(other);
        }
        better
    };
    let mut root = None;
    // A named root's tree start races no k-means start.
    if trees.is_empty() || (!opts.trunk && !named) {
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
        keep(best_kmeans);
    }
    if !trees.is_empty() {
        // Every tree's starts side by side: on a trunk from its own branch
        // times and from each of BRANCH_STARTS; apart, the one.
        let per_tree = if trunk { 1 + BRANCH_STARTS.len() } else { 1 };
        let mut starts = Vec::with_capacity(trees.len() * per_tree);
        let mut at = Vec::new();
        for places in &trees {
            if trunk {
                at.push(branch_times(places, opts.lineages));
                at.extend(BRANCH_STARTS.iter().map(|&tb| vec![tb; opts.lineages - 1]));
            }
            starts.extend(std::iter::repeat_n(
                params_from(reads, opts, places),
                per_tree,
            ));
        }
        let all = climbed(&starts, trunk.then_some(at.as_slice()))?;
        let mut top: Option<usize> = None;
        for (t, (places, own)) in trees
            .iter()
            .zip(all.objectives.chunks(per_tree))
            .enumerate()
        {
            let k = (0..per_tree)
                .reduce(|a, b| if beats(own[b], own[a]) { b } else { a })
                .expect("a start a tree");
            log::info!("ode climb: a tree start finished at {}", own[k]);
            tried.push(TriedRoot {
                pbs: root_region(places),
                objective: own[k],
            });
            if top.is_none_or(|b| beats(own[k], all.objectives[b])) {
                top = Some(t * per_tree + k);
            }
        }
        if let Some(b) = top {
            if keep((all.start(b)?, all.objectives[b])) {
                root = Some(b / per_tree);
            }
        }
    }
    let (fit, objective) = best.expect("a start");
    if objective.is_nan() {
        return Err(legume_numeric::candle::candle_core::Error::Msg(
            "ode-mixture climb: every start diverged (objective NaN)".into(),
        ));
    }
    Ok(Climbed {
        fit,
        objective,
        roots: tried,
        root,
        unrooted,
    })
}

/// Starts climbed together ([`climb_all`]): the batches' fits and every
/// start's objective, in order.
struct Climbs {
    fits: Vec<Fit>,
    per_batch: usize,
    objectives: Vec<f32>,
}

impl Climbs {
    /// Start `b` alone.
    fn start(&self, b: usize) -> Result<Fit> {
        self.fits[b / self.per_batch].start(b % self.per_batch)
    }
}

/// Every one of `starts` climbed `opts.steps` together, on a trunk from
/// `branch_at` (a list a start) or apart, in batches of
/// [`starts_per_batch`].
fn climb_all(
    starts: &[Params],
    branch_at: Option<&[Vec<f64>]>,
    reads: &DeviceReads,
    prior: &GridPrior,
    opts: &Climb,
    dev: &Device,
) -> Result<Climbs> {
    let (p, g) = reads.shape();
    let positions = (opts.points - 1) * opts.sub.max(1) + 1;
    let per_batch = starts_per_batch(opts.lineages, positions, p, g);
    let mut at = branch_at.map(|a| a.chunks(per_batch));
    let mut fits = Vec::new();
    let mut objectives = Vec::with_capacity(starts.len());
    for batch in starts.chunks(per_batch) {
        let mut fit =
            Fit::from_starts(batch, prior, dev)?.with_share(opts.jitter.then_some(SHARE_START))?;
        if let Some(at) = at.as_mut().and_then(Iterator::next) {
            fit = fit.with_trunks(batch, prior, at)?;
        }
        objectives.extend(fit.adam(reads, opts.steps, opts.lr, None)?);
        fits.push(fit);
    }
    Ok(Climbs {
        fits,
        per_batch,
        objectives,
    })
}
