//! The rank arithmetic of a critique: parameter checks, the committee and
//! its answers, merges, far shares and k-means++ seeding.

use legume_numeric::matrix::table::TableReader;
use rand::distr::weighted::WeightedIndex;
use rand::distr::Distribution;
use rand::rngs::SmallRng;
use rand::seq::IndexedRandom;
use rayon::prelude::*;
use rustc_hash::FxHashSet;
pub(crate) use senna::cell_labels::{label_composition, tolerant_align};

pub(crate) fn check_params(knn: usize, far_frac: f64, min_cells: usize) -> anyhow::Result<()> {
    anyhow::ensure!(knn >= 1, "--knn must be at least 1");
    anyhow::ensure!(
        far_frac > 0.0 && far_frac < 1.0,
        "--far-frac must lie strictly between 0 and 1 (got {far_frac})"
    );
    anyhow::ensure!(min_cells >= 1, "--min-cells must be at least 1");
    Ok(())
}

/// Rank of a point at distance `d` among `sorted` (ascending) distances: one
/// plus the number strictly closer.
pub(crate) fn rank_of(sorted: &[f32], d: f32) -> u32 {
    sorted.partition_point(|&x| x < d) as u32 + 1
}

/// Near is rank ≤ `near`; far is rank > `far`.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(crate) struct Bounds {
    pub near: u32,
    pub far: u32,
}

impl Bounds {
    /// `far = max(2·knn, ⌈P·frac⌉)`, well clear of near. The `P·frac` part does
    /// the work: a far set by `knn` alone does not carry over between samples
    /// of different sizes, while a share of the level does; `2·knn` only
    /// guards tiny levels. `None` when no rank
    /// can lie beyond it: ranks run from 1 to `P − 1`.
    pub(crate) fn for_level(n_pb: usize, knn: usize, frac: f64) -> Option<Self> {
        let far = (2 * knn).max((n_pb as f64 * frac).ceil() as usize);
        (n_pb.saturating_sub(1) > far).then_some(Bounds {
            near: knn as u32,
            far: far as u32,
        })
    }
}

/// The known ranks of pair `c` over the given models (`u32::MAX` is unknown),
/// sorted.
pub(crate) fn known_ranks(ranks: &[Vec<u32>], models: &[usize], c: usize) -> Vec<u32> {
    let mut known: Vec<u32> = models
        .iter()
        .map(|&o| ranks[o][c])
        .filter(|&r| r != u32::MAX)
        .collect();
    known.sort_unstable();
    known
}

pub(crate) fn median(sorted: &[u32]) -> Option<f64> {
    let m = sorted.len();
    match m {
        0 => None,
        _ if m % 2 == 1 => Some(f64::from(sorted[m / 2])),
        _ => Some(0.5 * (f64::from(sorted[m / 2 - 1]) + f64::from(sorted[m / 2]))),
    }
}

/// The models that answer `model`: all the others when `size` is 0 or covers
/// them; otherwise a random subset of `size` of them (query by bagging), sorted.
pub(crate) fn committee(
    n_models: usize,
    model: usize,
    size: usize,
    rng: &mut SmallRng,
) -> Vec<usize> {
    let others: Vec<usize> = (0..n_models).filter(|&o| o != model).collect();
    if size == 0 || size >= others.len() {
        return others;
    }
    let mut chosen: Vec<usize> = others.sample(rng, size).copied().collect();
    chosen.sort_unstable();
    chosen
}

/// Per pair, the committee's answer: the median of its members' known ranks
/// (`u32::MAX` is unknown). The asking model is never on its own committee.
pub(crate) fn answers(ranks: &[Vec<u32>], committee: &[usize]) -> Vec<Option<f64>> {
    let n = ranks.first().map_or(0, Vec::len);
    (0..n)
        .map(|c| median(&known_ranks(ranks, committee, c)))
        .collect()
}

/// Which pairs the model merges: it keeps the pair near while the answer holds
/// it far.
pub(crate) fn merges(own: &[u32], answers: &[Option<f64>], b: Bounds) -> Vec<bool> {
    own.iter()
        .zip(answers)
        .map(|(&r, a)| r <= b.near && a.is_some_and(|o| o > f64::from(b.far)))
        .collect()
}

/// The share of the committee members with a known rank for pair `c` that hold
/// it far: a merge's label weight.
pub(crate) fn far_share(ranks: &[Vec<u32>], committee: &[usize], c: usize, b: Bounds) -> f32 {
    let known = known_ranks(ranks, committee, c);
    if known.is_empty() {
        return 0.0;
    }
    known.iter().filter(|&&r| r > b.far).count() as f32 / known.len() as f32
}

/// How far apart two pairs are: their endpoints' distances, matched the better
/// way round.
pub(crate) fn pair_distance(
    d: &impl Fn(usize, usize) -> f32,
    p: (usize, usize),
    q: (usize, usize),
) -> f32 {
    (d(p.0, q.0) + d(p.1, q.1)).min(d(p.0, q.1) + d(p.1, q.0))
}

/// k-means++ seeding: draw up to `n` items, each with probability proportional
/// to its weight times its squared distance to the items already drawn (the
/// weight alone for the first). Items with no weight are never drawn; drawing
/// stops early when nothing is left to draw.
pub(crate) fn kmeanspp(
    weights: &[f64],
    dist: &(impl Fn(usize, usize) -> f64 + Sync),
    n: usize,
    rng: &mut SmallRng,
) -> Vec<usize> {
    let mut picked: Vec<usize> = Vec::new();
    // Squared distance to the nearest item drawn so far: unbounded before the
    // first draw, when the weight alone counts, and 0 for a drawn item, so it
    // is never drawn again.
    let mut near2 = vec![f64::INFINITY; weights.len()];
    while picked.len() < n {
        let score: Vec<f64> = weights
            .iter()
            .zip(&near2)
            .map(|(w, &d2)| w.max(0.0) * if picked.is_empty() { 1.0 } else { d2 })
            .collect();
        // `Err` when nothing has weight, or the total is not finite.
        let Ok(draw) = WeightedIndex::new(&score) else {
            break;
        };
        let i = draw.sample(rng);
        picked.push(i);
        near2.par_iter_mut().enumerate().for_each(|(j, slot)| {
            let dj = dist(i, j);
            *slot = slot.min(dj * dj);
        });
    }
    picked
}

/// Pairs in a model's top-`k` (unknown ranks never count).
pub(crate) fn near_count(own: &[u32], b: Bounds) -> usize {
    own.iter().filter(|&&r| r <= b.near).count()
}

/// Merges over near pairs; `NaN` without near pairs.
pub(crate) fn merge_rate(merges: usize, near: usize) -> f32 {
    if near == 0 {
        f32::NAN
    } else {
        merges as f32 / near as f32
    }
}

/// `base`, `base_2`, … until every name is unique, including against names
/// that already carry a suffix.
pub(crate) fn unique_names(bases: Vec<String>) -> Vec<String> {
    let mut used: FxHashSet<String> = FxHashSet::default();
    bases
        .into_iter()
        .map(|base| {
            let mut name = base.clone();
            let mut n = 1;
            while used.contains(&name) {
                n += 1;
                name = format!("{base}_{n}");
            }
            used.insert(name.clone());
            name
        })
        .collect()
}

/// `cell → label` from a table with a header (parquet, or delimited text,
/// gzip or not): the first column names the cell, `column` (else the second
/// column) holds the label. Missing labels (empty, `NA`, `NaN`) are left out.
pub(crate) fn read_cell_labels<M: FromIterator<(Box<str>, Box<str>)>>(
    path: &str,
    column: Option<&str>,
) -> anyhow::Result<M> {
    let table = TableReader::open(path)?;
    let at = match column {
        Some(c) => table.select(&[c])?[0],
        None => 1,
    };
    table
        .rows(&[0, at])?
        .filter_map(|row| {
            row.map(|row| {
                let [cell, label]: [Box<str>; 2] = row.try_into().expect("two columns asked for");
                (!matches!(&*label, "" | "NA" | "NaN")).then_some((cell, label))
            })
            .transpose()
        })
        .collect()
}

/// The shared mass of two label compositions, `Σ_k min(a_k, b_k)`: 1 for the
/// same mix, 0 for no label in common.
pub(crate) fn overlap(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x.min(*y)).sum()
}

/// Mean of the finite `values` where `pick`; `NaN` when there is none.
pub(crate) fn mean_over(values: &[f32], pick: &[bool]) -> f32 {
    let (sum, n) = values
        .iter()
        .zip(pick)
        .filter(|&(v, &p)| p && v.is_finite())
        .fold((0.0f32, 0usize), |(s, n), (v, _)| (s + v, n + 1));
    if n == 0 {
        f32::NAN
    } else {
        sum / n as f32
    }
}
