//! `senna critique` — fits that question each other: a report, nothing trained.
//!
//! Several fits of the same cells place pseudobulks differently. This command
//! finds the pseudobulk pairs on which they disagree, entirely in the models'
//! latent spaces: no gene counts are read. Nothing is trained and no model is
//! changed.
//!
//! 1. **One partition.** The cell → pseudobulk partition of one run (`--partition`)
//!    defines the pseudobulks at every level. The models need not have trained on
//!    it: each model's view is its own cell latent averaged over the partition's
//!    pseudobulks, so independently fitted runs compare as well.
//! 2. **Views.** Per model and level, the averaged latent in the metric its
//!    [`CellSpace`] calls for: Hellinger on θ, cosine on an embedding, Euclidean
//!    on z-scored signed scores.
//! 3. **Candidates.** The union of every model's top-`k` pseudobulk pairs. Each pair
//!    gets a rank in every model: the smaller of its two directional ranks.
//! 4. **Close and far.** Near is rank ≤ `k`; far is beyond `max(2k, P/4)` by
//!    default, well clear of near.
//! 5. **Merges.** Each model is judged against the median rank of the *other*
//!    models, so its own rank never votes. It *merges* a pair it keeps near while
//!    the others hold it far; the critique is "push these two apart". The reverse,
//!    holding far what the others keep near, is not a critique: checked against
//!    expert cell types, the lone model that separates is usually right.
//! 6. **Report card.** A model's merge rate is its merges over its near pairs:
//!    high, and it lumps together states the others separate (a topic model's
//!    resolution limit, a VAE's mode collapse).
//!
//! With `--cell-labels`, each pair's label-composition overlap checks the merges:
//! a merged pair should share few labels compared with the model's near pairs.
//!
//! Levels are numbered as in `cell_to_pb.parquet`: `0` is the coarsest.

use anyhow::Context;
use legume_numeric::matrix::knn::all_pairs::knn_rows_l2;
use legume_numeric::matrix::knn::metric::l2_sq;
use legume_numeric::matrix::parquet::{write_named_table, Column};
use log::warn;
use rand::distr::weighted::WeightedIndex;
use rand::distr::Distribution;
use rand::rngs::SmallRng;
use rand::seq::IndexedRandom;
use rand::SeedableRng;
use rayon::prelude::*;
use rustc_hash::{FxHashMap, FxHashSet};
use senna::embed_common::*;
use senna::run_manifest::{
    load_cell_to_pb_raw, resolve, CellSpace, InheritedPartition, RunManifest,
};
use std::path::{Path, PathBuf};

#[derive(Args, Debug)]
pub struct CritiqueArgs {
    #[arg(
        required = true,
        num_args = 2..,
        value_delimiter = ',',
        help = "Run manifests to compare ({run}.senna.json or a run prefix); two or more",
        long_help = "The fits whose pseudobulk views are compared.\n\
                     Any kind that writes a per-cell latent or cell embedding:\n\
                     topic, masked-*, vae, svd, bge, simba, gem."
    )]
    runs: Vec<Box<str>>,

    #[arg(
        long,
        help = "Run whose cell → pseudobulk partition is used (default: the first model that has one)",
        long_help = "A run that wrote {out}.cell_to_pb.parquet (topic, masked-*, vae).\n\
                     The models need not have trained on this partition.\n\
                     Default: the first model that has one."
    )]
    partition: Option<Box<str>>,

    #[arg(
        long,
        short,
        required = true,
        help = "Output prefix",
        long_help = "Writes:\n  \
                     {out}.critique.pairs.parquet           one row per candidate pair\n  \
                     {out}.critique.summary.parquet         merges and merge rate per model and level\n  \
                     {out}.critique.labels.{model}.parquet  with --questions: pairs to push apart\n  \
                     {out}.critique.json                    inputs and parameters"
    )]
    out: Box<str>,

    #[arg(
        long,
        default_value_t = 15,
        help = "Neighbours per pseudobulk in each model's view; near is rank ≤ this"
    )]
    knn: usize,

    #[arg(
        long,
        default_value_t = 0.25,
        help = "Far is beyond rank max(2·knn, this fraction of the pseudobulks)",
        long_help = "Far is beyond rank max(2·knn, this fraction of the pseudobulks).\n\
                     Near is rank ≤ --knn; far must be well clear of it, so that one\n\
                     model at rank 15 and another at 16 is not a disagreement.\n\
                     A level too small to hold a rank beyond far is skipped."
    )]
    far_frac: f64,

    #[arg(
        long,
        default_value_t = 10,
        help = "Pseudobulks with fewer cells are left out"
    )]
    min_cells: usize,

    #[arg(
        long,
        default_value_t = 0,
        help = "Models answering each model: a random subset of this size (0 = all others)",
        long_help = "Each model is answered by the median rank of its committee.\n\
                     0 takes every other model. A smaller committee is drawn at\n\
                     random per model and level (query by bagging), so an odd\n\
                     model only sometimes answers."
    )]
    committee: usize,

    #[arg(
        long,
        default_value_t = 0,
        help = "Questions drawn per model and level; their merges become labels (0 = none)",
        long_help = "Draws a model's questions among its near pairs by k-means++\n\
                     seeding: weight by disagreement (how far the answer lies beyond\n\
                     the model's own rank) squared, times squared distance to the\n\
                     questions already drawn. The drawn questions the answer holds\n\
                     far are written to {out}.critique.labels.{model}.parquet,\n\
                     the pairs that model is to push apart."
    )]
    questions: usize,

    #[arg(
        long,
        default_value_t = 42,
        help = "Seed for committees and question draws"
    )]
    seed: u64,

    #[arg(
        long,
        default_value_t = 20_000,
        help = "Levels with more pseudobulks than this are skipped"
    )]
    max_pb: usize,

    #[arg(
        long,
        requires = "label_column",
        help = "Per-cell labels to check the critique against (TSV, may be gzipped)",
        long_help = "A tab-separated table with a header (gzipped is fine); the first\n\
                     column names the cell, as in the partition. Each pseudobulk is\n\
                     described by its label composition, and each pair by the overlap\n\
                     of the two compositions: 1 for the same mix of labels, 0 for no\n\
                     label in common. A merge is borne out when its pair's overlap is\n\
                     low against the model's other near pairs. Labels only check the\n\
                     merges; they never change them."
    )]
    cell_labels: Option<Box<str>>,

    #[arg(
        long,
        requires = "cell_labels",
        help = "Column of --cell-labels holding the label (e.g. CellType)"
    )]
    label_column: Option<Box<str>>,
}

pub(crate) fn check_params(knn: usize, far_frac: f64, min_cells: usize) -> anyhow::Result<()> {
    anyhow::ensure!(knn >= 1, "--knn must be at least 1");
    anyhow::ensure!(
        far_frac > 0.0 && far_frac < 1.0,
        "--far-frac must lie strictly between 0 and 1 (got {far_frac})"
    );
    anyhow::ensure!(min_cells >= 1, "--min-cells must be at least 1");
    Ok(())
}

/// Position of every source name in `target`, or `usize::MAX` when absent.
/// Identical lists match by position, repeats included. Otherwise a name
/// that would be matched is refused when it repeats on either side, since no
/// order says which row it means; a repeat that matches nothing is harmless.
pub(crate) fn tolerant_align(
    source: &[Box<str>],
    target: &[Box<str>],
) -> anyhow::Result<Vec<usize>> {
    if source == target {
        return Ok((0..source.len()).collect());
    }
    // A name repeated in `target` maps to `REPEATED`, refused only if looked up.
    const REPEATED: usize = usize::MAX - 1;
    let mut at: FxHashMap<&str, usize> =
        FxHashMap::with_capacity_and_hasher(target.len(), Default::default());
    for (i, n) in target.iter().enumerate() {
        at.entry(n.as_ref())
            .and_modify(|j| *j = REPEATED)
            .or_insert(i);
    }
    let mut seen = vec![false; target.len()];
    source
        .iter()
        .map(|n| match at.get(n.as_ref()).copied() {
            None => Ok(usize::MAX),
            Some(REPEATED) => Err(repeated(n)),
            Some(j) if std::mem::replace(&mut seen[j], true) => Err(repeated(n)),
            Some(j) => Ok(j),
        })
        .collect()
}

fn repeated(name: &str) -> anyhow::Error {
    anyhow::anyhow!(
        "cell name {name} appears more than once, so cells cannot be matched by name; \
         make the names unique (e.g. suffix each sample's barcodes)"
    )
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

/// `cell → label` from a delimited table with a header: the first column names
/// the cell, `column` holds the label. Empty and `NA` labels are left out.
pub(crate) fn parse_cell_labels(
    reader: impl std::io::BufRead,
    column: &str,
) -> anyhow::Result<FxHashMap<Box<str>, Box<str>>> {
    let mut lines = reader.lines();
    let header = lines
        .next()
        .ok_or_else(|| anyhow::anyhow!("the label table is empty"))??;
    let at = header
        .split('\t')
        .position(|h| h.trim() == column)
        .ok_or_else(|| anyhow::anyhow!("no column `{column}` in the label table's header"))?;
    let mut out = FxHashMap::default();
    for line in lines {
        let line = line?;
        let fields: Vec<&str> = line.split('\t').collect();
        let (Some(cell), Some(label)) = (fields.first(), fields.get(at)) else {
            continue;
        };
        let label = label.trim();
        if !label.is_empty() && label != "NA" {
            out.insert(Box::from(cell.trim()), Box::from(label));
        }
    }
    Ok(out)
}

/// Per pseudobulk, the fraction of its labelled cells carrying each label;
/// `None` for a pseudobulk with no labelled cell.
pub(crate) fn label_composition(
    pb_of_cell: &[usize],
    label_of_cell: &[Option<u32>],
    n_pb: usize,
    n_labels: usize,
) -> Vec<Option<Vec<f32>>> {
    let mut count = vec![vec![0u32; n_labels]; n_pb];
    for (&pb, label) in pb_of_cell.iter().zip(label_of_cell) {
        if let (true, Some(k)) = (pb != usize::MAX, label) {
            count[pb][*k as usize] += 1;
        }
    }
    count
        .into_iter()
        .map(|c| {
            let total: u32 = c.iter().sum();
            (total > 0).then(|| c.iter().map(|&n| n as f32 / total as f32).collect())
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

/// The partition cells' labels, as indices into a sorted vocabulary.
struct CellLabels {
    of_cell: Vec<Option<u32>>,
    n_labels: usize,
}

fn load_cell_labels(
    path: &str,
    column: &str,
    part_cells: &[Box<str>],
) -> anyhow::Result<CellLabels> {
    let map = parse_cell_labels(
        legume_numeric::matrix::common_io::open_buf_reader(path)?,
        column,
    )?;
    let mut vocab: Vec<Box<str>> = map.values().cloned().collect();
    vocab.sort_unstable();
    vocab.dedup();
    let index: FxHashMap<&str, u32> = vocab
        .iter()
        .enumerate()
        .map(|(i, v)| (v.as_ref(), i as u32))
        .collect();
    let of_cell: Vec<Option<u32>> = part_cells
        .iter()
        .map(|c| map.get(c).map(|l| index[l.as_ref()]))
        .collect();
    let n = of_cell.iter().filter(|l| l.is_some()).count();
    info!(
        "Cell labels `{column}`: {} labels; {n} of the partition's {} cells labelled",
        vocab.len(),
        part_cells.len()
    );
    anyhow::ensure!(n > 0, "no partition cell is named in {path}");
    Ok(CellLabels {
        of_cell,
        n_labels: vocab.len(),
    })
}

/// The pseudobulks kept at one level.
struct Level {
    /// Partition cell → kept pseudobulk index, or `usize::MAX`.
    pb_of_cell: Vec<usize>,
    /// Kept index → the partition's own pseudobulk id.
    pb_id: Vec<usize>,
    n_cells: Vec<usize>,
}

impl Level {
    fn new(assignment: &[usize], min_cells: usize) -> Self {
        let n_pb = assignment
            .iter()
            .filter(|&&p| p != usize::MAX)
            .max()
            .map_or(0, |&p| p + 1);
        let mut count = vec![0usize; n_pb];
        for &p in assignment {
            if p != usize::MAX {
                count[p] += 1;
            }
        }
        let mut kept_of = vec![usize::MAX; n_pb];
        let mut pb_id = Vec::new();
        let mut n_cells = Vec::new();
        for (p, &c) in count.iter().enumerate() {
            if c >= min_cells {
                kept_of[p] = pb_id.len();
                pb_id.push(p);
                n_cells.push(c);
            }
        }
        let pb_of_cell = assignment
            .iter()
            .map(|&p| if p == usize::MAX { p } else { kept_of[p] })
            .collect();
        Level {
            pb_of_cell,
            pb_id,
            n_cells,
        }
    }

    fn n_pb(&self) -> usize {
        self.pb_id.len()
    }
}

/// One model's pseudobulk view at one level: only the pseudobulks it has cells
/// in, already in its metric, so Euclidean distance is the right distance.
pub(crate) struct View {
    /// Kept pseudobulk index of each row of `x`.
    pbs: Vec<usize>,
    /// Kept pseudobulk index → row of `x`, or `usize::MAX`.
    slot: Vec<usize>,
    x: Mat,
}

impl View {
    /// Keep the rows of a `P × d` matrix that are marked `valid` and finite. A
    /// non-finite row says nothing about distance, so it leaves the view.
    pub(crate) fn compact(full: Mat, valid: &[bool]) -> Self {
        let pbs: Vec<usize> = (0..valid.len())
            .filter(|&i| valid[i] && full.row(i).iter().all(|v| v.is_finite()))
            .collect();
        let mut slot = vec![usize::MAX; valid.len()];
        for (row, &pb) in pbs.iter().enumerate() {
            slot[pb] = row;
        }
        View {
            x: full.select_rows(&pbs),
            pbs,
            slot,
        }
    }

    /// The model's latent (θ already exponentiated for a simplex) averaged over
    /// each pseudobulk of `level`, in the metric of `space`.
    fn build(latent: &Mat, row_of_cell: &[usize], space: CellSpace, level: &Level) -> Self {
        let p = level.n_pb();
        let mut sum = Mat::zeros(p, latent.ncols());
        let mut count = vec![0usize; p];
        for (&pb, &row) in level.pb_of_cell.iter().zip(row_of_cell) {
            if pb != usize::MAX && row != usize::MAX {
                count[pb] += 1;
                let mut s = sum.row_mut(pb);
                s += latent.row(row);
            }
        }
        for (i, &c) in count.iter().enumerate() {
            if c > 0 {
                let mut r = sum.row_mut(i);
                r /= c as f32;
            }
        }
        let valid: Vec<bool> = count.iter().map(|&c| c > 0).collect();
        let mut view = View::compact(sum, &valid);
        match space {
            CellSpace::LogSimplex => view.x.apply(|v| *v = v.max(0.0).sqrt()),
            CellSpace::Embedding => l2_normalize_rows_inplace(&mut view.x),
            CellSpace::Signed => {
                for mut col in view.x.column_iter_mut() {
                    let n = col.len().max(1) as f32;
                    let mean = col.sum() / n;
                    let sd = (col.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / n)
                        .sqrt()
                        .max(1e-12);
                    col.apply(|v| *v = (*v - mean) / sd);
                }
            }
        }
        view
    }

    /// Distance between two kept pseudobulks in this view; `+∞` when either is
    /// not in it.
    pub(crate) fn dist(&self, a: usize, b: usize) -> f32 {
        let (sa, sb) = (self.slot[a], self.slot[b]);
        if sa == usize::MAX || sb == usize::MAX {
            return f32::INFINITY;
        }
        // No temporary: the rows are strided, so sum the squares in place.
        self.x
            .row(sa)
            .iter()
            .zip(self.x.row(sb).iter())
            .map(|(p, q)| (p - q) * (p - q))
            .sum::<f32>()
            .sqrt()
    }

    /// Every row's top-`k` pairs, as `(min, max)` kept indices.
    pub(crate) fn top_k_pairs(&self, k: usize) -> Vec<(u32, u32)> {
        let (nbrs, _) = knn_rows_l2(&self.x, k);
        nbrs.iter()
            .enumerate()
            .flat_map(|(i, js)| {
                let a = self.pbs[i];
                js.iter().map(move |&j| {
                    let b = self.pbs[j];
                    (a.min(b) as u32, a.max(b) as u32)
                })
            })
            .collect()
    }

    /// Every candidate's rank: the smaller of the two directional ranks, or
    /// `u32::MAX` when the model has no cells in either pseudobulk.
    pub(crate) fn pair_ranks(&self, pairs: &[(u32, u32)]) -> Vec<u32> {
        let n = self.x.nrows();
        let d = self.x.ncols();
        // One contiguous column per pseudobulk.
        let xt = self.x.transpose();
        let col = |i: usize| &xt.as_slice()[i * d..(i + 1) * d];
        let mut by_row: Vec<Vec<(usize, usize)>> = vec![Vec::new(); n];
        for (c, &(a, b)) in pairs.iter().enumerate() {
            let (sa, sb) = (self.slot[a as usize], self.slot[b as usize]);
            if sa != usize::MAX && sb != usize::MAX {
                by_row[sa].push((sb, c));
                by_row[sb].push((sa, c));
            }
        }
        let directional: Vec<Vec<(usize, u32)>> = (0..n)
            .into_par_iter()
            .map(|i| {
                if by_row[i].is_empty() {
                    return Vec::new();
                }
                let dist: Vec<f32> = (0..n)
                    .map(|j| {
                        if j == i {
                            f32::INFINITY
                        } else {
                            l2_sq(col(i), col(j))
                        }
                    })
                    .collect();
                let mut sorted = dist.clone();
                sorted.sort_by(f32::total_cmp);
                sorted.pop(); // `i` itself, at +∞
                by_row[i]
                    .iter()
                    .map(|&(j, c)| (c, rank_of(&sorted, dist[j])))
                    .collect()
            })
            .collect();
        let mut rank = vec![u32::MAX; pairs.len()];
        for (c, r) in directional.into_iter().flatten() {
            rank[c] = rank[c].min(r);
        }
        rank
    }
}

/// One fit, reduced to its views at the judged levels.
struct Model {
    kind: &'static str,
    prefix: String,
    coverage: f64,
    /// Per level; `None` at a skipped level.
    views: Vec<Option<View>>,
}

/// A judged level: its bounds, or why it was skipped.
enum Judged {
    Bounds(Bounds),
    Skipped(String),
}

pub fn run_critique(args: &CritiqueArgs) -> anyhow::Result<()> {
    mkdir_parent(&args.out)?;
    check_params(args.knn, args.far_frac, args.min_cells)?;

    let mut manifests = Vec::with_capacity(args.runs.len());
    for run in &args.runs {
        manifests.push(senna::run_manifest::load_for(run)?);
    }
    let names = unique_names(
        manifests
            .iter()
            .map(|(m, _)| {
                Path::new(&m.prefix).file_name().map_or_else(
                    || m.kind.as_str().to_string(),
                    |s| s.to_string_lossy().into_owned(),
                )
            })
            .collect(),
    );

    let (part_path, (assignments, part_cells)) = load_partition(args, &manifests)?;
    let labels = match (args.cell_labels.as_deref(), args.label_column.as_deref()) {
        (Some(path), Some(column)) => Some(load_cell_labels(path, column, &part_cells)?),
        _ => None,
    };
    let levels: Vec<Level> = assignments
        .iter()
        .map(|a| Level::new(a, args.min_cells))
        .collect();
    let judged: Vec<Judged> = levels
        .iter()
        .enumerate()
        .map(|(l, lv)| {
            let p = lv.n_pb();
            let j = if p > args.max_pb {
                Judged::Skipped(format!("{p} pseudobulks exceed --max-pb {}", args.max_pb))
            } else {
                match Bounds::for_level(p, args.knn, args.far_frac) {
                    Some(b) => Judged::Bounds(b),
                    None => Judged::Skipped(format!(
                        "{p} pseudobulks leave no rank beyond far; lower --knn to judge this level"
                    )),
                }
            };
            match &j {
                Judged::Bounds(b) => info!(
                    "Level {l}: {p} pseudobulks; near ≤ {}, far > {}",
                    b.near, b.far
                ),
                Judged::Skipped(why) => warn!("Level {l} skipped: {why}"),
            }
            j
        })
        .collect();

    let mut models = Vec::with_capacity(manifests.len());
    for ((m, dir), name) in manifests.iter().zip(&names) {
        models.push(load_model(m, dir, name, &part_cells, &levels, &judged)?);
    }

    let mut pair_tab = PairTable::default();
    let mut summary_tab = SummaryTable::default();
    let mut label_tabs: Vec<LabelTable> =
        (0..models.len()).map(|_| LabelTable::default()).collect();
    let mut level_json = Vec::new();
    for (l, (lv, j)) in levels.iter().zip(&judged).enumerate() {
        let b = match j {
            Judged::Bounds(b) => *b,
            Judged::Skipped(why) => {
                level_json
                    .push(serde_json::json!({ "level": l, "n_pb": lv.n_pb(), "skipped": why }));
                continue;
            }
        };
        let views: Vec<&View> = models
            .iter()
            .map(|m| m.views[l].as_ref().expect("a view at every judged level"))
            .collect();
        let pairs = candidates(&views, args.knn);
        let ranks: Vec<Vec<u32>> = views.iter().map(|v| v.pair_ranks(&pairs)).collect();
        let committees: Vec<Vec<usize>> = (0..models.len())
            .map(|mi| {
                committee(
                    models.len(),
                    mi,
                    args.committee,
                    &mut draw_rng(args.seed, l, mi, 0),
                )
            })
            .collect();
        let answered: Vec<Vec<Option<f64>>> =
            committees.iter().map(|c| answers(&ranks, c)).collect();
        let merged: Vec<Vec<bool>> = ranks
            .iter()
            .zip(&answered)
            .map(|(rk, a)| merges(rk, a, b))
            .collect();
        let judged = JudgedLevel {
            bounds: b,
            pairs: &pairs,
            ranks: &ranks,
            committees: &committees,
            answered: &answered,
            merged: &merged,
        };
        if args.questions > 0 {
            for (mi, label_tab) in label_tabs.iter_mut().enumerate() {
                let rng = &mut draw_rng(args.seed, l, mi, 1);
                let asked = draw_questions(views[mi], &judged, mi, lv.n_pb(), args.questions, rng);
                let n_labels = label_tab.push(l, lv, &judged, mi, &asked, &names);
                info!(
                    "Level {l}, {}: {} questions drawn, {n_labels} labels",
                    names[mi],
                    asked.len()
                );
            }
        }
        pair_tab.push_level(l, lv, &judged, &names);
        // With cell labels: each pair's composition overlap, NaN where unknown.
        let overlaps: Option<Vec<f32>> = labels.as_ref().map(|cl| {
            let comp = label_composition(&lv.pb_of_cell, &cl.of_cell, lv.n_pb(), cl.n_labels);
            pairs
                .iter()
                .map(|&(a, b_)| match (&comp[a as usize], &comp[b_ as usize]) {
                    (Some(x), Some(y)) => overlap(x, y),
                    _ => f32::NAN,
                })
                .collect()
        });
        if let Some(ov) = &overlaps {
            pair_tab.label_overlap.extend_from_slice(ov);
        }

        for (mi, name) in names.iter().enumerate() {
            let near = near_count(&ranks[mi], b);
            let n_merged = merged[mi].iter().filter(|&&m| m).count();
            summary_tab.push(l, name, lv.n_pb(), b, near, n_merged);
            info!(
                "Level {l}, {name}: {n_merged} merges of {near} near pairs (rate {:.3})",
                merge_rate(n_merged, near)
            );
            if let Some(ov) = &overlaps {
                let is_near: Vec<bool> = ranks[mi].iter().map(|&r| r <= b.near).collect();
                let (m, n) = (mean_over(ov, &merged[mi]), mean_over(ov, &is_near));
                summary_tab.merge_overlap.push(m);
                summary_tab.near_overlap.push(n);
                info!(
                    "Level {l}, {name}: label overlap of merges {m:.2}, of its near pairs {n:.2}"
                );
            }
        }

        info!("Level {l}: {} candidate pairs", pairs.len());
        level_json.push(serde_json::json!({
            "level": l,
            "n_pb": lv.n_pb(),
            "n_pairs": pairs.len(),
            "near_rank": b.near,
            "far_rank": b.far,
        }));
    }

    pair_tab.write(&format!("{}.critique.pairs.parquet", args.out), &names)?;
    summary_tab.write(&format!("{}.critique.summary.parquet", args.out))?;
    if args.questions > 0 {
        for (tab, name) in label_tabs.iter().zip(&names) {
            tab.write(&format!("{}.critique.labels.{name}.parquet", args.out))?;
        }
    }
    let record = serde_json::json!({
        "partition": part_path,
        "models": models.iter().zip(&names).map(|(m, name)| serde_json::json!({
            "name": name,
            "kind": m.kind,
            "prefix": m.prefix,
            "coverage": m.coverage,
        })).collect::<Vec<_>>(),
        "params": {
            "knn": args.knn,
            "far_frac": args.far_frac,
            "committee": args.committee,
            "questions": args.questions,
            "seed": args.seed,
            "min_cells": args.min_cells,
            "cell_labels": args.cell_labels,
            "label_column": args.label_column,
        },
        "levels": level_json,
    });
    std::fs::write(
        format!("{}.critique.json", args.out),
        serde_json::to_string_pretty(&record)?,
    )?;
    info!(
        "Wrote {}.critique.{{pairs,summary}}.parquet{} and .critique.json",
        args.out,
        if args.questions > 0 {
            ", .critique.labels.{model}.parquet"
        } else {
            ""
        }
    );
    Ok(())
}

/// The union of every view's top-`k` pairs, sorted.
fn candidates(views: &[&View], k: usize) -> Vec<(u32, u32)> {
    let mut pairs: Vec<(u32, u32)> = views.iter().flat_map(|v| v.top_k_pairs(k)).collect();
    pairs.sort_unstable();
    pairs.dedup();
    pairs
}

/// The partition run's per-cell partition: `(path, levels coarsest-first, cells)`.
///
/// The per-cell `cell_to_pb`, not `cell_to_pb_all`: that one also holds the
/// cells QC kept out of the per-cell outputs, which no model has a latent row for.
fn load_partition(
    args: &CritiqueArgs,
    manifests: &[(RunManifest, PathBuf)],
) -> anyhow::Result<(String, InheritedPartition)> {
    let (pm, pdir) = match args.partition.as_deref() {
        Some(p) => senna::run_manifest::load_for(p)?,
        None => manifests
            .iter()
            .find(|(m, _)| m.outputs.cell_to_pb.is_some())
            .cloned()
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "none of the models wrote a cell → pseudobulk partition; \
                     pass --partition with a topic, masked or vae run"
                )
            })?,
    };
    let rel = pm.outputs.cell_to_pb.as_deref().ok_or_else(|| {
        anyhow::anyhow!(
            "{}: no cell_to_pb in the manifest; the partition run must be a topic, masked or vae run",
            pm.prefix
        )
    })?;
    let path = resolve(&pdir, rel).to_string_lossy().into_owned();
    info!("Partition: {path}");
    Ok((path.clone(), load_cell_to_pb_raw(&path)?))
}

/// Read one model's cell latent, align it to the partition, and reduce it to
/// its views at the judged levels; the latent itself is not kept.
fn load_model(
    m: &RunManifest,
    dir: &Path,
    name: &str,
    part_cells: &[Box<str>],
    levels: &[Level],
    judged: &[Judged],
) -> anyhow::Result<Model> {
    let rel = m.outputs.geometry_latent().ok_or_else(|| {
        anyhow::anyhow!(
            "{}: the manifest names no latent or cell embedding",
            m.prefix
        )
    })?;
    let path = resolve(dir, rel).to_string_lossy().into_owned();
    let MatWithNames {
        rows,
        mat: mut latent,
        ..
    } = Mat::from_parquet_with_row_names(&path, Some(0))?;
    let row_of_cell = tolerant_align(part_cells, &rows)
        .with_context(|| format!("model {name}: matching {path} to the partition's cells"))?;
    let matched = row_of_cell.iter().filter(|&&r| r != usize::MAX).count();
    let coverage = matched as f64 / part_cells.len().max(1) as f64;
    let space = m.kind.cell_space();
    info!(
        "Model {name} ({}): {} cells × {} dims from {path}; {:.1}% of the partition's cells",
        m.kind.as_str(),
        rows.len(),
        latent.ncols(),
        100.0 * coverage
    );
    anyhow::ensure!(
        coverage >= 0.9,
        "{name}: only {:.1}% of the partition's cells have a latent row; \
         is it a fit of the same data?",
        100.0 * coverage
    );
    if space == CellSpace::LogSimplex {
        latent.apply(|v| *v = v.exp());
    }
    let views = levels
        .iter()
        .zip(judged)
        .map(|(lv, j)| {
            matches!(j, Judged::Bounds(_)).then(|| View::build(&latent, &row_of_cell, space, lv))
        })
        .collect();
    Ok(Model {
        kind: m.kind.as_str(),
        prefix: m.prefix.clone(),
        coverage,
        views,
    })
}

/// A reproducible generator for one draw: the run's seed, the level, the model
/// and what is drawn (0 = committee, 1 = questions).
fn draw_rng(seed: u64, level: usize, model: usize, what: u64) -> SmallRng {
    let salt = ((level as u64) << 40) ^ ((model as u64) << 8) ^ what;
    SmallRng::seed_from_u64(seed ^ salt.wrapping_mul(0x9E37_79B9_7F4A_7C15))
}

/// Draw up to `n` of model `mi`'s questions among its near pairs: k-means++
/// seeding, weighted by disagreement squared (how far the answer lies beyond
/// the model's own rank, over `P`) and spread by distance in the model's view.
/// Returns indices into the level's pairs.
fn draw_questions(
    view: &View,
    j: &JudgedLevel,
    mi: usize,
    n_pb: usize,
    n: usize,
    rng: &mut SmallRng,
) -> Vec<usize> {
    let own = &j.ranks[mi];
    let pool: Vec<usize> = (0..j.pairs.len())
        .filter(|&c| own[c] <= j.bounds.near)
        .collect();
    let weights: Vec<f64> = pool
        .iter()
        .map(|&c| {
            let gap = j.answered[mi][c].map_or(0.0, |a| (a - f64::from(own[c])).max(0.0));
            (gap / n_pb.max(1) as f64).powi(2)
        })
        .collect();
    let at = |i: usize| (j.pairs[pool[i]].0 as usize, j.pairs[pool[i]].1 as usize);
    let d = |x: usize, y: usize| view.dist(x, y);
    let dist = |i: usize, k: usize| f64::from(pair_distance(&d, at(i), at(k)));
    kmeanspp(&weights, &dist, n, rng)
        .into_iter()
        .map(|i| pool[i])
        .collect()
}

/// The comma-joined names of the picked models.
fn names_of(names: &[String], pick: impl Iterator<Item = usize>) -> Box<str> {
    pick.map(|o| names[o].as_str())
        .collect::<Vec<_>>()
        .join(",")
        .into()
}

/// The `level:pb_a-pb_b` row keys shared by the pair-keyed tables.
fn pair_keys(level: &[i32], pb_a: &[i32], pb_b: &[i32]) -> Vec<Box<str>> {
    (0..level.len())
        .map(|i| format!("{}:{}-{}", level[i], pb_a[i], pb_b[i]).into())
        .collect()
}

/// One model's labels, as `{out}.critique.labels.{model}.parquet`: the drawn
/// questions the answer made merges, the pairs that model is to push apart.
/// `channel` is `A` throughout (pseudobulk pairs, plan §7); channel B, gene
/// pairs, is not built.
#[derive(Default)]
struct LabelTable {
    level: Vec<i32>,
    pb_a: Vec<i32>,
    pb_b: Vec<i32>,
    weight: Vec<f32>,
    answer_rank: Vec<f32>,
    own_rank: Vec<i32>,
    answered_by: Vec<Box<str>>,
}

impl LabelTable {
    /// Add model `mi`'s drawn questions that are merges; returns how many.
    fn push(
        &mut self,
        l: usize,
        lv: &Level,
        j: &JudgedLevel,
        mi: usize,
        asked: &[usize],
        names: &[String],
    ) -> usize {
        let committee = &j.committees[mi];
        let by = names_of(names, committee.iter().copied());
        let before = self.level.len();
        for &c in asked.iter().filter(|&&c| j.merged[mi][c]) {
            let (a, b) = (j.pairs[c].0 as usize, j.pairs[c].1 as usize);
            self.level.push(l as i32);
            self.pb_a.push(lv.pb_id[a] as i32);
            self.pb_b.push(lv.pb_id[b] as i32);
            self.weight.push(far_share(j.ranks, committee, c, j.bounds));
            self.answer_rank
                .push(j.answered[mi][c].map_or(f32::NAN, |x| x as f32));
            self.own_rank.push(j.ranks[mi][c] as i32);
            self.answered_by.push(by.clone());
        }
        self.level.len() - before
    }

    fn write(&self, path: &str) -> anyhow::Result<()> {
        let key = pair_keys(&self.level, &self.pb_a, &self.pb_b);
        let channel: Vec<Box<str>> = vec![Box::from("A"); self.level.len()];
        let cols: Vec<(Box<str>, Column)> = vec![
            ("channel".into(), Column::Str(&channel)),
            ("level".into(), Column::I32(&self.level)),
            ("pb_a".into(), Column::I32(&self.pb_a)),
            ("pb_b".into(), Column::I32(&self.pb_b)),
            ("weight".into(), Column::F32(&self.weight)),
            ("answer_rank".into(), Column::F32(&self.answer_rank)),
            ("own_rank".into(), Column::I32(&self.own_rank)),
            ("answered_by".into(), Column::Str(&self.answered_by)),
        ];
        write_named_table(path, "pair", &key, &cols)
    }
}

/// One level's candidate pairs and what the models said about them.
struct JudgedLevel<'a> {
    bounds: Bounds,
    pairs: &'a [(u32, u32)],
    /// `ranks[model][pair]`, `u32::MAX` where unknown.
    ranks: &'a [Vec<u32>],
    /// Per model, the models that answered it.
    committees: &'a [Vec<usize>],
    /// `answered[model][pair]`: the committee's median rank.
    answered: &'a [Vec<Option<f64>>],
    /// `merged[model][pair]`.
    merged: &'a [Vec<bool>],
}

/// One row per candidate pair, as `{out}.critique.pairs.parquet`.
#[derive(Default)]
struct PairTable {
    level: Vec<i32>,
    pb_a: Vec<i32>,
    pb_b: Vec<i32>,
    n_cells_a: Vec<i32>,
    n_cells_b: Vec<i32>,
    /// Per model; `NaN` where the model has no cells in either pseudobulk.
    rank: Vec<Vec<f32>>,
    median_rank: Vec<f32>,
    /// Largest minus smallest known rank. Not `rank_spread`: per-model columns
    /// are `rank_{model}`, and a run named `spread` would collide.
    spread: Vec<f32>,
    /// The models that merged the pair, comma-separated; empty when none did.
    merged_by: Vec<Box<str>>,
    /// With `--cell-labels`: the overlap of the two label compositions.
    label_overlap: Vec<f32>,
}

impl PairTable {
    /// Add one level's pairs.
    fn push_level(&mut self, l: usize, lv: &Level, j: &JudgedLevel, names: &[String]) {
        let JudgedLevel {
            pairs,
            ranks,
            merged,
            ..
        } = *j;
        let all: Vec<usize> = (0..ranks.len()).collect();
        self.rank.resize_with(ranks.len(), Vec::new);
        for (c, &(a, b)) in pairs.iter().enumerate() {
            let sorted = known_ranks(ranks, &all, c);
            let (a, b) = (a as usize, b as usize);
            self.level.push(l as i32);
            self.pb_a.push(lv.pb_id[a] as i32);
            self.pb_b.push(lv.pb_id[b] as i32);
            self.n_cells_a.push(lv.n_cells[a] as i32);
            self.n_cells_b.push(lv.n_cells[b] as i32);
            for (col, rk) in self.rank.iter_mut().zip(ranks) {
                col.push(if rk[c] == u32::MAX {
                    f32::NAN
                } else {
                    rk[c] as f32
                });
            }
            self.median_rank
                .push(median(&sorted).map_or(f32::NAN, |m| m as f32));
            self.spread.push(match (sorted.first(), sorted.last()) {
                (Some(lo), Some(hi)) => (hi - lo) as f32,
                _ => f32::NAN,
            });
            self.merged_by
                .push(names_of(names, (0..merged.len()).filter(|&m| merged[m][c])));
        }
    }

    fn write(&self, path: &str, names: &[String]) -> anyhow::Result<()> {
        let key = pair_keys(&self.level, &self.pb_a, &self.pb_b);
        let rank_names: Vec<Box<str>> = names.iter().map(|n| format!("rank_{n}").into()).collect();
        let mut cols: Vec<(Box<str>, Column)> = vec![
            ("level".into(), Column::I32(&self.level)),
            ("pb_a".into(), Column::I32(&self.pb_a)),
            ("pb_b".into(), Column::I32(&self.pb_b)),
            ("n_cells_a".into(), Column::I32(&self.n_cells_a)),
            ("n_cells_b".into(), Column::I32(&self.n_cells_b)),
        ];
        for (name, col) in rank_names.iter().zip(&self.rank) {
            cols.push((name.clone(), Column::F32(col)));
        }
        cols.push(("median_rank".into(), Column::F32(&self.median_rank)));
        cols.push(("spread".into(), Column::F32(&self.spread)));
        cols.push(("merged_by".into(), Column::Str(&self.merged_by)));
        if !self.label_overlap.is_empty() {
            cols.push(("label_overlap".into(), Column::F32(&self.label_overlap)));
        }
        write_named_table(path, "pair", &key, &cols)
    }
}

/// One row per model and level, as `{out}.critique.summary.parquet`.
#[derive(Default)]
struct SummaryTable {
    model: Vec<Box<str>>,
    level: Vec<i32>,
    n_pb: Vec<i32>,
    far_rank: Vec<i32>,
    near: Vec<i32>,
    merges: Vec<i32>,
    /// Merges over near pairs: the report card.
    merge_rate: Vec<f32>,
    /// With `--cell-labels`: mean label overlap of the merged pairs, and of
    /// all the model's near pairs (the base a merge is judged against).
    merge_overlap: Vec<f32>,
    near_overlap: Vec<f32>,
}

impl SummaryTable {
    fn push(
        &mut self,
        level: usize,
        model: &str,
        n_pb: usize,
        b: Bounds,
        near: usize,
        merges: usize,
    ) {
        self.model.push(model.into());
        self.level.push(level as i32);
        self.n_pb.push(n_pb as i32);
        self.far_rank.push(b.far as i32);
        self.near.push(near as i32);
        self.merges.push(merges as i32);
        self.merge_rate.push(merge_rate(merges, near));
    }

    fn write(&self, path: &str) -> anyhow::Result<()> {
        let mut cols: Vec<(Box<str>, Column)> = vec![
            ("level".into(), Column::I32(&self.level)),
            ("n_pb".into(), Column::I32(&self.n_pb)),
            ("far_rank".into(), Column::I32(&self.far_rank)),
            ("near_pairs".into(), Column::I32(&self.near)),
            ("merges".into(), Column::I32(&self.merges)),
            ("merge_rate".into(), Column::F32(&self.merge_rate)),
        ];
        if !self.merge_overlap.is_empty() {
            cols.push(("merge_overlap".into(), Column::F32(&self.merge_overlap)));
            cols.push(("near_overlap".into(), Column::F32(&self.near_overlap)));
        }
        write_named_table(path, "model", &self.model, &cols)
    }
}

#[cfg(test)]
#[path = "critique_tests.rs"]
mod tests;
