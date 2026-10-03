//! `senna critique` — stage 0a of `docs/peer-critique-plan.md`: report only.
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
//! 4. **Consensus.** Near is rank ≤ `k`; far is beyond `max(3k, P/4)` by default,
//!    well clear of near. A pair's label is read off the median rank over all
//!    models: `similar`, `different` or `ambiguous`.
//! 5. **Charges.** Each model is judged against the median rank of the *other*
//!    models, so its own rank never votes. It *merges* a pair it keeps near while
//!    the others hold it far, and *splits* a pair it holds far while the others
//!    keep it near. A model with many merges and few splits packs states the
//!    others separate: the signature of mode collapse.
//!
//! The latents alone cannot say which side of a disagreement is right; the
//! others' consensus stands in for it. Models that share a mistake are not
//! charged for it.
//!
//! Levels are numbered as in `cell_to_pb.parquet`: `0` is the coarsest.

use legume_numeric::matrix::knn::all_pairs::knn_rows_l2;
use legume_numeric::matrix::knn::metric::l2_sq;
use legume_numeric::matrix::parquet::{write_named_table, Column};
use log::warn;
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
                     The models need not have trained on this partition."
    )]
    partition: Option<Box<str>>,

    #[arg(
        long,
        short,
        required = true,
        help = "Output prefix",
        long_help = "Writes:\n  \
                     {out}.critique.pairs.parquet     one row per candidate pair\n  \
                     {out}.critique.summary.parquet   merges and splits per model and level\n  \
                     {out}.critique.cells.parquet     per cell, its pseudobulk's label per model and level\n  \
                     {out}.critique.json              inputs and parameters"
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
        help = "Far is beyond rank max(3·knn, this fraction of the pseudobulks)",
        long_help = "Near is rank ≤ --knn. Far must be well clear of it, so that one\n\
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
        default_value_t = 20_000,
        help = "Levels with more pseudobulks than this are skipped"
    )]
    max_pb: usize,
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
pub(crate) fn tolerant_align(source: &[Box<str>], target: &[Box<str>]) -> Vec<usize> {
    let at: FxHashMap<&str, usize> = target
        .iter()
        .enumerate()
        .map(|(i, n)| (n.as_ref(), i))
        .collect();
    source
        .iter()
        .map(|n| at.get(n.as_ref()).copied().unwrap_or(usize::MAX))
        .collect()
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
    /// `far = max(3·knn, ⌈P·frac⌉)`, well clear of near. `None` when no rank
    /// can lie beyond it: ranks run from 1 to `P − 1`.
    pub(crate) fn for_level(n_pb: usize, knn: usize, frac: f64) -> Option<Self> {
        let far = (3 * knn).max((n_pb as f64 * frac).ceil() as usize);
        (n_pb.saturating_sub(1) > far).then_some(Bounds {
            near: knn as u32,
            far: far as u32,
        })
    }
}

/// The known ranks of pair `c` over all models (`u32::MAX` is unknown), sorted.
pub(crate) fn sorted_known(ranks: &[Vec<u32>], c: usize) -> Vec<u32> {
    let mut known: Vec<u32> = ranks
        .iter()
        .map(|rk| rk[c])
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

/// The median of `sorted` with one occurrence of `own` left out: a model's
/// rank never votes on itself.
pub(crate) fn median_without(sorted: &[u32], own: u32) -> Option<f64> {
    match sorted.iter().position(|&r| r == own) {
        Some(at) => {
            let mut others = sorted.to_vec();
            others.remove(at);
            median(&others)
        }
        None => median(sorted),
    }
}

/// What the models together say about a pair.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(crate) enum PairLabel {
    Similar,
    Different,
    Ambiguous,
}

impl PairLabel {
    pub(crate) fn of(median: Option<f64>, b: Bounds) -> Self {
        match median {
            Some(m) if m <= f64::from(b.near) => PairLabel::Similar,
            Some(m) if m > f64::from(b.far) => PairLabel::Different,
            _ => PairLabel::Ambiguous,
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            PairLabel::Similar => "similar",
            PairLabel::Different => "different",
            PairLabel::Ambiguous => "ambiguous",
        }
    }
}

/// What a pair is charged to one model as.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(crate) enum Charge {
    None,
    Merge,
    Split,
}

/// Judge one model, pair by pair, against the median of the others. `own` is
/// the model's ranks, `sorted` every pair's known ranks over all models.
pub(crate) fn charges(own: &[u32], sorted: &[Vec<u32>], b: Bounds) -> Vec<Charge> {
    own.iter()
        .zip(sorted)
        .map(|(&r, all)| {
            if r == u32::MAX {
                return Charge::None;
            }
            match median_without(all, r) {
                Some(o) if r <= b.near && o > f64::from(b.far) => Charge::Merge,
                Some(o) if r > b.far && o <= f64::from(b.near) => Charge::Split,
                _ => Charge::None,
            }
        })
        .collect()
}

/// Pairs in a model's top-`k` (unknown ranks never count).
pub(crate) fn near_count(own: &[u32], b: Bounds) -> usize {
    own.iter().filter(|&&r| r <= b.near).count()
}

/// Per kept pseudobulk: `merge`, `split`, `merge+split` or `consistent`, from
/// the charged pairs it belongs to; `unseen` where the model has no view of it.
pub(crate) fn pb_labels(
    pairs: &[(u32, u32)],
    charges: &[Charge],
    seen: &[bool],
) -> Vec<&'static str> {
    let mut flag = vec![0u8; seen.len()];
    for (&(a, b), &ch) in pairs.iter().zip(charges) {
        let bit = match ch {
            Charge::None => continue,
            Charge::Merge => 1,
            Charge::Split => 2,
        };
        flag[a as usize] |= bit;
        flag[b as usize] |= bit;
    }
    flag.iter()
        .zip(seen)
        .map(|(f, &seen)| match (seen, f) {
            (false, _) => "unseen",
            (true, 1) => "merge",
            (true, 2) => "split",
            (true, 3) => "merge+split",
            _ => "consistent",
        })
        .collect()
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

    /// Which kept pseudobulks the view holds.
    fn seen(&self) -> Vec<bool> {
        self.slot.iter().map(|&r| r != usize::MAX).collect()
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
    let mut cell_cols: Vec<(Box<str>, Vec<Box<str>>)> = Vec::new();
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
        let sorted: Vec<Vec<u32>> = (0..pairs.len()).map(|c| sorted_known(&ranks, c)).collect();
        let label_count = pair_tab.push_level(l, lv, b, &pairs, &ranks, &sorted);

        for (mi, name) in names.iter().enumerate() {
            let ch = charges(&ranks[mi], &sorted, b);
            let pb_label = pb_labels(&pairs, &ch, &views[mi].seen());
            cell_cols.push((
                format!("{name}.L{l}").into(),
                lv.pb_of_cell
                    .iter()
                    .map(|&pb| Box::from(pb_label.get(pb).copied().unwrap_or("")))
                    .collect(),
            ));
            let row = summary_tab.push(l, name, lv.n_pb(), b, near_count(&ranks[mi], b), &ch);
            info!("Level {l}, {name}: merges {}, splits {}", row.0, row.1);
        }

        info!(
            "Level {l}: {} candidate pairs; similar {}, different {}, ambiguous {}",
            pairs.len(),
            label_count[0],
            label_count[1],
            label_count[2],
        );
        level_json.push(serde_json::json!({
            "level": l,
            "n_pb": lv.n_pb(),
            "n_pairs": pairs.len(),
            "similar": label_count[0],
            "different": label_count[1],
            "ambiguous": label_count[2],
            "near_rank": b.near,
            "far_rank": b.far,
        }));
    }

    pair_tab.write(&format!("{}.critique.pairs.parquet", args.out), &names)?;
    summary_tab.write(&format!("{}.critique.summary.parquet", args.out))?;
    let cols: Vec<(Box<str>, Column)> = cell_cols
        .iter()
        .map(|(n, v)| (n.clone(), Column::Str(v)))
        .collect();
    write_named_table(
        &format!("{}.critique.cells.parquet", args.out),
        "cell",
        &part_cells,
        &cols,
    )?;
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
            "min_cells": args.min_cells,
        },
        "levels": level_json,
    });
    std::fs::write(
        format!("{}.critique.json", args.out),
        serde_json::to_string_pretty(&record)?,
    )?;
    info!(
        "Wrote {}.critique.{{pairs,summary,cells}}.parquet and .critique.json",
        args.out
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
    let row_of_cell = tolerant_align(part_cells, &rows);
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
    label: Vec<Box<str>>,
}

impl PairTable {
    /// Add one level's pairs; returns how many are similar, different and ambiguous.
    fn push_level(
        &mut self,
        l: usize,
        lv: &Level,
        b: Bounds,
        pairs: &[(u32, u32)],
        ranks: &[Vec<u32>],
        sorted: &[Vec<u32>],
    ) -> [usize; 3] {
        self.rank.resize_with(ranks.len(), Vec::new);
        let mut count = [0usize; 3];
        for (c, &(a, b_)) in pairs.iter().enumerate() {
            let (a, b_) = (a as usize, b_ as usize);
            let med = median(&sorted[c]);
            let label = PairLabel::of(med, b);
            count[label as usize] += 1;
            self.level.push(l as i32);
            self.pb_a.push(lv.pb_id[a] as i32);
            self.pb_b.push(lv.pb_id[b_] as i32);
            self.n_cells_a.push(lv.n_cells[a] as i32);
            self.n_cells_b.push(lv.n_cells[b_] as i32);
            for (col, rk) in self.rank.iter_mut().zip(ranks) {
                col.push(if rk[c] == u32::MAX {
                    f32::NAN
                } else {
                    rk[c] as f32
                });
            }
            self.median_rank.push(med.map_or(f32::NAN, |m| m as f32));
            self.spread
                .push(match (sorted[c].first(), sorted[c].last()) {
                    (Some(lo), Some(hi)) => (hi - lo) as f32,
                    _ => f32::NAN,
                });
            self.label.push(label.as_str().into());
        }
        count
    }

    fn write(&self, path: &str, names: &[String]) -> anyhow::Result<()> {
        let key: Vec<Box<str>> = (0..self.level.len())
            .map(|i| format!("{}:{}-{}", self.level[i], self.pb_a[i], self.pb_b[i]).into())
            .collect();
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
        cols.push(("label".into(), Column::Str(&self.label)));
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
    splits: Vec<i32>,
    merge_share: Vec<f32>,
}

impl SummaryTable {
    /// Add one model's row; returns its `(merges, splits)`.
    fn push(
        &mut self,
        level: usize,
        model: &str,
        n_pb: usize,
        b: Bounds,
        near: usize,
        charges: &[Charge],
    ) -> (usize, usize) {
        let merges = charges.iter().filter(|&&c| c == Charge::Merge).count();
        let splits = charges.iter().filter(|&&c| c == Charge::Split).count();
        self.model.push(model.into());
        self.level.push(level as i32);
        self.n_pb.push(n_pb as i32);
        self.far_rank.push(b.far as i32);
        self.near.push(near as i32);
        self.merges.push(merges as i32);
        self.splits.push(splits as i32);
        self.merge_share.push(if merges + splits == 0 {
            f32::NAN
        } else {
            merges as f32 / (merges + splits) as f32
        });
        (merges, splits)
    }

    fn write(&self, path: &str) -> anyhow::Result<()> {
        let cols: Vec<(Box<str>, Column)> = vec![
            ("level".into(), Column::I32(&self.level)),
            ("n_pb".into(), Column::I32(&self.n_pb)),
            ("far_rank".into(), Column::I32(&self.far_rank)),
            ("near_pairs".into(), Column::I32(&self.near)),
            ("merges".into(), Column::I32(&self.merges)),
            ("splits".into(), Column::I32(&self.splits)),
            ("merge_share".into(), Column::F32(&self.merge_share)),
        ];
        write_named_table(path, "model", &self.model, &cols)
    }
}

#[cfg(test)]
#[path = "critique_tests.rs"]
mod tests;
