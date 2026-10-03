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

use legume_numeric::matrix::parquet::{write_named_table, Column};
use log::warn;
use rayon::prelude::*;
use rustc_hash::{FxHashMap, FxHashSet};
use senna::embed_common::*;
use senna::run_manifest::{load_cell_to_pb_raw, resolve, CellSpace, RunManifest};
use std::path::Path;

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
                     model at rank 15 and another at 16 is not a disagreement."
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

/// The rank beyond which a model holds a pair far: `max(3·knn, ⌈P·frac⌉)`,
/// well clear of near (`≤ knn`).
pub(crate) fn far_rank(n_pb: usize, knn: usize, frac: f64) -> u32 {
    (3 * knn).max((n_pb as f64 * frac).ceil() as usize) as u32
}

/// Median of the known ranks (`u32::MAX` is unknown); `None` when none is known.
pub(crate) fn median_rank(ranks: impl Iterator<Item = u32>) -> Option<f64> {
    let mut known: Vec<u32> = ranks.filter(|&r| r != u32::MAX).collect();
    if known.is_empty() {
        return None;
    }
    known.sort_unstable();
    let m = known.len();
    Some(if m % 2 == 1 {
        f64::from(known[m / 2])
    } else {
        0.5 * (f64::from(known[m / 2 - 1]) + f64::from(known[m / 2]))
    })
}

/// The median rank of pair `c` over every model except `model`.
pub(crate) fn others_consensus(ranks: &[Vec<u32>], model: usize, c: usize) -> Option<f64> {
    median_rank(
        ranks
            .iter()
            .enumerate()
            .filter(|&(o, _)| o != model)
            .map(|(_, rk)| rk[c]),
    )
}

/// What the models together say about a pair.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(crate) enum PairLabel {
    Similar,
    Different,
    Ambiguous,
}

impl PairLabel {
    /// From the median rank over all models: near is `≤ knn`, far is `> far`.
    pub(crate) fn of(median: Option<f64>, knn: u32, far: u32) -> Self {
        match median {
            Some(m) if m <= f64::from(knn) => PairLabel::Similar,
            Some(m) if m > f64::from(far) => PairLabel::Different,
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

/// One model's charges at one level.
#[derive(Debug, Default, PartialEq, Eq)]
pub(crate) struct Tally {
    /// Pairs in the model's top-`k`.
    pub near: usize,
    /// Near pairs the other models hold far.
    pub merges: usize,
    /// Pairs the model holds far while the other models keep them near.
    pub splits: usize,
}

/// Judge `model` against the median of the others, pair by pair.
/// `ranks[model][pair]`, `u32::MAX` where unknown.
pub(crate) fn tally(ranks: &[Vec<u32>], model: usize, knn: u32, far: u32) -> (Tally, Vec<Charge>) {
    let n = ranks[model].len();
    let mut t = Tally::default();
    let mut charges = vec![Charge::None; n];
    for (c, charge) in charges.iter_mut().enumerate() {
        let r = ranks[model][c];
        if r == u32::MAX {
            continue;
        }
        let Some(others) = others_consensus(ranks, model, c) else {
            continue;
        };
        if r <= knn {
            t.near += 1;
            if others > f64::from(far) {
                t.merges += 1;
                *charge = Charge::Merge;
            }
        } else if r > far && others <= f64::from(knn) {
            t.splits += 1;
            *charge = Charge::Split;
        }
    }
    (t, charges)
}

/// Per kept pseudobulk: `merge`, `split`, `merge+split` or `consistent`, from
/// the charged pairs it belongs to.
pub(crate) fn pb_labels(n_pb: usize, pairs: &[(u32, u32)], charges: &[Charge]) -> Vec<Box<str>> {
    let mut flag = vec![0u8; n_pb];
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
        .map(|f| match f {
            1 => "merge",
            2 => "split",
            3 => "merge+split",
            _ => "consistent",
        })
        .map(Box::from)
        .collect()
}

/// One fit, aligned to the partition's cells.
struct Model {
    name: String,
    space: CellSpace,
    /// Partition cell → row of `latent`, or `usize::MAX`.
    row_of_cell: Vec<usize>,
    latent: Mat,
    coverage: f64,
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

/// A model's pseudobulk view at one level: rows in the metric of its space,
/// and which rows have any cells.
fn pb_view(model: &Model, level: &Level) -> (Mat, Vec<bool>) {
    let p = level.n_pb();
    let d = model.latent.ncols();
    let mut sum = Mat::zeros(p, d);
    let mut count = vec![0usize; p];
    for (cell, &pb) in level.pb_of_cell.iter().enumerate() {
        let row = model.row_of_cell[cell];
        if pb == usize::MAX || row == usize::MAX {
            continue;
        }
        count[pb] += 1;
        for k in 0..d {
            let x = model.latent[(row, k)];
            sum[(pb, k)] += match model.space {
                CellSpace::LogSimplex => x.exp(),
                CellSpace::Embedding | CellSpace::Signed => x,
            };
        }
    }
    let valid: Vec<bool> = count.iter().map(|&c| c > 0).collect();
    for (i, &c) in count.iter().enumerate() {
        if c > 0 {
            let mut r = sum.row_mut(i);
            r /= c as f32;
        }
    }
    match model.space {
        CellSpace::LogSimplex => sum.apply(|x| *x = x.max(0.0).sqrt()),
        CellSpace::Embedding => {
            for (i, ok) in valid.iter().enumerate() {
                if *ok {
                    let mut r = sum.row_mut(i);
                    let n = r.norm();
                    if n > 0.0 {
                        r /= n;
                    }
                }
            }
        }
        CellSpace::Signed => {
            for k in 0..d {
                let vals: Vec<f32> = (0..p).filter(|&i| valid[i]).map(|i| sum[(i, k)]).collect();
                if vals.len() < 2 {
                    continue;
                }
                let mean = vals.iter().sum::<f32>() / vals.len() as f32;
                let var = vals.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / vals.len() as f32;
                let sd = var.sqrt().max(1e-12);
                for i in 0..p {
                    sum[(i, k)] = (sum[(i, k)] - mean) / sd;
                }
            }
        }
    }
    (sum, valid)
}

/// Squared Euclidean distances from row `i` to every row; `+∞` for `i` itself
/// and for rows without cells.
fn row_distances(view: &Mat, valid: &[bool], i: usize) -> Vec<f32> {
    (0..view.nrows())
        .map(|j| {
            if j == i || !valid[j] {
                f32::INFINITY
            } else {
                (view.row(i) - view.row(j)).norm_squared()
            }
        })
        .collect()
}

/// A view's top-`k` pairs, as `(min, max)` kept indices.
fn top_k_pairs(view: &Mat, valid: &[bool], k: usize) -> Vec<(u32, u32)> {
    (0..view.nrows())
        .into_par_iter()
        .filter(|&i| valid[i])
        .flat_map_iter(|i| {
            let d = row_distances(view, valid, i);
            let mut idx: Vec<usize> = (0..d.len()).filter(|&j| d[j].is_finite()).collect();
            idx.sort_by(|&a, &b| d[a].total_cmp(&d[b]));
            idx.truncate(k);
            idx.into_iter()
                .map(move |j| (i.min(j) as u32, i.max(j) as u32))
        })
        .collect()
}

/// Every candidate's rank in one view: the smaller of the two directional
/// ranks, or `u32::MAX` when the model has no cells in either pseudobulk.
fn pair_ranks(view: &Mat, valid: &[bool], pairs: &[(u32, u32)]) -> Vec<u32> {
    let p = view.nrows();
    let mut by_row: Vec<Vec<(usize, usize)>> = vec![Vec::new(); p];
    for (c, &(a, b)) in pairs.iter().enumerate() {
        by_row[a as usize].push((b as usize, c));
        by_row[b as usize].push((a as usize, c));
    }
    let directional: Vec<Vec<(usize, u32)>> = (0..p)
        .into_par_iter()
        .map(|i| {
            if !valid[i] || by_row[i].is_empty() {
                return Vec::new();
            }
            let d = row_distances(view, valid, i);
            let mut sorted: Vec<f32> = d.iter().copied().filter(|x| x.is_finite()).collect();
            sorted.sort_by(f32::total_cmp);
            by_row[i]
                .iter()
                .filter(|&&(j, _)| valid[j])
                .map(|&(j, c)| (c, rank_of(&sorted, d[j])))
                .collect()
        })
        .collect();
    let mut rank = vec![u32::MAX; pairs.len()];
    for (c, r) in directional.into_iter().flatten() {
        rank[c] = rank[c].min(r);
    }
    rank
}

pub fn run_critique(args: &CritiqueArgs) -> anyhow::Result<()> {
    mkdir_parent(&args.out)?;
    anyhow::ensure!(args.min_cells >= 1, "--min-cells must be at least 1");

    // Models, in the order given.
    let mut manifests = Vec::with_capacity(args.runs.len());
    for run in &args.runs {
        manifests.push(senna::run_manifest::load_for(run)?);
    }
    let names = model_names(&manifests);

    // The partition.
    let (pm, pdir) = match args.partition.as_deref() {
        Some(p) => senna::run_manifest::load_for(p)?,
        None => manifests
            .iter()
            .find(|(m, dir)| m.cell_to_pb_path(dir).is_some())
            .cloned()
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "none of the models wrote a cell → pseudobulk partition; \
                     pass --partition with a topic, masked or vae run"
                )
            })?,
    };
    let part_path = pm.cell_to_pb_path(&pdir).ok_or_else(|| {
        anyhow::anyhow!(
            "{}: no cell_to_pb in the manifest; the partition run must be a topic, masked or vae run",
            pm.prefix
        )
    })?;
    info!("Partition: {part_path}");
    let (assignments, part_cells) = load_cell_to_pb_raw(&part_path)?;
    let levels: Vec<Level> = assignments
        .iter()
        .map(|a| Level::new(a, args.min_cells))
        .collect();
    for (l, lv) in levels.iter().enumerate() {
        info!(
            "Level {l}: {} pseudobulks with ≥ {} cells",
            lv.n_pb(),
            args.min_cells
        );
    }

    let mut models = Vec::with_capacity(manifests.len());
    for ((m, dir), name) in manifests.iter().zip(&names) {
        models.push(load_model(m, dir, name, &part_cells)?);
    }

    let knn = args.knn as u32;
    let mut pair_tab = PairTable::new(names.len());
    let mut summary_tab = SummaryTable::default();
    let mut cell_cols: Vec<(Box<str>, Vec<Box<str>>)> = Vec::new();
    let mut level_json = Vec::new();
    for (l, lv) in levels.iter().enumerate() {
        let p = lv.n_pb();
        if p < 3 {
            continue;
        }
        if p > args.max_pb {
            warn!(
                "Level {l}: {p} pseudobulks exceed --max-pb {}; skipped",
                args.max_pb
            );
            continue;
        }
        let far = far_rank(p, args.knn, args.far_frac);
        let views: Vec<(Mat, Vec<bool>)> = models.iter().map(|m| pb_view(m, lv)).collect();
        let mut cand: FxHashSet<(u32, u32)> = FxHashSet::default();
        for (view, valid) in &views {
            cand.extend(top_k_pairs(view, valid, args.knn));
        }
        let mut pairs: Vec<(u32, u32)> = cand.into_iter().collect();
        pairs.sort_unstable();
        let ranks: Vec<Vec<u32>> = views
            .iter()
            .map(|(view, valid)| pair_ranks(view, valid, &pairs))
            .collect();

        let mut labels = Vec::with_capacity(pairs.len());
        for (c, &(a, b)) in pairs.iter().enumerate() {
            let (a, b) = (a as usize, b as usize);
            let median = median_rank(ranks.iter().map(|rk| rk[c]));
            let label = PairLabel::of(median, knn, far);
            labels.push(label);
            let known: Vec<u32> = ranks
                .iter()
                .map(|rk| rk[c])
                .filter(|&x| x != u32::MAX)
                .collect();
            let t = &mut pair_tab;
            t.key
                .push(format!("{l}:{}-{}", lv.pb_id[a], lv.pb_id[b]).into());
            t.level.push(l as i32);
            t.pb_a.push(lv.pb_id[a] as i32);
            t.pb_b.push(lv.pb_id[b] as i32);
            t.n_cells_a.push(lv.n_cells[a] as i32);
            t.n_cells_b.push(lv.n_cells[b] as i32);
            for (col, rk) in t.rank.iter_mut().zip(&ranks) {
                col.push(if rk[c] == u32::MAX {
                    f32::NAN
                } else {
                    rk[c] as f32
                });
            }
            t.median_rank.push(median.map_or(f32::NAN, |m| m as f32));
            t.rank_spread
                .push(match (known.iter().max(), known.iter().min()) {
                    (Some(hi), Some(lo)) => (hi - lo) as f32,
                    _ => f32::NAN,
                });
            t.label.push(label.as_str().into());
        }

        for (mi, name) in names.iter().enumerate() {
            let (t, charges) = tally(&ranks, mi, knn, far);
            let pb_label = pb_labels(p, &pairs, &charges);
            cell_cols.push((
                format!("{name}.L{l}").into(),
                lv.pb_of_cell
                    .iter()
                    .map(|&pb| {
                        if pb == usize::MAX {
                            Box::from("")
                        } else {
                            pb_label[pb].clone()
                        }
                    })
                    .collect(),
            ));
            summary_tab.push(l, name, p, far, &t);
            info!(
                "Level {l}, {name}: {} near pairs; merges {}, splits {}",
                t.near, t.merges, t.splits
            );
        }

        let count = |x: PairLabel| labels.iter().filter(|&&y| y == x).count();
        info!(
            "Level {l}: {} candidate pairs; similar {}, different {}, ambiguous {} (far > {far})",
            pairs.len(),
            count(PairLabel::Similar),
            count(PairLabel::Different),
            count(PairLabel::Ambiguous),
        );
        level_json.push(serde_json::json!({
            "level": l,
            "n_pb": p,
            "n_pairs": pairs.len(),
            "similar": count(PairLabel::Similar),
            "different": count(PairLabel::Different),
            "ambiguous": count(PairLabel::Ambiguous),
            "far_rank": far,
        }));
    }

    // Write.
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
        "models": models.iter().zip(&manifests).map(|(m, (man, _))| serde_json::json!({
            "name": m.name,
            "kind": man.kind.as_str(),
            "prefix": man.prefix,
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

/// Short, unique names: the run prefix's file name, with `_2`, `_3`, … on repeats.
fn model_names(manifests: &[(RunManifest, std::path::PathBuf)]) -> Vec<String> {
    let mut seen: FxHashMap<String, usize> = FxHashMap::default();
    manifests
        .iter()
        .map(|(m, _)| {
            let base = Path::new(&m.prefix).file_name().map_or_else(
                || m.kind.as_str().to_string(),
                |s| s.to_string_lossy().into_owned(),
            );
            let n = seen.entry(base.clone()).or_insert(0);
            *n += 1;
            if *n == 1 {
                base
            } else {
                format!("{base}_{n}")
            }
        })
        .collect()
}

fn load_model(
    m: &RunManifest,
    dir: &Path,
    name: &str,
    part_cells: &[Box<str>],
) -> anyhow::Result<Model> {
    let rel = m.outputs.geometry_latent().ok_or_else(|| {
        anyhow::anyhow!(
            "{}: the manifest names no latent or cell embedding",
            m.prefix
        )
    })?;
    let path = resolve(dir, rel).to_string_lossy().into_owned();
    let MatWithNames { rows, mat, .. } = Mat::from_parquet_with_row_names(&path, Some(0))?;
    let row_of_cell = tolerant_align(part_cells, &rows);
    let matched = row_of_cell.iter().filter(|&&r| r != usize::MAX).count();
    let coverage = matched as f64 / part_cells.len().max(1) as f64;
    info!(
        "Model {name} ({}): {} cells × {} dims from {path}; {:.1}% of the partition's cells",
        m.kind.as_str(),
        rows.len(),
        mat.ncols(),
        100.0 * coverage
    );
    anyhow::ensure!(
        coverage >= 0.9,
        "{name}: only {:.1}% of the partition's cells have a latent row; \
         is it a fit of the same data?",
        100.0 * coverage
    );
    Ok(Model {
        name: name.to_string(),
        space: m.kind.cell_space(),
        row_of_cell,
        latent: mat,
        coverage,
    })
}

/// One row per candidate pair, as `{out}.critique.pairs.parquet`.
struct PairTable {
    key: Vec<Box<str>>,
    level: Vec<i32>,
    pb_a: Vec<i32>,
    pb_b: Vec<i32>,
    n_cells_a: Vec<i32>,
    n_cells_b: Vec<i32>,
    /// Per model; `NaN` where the model has no cells in either pseudobulk.
    rank: Vec<Vec<f32>>,
    median_rank: Vec<f32>,
    rank_spread: Vec<f32>,
    label: Vec<Box<str>>,
}

impl PairTable {
    fn new(n_models: usize) -> Self {
        PairTable {
            key: Vec::new(),
            level: Vec::new(),
            pb_a: Vec::new(),
            pb_b: Vec::new(),
            n_cells_a: Vec::new(),
            n_cells_b: Vec::new(),
            rank: vec![Vec::new(); n_models],
            median_rank: Vec::new(),
            rank_spread: Vec::new(),
            label: Vec::new(),
        }
    }

    fn write(&self, path: &str, names: &[String]) -> anyhow::Result<()> {
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
        cols.push(("rank_spread".into(), Column::F32(&self.rank_spread)));
        cols.push(("label".into(), Column::Str(&self.label)));
        write_named_table(path, "pair", &self.key, &cols)
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
    fn push(&mut self, level: usize, model: &str, n_pb: usize, far: u32, t: &Tally) {
        self.model.push(model.into());
        self.level.push(level as i32);
        self.n_pb.push(n_pb as i32);
        self.far_rank.push(far as i32);
        self.near.push(t.near as i32);
        self.merges.push(t.merges as i32);
        self.splits.push(t.splits as i32);
        let total = t.merges + t.splits;
        self.merge_share.push(if total == 0 {
            f32::NAN
        } else {
            t.merges as f32 / total as f32
        });
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
