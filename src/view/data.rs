//! Everything `senna view` shows, resolved from a run manifest.
//!
//! Layouts come from `manifest.layout.methods` (falling back to the top-level
//! `layout.cell_coords` for manifests written before methods were kept apart).
//! Group labels come from whatever the run has: clusters, annotation, topic
//! argmax for cells; marker membership and topic argmax for features. Labels
//! are joined to points by name, so a layout and a label table written by
//! different commands still line up.

use rustc_hash::FxHashMap as HashMap;
use senna::embed_common::*;
use senna::run_manifest::{self, RunManifest};
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};

/// What a point stands for.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Axis {
    Cells,
    Features,
}

/// One set of 2D points.
pub struct Points {
    pub names: Vec<Box<str>>,
    pub xy: Vec<[f32; 2]>,
    /// A fixed random permutation: any prefix is a uniform subsample, which is
    /// what lets the renderer draw progressively.
    pub order: Vec<u32>,
    pub bounds: [f32; 4],
}

impl Points {
    pub(crate) fn new(names: Vec<Box<str>>, xy: Vec<[f32; 2]>) -> Self {
        use rand::seq::SliceRandom;
        use rand::SeedableRng;
        let mut order: Vec<u32> = (0..xy.len() as u32)
            .filter(|&i| {
                let [x, y] = xy[i as usize];
                x.is_finite() && y.is_finite()
            })
            .collect();
        order.shuffle(&mut rand::rngs::SmallRng::seed_from_u64(7));
        let mut b = [
            f32::INFINITY,
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::NEG_INFINITY,
        ];
        for &i in &order {
            let [x, y] = xy[i as usize];
            b = [b[0].min(x), b[1].min(y), b[2].max(x), b[3].max(y)];
        }
        if order.is_empty() {
            b = [0.0, 0.0, 1.0, 1.0];
        }
        Self {
            names,
            xy,
            order,
            bounds: b,
        }
    }
}

/// A view of one layout method on one axis.
pub struct Space {
    pub method: String,
    /// `cells`, `features on cells`, or `features`.
    pub title: &'static str,
    pub axis: Axis,
    pub points: Points,
    /// For features placed on a cell map: that map's cells, drawn muted
    /// underneath for context.
    pub backdrop: Option<usize>,
    /// For a layout of one group computed in the viewer: the space it was
    /// zoomed from.
    pub parent: Option<usize>,
}

/// A categorical labelling keyed by point name.
pub struct Labels {
    pub title: String,
    pub axis: Axis,
    pub levels: Vec<Box<str>>,
    pub by_name: HashMap<Box<str>, u32>,
}

/// Group id per point, `NONE` where the point has no label.
pub const NONE: u32 = u32::MAX;

impl Labels {
    pub(crate) fn from_pairs<I: IntoIterator<Item = (Box<str>, Box<str>)>>(
        title: &str,
        axis: Axis,
        pairs: I,
        skip: &[&str],
    ) -> Self {
        let mut levels: Vec<Box<str>> = Vec::new();
        let mut index: HashMap<Box<str>, u32> = HashMap::default();
        let mut by_name = HashMap::default();
        for (name, level) in pairs {
            if skip.contains(&level.as_ref()) {
                continue;
            }
            let id = *index.entry(level.clone()).or_insert_with(|| {
                levels.push(level);
                (levels.len() - 1) as u32
            });
            by_name.entry(name).or_insert(id);
        }
        Self {
            title: title.to_string(),
            axis,
            levels,
            by_name,
        }
    }

    /// Group id of every point in `points`.
    #[must_use]
    pub fn align(&self, points: &Points) -> Vec<u32> {
        points
            .names
            .iter()
            .map(|n| self.by_name.get(n).copied().unwrap_or(NONE))
            .collect()
    }
}

pub struct Dataset {
    pub prefix: String,
    pub spaces: Vec<Space>,
    pub labels: Vec<Labels>,
    /// The manifest and its directory, for loading feature activity on demand.
    pub run: Option<(RunManifest, PathBuf)>,
    /// Where this manifest sits among annotation rounds.
    pub round: Option<super::rounds::Round>,
}

fn read_xy(path: &Path) -> anyhow::Result<Points> {
    let MatWithNames { rows, cols, mat } =
        Mat::from_parquet_with_row_names(&path.to_string_lossy(), Some(0))?;
    let col = |name: &str, default: usize| {
        cols.iter()
            .position(|c| c.as_ref() == name)
            .unwrap_or(default)
    };
    let (cx, cy) = (col("x", 0), col("y", 1));
    anyhow::ensure!(
        mat.ncols() > cx.max(cy),
        "{} has fewer than two coordinate columns",
        path.display()
    );
    let xy = (0..mat.nrows())
        .map(|i| [mat[(i, cx)], mat[(i, cy)]])
        .collect();
    Ok(Points::new(rows, xy))
}

/// Rows of a text table, `#` comments and blank lines skipped.
pub(super) fn read_rows(path: &Path) -> anyhow::Result<Vec<Vec<String>>> {
    let f =
        std::fs::File::open(path).map_err(|e| anyhow::anyhow!("open {}: {e}", path.display()))?;
    let mut out = Vec::new();
    for line in BufReader::new(f).lines() {
        let line = line?;
        let line = line.trim_end();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        out.push(line.split('\t').map(str::to_string).collect());
    }
    Ok(out)
}

/// Integer ids from a one-column parquet (NaN = unassigned), as `C{id}`.
fn read_cluster_labels(path: &Path) -> anyhow::Result<Labels> {
    let MatWithNames { rows, mat, .. } =
        Mat::from_parquet_with_row_names(&path.to_string_lossy(), Some(0))?;
    let mut pairs: Vec<(Box<str>, i64)> = rows
        .into_iter()
        .enumerate()
        .filter_map(|(i, n)| {
            let v = mat[(i, 0)];
            (v.is_finite() && v >= 0.0).then_some((n, v as i64))
        })
        .collect();
    // Level order follows the ids, not first appearance.
    pairs.sort_by_key(|&(_, id)| id);
    Ok(Labels::from_pairs(
        "cluster",
        Axis::Cells,
        pairs
            .into_iter()
            .map(|(n, id)| (n, format!("C{id}").into_boxed_str())),
        &[],
    ))
}

/// Argmax over each row of an N × K table, as `T{k}`.
fn read_argmax_labels(path: &Path, title: &str, axis: Axis) -> anyhow::Result<Labels> {
    let MatWithNames { rows, mat, .. } =
        Mat::from_parquet_with_row_names(&path.to_string_lossy(), Some(0))?;
    let k = mat.ncols();
    let mut pairs: Vec<(Box<str>, usize)> = rows
        .into_iter()
        .enumerate()
        .map(|(i, n)| {
            let best = (0..k)
                .max_by(|&a, &b| mat[(i, a)].total_cmp(&mat[(i, b)]))
                .unwrap_or(0);
            (n, best)
        })
        .collect();
    pairs.sort_by_key(|&(_, t)| t);
    Ok(Labels::from_pairs(
        title,
        axis,
        pairs
            .into_iter()
            .map(|(n, t)| (n, format!("T{t}").into_boxed_str())),
        &[],
    ))
}

/// Per-row argmax of a column-simplex dictionary after normalizing each row,
/// so a feature goes to the topic it is most specific to rather than to the
/// topic with the largest overall mass.
fn read_dictionary_labels(path: &Path) -> anyhow::Result<Labels> {
    let MatWithNames { rows, mut mat, .. } =
        Mat::from_parquet_with_row_names(&path.to_string_lossy(), Some(0))?;
    // Stored in log space; a row-wise max-subtract keeps exp finite.
    for mut row in mat.row_iter_mut() {
        let m = row.max();
        row.apply(|v| *v = (*v - m).exp());
    }
    let k = mat.ncols();
    let pairs = rows.into_iter().enumerate().map(|(i, n)| {
        let t = (0..k)
            .max_by(|&a, &b| mat[(i, a)].total_cmp(&mat[(i, b)]))
            .unwrap_or(0);
        (n, format!("T{t}").into_boxed_str())
    });
    let mut labels = Labels::from_pairs("topic", Axis::Features, pairs, &[]);
    sort_levels_naturally(&mut labels);
    Ok(labels)
}

/// Renumber levels in natural order (`T2` before `T10`).
pub(super) fn sort_levels_naturally(labels: &mut Labels) {
    let key = |s: &str| {
        let digits: String = s.chars().filter(char::is_ascii_digit).collect();
        (digits.parse::<u64>().unwrap_or(u64::MAX), s.to_string())
    };
    let mut idx: Vec<usize> = (0..labels.levels.len()).collect();
    idx.sort_by_key(|&i| key(&labels.levels[i]));
    let mut remap = vec![0u32; idx.len()];
    for (new, &old) in idx.iter().enumerate() {
        remap[old] = new as u32;
    }
    labels.levels = idx.iter().map(|&i| labels.levels[i].clone()).collect();
    for v in labels.by_name.values_mut() {
        *v = remap[*v as usize];
    }
}

/// `feature<TAB>group` marker table, as used by annotation; a feature listed
/// under several groups keeps the first.
fn read_marker_labels(path: &Path) -> anyhow::Result<Labels> {
    let rows = read_rows(path)?;
    let pairs = rows.into_iter().filter_map(|r| {
        let mut it = r.into_iter();
        Some((it.next()?.into_boxed_str(), it.next()?.into_boxed_str()))
    });
    let mut labels = Labels::from_pairs("markers", Axis::Features, pairs, &[]);
    // A header row, if present, becomes a one-member level; drop it.
    labels
        .by_name
        .retain(|n, _| !n.eq_ignore_ascii_case("gene"));
    sort_levels_naturally(&mut labels);
    Ok(labels)
}

/// Layout files named by the `senna layout` convention,
/// `{prefix}.{method}.{what}.parquet`, relative to the manifest directory.
fn discover_layouts(dir: &Path, prefix: &str) -> Vec<(String, run_manifest::LayoutEntry)> {
    let base = Path::new(prefix)
        .file_name()
        .map_or_else(|| prefix.to_string(), |b| b.to_string_lossy().into_owned());
    let found = |method: &str, what: &str| {
        let rel = format!("{base}.{method}.{what}.parquet");
        dir.join(&rel).exists().then_some(rel)
    };
    ["umap", "phate", "tsne"]
        .iter()
        .filter_map(|&method| {
            let e = run_manifest::LayoutEntry {
                cell_coords: found(method, "cell_coords"),
                pb_coords: found(method, "pb_coords"),
                feature_on_cell_coords: found(method, "feature_on_cell_coords"),
                feature_coords: found(method, "feature_coords"),
            };
            (e != run_manifest::LayoutEntry::default()).then(|| (method.to_string(), e))
        })
        .collect()
}

impl Dataset {
    pub fn load(from: &str) -> anyhow::Result<Self> {
        let manifest_path = PathBuf::from(from);
        let (m, dir) = RunManifest::load(&manifest_path)?;
        let at = |rel: &str| run_manifest::resolve(&dir, rel);

        let mut spaces = Vec::new();
        let mut methods: Vec<(String, run_manifest::LayoutEntry)> = m
            .layout
            .methods
            .iter()
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect();
        if methods.is_empty() {
            // The per-method table can be lost when a tool with an older
            // manifest schema rewrites the file; the files are still there.
            methods = discover_layouts(&dir, &m.prefix);
            if !methods.is_empty() {
                log::warn!(
                    "view: manifest lists no layout methods; found {} on disk",
                    methods.len()
                );
            }
        }
        if methods.is_empty() {
            if let Some(cc) = &m.layout.cell_coords {
                methods.push((
                    "layout".into(),
                    run_manifest::LayoutEntry {
                        cell_coords: Some(cc.clone()),
                        ..Default::default()
                    },
                ));
            }
        }
        // The layout run last comes first; without a record of it, the one
        // the top-level slot points at.
        let current = m.layout.current.clone().or_else(|| {
            let cc = m.layout.cell_coords.as_deref()?;
            methods
                .iter()
                .find(|(_, e)| e.cell_coords.as_deref() == Some(cc))
                .map(|(k, _)| k.clone())
        });
        if let Some(cur) = &current {
            if let Some(i) = methods.iter().position(|(k, _)| k == cur) {
                let e = methods.remove(i);
                methods.insert(0, e);
            }
        }

        for (method, e) in &methods {
            let cell_space = if let Some(p) = &e.cell_coords {
                info!("view: {method} cells from {p}");
                spaces.push(Space {
                    method: method.clone(),
                    title: "cells",
                    axis: Axis::Cells,
                    points: read_xy(&at(p))?,
                    backdrop: None,
                    parent: None,
                });
                Some(spaces.len() - 1)
            } else {
                None
            };
            if let Some(p) = &e.feature_on_cell_coords {
                spaces.push(Space {
                    method: method.clone(),
                    title: "features on cells",
                    axis: Axis::Features,
                    points: read_xy(&at(p))?,
                    backdrop: cell_space,
                    parent: None,
                });
            }
            if let Some(p) = &e.feature_coords {
                spaces.push(Space {
                    method: method.clone(),
                    title: "features",
                    axis: Axis::Features,
                    points: read_xy(&at(p))?,
                    backdrop: None,
                    parent: None,
                });
            }
        }
        anyhow::ensure!(
            !spaces.is_empty(),
            "{from} has no layout yet; run `senna layout umap --from {from}` first"
        );

        let mut labels = Vec::new();
        let mut try_add = |what: &str, r: anyhow::Result<Labels>| match r {
            Ok(l) if !l.levels.is_empty() => labels.push(l),
            Ok(_) => {}
            Err(e) => log::warn!("view: skipping {what}: {e}"),
        };
        let round = super::rounds::Round::load(&m, &dir, &manifest_path);
        if let Some(p) = &m.annotate.argmax {
            match super::rounds::read_argmax(&at(p)) {
                Ok(current) => {
                    let mut ann = Labels::from_pairs(
                        "annotation",
                        Axis::Cells,
                        current.iter().map(|(c, l)| (c.clone(), l.clone())),
                        &["unassigned"],
                    );
                    sort_levels_naturally(&mut ann);
                    try_add("annotation", Ok(ann));
                    // Against the round this one was made from, when there is one.
                    let source_argmax = round.source.as_deref().and_then(|src| {
                        let (sm, sdir) = RunManifest::load(src).ok()?;
                        Some(run_manifest::resolve(&sdir, sm.annotate.argmax.as_deref()?))
                    });
                    if let Some(sp) = source_argmax {
                        match super::rounds::comparisons(&current, &sp) {
                            Ok([before, changed]) => {
                                try_add("changed", Ok(changed));
                                try_add("previous annotation", Ok(before));
                            }
                            Err(e) => log::warn!("view: skipping round comparison: {e}"),
                        }
                    }
                }
                Err(e) => log::warn!("view: skipping annotation: {e}"),
            }
        }
        if let Some(p) = &m.cluster.clusters {
            try_add("clusters", read_cluster_labels(&at(p)));
        }
        // An embedding run's `latent` is log θ only when Z went to
        // `cell_embedding`; older manifests kept Z there instead.
        let latent_is_topics = m.kind.latent_is_log_simplex() || m.outputs.cell_embedding.is_some();
        if let Some(p) = m.outputs.latent.as_deref().filter(|_| latent_is_topics) {
            try_add("topics", read_argmax_labels(&at(p), "topic", Axis::Cells));
        }
        if let Some(p) = &m.annotate.markers {
            try_add("markers", read_marker_labels(&at(p)));
        }
        if let Some(p) = &m.outputs.softmax_dictionary {
            try_add("feature topics", read_dictionary_labels(&at(p)));
        }

        Ok(Self {
            prefix: run_manifest::derive_out_prefix(from),
            spaces,
            labels,
            run: Some((m, dir)),
            round: Some(round),
        })
    }
}
