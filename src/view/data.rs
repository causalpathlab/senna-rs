//! Everything `senna view` shows, resolved from a run manifest.
//!
//! Layouts come from `manifest.layout.methods` (falling back to the top-level
//! `layout.cell_coords` for manifests written before methods were kept apart).
//! Group labels come from whatever the run has: clusters, annotation, topic
//! argmax for cells; marker membership and topic argmax for features. Labels
//! are joined to points by name, so a layout and a label table written by
//! different commands still line up.

use legume_numeric::matrix::common_io::read_lines_of_words_delim;
use rustc_hash::FxHashMap as HashMap;
use senna::embed_common::*;
use senna::run_manifest::{self, LayoutEntry, RunManifest};
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

/// Which points a space shows.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum SpaceKind {
    Cells,
    /// Features placed on a cell layout, drawn over that layout's cells.
    FeaturesOnCells,
    /// The feature embedding laid out on its own.
    Features,
}

impl SpaceKind {
    #[must_use]
    pub fn axis(self) -> Axis {
        match self {
            SpaceKind::Cells => Axis::Cells,
            SpaceKind::FeaturesOnCells | SpaceKind::Features => Axis::Features,
        }
    }

    #[must_use]
    pub fn title(self) -> &'static str {
        match self {
            SpaceKind::Cells => "cells",
            SpaceKind::FeaturesOnCells => "features on cells",
            SpaceKind::Features => "features",
        }
    }

    /// The title as a file-name and flag-friendly word.
    #[must_use]
    pub fn slug(self) -> &'static str {
        match self {
            SpaceKind::Cells => "cells",
            SpaceKind::FeaturesOnCells => "features-on-cells",
            SpaceKind::Features => "features",
        }
    }
}

/// A view of one layout method on one axis.
pub struct Space {
    pub method: String,
    pub kind: SpaceKind,
    pub points: Points,
    /// For features placed on a cell map: that map's cells, drawn muted
    /// underneath for context.
    pub backdrop: Option<usize>,
    /// For a layout of one group computed in the viewer: the space it was
    /// zoomed from.
    pub parent: Option<usize>,
}

impl Space {
    #[must_use]
    pub fn axis(&self) -> Axis {
        self.kind.axis()
    }

    #[must_use]
    pub fn title(&self) -> &'static str {
        self.kind.title()
    }
}

/// What a grouping is. Code dispatches on this; `title` is what the user
/// sees and names in `--colour-by`, and keys saved styles.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum LabelKind {
    Annotation,
    /// Fine cell types, when the round's annotation is coarse.
    FineAnnotation,
    Cluster,
    Topic,
    Markers,
    FeatureTopic,
    /// Cells whose label differs from the source round's.
    Changed,
    /// The source round's annotation.
    Previous,
}

impl LabelKind {
    #[must_use]
    pub fn title(self) -> &'static str {
        match self {
            LabelKind::Annotation => "annotation",
            LabelKind::FineAnnotation => "fine annotation",
            LabelKind::Cluster => "cluster",
            LabelKind::Topic => "topic",
            LabelKind::Markers => "markers",
            LabelKind::FeatureTopic => "feature topics",
            LabelKind::Changed => "changed",
            LabelKind::Previous => "previous annotation",
        }
    }

    #[must_use]
    pub fn axis(self) -> Axis {
        match self {
            LabelKind::Markers | LabelKind::FeatureTopic => Axis::Features,
            _ => Axis::Cells,
        }
    }
}

/// A categorical labelling keyed by point name.
pub struct Labels {
    pub kind: LabelKind,
    pub levels: Vec<Box<str>>,
    /// For clusters: the numeric id of each level.
    pub ids: Vec<i64>,
    pub by_name: HashMap<Box<str>, u32>,
}

/// Group id per point, `NONE` where the point has no label.
pub const NONE: u32 = u32::MAX;

impl Labels {
    /// Group `(point, level)` pairs, leaving out levels in `skip`. A point
    /// listed twice keeps its first level. Levels are numbered in natural
    /// order (`T2` before `T10`).
    pub(crate) fn new<I: IntoIterator<Item = (Box<str>, Box<str>)>>(
        kind: LabelKind,
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
        let mut labels = Self {
            kind,
            levels,
            ids: Vec::new(),
            by_name,
        };
        labels.sort_naturally();
        labels
    }

    /// Clusters from `(point, id)` pairs, named `C{id}` and ordered by id.
    pub(crate) fn clusters<I: IntoIterator<Item = (Box<str>, i64)>>(pairs: I) -> Self {
        let pairs: Vec<(Box<str>, i64)> = pairs.into_iter().collect();
        let mut labels = Self::new(
            LabelKind::Cluster,
            pairs
                .iter()
                .map(|(n, id)| (n.clone(), format!("C{id}").into_boxed_str())),
            &[],
        );
        labels.ids = labels
            .levels
            .iter()
            .map(|l| l[1..].parse().expect("written as C{id} above"))
            .collect();
        labels
    }

    #[must_use]
    pub fn title(&self) -> &'static str {
        self.kind.title()
    }

    #[must_use]
    pub fn axis(&self) -> Axis {
        self.kind.axis()
    }

    /// Renumber levels in natural order.
    fn sort_naturally(&mut self) {
        let key = |s: &str| {
            let digits: String = s.chars().filter(char::is_ascii_digit).collect();
            (digits.parse::<u64>().unwrap_or(u64::MAX), s.to_string())
        };
        let mut idx: Vec<usize> = (0..self.levels.len()).collect();
        idx.sort_by_key(|&i| key(&self.levels[i]));
        let mut remap = vec![0u32; idx.len()];
        for (new, &old) in idx.iter().enumerate() {
            remap[old] = new as u32;
        }
        self.levels = idx.iter().map(|&i| self.levels[i].clone()).collect();
        for v in self.by_name.values_mut() {
            *v = remap[*v as usize];
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

/// `(first, second)` columns of a tab-separated table (gzip allowed), header
/// row skipped.
pub(super) fn read_pairs(path: &Path) -> anyhow::Result<Vec<(Box<str>, Box<str>)>> {
    let lines = read_lines_of_words_delim(&path.to_string_lossy(), &['\t'][..], -1)?.lines;
    Ok(lines
        .into_iter()
        .skip(1)
        .filter_map(|r| {
            let mut it = r.into_iter();
            Some((it.next()?, it.next()?))
        })
        .collect())
}

/// Column of the largest entry in row `i`.
fn row_argmax(m: &Mat, i: usize) -> usize {
    (0..m.ncols())
        .max_by(|&a, &b| m[(i, a)].total_cmp(&m[(i, b)]))
        .unwrap_or(0)
}

/// Integer ids from a one-column parquet (NaN = unassigned).
fn read_cluster_labels(path: &Path) -> anyhow::Result<Labels> {
    let MatWithNames { rows, mat, .. } =
        Mat::from_parquet_with_row_names(&path.to_string_lossy(), Some(0))?;
    Ok(Labels::clusters(rows.into_iter().enumerate().filter_map(
        |(i, n)| {
            let v = mat[(i, 0)];
            (v.is_finite() && v >= 0.0).then_some((n, v as i64))
        },
    )))
}

/// Topic argmax per cell from log θ (`cells × K`), as `T{k}`.
fn read_topic_labels(path: &Path) -> anyhow::Result<Labels> {
    let MatWithNames { rows, mat, .. } =
        Mat::from_parquet_with_row_names(&path.to_string_lossy(), Some(0))?;
    let pairs = rows
        .into_iter()
        .enumerate()
        .map(|(i, n)| (n, format!("T{}", row_argmax(&mat, i)).into_boxed_str()));
    Ok(Labels::new(LabelKind::Topic, pairs, &[]))
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
    let pairs = rows
        .into_iter()
        .enumerate()
        .map(|(i, n)| (n, format!("T{}", row_argmax(&mat, i)).into_boxed_str()));
    Ok(Labels::new(LabelKind::FeatureTopic, pairs, &[]))
}

/// `feature<TAB>group` marker table, read the way annotation reads it; a
/// feature listed under several groups keeps the first.
fn read_marker_labels(path: &Path) -> anyhow::Result<Labels> {
    let pairs = data_beans::aux::gene_sets::read_membership_pairs(&path.to_string_lossy())?;
    Ok(Labels::new(LabelKind::Markers, pairs, &[]))
}

/// The run's layouts, the most recently computed first.
fn ordered_methods(m: &RunManifest) -> Vec<(String, LayoutEntry)> {
    let mut methods: Vec<(String, LayoutEntry)> = m
        .layout
        .methods
        .iter()
        .map(|(k, v)| (k.clone(), v.clone()))
        .collect();
    if methods.is_empty() {
        if let Some(cc) = &m.layout.cell_coords {
            methods.push((
                "layout".into(),
                LayoutEntry {
                    cell_coords: Some(cc.clone()),
                    ..Default::default()
                },
            ));
        }
    }
    if let Some(i) = m
        .layout
        .current
        .as_ref()
        .and_then(|cur| methods.iter().position(|(k, _)| k == cur))
    {
        let e = methods.remove(i);
        methods.insert(0, e);
    }
    methods
}

fn load_spaces(m: &RunManifest, dir: &Path) -> anyhow::Result<Vec<Space>> {
    let at = |rel: &str| run_manifest::resolve(dir, rel);
    let mut spaces: Vec<Space> = Vec::new();
    for (method, e) in ordered_methods(m) {
        let mut cells = None;
        for (slot, kind) in [
            (&e.cell_coords, SpaceKind::Cells),
            (&e.feature_on_cell_coords, SpaceKind::FeaturesOnCells),
            (&e.feature_coords, SpaceKind::Features),
        ] {
            let Some(p) = slot else { continue };
            spaces.push(Space {
                method: method.clone(),
                kind,
                points: read_xy(&at(p))?,
                backdrop: (kind == SpaceKind::FeaturesOnCells)
                    .then_some(cells)
                    .flatten(),
                parent: None,
            });
            if kind == SpaceKind::Cells {
                cells = Some(spaces.len() - 1);
            }
        }
    }
    Ok(spaces)
}

/// Every grouping the run carries. One that fails to read is skipped with a
/// warning rather than failing the view.
fn load_labels(m: &RunManifest, dir: &Path, round: &super::rounds::Round) -> Vec<Labels> {
    let at = |rel: &str| run_manifest::resolve(dir, rel);
    let mut labels = Vec::new();
    let mut keep = |what: &str, r: anyhow::Result<Labels>| match r {
        Ok(l) if !l.levels.is_empty() => labels.push(l),
        Ok(_) => {}
        Err(e) => log::warn!("view: skipping {what}: {e}"),
    };
    if let Some(p) = &m.annotate.argmax {
        match super::rounds::read_argmax(&at(p)) {
            Ok(current) => {
                keep(
                    "annotation",
                    Ok(Labels::new(
                        LabelKind::Annotation,
                        current.iter().map(|(c, l)| (c.clone(), l.clone())),
                        &[super::rounds::UNASSIGNED],
                    )),
                );
                // Against the round this one was made from, when there is one;
                // a source without annotation counts as all unassigned.
                let previous = round.source.as_deref().and_then(|src| {
                    let (sm, sdir) = RunManifest::load(src).ok()?;
                    Some(match sm.annotate.argmax.as_deref() {
                        Some(p) => super::rounds::read_argmax(&run_manifest::resolve(&sdir, p)),
                        None => Ok(Default::default()),
                    })
                });
                match previous {
                    Some(Ok(previous)) => {
                        let [before, changed] = super::rounds::comparisons(&current, &previous);
                        keep("changed", Ok(changed));
                        keep("previous annotation", Ok(before));
                    }
                    Some(Err(e)) => log::warn!("view: skipping round comparison: {e}"),
                    None => {}
                }
            }
            Err(e) => log::warn!("view: skipping annotation: {e}"),
        }
    }
    if let Some(p) = super::rounds::annotate_str(m, "fine_argmax") {
        keep(
            "fine annotation",
            super::rounds::read_argmax(&at(p)).map(|fine| {
                Labels::new(
                    LabelKind::FineAnnotation,
                    fine,
                    &[super::rounds::UNASSIGNED],
                )
            }),
        );
    }
    if let Some(p) = &m.cluster.clusters {
        keep("clusters", read_cluster_labels(&at(p)));
    }
    // An embedding run's `latent` is log θ only when Z went to
    // `cell_embedding`; older manifests kept Z there instead.
    let latent_is_topics = m.kind.latent_is_log_simplex() || m.outputs.cell_embedding.is_some();
    if let Some(p) = m.outputs.latent.as_deref().filter(|_| latent_is_topics) {
        keep("topics", read_topic_labels(&at(p)));
    }
    if let Some(p) = &m.annotate.markers {
        keep("markers", read_marker_labels(&at(p)));
    }
    if let Some(p) = &m.outputs.softmax_dictionary {
        keep("feature topics", read_dictionary_labels(&at(p)));
    }
    labels
}

impl Dataset {
    pub fn load(from: &str) -> anyhow::Result<Self> {
        let manifest_path = PathBuf::from(from);
        let (m, dir) = RunManifest::load(&manifest_path)?;
        let spaces = load_spaces(&m, &dir)?;
        anyhow::ensure!(
            !spaces.is_empty(),
            "{from} has no layout yet; run `senna layout umap --from {from}` first"
        );
        let round = super::rounds::Round::load(&m, &dir, &manifest_path);
        let labels = load_labels(&m, &dir, &round);
        Ok(Self {
            prefix: run_manifest::derive_out_prefix(from),
            spaces,
            labels,
            run: Some((m, dir)),
            round: Some(round),
        })
    }
}
