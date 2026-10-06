//! Features on the map: suggestions, the picked feature's activity, and
//! features near a clicked cell.

use super::*;

/// How many neighbours a click lists, in turn; the second by default.
pub(crate) const NEAR_COUNTS: [usize; 5] = [6, 12, 20, 30, 50];

/// A sidebar row: its text, and the feature it names, if it names one
/// (arrows step through those rows, a click shows its feature).
pub(crate) type PaneRow = (String, Option<Box<str>>);

/// Names ranked best first with their scores, or why there are none.
type Ranked = Result<Vec<(Box<str>, f32)>, String>;

/// What a set of neighbours is drawn around.
#[derive(Clone, Copy, PartialEq)]
pub(crate) enum Centre {
    /// A clicked cell, placed where the cells are.
    Cell,
    /// A clicked cluster label: the cluster's label anchor on cell map
    /// `space`, drawn only there.
    Group { space: usize, xy: [f32; 2] },
    /// A clicked feature, placed where the features are.
    Feature,
    /// Nothing: a pinned feature name on its own.
    None,
}

/// What the scores of a set of neighbours measure.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Metric {
    /// Euclidean distance in the cells' space, the metric the cell map is
    /// laid out and features are placed on it by; scores are minus it.
    Distance,
    /// Cosine of feature embeddings, the metric a feature map is laid out by.
    Cosine,
    /// `ρ_g · (z − z̄)`: how far the cells lean the feature's way, for a run
    /// with no co-embedding to place features by.
    Direction,
}

/// Features shown around one clicked cell, or features and cells around one
/// clicked feature.
#[derive(Clone)]
pub(crate) struct Near {
    pub name: Box<str>,
    pub centre: Centre,
    /// What the scores measure.
    pub metric: Metric,
    pub features: Vec<(Box<str>, f32)>,
    /// Cells nearest a clicked feature (runs with cells in an embedding).
    pub cells: Vec<(Box<str>, f32)>,
}

/// The run's feature embedding with every row L2-normalized, so a product of
/// rows is their cosine.
pub(crate) struct FeatureEmbedding {
    axis: activity::Axis,
    rows: Mat,
    /// The co-embedding as written, rows as `rows`, when the run has one:
    /// where each feature sits among the cells.
    places: Option<Mat>,
}

impl FeatureEmbedding {
    fn load(
        run: Option<&(senna::run_manifest::RunManifest, std::path::PathBuf)>,
    ) -> Result<Self, String> {
        let (m, dir) = run.ok_or("no run to read a feature embedding from")?;
        // The same table the feature map is laid out from, by default.
        use crate::postprocess::fit_layout_features::{read_feature_rows, FeatureSpace};
        let (names, rows) = read_feature_rows(m, dir, FeatureSpace::Auto)
            .map_err(|e| format!("no feature embedding: {e}"))?;
        // Auto reads the co-embedding when there is one: the same rows,
        // before they were normalized.
        let places = m.outputs.feature_coembedding.as_deref().and_then(|rel| {
            let path = senna::run_manifest::resolve(dir, rel);
            Mat::from_parquet_with_row_names(&path.to_string_lossy(), Some(0))
                .ok()
                .filter(|t| t.rows == names)
                .map(|t| t.mat)
        });
        Ok(Self {
            axis: activity::Axis::new(names),
            rows,
            places,
        })
    }

    /// The `top` features whose co-embedding is nearest `z` (Euclidean),
    /// scored by minus the distance, `skip` left out; none without a
    /// co-embedding of `z`'s width.
    fn near_place(
        &self,
        z: &activity::Vector,
        skip: Option<usize>,
        top: usize,
    ) -> Option<Vec<(Box<str>, f32)>> {
        let places = self.places.as_ref().filter(|p| p.ncols() == z.len())?;
        let mut scores: Vec<f32> = places
            .row_iter()
            .map(|r| -(r.transpose() - z).norm())
            .collect();
        if let Some(i) = skip {
            scores[i] = f32::NAN;
        }
        Some(activity::best(&scores, &self.axis.names, top))
    }

    /// Where `feature` sits among the cells, and its row; none without a
    /// co-embedding.
    fn place_of(&self, feature: &str) -> Option<(activity::Vector, usize)> {
        let places = self.places.as_ref()?;
        let i = self.axis.index.match_gene(feature)?;
        Some((places.row(i).transpose(), i))
    }

    /// The `top` features of highest cosine to `feature`, itself left out.
    fn near(&self, feature: &str, top: usize) -> Result<Vec<(Box<str>, f32)>, String> {
        let i = self
            .axis
            .index
            .match_gene(feature)
            .ok_or_else(|| format!("{feature} is not in the feature embedding"))?;
        let mut cos = &self.rows * self.rows.row(i).transpose();
        cos[i] = f32::NAN;
        Ok(activity::best(cos.as_slice(), &self.axis.names, top))
    }
}

impl Scene {
    /// The suggestions, when they were made for the view on screen.
    fn current_suggestions(&self) -> Option<&Suggestions> {
        self.suggestions.as_ref().filter(|s| s.space == self.space)
    }

    /// The activity drawn, when it was computed for the view on screen.
    pub(super) fn current_shown(&self) -> Option<&Shown> {
        self.shown.as_ref().filter(|s| s.space == self.space)
    }

    pub(super) fn markers(&self) -> Option<&data::Labels> {
        Some(&self.data.labels[self.label_index(LabelKind::Markers)?])
    }

    /// Marker features of `group`, matched loosely (case, and spaces, commas
    /// and underscores, which annotation tools rewrite).
    pub(super) fn marker_features(&self, group: &str) -> Vec<Box<str>> {
        let norm = rounds::label_key;
        let Some(m) = self.markers() else {
            return Vec::new();
        };
        let Some(id) = m.levels.iter().position(|l| norm(l) == norm(group)) else {
            return Vec::new();
        };
        let mut out: Vec<Box<str>> = m
            .by_name
            .iter()
            .filter(|&(_, &g)| g as usize == id)
            .map(|(n, _)| n.clone())
            .collect();
        out.sort();
        out
    }

    /// Rank features for the view on screen: what distinguishes the focused
    /// group from the rest of the view, or, with nothing focused, what varies
    /// most here. Shows the first one straight away.
    pub fn suggest(&mut self) {
        const TOP: usize = 100;
        if self.current().axis() != Axis::Cells {
            self.note = Some("suggestions work on a cell view".into());
            return;
        }
        let space = self.space;
        let mask: Option<Vec<bool>> = match (self.focus, self.groups()) {
            (Some(f), Some(g)) => Some(g.iter().map(|&x| x == f).collect()),
            _ => None,
        };
        let title = match self.focused_name() {
            Some(name) => format!("features that set {name} apart in this view"),
            None => "features that vary most in this view".to_string(),
        };
        let Some((activity, data)) = self.activity_and_data() else {
            self.note = Some("no manifest to read the model from".into());
            return;
        };
        let universe = &data.spaces[space].points.names;
        match activity.suggest(space, universe, mask.as_deref(), TOP) {
            Ok(list) if !list.is_empty() => {
                let first = list[0].0.clone();
                self.suggestions = Some(Suggestions { space, title, list });
                self.set_pick(Pick::One(first));
            }
            Ok(_) => self.note = Some("no feature stands out here".into()),
            Err(e) => self.note = Some(e),
        }
    }

    /// Keep a search's `matches` as the sidebar's list, `chosen` shown; the
    /// arrows then step through the matches as through suggestions.
    pub fn keep_matches(&mut self, query: &str, matches: Vec<Box<str>>, chosen: Box<str>) {
        self.suggestions = Some(Suggestions {
            space: self.space,
            title: format!("matches for “{query}”"),
            list: matches.into_iter().map(|m| (m, f32::NAN)).collect(),
        });
        self.set_pick(Pick::One(chosen));
    }

    /// Drop the suggestions. Returns whether there were any.
    pub fn clear_suggestions(&mut self) -> bool {
        self.suggestions.take().is_some()
    }

    /// The feature shown on the map, when it is one feature.
    fn shown_feature(&self) -> Option<&str> {
        match &self.pick {
            Some(Pick::One(f)) => Some(f.as_ref()),
            _ => None,
        }
    }

    /// Panel rows for the suggestions of the view on screen, the feature
    /// being shown marked; each feature's row carries its name.
    pub fn suggestion_rows(&self) -> Option<Vec<PaneRow>> {
        let Suggestions { title, list, .. } = self.current_suggestions()?;
        let shown = self.shown_feature();
        let mut out: Vec<PaneRow> = vec![
            (title.clone(), None),
            ("↑ ↓ step · o observed · x close".into(), None),
        ];
        out.extend(list.iter().enumerate().map(|(k, (f, v))| {
            let mark = if shown == Some(f.as_ref()) {
                "▸"
            } else {
                " "
            };
            // A search's matches have no score to show.
            let text = if v.is_nan() {
                format!("{mark} {:>2}  {f}", k + 1)
            } else {
                format!("{mark} {:>2}  {f:<16} {v:>6.2}", k + 1)
            };
            (text, Some(f.clone()))
        }));
        Some(out)
    }

    /// Features `g` steps through: the current suggestions, else the focused
    /// group's markers, else every marker grouped by type.
    pub(super) fn feature_list(&self) -> Vec<Box<str>> {
        if let Some(sug) = self.current_suggestions() {
            return sug.list.iter().map(|(f, _)| f.clone()).collect();
        }
        if let Some(name) = self.focused_name() {
            let own = self.marker_features(&name);
            if !own.is_empty() {
                return own;
            }
        }
        let Some(m) = self.markers() else {
            return Vec::new();
        };
        let mut all: Vec<(u32, Box<str>)> =
            m.by_name.iter().map(|(n, &g)| (g, n.clone())).collect();
        all.sort();
        all.into_iter().map(|(_, n)| n).collect()
    }

    pub fn step_feature(&mut self, delta: i64) {
        let list = self.feature_list();
        if list.is_empty() {
            self.note = Some("no marker table in this run; press / to search a feature".into());
            return;
        }
        let at = match &self.pick {
            Some(Pick::One(n)) => list.iter().position(|x| x == n),
            _ => None,
        };
        let n = list.len() as i64;
        let next = match at {
            Some(i) => (i as i64 + delta).rem_euclid(n),
            None if delta > 0 => 0,
            None => n - 1,
        };
        self.set_pick(Pick::One(list[next as usize].clone()));
    }

    pub fn set_pick(&mut self, pick: Pick) {
        self.pick = Some(pick);
        self.refresh_activity();
    }

    /// Show the focused group's marker set as one activity.
    pub fn pick_marker_set(&mut self) {
        let Some(name) = self.focused_name() else {
            self.note = Some("focus a group first ([ ] or click), then press a".into());
            return;
        };
        if self.marker_features(&name).is_empty() {
            self.note = Some(format!("no markers listed for {name}"));
            return;
        }
        self.set_pick(Pick::Markers(name));
    }

    pub fn toggle_source(&mut self) {
        self.source = self.source.other();
        // A click's list holds the features with values in the source shown;
        // suggestions on screen stay as they are.
        if self.current_suggestions().is_none() {
            let shown = self.pick.clone();
            self.rerun_near();
            self.pick = shown;
        }
        self.refresh_activity();
    }

    /// Drop the activity view. Returns whether there was one.
    pub fn clear_pick(&mut self) -> bool {
        self.shown = None;
        self.pick.take().is_some()
    }

    /// Feature names searchable under the current source.
    pub fn searchable(&mut self) -> Vec<Box<str>> {
        let source = self.source;
        self.activity()
            .and_then(|a| a.feature_names(source).map(<[_]>::to_vec).ok())
            .unwrap_or_default()
    }

    pub(super) fn activity(&mut self) -> Option<&mut Activity> {
        if self.activity.is_none() {
            let (m, dir) = self.data.run.clone()?;
            self.activity = Some(Activity::new(m, dir));
        }
        self.activity.as_mut()
    }

    /// The activity (made on first use) with the data beside it, borrowed
    /// apart so both can be used at once.
    pub(super) fn activity_and_data(&mut self) -> Option<(&mut Activity, &Dataset)> {
        self.activity()?;
        Some((self.activity.as_mut()?, &self.data))
    }

    /// The run's cells, as its first cell map names them.
    pub fn cell_names(&self) -> Option<&[Box<str>]> {
        self.data
            .spaces
            .iter()
            .find(|s| s.axis() == Axis::Cells)
            .map(|s| s.points.names.as_slice())
    }

    /// Recompute the activity drawn for the current pick, space and source.
    /// Cell spaces only; on a feature space the pick is marked instead.
    pub(super) fn refresh_activity(&mut self) {
        let Some(pick) = self.pick.clone() else {
            self.shown = None;
            return;
        };
        if self.current().axis() != Axis::Cells {
            self.shown = None;
            return;
        }
        let (space, source) = (self.space, self.source);
        if matches!(&self.shown, Some(s) if s.space == space && s.pick == pick && s.source == source)
        {
            return;
        }
        let set = match &pick {
            Pick::Markers(group) => self.marker_features(group),
            Pick::One(_) => Vec::new(),
        };
        let Some((activity, data)) = self.activity_and_data() else {
            self.note = Some("no manifest to read activity from".into());
            return;
        };
        let names = &data.spaces[space].points.names;
        let result = match &pick {
            Pick::One(f) => activity
                .levels(f, source, space, names)
                .map(|(l, spelled)| (l, format!("{spelled} · {}", source.name()))),
            Pick::Markers(group) => activity
                .set_levels(&set, source, space, names)
                .map(|(l, used)| (l, format!("{group} markers ({used}) · {}", source.name()))),
        };
        match result {
            Ok((levels, title)) => {
                self.shown_ids += 1;
                self.shown = Some(Shown {
                    id: self.shown_ids,
                    space,
                    pick,
                    source,
                    title,
                    levels,
                });
            }
            Err(e) => {
                self.shown = None;
                let no_counts = source == Source::Observed
                    && self
                        .activity
                        .as_ref()
                        .is_some_and(Activity::observed_failed);
                if no_counts {
                    self.missing_data = self
                        .activity
                        .as_ref()
                        .and_then(|a| a.missing_inputs().into_iter().next());
                    // Without the data, the model's expectation is what there
                    // is; if that fails too, its reason is said, not hidden.
                    self.source = Source::Expected;
                    self.note = None;
                    self.refresh_activity();
                    let then = match (&self.shown, self.note.take()) {
                        (Some(_), _) => "showing the model's expectation".to_string(),
                        (None, why) => format!(
                            "the model's expectation failed too: {}",
                            why.unwrap_or_else(|| "nothing to show".into())
                        ),
                    };
                    self.note = Some(format!("no observed counts: {e} · {then}"));
                } else {
                    self.note = Some(e);
                }
            }
        }
    }

    /// Features nearest cell `cell` (see [`Self::rank_near`]).
    pub fn show_near(&mut self, cell: &str) {
        if self.current().kind != SpaceKind::Cells {
            return;
        }
        let (metric, found) = self.rank_near(&[cell]);
        // The sidebar shows what was asked last: this click's list.
        self.clear_suggestions();
        self.keep_near(cell.into(), Centre::Cell, metric, found);
    }

    /// The features near `cells`: those whose co-embedding is nearest the
    /// cells' mean embedding (Euclidean, as the map is laid out), or, on a
    /// run with no co-embedding, those most up in the cells.
    fn rank_near(&mut self, cells: &[&str]) -> (Metric, Ranked) {
        let top = self.near_count;
        let z = match self.activity() {
            Some(a) => a.cells_z(cells.iter().copied()),
            None => return (Metric::Distance, Err("no run to find features in".into())),
        };
        let embedding = self
            .feature_embedding
            .get_or_insert_with(|| FeatureEmbedding::load(self.data.run.as_ref()));
        if let (Ok(z), Ok(e)) = (&z, embedding.as_ref()) {
            if let Some(found) = e.near_place(z, None, top) {
                return (Metric::Distance, Ok(found));
            }
        }
        let found = match self.activity() {
            Some(a) => a.near_cells(cells.iter().copied(), top),
            None => Err("no run to find features in".into()),
        };
        (Metric::Direction, found)
    }

    /// The features of `list` that have values to draw on this cell map from
    /// the source on screen: one without would show as missing everywhere.
    fn with_values(&mut self, list: Vec<(Box<str>, f32)>) -> Vec<(Box<str>, f32)> {
        if self.current().axis() != Axis::Cells {
            return list;
        }
        let (space, source) = (self.space, self.source);
        let Some((activity, data)) = self.activity_and_data() else {
            return list;
        };
        let names = &data.spaces[space].points.names;
        list.into_iter()
            .filter(|(f, _)| activity.shows(f, source, space, names))
            .collect()
    }

    /// List the neighbours of the last click again (after its count or the
    /// source changed).
    fn rerun_near(&mut self) {
        let again = self.near.as_ref().map(|n| (n.name.clone(), n.centre));
        match again {
            Some((name, Centre::Cell)) => self.show_near(&name),
            Some((name, Centre::Feature)) => self.show_near_feature(&name),
            Some((name, Centre::Group { .. })) => {
                if let Some(g) = self.levels().iter().position(|l| **l == *name) {
                    self.show_near_group(g as u32);
                }
            }
            _ => {}
        }
    }

    /// List `step` more or fewer neighbours along [`NEAR_COUNTS`], and list
    /// those of the last click again.
    pub fn step_near_count(&mut self, step: isize) {
        let n = NEAR_COUNTS.len() as isize;
        let at = NEAR_COUNTS
            .iter()
            .position(|&c| c >= self.near_count)
            .unwrap_or(1) as isize;
        self.near_count = NEAR_COUNTS[(at + step).clamp(0, n - 1) as usize];
        self.rerun_near();
    }

    /// Features nearest group `g` of the grouping on screen, as for a cell
    /// (see [`Self::rank_near`]), drawn from its label.
    pub fn show_near_group(&mut self, g: u32) {
        let (Some(xy), Some(groups)) = (self.group_centre(g), self.groups()) else {
            return;
        };
        // Members by index, their names read from a shared handle on the
        // points: no copy per cell.
        let members: Vec<usize> = (0..groups.len()).filter(|&i| groups[i] == g).collect();
        let points = self.current().points.clone();
        let (name, space) = (self.levels()[g as usize].clone(), self.space);
        let cells: Vec<&str> = members.iter().map(|&i| &*points.names[i]).collect();
        let (metric, found) = self.rank_near(&cells);
        self.keep_near(name, Centre::Group { space, xy }, metric, found);
    }

    /// Show the features `found` around `centre`, or say why there are none.
    fn keep_near(
        &mut self,
        name: Box<str>,
        centre: Centre,
        metric: Metric,
        found: Result<Vec<(Box<str>, f32)>, String>,
    ) {
        match found.map(|f| self.with_values(f)) {
            Ok(features) => {
                if self.feature_space().is_none() {
                    self.note = Some(if self.root() == self.space {
                        format!(
                            "no features placed on this {} map; `senna layout {}` places them",
                            self.current().method,
                            self.current().method
                        )
                    } else {
                        "features are placed on the full map only; Z to go back".into()
                    });
                }
                self.near = Some(Near {
                    name,
                    centre,
                    metric,
                    features,
                    cells: Vec::new(),
                });
            }
            Err(e) => {
                self.near = None;
                self.note = Some(e);
            }
        }
    }

    /// Features nearest feature `feature`, and on a run with cells, the
    /// cells nearest it, by the metric of the map on screen: on a cell map,
    /// Euclidean from where it sits among the cells (its co-embedding); on a
    /// feature map, cosine of feature embeddings, and cells most up in it.
    pub fn show_near_feature(&mut self, feature: &str) {
        let top = self.near_count;
        let on_cells = self.current().kind == SpaceKind::FeaturesOnCells;
        let embedding = self
            .feature_embedding
            .get_or_insert_with(|| FeatureEmbedding::load(self.data.run.as_ref()));
        let placed = embedding.as_ref().ok().filter(|_| on_cells).and_then(|e| {
            let (at, i) = e.place_of(feature)?;
            Some((e.near_place(&at, Some(i), top)?, at))
        });
        let has_cells = self.data.has_axis(Axis::Cells);
        let (metric, features, cells) = match placed {
            Some((features, at)) => {
                let cells = match self.activity().filter(|_| has_cells) {
                    Some(a) => a.cells_near(&at, top),
                    None => Ok(Vec::new()),
                };
                (Metric::Distance, Ok(features), cells)
            }
            None => {
                let features = embedding
                    .as_ref()
                    .map_err(Clone::clone)
                    .and_then(|e| e.near(feature, top));
                let cells = match self.activity().filter(|_| has_cells) {
                    Some(a) => a.near_feature(feature, top),
                    None => Ok(Vec::new()),
                };
                (Metric::Cosine, features, cells)
            }
        };
        // Whatever failed is said, even when the other half has neighbours.
        let (features, cells, why) = match (features, cells) {
            (Ok(f), Ok(c)) => (f, c, None),
            (Err(e), Ok(c)) => (Vec::new(), c, Some(e)),
            (Ok(f), Err(e)) => (f, Vec::new(), Some(e)),
            (Err(e), Err(_)) => (Vec::new(), Vec::new(), Some(e)),
        };
        self.note = why;
        self.near = (!features.is_empty() || !cells.is_empty()).then(|| Near {
            name: feature.into(),
            centre: Centre::Feature,
            metric,
            features,
            cells,
        });
    }

    /// Pin the names near the clicked cell, else the feature shown; with
    /// nothing new to pin, clear the pins.
    pub fn pin(&mut self) {
        if let Some(n) = self.near.take() {
            self.note = Some(format!(
                "pinned {} names · k again clears",
                n.features.len() + n.cells.len()
            ));
            self.locked.push(n);
            return;
        }
        if let Some(Pick::One(f)) = &self.pick {
            let pinned = self
                .locked
                .iter()
                .any(|n| n.features.iter().any(|(x, _)| x == f));
            if !pinned {
                let placed = self
                    .feature_positions()
                    .into_iter()
                    .chain(
                        (self.current().axis() == Axis::Features).then(|| &*self.current().points),
                    )
                    .any(|p| p.names.iter().any(|n| n == f));
                self.note = Some(if placed {
                    format!("pinned {f} · k again clears")
                } else {
                    format!("{f} has no place on this map")
                });
                if placed {
                    self.locked.push(Near {
                        name: f.clone(),
                        centre: Centre::None,
                        metric: Metric::Distance,
                        features: vec![(f.clone(), f32::NAN)],
                        cells: Vec::new(),
                    });
                }
                return;
            }
        }
        if self.locked.is_empty() {
            self.note = Some("click a cell or show a feature, then p pins its names".into());
        } else {
            self.locked.clear();
            self.note = Some("pins cleared".into());
        }
    }

    /// Where features sit on the current cell map: the features-on-cells
    /// layout of the same method, if the run has one. A zoomed layout is laid
    /// out afresh, so features placed on the full map do not belong on it.
    pub fn feature_positions(&self) -> Option<&super::data::Points> {
        Some(&self.data.spaces[self.feature_space()?].points)
    }

    /// The space `feature_positions` are the points of.
    pub(super) fn feature_space(&self) -> Option<usize> {
        if self.root() != self.space {
            return None;
        }
        let method = &self.current().method;
        self.data
            .spaces
            .iter()
            .position(|s| s.kind == SpaceKind::FeaturesOnCells && s.method == *method)
    }

    /// Where a clicked cell and the features near it are drawn on the view
    /// on screen: `(cell space, feature space)`, either absent when the view
    /// has no place for it.
    pub(super) fn near_spaces(&self) -> (Option<usize>, Option<usize>) {
        let space = self.current();
        match space.kind {
            SpaceKind::Cells => (Some(self.space), self.feature_space()),
            SpaceKind::FeaturesOnCells => (space.backdrop, Some(self.space)),
            SpaceKind::Features => (None, Some(self.space)),
        }
    }

    /// The point named `name` in space `k` (its first, when repeated). Each
    /// space's index is built once and kept: rebuilding a cell map's on every
    /// view switch would stall the redraw.
    pub(super) fn point_of(&self, k: usize, name: &str) -> Option<usize> {
        let mut cache = self.name_index.borrow_mut();
        let index = cache.entry(k).or_insert_with(|| {
            let mut first = NameIndex::new();
            for (i, n) in self.data.spaces[k].points.names.iter().enumerate() {
                first.entry(n.clone()).or_insert(i);
            }
            first
        });
        index.get(name).copied()
    }

    /// Panel rows for the features near the clicked cell or feature, the
    /// feature being shown marked; each feature's row carries its name.
    pub fn near_rows(&self) -> Option<Vec<PaneRow>> {
        let near = self.near.as_ref()?;
        let shown = self.shown_feature();
        let mut out: Vec<PaneRow> = Vec::new();
        let say = match near.metric {
            Metric::Distance => "nearest (distance)",
            Metric::Cosine => "nearest (cosine)",
            Metric::Direction => "most up (ρ·(z − z̄))",
        };
        for (what, list) in [("features", &near.features), ("cells", &near.cells)] {
            if list.is_empty() {
                continue;
            }
            // Cells near a feature on a feature map lean its way.
            let say = match (what, near.metric) {
                ("cells", Metric::Cosine) => "most up in it",
                _ => say,
            };
            let features = what == "features";
            out.push((format!("{what} {say}: {}", near.name), None));
            out.extend(list.iter().map(|(f, v)| {
                let mark = if features && shown == Some(f.as_ref()) {
                    "▸"
                } else {
                    " "
                };
                let text = match near.metric {
                    Metric::Distance => format!("{mark} {f:<14} {:.2}", -v),
                    _ => format!("{mark} {f:<14} {v:+.2}"),
                };
                (text, features.then(|| f.clone()))
            }));
            out.push((String::new(), None));
        }
        out.push(("↑ ↓ step · p pins their names on the map".into(), None));
        Some(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn features_near_a_feature_rank_by_cosine_and_leave_it_out() {
        let names = ["a", "b", "c", "d"];
        // Lengths differ on purpose: only direction counts.
        let mut rows = Mat::from_row_slice(4, 2, &[1.0, 0.0, 5.0, 0.5, 0.0, 1.0, -2.0, 0.0]);
        l2_normalize_rows_inplace(&mut rows);
        let e = FeatureEmbedding {
            axis: activity::Axis::new(names.map(Into::into).to_vec()),
            rows,
            places: None,
        };
        let near = e.near("a", 2).unwrap();
        let got: Vec<&str> = near.iter().map(|(n, _)| n.as_ref()).collect();
        assert_eq!(got, ["b", "c"]);
        assert!((near[1].1).abs() < 1e-6);
        assert!(e.near("z", 2).is_err());
    }

    #[test]
    fn features_near_a_place_rank_by_distance_as_the_map_is_laid_out() {
        let names = ["near", "long", "off"];
        // "long" points the same way as the query but sits far out: cosine
        // would tie it with "near", distance does not.
        let places = Mat::from_row_slice(3, 2, &[1.0, 0.0, 10.0, 0.0, 0.0, 1.5]);
        let mut rows = places.clone();
        l2_normalize_rows_inplace(&mut rows);
        let e = FeatureEmbedding {
            axis: activity::Axis::new(names.map(Into::into).to_vec()),
            rows,
            places: Some(places),
        };
        let z = activity::Vector::from_vec(vec![1.2, 0.0]);
        let got = e.near_place(&z, None, 3).unwrap();
        let order: Vec<&str> = got.iter().map(|(n, _)| &**n).collect();
        assert_eq!(order, ["near", "off", "long"]);
        assert!((got[0].1 + 0.2).abs() < 1e-6);
        // A feature's own place: itself left out.
        let (at, i) = e.place_of("near").unwrap();
        let got = e.near_place(&at, Some(i), 1).unwrap();
        assert_eq!(&*got[0].0, "off");
        // No co-embedding of this width: no ranking by place.
        let wide = activity::Vector::from_vec(vec![1.0, 0.0, 0.0]);
        assert!(e.near_place(&wide, None, 1).is_none());
    }
}
