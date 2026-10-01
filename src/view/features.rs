//! Features on the map: suggestions, the picked feature's activity, and
//! features near a clicked cell.

use super::*;

const NEAR: usize = 12;

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

/// Features shown around one clicked cell, or features and cells around one
/// clicked feature.
#[derive(Clone)]
pub(crate) struct Near {
    pub name: Box<str>,
    pub centre: Centre,
    pub features: Vec<(Box<str>, f32)>,
    /// Cells nearest a clicked feature (runs with cells in an embedding).
    pub cells: Vec<(Box<str>, f32)>,
}

/// The run's feature embedding with every row L2-normalized, so a product of
/// rows is their cosine.
pub(crate) struct FeatureEmbedding {
    axis: activity::Axis,
    rows: Mat,
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
        Ok(Self {
            axis: activity::Axis::new(names),
            rows,
        })
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

    /// Drop the suggestions. Returns whether there were any.
    pub fn clear_suggestions(&mut self) -> bool {
        self.suggestions.take().is_some()
    }

    /// Panel text for the suggestions of the view on screen, marking the
    /// feature being shown.
    pub fn suggestion_lines(&self) -> Option<Vec<String>> {
        let Suggestions { title, list, .. } = self.current_suggestions()?;
        let shown = match &self.pick {
            Some(Pick::One(f)) => Some(f.as_ref()),
            _ => None,
        };
        let mut out = vec![title.clone(), "g / G step · o observed · x close".into()];
        out.extend(list.iter().enumerate().map(|(k, (f, v))| {
            let mark = if shown == Some(f.as_ref()) {
                "▸"
            } else {
                " "
            };
            format!("{mark} {:>2}  {f:<16} {v:>6.2}", k + 1)
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

    /// Features nearest cell `cell` in the embedding.
    pub fn show_near(&mut self, cell: &str) {
        if self.current().kind != SpaceKind::Cells {
            return;
        }
        let Some(activity) = self.activity() else {
            return;
        };
        let found = activity.near_cells([cell], NEAR);
        self.keep_near(cell.into(), Centre::Cell, found);
    }

    /// Features nearest group `g` of the grouping on screen: those most up
    /// in its cells against the average cell, drawn from its label.
    pub fn show_near_group(&mut self, g: u32) {
        let (Some(xy), Some(groups)) = (self.group_centre(g), self.groups()) else {
            return;
        };
        // Members by index, their names read from a shared handle on the
        // points: no copy per cell.
        let members: Vec<usize> = (0..groups.len()).filter(|&i| groups[i] == g).collect();
        let points = self.current().points.clone();
        let (name, space) = (self.levels()[g as usize].clone(), self.space);
        let Some(activity) = self.activity() else {
            return;
        };
        let found = activity.near_cells(members.iter().map(|&i| &*points.names[i]), NEAR);
        self.keep_near(name, Centre::Group { space, xy }, found);
    }

    /// Show the features `found` around `centre`, or say why there are none.
    fn keep_near(
        &mut self,
        name: Box<str>,
        centre: Centre,
        found: Result<Vec<(Box<str>, f32)>, String>,
    ) {
        match found {
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

    /// Features nearest feature `feature` in the run's feature embedding,
    /// by cosine, and on a run with cells, the cells nearest it.
    pub fn show_near_feature(&mut self, feature: &str) {
        let embedding = self
            .feature_embedding
            .get_or_insert_with(|| FeatureEmbedding::load(self.data.run.as_ref()));
        let features = embedding
            .as_ref()
            .map_err(Clone::clone)
            .and_then(|e| e.near(feature, NEAR));
        let has_cells = self.data.has_axis(Axis::Cells);
        let cells = match self.activity().filter(|_| has_cells) {
            Some(a) => a.near_feature(feature, NEAR),
            None => Ok(Vec::new()),
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

    /// Sidebar text for the features near the clicked cell or feature.
    pub fn near_lines(&self) -> Option<Vec<String>> {
        let near = self.near.as_ref()?;
        let mut out = Vec::new();
        for (what, list) in [("features", &near.features), ("cells", &near.cells)] {
            if list.is_empty() {
                continue;
            }
            out.push(format!("{what} nearest {}", near.name));
            out.extend(list.iter().map(|(f, v)| format!("  {f:<14} {v:+.2}")));
            out.push(String::new());
        }
        out.push("p pins their names on the map".into());
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
        };
        let near = e.near("a", 2).unwrap();
        let got: Vec<&str> = near.iter().map(|(n, _)| n.as_ref()).collect();
        assert_eq!(got, ["b", "c"]);
        assert!((near[1].1).abs() < 1e-6);
        assert!(e.near("z", 2).is_err());
    }
}
