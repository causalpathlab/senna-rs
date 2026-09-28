//! Features on the map: suggestions, the picked feature's activity, and
//! features near a clicked cell.

use super::*;

const NEAR: usize = 12;

/// Features shown around one clicked cell.
pub(crate) struct Near {
    pub cell: Box<str>,
    pub features: Vec<(Box<str>, f32)>,
}

impl Scene {
    pub fn has_suggestions(&self) -> bool {
        self.current_suggestions().is_some()
    }

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
        const TOP: usize = 25;
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
                self.note = Some(e);
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
        match activity.near_cell(cell, NEAR) {
            Ok(features) => {
                self.near = Some(Near {
                    cell: cell.into(),
                    features,
                });
            }
            Err(e) => {
                self.near = None;
                self.note = Some(e);
            }
        }
    }

    /// Pin the names near the clicked cell, else the feature shown; with
    /// nothing new to pin, clear the pins.
    pub fn pin(&mut self) {
        if let Some(n) = self.near.take() {
            self.note = Some(format!(
                "pinned {} names · k or l again clears",
                n.features.len()
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
                        (self.current().axis() == Axis::Features).then(|| &self.current().points),
                    )
                    .any(|p| p.names.iter().any(|n| n == f));
                self.note = Some(if placed {
                    format!("pinned {f} · k or l again clears")
                } else {
                    format!("{f} has no place on this map")
                });
                if placed {
                    self.locked.push(Near {
                        cell: f.clone(),
                        features: vec![(f.clone(), f32::NAN)],
                    });
                }
                return;
            }
        }
        if self.locked.is_empty() {
            self.note = Some("click a cell or show a feature, then k pins its names".into());
        } else {
            self.locked.clear();
            self.note = Some("pins cleared".into());
        }
    }

    /// Where features sit on the current cell map: the features-on-cells
    /// layout of the same method, if the run has one.
    pub fn feature_positions(&self) -> Option<&super::data::Points> {
        let root = &self.data.spaces[self.root()];
        self.data
            .spaces
            .iter()
            .find(|s| s.kind == SpaceKind::FeaturesOnCells && s.method == root.method)
            .map(|s| &s.points)
    }

    /// Sidebar text for the features near the clicked cell.
    pub fn near_lines(&self) -> Option<Vec<String>> {
        let near = self.near.as_ref()?;
        let mut out = vec![format!("features nearest {}", near.cell)];
        out.extend(
            near.features
                .iter()
                .map(|(f, v)| format!("  {f:<14} {v:+.2}")),
        );
        out.push(String::new());
        out.push("k or l pins their names on the map".into());
        Some(out)
    }
}
