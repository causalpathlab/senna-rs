//! Relabel mode: visit clusters one at a time, gather the evidence for each
//! (differentially expressed features, the candidate types' markers, the
//! features nearest a clicked cell), stage decisions in a draft, and hand the
//! whole draft to lupin as one round at the end. Everything here only stages;
//! lupin applies.

use super::data::{LabelKind, SpaceKind, NONE};
use super::review::{Draft, Mark, Merge, Verdict};
use super::{Axis, Pick, Scene};
use std::collections::BTreeMap;

/// Rows of the review list: differentially expressed features first.
const TOP_DE: usize = 15;
/// A DE feature this far above the rest of the view (expected log fold
/// change) is proposed as a marker of the target type.
const PROPOSE_ADD: f32 = 0.5;
/// Markers listed per candidate type.
const MARKERS_PER_TYPE: usize = 6;
/// Candidate types added from how well their markers fit the cluster.
const FITTING_TYPES: usize = 3;
/// A type's fit is the mean of its best this-many marker scores, missing
/// ones counted as zero, so a type needs several strong markers to fit well
/// rather than one or two.
const FIT_TOP: usize = 5;

/// Type names compare as lupin compares them: case-insensitive, with spaces
/// and underscores the same.
fn same_type(a: &str, b: &str) -> bool {
    let norm = |s: &str| s.to_lowercase().replace(' ', "_");
    norm(a) == norm(b)
}
/// A listed marker this low in the cluster is proposed for dropping.
const PROPOSE_DROP: f32 = 0.1;
/// Features shown around a clicked cell.
const NEAR: usize = 12;
/// A cluster whose best-fitting type differs from its label is flagged only
/// when that fit is at least this good.
const SUGGEST_FIT: f32 = 0.3;

/// One feature in the review list.
pub(crate) struct Row {
    pub feature: Box<str>,
    /// Expected log fold change, this cluster over the rest of the view.
    pub score: f32,
    /// Candidate types that list it as a marker.
    pub marker_of: Vec<String>,
    /// What the view suggests doing with it, if anything.
    pub proposal: Option<Mark>,
}

/// Features shown around one clicked cell.
pub(crate) struct Near {
    pub cell: Box<str>,
    pub features: Vec<(Box<str>, f32)>,
}

pub(crate) struct Review {
    /// Clusters in visiting order: unassigned first, then by size.
    pub order: Vec<i64>,
    pub at: usize,
    pub draft: Draft,
    pub candidates: Vec<String>,
    /// How well each candidate type's markers fit the cluster: the mean
    /// expected log fold change of its markers, and how many were scored.
    pub fits: Vec<(String, f32, usize)>,
    /// The type `+` adds markers to and a label defaults to.
    pub target: Option<String>,
    pub rows: Vec<Row>,
    pub row: usize,
    /// Every cluster at a glance, in visiting order.
    pub overview: Vec<Overview>,
    /// Merge mode, when on.
    pub merge: Option<MergeSel>,
    /// The last answer from lupin's preview, as text.
    pub preview: Option<Vec<String>>,
}

/// One cluster in the overview.
pub(crate) struct Overview {
    pub id: i64,
    pub size: usize,
    /// The round's current label (`None`: unassigned).
    pub label: Option<String>,
    /// The type whose markers fit it best, and how well.
    pub best: Option<(String, f32)>,
}

impl Overview {
    /// Whether its markers point somewhere other than its label.
    #[must_use]
    pub fn suggests_change(&self) -> bool {
        match (&self.label, &self.best) {
            (_, None) => false,
            (None, Some(_)) => true,
            (Some(l), Some((b, fit))) => *fit >= SUGGEST_FIT && !same_type(l, b),
        }
    }
}

/// Merge mode: clusters chosen by keyboard, and what they would be together.
pub(crate) struct MergeSel {
    /// Index into the overview.
    pub cursor: usize,
    pub chosen: std::collections::BTreeSet<i64>,
    /// Per level of the cluster grouping: chosen or not (for drawing).
    pub levels: Vec<bool>,
    /// The best-fitting type of the chosen clusters together.
    pub best: Option<(String, f32)>,
}

impl Review {
    #[must_use]
    pub fn cluster(&self) -> i64 {
        self.order[self.at]
    }
}

impl Scene {
    /// Enter relabel mode on the round on screen.
    pub fn enter_review(&mut self) -> Result<(), String> {
        if self.current().axis() != Axis::Cells {
            return Err("relabel from a cell view".into());
        }
        let round = self
            .data
            .round
            .as_ref()
            .ok_or("this view has no annotation round to relabel")?;
        let li = self
            .data
            .labels
            .iter()
            .position(|l| l.kind == LabelKind::Cluster)
            .ok_or("this round has no clusters to relabel")?;
        let draft = Draft::load(&round.path);
        self.colour = Some(li);
        self.focus = None;
        self.refresh_groups();
        let overview = self.cluster_overview(li);
        self.review = Some(Review {
            order: overview.iter().map(|o| o.id).collect(),
            at: 0,
            draft,
            candidates: Vec::new(),
            fits: Vec::new(),
            target: None,
            rows: Vec::new(),
            row: 0,
            overview,
            merge: None,
            preview: None,
        });
        self.visit(0);
        Ok(())
    }

    /// Every cluster of grouping `li` with its size, label and best-fitting
    /// type, in visiting order: unassigned first, then those whose markers
    /// suggest a change, then the rest; larger first within each.
    fn cluster_overview(&mut self, li: usize) -> Vec<Overview> {
        let space = self.space;
        let markers = self.markers_by_type();
        let clusters = &self.data.labels[li];
        let ids = clusters.ids.clone();
        let n = ids.len();
        let groups = clusters.align(&self.data.spaces[space].points);
        let mut size = vec![0usize; n];
        for &g in &groups {
            if (g as usize) < n {
                size[g as usize] += 1;
            }
        }
        let labels: Vec<Option<String>> = ids
            .iter()
            .map(|id| {
                self.data
                    .round
                    .as_ref()
                    .and_then(|r| r.call(&id.to_string()).0)
            })
            .collect();
        let mut best: Vec<Option<(String, f32)>> = vec![None; n];
        if self.activity().is_some() {
            let names = &self.data.spaces[space].points.names;
            let activity = self.activity.as_mut().expect("made above");
            if let Ok((per, features)) = activity.cluster_contrasts(space, names, &groups, n) {
                for (g, scores) in per.iter().enumerate() {
                    if scores.is_empty() {
                        continue;
                    }
                    let score_of: BTreeMap<&str, f32> = features
                        .iter()
                        .zip(scores)
                        .map(|(f, &v)| (f.as_ref(), v))
                        .collect();
                    best[g] = type_fits(&score_of, &markers)
                        .into_iter()
                        .next()
                        .map(|(t, fit, _)| (t, fit));
                }
            }
        }
        let mut out: Vec<Overview> = (0..n)
            .map(|g| Overview {
                id: ids[g],
                size: size[g],
                label: labels[g].clone(),
                best: best[g].clone(),
            })
            .collect();
        out.sort_by_key(|o| (o.label.is_some(), !o.suggests_change(), usize::MAX - o.size));
        out
    }

    /// Enter merge mode with the current cluster chosen.
    pub fn begin_merge(&mut self) {
        let Some(r) = self.review.as_mut() else {
            return;
        };
        let id = r.cluster();
        r.merge = Some(MergeSel {
            cursor: r.at,
            chosen: std::iter::once(id).collect(),
            levels: Vec::new(),
            best: None,
        });
        self.refresh_merge();
    }

    /// Choose or unchoose the cluster under the merge cursor.
    pub fn toggle_merge_cursor(&mut self) {
        let Some(r) = self.review.as_mut() else {
            return;
        };
        let Some(m) = r.merge.as_mut() else { return };
        let id = r.overview[m.cursor].id;
        if !m.chosen.remove(&id) {
            m.chosen.insert(id);
        }
        self.refresh_merge();
    }

    /// Recompute what the chosen clusters would be together, and which to
    /// draw.
    fn refresh_merge(&mut self) {
        let Some(li) = self.colour else { return };
        let Some(chosen) = self
            .review
            .as_ref()
            .and_then(|r| r.merge.as_ref())
            .map(|m| m.chosen.clone())
        else {
            return;
        };
        let ids = &self.data.labels[li].ids;
        let levels: Vec<bool> = ids.iter().map(|id| chosen.contains(id)).collect();
        let space = self.space;
        let markers = self.markers_by_type();
        let mask: Vec<bool> = self
            .groups()
            .map(|g| {
                g.iter()
                    .map(|&x| levels.get(x as usize).copied().unwrap_or(false))
                    .collect()
            })
            .unwrap_or_default();
        let mut best = None;
        if !mask.is_empty() && self.activity().is_some() {
            let names = &self.data.spaces[space].points.names;
            let activity = self.activity.as_mut().expect("made above");
            if let Ok((scores, features)) = activity.contrast(space, names, Some(&mask)) {
                let score_of: BTreeMap<&str, f32> = features
                    .iter()
                    .zip(&scores)
                    .map(|(f, &v)| (f.as_ref(), v))
                    .collect();
                best = type_fits(&score_of, &markers)
                    .into_iter()
                    .next()
                    .map(|(t, fit, _)| (t, fit));
            }
        }
        if let Some(m) = self.review.as_mut().and_then(|r| r.merge.as_mut()) {
            m.levels = levels;
            m.best = best;
        }
    }

    /// Sidebar text for merge mode.
    pub fn merge_lines(&self) -> Option<Vec<String>> {
        let r = self.review.as_ref()?;
        let m = r.merge.as_ref()?;
        let chosen: Vec<&Overview> = r
            .overview
            .iter()
            .filter(|o| m.chosen.contains(&o.id))
            .collect();
        let total: usize = chosen.iter().map(|o| o.size).sum();
        let mut out = vec![
            format!("merge · {} clusters · {total} cells", chosen.len()),
            match &m.best {
                Some((t, fit)) => format!("together, markers fit {t} best ({fit:+.2})"),
                None => "together: no marker fit".into(),
            },
            if chosen.len() < 2 {
                "next: choose at least one more cluster (↑↓, space)".into()
            } else {
                "next: enter to name the merged cluster".into()
            },
            String::new(),
        ];
        for o in chosen {
            out.push(format!(
                "  C{:<4} {:>6} cells  now {}",
                o.id,
                o.size,
                o.label.as_deref().unwrap_or("unassigned")
            ));
        }
        Some(out)
    }

    /// Sidebar text for the cluster overview (the left panel).
    pub fn overview_lines(&self) -> Option<Vec<String>> {
        let r = self.review.as_ref()?;
        let cursor = r.merge.as_ref().map_or(r.at, |m| m.cursor);
        let cursor_best = r
            .overview
            .get(cursor)
            .and_then(|o| o.best.as_ref())
            .map(|b| b.0.clone());
        let merged: std::collections::BTreeSet<i64> = r
            .draft
            .merges
            .iter()
            .flat_map(|m| m.clusters.iter().copied())
            .collect();
        let mut out = vec![
            format!("clusters · {} decided", r.draft.decided()),
            "? unassigned  → markers suggest  ✓ decided".into(),
            String::new(),
        ];
        for (k, o) in r.overview.iter().enumerate() {
            let here = if k == cursor { "▸" } else { " " };
            let pick = match &r.merge {
                Some(m) if m.chosen.contains(&o.id) => "[x]",
                Some(_) => "[ ]",
                None => "",
            };
            let staged = r.draft.clusters.get(&o.id).and_then(|c| c.verdict.as_ref());
            let status = if let Some(v) = staged {
                match v {
                    Verdict::Label { label, .. } | Verdict::Keep { label, .. } => {
                        format!("✓ {label}")
                    }
                }
            } else if merged.contains(&o.id) {
                "✓ merged".into()
            } else if o.label.is_none() {
                match &o.best {
                    Some((b, _)) => format!("? → {b}"),
                    None => "?".into(),
                }
            } else if o.suggests_change() {
                format!("→ {}", o.best.as_ref().map_or("", |b| b.0.as_str()))
            } else {
                o.label.clone().unwrap_or_default()
            };
            let similar = r.merge.is_some()
                && k != cursor
                && cursor_best.is_some()
                && o.best.as_ref().map(|b| &b.0) == cursor_best.as_ref();
            out.push(format!(
                "{here}{pick}{}C{:<4}{:>6} {status}",
                if similar { "≈" } else { " " },
                o.id,
                o.size
            ));
        }
        Some(out)
    }

    /// Leave relabel mode, keeping the draft on disk.
    pub fn leave_review(&mut self) {
        if let Some(r) = self.review.take() {
            if let Err(e) = r.draft.save() {
                self.note = Some(format!("could not save the draft: {e}"));
            }
        }
    }

    /// Go to the `i`-th cluster of the visiting order and gather its evidence.
    pub fn visit(&mut self, i: usize) {
        let Some(r) = self.review.as_mut() else {
            return;
        };
        r.at = i.min(r.order.len().saturating_sub(1));
        let id = r.cluster();
        let called = self
            .data
            .round
            .as_ref()
            .map(|round| round.candidates(&id.to_string()))
            .unwrap_or_default();
        let staged = r.draft.clusters.get(&id).and_then(|c| match &c.verdict {
            Some(Verdict::Label { label, .. } | Verdict::Keep { label, .. }) => Some(label.clone()),
            None => None,
        });
        // Focus the cluster, so it is drawn on top.
        let li = self.colour;
        self.focus = li.and_then(|li| {
            let c = &self.data.labels[li];
            c.ids.iter().position(|&x| x == id).map(|g| g as u32)
        });
        let scores = self.cluster_scores();
        let markers = self.markers_by_type();
        let fits = type_fits(
            &scores.iter().map(|(f, &v)| (f.as_ref(), v)).collect(),
            &markers,
        );
        // Candidates: lupin's calls (spelled as the marker table spells them),
        // then the types whose markers fit best.
        let mut candidates: Vec<String> = called
            .into_iter()
            .map(|c| {
                markers
                    .keys()
                    .find(|t| same_type(t, &c))
                    .cloned()
                    .unwrap_or(c)
            })
            .collect();
        candidates.dedup();
        for (t, fit, _) in fits.iter().take(FITTING_TYPES) {
            if *fit > 0.0 && !candidates.contains(t) {
                candidates.push(t.clone());
            }
        }
        let fit_of = |t: &str| fits.iter().find(|f| f.0 == t).map(|f| f.1);
        // The target: what was staged, else the best-fitting candidate.
        let target = staged.clone().or_else(|| {
            candidates
                .iter()
                .max_by(|a, b| {
                    fit_of(a)
                        .unwrap_or(f32::NEG_INFINITY)
                        .total_cmp(&fit_of(b).unwrap_or(f32::NEG_INFINITY))
                })
                .cloned()
        });
        // Markers listed: the target's, and the best-fitting other candidate's.
        let alternative = candidates
            .iter()
            .filter(|c| Some(c.as_str()) != target.as_deref())
            .max_by(|a, b| {
                fit_of(a)
                    .unwrap_or(f32::NEG_INFINITY)
                    .total_cmp(&fit_of(b).unwrap_or(f32::NEG_INFINITY))
            })
            .cloned();
        let listed: Vec<String> = target.iter().cloned().chain(alternative).collect();
        let rows = review_rows(
            &scores,
            &markers,
            &listed,
            target.as_deref(),
            staged.as_deref(),
        );
        candidates.sort_by(|a, b| {
            fit_of(b)
                .unwrap_or(f32::NEG_INFINITY)
                .total_cmp(&fit_of(a).unwrap_or(f32::NEG_INFINITY))
        });
        let r = self.review.as_mut().expect("checked above");
        r.fits = candidates
            .iter()
            .filter_map(|c| fits.iter().find(|f| &f.0 == c).cloned())
            .collect();
        r.candidates = candidates;
        r.target = target;
        r.rows = rows;
        r.row = 0;
        r.preview = None;
    }

    /// Every feature's expected log fold change, the focused cluster over the
    /// rest of the view.
    fn cluster_scores(&mut self) -> BTreeMap<Box<str>, f32> {
        let space = self.space;
        let (Some(f), Some(groups)) = (self.focus, self.groups()) else {
            return BTreeMap::new();
        };
        let mask: Vec<bool> = groups.iter().map(|&g| g == f).collect();
        if self.activity().is_none() {
            return BTreeMap::new();
        }
        let names = &self.data.spaces[space].points.names;
        let activity = self.activity.as_mut().expect("made above");
        match activity.contrast(space, names, Some(&mask)) {
            Ok((scores, features)) => features.iter().cloned().zip(scores).collect(),
            Err(e) => {
                self.note = Some(e);
                BTreeMap::new()
            }
        }
    }

    /// The marker table, as the features listed under each type.
    fn markers_by_type(&self) -> BTreeMap<String, Vec<Box<str>>> {
        let mut out: BTreeMap<String, Vec<Box<str>>> = BTreeMap::new();
        if let Some(m) = self.markers() {
            for (f, &g) in &m.by_name {
                out.entry(m.levels[g as usize].to_string())
                    .or_default()
                    .push(f.clone());
            }
        }
        out
    }

    /// Stage `mark` for the selected row's feature (or clear it with `None`).
    pub fn mark_row(&mut self, mark: Option<bool>) {
        let Some(r) = self.review.as_mut() else {
            return;
        };
        let Some(row) = r.rows.get(r.row) else { return };
        let id = r.cluster();
        let feature = row.feature.to_string();
        let score = row.score;
        let m = match mark {
            None => {
                r.draft.cluster(id).marks.remove(&feature);
                return;
            }
            Some(true) => match &r.target {
                Some(t) => Mark::Include {
                    cell_type: t.clone(),
                },
                None => {
                    self.note = Some("choose a target type first (tab)".into());
                    return;
                }
            },
            Some(false) => {
                let t = row.marker_of.first().cloned().or_else(|| r.target.clone());
                match t {
                    Some(t) => Mark::Exclude { cell_type: t },
                    None => {
                        self.note = Some("this feature is not anyone's marker".into());
                        return;
                    }
                }
            }
        };
        r.draft.cluster(id).marks.insert(feature, (m, score));
    }

    /// Stage every proposal of the current cluster.
    pub fn accept_proposals(&mut self) {
        let Some(r) = self.review.as_mut() else {
            return;
        };
        let id = r.cluster();
        let props: Vec<(String, Mark, f32)> = r
            .rows
            .iter()
            .filter_map(|row| Some((row.feature.to_string(), row.proposal.clone()?, row.score)))
            .collect();
        let n = props.len();
        let c = r.draft.cluster(id);
        for (f, m, v) in props {
            c.marks.insert(f, (m, v));
        }
        self.note = Some(format!("staged {n} proposed marker edits"));
    }

    /// Cycle the target type through the candidates.
    pub fn next_target(&mut self) {
        let Some(r) = self.review.as_mut() else {
            return;
        };
        if r.candidates.is_empty() {
            self.note = Some("no candidate types; type one with L".into());
            return;
        }
        let at = r
            .target
            .as_ref()
            .and_then(|t| r.candidates.iter().position(|c| c == t));
        r.target = Some(r.candidates[at.map_or(0, |i| (i + 1) % r.candidates.len())].clone());
    }

    /// Stage a verdict for the current cluster.
    pub fn stage_verdict(&mut self, verdict: Verdict) {
        if let Some(r) = self.review.as_mut() {
            if let Verdict::Label { label, .. } = &verdict {
                r.target = Some(label.clone());
            }
            let id = r.cluster();
            r.draft.cluster(id).verdict = Some(verdict);
        }
    }

    /// Stage a merge of the clusters chosen in merge mode, and leave it.
    pub fn stage_merge(&mut self, label: String, rationale: String) {
        let Some(r) = self.review.as_mut() else {
            return;
        };
        let Some(m) = r.merge.take() else { return };
        r.draft.merges.push(Merge {
            clusters: m.chosen.into_iter().collect(),
            label,
            rationale,
        });
    }

    /// A rationale drafted from what was staged for the current cluster.
    pub fn drafted_rationale(&self) -> String {
        let Some(r) = self.review.as_ref() else {
            return String::new();
        };
        let id = r.cluster();
        let kept: Vec<String> = r
            .draft
            .clusters
            .get(&id)
            .map(|c| {
                let mut kept: Vec<(&String, f32)> = c
                    .marks
                    .iter()
                    .filter(|(_, (m, _))| matches!(m, Mark::Include { .. }))
                    .map(|(f, (_, v))| (f, *v))
                    .collect();
                kept.sort_by(|a, b| b.1.total_cmp(&a.1));
                kept.iter()
                    .take(5)
                    .map(|(f, v)| format!("{f} {v:+.1}"))
                    .collect()
            })
            .unwrap_or_default();
        if kept.is_empty() {
            let top: Vec<String> = r
                .rows
                .iter()
                .take(3)
                .map(|row| format!("{} {:+.1}", row.feature, row.score))
                .collect();
            format!("top features here: {}", top.join(", "))
        } else {
            format!("marked by {}", kept.join(", "))
        }
    }

    /// Features nearest cell `cell` in the embedding; shown on the map and in
    /// the sidebar.
    pub fn show_near(&mut self, cell: &str) {
        if self.current().kind != SpaceKind::Cells {
            return;
        }
        let Some(activity) = self.activity() else {
            return;
        };
        match activity.near_cell(cell, NEAR) {
            Ok(features) => {
                self.near = Some(super::relabel::Near {
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

    /// Pin feature names on the map: the features nearest the clicked cell,
    /// else the feature on screen. Pinned names stay while other cells are
    /// clicked, the camera moves or the colouring changes. With nothing new
    /// to pin, the pins are cleared.
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

    /// Sidebar text for relabel mode.
    pub fn review_lines(&self) -> Option<Vec<String>> {
        let r = self.review.as_ref()?;
        let id = r.cluster();
        let (label, _) = self.cluster_call(id);
        let staged = r.draft.clusters.get(&id);
        let size = self.focus.and_then(|f| {
            let g = self.groups()?;
            Some(g.iter().filter(|&&x| x == f && x != NONE).count())
        });
        let marks = staged.map_or(0, |c| c.marks.len());
        let verdict = staged.and_then(|c| c.verdict.as_ref());
        let merged = r.draft.merges.iter().find(|m| m.clusters.contains(&id));
        let next = if verdict.is_some() || merged.is_some() {
            "next: ] for the next cluster · S when done (p previews)"
        } else if marks == 0 {
            "next: check the fit below, then + / - features (a accepts ?), then L"
        } else {
            "next: L to label it (or K to keep it, M to merge it with others)"
        };
        let mut out = vec![
            format!(
                "relabel · C{id} ({} of {}) · {} decided",
                r.at + 1,
                r.order.len(),
                r.draft.decided()
            ),
            format!(
                "{} cells · now {}",
                size.unwrap_or(0),
                label.as_deref().unwrap_or("unassigned")
            ),
            next.to_string(),
        ];
        if let Some(v) = staged.and_then(|c| c.verdict.as_ref()) {
            out.push(match v {
                Verdict::Label { label, .. } => format!("staged: label {label}"),
                Verdict::Keep { label, .. } => format!("staged: keep {label}"),
            });
        }
        if let Some(m) = merged {
            let others: Vec<String> = m
                .clusters
                .iter()
                .filter(|&&c| c != id)
                .map(|c| format!("C{c}"))
                .collect();
            out.push(format!(
                "staged: merge with {} as {}",
                others.join(" "),
                m.label
            ));
        }
        out.push(format!("target {}", r.target.as_deref().unwrap_or("-")));
        out.push("markers fit here (tab picks the target):".into());
        for c in &r.candidates {
            let fit = r.fits.iter().find(|f| &f.0 == c);
            let mark = if r.target.as_ref() == Some(c) {
                "▸"
            } else {
                " "
            };
            out.push(match fit {
                Some((_, v, n)) => format!(" {mark} {c:<22} {v:+.2}  ({n} markers)"),
                None => format!(" {mark} {c:<22}     ·"),
            });
        }
        out.push(String::new());
        out.push("   mark feature       here  marker of".into());
        for (k, row) in r.rows.iter().enumerate() {
            let staged = staged.and_then(|c| c.marks.get(row.feature.as_ref()));
            let sign = match (staged, &row.proposal) {
                (Some((Mark::Include { .. }, _)), _) => "+",
                (Some((Mark::Exclude { .. }, _)), _) => "−",
                (None, Some(Mark::Include { .. })) => "?+",
                (None, Some(Mark::Exclude { .. })) => "?−",
                (None, None) => "",
            };
            let cursor = if k == r.row { "▸" } else { " " };
            let score = if row.score.is_finite() {
                format!("{:+5.1}", row.score)
            } else {
                "    ·".into()
            };
            out.push(format!(
                "{cursor}{sign:<2} {:<12} {score}  {}",
                row.feature,
                row.marker_of.join(",")
            ));
        }
        if let Some(near) = &self.near {
            out.push(String::new());
            out.push(format!("near {} (k pins their names)", near.cell));
            let names: Vec<&str> = near.features.iter().map(|(f, _)| f.as_ref()).collect();
            out.push(format!("  {}", names.join(" ")));
        }
        if let Some(p) = &r.preview {
            out.push(String::new());
            out.extend(p.iter().cloned());
        }
        Some(out)
    }

    /// Show the selected row's feature on the map.
    pub fn show_row(&mut self) {
        let f = self
            .review
            .as_ref()
            .and_then(|r| r.rows.get(r.row))
            .map(|row| row.feature.clone());
        if let Some(f) = f {
            self.set_pick(Pick::One(f));
        }
    }
}

impl Scene {
    /// Sidebar text for the features near the clicked cell, outside relabel
    /// mode.
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

/// How well each type's markers fit, best first: the mean of its best
/// `FIT_TOP` marker scores (missing ones as zero), with how many it has.
fn type_fits(
    score_of: &BTreeMap<&str, f32>,
    markers: &BTreeMap<String, Vec<Box<str>>>,
) -> Vec<(String, f32, usize)> {
    let mut fits: Vec<(String, f32, usize)> = markers
        .iter()
        .filter_map(|(t, ms)| {
            let mut v: Vec<f32> = ms
                .iter()
                .filter_map(|m| score_of.get(m.as_ref()).copied())
                .filter(|v| v.is_finite())
                .collect();
            if v.is_empty() {
                return None;
            }
            v.sort_by(|a, b| b.total_cmp(a));
            let fit = v.iter().take(FIT_TOP).sum::<f32>() / FIT_TOP as f32;
            Some((t.clone(), fit, v.len()))
        })
        .collect();
    fits.sort_by(|a, b| b.1.total_cmp(&a.1));
    fits
}

/// The cluster's features: the most differentially expressed, then each
/// candidate type's best-scoring markers, with proposals. `+` goes to the
/// target type; `−` only to markers of the type the cluster was labelled
/// (a low marker of some other candidate says the candidate is wrong, not
/// that its marker is).
fn review_rows(
    scores: &BTreeMap<Box<str>, f32>,
    markers: &BTreeMap<String, Vec<Box<str>>>,
    candidates: &[String],
    target: Option<&str>,
    labelled: Option<&str>,
) -> Vec<Row> {
    let marker_of = |f: &str| -> Vec<String> {
        candidates
            .iter()
            .filter(|t| {
                markers
                    .get(*t)
                    .is_some_and(|ms| ms.iter().any(|m| m.as_ref() == f))
            })
            .cloned()
            .collect()
    };
    let target_lists = |f: &str| {
        target.is_some_and(|t| {
            markers
                .get(t)
                .is_some_and(|ms| ms.iter().any(|m| m.as_ref() == f))
        })
    };
    let mut ranked: Vec<(&Box<str>, f32)> = scores
        .iter()
        .map(|(f, &v)| (f, v))
        .filter(|(_, v)| v.is_finite())
        .collect();
    ranked.sort_by(|a, b| b.1.total_cmp(&a.1));
    let mut rows: Vec<Row> = ranked
        .into_iter()
        .take(TOP_DE)
        .map(|(f, v)| Row {
            feature: f.clone(),
            score: v,
            marker_of: marker_of(f),
            proposal: (v >= PROPOSE_ADD && !target_lists(f))
                .then(|| {
                    target.map(|t| Mark::Include {
                        cell_type: t.to_string(),
                    })
                })
                .flatten(),
        })
        .collect();
    for t in candidates {
        let Some(ms) = markers.get(t) else { continue };
        let mut listed: Vec<(&Box<str>, f32)> = ms
            .iter()
            .map(|m| (m, scores.get(m).copied().unwrap_or(f32::NAN)))
            .collect();
        listed.sort_by(|a, b| {
            let key = |v: f32| if v.is_finite() { v } else { f32::NEG_INFINITY };
            key(b.1).total_cmp(&key(a.1))
        });
        for (m, v) in listed.into_iter().take(MARKERS_PER_TYPE) {
            if rows.iter().any(|r| r.feature == *m) {
                continue;
            }
            let proposal =
                (labelled == Some(t.as_str()) && v < PROPOSE_DROP).then(|| Mark::Exclude {
                    cell_type: t.clone(),
                });
            rows.push(Row {
                feature: m.clone(),
                score: v,
                marker_of: marker_of(m),
                proposal,
            });
        }
    }
    rows
}

/// Lupin's preview, as sidebar lines.
#[must_use]
pub fn preview_lines(v: &serde_json::Value) -> Vec<String> {
    let mut out = vec![format!(
        "preview · {} cells would change{}",
        v["cells_changed"].as_u64().unwrap_or(0),
        if v["rescored"].as_bool() == Some(false) {
            " (calls not rescored)"
        } else {
            ""
        }
    )];
    if let Some(cs) = v["clusters"].as_object() {
        for (id, c) in cs {
            let s = |k: &str| c[k].as_str().unwrap_or("unassigned").to_string();
            let top = c["calls"][0]["label"].as_str().unwrap_or("-");
            out.push(format!(
                "  C{id}: {} → {}   top call {top}",
                s("label_before"),
                s("label_after")
            ));
        }
    }
    if let Some(ms) = v["markers"].as_object() {
        for (t, m) in ms {
            let n = |k: &str| m[k].as_array().map_or(0, Vec::len);
            out.push(format!("  markers {t}: +{} −{}", n("added"), n("dropped")));
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_preview_reads_as_before_and_after_per_cluster() {
        let v: serde_json::Value = serde_json::from_str(
            r#"{"rescored":false,"clusters":{"7":{"label_before":"CT1","label_after":"CT2",
                "top_before":"CT1","calls":[{"label":"CT1","score":null,"q":null,"support":null}]}},
                "cells_changed":388,"markers":{"CT2":{"added":["GENE1","GENE2"],"dropped":[]}}}"#,
        )
        .unwrap();
        let lines = preview_lines(&v);
        assert_eq!(
            lines[0],
            "preview · 388 cells would change (calls not rescored)"
        );
        assert!(lines[1].contains("C7: CT1 → CT2"));
        assert!(lines[2].contains("markers CT2: +2 −0"));
    }

    #[test]
    fn type_names_match_across_spaces_and_underscores() {
        assert!(same_type("B_cells", "b cells"));
        assert!(!same_type("B cells", "B cells memory"));
    }
}
