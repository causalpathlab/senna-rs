//! Relabel mode: visit clusters, stage decisions in a draft; lupin applies it.

use super::data::{group_counts, LabelKind, NONE};
use super::review::{Draft, Mark, Merge, Verdict};
use super::{Axis, Pick, Scene};
use std::collections::BTreeMap;

mod merge;
mod panel;

pub(crate) use panel::preview_lines;

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

/// A listed marker this low in the cluster is proposed for dropping.
const PROPOSE_DROP: f32 = 0.1;
/// A cluster whose best-fitting type differs from its label is flagged only
/// when that fit is at least this good.
const SUGGEST_FIT: f32 = 0.3;

/// Type names compare as lupin compares them: case-insensitive, with spaces
/// and underscores the same.
fn same_type(a: &str, b: &str) -> bool {
    super::rounds::label_key(a) == super::rounds::label_key(b)
}

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

pub(crate) struct Review {
    /// Index into `overview` of the cluster being visited.
    pub at: usize,
    pub draft: Draft,
    pub candidates: Vec<String>,
    /// Each candidate's fit (see `type_fits`) and how many markers it scored.
    pub fits: Vec<(String, f32, usize)>,
    /// The type `+` adds markers to and a label defaults to.
    pub target: Option<String>,
    pub rows: Vec<Row>,
    pub row: usize,
    /// Every cluster, in visiting order.
    pub overview: Vec<Overview>,
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
    /// Its label is a coarse group, to be refined to one of its members.
    pub coarse: bool,
}

impl Overview {
    /// Whether its markers point somewhere other than its label.
    #[must_use]
    pub fn suggests_change(&self) -> bool {
        match (&self.label, &self.best) {
            (_, None) => false,
            (None, Some(_)) => true,
            _ if self.coarse => false,
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
    pub(crate) fn new(overview: Vec<Overview>, draft: Draft) -> Self {
        Self {
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
        }
    }

    #[must_use]
    pub fn cluster(&self) -> i64 {
        self.overview[self.at].id
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
            .label_index(LabelKind::Cluster)
            .ok_or("this round has no clusters to relabel")?;
        let draft = Draft::load(&round.path);
        self.colour = Some(li);
        self.focus = None;
        self.refresh_groups();
        let overview = self.cluster_overview(li);
        self.review = Some(Review::new(overview, draft));
        self.visit(0);
        Ok(())
    }

    /// Every cluster of grouping `li`, in visiting order.
    fn cluster_overview(&mut self, li: usize) -> Vec<Overview> {
        let space = self.space;
        let markers = self.markers_by_type();
        let clusters = &self.data.labels[li];
        let ids = clusters.ids.clone();
        let n = ids.len();
        let groups = clusters.align(&self.data.spaces[space].points);
        let size = group_counts(groups.iter().copied(), n);
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
        if let Some((activity, data)) = self.activity_and_data() {
            let names = &data.spaces[space].points.names;
            if let Ok((per, features)) = activity.cluster_contrasts(space, names, &groups, n) {
                for (g, scores) in per.iter().enumerate() {
                    if !scores.is_empty() {
                        best[g] = best_fit(features, scores, &markers);
                    }
                }
            }
        }
        let mut out: Vec<Overview> = (0..n)
            .map(|g| Overview {
                id: ids[g],
                size: size[g],
                label: labels[g].clone(),
                best: best[g].clone(),
                coarse: labels[g].as_deref().is_some_and(|l| {
                    self.data
                        .round
                        .as_ref()
                        .is_some_and(|r| r.members_of(l).is_some())
                }),
            })
            .collect();
        out.sort_by_key(|o| {
            (
                o.label.is_some(),
                !(o.suggests_change() || o.coarse),
                usize::MAX - o.size,
            )
        });
        out
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
        r.at = i.min(r.overview.len().saturating_sub(1));
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
        let spell = |x: String| {
            markers
                .keys()
                .find(|t| same_type(t, &x))
                .cloned()
                .unwrap_or(x)
        };
        let mut candidates: Vec<String> = called.into_iter().map(spell).collect();
        candidates.dedup();
        // A coarse call is refined to one of its members.
        let (current, _) = self.cluster_call(id);
        let members: Vec<String> = current
            .as_deref()
            .and_then(|l| self.data.round.as_ref()?.members_of(l))
            .map(<[String]>::to_vec)
            .unwrap_or_default();
        for m in &members {
            let m = spell(m.clone());
            if !candidates.contains(&m) {
                candidates.push(m);
            }
        }
        for (t, fit, _) in fits.iter().take(FITTING_TYPES) {
            if *fit > 0.0 && !candidates.contains(t) {
                candidates.push(t.clone());
            }
        }
        let fit = |t: &str| {
            fits.iter()
                .find(|f| f.0 == t)
                .map_or(f32::NEG_INFINITY, |f| f.1)
        };
        let target = staged.clone().or_else(|| {
            candidates
                .iter()
                .filter(|c| members.is_empty() || members.iter().any(|m| same_type(m, c)))
                .max_by(|a, b| fit(a).total_cmp(&fit(b)))
                .cloned()
        });
        // Markers listed: the target's, and the best-fitting other candidate's.
        let alternative = candidates
            .iter()
            .filter(|c| Some(c.as_str()) != target.as_deref())
            .max_by(|a, b| fit(a).total_cmp(&fit(b)))
            .cloned();
        let listed: Vec<String> = target.iter().cloned().chain(alternative).collect();
        let rows = review_rows(
            &scores,
            &markers,
            &listed,
            target.as_deref(),
            staged.as_deref(),
        );
        candidates.sort_by(|a, b| fit(b).total_cmp(&fit(a)));
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
        let Some((activity, data)) = self.activity_and_data() else {
            return BTreeMap::new();
        };
        let names = &data.spaces[space].points.names;
        let scores = activity
            .contrast(space, names, Some(&mask))
            .map(|(scores, features)| features.iter().cloned().zip(scores).collect());
        match scores {
            Ok(scores) => scores,
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

/// The type whose markers fit best, given each feature's score.
fn best_fit(
    features: &[Box<str>],
    scores: &[f32],
    markers: &BTreeMap<String, Vec<Box<str>>>,
) -> Option<(String, f32)> {
    let score_of: BTreeMap<&str, f32> = features
        .iter()
        .zip(scores)
        .map(|(f, &v)| (f.as_ref(), v))
        .collect();
    type_fits(&score_of, markers)
        .into_iter()
        .next()
        .map(|(t, fit, _)| (t, fit))
}

/// Top DE features, then each candidate's best markers, with proposals. `−`
/// only for the labelled type: a low marker of another type faults the type.
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn type_names_match_across_spaces_commas_and_underscores() {
        assert!(same_type("CT_1", "ct 1"));
        assert!(same_type("CT 1, a", "CT_1_a"));
        assert!(!same_type("CT 1", "CT 1 a"));
    }
}
