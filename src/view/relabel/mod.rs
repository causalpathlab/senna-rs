//! Relabel mode: visit clusters, stage decisions in a draft; lupin applies it.

use super::activity::{Activity, GroupSums, Source};
use super::data::LabelKind;
use super::review::{Draft, Mark, Merge, Verdict};
use super::{Axis, Pick, Scene};
use fit::Evidence;
use std::collections::BTreeMap;
use std::rc::Rc;

mod fit;
mod live;
mod merge;
mod panel;

pub(crate) use live::{parse_scores, Live};
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

/// The space and grouping per-group sums were taken on.
type SumsKey = (usize, Option<usize>);

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
    /// lupin's scores with the staged marker edits, when there are any.
    pub live: Option<Live>,
    evidence: Rc<Evidence>,
    /// Per-group sums for the (space, grouping) on screen, taken once.
    sums: Option<(SumsKey, Result<GroupSums, String>)>,
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
    /// lupin's rescored top call, when it disagrees with the label.
    pub disputed: Option<String>,
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
            live: None,
            evidence: Rc::default(),
            sums: None,
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
        let markers = self.markers_by_type();
        let features = self
            .activity()
            .and_then(|a| a.feature_names(Source::Expected).ok().map(<[_]>::to_vec))
            .unwrap_or_default();
        let mut review = Review::new(Vec::new(), draft);
        review.evidence = Rc::new(Evidence::new(markers, features));
        self.review = Some(review);
        let overview = self.cluster_overview(li);
        if let Some(r) = self.review.as_mut() {
            r.overview = overview;
        }
        self.visit(0);
        Ok(())
    }

    /// Every cluster of grouping `li` (the one on screen), in visiting order.
    fn cluster_overview(&mut self, li: usize) -> Vec<Overview> {
        let clusters = &self.data.labels[li];
        let ids = clusters.ids.clone();
        let n = ids.len();
        let size = self
            .group_sizes()
            .map_or_else(|| vec![0; n], <[usize]>::to_vec);
        let labels: Vec<Option<String>> = ids
            .iter()
            .map(|id| {
                self.data
                    .round
                    .as_ref()
                    .and_then(|r| r.label(&id.to_string()))
            })
            .collect();
        let per = match self.review_sums() {
            Some((activity, Ok(sums))) => activity.cluster_contrasts(sums).ok().map(|(p, _)| p),
            _ => None,
        };
        let evidence = self.evidence();
        let best: Vec<Option<(String, f32)>> = (0..n)
            .map(|g| {
                let scores = per.as_ref()?.get(g).filter(|s| !s.is_empty())?;
                evidence.best_fit(scores)
            })
            .collect();
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
                disputed: self
                    .data
                    .round
                    .as_ref()
                    .and_then(|r| r.evidence(&ids[g].to_string()))
                    .filter(|e| !e.agrees)
                    .map(|e| e.top.unwrap_or_else(|| super::rounds::UNASSIGNED.into())),
            })
            .collect();
        out.sort_by_key(|o| {
            (
                o.label.is_some(),
                !(o.suggests_change() || o.coarse || o.disputed.is_some()),
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
        let evidence = self.evidence();
        let markers = &evidence.markers;
        let fits = evidence.type_fits(&scores);
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
        let current = self.cluster_label(id);
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
        let rows = evidence.review_rows(&scores, &listed, target.as_deref(), staged.as_deref());
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
    /// rest of the view, in the model's feature order (empty: none).
    fn cluster_scores(&mut self) -> Vec<f32> {
        let Some(f) = self.focus else {
            return Vec::new();
        };
        let scores = match self.review_sums() {
            Some((activity, Ok(sums))) => activity
                .union_contrast(sums, |g| g == f as usize)
                .map(|(v, _)| v),
            Some((_, Err(e))) => Err(e.clone()),
            None => return Vec::new(),
        };
        scores.unwrap_or_else(|e| {
            self.note = Some(e);
            Vec::new()
        })
    }

    fn evidence(&self) -> Rc<Evidence> {
        self.review
            .as_ref()
            .map_or_else(Rc::default, |r| r.evidence.clone())
    }

    /// The activity, and the review's per-group sums for the space and
    /// grouping on screen (taken on first use for each).
    fn review_sums(&mut self) -> Option<(&mut Activity, &Result<GroupSums, String>)> {
        let key = (self.space, self.colour);
        let n = self.levels().len();
        self.activity()?;
        let (_, _, groups, _) = self.groups.as_ref()?;
        let r = self.review.as_mut()?;
        let activity = self.activity.as_mut()?;
        if r.sums.as_ref().is_none_or(|(k, _)| *k != key) {
            let names = &self.data.spaces[key.0].points.names;
            r.sums = Some((key, activity.group_sums(key.0, names, groups, n)));
        }
        Some((activity, &r.sums.as_ref()?.1))
    }

    /// The marker table, as the features listed under each type.
    fn markers_by_type(&self) -> BTreeMap<String, Vec<Box<str>>> {
        let mut out: BTreeMap<String, Vec<Box<str>>> = BTreeMap::new();
        if let Some(m) = self.markers() {
            for (f, &g) in m.by_name.iter() {
                out.entry(m.levels[g as usize].to_string())
                    .or_default()
                    .push(f.clone());
            }
        }
        out
    }

    /// Stage `mark` for the selected row's feature (or clear it with `None`),
    /// say what it did, and move to the next row.
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
                let had = r.draft.cluster(id).marks.remove(&feature).is_some();
                r.row = (r.row + 1).min(r.rows.len() - 1);
                if had {
                    self.note = Some(format!("{feature}: mark cleared"));
                }
                return;
            }
            Some(true) => match &r.target {
                Some(t) if r.evidence.lists(t, &feature) => {
                    self.note = Some(format!("{feature} is already one of {t}'s markers"));
                    return;
                }
                Some(t) => Mark::Include {
                    cell_type: t.clone(),
                },
                None => {
                    self.note = Some("choose a target type first (tab)".into());
                    return;
                }
            },
            // Only a type that lists the feature can drop it.
            Some(false) => match row.marker_of.first() {
                Some(t) => Mark::Exclude {
                    cell_type: t.clone(),
                },
                None => {
                    self.note = Some(format!("{feature} is not a listed marker; nothing to drop"));
                    return;
                }
            },
        };
        self.note = Some(match &m {
            Mark::Include { cell_type } => format!("{feature}: add to {cell_type}'s markers"),
            Mark::Exclude { cell_type } => format!("{feature}: drop from {cell_type}'s markers"),
        });
        r.draft.cluster(id).marks.insert(feature, (m, score));
        r.row = (r.row + 1).min(r.rows.len() - 1);
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

    /// The next cluster after the one visited that has no verdict and is in
    /// no merge, going round; `None` when every cluster is decided.
    pub fn next_undecided(&self) -> Option<usize> {
        let r = self.review.as_ref()?;
        let n = r.overview.len();
        (1..=n).map(|k| (r.at + k) % n).find(|&i| {
            let id = r.overview[i].id;
            r.draft
                .clusters
                .get(&id)
                .is_none_or(|c| c.verdict.is_none())
                && !r.draft.merges.iter().any(|m| m.clusters.contains(&id))
        })
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
