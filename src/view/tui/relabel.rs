//! Relabel mode's keys: visit clusters, mark features, stage verdicts and
//! merges, preview with lupin, and submit the draft as one round.

use super::*;
use crate::view::decide::relabel;
use crate::view::review::Verdict;

impl App {
    /// `R`: enter relabel mode, or leave it (the draft is kept on disk).
    pub(super) fn toggle_review(&mut self) {
        if self.scene.review.is_some() {
            self.scene.leave_review();
            self.message = Some("left relabel mode · draft kept".into());
        } else if let Err(e) = self.scene.enter_review() {
            self.message = Some(e);
        }
        self.settle();
    }

    /// Keys of relabel mode. Returns whether the key was one of them.
    pub(super) fn review_key(&mut self, k: KeyEvent) -> bool {
        let Some(r) = self.scene.review.as_mut() else {
            return false;
        };
        if r.merge.is_some() {
            return self.merge_key(k);
        }
        let n_rows = r.rows.len();
        let n_clusters = r.overview.len().max(1);
        match k.code {
            KeyCode::Char(']') => {
                let next = (r.at + 1) % n_clusters;
                self.change(|s| s.visit(next));
            }
            KeyCode::Char('[') => {
                let prev = (r.at + n_clusters - 1) % n_clusters;
                self.change(|s| s.visit(prev));
            }
            KeyCode::Down => r.row = (r.row + 1).min(n_rows.saturating_sub(1)),
            KeyCode::Up => r.row = r.row.saturating_sub(1),
            KeyCode::Enter => self.change(Scene::show_row),
            // These change the draft and the panels, not the map.
            KeyCode::Char('+' | '=') => self.change_text(|s| s.mark_row(Some(true))),
            KeyCode::Char('-' | '_') => self.change_text(|s| s.mark_row(Some(false))),
            KeyCode::Char(' ') => self.change_text(|s| s.mark_row(None)),
            KeyCode::Char('a') => self.change_text(Scene::accept_proposals),
            KeyCode::Tab => self.change_text(Scene::next_target),
            KeyCode::Char('L') => self.begin_staged(Action::Label),
            KeyCode::Char('K') => self.begin_staged(Action::Keep),
            KeyCode::Char('M') => self.change(Scene::begin_merge),
            KeyCode::Char('p') => self.send_draft(Mode::Preview),
            KeyCode::Char('S') => self.send_draft(Mode::Next),
            KeyCode::Esc | KeyCode::Char('R') => self.toggle_review(),
            _ => return false,
        }
        true
    }

    /// Keys of merge mode: move through the cluster list, choose clusters,
    /// then name the merged cluster.
    fn merge_key(&mut self, k: KeyEvent) -> bool {
        let Some(r) = self.scene.review.as_mut() else {
            return false;
        };
        let n = r.overview.len();
        let Some(m) = r.merge.as_mut() else {
            return false;
        };
        match k.code {
            KeyCode::Down => m.cursor = (m.cursor + 1).min(n.saturating_sub(1)),
            KeyCode::Up => m.cursor = m.cursor.saturating_sub(1),
            KeyCode::Char(' ') => self.change(Scene::toggle_merge_cursor),
            KeyCode::Enter => self.begin_staged(Action::Merge),
            KeyCode::Esc | KeyCode::Char('M') => {
                r.merge = None;
                self.message = Some("merge cancelled".into());
                self.restart();
            }
            _ => return false,
        }
        true
    }

    /// Ask for a label (pre-filled with the target type, or for a merge the
    /// type the chosen clusters fit best) and a rationale (pre-filled from
    /// what was staged), then stage the decision.
    fn begin_staged(&mut self, action: Action) {
        let Some(r) = self.scene.review.as_ref() else {
            return;
        };
        let id = r.cluster();
        let (clusters, target) = if action == Action::Merge {
            let Some(m) = r.merge.as_ref() else { return };
            if m.chosen.len() < 2 {
                self.message = Some("choose at least two clusters (↑↓, space)".into());
                return;
            }
            (
                m.chosen.iter().copied().collect::<Vec<_>>(),
                m.best.as_ref().map(|b| b.0.clone()).unwrap_or_default(),
            )
        } else {
            (vec![id], r.target.clone().unwrap_or_default())
        };
        let current = self.scene.cluster_call(id).0;
        let label = if action == Action::Keep {
            match current {
                Some(l) => l,
                None => {
                    self.message = Some(format!("C{id} has no call to keep"));
                    return;
                }
            }
        } else {
            String::new()
        };
        let why = action == Action::Keep;
        let prefill = if action == Action::Merge {
            let ids: Vec<String> = clusters.iter().map(|c| format!("C{c}")).collect();
            let fit = r
                .merge
                .as_ref()
                .and_then(|m| m.best.as_ref())
                .map(|(t, v)| format!("; together their markers fit {t} ({v:+.2})"))
                .unwrap_or_default();
            format!("{} share one program{fit}", ids.join(" "))
        } else {
            self.scene.drafted_rationale()
        };
        self.modal = Some(Modal::Prompt(Prompt {
            decision: Decision {
                action,
                clusters,
                features: Vec::new(),
                label,
                rationale: String::new(),
                evidence: Vec::new(),
            },
            why,
            input: if why { prefill.clone() } else { target },
            known: self.scene.known_labels(),
            why_prefill: prefill,
        }));
    }

    /// Put a typed decision into the draft.
    pub(super) fn stage(&mut self, d: Decision) {
        // Only a merge changes the map (its clusters stop being marked).
        let merge = d.action == Action::Merge;
        match d.action {
            Action::Merge => self.scene.stage_merge(d.label, d.rationale),
            Action::Keep => self.scene.stage_verdict(Verdict::Keep {
                label: d.label,
                rationale: d.rationale,
            }),
            _ => self.scene.stage_verdict(Verdict::Label {
                label: d.label,
                rationale: d.rationale,
            }),
        }
        if let Some(r) = self.scene.review.as_ref() {
            let _ = r.draft.save();
        }
        self.message = Some("staged · ] next cluster · S submits everything".into());
        if merge {
            self.restart();
        }
    }

    /// Send the whole draft to lupin: a preview, or the next round.
    fn send_draft(&mut self, mode: Mode) {
        let Some(r) = self.scene.review.as_ref() else {
            return;
        };
        if r.draft.is_empty() {
            self.message = Some("nothing staged yet".into());
            return;
        }
        let _ = r.draft.save();
        let round = self
            .from
            .canonicalize()
            .unwrap_or_else(|_| self.from.clone());
        let lines = r.draft.lines(&round.to_string_lossy());
        let sent: Vec<String> = r.draft.decisions().iter().map(Decision::summary).collect();
        let lupin = self.lupin.clone();
        let done = Pending::spawn(move || relabel(&lupin, &round, &lines, mode));
        self.relabeling = Some(Relabeling {
            job: RelabelJob::Draft(mode),
            started: std::time::Instant::now(),
            sent,
            progress: Default::default(),
            done,
        });
    }
}
