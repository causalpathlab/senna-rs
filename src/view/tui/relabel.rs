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
        } else if let Err(e) = {
            // Relabel works on the map.
            self.scene.chart = None;
            self.scene.enter_review()
        } {
            self.message = Some(e);
        }
        self.settle();
    }

    /// Keys of relabel mode. Returns whether the key was one of them.
    pub(super) fn review_key(&mut self, k: KeyEvent) -> bool {
        if self.scene.review.is_none() {
            return false;
        }
        // Relabel works on the cluster colouring of this cell view: keys that
        // would change it, or the round, wait — in merge mode too.
        if matches!(
            k.code,
            KeyCode::Char('c' | ',' | '.' | 'v') | KeyCode::BackTab
        ) {
            self.message = Some("leave relabel mode first (R)".into());
            return true;
        }
        // Back to the cluster: drop a feature or suggestions on show, keep
        // the focus.
        if k.code == KeyCode::Char('x') {
            self.info = None;
            self.scene.clear_suggestions();
            self.scene.clear_pick();
            self.restart();
            return true;
        }
        let Some(r) = self.scene.review.as_mut() else {
            return false;
        };
        if r.merge.is_some() {
            return self.merge_key(k);
        }
        let n_rows = r.rows.len();
        let n_clusters = r.overview.len().max(1);
        match k.code {
            KeyCode::Char(']') | KeyCode::Right => {
                let next = (r.at + 1) % n_clusters;
                self.change(|s| s.visit(next));
            }
            KeyCode::Char('[') | KeyCode::Left => {
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
            KeyCode::Char('S') => self.confirm_submit(),
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
            // The map marks the cluster under the cursor: draw it again.
            KeyCode::Down => {
                m.cursor = (m.cursor + 1).min(n.saturating_sub(1));
                self.restart();
            }
            KeyCode::Up => {
                m.cursor = m.cursor.saturating_sub(1);
                self.restart();
            }
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
        let current = self.scene.cluster_label(id);
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
        let done = match self.scene.next_undecided() {
            Some(i) => {
                self.change(|s| s.visit(i));
                "staged · on to the next undecided cluster".to_string()
            }
            None => {
                if merge {
                    self.restart();
                }
                "staged · every cluster is decided · S hands them to lupin".to_string()
            }
        };
        self.message = Some(done);
    }

    /// Keep the scores on screen current with the staged marker edits: take
    /// lupin's answer when it comes, and start it again on newer edits (the
    /// running one is killed). With no edits the round's own scores return.
    /// Returns whether anything on screen changed.
    pub(super) fn keep_scores_current(&mut self) -> bool {
        let Some(r) = self.scene.review.as_mut() else {
            self.rescoring = None;
            return false;
        };
        let want = r.draft.marker_edits();
        if let Some(job) = &self.rescoring {
            let live = r.live.as_mut().filter(|l| l.edits == want);
            match live {
                Some(live) => {
                    let Some(reply) = job.poll() else {
                        return false;
                    };
                    self.rescoring = None;
                    let scores = reply.and_then(|v| crate::view::relabel::parse_scores(&v));
                    self.message = Some(match &scores {
                        Ok(_) => "lupin rescored the edited cell types".into(),
                        Err(e) => format!("lupin could not rescore: {e}"),
                    });
                    live.scores = Some(scores);
                    return true;
                }
                // The edits moved on: stop it.
                None => self.rescoring = None,
            }
        }
        if r.live.as_ref().map_or(want.is_empty(), |l| l.edits == want) {
            return false;
        }
        if want.is_empty() {
            r.live = None;
            return true;
        }
        let round = self
            .from
            .canonicalize()
            .unwrap_or_else(|_| self.from.clone());
        let lines: Vec<_> = r
            .draft
            .marker_decisions()
            .iter()
            .map(|d| d.to_json(&round.to_string_lossy()))
            .collect();
        let scores = match Rescore::spawn(&self.lupin, &round, &lines) {
            Ok(job) => {
                self.rescoring = Some(job);
                None
            }
            Err(e) => Some(Err(e)),
        };
        r.live = Some(crate::view::relabel::Live {
            edits: want,
            scores,
        });
        true
    }

    /// Send the whole draft to lupin: a preview, or the next round.
    /// Ask before `S` hands lupin the draft: it makes a new round.
    fn confirm_submit(&mut self) {
        let Some(r) = self.scene.review.as_ref() else {
            return;
        };
        if r.draft.is_empty() {
            self.message = Some("nothing staged yet".into());
            return;
        }
        let lines = r.draft.decisions().iter().map(Decision::summary).collect();
        self.modal = Some(Modal::Submit(lines));
    }

    pub(super) fn send_draft(&mut self, mode: Mode) {
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
