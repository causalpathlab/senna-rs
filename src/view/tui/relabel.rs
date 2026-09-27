//! Relabel mode's keys: visit clusters, mark features, stage verdicts and
//! merges, preview with lupin, and submit the draft as one round.

use super::*;
use crate::view::decide::{relabel, Mode};
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
        let n_rows = r.rows.len();
        let n_clusters = r.order.len();
        match k.code {
            KeyCode::Char(']') => {
                let next = (r.at + 1) % n_clusters.max(1);
                self.change(|s| s.visit(next));
            }
            KeyCode::Char('[') => {
                let prev = (r.at + n_clusters.max(1) - 1) % n_clusters.max(1);
                self.change(|s| s.visit(prev));
            }
            KeyCode::Down => r.row = (r.row + 1).min(n_rows.saturating_sub(1)),
            KeyCode::Up => r.row = r.row.saturating_sub(1),
            KeyCode::Enter => self.change(Scene::show_row),
            KeyCode::Char('+' | '=') => self.change(|s| s.mark_row(Some(true))),
            KeyCode::Char('-' | '_') => self.change(|s| s.mark_row(Some(false))),
            KeyCode::Char(' ') => self.change(|s| s.mark_row(None)),
            KeyCode::Char('a') => self.change(Scene::accept_proposals),
            KeyCode::Tab => self.change(Scene::next_target),
            KeyCode::Char('L') => self.begin_staged(Action::Label),
            KeyCode::Char('K') => self.begin_staged(Action::Keep),
            KeyCode::Char('M') => self.begin_staged(Action::Merge),
            KeyCode::Char('v') => {
                let id = r.cluster();
                if let Some(i) = r.marked.iter().position(|&m| m == id) {
                    r.marked.remove(i);
                } else {
                    r.marked.push(id);
                }
            }
            KeyCode::Char('p') => self.send_draft(Mode::Preview),
            KeyCode::Char('S') => self.send_draft(Mode::Next),
            KeyCode::Esc | KeyCode::Char('R') => self.toggle_review(),
            _ => return false,
        }
        true
    }

    /// Ask for a label (pre-filled with the target type) and a rationale
    /// (pre-filled from what was staged), then stage the verdict.
    fn begin_staged(&mut self, action: Action) {
        let Some(r) = self.scene.review.as_ref() else {
            return;
        };
        let id = r.cluster();
        let clusters = if action == Action::Merge {
            let mut c = r.marked.clone();
            if !c.contains(&id) {
                c.push(id);
            }
            if c.len() < 2 {
                self.message = Some("mark the clusters to merge with v first".into());
                return;
            }
            c
        } else {
            vec![id]
        };
        let target = r.target.clone().unwrap_or_default();
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
        let prefill = self.scene.drafted_rationale();
        self.prompt = Some(Prompt {
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
        });
    }

    /// Put a typed decision into the draft.
    pub(super) fn stage(&mut self, d: Decision) {
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
        self.restart();
    }

    /// Send the whole draft to lupin: a preview, or the next round.
    fn send_draft(&mut self, mode: Mode) {
        if self.relabeling.is_some() {
            self.message = Some("lupin is still busy with the last request".into());
            return;
        }
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
        let n = lines.len();
        let lupin = self.lupin.clone();
        let (tx, rx) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            let _ = tx.send(relabel(&lupin, &round, &lines, mode));
        });
        self.relabeling = Some(Relabeling {
            job: match mode {
                Mode::Next => RelabelJob::Submit,
                Mode::Preview => RelabelJob::Preview,
            },
            done: rx,
        });
        self.message = Some(match mode {
            Mode::Next => format!("sending {n} decisions to lupin as one round…"),
            Mode::Preview => format!("asking lupin to preview {n} decisions…"),
        });
    }
}
