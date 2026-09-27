//! The decision prompt, and what comes back from lupin: rounds it wrote for
//! this view, and rounds a watcher wrote.

use super::*;
use crate::view::decide::Reply;

impl App {
    pub(super) fn prompt_key(&mut self, k: KeyEvent) {
        let Some(p) = self.prompt.as_mut() else {
            return;
        };
        match k.code {
            KeyCode::Esc => {
                self.prompt = None;
                self.message = Some("decision cancelled".into());
            }
            KeyCode::Backspace => {
                p.input.pop();
            }
            KeyCode::Tab if !p.why => {
                let low = p.input.to_lowercase();
                if let Some(k) = p.known.iter().find(|k| k.to_lowercase().starts_with(&low)) {
                    p.input = k.to_string();
                }
            }
            KeyCode::Enter => {
                let text = p.input.trim().to_string();
                if text.is_empty() {
                    return;
                }
                if p.why {
                    p.decision.rationale = text;
                    let p = self.prompt.take().expect("checked above");
                    self.stage(p.decision);
                } else {
                    p.decision.label = text;
                    p.why = true;
                    p.input = std::mem::take(&mut p.why_prefill);
                }
            }
            KeyCode::Char(c) => p.input.push(c),
            _ => {}
        }
    }

    /// Take lupin's answer, if it has come: open the round it wrote, or say
    /// why it refused (opening the latest round when this one was stale).
    /// Returns whether it did anything.
    pub(super) fn finish_relabel(&mut self) -> bool {
        let Some(r) = &self.relabeling else {
            return false;
        };
        let reply = match r.done.try_recv() {
            Ok(reply) => reply,
            Err(std::sync::mpsc::TryRecvError::Empty) => return false,
            Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                Err("the lupin call stopped without an answer".into())
            }
        };
        let r = self.relabeling.take().expect("checked above");
        match reply {
            Ok(Reply::Round(path)) => {
                if matches!(r.job, RelabelJob::Submit) {
                    if let Some(review) = self.scene.review.take() {
                        review.draft.discard();
                    }
                }
                self.open_round(&path, "new round");
            }
            Ok(Reply::Preview(v)) => {
                if let Some(review) = self.scene.review.as_mut() {
                    review.preview = Some(crate::view::relabel::preview_lines(&v));
                }
                self.message = Some("preview from lupin in the sidebar".into());
            }
            Ok(Reply::Refused { reason, latest }) => {
                if let Some(l) = latest.filter(|l| !same_file(l, &self.from)) {
                    // A newer round exists: say so, and leave it to the user
                    // to open it (`.`) and decide again there.
                    let name = l
                        .file_name()
                        .map_or_else(String::new, |n| n.to_string_lossy().into_owned());
                    self.message = Some(format!(
                        "lupin refused: this is not the latest round ({name} is) · . opens it; decide again there"
                    ));
                } else {
                    self.message = Some(format!("lupin refused: {reason}"));
                }
            }
            Err(e) => self.message = Some(e),
        }
        true
    }

    /// Open the watcher's latest round when it moves on, and show a refused
    /// batch's reason. Returns whether a round was opened.
    pub(super) fn follow_watcher(&mut self) -> bool {
        let Some(w) = self.watcher.as_mut() else {
            return false;
        };
        if !w.refresh() {
            return false;
        }
        if let Some(e) = &w.error {
            self.message = Some(format!("lupin refused the last decisions: {e}"));
        }
        let latest = w.latest.clone();
        match latest {
            Some(l) if !same_file(&l, &self.from) => {
                self.open_round(&l, "new round");
                true
            }
            _ => false,
        }
    }
}

/// A decision being typed in the status line: first the label (unless the
/// action keeps the current one), then the rationale.
pub(super) struct Prompt {
    pub(super) decision: Decision,
    /// Typing the rationale (after the label).
    pub(super) why: bool,
    pub(super) input: String,
    /// Label completions offered for the current input.
    pub(super) known: Vec<Box<str>>,
    /// What the rationale starts as, once the label is in.
    pub(super) why_prefill: String,
}

impl Prompt {
    pub(super) fn subject(&self) -> String {
        let d = &self.decision;
        let ids: Vec<String> = d.clusters.iter().map(|c| format!("C{c}")).collect();
        match d.action {
            Action::Label => format!("label {}", ids.join(" ")),
            Action::Merge => format!("merge {}", ids.join(" ")),
            Action::Keep => format!("keep {} as {}", ids.join(" "), d.label),
            Action::MarkersAdd => format!("add {} to markers of", d.features.join(" ")),
            Action::MarkersDrop => format!("drop {} from markers of", d.features.join(" ")),
        }
    }

    pub(super) fn line(&self) -> String {
        if self.why {
            format!(
                "{} · why? {}▏  (enter sends · esc cancels)",
                self.subject(),
                self.input
            )
        } else {
            let hint: Vec<&str> = self
                .known
                .iter()
                .filter(|k| k.to_lowercase().starts_with(&self.input.to_lowercase()))
                .take(4)
                .map(AsRef::as_ref)
                .collect();
            format!(
                "{} as: {}▏  tab: {}",
                self.subject(),
                self.input,
                hint.join(" · ")
            )
        }
    }
}
