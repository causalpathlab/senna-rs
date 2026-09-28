//! The decision prompt, and what comes back from lupin: rounds it wrote for
//! this view, and rounds a watcher wrote.

use super::*;
use crate::view::decide::Reply;
use crate::view::LabelKind;

/// How long a popup stays.
const TOAST_FOR: std::time::Duration = std::time::Duration::from_millis(2500);

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
            Ok(Reply::Round(path)) if matches!(r.job, RelabelJob::Annotate) => {
                self.open_round(&path, "annotated");
                self.scene.colour_by(LabelKind::Annotation);
                let name = path
                    .file_name()
                    .map_or_else(String::new, |n| n.to_string_lossy().into_owned());
                self.info = Some(vec![
                    format!("lupin annotated this run: {name}"),
                    String::new(),
                    "coloured by annotation; click a cell for its cluster's calls".into(),
                    "R relabels cluster by cluster".into(),
                    String::new(),
                    "x closes this".into(),
                ]);
                self.message = Some(format!(
                    "lupin annotated in {:.0} s",
                    r.started.elapsed().as_secs_f32()
                ));
                self.pop(format!("✓ lupin wrote {name}"));
                self.restart();
            }
            Ok(Reply::Refused { reason, .. }) if matches!(r.job, RelabelJob::Annotate) => {
                self.message = Some(format!("lupin could not annotate: {reason}"));
            }
            Ok(Reply::Round(path)) => {
                if matches!(r.job, RelabelJob::Submit) {
                    if let Some(review) = self.scene.review.take() {
                        review.draft.discard();
                    }
                }
                self.open_round(&path, "new round");
                // Show what the round changed: the changed cells coloured,
                // and what was applied, in the sidebar.
                let name = path
                    .file_name()
                    .map_or_else(String::new, |n| n.to_string_lossy().into_owned());
                let mut lines = vec![
                    format!("lupin wrote {name}"),
                    String::new(),
                    "applied:".into(),
                ];
                lines.extend(r.sent.iter().map(|l| format!("  {l}")));
                lines.push(String::new());
                lines.extend(self.scene.show_changes());
                lines.push(String::new());
                lines.push("x closes this · , goes back to the round before".into());
                self.info = Some(lines);
                self.message = Some(format!(
                    "lupin applied {} decisions in {:.1} s · coloured by what changed",
                    r.sent.len(),
                    r.started.elapsed().as_secs_f32()
                ));
                self.pop(format!("✓ lupin wrote {name}"));
                self.restart();
            }
            Ok(Reply::Preview(v)) => {
                if let Some(review) = self.scene.review.as_mut() {
                    review.preview = Some(crate::view::relabel::preview_lines(&v));
                }
                self.message = Some("preview from lupin in the sidebar".into());
                self.pop("✓ lupin answered: preview in the sidebar".into());
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

    /// Show `text` in a short popup over the map.
    pub(super) fn pop(&mut self, text: String) {
        self.toast = Some((text, std::time::Instant::now() + TOAST_FOR));
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
