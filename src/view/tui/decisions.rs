//! The decision prompt, and what comes back from lupin: rounds it wrote for
//! this view, and rounds a watcher wrote.

use super::*;
use crate::view::decide::Reply;
use crate::view::rounds::label_key;
use crate::view::LabelKind;

impl App {
    /// A key while typing a decision. Returns whether anything changed.
    pub(super) fn prompt_key(&mut self, k: KeyEvent) -> bool {
        let Some(Modal::Prompt(p)) = self.modal.as_mut() else {
            return false;
        };
        if k.code != KeyCode::Tab {
            p.cycle = None;
        }
        match k.code {
            KeyCode::Esc => {
                self.modal = None;
                self.message = Some("decision cancelled".into());
            }
            KeyCode::Tab if !p.why => {
                let Some(k) = p.next_completion() else {
                    return false;
                };
                p.input = k;
            }
            KeyCode::Enter => {
                let text = p.input.trim().to_string();
                if text.is_empty() {
                    return false;
                }
                if p.why {
                    p.decision.rationale = text;
                    if let Some(Modal::Prompt(p)) = self.modal.take() {
                        self.stage(p.decision);
                    }
                } else {
                    p.decision.label = text;
                    p.why = true;
                    p.input = std::mem::take(&mut p.why_prefill);
                }
            }
            _ if crate::tui::edit_line(&mut p.input, &k) => {}
            _ => return false,
        }
        true
    }

    /// Take lupin's answer, if it has come: open the round it wrote, or say
    /// why it refused (opening the latest round when this one was stale).
    /// Returns whether it did anything.
    pub(super) fn finish_relabel(&mut self) -> bool {
        let Some(reply) = self
            .relabeling
            .as_ref()
            .and_then(|r| r.done.poll("the lupin call stopped without an answer"))
        else {
            return false;
        };
        let r = self.relabeling.take().expect("checked above");
        match reply {
            Ok(Reply::Round(path)) if matches!(r.job, RelabelJob::Annotate) => {
                self.open_round(&path, "annotated");
                self.scene.colour_by(LabelKind::Annotation);
                let name = files::name(&path);
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
                if matches!(r.job, RelabelJob::Draft(Mode::Next)) {
                    if let Some(review) = self.scene.review.take() {
                        review.draft.discard();
                    }
                }
                self.open_round(&path, "new round");
                // Show what the round changed: the changed cells coloured,
                // and what was applied, in the sidebar.
                let name = files::name(&path);
                let mut lines = vec![format!("lupin wrote {name}"), String::new()];
                lines.push("applied:".into());
                lines.extend(r.sent.iter().map(|l| format!("  {l}")));
                lines.push(String::new());
                lines.extend(self.scene.show_changes());
                lines.push(String::new());
                lines.push("x closes this · , goes back to the round before".into());
                self.info = Some(lines);
                let secs = r.started.elapsed().as_secs_f32();
                self.message = Some(format!(
                    "lupin applied {} decisions in {secs:.1} s · coloured by what changed",
                    r.sent.len()
                ));
                self.pop(format!("✓ lupin wrote {name}"));
                self.restart();
            }
            Ok(Reply::Preview(v)) => {
                if let Some(review) = self.scene.review.as_mut() {
                    let alpha = crate::view::rounds::fdr_alpha(self.scene.data.round.as_ref());
                    review.preview = Some(crate::view::relabel::preview_lines(&v, alpha));
                }
                self.message = Some("preview from lupin in the sidebar".into());
                self.pop("✓ lupin answered: preview in the sidebar".into());
            }
            Ok(Reply::Refused { reason, latest }) => {
                if let Some(l) = latest.filter(|l| !same_file(l, &self.from)) {
                    // A newer round exists: say so, and leave it to the user
                    // to open it (`.`) and decide again there.
                    let name = files::name(&l);
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
    /// The prefix a run of Tabs started from and the candidate it is on, so
    /// the list does not narrow to the label just filled in.
    pub(super) cycle: Option<(String, usize)>,
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

    /// Known labels the typed label could complete to: keyed as lupin keys
    /// type names, the prefix matched without regard to case so a lower-case
    /// start still finds them.
    fn completions_of<'a>(&'a self, prefix: &str) -> impl Iterator<Item = &'a str> {
        let typed = fold(prefix);
        self.known
            .iter()
            .filter(move |k| fold(k).starts_with(&typed))
            .map(AsRef::as_ref)
    }

    fn completions(&self) -> impl Iterator<Item = &str> {
        self.completions_of(&self.input)
    }

    /// The completion a Tab moves to. A run of Tabs keeps the prefix it began
    /// with and steps through every match, labels that differ only in case
    /// included; the first Tab goes on from the label already typed.
    fn next_completion(&mut self) -> Option<String> {
        let (prefix, at) = match &self.cycle {
            Some((p, i)) => (p.clone(), Some(*i)),
            None => (self.input.clone(), None),
        };
        let all: Vec<&str> = self.completions_of(&prefix).collect();
        if all.is_empty() {
            return None;
        }
        let next = match at {
            Some(i) => (i + 1) % all.len(),
            None => all
                .iter()
                .position(|k| *k == self.input)
                .map_or(0, |i| (i + 1) % all.len()),
        };
        let label = all[next].to_string();
        self.cycle = Some((prefix, next));
        Some(label)
    }

    pub(super) fn line(&self) -> String {
        if self.why {
            format!(
                "{} · why? {}▏  (enter sends · esc cancels)",
                self.subject(),
                self.input
            )
        } else {
            let hint: Vec<&str> = self.completions().take(4).collect();
            format!(
                "{} as: {}▏  tab: {}",
                self.subject(),
                self.input,
                hint.join(" · ")
            )
        }
    }
}

/// A label as completion compares it: keyed, then case-folded.
fn fold(s: &str) -> String {
    label_key(s).to_lowercase()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tab_steps_through_labels_that_differ_only_in_case() {
        let mut p = Prompt {
            decision: Decision {
                action: Action::Label,
                clusters: vec![1],
                features: Vec::new(),
                label: String::new(),
                rationale: String::new(),
                evidence: Vec::new(),
            },
            why: false,
            input: "t".into(),
            known: ["T cell", "T cell memory", "t cell", "B cell"]
                .map(Box::from)
                .to_vec(),
            why_prefill: String::new(),
            cycle: None,
        };
        let mut seen = Vec::new();
        for _ in 0..4 {
            p.input = p.next_completion().unwrap();
            seen.push(p.input.clone());
        }
        assert_eq!(seen, ["T cell", "T cell memory", "t cell", "T cell"]);
        p.cycle = None;
        p.input = "zzz".into();
        assert_eq!(p.next_completion(), None);
    }
}
