//! Annotation decisions: the prompt, merge marks, sending a decision to
//! lupin and following the rounds it writes.

use super::*;

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
                    self.send(&p.decision);
                } else {
                    p.decision.label = text;
                    p.why = true;
                    p.input.clear();
                }
            }
            KeyCode::Char(c) => p.input.push(c),
            _ => {}
        }
    }

    /// The cluster a decision is about: the focused cluster when colouring by
    /// cluster, else the one last clicked.
    pub(super) fn target(&self) -> Option<i64> {
        self.scene.focused_cluster().or(self.clicked)
    }

    pub(super) fn toggle_mark(&mut self) {
        let Some(c) = self.target() else {
            self.message = Some("focus or click a cluster to mark it".into());
            return;
        };
        if let Some(i) = self.marked.iter().position(|&m| m == c) {
            self.marked.remove(i);
        } else {
            self.marked.push(c);
        }
    }

    /// Start typing a decision of kind `action`.
    pub(super) fn begin(&mut self, action: Action) {
        if self.watcher.is_none() {
            self.watcher = Watcher::find(&self.from, self.decisions.as_deref());
        }
        if self.watcher.is_none() {
            self.message = Some(
                "no `lupin relabel --watch` found beside this round; start one, or pass --decisions"
                    .into(),
            );
            return;
        }
        let mut decision = Decision {
            action,
            clusters: Vec::new(),
            features: Vec::new(),
            label: String::new(),
            rationale: String::new(),
            evidence: Vec::new(),
        };
        let mut input = String::new();
        match action {
            Action::Merge => {
                if self.marked.len() < 2 {
                    self.message = Some("mark two or more clusters with v first".into());
                    return;
                }
                decision.clusters = self.marked.clone();
            }
            Action::Label | Action::Keep => {
                let Some(c) = self.target() else {
                    self.message = Some("focus or click a cluster first".into());
                    return;
                };
                decision.clusters = vec![c];
                let (label, top) = self.scene.cluster_call(c);
                if let Some((call, support)) = top {
                    decision.evidence.push(serde_json::json!({
                        "kind": "marker", "term": call.clone(), "stat": "support", "value": support,
                    }));
                    input = call;
                }
                if action == Action::Keep {
                    let Some(l) = label else {
                        self.message = Some(format!("C{c} has no current call to keep"));
                        return;
                    };
                    decision.label = l;
                }
            }
            Action::MarkersAdd | Action::MarkersDrop => {
                let Some(Pick::One(f)) = self.scene.pick.clone() else {
                    self.message = Some("show a feature first (n, g or /)".into());
                    return;
                };
                if let Some(v) = self.scene.suggestion_score(&f) {
                    decision.evidence.push(serde_json::json!({
                        "kind": "marker", "term": f.as_ref(), "stat": "expected_lfc", "value": v,
                    }));
                }
                decision.features = vec![f];
                input = self
                    .scene
                    .focused_name()
                    .map(|n| n.to_string())
                    .unwrap_or_default();
            }
        }
        // Keep reuses the cluster's current label; everything else asks for one.
        let why = action == Action::Keep;
        self.prompt = Some(Prompt {
            decision,
            why,
            input: if why { String::new() } else { input },
            known: self.scene.known_labels(),
        });
    }

    pub(super) fn send(&mut self, d: &Decision) {
        let Some(w) = &self.watcher else { return };
        let line = d.to_json(&w.round_ref(&self.from));
        self.message = Some(match w.append(&line) {
            Ok(()) => {
                if d.action == Action::Merge {
                    self.marked.clear();
                }
                format!(
                    "sent: {} {} · lupin will write the next round",
                    d.action.name(),
                    d.label
                )
            }
            Err(e) => format!("could not write the decision: {e}"),
        });
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
    decision: Decision,
    /// Typing the rationale (after the label).
    why: bool,
    input: String,
    /// Label completions offered for the current input.
    known: Vec<Box<str>>,
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
