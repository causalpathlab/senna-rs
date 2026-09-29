//! What is being typed in the status line: at most one at a time.

use super::annotate::path_options;
use super::help::Context;
use super::*;

pub(super) enum Modal {
    /// Feature search: the query and its current matches.
    Search(String, Vec<Box<str>>),
    /// A decision: its label, then its rationale.
    Prompt(Prompt),
    /// The markers file for `lupin annotate`, and the entries its last part
    /// could complete to (read once per edit, not per draw).
    MarkersFile(String, Vec<(String, bool)>),
    /// Submitting the relabel draft: one line per decision it hands lupin.
    Submit(Vec<String>),
}

impl Modal {
    pub(super) fn markers_file(input: String) -> Self {
        let options = path_options(&input);
        Modal::MarkersFile(input, options)
    }

    pub(super) fn context(&self) -> Context {
        match self {
            Modal::Search(..) => Context::Search,
            Modal::Prompt(_) => Context::Prompt,
            Modal::MarkersFile(..) => Context::MarkersFile,
            Modal::Submit(_) => Context::Submit,
        }
    }

    /// The status line while typing; none when a popup asks instead.
    pub(super) fn line(&self) -> Option<String> {
        Some(match self {
            Modal::Search(q, hits) => {
                let shown: Vec<&str> = hits.iter().take(6).map(AsRef::as_ref).collect();
                format!("/{q}   {}", shown.join("  "))
            }
            Modal::Prompt(p) => p.line(),
            Modal::Submit(_) => return None,
            Modal::MarkersFile(input, options) => {
                let hint = if options.len() > 1 {
                    let names: Vec<&str> =
                        options.iter().take(4).map(|(n, _)| n.as_str()).collect();
                    format!("   tab: {}", names.join(" · "))
                } else {
                    String::new()
                };
                format!("markers file for lupin annotate: {input}▏{hint}")
            }
        })
    }
}

impl App {
    /// A key while typing. Returns whether anything changed.
    pub(super) fn modal_key(&mut self, k: KeyEvent) -> bool {
        match self.modal {
            Some(Modal::Search(..)) => self.search_key(k),
            Some(Modal::Prompt(_)) => self.prompt_key(k),
            Some(Modal::MarkersFile(..)) => self.markers_key(k),
            Some(Modal::Submit(_)) => {
                self.modal = None;
                if matches!(k.code, KeyCode::Char('S') | KeyCode::Enter) {
                    self.send_draft(Mode::Next);
                } else {
                    self.message = Some("not submitted · the draft is kept".into());
                }
                true
            }
            None => false,
        }
    }
}
