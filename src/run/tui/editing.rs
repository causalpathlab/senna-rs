//! Typing a line into a field, and the file browser that fills one.

use super::{batches, batches::Batch, data::Pick, header_folder, App, Editor, Target};
use legume_tui::browse::{Browser, Outcome};
use ratatui::crossterm::event::{KeyCode, KeyEvent};

impl App {
    // ───────────── typing a line ─────────────

    pub(super) fn edit(&mut self, target: Target, text: String) {
        self.editor = Some(Editor { target, text });
    }

    pub(super) fn editor_key(&mut self, k: KeyEvent) {
        let Some(e) = self.editor.as_mut() else {
            return;
        };
        match k.code {
            KeyCode::Esc => {
                if e.target == Target::Filter {
                    self.filter.clear();
                    self.field_row = 0;
                }
                self.editor = None;
            }
            KeyCode::Enter => {
                let Editor { target, text } = self.editor.take().unwrap();
                match target {
                    Target::Out(i) => {
                        let t = text.trim();
                        let r = &mut self.rows[i];
                        if t.is_empty() {
                            // Back to the header's.
                            r.typed = false;
                            self.refresh_outs();
                        } else {
                            r.out = t.to_string();
                            r.typed = true;
                        }
                    }
                    Target::Header => self.set_header(&text),
                    Target::Field(..) | Target::Step(..) | Target::Shared(_) => {
                        self.set_text(&target, &text)
                    }
                    Target::Filter => {}
                    Target::BatchName(i) => {
                        let name = text.trim();
                        // No name, or senna's own: back to senna's rule.
                        self.pairs[i].batch = if name.is_empty()
                            || (self.pairs[i].tags.is_none()
                                && name == batches::own_name(&self.pairs[i].data))
                        {
                            Batch::Own
                        } else {
                            Batch::Named(name.to_string())
                        };
                    }
                    Target::Label(i, l) => {
                        if let Batch::Labels {
                            counts, renamed, ..
                        } = &mut self.pairs[i].batch
                        {
                            let label = counts[l].0.clone();
                            let name = text.trim();
                            if name.is_empty() || name == label {
                                renamed.remove(&label);
                            } else {
                                renamed.insert(label, name.to_string());
                            }
                        }
                    }
                }
            }
            // Type, backspace, ctrl+u clears; the filter narrows the flags
            // as it is typed.
            _ if crate::tui::edit_line(&mut e.text, &k) && e.target == Target::Filter => {
                self.filter.clone_from(&e.text);
                self.field_row = 0;
            }
            _ => {}
        }
    }

    // ───────────── the file browser ─────────────

    pub(super) fn browse(&mut self, want: Pick) {
        self.browser = Some(Browser::open(self.browse_dir.clone(), want, None));
    }

    pub(super) fn browser_key(&mut self, k: KeyEvent) {
        let Some(b) = self.browser.as_mut() else {
            return;
        };
        // Chosen files, or none when cancelled; either way it closes.
        let chosen = match b.key(k) {
            Outcome::Ignored | Outcome::Moved => return,
            Outcome::Cancelled => None,
            Outcome::Chosen(c) => Some(c.files()),
        };
        let Some(b) = self.browser.take() else { return };
        let target = self.browsing.take();
        // Browsing for data remembers the folder, chosen from or not.
        if matches!(b.want, Pick::Data | Pick::Batch) {
            self.browse_dir = b.dir;
        }
        let Some(paths) = chosen else { return };
        match (b.want, target) {
            (Pick::Batch, _) => self.take_batches(&paths),
            (Pick::Data, _) => self.take_data(paths),
            // In full: each fit runs in its own output folder.
            (Pick::Table | Pick::File, Some(target)) => {
                self.set_text(&target, &paths[0].to_string_lossy());
            }
            (Pick::Table | Pick::File, None) => {}
        }
    }

    /// Browse for one file to fill flag `target`: an earlier run's feature
    /// table, or any file. It opens where the results go, else where the
    /// data was found.
    pub(super) fn browse_for(&mut self, target: Target, want: Pick) {
        let results = self.here.join(header_folder(&self.header));
        let dir = if want == Pick::Table && results.is_dir() {
            results
        } else {
            self.browse_dir.clone()
        };
        self.browsing = Some(target);
        self.browser = Some(Browser::open(dir, want, None));
    }
}
