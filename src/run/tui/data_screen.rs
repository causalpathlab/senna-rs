//! The data and methods screens: taking in data and batch files, relabelling, and their keys.

use super::{batches, batches::Batch, data, data::Pair, data::Pick, App, Screen, Target};
use ratatui::crossterm::event::{KeyCode, KeyEvent};
use std::path::{Path, PathBuf};

impl App {
    pub(super) fn take_data(&mut self, paths: Vec<PathBuf>) {
        let mut added = 0;
        // Each folder's label files, listed once for all its data.
        let mut labels: std::collections::HashMap<PathBuf, Vec<PathBuf>> =
            std::collections::HashMap::new();
        for p in paths {
            if self.pairs.iter().any(|q| q.data == p) {
                continue;
            }
            let mut pair = Pair::pending(p);
            let dir = pair.data.parent().unwrap_or(Path::new(".")).to_path_buf();
            let near = labels
                .entry(dir)
                .or_insert_with_key(|d| data::label_files_in(d));
            if let Some(file) = data::beside(&pair.data, near) {
                pair.batch = Batch::labels(file).unwrap_or(Batch::Own);
            }
            // Opening a large backend takes a while: not here.
            let (tx, path) = (self.described.0.clone(), pair.data.clone());
            std::thread::spawn(move || {
                let d = data::describe(&path);
                let _ = tx.send((path, d));
            });
            self.describing += 1;
            self.pairs.push(pair);
            added += 1;
        }
        self.pair_row = self.pairs.len().saturating_sub(1);
        let found = self
            .pairs
            .iter()
            .filter(|p| matches!(p.batch, Batch::Labels { .. }))
            .count();
        self.message = Some(format!(
            "{added} data file{} added; {found} of {} with a label file",
            if added == 1 { "" } else { "s" },
            self.pairs.len()
        ));
    }

    /// One batch file goes to the data row under the cursor; several are
    /// paired with all rows by name.
    pub(super) fn take_batches(&mut self, paths: &[PathBuf]) {
        let row = self.pair_row;
        let chosen: Vec<Option<PathBuf>> = if let [one] = paths {
            (0..self.pairs.len())
                .map(|i| (i == row).then(|| one.clone()))
                .collect()
        } else {
            let data: Vec<PathBuf> = self.pairs.iter().map(|p| p.data.clone()).collect();
            let (chosen, paired) = data::assign(&data, paths);
            let n = data.len();
            self.message = Some(match paired {
                data::Paired::ByName(_) => format!("{n} batch files paired by name"),
                data::Paired::InOrder => format!(
                    "no names match: {n} batch files paired in the order listed; check them"
                ),
                data::Paired::Partly(k) => format!(
                    "{k} of {n} data files matched a batch file by name; b sets the rest one by one"
                ),
            });
            chosen
        };
        for (p, file) in self.pairs.iter_mut().zip(chosen) {
            let Some(file) = file else { continue };
            match Batch::labels(file.clone()) {
                Ok(b) => p.batch = b,
                Err(e) => {
                    let why = format!("{}: {e}", crate::tui::name(&file));
                    self.message = Some(match self.message.take() {
                        Some(m) => format!("{m}; {why}"),
                        None => why,
                    });
                }
            }
        }
    }

    /// Keys in the list of a label file's labels: Enter renames one.
    pub(super) fn relabel_key(&mut self, k: KeyEvent) {
        let Some((row, at)) = self.relabel else {
            return;
        };
        let n = match &self.pairs.get(row).map(|p| &p.batch) {
            Some(Batch::Labels { counts, .. }) => counts.len(),
            _ => 0,
        };
        let at = match k.code {
            KeyCode::Esc | KeyCode::Char('q') => {
                self.relabel = None;
                return;
            }
            _ if legume_tui::keys::list_delta(&k, true).is_some() => {
                legume_tui::keys::list_move(&k, at, n, true).unwrap_or(at)
            }
            KeyCode::Enter if at < n => {
                let now = match &self.pairs[row].batch {
                    Batch::Labels {
                        counts, renamed, ..
                    } => renamed.get(&counts[at].0).unwrap_or(&counts[at].0).clone(),
                    _ => String::new(),
                };
                self.edit(Target::Label(row, at), now);
                at
            }
            _ => at,
        };
        self.relabel = Some((row, at));
    }

    // ───────────── data ─────────────

    pub(super) fn data_key(&mut self, k: KeyEvent) {
        let last = self.pairs.len().saturating_sub(1);
        if let Some(row) = legume_tui::keys::list_move(&k, self.pair_row, self.pairs.len(), true) {
            self.pair_row = row;
            return;
        }
        match k.code {
            KeyCode::Char('a') => self.browse(Pick::Data),
            KeyCode::Char('b') if !self.pairs.is_empty() => self.browse(Pick::Batch),
            KeyCode::Char('x') => {
                if let Some(p) = self.pairs.get_mut(self.pair_row) {
                    p.batch = Batch::Own;
                }
            }
            KeyCode::Char('X') => self.pairs.iter_mut().for_each(|p| p.batch = Batch::Own),
            KeyCode::Char('n') => {
                if let Some(p) = self.pairs.get(self.pair_row) {
                    let now = match &p.batch {
                        Batch::Named(name) => name.clone(),
                        _ if p.tags.is_none() => batches::own_name(&p.data),
                        _ => String::new(),
                    };
                    self.edit(Target::BatchName(self.pair_row), now);
                }
            }
            KeyCode::Char('e') => {
                let labelled = self
                    .pairs
                    .get(self.pair_row)
                    .is_some_and(|p| matches!(p.batch, Batch::Labels { .. }));
                if labelled {
                    self.relabel = Some((self.pair_row, 0));
                } else {
                    self.message = Some("e renames the labels of a label file: b picks one".into());
                }
            }
            KeyCode::Char('d') | KeyCode::Delete if !self.pairs.is_empty() => {
                self.pairs.remove(self.pair_row);
                self.pair_row = self.pair_row.min(self.pairs.len().saturating_sub(1));
            }
            KeyCode::Char('K') if self.pair_row > 0 => {
                self.pairs.swap(self.pair_row, self.pair_row - 1);
                self.pair_row -= 1;
            }
            KeyCode::Char('J') if self.pair_row < last => {
                self.pairs.swap(self.pair_row, self.pair_row + 1);
                self.pair_row += 1;
            }
            KeyCode::Enter => self.screen = Screen::Methods,
            _ => {}
        }
    }

    // ───────────── methods ─────────────

    pub(super) fn methods_key(&mut self, k: KeyEvent) {
        if let Some(row) = legume_tui::keys::list_move(&k, self.method_row, self.rows.len(), true) {
            self.method_row = row;
            return;
        }
        match k.code {
            KeyCode::Char(' ') => {
                let r = &mut self.rows[self.method_row];
                r.on = !r.on;
            }
            KeyCode::Char('O') => {
                let header = self.header.clone();
                self.edit(Target::Header, header);
            }
            KeyCode::Char('o') => {
                let out = self.rows[self.method_row].out.clone();
                self.edit(Target::Out(self.method_row), out);
            }
            KeyCode::Enter | KeyCode::Right => {
                // Queuing the second fit is when the shared page first
                // exists, so show it then; otherwise the method's own page.
                let before = self.shared_rows().len();
                self.rows[self.method_row].on = true;
                if before == 1 && self.shared_rows().len() == 2 {
                    self.go_to(Screen::Params);
                } else {
                    self.show_params(self.method_row);
                }
            }
            _ => {}
        }
    }
}
