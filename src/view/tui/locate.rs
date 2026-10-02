//! A run's data file that is not where its manifest says, as when a manifest
//! is copied from another machine: the view asks where it is, then writes
//! the answer into the manifest so every later reader finds it too.

use super::browse::Missing;
use super::*;
use crate::tui::browse::{Browser, Outcome};
use crate::view::activity::Activity;
use senna::run_manifest::{self, RunManifest};
use std::path::{Path, PathBuf};

impl App {
    /// Ask where the first missing data file is, when the scene found one
    /// missing, this view is on screen, and nothing else is being asked or
    /// rewriting the run (a recompute would save over the answer).
    pub(super) fn ask_missing_data(&mut self, on_screen: bool) -> bool {
        if !on_screen
            || self.modal.is_some()
            || self.relabeling.is_some()
            || self.recomputing.is_some()
        {
            return false;
        }
        let Some((index, recorded)) = self.scene.missing_data.take() else {
            return false;
        };
        self.ask_for_data(index, recorded);
        true
    }

    /// Open the browser on data file `index`, recorded at `recorded`.
    fn ask_for_data(&mut self, index: usize, recorded: String) {
        let dir = start_dir(&self.from);
        self.modal = Some(Modal::DataFile(
            Browser::open(dir, Missing(recorded), None),
            index,
        ));
    }

    /// A key in the browser. Returns whether anything changed.
    pub(super) fn data_file_key(&mut self, k: KeyEvent) -> bool {
        let Some(Modal::DataFile(b, index)) = self.modal.as_mut() else {
            return false;
        };
        let index = *index;
        match b.key(k) {
            Outcome::Ignored => return false,
            Outcome::Moved => {}
            Outcome::Cancelled => {
                self.modal = None;
                self.data_pending = None;
                self.message = Some(
                    "no data for observed counts · nothing recorded · showing the model's \
                     expectation"
                        .into(),
                );
            }
            Outcome::Chosen(c) => {
                self.modal = None;
                self.use_data(&c.file(), index);
            }
        }
        true
    }

    /// Point data file `index` at `chosen`. While others are still missing,
    /// ask for the next; once all are here, check they hold the run's cells,
    /// and only then record them in the manifest and show the observed
    /// counts. Nothing is written for data that is not this run's, nor for a
    /// search given up.
    fn use_data(&mut self, chosen: &Path, index: usize) {
        let pending = self.data_pending.take();
        let moved = RunManifest::load(&self.from).and_then(|(mut m, dir)| {
            if let Some(pending) = pending {
                m.data = pending;
            }
            let also = m.relocate_data(&dir, index, chosen)?;
            Ok((m, dir, also))
        });
        let (m, dir, also) = match moved {
            Ok(moved) => moved,
            Err(e) => return self.refuse(format!("could not record the data file: {e}")),
        };
        let name = files::name(chosen);
        let mut activity = Activity::new(m.clone(), dir);
        if let Some((next, recorded)) = activity.missing_inputs().into_iter().next() {
            self.message = Some(format!(
                "{name} found · where is {}?",
                files::name(Path::new(&recorded))
            ));
            self.data_pending = Some(m.data);
            return self.ask_for_data(next, recorded);
        }
        let found = self
            .scene
            .cell_names()
            .map(|cells| (activity.observed_cells_found(cells), cells.len()));
        let warning = match found {
            Some((Err(e), _)) => return self.refuse(format!("{name} does not read: {e}")),
            Some((Ok(0), n)) => {
                return self.refuse(format!("none of this run's {n} cells are in {name}"))
            }
            Some((Ok(k), n)) if k * 10 < n * 9 => Some(format!(
                "only {k} of this run's {n} cells are in {name}: is it this run's data?"
            )),
            _ => None,
        };
        if let Err(e) = m.save(&self.from) {
            return self.refuse(format!("could not save the manifest: {e}"));
        }
        let from = self.from.clone();
        self.open_round(&from, "reloaded");
        // The data is open already: the scene keeps it rather than reading
        // it again.
        self.scene.activity = Some(activity);
        self.change(|s| {
            s.source = crate::view::activity::Source::Observed;
            s.refresh_activity();
        });
        if let Some(w) = warning {
            self.message = Some(w);
        } else if self.scene.source == crate::view::activity::Source::Observed {
            // Else it fell back to the expectation, and the note says why.
            let more = match also {
                0 => String::new(),
                n => format!(" (and {n} more moved with it)"),
            };
            self.message = Some(format!(
                "data: {}{more} · recorded in {}",
                crate::tui::shown(chosen),
                files::name(&from)
            ));
        }
    }

    /// Give up on the chosen data, saying `why`: the manifest is untouched.
    fn refuse(&mut self, why: String) {
        self.message = Some(format!("{why} · not recorded"));
    }
}

/// Where to start looking: the folder the viewer runs from, else the
/// run's own folder.
fn start_dir(from: &Path) -> PathBuf {
    std::env::current_dir()
        .ok()
        .or_else(|| std::fs::canonicalize(run_manifest::manifest_dir(from)).ok())
        .unwrap_or_else(|| PathBuf::from("/"))
}
