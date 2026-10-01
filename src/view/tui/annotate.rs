//! A first annotation from the view: browse to a marker panel, then run
//! `lupin annotate` on this run and open the round it writes.

use super::browse::Panels;
use super::*;
use crate::tui::browse::{Browser, Entry, Outcome};
use crate::view::decide;
use data_beans::utilities::name_matching::GeneIndex;
use std::path::{Path, PathBuf};

impl App {
    /// Open the marker-panel browser where the run's own panel is, else
    /// where a file nearby whose name mentions markers is, else beside the
    /// run.
    pub(super) fn ask_markers(&mut self) {
        if self.scene.review.is_some() {
            self.message = Some("leave relabel mode first (R)".into());
            return;
        }
        let names = self.scene.searchable();
        let genes = (!names.is_empty()).then(|| GeneIndex::build(&names));
        let guess = self.markers_guess();
        let dir = guess
            .as_deref()
            .and_then(Path::parent)
            .map(Path::to_path_buf)
            .or_else(|| {
                self.from
                    .canonicalize()
                    .ok()?
                    .parent()
                    .map(Path::to_path_buf)
            })
            .or_else(|| std::env::current_dir().ok())
            .unwrap_or_else(|| PathBuf::from("/"));
        let dir = dir.canonicalize().unwrap_or(dir);
        let select = guess.as_deref().map(files::name);
        self.modal = Some(Modal::MarkersFile(Browser::open(
            dir,
            Panels(genes),
            select.as_deref(),
        )));
    }

    /// The run's recorded marker panel, if it is still there.
    fn markers_guess(&self) -> Option<PathBuf> {
        let (m, dir) = self.scene.data.run.as_ref()?;
        let p = senna::run_manifest::resolve(dir, m.annotate.markers.as_deref()?);
        p.is_file().then_some(p)
    }

    /// A key in the browser. Returns whether anything changed.
    pub(super) fn markers_key(&mut self, k: KeyEvent) -> bool {
        let Some(Modal::MarkersFile(b)) = self.modal.as_mut() else {
            return false;
        };
        match b.key(k) {
            Outcome::Ignored => return false,
            Outcome::Moved => {}
            Outcome::Cancelled => {
                self.modal = None;
                self.message = Some("annotation cancelled".into());
            }
            Outcome::Chosen(paths) => match b.current() {
                Some(Entry::File(name, p)) if p.types < 2 => {
                    self.message = Some(format!("{name} names one cell type; lupin needs several"));
                }
                _ => {
                    self.modal = None;
                    if let Some(path) = paths.into_iter().next() {
                        self.run_annotate(path);
                    }
                }
            },
        }
        true
    }

    fn run_annotate(&mut self, markers: PathBuf) {
        let run = self
            .from
            .canonicalize()
            .unwrap_or_else(|_| self.from.clone());
        let out = decide::annotate_out(&run);
        let lupin = self.lupin.clone();
        let progress: std::sync::Arc<std::sync::Mutex<String>> = Default::default();
        let shared = progress.clone();
        let done = Pending::spawn(move || decide::annotate(&lupin, &run, &markers, &out, &shared));
        self.relabeling = Some(Relabeling {
            job: RelabelJob::Annotate,
            started: std::time::Instant::now(),
            sent: Vec::new(),
            progress,
            done,
        });
    }
}
