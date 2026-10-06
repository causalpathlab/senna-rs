//! `r`: have senna compute a run's layouts or clusterings again, from a
//! menu, while the view stays usable; the run reloads when it is done.

use super::*;
use crate::view::recompute::{self, Menu, Step, Target};

/// senna recomputing on a worker thread.
pub(super) struct Recomputing {
    started: std::time::Instant,
    /// What was asked for, for the status line and the reload message.
    what: String,
    /// The run being rewritten, reloaded when it is done.
    from: std::path::PathBuf,
    /// The layout asked for, shown once the run reloads: the map on screen
    /// is otherwise kept, which would hide a new method behind the old.
    show: Option<(String, crate::view::SpaceKind)>,
    progress: std::sync::Arc<std::sync::Mutex<String>>,
    stopper: std::sync::Arc<crate::tui::child::Stopper>,
    done: Pending<()>,
}

/// A view that goes (the viewer quits) stops what senna does for it,
/// rather than leave it rewriting the run unseen.
impl Drop for Recomputing {
    fn drop(&mut self) {
        self.stopper.kill();
    }
}

impl App {
    /// Open the menu for the run on screen, the map on screen chosen.
    pub(super) fn open_recompute(&mut self) {
        self.open_recompute_with(None);
    }

    /// Open the menu with the map on screen chosen, laid out with `method`
    /// when given (else the one on screen).
    pub(super) fn open_recompute_with(&mut self, method: Option<&str>) {
        if self.recomputing.is_some() {
            self.message = Some("already recomputing · it reloads when done".into());
            return;
        }
        let target = match Target::load(&self.from.to_string_lossy()) {
            Ok(t) => t,
            Err(e) => {
                self.message = Some(format!("cannot recompute: {e}"));
                return;
            }
        };
        if target.offered().is_empty() {
            self.message = Some("nothing to recompute for this run".into());
            return;
        }
        // A zoom was laid out from its root map: redo that one.
        let kind = self.scene.current().kind;
        let method = method.map_or_else(
            || self.scene.data.spaces[self.scene.root()].method.clone(),
            str::to_string,
        );
        let on_screen = Step::on_screen(kind, &method);
        let menu = Menu::new(&target, on_screen, &method);
        self.modal = Some(Modal::Recompute(target, menu));
    }

    /// A key in the menu. Returns whether anything changed.
    pub(super) fn recompute_key(&mut self, k: KeyEvent) -> bool {
        let Some(Modal::Recompute(_, menu)) = self.modal.as_mut() else {
            return false;
        };
        match k.code {
            KeyCode::Up | KeyCode::Char('k') => menu.step(-1),
            KeyCode::Down | KeyCode::Char('j') => menu.step(1),
            KeyCode::Char(' ') => menu.toggle(),
            KeyCode::Left | KeyCode::Char('h') => menu.step_setting(-1),
            KeyCode::Right | KeyCode::Char('l') => menu.step_setting(1),
            KeyCode::Enter => self.start_recompute(),
            KeyCode::Char('r') if k.modifiers.contains(KeyModifiers::CONTROL) => {
                self.start_recompute();
            }
            KeyCode::Esc => self.modal = None,
            _ => return false,
        }
        true
    }

    fn start_recompute(&mut self) {
        let Some(Modal::Recompute(target, menu)) = self.modal.take() else {
            return;
        };
        let chosen = menu.chosen();
        if chosen.is_empty() {
            self.message = Some("nothing chosen · space chooses".into());
            return;
        }
        self.launch(target, chosen);
    }

    /// `T`: resolve topics for a run that has none, one per cell cluster,
    /// as senna does for the `r` menu's steps; `H` then draws them.
    pub(super) fn resolve_topics(&mut self) {
        if self.recomputing.is_some() {
            self.message = Some("already recomputing · it reloads when done".into());
            return;
        }
        let target = match Target::load(&self.from.to_string_lossy()) {
            Ok(t) => t,
            Err(e) => {
                self.message = Some(format!("cannot make topics: {e}"));
                return;
            }
        };
        if !target.can_resolve_topics {
            self.message = Some(if self.scene.run_has_latent() {
                "this run has its topics (or cell factors) already · H draws topics".into()
            } else {
                "topics need the run's cell and gene embeddings".into()
            });
            return;
        }
        self.launch(target, vec![(Step::Topics, "")]);
    }

    /// Have senna run `chosen` for `target` on a worker thread.
    fn launch(&mut self, target: Target, chosen: Vec<(Step, &'static str)>) {
        let what = chosen
            .iter()
            .map(|(s, m)| match s.say(m) {
                Some(said) => format!("{} ({said})", s.label()),
                None => s.label().to_string(),
            })
            .collect::<Vec<_>>()
            .join(", ");
        let show = chosen.iter().find_map(|(s, m)| s.shows(m));
        let progress: std::sync::Arc<std::sync::Mutex<String>> = Default::default();
        let stopper = std::sync::Arc::new(crate::tui::child::Stopper::default());
        let (shared, stops) = (progress.clone(), stopper.clone());
        let done = Pending::spawn(move || recompute::run(&target, &chosen, &shared, &stops));
        self.recomputing = Some(Recomputing {
            started: std::time::Instant::now(),
            what,
            from: self.from.clone(),
            show,
            progress,
            stopper,
            done,
        });
    }

    /// Take a finished recompute: reload the run (what finished before a
    /// failure is kept too). Returns whether it did.
    pub(super) fn finish_recompute(&mut self) -> bool {
        let Some(result) = self
            .recomputing
            .as_ref()
            .and_then(|r| r.done.poll("senna stopped without an answer"))
        else {
            return false;
        };
        let r = self.recomputing.take().expect("checked above");
        self.open_round(&r.from.clone(), "reloaded");
        if let Some((method, kind)) = &r.show {
            self.show_layout(method, *kind);
        }
        match result {
            Ok(()) => self.pop(format!("recomputed {}", r.what)),
            Err(e) if e == "stopped" => {
                self.message = Some(format!(
                    "stopped recomputing {} · what finished before is kept",
                    r.what
                ));
            }
            Err(e) => self.message = Some(e),
        }
        true
    }

    /// Put the map of `method` on screen, of `kind` when the run has it.
    /// Returns whether the run has that method at all.
    pub(super) fn show_layout(&mut self, method: &str, kind: crate::view::SpaceKind) -> bool {
        let spaces = &self.scene.data.spaces;
        let same = spaces
            .iter()
            .position(|s| s.method == method && s.kind == kind);
        let any = spaces.iter().position(|s| s.method == method);
        match same.or(any) {
            Some(i) => {
                if i != self.scene.space {
                    self.switch_space(i);
                }
                true
            }
            None => false,
        }
    }

    /// While senna rewrites the run: esc stops it, and nothing that changes
    /// the run (another round, lupin) starts. Returns whether `k` was taken.
    pub(super) fn guard_recompute(&mut self, k: KeyEvent) -> bool {
        let Some(r) = &self.recomputing else {
            return false;
        };
        match k.code {
            KeyCode::Esc => {
                self.message = Some(if r.stopper.stop() {
                    "senna killed".into()
                } else {
                    "interrupting senna… · esc again kills it".into()
                });
            }
            KeyCode::Char(',' | '.' | 'A' | 'S' | 'P') => {
                self.message = Some(format!(
                    "senna is recomputing {} for this run · wait, or esc stops it",
                    r.what
                ));
            }
            _ => return false,
        }
        true
    }

    /// The status line while senna works.
    pub(super) fn recompute_line(&self) -> Option<String> {
        let r = self.recomputing.as_ref()?;
        let now = r.progress.lock().map(|p| p.clone()).unwrap_or_default();
        Some(format!(
            "recomputing {} · {now} · {:.0} s · reloads when done · esc stops",
            r.what,
            r.started.elapsed().as_secs_f32()
        ))
    }

    /// The menu, as a popup over the map.
    pub(super) fn recompute_lines(menu: &Menu, run: &str) -> Vec<Line<'static>> {
        let bold = Style::default().add_modifier(ratatui::style::Modifier::BOLD);
        let mut lines = vec![
            Line::from(Span::styled(format!(" Recompute for {run}"), bold)),
            Line::from(""),
        ];
        for (k, item) in menu.items.iter().enumerate() {
            let here = k == menu.at;
            let cursor = if here { " ▸ " } else { "   " };
            let mark = if item.on { "[x]" } else { "[ ]" };
            let method = item
                .step
                .say(item.setting())
                .map_or_else(String::new, |said| format!("‹ {said} ›"));
            // The cursor's line is a dark bar; chosen lines stand out in bold.
            let style = match (here, item.on) {
                (true, _) => selected(),
                (false, true) => bold,
                (false, false) => Style::default(),
            };
            let text = format!("{cursor}{mark} {:<26}{method} ", item.step.label());
            lines.push(Line::from(Span::styled(text, style)));
        }
        lines.push(Line::from(""));
        lines.push(Line::from(Span::styled(
            " replaces those files of the run; the view reloads when done",
            hint(),
        )));
        lines.push(Line::from(
            " ↑ ↓ move   space choose   ← → method or resolution   ctrl+r / enter run   esc cancel",
        ));
        lines
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A view whose run senna is recomputing (a job that never answers).
    fn busy() -> (
        App,
        std::sync::Arc<crate::tui::child::Stopper>,
        std::sync::mpsc::Sender<Result<(), String>>,
    ) {
        let mut app = App::new(
            crate::view::tests::scene(),
            Picker::halfblocks(),
            "r.senna.json".into(),
            "lupin".into(),
        );
        let (tx, rx) = std::sync::mpsc::channel();
        let stopper = std::sync::Arc::new(crate::tui::child::Stopper::default());
        app.recomputing = Some(Recomputing {
            started: std::time::Instant::now(),
            what: "cell layout (umap)".into(),
            from: "r.senna.json".into(),
            show: None,
            progress: Default::default(),
            stopper: stopper.clone(),
            done: Pending(rx),
        });
        (app, stopper, tx)
    }

    fn press(app: &mut App, c: KeyCode) {
        app.key(KeyEvent::new(c, KeyModifiers::NONE));
    }

    #[test]
    fn nothing_that_changes_the_run_starts_while_senna_rewrites_it() {
        let (mut app, stopper, _tx) = busy();
        for c in [',', '.', 'A', 'S', 'P'] {
            app.message = None;
            press(&mut app, KeyCode::Char(c));
            let said = app.message.clone().unwrap_or_default();
            assert!(said.contains("recomputing"), "{c}: {said}");
            assert!(app.modal.is_none() && app.relabeling.is_none(), "{c}");
        }
        assert!(!stopper.is_stopped());
        // Esc stops senna.
        press(&mut app, KeyCode::Esc);
        assert!(stopper.is_stopped());
    }

    #[test]
    fn a_finished_recompute_reloads_the_run_it_was_for() {
        let (mut app, _stopper, tx) = busy();
        app.from = "elsewhere.senna.json".into();
        tx.send(Ok(())).unwrap();
        assert!(app.finish_recompute());
        // It tried the recomputed run (not the one now on screen).
        let said = app.message.clone().unwrap_or_default();
        assert!(said.contains("r.senna.json"), "{said}");
    }

    #[test]
    fn a_recomputed_layout_is_put_on_screen() {
        let (mut app, _stopper, _tx) = busy();
        assert_eq!(app.scene.current().method, "umap");
        assert!(app.show_layout("phate", crate::view::SpaceKind::Cells));
        assert_eq!(app.scene.current().method, "phate");
        assert_eq!(app.scene.current().kind, crate::view::SpaceKind::Cells);
        // A method the run lacks leaves the map as it is.
        assert!(!app.show_layout("tsne", crate::view::SpaceKind::Cells));
        assert_eq!(app.scene.current().method, "phate");
    }

    #[test]
    fn a_layout_step_shows_its_method_and_a_clustering_nothing() {
        use crate::view::SpaceKind;
        assert_eq!(
            Step::CellLayout.shows("tsne"),
            Some(("tsne".into(), SpaceKind::Cells))
        );
        assert_eq!(
            Step::FeatureLayout.shows("phate"),
            Some(("phate".into(), SpaceKind::Features))
        );
        assert_eq!(Step::CellClusters.shows("1"), None);
    }

    #[test]
    fn quitting_stops_senna() {
        let (app, stopper, _tx) = busy();
        drop(app);
        assert!(stopper.is_stopped());
    }
}
