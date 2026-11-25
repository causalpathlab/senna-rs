//! The run screen: the clones step, the queue and its log, and choices a fit asks for.

use super::jobs::{Keep, Queue};
use super::{jobs, rerun_argv, script, App, Choice, Screen};
use ratatui::crossterm::event::{KeyCode, KeyEvent};
use std::path::Path;

impl App {
    // ───────────── run ─────────────

    /// Whether the queue waits on the clones of `mung clones`.
    pub(super) fn asking(&self) -> bool {
        self.queue.as_ref().is_some_and(Queue::asking)
    }

    /// Keys while the queue waits on the clones: keep them, run without,
    /// or stop.
    pub(super) fn clones_key(&mut self, k: KeyEvent) {
        let keep = match k.code {
            KeyCode::Up | KeyCode::Char('k') => {
                self.clones_scroll = self.clones_scroll.saturating_sub(1);
                return;
            }
            KeyCode::Down | KeyCode::Char('j') => {
                self.clones_scroll += 1;
                return;
            }
            KeyCode::Char('y') => Keep::Use,
            KeyCode::Char('n') => Keep::Without,
            KeyCode::Char('s') => Keep::Stop,
            _ => return,
        };
        if let Some(q) = &self.queue {
            q.answer(keep);
        }
        self.message = Some(
            match keep {
                Keep::Use => "the fits keep the clones apart",
                Keep::Without => "the fits run without the clones",
                Keep::Stop => "stopped after mung clones",
            }
            .into(),
        );
    }

    pub(super) fn run_key(&mut self, k: KeyEvent) {
        let page = self.log_rows.get().max(1);
        match k.code {
            KeyCode::Up | KeyCode::Char('k') => self.scroll_log(-1),
            KeyCode::Down | KeyCode::Char('j') => self.scroll_log(1),
            KeyCode::PageUp => self.scroll_log(-(page as isize)),
            KeyCode::PageDown => self.scroll_log(page as isize),
            KeyCode::Home => self.log_top = Some(0),
            KeyCode::End => self.log_top = None,
            KeyCode::Char('s') => {
                if let Some(q) = &self.queue {
                    self.message = Some(if q.stop() {
                        "killed".into()
                    } else {
                        "interrupted: the fit wraps up and stops · s again kills it".into()
                    });
                }
            }
            KeyCode::Char('v') => {
                let done = self.queue.as_ref().map(Queue::done).unwrap_or_default();
                if self.running() {
                    self.message = Some("still running".into());
                } else if done.is_empty() {
                    self.message = Some("no fit finished".into());
                } else {
                    self.view = done;
                    self.quit = true;
                }
            }
            _ => {}
        }
    }

    /// Once the queue is through, the first open choice a finished fit
    /// left in its manifest, put to the user.
    pub(super) fn read_choices(&mut self) {
        let Some(q) = &mut self.queue else { return };
        if q.choices_read || !q.finished() {
            return;
        }
        q.choices_read = true;
        self.choice = q
            .jobs
            .iter()
            .zip(q.states())
            .enumerate()
            .filter(|(_, (j, s))| j.is_fit() && *s == jobs::State::Done)
            .find_map(|(i, (j, _))| {
                let run = j.result();
                let (m, _) = senna::run_manifest::RunManifest::load(&run).ok()?;
                let (decisions, _) = crate::view::typed_decisions(&run, &m.decisions);
                let decision = decisions.into_iter().next()?;
                Some(Choice {
                    job: i,
                    decision,
                    row: 0,
                })
            });
        if self.choice.is_some() {
            self.screen = Screen::Run;
        }
    }

    /// Keys while a finished fit's choice is put: pick an option and rerun
    /// with it, or leave it.
    pub(super) fn choice_key(&mut self, k: KeyEvent) {
        let Some(c) = &mut self.choice else { return };
        let n = c.decision.options.len();
        if let Some(row) = legume_tui::keys::list_move(&k, c.row, n, true) {
            c.row = row;
            return;
        }
        match k.code {
            KeyCode::Esc | KeyCode::Char('q') => {
                let left = self
                    .choice
                    .take()
                    .and_then(|c| self.queue.as_ref().map(|q| q.jobs[c.job].result()));
                self.message =
                    left.map(|m| format!("left open; the choice stays in {}", self.shown(&m)));
            }
            KeyCode::Enter => {
                let Some(c) = self.choice.take() else { return };
                let Some(q) = &self.queue else { return };
                let mut job = q.jobs[c.job].clone();
                let option = &c.decision.options[c.row];
                let out = format!("{}_{}{}", job.out, c.decision.topic, c.row);
                // The fit's queued steps follow the rerun, on its new out.
                let steps = q
                    .jobs
                    .iter()
                    .filter(|j| j.step.is_some_and(|of| of.fit == c.job))
                    .map(|j| {
                        let mut j = j.clone();
                        if let Some(of) = j.step {
                            let step = &self.rows[j.row].steps[of.step];
                            j.argv = self.step_argv(step, &out, 0);
                        }
                        j.out.clone_from(&out);
                        j.lane = 0;
                        j.step = j.step.map(|of| jobs::StepOf { fit: 0, ..of });
                        j
                    })
                    .collect::<Vec<_>>();
                job.argv = rerun_argv(&job.argv, &out, &option.rerun_args);
                job.out = out;
                job.lane = 0;
                self.queue = Some(Queue::start(std::iter::once(job).chain(steps).collect()));
                self.log_top = None;
                self.message = Some(format!("rerunning with {}", option.label));
            }
            _ => {}
        }
    }

    /// Move the Run screen's log by `d` lines; at its end it follows the
    /// newest again.
    fn scroll_log(&mut self, d: isize) {
        let Some(q) = &self.queue else { return };
        let (first, end) = q
            .shared
            .lock()
            .map(|s| (s.dropped, s.dropped + s.log.len()))
            .unwrap_or_default();
        let last_top = end.saturating_sub(self.log_rows.get()).max(first);
        let top = self.log_top.unwrap_or(last_top).clamp(first, last_top);
        let top = top.saturating_add_signed(d).clamp(first, last_top);
        self.log_top = (top < last_top).then_some(top);
    }

    /// What is printed once the terminal is given back: each script.
    /// A path as shown: relative to where senna run started when under it.
    pub(super) fn shown(&self, p: &Path) -> String {
        p.strip_prefix(&self.here)
            .ok()
            .filter(|r| !r.as_os_str().is_empty())
            .unwrap_or(p)
            .to_string_lossy()
            .into_owned()
    }

    pub(super) fn summary(&self) -> Vec<String> {
        let Some(q) = &self.queue else {
            return Vec::new();
        };
        q.jobs
            .iter()
            .zip(q.states())
            .map(|(j, s)| {
                let what = match s {
                    jobs::State::Done => "done".to_string(),
                    jobs::State::Failed(why) => format!("failed: {why}"),
                    jobs::State::Stopped => "stopped".to_string(),
                    jobs::State::Waiting | jobs::State::Running => "not finished".to_string(),
                };
                // A step's fit script refuses to rerun over the fit it made,
                // so a step is given again as its own command.
                if j.step.is_some() {
                    return format!(
                        "{} {what}; again with: (cd {} && senna {})",
                        j.method,
                        script::quote(&self.shown(&j.dir)),
                        script::line(&j.argv)
                    );
                }
                let script = j.script();
                if script.exists() {
                    format!(
                        "{} {what}; again with: bash {}",
                        j.method,
                        self.shown(&script)
                    )
                } else {
                    format!("{} {what}", j.method)
                }
            })
            .collect()
    }
}
