//! Planning the queued commands, and the confirm popup that shows them.

use super::jobs::{Job, Queue, Tool, CLONES_FLAG};
use super::{batches, form, script, share, App, Builder, Planned, Row, Screen};
use ratatui::crossterm::event::{KeyCode, KeyEvent};
use std::path::{Path, PathBuf};

impl App {
    // ───────────── confirm ─────────────

    /// Where row `i` writes: its folder and the prefix in it.
    fn out_of(&self, i: usize) -> (PathBuf, String) {
        let abs = self.here.join(&self.rows[i].out);
        let dir = abs
            .parent()
            .map_or_else(|| self.here.clone(), script::resolved);
        let name = abs
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_default();
        (dir, name)
    }

    /// The command `tool`'s lines are checked against: a mung row exists
    /// only when mung described itself.
    pub(super) fn command_of(&self, tool: Tool) -> &clap::Command {
        match tool {
            Tool::Senna => &self.cli,
            Tool::Mung => self
                .mung
                .as_ref()
                .expect("a mung row exists only when mung described itself"),
        }
    }

    pub(super) fn program_of(&self, tool: Tool) -> PathBuf {
        match tool {
            Tool::Senna => self.senna_program.clone(),
            Tool::Mung => self.mung_program.clone(),
        }
    }

    /// The queued `mung clones` row, if one is.
    pub(super) fn clones_row(&self) -> Option<usize> {
        self.rows.iter().position(|r| r.on && r.tool == Tool::Mung)
    }

    /// How senna fit `r` takes the clone table `table` of the queued
    /// `mung clones`: the table, or a warning that it cannot, or a problem.
    fn clones_for(r: &Row, table: &str) -> (Option<String>, Option<String>, Option<String>) {
        match r.form.fields.iter().find(|f| f.long == CLONES_FLAG) {
            Some(f) if !f.is_default() => (
                None,
                None,
                Some(format!(
                    "--{CLONES_FLAG} is set here and mung clones is queued: reset one"
                )),
            ),
            Some(_) => (Some(table.to_string()), None, None),
            None => (
                None,
                Some(format!("takes no --{CLONES_FLAG}: runs without the clones")),
                None,
            ),
        }
    }

    /// The queued fits, each checked.
    pub(super) fn plan(&self) -> Vec<Planned> {
        let mut outs: Vec<PathBuf> = Vec::new();
        let mut planned = Vec::new();
        let table = self.clones_row().map(|i| {
            let (dir, out) = self.out_of(i);
            dir.join(format!("{out}.{}", Tool::Mung.result()))
        });
        // Each partition a queued fit builds: the first fit to build it, by
        // its manifest and name.
        let mut builders: Vec<(Vec<(String, String)>, Builder)> = Vec::new();
        // The GPU the next fit on `--device cuda` takes.
        let mut next_gpu = 0;
        for (i, r) in self.rows.iter().enumerate().filter(|(_, r)| r.on) {
            let (dir, out) = self.out_of(i);
            let rel = |p: &Path| script::relative(p, &dir).to_string_lossy().into_owned();
            let data: Vec<String> = self.pairs.iter().map(|p| rel(&p.data)).collect();
            let (batch_files, labels, batch_problem) = if r.form.takes_batches() {
                match batches::files(&self.pairs, &dir, &out) {
                    Ok((passed, written)) => (passed, written, None),
                    Err(why) => (Vec::new(), Vec::new(), Some(why)),
                }
            } else {
                (Vec::new(), Vec::new(), None)
            };
            let batch_files: Vec<String> = batch_files.iter().map(|p| rel(p)).collect();
            // The clones go to every fit that takes them.
            let (clones, warning, clones_problem) = match (r.tool, &table) {
                (Tool::Senna, Some(t)) => Self::clones_for(r, &rel(t)),
                _ => (None, None, None),
            };
            let mut notes: Vec<String> = warning.into_iter().collect();
            // A partition another queued fit builds is collapsed on, not
            // built again.
            let mut pb_from = None;
            if let Some(key) = share::key(&r.form, clones.is_some()) {
                match builders.iter().find(|(k, _)| *k == key) {
                    Some((_, b)) => {
                        pb_from = Some(rel(&b.manifest));
                        notes.push(format!(
                            "collapses on {}'s pseudobulks (--pb-from; its pb tree stays there)",
                            b.method
                        ));
                    }
                    // bge collapses on a partition but keeps none of its own
                    // for another fit.
                    None if r.form.name == "bge" => {}
                    None => builders.push((
                        key,
                        Builder {
                            manifest: dir.join(format!("{out}.{}", Tool::Senna.result())),
                            method: r.label.clone(),
                        },
                    )),
                }
            }
            // On several GPUs, the fits on one take them in turn (a
            // `--device-no` given by hand stays): each GPU is a lane, its
            // fits run in turn and the GPUs side by side.
            let mut argv = r.form.argv(&data, &batch_files, &out);
            let lane = match self.gpu_of(r) {
                Some(Some(given)) => given,
                Some(None) => {
                    let gpu = next_gpu % self.gpus;
                    next_gpu += 1;
                    if gpu > 0 {
                        argv.extend(["--device-no".to_string(), gpu.to_string()]);
                    }
                    gpu
                }
                None => 0,
            };
            let job = Job {
                tool: r.tool,
                program: self.program_of(r.tool),
                row: i,
                method: r.label.clone(),
                dir: dir.clone(),
                out: out.clone(),
                argv,
                labels,
                clones,
                pb_from,
                lane,
            };
            let mut blamed = None;
            let problem = if self.pairs.is_empty() {
                Some("no data files: add some on the Data screen".to_string())
            } else if let Some(p) = batch_problem.or(clones_problem) {
                Some(p)
            } else if let Some(there) = [job.result(), job.script(), job.batches()]
                .into_iter()
                .find(|p| p.exists())
            {
                Some(format!(
                    "{} exists: change --out (o on Methods)",
                    self.shown(&there)
                ))
            } else if outs.contains(&dir.join(&out)) {
                Some("another queued method writes the same --out".to_string())
            } else if dir.exists() && !dir.is_dir() {
                // A folder not there yet is made when the run starts.
                Some(format!("{} is not a folder", self.shown(&dir)))
            } else {
                form::check(self.command_of(r.tool), &job.command())
                    .err()
                    .inspect(|why| {
                        blamed = form::blamed(why, &r.form.fields).map(str::to_string);
                    })
            };
            outs.push(dir.join(&out));
            planned.push(Planned {
                warning: (!notes.is_empty()).then(|| notes.join("; ")),
                job,
                problem,
                blamed,
            });
        }
        planned
    }

    /// Whether row `r`'s fit runs on a GPU while there are several to
    /// spread fits over: `Some(Some(d))` when `--device-no d` is given,
    /// `Some(None)` when it is free to take one; `None` otherwise.
    fn gpu_of(&self, r: &Row) -> Option<Option<usize>> {
        let value = |long: &str| r.form.fields.iter().find(|f| f.long == long);
        if self.gpus < 2 || value("device").is_none_or(|f| f.value.trim() != "cuda") {
            return None;
        }
        Some(
            value("device-no")
                .filter(|f| !f.is_default())
                .and_then(|f| f.value.trim().parse().ok())
                .filter(|&d: &usize| d < self.gpus),
        )
    }

    pub(super) fn open_confirm(&mut self) {
        if !self.rows.iter().any(|r| r.on) {
            self.message = Some("no method queued: space on the Methods screen picks some".into());
            self.screen = Screen::Methods;
            return;
        }
        if self.running() {
            self.message = Some("fits are still running".into());
            return;
        }
        self.confirm = Some(self.plan());
        self.confirm_scroll = 0;
    }

    pub(super) fn confirm_key(&mut self, k: KeyEvent) {
        match k.code {
            KeyCode::Esc | KeyCode::Char('q') => self.confirm = None,
            _ if legume_tui::keys::list_delta(&k, true).is_some() => {
                let n = self.confirm_max.get() + 1;
                self.confirm_scroll = legume_tui::keys::list_move(&k, self.confirm_scroll, n, true)
                    .unwrap_or(self.confirm_scroll);
            }
            KeyCode::Char('c') => {
                let text = self.copy_text();
                let attached = legume_tui::current().is_none_or(legume_tui::Session::attached);
                self.message = Some(match legume_tui::keys::copy(&text, attached) {
                    Ok(how) => format!("commands copied ({how})"),
                    Err(e) => format!("cannot copy: {e}"),
                });
            }
            // Enter runs it, as does the key that opened it.
            _ if k.code == KeyCode::Enter || crate::tui::is_run_key(&k) => {
                let Some(planned) = self.confirm.take() else {
                    return;
                };
                if let Some(p) = planned.iter().find(|p| p.problem.is_some()) {
                    if let Some(flag) = &p.blamed {
                        self.go_to_flag(p.job.row, flag);
                    }
                    self.message = Some(format!(
                        "{}: {}",
                        p.job.method,
                        p.problem.clone().unwrap_or_default()
                    ));
                    return;
                }
                let jobs = planned.into_iter().map(|p| p.job).collect();
                self.queue = Some(Queue::start(jobs));
                self.clones_scroll = 0;
                self.screen = Screen::Run;
            }
            _ => {}
        }
    }

    /// Show `--flag` of `method` on the parameters screen, the advanced
    /// rows too when it is one of them.
    fn go_to_flag(&mut self, m: usize, flag: &str) {
        let Some(at) = self.rows[m].form.fields.iter().position(|f| f.long == flag) else {
            return;
        };
        self.show_params(m);
        self.advanced |= self.rows[m].form.fields[at].advanced;
        self.filter.clear();
        self.field_row = self.row_of(at).unwrap_or(0);
    }

    /// Where field `i` is listed on the parameters screen.
    pub(super) fn row_of(&self, i: usize) -> Option<usize> {
        self.visible().iter().position(|&j| j == i)
    }

    /// The queued commands as one would type them where senna run started.
    fn copy_text(&self) -> String {
        self.confirm
            .iter()
            .flatten()
            .map(|p| {
                let cmd = format!("{} {}", p.job.tool.name(), script::line(&p.job.command()));
                if p.job.dir == self.here {
                    cmd
                } else {
                    let to = script::relative(&p.job.dir, &self.here);
                    format!("(cd {} && {cmd})", script::quote(&to.to_string_lossy()))
                }
            })
            .collect::<Vec<_>>()
            .join("\n")
    }
}
