//! The queued fits, run one after another on a worker thread: each writes
//! its `{out}.cmd.sh`, then starts senna (or mung) in the script's folder
//! with the same command, its log kept for the screen.
//!
//! A `mung clones` step goes first. Once it is done the queue waits for
//! the user, who has seen its clones, to keep `--cnv-clones` on the fits
//! after it, run them without, or stop.

use std::collections::VecDeque;
use std::path::PathBuf;
use std::process::Command;
use std::sync::{Arc, Mutex};

use super::script;
use crate::tui::child::{run_one, Failed, Stopper};

/// Log lines kept for the screen.
const KEEP: usize = 2000;

/// The program a job starts.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Tool {
    Senna,
    Mung,
}

impl Tool {
    /// The program as the script names it, overridable from the shell.
    #[must_use]
    pub fn word(self) -> &'static str {
        match self {
            Tool::Senna => "\"${SENNA:-senna}\"",
            Tool::Mung => "\"${MUNG:-mung}\"",
        }
    }

    /// What a finished run leaves at `{out}.`, and the script refuses to
    /// run over.
    #[must_use]
    pub fn result(self) -> &'static str {
        match self {
            Tool::Senna => "senna.json",
            Tool::Mung => "clones.parquet",
        }
    }
}

/// The flag that hands the clones of `mung clones` to a senna fit.
pub const CLONES_FLAG: &str = "--cnv-clones";

/// After `mung clones`: what the fits after it do.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Keep {
    /// Run with `--cnv-clones`.
    Use,
    /// Run without.
    Without,
    /// Run none.
    Stop,
}

/// One fit to run.
#[derive(Clone, Debug)]
pub struct Job {
    pub tool: Tool,
    pub method: String,
    /// Where it runs and the script goes.
    pub dir: PathBuf,
    /// The `--out` prefix, a name in `dir`.
    pub out: String,
    /// The command line without the program, paths relative to `dir`.
    pub argv: Vec<String>,
    /// Batch label files to write before it starts.
    pub labels: Vec<super::batches::Written>,
    /// Whether `argv` passes the clones of the `mung clones` step queued
    /// before it.
    pub clones: bool,
}

impl Job {
    /// What it leaves when done: `{out}.senna.json`, or mung's clones.
    #[must_use]
    pub fn result(&self) -> PathBuf {
        self.dir
            .join(format!("{}.{}", self.out, self.tool.result()))
    }

    /// `argv` without the clones of the `mung clones` step.
    fn without_clones(&mut self) {
        if let Some(k) = self.argv.iter().position(|w| w == CLONES_FLAG) {
            self.argv.drain(k..(k + 2).min(self.argv.len()));
        }
        self.clones = false;
    }

    #[must_use]
    pub fn script(&self) -> PathBuf {
        self.dir.join(format!("{}.cmd.sh", self.out))
    }

    /// Where the batch label files it writes go.
    #[must_use]
    pub fn batches(&self) -> PathBuf {
        self.dir.join(format!("{}.batches", self.out))
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum State {
    Waiting,
    Running,
    Done,
    Failed(String),
    Stopped,
}

/// What the screen reads while the queue runs.
#[derive(Default)]
pub struct Shared {
    pub states: Vec<State>,
    pub log: VecDeque<String>,
    /// The last line each job wrote.
    pub last: Vec<String>,
    pub finished: bool,
    /// The `mung clones` step waiting for [`Queue::answer`].
    pub asking: Option<usize>,
    answer: Option<Keep>,
}

/// The queue, running.
pub struct Queue {
    pub jobs: Vec<Job>,
    pub shared: Arc<Mutex<Shared>>,
    stopper: Arc<Stopper>,
    worker: Mutex<Option<std::thread::JoinHandle<()>>>,
}

impl Queue {
    /// Start `jobs` in order: senna's with `senna` (senna itself), mung's
    /// with `mung`.
    pub fn start(jobs: Vec<Job>, senna: PathBuf, mung: PathBuf) -> Self {
        let shared = Arc::new(Mutex::new(Shared {
            states: vec![State::Waiting; jobs.len()],
            last: vec![String::new(); jobs.len()],
            ..Shared::default()
        }));
        let stopper = Arc::new(Stopper::default());
        let (s, st, js) = (shared.clone(), stopper.clone(), jobs.clone());
        let worker = std::thread::spawn(move || {
            let mut js = js;
            // Why the fits that wanted the clones cannot have them.
            let mut no_clones: Option<String> = None;
            for i in 0..js.len() {
                let job = js[i].clone();
                let state = if st.is_stopped() {
                    State::Stopped
                } else if let (true, Some(why)) = (job.clones, &no_clones) {
                    State::Failed(why.clone())
                } else {
                    set(&s, i, State::Running);
                    let program = match job.tool {
                        Tool::Senna => &senna,
                        Tool::Mung => &mung,
                    };
                    run_job(&job, program, i, &s, &st)
                };
                let done = state == State::Done;
                set(&s, i, state);
                if job.tool != Tool::Mung {
                    continue;
                }
                if !done {
                    no_clones = Some(format!("{} did not finish", job.method));
                    continue;
                }
                match ask(&s, &st, i) {
                    Keep::Use => {}
                    Keep::Without => js[i + 1..].iter_mut().for_each(Job::without_clones),
                    Keep::Stop => st.stop(),
                }
            }
            if let Ok(mut s) = s.lock() {
                s.finished = true;
            }
        });
        Queue {
            jobs,
            shared,
            stopper,
            worker: Mutex::new(Some(worker)),
        }
    }

    /// The `mung clones` step waiting for an answer, if one is.
    #[must_use]
    pub fn asking(&self) -> Option<usize> {
        self.shared.lock().ok().and_then(|s| s.asking)
    }

    /// Tell the queue waiting after `mung clones` what to do.
    pub fn answer(&self, keep: Keep) {
        if let Ok(mut s) = self.shared.lock() {
            s.answer = Some(keep);
        }
    }

    /// Kill the fit running and start none after it.
    pub fn stop(&self) {
        self.stopper.stop();
    }

    /// Wait for the worker to finish: after [`Queue::stop`], until the fit
    /// it may have been starting is killed too.
    pub fn join(&self) {
        let worker = self.worker.lock().ok().and_then(|mut w| w.take());
        if let Some(w) = worker {
            let _ = w.join();
        }
    }

    #[must_use]
    pub fn finished(&self) -> bool {
        self.shared.lock().map_or(true, |s| s.finished)
    }

    /// Each job's state now.
    #[must_use]
    pub fn states(&self) -> Vec<State> {
        self.shared
            .lock()
            .map(|s| s.states.clone())
            .unwrap_or_default()
    }

    /// Manifests of the fits that finished.
    #[must_use]
    pub fn done(&self) -> Vec<PathBuf> {
        self.jobs
            .iter()
            .zip(self.states())
            .filter(|(j, s)| j.tool == Tool::Senna && *s == State::Done && j.result().exists())
            .map(|(j, _)| j.result())
            .collect()
    }
}

/// Wait for the user's answer about the clones of job `i`; stopping the
/// queue answers it too.
fn ask(s: &Mutex<Shared>, stopper: &Stopper, i: usize) -> Keep {
    if let Ok(mut s) = s.lock() {
        s.asking = Some(i);
    }
    loop {
        let answer = if stopper.is_stopped() {
            Some(Keep::Stop)
        } else {
            s.lock().ok().and_then(|mut s| s.answer.take())
        };
        if let Some(a) = answer {
            if let Ok(mut s) = s.lock() {
                s.asking = None;
            }
            return a;
        }
        std::thread::sleep(std::time::Duration::from_millis(100));
    }
}

fn set(s: &Mutex<Shared>, i: usize, state: State) {
    if let Ok(mut s) = s.lock() {
        s.states[i] = state;
    }
}

fn say(s: &Mutex<Shared>, i: usize, line: String) {
    if let Ok(mut s) = s.lock() {
        if s.log.len() >= KEEP {
            s.log.pop_front();
        }
        s.last[i].clone_from(&line);
        s.log.push_back(line);
    }
}

fn run_job(
    job: &Job,
    program: &std::path::Path,
    i: usize,
    s: &Mutex<Shared>,
    stopper: &Stopper,
) -> State {
    if job.result().exists() {
        return State::Failed(format!("{} exists", job.result().display()));
    }
    for w in &job.labels {
        if let Err(e) = super::batches::write(w) {
            return State::Failed(format!("cannot write the batch labels: {e}"));
        }
    }
    if let Err(e) = script::write(&job.script(), &job.out, &job.argv, job.tool) {
        return State::Failed(format!("cannot write the script: {e}"));
    }
    say(
        s,
        i,
        format!("── {} · {}", job.method, crate::tui::shown(&job.script())),
    );
    let mut command = Command::new(program);
    command.args(&job.argv).current_dir(&job.dir);
    // The log level the script sets, so the run is the one it records.
    if std::env::var_os("RUST_LOG").is_none() {
        command.env("RUST_LOG", script::LOG_LEVEL);
    }
    match run_one(command, stopper, |line| say(s, i, line.to_string())) {
        Ok(()) => State::Done,
        Err(Failed::Stopped) => State::Stopped,
        Err(Failed::Start(why) | Failed::Exit(why)) => State::Failed(why),
    }
}

#[cfg(test)]
#[path = "tests/jobs.rs"]
mod tests;
