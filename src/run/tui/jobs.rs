//! The queued fits, run one after another on a worker thread: each writes
//! its `{out}.cmd.sh`, then starts senna in the script's folder with the
//! same command, its log kept for the screen.

use std::collections::VecDeque;
use std::path::PathBuf;
use std::process::Command;
use std::sync::{Arc, Mutex};

use super::script;
use crate::tui::child::{run_one, Failed, Stopper};

/// Log lines kept for the screen.
const KEEP: usize = 2000;

/// One fit to run.
#[derive(Clone, Debug)]
pub struct Job {
    pub method: String,
    /// Where it runs and the script goes.
    pub dir: PathBuf,
    /// The `--out` prefix, a name in `dir`.
    pub out: String,
    /// The command line without the program, paths relative to `dir`.
    pub argv: Vec<String>,
    /// Batch label files to write before it starts.
    pub labels: Vec<super::batches::Written>,
}

impl Job {
    #[must_use]
    pub fn manifest(&self) -> PathBuf {
        self.dir.join(format!("{}.senna.json", self.out))
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
}

/// The queue, running.
pub struct Queue {
    pub jobs: Vec<Job>,
    pub shared: Arc<Mutex<Shared>>,
    stopper: Arc<Stopper>,
    worker: Mutex<Option<std::thread::JoinHandle<()>>>,
}

impl Queue {
    /// Start `jobs` in order with `program` (senna itself).
    pub fn start(jobs: Vec<Job>, program: PathBuf) -> Self {
        let shared = Arc::new(Mutex::new(Shared {
            states: vec![State::Waiting; jobs.len()],
            last: vec![String::new(); jobs.len()],
            ..Shared::default()
        }));
        let stopper = Arc::new(Stopper::default());
        let (s, st, js) = (shared.clone(), stopper.clone(), jobs.clone());
        let worker = std::thread::spawn(move || {
            for (i, job) in js.iter().enumerate() {
                let state = if st.is_stopped() {
                    State::Stopped
                } else {
                    set(&s, i, State::Running);
                    run_job(job, &program, i, &s, &st)
                };
                set(&s, i, state);
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
            .filter(|(j, s)| *s == State::Done && j.manifest().exists())
            .map(|(j, _)| j.manifest())
            .collect()
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
    if job.manifest().exists() {
        return State::Failed(format!("{} exists", job.manifest().display()));
    }
    for w in &job.labels {
        if let Err(e) = super::batches::write(w) {
            return State::Failed(format!("cannot write the batch labels: {e}"));
        }
    }
    if let Err(e) = script::write(&job.script(), &job.out, &job.argv) {
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
