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
use std::sync::mpsc::{Receiver, Sender};
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
    /// The program's name.
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            Tool::Senna => "senna",
            Tool::Mung => "mung",
        }
    }

    /// The program as the script names it, overridable from the shell
    /// (`$SENNA`, `$MUNG`).
    #[must_use]
    pub fn word(self) -> String {
        let name = self.name();
        format!("\"${{{}:-{name}}}\"", name.to_uppercase())
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

/// The flag that hands the clones of `mung clones` to a senna fit, as
/// its form lists it.
pub const CLONES_FLAG: &str = "cnv-clones";

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
    /// The program started: senna itself, or mung.
    pub program: PathBuf,
    /// The row it comes from.
    pub row: usize,
    pub method: String,
    /// Where it runs and the script goes.
    pub dir: PathBuf,
    /// The `--out` prefix, a name in `dir`.
    pub out: String,
    /// The command line without the program, paths relative to `dir`.
    pub argv: Vec<String>,
    /// Batch label files to write before it starts.
    pub labels: Vec<super::batches::Written>,
    /// The clone table of the `mung clones` step queued before it, as
    /// passed: relative to `dir`.
    pub clones: Option<String>,
}

impl Job {
    /// What it leaves when done: `{out}.senna.json`, or mung's clones.
    #[must_use]
    pub fn result(&self) -> PathBuf {
        self.dir
            .join(format!("{}.{}", self.out, self.tool.result()))
    }

    /// The command it runs and records, without the program: `argv`, then
    /// the clones.
    #[must_use]
    pub fn command(&self) -> Vec<String> {
        let mut c = self.argv.clone();
        if let Some(table) = &self.clones {
            c.extend([format!("--{CLONES_FLAG}"), table.clone()]);
        }
        c
    }

    /// Whether the queue waits after it for the user to judge its clones.
    #[must_use]
    pub fn asks(&self) -> bool {
        self.tool == Tool::Mung
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
    /// What the `mung clones` step waiting for [`Queue::answer`] found.
    pub asking: Option<Told>,
}

/// The clones of a finished `mung clones` step: the summary's lines and a
/// word on keeping them, or why the table cannot be read.
pub type Told = Result<(Vec<String>, &'static str), String>;

/// The queue, running.
pub struct Queue {
    pub jobs: Vec<Job>,
    pub shared: Arc<Mutex<Shared>>,
    stopper: Arc<Stopper>,
    answers: Sender<Keep>,
    worker: Mutex<Option<std::thread::JoinHandle<()>>>,
}

impl Queue {
    /// Start `jobs` in order, each with its program.
    pub fn start(jobs: Vec<Job>) -> Self {
        let shared = Arc::new(Mutex::new(Shared {
            states: vec![State::Waiting; jobs.len()],
            last: vec![String::new(); jobs.len()],
            ..Shared::default()
        }));
        let stopper = Arc::new(Stopper::default());
        let (answers, answered) = std::sync::mpsc::channel();
        let (s, st, mut js) = (shared.clone(), stopper.clone(), jobs.clone());
        let worker = std::thread::spawn(move || {
            // Why the fits that wanted the clones cannot have them.
            let mut no_clones: Option<String> = None;
            for i in 0..js.len() {
                let job = &js[i];
                let state = if st.is_stopped() {
                    State::Stopped
                } else if let (Some(_), Some(why)) = (&job.clones, &no_clones) {
                    State::Failed(why.clone())
                } else {
                    set(&s, i, State::Running);
                    run_job(job, i, &s, &st)
                };
                let done = state == State::Done;
                set(&s, i, state);
                if !job.asks() {
                    continue;
                }
                if !done {
                    no_clones = Some(format!("{} did not finish", job.method));
                    continue;
                }
                match ask(&s, &answered, told(job)) {
                    Keep::Use => {}
                    Keep::Without => js[i + 1..].iter_mut().for_each(|j| j.clones = None),
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
            answers,
            worker: Mutex::new(Some(worker)),
        }
    }

    /// Whether a `mung clones` step waits for an answer.
    #[must_use]
    pub fn asking(&self) -> bool {
        self.shared.lock().is_ok_and(|s| s.asking.is_some())
    }

    /// Tell the queue waiting after `mung clones` what to do.
    pub fn answer(&self, keep: Keep) {
        let _ = self.answers.send(keep);
    }

    /// Kill the fit running and start none after it.
    pub fn stop(&self) {
        self.stopper.stop();
        // A queue waiting on the clones stops waiting.
        self.answer(Keep::Stop);
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

/// The clones of finished job `job`, told for the user to judge.
fn told(job: &Job) -> Told {
    senna::clone_strata::read(&job.result().to_string_lossy())
        .map(|t| {
            let s = super::clones::of(&t);
            (s.lines(), s.advice())
        })
        .map_err(|e| e.to_string())
}

/// Show what a `mung clones` step found and wait for the user's answer;
/// a queue gone answers stop.
fn ask(s: &Mutex<Shared>, answered: &Receiver<Keep>, told: Told) -> Keep {
    if let Ok(mut s) = s.lock() {
        s.asking = Some(told);
    }
    let keep = answered.recv().unwrap_or(Keep::Stop);
    if let Ok(mut s) = s.lock() {
        s.asking = None;
    }
    keep
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

fn run_job(job: &Job, i: usize, s: &Mutex<Shared>, stopper: &Stopper) -> State {
    if job.result().exists() {
        return State::Failed(format!("{} exists", job.result().display()));
    }
    for w in &job.labels {
        if let Err(e) = super::batches::write(w) {
            return State::Failed(format!("cannot write the batch labels: {e}"));
        }
    }
    let argv = job.command();
    if let Err(e) = script::write(&job.script(), &job.out, &argv, job.tool) {
        return State::Failed(format!("cannot write the script: {e}"));
    }
    say(
        s,
        i,
        format!("── {} · {}", job.method, crate::tui::shown(&job.script())),
    );
    let mut command = Command::new(&job.program);
    command.args(&argv).current_dir(&job.dir);
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
