//! The queued fits, run one after another in each lane: each writes its
//! `{out}.cmd.sh`, then starts senna (or mung) in the script's folder with
//! the same command, its log kept for the screen. Lanes run side by side,
//! one per GPU the fits are spread over; a fit collapsing on another's
//! partition waits for it whichever lane it is in.
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
use legume_tui::child::{run_one, Failed, Progress, Said, Stopper};

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

/// The method that fits time on two-track counts, not an embedding.
pub const TD_RBM: &str = "td-rbm";

/// The flag that hands the clones of `mung clones` to a senna fit, as
/// its form lists it.
pub const CLONES_FLAG: &str = "cnv-clones";
/// The flag a fit collapses on another's pseudobulk partition with.
pub const PB_FROM_FLAG: &str = "pb-from";

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

/// The GPUs fits can be spread over: those `CUDA_VISIBLE_DEVICES` names,
/// else those `nvidia-smi -L` lists; 1 when neither tells (no spreading).
#[must_use]
pub fn gpus() -> usize {
    if let Ok(visible) = std::env::var("CUDA_VISIBLE_DEVICES") {
        return visible
            .split(',')
            .filter(|d| !d.trim().is_empty())
            .count()
            .max(1);
    }
    Command::new("nvidia-smi")
        .arg("-L")
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map_or(1, |o| {
            String::from_utf8_lossy(&o.stdout)
                .lines()
                .filter(|l| l.starts_with("GPU "))
                .count()
                .max(1)
        })
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
    /// The manifest of a fit queued before it whose pseudobulk partition it
    /// collapses on, as passed: relative to `dir`.
    pub pb_from: Option<String>,
    /// The lane it runs in: its GPU when the fits are spread over several,
    /// else 0.
    pub lane: usize,
}

impl Job {
    /// What it leaves when done: `{out}.senna.json`, or mung's clones.
    #[must_use]
    pub fn result(&self) -> PathBuf {
        self.dir
            .join(format!("{}.{}", self.out, self.tool.result()))
    }

    /// The command it runs and records, without the program: `argv`, then
    /// the clones and the partition it collapses on.
    #[must_use]
    pub fn command(&self) -> Vec<String> {
        let mut c = self.argv.clone();
        if let Some(table) = &self.clones {
            c.extend([format!("--{CLONES_FLAG}"), table.clone()]);
        }
        if let Some(run) = &self.pb_from {
            c.extend([format!("--{PB_FROM_FLAG}"), run.clone()]);
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

/// Steps kept in a job's trend, at most drawn in its sparkline.
pub const TREND: usize = 64;

/// What the screen reads while the queue runs.
#[derive(Default)]
pub struct Shared {
    pub states: Vec<State>,
    pub log: VecDeque<String>,
    /// How many lines went off the front of `log`: the first kept is line
    /// `dropped` of all.
    pub dropped: usize,
    /// The last line each job wrote.
    pub last: Vec<String>,
    /// Where each job's progress bar stands, while one is drawn.
    pub progress: Vec<Option<Progress>>,
    /// The score of each epoch a job's bar has shown, the newest last.
    pub trend: Vec<Vec<f64>>,
    /// The bar step each job's newest trend entry belongs to.
    pub trend_pos: Vec<u64>,
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
    /// One per lane.
    stoppers: Vec<Arc<Stopper>>,
    answers: Sender<Keep>,
    workers: Mutex<Vec<std::thread::JoinHandle<()>>>,
}

impl Queue {
    /// Start `jobs` in order, each with its program, every lane on its own
    /// worker.
    pub fn start(jobs: Vec<Job>) -> Self {
        let shared = Arc::new(Mutex::new(Shared {
            states: vec![State::Waiting; jobs.len()],
            last: vec![String::new(); jobs.len()],
            progress: vec![None; jobs.len()],
            trend: vec![Vec::new(); jobs.len()],
            trend_pos: vec![0; jobs.len()],
            ..Shared::default()
        }));
        let (answers, answered) = std::sync::mpsc::channel();
        // A queue asking about clones runs as one lane: the fits after the
        // step wait on the answer.
        let lanes = if jobs.iter().any(Job::asks) {
            1
        } else {
            jobs.iter().map(|j| j.lane + 1).max().unwrap_or(1)
        };
        let stoppers: Vec<Arc<Stopper>> = (0..lanes).map(|_| Arc::default()).collect();
        let open = Arc::new(std::sync::atomic::AtomicUsize::new(lanes));
        let answered = Arc::new(Mutex::new(answered));
        let workers = (0..lanes)
            .map(|lane| {
                let (s, st, js) = (shared.clone(), stoppers[lane].clone(), jobs.clone());
                let (open, answered) = (open.clone(), answered.clone());
                std::thread::spawn(move || {
                    run_lane(&js, lane, lanes, &s, &st, &answered);
                    if open.fetch_sub(1, std::sync::atomic::Ordering::SeqCst) == 1 {
                        if let Ok(mut s) = s.lock() {
                            s.finished = true;
                        }
                    }
                })
            })
            .collect();
        Queue {
            jobs,
            shared,
            stoppers,
            answers,
            workers: Mutex::new(workers),
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

    /// Interrupt the fits running, as Ctrl+C would, and start none after
    /// them; asked again, kill them. Returns whether this stop kills.
    pub fn stop(&self) -> bool {
        // Every lane, so not `any`: it would stop at the first.
        let kills: Vec<bool> = self.stoppers.iter().map(|st| st.stop()).collect();
        let kill = kills.contains(&true);
        // A queue waiting on the clones stops waiting.
        self.answer(Keep::Stop);
        kill
    }

    /// Stop, give the fit running `grace` to wrap up, then kill it and
    /// wait for the worker.
    pub fn stop_within(&self, grace: std::time::Duration) {
        self.stop();
        let start = std::time::Instant::now();
        while start.elapsed() < grace && !self.worker_done() {
            std::thread::sleep(std::time::Duration::from_millis(50));
        }
        if !self.worker_done() {
            self.stoppers.iter().for_each(|st| st.kill());
        }
        self.join();
    }

    fn worker_done(&self) -> bool {
        self.workers
            .lock()
            .map_or(true, |w| w.iter().all(std::thread::JoinHandle::is_finished))
    }

    /// Wait for the workers to finish: after [`Queue::stop`], until the
    /// fits they may have been starting are killed too.
    pub fn join(&self) {
        let workers = self
            .workers
            .lock()
            .map(|mut w| std::mem::take(&mut *w))
            .unwrap_or_default();
        for w in workers {
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

/// Run lane `lane`'s jobs of `js` in order (every job when there is one
/// lane). A job collapsing on another's partition waits for it, and fails
/// when it did not finish.
fn run_lane(
    js: &[Job],
    lane: usize,
    lanes: usize,
    s: &Mutex<Shared>,
    st: &Stopper,
    answered: &Mutex<Receiver<Keep>>,
) {
    let mut js = js.to_vec();
    // Why the fits that wanted the clones cannot have them.
    let mut no_clones: Option<String> = None;
    for i in 0..js.len() {
        if lanes > 1 && js[i].lane != lane {
            continue;
        }
        let job = &js[i];
        // The fit whose partition it collapses on, once that one is through.
        let source = job.pb_from.as_ref().and_then(|run| {
            let run = super::script::normalize(&job.dir.join(run));
            js.iter()
                .position(|j| super::script::normalize(&j.result()) == run)
        });
        let source_state = source.map(|k| loop {
            let state = s
                .lock()
                .map(|s| s.states[k].clone())
                .unwrap_or(State::Stopped);
            if !matches!(state, State::Waiting | State::Running) || st.is_stopped() {
                break state;
            }
            std::thread::sleep(std::time::Duration::from_millis(200));
        });
        let state = if st.is_stopped() {
            State::Stopped
        } else if let (Some(_), Some(why)) = (&job.clones, &no_clones) {
            State::Failed(why.clone())
        } else if let Some(k) = source.filter(|_| source_state.as_ref() != Some(&State::Done)) {
            State::Failed(format!(
                "{} did not finish: no pseudobulks to collapse on",
                js[k].method
            ))
        } else {
            set(s, i, State::Running);
            run_job(job, i, lanes > 1, s, st)
        };
        let done = state == State::Done;
        set(s, i, state);
        if !job.asks() {
            continue;
        }
        if !done {
            no_clones = Some(format!("{} did not finish", job.method));
            continue;
        }
        let answer = answered
            .lock()
            .map_or(Keep::Stop, |a| ask(s, &a, told(&js[i])));
        match answer {
            Keep::Use => {}
            Keep::Without => js[i + 1..].iter_mut().for_each(|j| j.clones = None),
            Keep::Stop => {
                st.stop();
            }
        }
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
            s.dropped += 1;
        }
        s.last[i].clone_from(&line);
        s.log.push_back(line);
    }
}

fn run_job(job: &Job, i: usize, tag: bool, s: &Mutex<Shared>, stopper: &Stopper) -> State {
    if job.result().exists() {
        return State::Failed(format!("{} exists", job.result().display()));
    }
    if let Err(e) = legume_numeric::matrix::common_io::mkdir(&job.dir.to_string_lossy()) {
        return State::Failed(format!("cannot make {}: {e}", job.dir.display()));
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
    crate::tui::own_group(&mut command)
        .args(&argv)
        .current_dir(&job.dir);
    // The log level the script sets, so the run is the one it records.
    if std::env::var_os("RUST_LOG").is_none() {
        command.env("RUST_LOG", script::LOG_LEVEL);
    }
    // Side by side, every line says whose it is.
    let each = |said| match said {
        Said::Line(line) if tag => say(s, i, format!("[{}] {line}", job.method)),
        Said::Line(line) => say(s, i, line),
        Said::Progress(p) => {
            if let Ok(mut s) = s.lock() {
                // A new bar (none before, or it restarted or changed length)
                // starts a new trend: its scores are not on the last one's scale.
                let new_bar = s.progress[i]
                    .as_ref()
                    .is_none_or(|q| p.pos < q.pos || p.len != q.len);
                if new_bar {
                    s.trend[i].clear();
                }
                if let Some(v) = p.score() {
                    // One score per step: a later frame of the same step is
                    // the step's own score, replacing the one that lagged.
                    let st = &mut *s;
                    let t = &mut st.trend[i];
                    match t.last_mut() {
                        Some(last) if st.trend_pos[i] == p.pos => *last = v,
                        _ => {
                            if t.len() >= TREND {
                                t.remove(0);
                            }
                            t.push(v);
                            st.trend_pos[i] = p.pos;
                        }
                    }
                }
                s.progress[i] = Some(p);
            }
        }
        _ => {}
    };
    let ran = run_one(command, stopper, each);
    if let Ok(mut s) = s.lock() {
        s.progress[i] = None;
        s.trend[i].clear();
    }
    match ran {
        Ok(()) => State::Done,
        Err(Failed::Stopped) => State::Stopped,
        Err(Failed::Start(why) | Failed::Exit(why)) => State::Failed(why),
        Err(other) => State::Failed(format!("{other:?}")),
    }
}

#[cfg(test)]
#[path = "tests/jobs.rs"]
mod tests;
