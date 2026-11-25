//! Terminal front end of `senna run`: data and batch files, methods,
//! their flags, a confirm popup with the exact commands, then the queue
//! running with its log.
//!
//! An optional `mung clones` step runs before the fits. mung is a separate
//! program, started as a child like lupin is from `senna view`; its flags
//! come from `mung describe clones`.

mod batches;
mod clones;
mod confirm;
mod data;
mod data_screen;
mod described;
mod draw;
mod editing;
mod form;
mod jobs;
mod params_screen;
mod run_screen;
mod script;
mod share;

use batches::Batch;
use data::Pair;
use data::Pick;
use form::{Field, Kind, Method};
use jobs::{Job, Queue, Tool};
use legume_tui::browse::Browser;
use legume_tui::Round;
use ratatui::crossterm::event::{
    Event, KeyCode, KeyEvent, KeyEventKind, KeyModifiers, MouseEventKind,
};
use std::path::{Path, PathBuf};
use std::sync::mpsc::{Receiver, Sender};
use std::time::{Duration, Instant};

/// The methods `senna run` sets up, in the order listed.
pub const METHODS: [&str; 10] = [
    "topic",
    "masked-topic",
    "masked-vae",
    "masked-sbp",
    "vae",
    "svd",
    "bge",
    "simba",
    "rbm",
    "ode",
];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Screen {
    Data,
    Methods,
    Params,
    Run,
}

impl Screen {
    const ALL: [Screen; 4] = [Screen::Data, Screen::Methods, Screen::Params, Screen::Run];

    fn title(self) -> &'static str {
        match self {
            Screen::Data => "Data",
            Screen::Methods => "Methods",
            Screen::Params => "Parameters",
            Screen::Run => "Run",
        }
    }
}

/// A method, whether it is queued, and where it writes.
struct Row {
    tool: Tool,
    /// Its name as listed: the method, or `mung clones`.
    label: String,
    form: Method,
    on: bool,
    /// The `--out` prefix: relative to where senna run started, or
    /// absolute.
    out: String,
    /// What its `--out` is named after under the output header.
    stem: &'static str,
    /// Whether `out` was typed by hand, so the header leaves it be.
    typed: bool,
}

impl Row {
    fn new(tool: Tool, form: Method, stem: &'static str) -> Self {
        let label = match tool {
            Tool::Senna => form.name.clone(),
            Tool::Mung => format!("{} {}", tool.name(), form.name),
        };
        Row {
            tool,
            label,
            form,
            on: false,
            out: stem.to_string(),
            stem,
            typed: false,
        }
    }
}

/// What a line being typed will become.
#[derive(Clone, Debug, PartialEq, Eq)]
enum Target {
    Out(usize),
    /// What every `--out` not typed by hand starts with.
    Header,
    /// Method row, field index.
    Field(usize, usize),
    /// A flag of the shared page, by name: in every queued fit that has it.
    Shared(String),
    Filter,
    /// The batch of data row `usize`, by name.
    BatchName(usize),
    /// A new name for label `.1` of data row `.0`'s label file.
    Label(usize, usize),
}

struct Editor {
    target: Target,
    text: String,
}

/// The queued fit that builds a pseudobulk partition others collapse on.
struct Builder {
    manifest: PathBuf,
    method: String,
}

/// One queued fit as the confirm popup shows it.
struct Planned {
    job: Job,
    /// What blocks it from running.
    problem: Option<String>,
    /// The flag clap blamed, to mark its row.
    blamed: Option<String>,
    /// Worth knowing, not blocking.
    warning: Option<String>,
}

pub(crate) struct App {
    cli: clap::Command,
    /// What every `--out` not typed by hand starts with: a name, or a
    /// folder when it ends in `/`.
    header: String,
    /// `mung` as described by itself, or why it is not there.
    mung: Result<clap::Command, String>,
    /// The GPUs fits on `--device cuda` are spread over, one lane each.
    gpus: usize,
    /// The programs the jobs start: senna itself, and mung.
    senna_program: PathBuf,
    mung_program: PathBuf,
    /// How far the clones popup is scrolled.
    clones_scroll: usize,
    /// The log line at the top of the Run screen's log, counted from the
    /// first line of the run; `None` follows the newest.
    log_top: Option<usize>,
    /// How many log lines the Run screen shows, as last drawn.
    log_rows: std::cell::Cell<usize>,
    here: PathBuf,
    screen: Screen,
    pairs: Vec<Pair>,
    pair_row: usize,
    browser: Option<Browser<Pick>>,
    /// Where the browser opens next.
    browse_dir: PathBuf,
    rows: Vec<Row>,
    method_row: usize,
    /// The method whose flags the parameters screen shows.
    param_method: usize,
    /// The parameters screen shows the flags the queued methods share,
    /// where a change sets every one of them.
    shared_page: bool,
    /// The flag a single-file browser fills when a file is chosen.
    browsing: Option<Target>,
    field_row: usize,
    advanced: bool,
    filter: String,
    editor: Option<Editor>,
    confirm: Option<Vec<Planned>>,
    confirm_scroll: usize,
    /// How far the confirm popup can scroll, as last drawn.
    confirm_max: std::cell::Cell<usize>,
    /// The method, command line and flag clap blamed, as last checked.
    blame: std::cell::RefCell<Option<Blame>>,
    /// Data files described on worker threads.
    described: (Sender<Read>, Receiver<Read>),
    /// The labels of a data row's label file, listed to rename: the row
    /// and the label under the cursor.
    relabel: Option<(usize, usize)>,
    /// Data files still being described.
    describing: usize,
    queue: Option<Queue>,
    message: Option<String>,
    quit: bool,
    /// The `?` page of every key: up, and how far it is scrolled; its
    /// furthest scroll as last drawn.
    help: bool,
    help_scroll: usize,
    help_max: std::cell::Cell<usize>,
    /// Manifests to open in `senna view` once the terminal is given back.
    view: Vec<PathBuf>,
    /// A choice a finished fit left open, put to the user on the Run screen.
    choice: Option<Choice>,
}

/// A choice a finished fit left open ([`senna::run_manifest::Decision`]):
/// the queue's job, the decision, and the option under the cursor.
struct Choice {
    job: usize,
    decision: senna::run_manifest::Decision,
    row: usize,
}

/// `argv` (a queued job's, its out given as `--out`) rerun as `out`, with
/// `args` (flags and their values) set: a flag it has already takes the
/// new value, any other is added.
fn rerun_argv(argv: &[String], out: &str, args: &[String]) -> Vec<String> {
    fn set(argv: &mut Vec<String>, flag: &str, value: &str) {
        match argv.iter().position(|w| w == flag) {
            Some(k) if k + 1 < argv.len() => argv[k + 1] = value.to_string(),
            _ => argv.extend([flag.to_string(), value.to_string()]),
        }
    }
    let mut argv = argv.to_vec();
    set(&mut argv, "--out", out);
    for pair in args.chunks(2) {
        if let [flag, value] = pair {
            set(&mut argv, flag, value);
        }
    }
    argv
}

/// How a fit given whole ended, as its job tells it: done with its
/// manifest, failed and why, or stopped.
fn fit_end(state: &jobs::State, manifest: &Path) -> legume_mcp::job::End {
    use legume_mcp::job::End;
    match state {
        jobs::State::Done => {
            let open = senna::run_manifest::RunManifest::load(manifest)
                .is_ok_and(|(m, _)| !m.decisions.is_empty());
            End::Done(if open {
                format!(
                    "{}; the fit left a choice to the user: view_open it for needs_input",
                    manifest.display()
                )
            } else {
                manifest.display().to_string()
            })
        }
        jobs::State::Failed(why) => End::Failed(why.clone()),
        jobs::State::Stopped | jobs::State::Waiting | jobs::State::Running => End::Cancelled,
    }
}

/// How far the queue's log has gone to the job: its lines up to here.
#[derive(Default)]
struct Sent(usize);

impl Sent {
    /// The lines of `shared`'s log not sent yet; those gone off its front
    /// unsent are lost.
    fn new_lines(&mut self, shared: &jobs::Shared) -> Vec<String> {
        let first = self.0.max(shared.dropped) - shared.dropped;
        self.0 = shared.dropped + shared.log.len();
        shared.log.iter().skip(first).cloned().collect()
    }
}

/// How often a fit given whole tells its job where its bars stand.
const SAY_STAGES_EVERY: Duration = Duration::from_secs(2);
/// A stage's label is cut to this many characters.
const STAGE_LABEL_MAX: usize = 120;

/// A fit's progress bar as its job's `job_status` shows it: what it counts
/// (the bar's message, else `last`, the fit's newest log line), its count,
/// and the times its frame drew. A spinner has no end and no time left.
fn stage_of(p: &legume_tui::child::Progress, last: &str) -> legume_mcp::job::Stage {
    let label = match p.what.trim() {
        "" => last.trim(),
        what => what,
    };
    let len = (p.len > 0).then_some(p.len);
    legume_mcp::job::Stage {
        label: label.chars().take(STAGE_LABEL_MAX).collect(),
        pos: p.pos,
        len,
        elapsed_s: p.elapsed.map_or(0, |d| d.as_secs()),
        eta_s: p.eta.filter(|_| len.is_some()).map(|d| d.as_secs()),
    }
}

/// How long a fit has to wrap up when senna run quits under it.
const STOP_GRACE: Duration = Duration::from_secs(10);

/// A data file and what reading it found.
type Read = (PathBuf, data::Described);

/// A method row, its command line, and the flag clap blamed in it.
type Blame = (usize, Vec<String>, Option<String>);

/// Where the queue stands, in a line: `fits running: 1 of 3 done`, with
/// failures and stops when there are any.
fn fits_status(states: &[jobs::State], finished: bool) -> String {
    use jobs::State;
    let count = |f: fn(&State) -> bool| states.iter().filter(|s| f(s)).count();
    let done = count(|s| *s == State::Done);
    let failed = count(|s| matches!(s, State::Failed(_)));
    let stopped = count(|s| *s == State::Stopped);
    let mut line = format!(
        "fits {}: {done} of {} done",
        if finished { "finished" } else { "running" },
        states.len()
    );
    if failed > 0 {
        line += &format!(", {failed} failed");
    }
    if stopped > 0 {
        line += &format!(", {stopped} stopped");
    }
    line
}

/// `senna view` on the runs `shown`, run by `senna` (this binary), as `v`
/// opens them once the screens are left; [`legume_tui::Session::run_child`]
/// runs it in this session as this one runs (plain too, when it was asked
/// to be).
fn view_command(mut senna: std::process::Command, shown: &[String]) -> std::process::Command {
    senna.arg("view").arg("-f").args(shown);
    senna
}

/// Run the screens; `cli` is senna's built command, the source of every
/// method's flags and the check of every command line.
pub fn run(cli: clap::Command, args: &super::RunArgs) -> anyhow::Result<()> {
    // First: hosted, this process only relays the terminal from here on,
    // and the program the host starts does the rest.
    let session = crate::tui::TOOL.session("run", &args.session())?;
    let start = args.dir.clone().unwrap_or_else(|| PathBuf::from("."));
    let mung = describe_mung(&mung_program());
    // A fit given whole (by `senna mcp`, say): senna run ends with it, and
    // its log and end go to its job, also when it cannot start.
    let fit = !args.fit.is_empty();
    let reporter = fit.then(legume_mcp::job::Reporter::from_env).flatten();
    let failed = |e: anyhow::Error| {
        if let Some(r) = &reporter {
            r.finish(legume_mcp::job::End::Failed(format!("{e:#}")));
        }
        e
    };
    let mut app = App::new(cli, mung, start).map_err(failed)?;
    let mut sent = Sent::default();
    let mut said_stages = Instant::now();
    let mut report = |sent: &mut Sent, q: &Queue| {
        let Some(r) = &reporter else { return };
        let now = Instant::now();
        let say_stages = now.duration_since(said_stages) >= SAY_STAGES_EVERY;
        // Taken off the lock, then written.
        let (lines, bars) = q
            .shared
            .lock()
            .map(|s| {
                let bars = say_stages.then(|| (s.progress.clone(), s.last.clone()));
                (sent.new_lines(&s), bars)
            })
            .unwrap_or_default();
        lines.iter().for_each(|l| r.line(l));
        if let Some((bars, last)) = bars {
            said_stages = now;
            let stages: Vec<_> = bars
                .iter()
                .zip(&last)
                .filter_map(|(p, last)| p.as_ref().map(|p| stage_of(p, last)))
                .collect();
            r.progress(&stages);
        }
    };
    let quiet = legume_tui::logs::quiet();
    // The mouse's buttons and wheel reported, never its movement, so a
    // moving pointer wakes nothing; shift+drag still selects text in most
    // terminals.
    let mut terminal = session
        .terminal_with(legume_tui::Modes::new().mouse(legume_tui::Mouse::Buttons))
        .map_err(|e| failed(e.into()))?;
    // Started once the screen is up: no fit runs that an error before it
    // would leave behind.
    if fit {
        app.start_fit(&args.fit).map_err(failed)?;
    }
    let result = (|| -> anyhow::Result<()> {
        while !app.quit {
            app.poll();
            if let Some(q) = &app.queue {
                report(&mut sent, q);
                // A fit given whole: done with it, done, without waiting
                // for a key.
                if fit && q.finished() {
                    break;
                }
            }
            // Detached, the fits are followed and nothing is drawn.
            session.draw(&mut terminal, |f| app.draw(f))?;
            session.status(&app.session_status());
            // Redraw often only while something changes on its own.
            let listing = app.browser.as_ref().is_some_and(|b| b.listing().is_some());
            let wanted = if app.running() || app.describing > 0 || listing {
                Duration::from_millis(200)
            } else {
                Duration::from_secs(60)
            };
            match session.poll(&mut terminal, wanted)? {
                Round::Event(Event::Key(k)) if k.kind != KeyEventKind::Release => app.key(k),
                // The wheel scrolls whatever list or page is in front, as
                // the arrow keys do.
                Round::Event(Event::Mouse(m)) => match m.kind {
                    MouseEventKind::ScrollUp => app.key(KeyEvent::from(KeyCode::Up)),
                    MouseEventKind::ScrollDown => app.key(KeyEvent::from(KeyCode::Down)),
                    _ => {}
                },
                // `senna kill` (SIGTERM) quits as ctrl+c does: the fits
                // running are stopped and wrapped up below.
                Round::Stopped => app.quit = true,
                _ => {}
            }
        }
        Ok(())
    })();
    if let Some(q) = &app.queue {
        // Wait for the worker, so no fit it was starting outlives us: the
        // fit running has a moment to wrap up, then it is killed.
        if !q.finished() {
            app.message = Some("stopping the fits…".into());
            let _ = session.draw(&mut terminal, |f| app.draw(f));
        }
        q.stop_within(STOP_GRACE);
        report(&mut sent, q);
    }
    if let (Some(r), Some(q)) = (&reporter, &app.queue) {
        // A fit done is done, whatever the screen ended with.
        let state = q.states().first().cloned().unwrap_or(jobs::State::Stopped);
        r.finish(match &result {
            Err(e) if state != jobs::State::Done => legume_mcp::job::End::Failed(format!("{e:#}")),
            _ => fit_end(&state, &q.jobs[0].result()),
        });
    }
    // An error it ends with stays drawn on a screen nobody is attached to,
    // and is said once by whoever looks next.
    if let Err(e) = &result {
        let _ = session.hold_error(&mut terminal, e);
    }
    session.restore();
    drop(quiet);
    result?;
    // Hosted, the summary is left for whoever looks next: the terminal
    // attached as the session ends, or the `senna sessions`, `attach` or
    // `kill` after a detached one.
    let summary = app.summary().join("\n");
    if !summary.is_empty() {
        if session.is_hosted() {
            session.leave(&summary);
        } else {
            println!("{summary}");
        }
    }
    // Not after a `senna kill`: it asked for the session to end.
    if !fit && !app.view.is_empty() && !session.stop_requested() {
        let shown: Vec<String> = app.view.iter().map(|p| app.shown(p)).collect();
        // This binary, found even when it was reinstalled while it ran.
        let senna = legume_mcp::own_command(std::iter::empty::<&str>())?;
        let mut view = view_command(senna, &shown);
        // In the same session: a `senna kill` reaches it through us.
        session.run_child(&mut view)?;
    }
    Ok(())
}

impl App {
    fn new(
        cli: clap::Command,
        mung: Result<clap::Command, String>,
        start: PathBuf,
    ) -> anyhow::Result<Self> {
        let here = std::env::current_dir()?;
        // `..` resolved, symlinks kept: the folders as the user knows them.
        let start = script::lexical(&here.join(start));
        // The clones step first: it runs before the fits.
        let mut rows = Vec::new();
        if let Ok(m) = &mung {
            rows.push(Row::new(Tool::Mung, Method::new(m, CLONES)?, "cnv"));
        }
        for m in METHODS {
            rows.push(Row::new(Tool::Senna, Method::new(&cli, m)?, m));
        }
        let mut app = App {
            cli,
            mung,
            gpus: jobs::gpus(),
            header: String::new(),
            senna_program: std::env::current_exe().unwrap_or_else(|_| PathBuf::from("senna")),
            mung_program: mung_program(),
            clones_scroll: 0,
            log_top: None,
            log_rows: std::cell::Cell::new(0),
            here,
            screen: Screen::Data,
            pairs: Vec::new(),
            pair_row: 0,
            browser: Some(Browser::open(start.clone(), Pick::Data, None)),
            browse_dir: start,
            rows,
            method_row: 0,
            param_method: 0,
            shared_page: false,
            browsing: None,
            field_row: 0,
            advanced: true,
            filter: String::new(),
            // The output header is asked first: every result is named
            // after it.
            editor: Some(Editor {
                target: Target::Header,
                text: String::new(),
            }),
            confirm: None,
            confirm_scroll: 0,
            confirm_max: std::cell::Cell::new(0),
            blame: std::cell::RefCell::new(None),
            described: std::sync::mpsc::channel(),
            relabel: None,
            describing: 0,
            queue: None,
            message: None,
            quit: false,
            help: false,
            help_scroll: 0,
            help_max: std::cell::Cell::new(0),
            view: Vec::new(),
            choice: None,
        };
        app.refresh_outs();
        Ok(app)
    }

    /// Take `text` as the output header, its folder made now: all of it
    /// when it ends in `/`, else the part before its last `/`. `~/`
    /// starts at home. A folder that is a file keeps the header as it was.
    fn set_header(&mut self, text: &str) {
        let header = crate::tui::home(text.trim());
        let folder = header_folder(&header);
        if !folder.is_empty() {
            // Without its last `/`, a file there reads as a file.
            let at = self.here.join(folder.trim_end_matches('/'));
            if at.exists() && !at.is_dir() {
                let shown = self.shown(&at);
                self.message = Some(format!("{shown} is a file, not a folder: header kept"));
                return;
            }
            let made = !at.exists();
            if let Err(e) = legume_numeric::matrix::common_io::mkdir(&at.to_string_lossy()) {
                self.message = Some(format!("cannot make {folder}: {e}; header kept"));
                return;
            }
            self.message = Some(if made {
                format!("made {folder} for this run's results")
            } else {
                format!("results go in {folder}")
            });
        }
        self.header = header;
        self.refresh_outs();
    }

    /// Every `--out` not typed by hand: under the output header, first
    /// free where the run writes.
    fn refresh_outs(&mut self) {
        for r in self.rows.iter_mut().filter(|r| !r.typed) {
            r.out = free_out(&self.here, &under(&self.header, r.stem));
        }
    }

    fn running(&self) -> bool {
        self.queue.as_ref().is_some_and(|q| !q.finished())
    }

    /// A fit given whole (`senna run -- METHOD DATA... ARGS`), as the queue
    /// runs it: from here, as it was given, its script beside its out.
    fn fit_job(&self, argv: &[String]) -> anyhow::Result<Job> {
        use anyhow::Context;
        let method = argv
            .first()
            .context("no method: senna run -- METHOD DATA... --out OUT")?;
        let row = self
            .rows
            .iter()
            .position(|r| r.tool == Tool::Senna && r.form.name == *method)
            .with_context(|| format!("senna has no method {method}"))?;
        let rest = &argv[1..];
        // `--out`'s value, wherever and however it is given.
        let out = rest
            .iter()
            .enumerate()
            .find_map(|(k, w)| match w.as_str() {
                "-o" | "--out" => rest.get(k + 1).cloned(),
                w => w.strip_prefix("--out=").map(String::from),
            })
            .context("no --out")?;
        // Run from here, as the fit was given: its paths read as they were
        // meant, its out named from here when it is under here, whole
        // otherwise, with its data files whole too.
        let out = match Path::new(&out).strip_prefix(&self.here) {
            Ok(under) => under.to_path_buf(),
            Err(_) => PathBuf::from(&out),
        };
        anyhow::ensure!(
            out.file_name().is_some(),
            "--out {} names no file",
            out.display()
        );
        let name = out.to_string_lossy().into_owned();
        let data = |w: &str| {
            if out.is_absolute() {
                self.here.join(w).to_string_lossy().into_owned()
            } else {
                w.to_string()
            }
        };
        let mut words = vec![method.clone()];
        let (mut in_data, mut rest) = (true, rest.iter());
        while let Some(w) = rest.next() {
            in_data &= !w.starts_with('-');
            match w.as_str() {
                "-o" | "--out" => {
                    rest.next();
                    words.extend(["--out".to_string(), name.clone()]);
                }
                w if w.starts_with("--out=") => words.extend(["--out".to_string(), name.clone()]),
                w if in_data => words.push(data(w)),
                w => words.push(w.to_string()),
            }
        }
        Ok(Job {
            tool: Tool::Senna,
            program: self.program_of(Tool::Senna),
            row,
            method: self.rows[row].label.clone(),
            dir: self.here.clone(),
            out: name,
            argv: words,
            labels: Vec::new(),
            clones: None,
            pb_from: None,
            lane: 0,
        })
    }

    /// Queue the fit given whole and show it running; senna run ends
    /// with it.
    fn start_fit(&mut self, argv: &[String]) -> anyhow::Result<()> {
        let job = self.fit_job(argv)?;
        self.queue = Some(Queue::start(vec![job]));
        self.screen = Screen::Run;
        Ok(())
    }

    /// One line for `senna sessions` and the resume picker.
    fn session_status(&self) -> String {
        match &self.queue {
            None => "setting up fits".into(),
            // Counted under the lock, the states not copied every round.
            Some(q) => match q.shared.lock() {
                Ok(s) => fits_status(&s.states, s.finished),
                Err(_) => fits_status(&[], true),
            },
        }
    }

    /// Screens reachable now: Run once a queue has started.
    fn screens(&self) -> Vec<Screen> {
        Screen::ALL
            .into_iter()
            .filter(|s| *s != Screen::Run || self.queue.is_some())
            .collect()
    }

    fn key(&mut self, k: KeyEvent) {
        let ctrl = k.modifiers.contains(KeyModifiers::CONTROL);
        if ctrl && k.code == KeyCode::Char('c') {
            self.quit = true;
            return;
        }
        self.message = None;
        if self.help {
            // The page scrolls; esc, q or ? closes it, other keys do nothing.
            match k.code {
                KeyCode::Esc | KeyCode::Char('q' | '?') => self.help = false,
                _ => {
                    let n = self.help_max.get() + 1;
                    if let Some(at) = legume_tui::keys::list_move(&k, self.help_scroll, n, true) {
                        self.help_scroll = at;
                    }
                }
            }
            return;
        }
        if self.editor.is_some() {
            self.editor_key(k);
        } else if self.browser.is_some() {
            self.browser_key(k);
        } else if self.confirm.is_some() {
            self.confirm_key(k);
        } else if self.choice.is_some() {
            self.choice_key(k);
        } else if self.relabel.is_some() {
            self.relabel_key(k);
        } else if self.asking() {
            self.clones_key(k);
        } else if !self.global_key(k) {
            match self.screen {
                Screen::Data => self.data_key(k),
                Screen::Methods => self.methods_key(k),
                Screen::Params => self.params_key(k),
                Screen::Run => self.run_key(k),
            }
        }
        self.settle();
    }

    /// Keep the parameters screen on a queued method and on a listed flag:
    /// methods are queued and screens reached by more than one key, and a
    /// reset can drop the advanced flag under the cursor.
    fn settle(&mut self) {
        let methods = self.param_methods();
        if !methods.contains(&self.param_method) {
            self.param_method = methods[0];
            self.field_row = 0;
        }
        if self.shared_page && self.shared_rows().len() < 2 {
            self.shared_page = false;
            self.field_row = 0;
        }
        self.field_row = self.field_row.min(self.visible().len().saturating_sub(1));
    }

    /// Switch to screen `s`. Arriving at the parameters with two fits or
    /// more queued opens the page of what they share.
    fn go_to(&mut self, s: Screen) {
        if s == Screen::Params && self.screen != s && self.shared_rows().len() > 1 {
            self.show_shared();
            self.message = Some("flags the queued fits share; ] for each fit's own".into());
        }
        self.screen = s;
    }

    /// Keys every screen shares. Returns whether `k` was one.
    fn global_key(&mut self, k: KeyEvent) -> bool {
        let screens = self.screens();
        let at = screens.iter().position(|s| *s == self.screen).unwrap_or(0);
        match k.code {
            KeyCode::Tab => self.go_to(screens[(at + 1) % screens.len()]),
            KeyCode::BackTab => self.go_to(screens[(at + screens.len() - 1) % screens.len()]),
            KeyCode::Char(c @ '1'..='4') => {
                if let Some(s) = screens.get(c as usize - '1' as usize) {
                    self.go_to(*s);
                }
            }
            _ if crate::tui::is_run_key(&k) => self.open_confirm(),
            KeyCode::Char('?') => {
                self.help = true;
                self.help_scroll = 0;
            }
            KeyCode::Char('q') => {
                if self.running() {
                    self.message =
                        Some("fits are running: s stops them, ctrl+c stops and quits".into());
                } else {
                    self.quit = true;
                }
            }
            _ => return false,
        }
        true
    }

    /// Take in what the workers found out about data files.
    fn poll(&mut self) {
        if let Some(b) = self.browser.as_mut() {
            b.poll();
        }
        self.read_choices();
        while let Ok((path, d)) = self.described.1.try_recv() {
            self.describing = self.describing.saturating_sub(1);
            if let Some(p) = self.pairs.iter_mut().find(|p| p.data == path) {
                p.described(d);
            }
        }
    }
}

/// The mung subcommand `senna run` offers.
const CLONES: &str = "clones";

/// The mung to start: `$SENNA_MUNG`, else `mung` on the PATH.
fn mung_program() -> PathBuf {
    std::env::var_os("SENNA_MUNG").map_or_else(|| PathBuf::from("mung"), PathBuf::from)
}

/// `mung clones` as clap would see it, read from `mung describe clones`;
/// why not, when that fails.
fn describe_mung(program: &Path) -> Result<clap::Command, String> {
    let name = program.display();
    let out = std::process::Command::new(program)
        .args(["describe", CLONES])
        .stdin(std::process::Stdio::null())
        .output()
        .map_err(|e| format!("cannot run {name} ({e}): install mung-cnv or set SENNA_MUNG"))?;
    if !out.status.success() {
        return Err(format!(
            "{name} cannot describe its flags: mung-cnv 0.2.7 or later needed"
        ));
    }
    described::Description::parse(&String::from_utf8_lossy(&out.stdout))
        .map(|d| d.command())
        .map_err(|e| format!("{name} describe: {e}"))
}

/// The folder part of an output header: all of it when it ends in `/`, else
/// what comes before its last `/`; empty for a bare name.
fn header_folder(header: &str) -> &str {
    header.rfind('/').map_or("", |at| &header[..=at])
}

/// `stem` under the output header `header`: in it when it is a folder
/// (ends in `/`) or ends in a separator, else after it and `_`.
fn under(header: &str, stem: &str) -> String {
    match header.chars().last() {
        None => stem.to_string(),
        Some('/' | '_' | '-' | '.') => format!("{header}{stem}"),
        Some(_) => format!("{header}_{stem}"),
    }
}

/// `{method}`, or `{method}-2`, … : the first prefix in `dir` with no
/// manifest or script yet.
fn free_out(dir: &Path, method: &str) -> String {
    let taken = |p: &str| {
        [
            Tool::Senna.result(),
            Tool::Mung.result(),
            "cmd.sh",
            "batches",
        ]
        .iter()
        .any(|end| dir.join(format!("{p}.{end}")).exists())
    };
    if !taken(method) {
        return method.to_string();
    }
    (2..)
        .map(|k| format!("{method}-{k}"))
        .find(|p| !taken(p))
        .unwrap_or_else(|| method.to_string())
}

#[cfg(test)]
#[path = "tests/app.rs"]
mod tests;
