//! Terminal front end of `senna run`: data and batch files, methods,
//! their flags, a confirm popup with the exact commands, then the queue
//! running with its log.
//!
//! An optional `mung clones` step runs before the fits. mung is a separate
//! program, started as a child like lupin is from `senna view`; its flags
//! come from `mung describe clones`.

mod batches;
mod clones;
mod data;
mod described;
mod draw;
mod form;
mod jobs;
mod script;

use crate::tui::browse::{Browser, Outcome};
use batches::Batch;
use data::Pair;
use data::Pick;
use form::{Field, Kind, Method};
use jobs::{Job, Keep, Queue, Tool, CLONES_FLAG};
use ratatui::crossterm::event::{self, Event, KeyCode, KeyEvent, KeyEventKind, KeyModifiers};
use std::path::{Path, PathBuf};
use std::sync::mpsc::{Receiver, Sender};
use std::time::Duration;

/// The embedding methods `senna run` sets up, in the order listed.
pub const METHODS: [&str; 9] = [
    "topic",
    "masked-topic",
    "masked-vae",
    "masked-sbp",
    "vae",
    "svd",
    "bge",
    "simba",
    "gem",
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
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Target {
    Out(usize),
    /// What every `--out` not typed by hand starts with.
    Header,
    /// Method row, field index.
    Field(usize, usize),
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
    /// Manifests to open in `senna view` once the terminal is given back.
    view: Vec<PathBuf>,
}

/// A data file and what reading it found.
type Read = (PathBuf, data::Described);

/// A method row, its command line, and the flag clap blamed in it.
type Blame = (usize, Vec<String>, Option<String>);

/// Run the screens; `cli` is senna's built command, the source of every
/// method's flags and the check of every command line.
pub fn run(cli: clap::Command, start: PathBuf) -> anyhow::Result<()> {
    let mung = describe_mung(&mung_program());
    let mut app = App::new(cli, mung, start)?;
    let level = log::max_level();
    log::set_max_level(log::LevelFilter::Off);
    let mut terminal = ratatui::init();
    let result = (|| -> anyhow::Result<()> {
        while !app.quit {
            app.poll();
            terminal.draw(|f| app.draw(f))?;
            // Redraw often only while something changes on its own.
            let wait = if app.running() || app.describing > 0 {
                Duration::from_millis(200)
            } else {
                Duration::from_secs(60)
            };
            if event::poll(wait)? {
                if let Event::Key(k) = event::read()? {
                    if k.kind != KeyEventKind::Release {
                        app.key(k);
                    }
                }
            }
        }
        Ok(())
    })();
    if let Some(q) = &app.queue {
        // Wait for the worker, so no fit it was starting outlives us.
        q.stop();
        q.join();
    }
    ratatui::restore();
    log::set_max_level(level);
    result?;
    for line in app.summary() {
        println!("{line}");
    }
    if !app.view.is_empty() {
        let exe = std::env::current_exe()?;
        let shown: Vec<String> = app.view.iter().map(|p| app.shown(p)).collect();
        std::process::Command::new(exe)
            .arg("view")
            .arg("-f")
            .args(&shown)
            .status()?;
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
            field_row: 0,
            advanced: false,
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
            view: Vec::new(),
        };
        app.refresh_outs();
        Ok(app)
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
        if self.editor.is_some() {
            self.editor_key(k);
        } else if self.browser.is_some() {
            self.browser_key(k);
        } else if self.confirm.is_some() {
            self.confirm_key(k);
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
        self.field_row = self.field_row.min(self.visible().len().saturating_sub(1));
    }

    /// Keys every screen shares. Returns whether `k` was one.
    fn global_key(&mut self, k: KeyEvent) -> bool {
        let screens = self.screens();
        let at = screens.iter().position(|s| *s == self.screen).unwrap_or(0);
        match k.code {
            KeyCode::Tab => self.screen = screens[(at + 1) % screens.len()],
            KeyCode::BackTab => self.screen = screens[(at + screens.len() - 1) % screens.len()],
            KeyCode::Char(c @ '1'..='4') => {
                if let Some(s) = screens.get(c as usize - '1' as usize) {
                    self.screen = *s;
                }
            }
            KeyCode::Char('G') => self.open_confirm(),
            KeyCode::Char('q') => {
                if self.running() {
                    self.message =
                        Some("fits are running: s stops them, ctrl-c stops and quits".into());
                } else {
                    self.quit = true;
                }
            }
            _ => return false,
        }
        true
    }

    // ───────────── typing a line ─────────────

    fn edit(&mut self, target: Target, text: String) {
        self.editor = Some(Editor { target, text });
    }

    fn editor_key(&mut self, k: KeyEvent) {
        let Some(e) = self.editor.as_mut() else {
            return;
        };
        match k.code {
            KeyCode::Esc => {
                if e.target == Target::Filter {
                    self.filter.clear();
                    self.field_row = 0;
                }
                self.editor = None;
            }
            KeyCode::Enter => {
                let Editor { target, text } = self.editor.take().unwrap();
                match target {
                    Target::Out(i) => {
                        let t = text.trim();
                        let r = &mut self.rows[i];
                        if t.is_empty() {
                            // Back to the header's.
                            r.typed = false;
                            self.refresh_outs();
                        } else {
                            r.out = t.to_string();
                            r.typed = true;
                        }
                    }
                    Target::Header => {
                        self.header = text.trim().to_string();
                        self.refresh_outs();
                    }
                    Target::Field(m, f) => {
                        let field = &mut self.rows[m].form.fields[f];
                        if text.trim().is_empty() && !field.default.is_empty() {
                            // Leaving it out means clap's default, so say that.
                            field.reset();
                            self.message = Some(format!(
                                "--{} cannot be left unset: back to its default {}",
                                field.long, field.default
                            ));
                        } else {
                            field.value = text;
                        }
                    }
                    Target::Filter => {}
                    Target::BatchName(i) => {
                        let name = text.trim();
                        // No name, or senna's own: back to senna's rule.
                        self.pairs[i].batch = if name.is_empty()
                            || (self.pairs[i].tags.is_none()
                                && name == batches::own_name(&self.pairs[i].data))
                        {
                            Batch::Own
                        } else {
                            Batch::Named(name.to_string())
                        };
                    }
                    Target::Label(i, l) => {
                        if let Batch::Labels {
                            counts, renamed, ..
                        } = &mut self.pairs[i].batch
                        {
                            let label = counts[l].0.clone();
                            let name = text.trim();
                            if name.is_empty() || name == label {
                                renamed.remove(&label);
                            } else {
                                renamed.insert(label, name.to_string());
                            }
                        }
                    }
                }
            }
            KeyCode::Backspace | KeyCode::Char(_) => {
                if let KeyCode::Char(c) = k.code {
                    e.text.push(c);
                } else {
                    e.text.pop();
                }
                // The filter narrows the flags as it is typed.
                if e.target == Target::Filter {
                    self.filter.clone_from(&e.text);
                    self.field_row = 0;
                }
            }
            _ => {}
        }
    }

    // ───────────── the file browser ─────────────

    fn browse(&mut self, want: Pick) {
        self.browser = Some(Browser::open(self.browse_dir.clone(), want, None));
    }

    fn browser_key(&mut self, k: KeyEvent) {
        let Some(b) = self.browser.as_mut() else {
            return;
        };
        // Chosen files, or none when cancelled; either way it closes.
        let chosen = match b.key(k) {
            Outcome::Ignored | Outcome::Moved => return,
            Outcome::Cancelled => None,
            Outcome::Chosen(c) => Some(c.files()),
        };
        let Some(b) = self.browser.take() else { return };
        self.browse_dir = b.dir;
        let Some(paths) = chosen else { return };
        if b.want == Pick::Batch {
            self.take_batches(&paths);
        } else {
            self.take_data(paths);
        }
    }

    fn take_data(&mut self, paths: Vec<PathBuf>) {
        let mut added = 0;
        // Each folder's label files, listed once for all its data.
        let mut labels: std::collections::HashMap<PathBuf, Vec<PathBuf>> =
            std::collections::HashMap::new();
        for p in paths {
            if self.pairs.iter().any(|q| q.data == p) {
                continue;
            }
            let mut pair = Pair::pending(p);
            let dir = pair.data.parent().unwrap_or(Path::new(".")).to_path_buf();
            let near = labels
                .entry(dir)
                .or_insert_with_key(|d| data::label_files_in(d));
            if let Some(file) = data::beside(&pair.data, near) {
                pair.batch = Batch::labels(file).unwrap_or(Batch::Own);
            }
            // Opening a large backend takes a while: not here.
            let (tx, path) = (self.described.0.clone(), pair.data.clone());
            std::thread::spawn(move || {
                let d = data::describe(&path);
                let _ = tx.send((path, d));
            });
            self.describing += 1;
            self.pairs.push(pair);
            added += 1;
        }
        self.pair_row = self.pairs.len().saturating_sub(1);
        let found = self
            .pairs
            .iter()
            .filter(|p| matches!(p.batch, Batch::Labels { .. }))
            .count();
        self.message = Some(format!(
            "{added} data file{} added; {found} of {} with a label file",
            if added == 1 { "" } else { "s" },
            self.pairs.len()
        ));
    }

    /// One batch file goes to the data row under the cursor; several are
    /// paired with all rows by name.
    fn take_batches(&mut self, paths: &[PathBuf]) {
        let row = self.pair_row;
        let chosen: Vec<Option<PathBuf>> = if let [one] = paths {
            (0..self.pairs.len())
                .map(|i| (i == row).then(|| one.clone()))
                .collect()
        } else {
            let data: Vec<PathBuf> = self.pairs.iter().map(|p| p.data.clone()).collect();
            let (chosen, paired) = data::assign(&data, paths);
            let n = data.len();
            self.message = Some(match paired {
                data::Paired::ByName(_) => format!("{n} batch files paired by name"),
                data::Paired::InOrder => format!(
                    "no names match: {n} batch files paired in the order listed; check them"
                ),
                data::Paired::Partly(k) => format!(
                    "{k} of {n} data files matched a batch file by name; b sets the rest one by one"
                ),
            });
            chosen
        };
        for (p, file) in self.pairs.iter_mut().zip(chosen) {
            let Some(file) = file else { continue };
            match Batch::labels(file.clone()) {
                Ok(b) => p.batch = b,
                Err(e) => {
                    let why = format!("{}: {e}", crate::tui::name(&file));
                    self.message = Some(match self.message.take() {
                        Some(m) => format!("{m}; {why}"),
                        None => why,
                    });
                }
            }
        }
    }

    /// Keys in the list of a label file's labels: Enter renames one.
    fn relabel_key(&mut self, k: KeyEvent) {
        let Some((row, at)) = self.relabel else {
            return;
        };
        let n = match &self.pairs.get(row).map(|p| &p.batch) {
            Some(Batch::Labels { counts, .. }) => counts.len(),
            _ => 0,
        };
        let at = match k.code {
            KeyCode::Esc | KeyCode::Char('q') => {
                self.relabel = None;
                return;
            }
            KeyCode::Up => at.saturating_sub(1),
            KeyCode::Down => (at + 1).min(n.saturating_sub(1)),
            KeyCode::Enter if at < n => {
                let now = match &self.pairs[row].batch {
                    Batch::Labels {
                        counts, renamed, ..
                    } => renamed.get(&counts[at].0).unwrap_or(&counts[at].0).clone(),
                    _ => String::new(),
                };
                self.edit(Target::Label(row, at), now);
                at
            }
            _ => at,
        };
        self.relabel = Some((row, at));
    }

    /// Take in what the workers found out about data files.
    fn poll(&mut self) {
        while let Ok((path, d)) = self.described.1.try_recv() {
            self.describing = self.describing.saturating_sub(1);
            if let Some(p) = self.pairs.iter_mut().find(|p| p.data == path) {
                p.described(d);
            }
        }
    }

    // ───────────── data ─────────────

    fn data_key(&mut self, k: KeyEvent) {
        let last = self.pairs.len().saturating_sub(1);
        match k.code {
            KeyCode::Up => self.pair_row = self.pair_row.saturating_sub(1),
            KeyCode::Down => self.pair_row = (self.pair_row + 1).min(last),
            KeyCode::Char('a') => self.browse(Pick::Data),
            KeyCode::Char('b') if !self.pairs.is_empty() => self.browse(Pick::Batch),
            KeyCode::Char('x') => {
                if let Some(p) = self.pairs.get_mut(self.pair_row) {
                    p.batch = Batch::Own;
                }
            }
            KeyCode::Char('X') => self.pairs.iter_mut().for_each(|p| p.batch = Batch::Own),
            KeyCode::Char('n') => {
                if let Some(p) = self.pairs.get(self.pair_row) {
                    let now = match &p.batch {
                        Batch::Named(name) => name.clone(),
                        _ if p.tags.is_none() => batches::own_name(&p.data),
                        _ => String::new(),
                    };
                    self.edit(Target::BatchName(self.pair_row), now);
                }
            }
            KeyCode::Char('e') => {
                let labelled = self
                    .pairs
                    .get(self.pair_row)
                    .is_some_and(|p| matches!(p.batch, Batch::Labels { .. }));
                if labelled {
                    self.relabel = Some((self.pair_row, 0));
                } else {
                    self.message = Some("e renames the labels of a label file: b picks one".into());
                }
            }
            KeyCode::Char('d') | KeyCode::Delete if !self.pairs.is_empty() => {
                self.pairs.remove(self.pair_row);
                self.pair_row = self.pair_row.min(self.pairs.len().saturating_sub(1));
            }
            KeyCode::Char('K') if self.pair_row > 0 => {
                self.pairs.swap(self.pair_row, self.pair_row - 1);
                self.pair_row -= 1;
            }
            KeyCode::Char('J') if self.pair_row < last => {
                self.pairs.swap(self.pair_row, self.pair_row + 1);
                self.pair_row += 1;
            }
            KeyCode::Enter => self.screen = Screen::Methods,
            _ => {}
        }
    }

    // ───────────── methods ─────────────

    fn methods_key(&mut self, k: KeyEvent) {
        let last = self.rows.len() - 1;
        match k.code {
            KeyCode::Up => self.method_row = self.method_row.saturating_sub(1),
            KeyCode::Down => self.method_row = (self.method_row + 1).min(last),
            KeyCode::Char(' ') => {
                let r = &mut self.rows[self.method_row];
                r.on = !r.on;
            }
            KeyCode::Char('O') => {
                let header = self.header.clone();
                self.edit(Target::Header, header);
            }
            KeyCode::Char('o') => {
                let out = self.rows[self.method_row].out.clone();
                self.edit(Target::Out(self.method_row), out);
            }
            KeyCode::Enter | KeyCode::Right => {
                self.rows[self.method_row].on = true;
                self.show_params(self.method_row);
            }
            _ => {}
        }
    }

    fn show_params(&mut self, m: usize) {
        if self.param_method != m {
            self.field_row = 0;
        }
        self.param_method = m;
        self.screen = Screen::Params;
    }

    /// Methods the parameters screen steps through: the queued ones, or
    /// all when none is.
    fn param_methods(&self) -> Vec<usize> {
        let on: Vec<usize> = (0..self.rows.len()).filter(|&i| self.rows[i].on).collect();
        if on.is_empty() {
            (0..self.rows.len()).collect()
        } else {
            on
        }
    }

    // ───────────── parameters ─────────────

    /// Field indices of the method shown, as listed: advanced ones only
    /// when asked for or changed, narrowed by the filter.
    fn visible(&self) -> Vec<usize> {
        let f = self.filter.to_lowercase();
        self.rows[self.param_method]
            .form
            .fields
            .iter()
            .enumerate()
            .filter(|(_, x)| self.advanced || !x.advanced || !x.is_default())
            .filter(|(_, x)| f.is_empty() || x.long.contains(&f))
            .map(|(i, _)| i)
            .collect()
    }

    fn field(&mut self) -> Option<(usize, &mut Field)> {
        let i = *self.visible().get(self.field_row)?;
        Some((i, &mut self.rows[self.param_method].form.fields[i]))
    }

    fn params_key(&mut self, k: KeyEvent) {
        let n = self.visible().len();
        let last = n.saturating_sub(1);
        match k.code {
            KeyCode::Up => self.field_row = self.field_row.saturating_sub(1),
            KeyCode::Down => self.field_row = (self.field_row + 1).min(last),
            KeyCode::PageUp => self.field_row = self.field_row.saturating_sub(10),
            KeyCode::PageDown => self.field_row = (self.field_row + 10).min(last),
            KeyCode::Home => self.field_row = 0,
            KeyCode::End => self.field_row = last,
            KeyCode::Char('[' | ']') => {
                let ms = self.param_methods();
                let at = ms.iter().position(|&m| m == self.param_method).unwrap_or(0);
                let d = if k.code == KeyCode::Char(']') {
                    1
                } else {
                    ms.len() - 1
                };
                self.show_params(ms[(at + d) % ms.len()]);
            }
            KeyCode::Char('a') => {
                let on = self.field().map(|(i, _)| i);
                self.advanced = !self.advanced;
                // Stay on the same flag when it is still listed.
                self.field_row = on.and_then(|i| self.row_of(i)).unwrap_or(0);
            }
            KeyCode::Char('/') => {
                let text = self.filter.clone();
                self.edit(Target::Filter, text);
            }
            KeyCode::Esc if !self.filter.is_empty() => {
                self.filter.clear();
                self.field_row = 0;
            }
            KeyCode::Char('r') => {
                if let Some((_, f)) = self.field() {
                    f.reset();
                }
            }
            KeyCode::Char('R') => {
                for f in &mut self.rows[self.param_method].form.fields {
                    f.reset();
                }
                self.message = Some("every flag back to its default".into());
            }
            KeyCode::Char(' ') | KeyCode::Left | KeyCode::Right | KeyCode::Enter => {
                let m = self.param_method;
                let Some((i, f)) = self.field() else { return };
                match (&f.kind, k.code) {
                    (Kind::Flag { .. }, KeyCode::Char(' ') | KeyCode::Enter) => f.toggle(),
                    (Kind::Choice(_), KeyCode::Left) => f.cycle(-1),
                    (Kind::Choice(_), _) => f.cycle(1),
                    (Kind::Text, KeyCode::Enter) => {
                        let text = f.value.clone();
                        self.edit(Target::Field(m, i), text);
                    }
                    _ => {}
                }
            }
            _ => {}
        }
    }

    // ───────────── confirm ─────────────

    /// Where row `i` writes: its folder and the prefix in it.
    fn out_of(&self, i: usize) -> (PathBuf, String) {
        let abs = self.here.join(&self.rows[i].out);
        let dir = abs
            .parent()
            .map_or_else(|| self.here.clone(), script::normalize);
        let name = abs
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_default();
        (dir, name)
    }

    /// The command `tool`'s lines are checked against: a mung row exists
    /// only when mung described itself.
    fn command_of(&self, tool: Tool) -> &clap::Command {
        match tool {
            Tool::Senna => &self.cli,
            Tool::Mung => self
                .mung
                .as_ref()
                .expect("a mung row exists only when mung described itself"),
        }
    }

    fn program_of(&self, tool: Tool) -> PathBuf {
        match tool {
            Tool::Senna => self.senna_program.clone(),
            Tool::Mung => self.mung_program.clone(),
        }
    }

    /// The queued `mung clones` row, if one is.
    fn clones_row(&self) -> Option<usize> {
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
    fn plan(&self) -> Vec<Planned> {
        let mut outs: Vec<PathBuf> = Vec::new();
        let mut planned = Vec::new();
        let table = self.clones_row().map(|i| {
            let (dir, out) = self.out_of(i);
            dir.join(format!("{out}.{}", Tool::Mung.result()))
        });
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
            let warning = warning.or_else(|| self.shape_warning(r));
            let job = Job {
                tool: r.tool,
                program: self.program_of(r.tool),
                row: i,
                method: r.label.clone(),
                dir: dir.clone(),
                out: out.clone(),
                argv: r.form.argv(&data, &batch_files, &out),
                labels,
                clones,
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
                warning,
                job,
                problem,
                blamed,
            });
        }
        planned
    }

    /// Whether the data look like what `method` reads: gem wants
    /// `{gene}/count/…` rows, the others plain features.
    fn shape_warning(&self, r: &Row) -> Option<String> {
        if r.tool != Tool::Senna {
            return None;
        }
        let method = r.form.name.as_str();
        let known: Vec<bool> = self.pairs.iter().filter_map(|p| p.gene_counts).collect();
        if known.is_empty() {
            return None;
        }
        let gene = known.iter().filter(|g| **g).count();
        if method == "gem" && gene < known.len() {
            Some("gem reads {gene}/count/{spliced|unspliced} rows; some files have none".into())
        } else if method != "gem" && gene == known.len() {
            Some("these files hold gem's gene-count rows".into())
        } else {
            None
        }
    }

    fn open_confirm(&mut self) {
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

    fn confirm_key(&mut self, k: KeyEvent) {
        match k.code {
            KeyCode::Esc | KeyCode::Char('q') => self.confirm = None,
            KeyCode::Up => self.confirm_scroll = self.confirm_scroll.saturating_sub(1),
            KeyCode::Down => {
                self.confirm_scroll = (self.confirm_scroll + 1).min(self.confirm_max.get());
            }
            KeyCode::Char('c') => {
                let text = self.copy_text();
                self.message = Some(match copy(&text) {
                    Ok(how) => format!("commands copied ({how})"),
                    Err(e) => format!("cannot copy: {e}"),
                });
            }
            // The key that opened the review runs it.
            KeyCode::Char('G') => {
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
    fn row_of(&self, i: usize) -> Option<usize> {
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

    // ───────────── run ─────────────

    /// Whether the queue waits on the clones of `mung clones`.
    fn asking(&self) -> bool {
        self.queue.as_ref().is_some_and(Queue::asking)
    }

    /// Keys while the queue waits on the clones: keep them, run without,
    /// or stop.
    fn clones_key(&mut self, k: KeyEvent) {
        let keep = match k.code {
            KeyCode::Up => {
                self.clones_scroll = self.clones_scroll.saturating_sub(1);
                return;
            }
            KeyCode::Down => {
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

    fn run_key(&mut self, k: KeyEvent) {
        let page = self.log_rows.get().max(1);
        match k.code {
            KeyCode::Up => self.scroll_log(-1),
            KeyCode::Down => self.scroll_log(1),
            KeyCode::PageUp => self.scroll_log(-(page as isize)),
            KeyCode::PageDown => self.scroll_log(page as isize),
            KeyCode::Home => self.log_top = Some(0),
            KeyCode::End => self.log_top = None,
            KeyCode::Char('s') => {
                if let Some(q) = &self.queue {
                    q.stop();
                    self.message = Some("stopping".into());
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
    fn shown(&self, p: &Path) -> String {
        p.strip_prefix(&self.here)
            .ok()
            .filter(|r| !r.as_os_str().is_empty())
            .unwrap_or(p)
            .to_string_lossy()
            .into_owned()
    }

    fn summary(&self) -> Vec<String> {
        let Some(q) = &self.queue else {
            return Vec::new();
        };
        q.jobs
            .iter()
            .zip(q.states())
            .map(|(j, s)| {
                let script = j.script();
                let what = match s {
                    jobs::State::Done => "done".to_string(),
                    jobs::State::Failed(why) => format!("failed: {why}"),
                    jobs::State::Stopped => "stopped".to_string(),
                    jobs::State::Waiting | jobs::State::Running => "not finished".to_string(),
                };
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

/// Put `text` on the clipboard: `pbcopy` where there is one, else the
/// terminal's OSC 52. Says which.
fn copy(text: &str) -> anyhow::Result<&'static str> {
    use std::io::Write;
    if let Ok(mut c) = std::process::Command::new("pbcopy")
        .stdin(std::process::Stdio::piped())
        .spawn()
    {
        if let Some(mut stdin) = c.stdin.take() {
            stdin.write_all(text.as_bytes())?;
        }
        if c.wait()?.success() {
            return Ok("pbcopy");
        }
    }
    let mut out = std::io::stdout();
    write!(out, "\x1b]52;c;{}\x07", {
        use base64::Engine;
        base64::engine::general_purpose::STANDARD.encode(text)
    })?;
    out.flush()?;
    Ok("terminal clipboard")
}

#[cfg(test)]
#[path = "tests/app.rs"]
mod tests;
