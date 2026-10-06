//! A file browser, one folder at a time: its folders, and the files that
//! are wanted, each with a line about it.
//!
//! What is wanted, how a file is described and which one to start on are
//! the caller's, through [`Wanted`]; the browser only moves, narrows, marks
//! and draws. A `.zarr` store is a folder on disk but a file here: it is
//! never opened, only offered when the caller takes stores.

use super::style::{bold, first_row, hint, selected};
use ratatui::crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
use ratatui::style::Style;
use ratatui::text::{Line, Span};
use std::borrow::Cow;
use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

/// What a browser looks for, and how it speaks of it.
pub(crate) trait Wanted {
    /// What is shown about a listed file: its size, a run's kind, a
    /// panel's counts.
    type About: Clone;

    /// The popup's words.
    fn header(&self) -> Header;

    /// The file `name` at `path` with what to show of it, if it is wanted.
    fn file(&self, path: &Path, name: &str) -> Option<Self::About>;

    /// The `.zarr` store `name` at `path` as a file, when stores are wanted.
    fn store(&self, _path: &Path, _name: &str) -> Option<Self::About> {
        None
    }

    /// One line about a file.
    fn describe<'a>(&self, about: &'a Self::About) -> Cow<'a, str>;

    /// Why file `name` cannot be taken, though it is listed.
    fn refuse(&self, _name: &str, _about: &Self::About) -> Option<String> {
        None
    }

    /// The file to start on among those listed in `dir`.
    fn best(&self, _dir: &Path, _files: &[(&str, &Self::About)]) -> Option<String> {
        None
    }

    /// Whether space marks several files to take together.
    fn many(&self) -> bool {
        false
    }

    /// A key of the caller's own, such as an option it offers while
    /// browsing (its header says which). Returns whether it was one.
    fn key(&mut self, _k: &KeyEvent) -> bool {
        false
    }
}

/// What the popup says about what is wanted.
pub(crate) struct Header {
    pub title: String,
    /// Lines under the title.
    pub notes: Vec<String>,
    /// The files wanted, as in "no {what} here".
    pub what: &'static str,
    /// What a star by the starting file means; no star without it.
    pub star: Option<&'static str>,
    /// What enter does to a file: "open", "annotate".
    pub verb: &'static str,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum Entry<A> {
    Up,
    Dir(String),
    File(String, A),
}

impl<A> Entry<A> {
    pub fn name(&self) -> &str {
        match self {
            Entry::Up => "..",
            Entry::Dir(n) | Entry::File(n, _) => n,
        }
    }

    fn is_file(&self) -> bool {
        matches!(self, Entry::File(..))
    }
}

/// What a key did in the browser.
#[derive(Debug, PartialEq, Eq)]
pub(crate) enum Outcome {
    /// Not a browser key.
    Ignored,
    Moved,
    Cancelled,
    Chosen(Chosen),
}

/// Files chosen, never none: the one under the cursor, or, where several
/// can be taken, the marked ones.
#[derive(Debug, PartialEq, Eq)]
pub(crate) struct Chosen {
    first: PathBuf,
    rest: Vec<PathBuf>,
}

impl Chosen {
    fn one(path: PathBuf) -> Self {
        Chosen {
            first: path,
            rest: Vec::new(),
        }
    }

    /// The file, where one is taken at a time.
    #[must_use]
    pub fn file(self) -> PathBuf {
        debug_assert!(
            self.rest.is_empty(),
            "several files chosen where one is taken"
        );
        self.first
    }

    /// Every file taken, in order.
    #[must_use]
    pub fn files(self) -> Vec<PathBuf> {
        std::iter::once(self.first).chain(self.rest).collect()
    }
}

pub(crate) struct Browser<W: Wanted> {
    pub dir: PathBuf,
    pub want: W,
    entries: Vec<Entry<W::About>>,
    /// Typed letters narrow the list to names containing them; typed from
    /// a `/`, they are a path the browser follows (see [`Self::follow`]).
    pub filter: String,
    /// Position among the shown entries.
    pub row: usize,
    /// The file to start on, as the caller chose it.
    best: Option<String>,
    /// Files marked with space, in any folder.
    pub marked: BTreeSet<PathBuf>,
    /// Why the file just chosen was refused.
    pub refused: Option<String>,
}

impl<W: Wanted> Browser<W> {
    pub fn open(dir: PathBuf, want: W, select: Option<&str>) -> Self {
        let mut b = Browser {
            dir,
            want,
            entries: Vec::new(),
            filter: String::new(),
            row: 0,
            best: None,
            marked: BTreeSet::new(),
            refused: None,
        };
        b.read(select);
        b
    }

    /// Read the current folder; the cursor goes to `select` if it is
    /// listed, else to the best file, else to the first entry after `..`.
    pub fn read(&mut self, select: Option<&str>) {
        self.load();
        self.filter.clear();
        let want = select.map(str::to_string).or_else(|| self.best.clone());
        self.row = want
            .and_then(|w| self.shown().iter().position(|e| e.name() == w))
            .unwrap_or(usize::from(self.shown().len() > 1));
    }

    /// List the current folder and find its best file; the filter stays.
    fn load(&mut self) {
        self.entries = list_dir(&self.dir, &self.want);
        let files: Vec<(&str, &W::About)> = self
            .entries
            .iter()
            .filter_map(|e| match e {
                Entry::File(n, a) => Some((n.as_str(), a)),
                _ => None,
            })
            .collect();
        self.best = self.want.best(&self.dir, &files);
    }

    /// The typed path's folder (up to its last `/`) and the name begun
    /// after it, when the filter is a path.
    fn typed_path(&self) -> Option<(&str, &str)> {
        if !self.filter.starts_with('/') {
            return None;
        }
        let at = self.filter.rfind('/')? + 1;
        Some((&self.filter[..at], &self.filter[at..]))
    }

    /// Follow a typed path, which stays as typed: list its folder as soon
    /// as that folder exists, and the folder the name after it reaches
    /// once no other entry there begins with that name (`/data2` lists
    /// `/data2/` at once; `/data` beside `/data2` waits for a `/`).
    fn follow(&mut self) {
        let Some((folder, name)) = self.typed_path() else {
            return;
        };
        let folder: PathBuf = Path::new(folder).components().collect();
        if !folder.is_dir() {
            return;
        }
        let into = folder.join(name);
        let only = || {
            std::fs::read_dir(&folder).map_or(0, |d| {
                d.flatten()
                    .filter(|e| e.file_name().to_string_lossy().starts_with(name))
                    .count()
            }) == 1
        };
        let target = if !name.is_empty() && into.is_dir() && only() {
            into
        } else {
            folder
        };
        if target != self.dir {
            self.dir = target;
            self.load();
        }
    }

    /// The folder a typed path reached past what was typed: the one its
    /// name was completed to.
    fn completed(&self) -> bool {
        self.typed_path().is_some_and(|(folder, name)| {
            !name.is_empty()
                && Path::new(folder)
                    .join(name)
                    .components()
                    .collect::<PathBuf>()
                    == self.dir
        })
    }

    /// The entries the filter lets through, `..` always first. Hidden ones
    /// only when the filter starts with `.`.
    pub fn shown(&self) -> Vec<&Entry<W::About>> {
        // A typed path narrows by the name begun after its last `/`, once
        // the browser is in that path's folder.
        let f = match self.typed_path() {
            // In the folder its name completed to: all of it.
            Some(_) if self.completed() => String::new(),
            Some((_, name)) => name.to_lowercase(),
            None => self.filter.to_lowercase(),
        };
        let hidden = f.starts_with('.');
        self.entries
            .iter()
            .filter(|e| {
                matches!(e, Entry::Up)
                    || ((hidden || !e.name().starts_with('.'))
                        && e.name().to_lowercase().contains(&f))
            })
            .collect()
    }

    /// The entry under the cursor.
    pub fn current(&self) -> Option<&Entry<W::About>> {
        self.shown().get(self.row).copied()
    }

    pub fn go_up(&mut self) {
        let from = super::name(&self.dir);
        if let Some(parent) = self.dir.parent().map(Path::to_path_buf) {
            self.dir = parent;
            self.read(Some(&from));
        }
    }

    /// The file under the cursor, if it is on one.
    fn file_here(&self) -> Option<PathBuf> {
        let e = self.current().filter(|e| e.is_file())?;
        Some(self.dir.join(e.name()))
    }

    /// Why `e` cannot be taken, if it is a file the caller refuses.
    fn refusal(&self, e: &Entry<W::About>) -> Option<String> {
        match e {
            Entry::File(name, about) => self.want.refuse(name, about),
            _ => None,
        }
    }

    /// Take the file under the cursor, unless the caller refuses it: then
    /// say why and stay.
    fn take_here(&mut self) -> Outcome {
        let Some(Entry::File(name, about)) = self.current() else {
            return Outcome::Ignored;
        };
        if let Some(why) = self.want.refuse(name, about) {
            self.refused = Some(why);
            return Outcome::Moved;
        }
        Outcome::Chosen(Chosen::one(self.dir.join(name)))
    }

    /// A key: move, open a folder, narrow, mark, cancel, or choose.
    pub fn key(&mut self, k: KeyEvent) -> Outcome {
        // A refusal stays on screen until a key changes something.
        let said = self.refused.take();
        let out = self.handle(k);
        if out == Outcome::Ignored && self.refused.is_none() {
            self.refused = said;
        }
        out
    }

    fn handle(&mut self, k: KeyEvent) -> Outcome {
        if self.want.key(&k) {
            return Outcome::Moved;
        }
        if self.want.many() {
            if let Some(o) = self.mark_key(k) {
                return o;
            }
        }
        let last = self.shown().len().saturating_sub(1);
        match k.code {
            KeyCode::Esc => return Outcome::Cancelled,
            KeyCode::Up => self.row = self.row.saturating_sub(1),
            KeyCode::Down => self.row = (self.row + 1).min(last),
            KeyCode::PageUp => self.row = self.row.saturating_sub(10),
            KeyCode::PageDown => self.row = (self.row + 10).min(last),
            KeyCode::Home => self.row = 0,
            KeyCode::End => self.row = last,
            KeyCode::Left => self.go_up(),
            KeyCode::Backspace => {
                if self.filter.pop().is_none() {
                    self.go_up();
                } else {
                    self.follow();
                    self.row = usize::from(self.shown().len() > 1);
                }
            }
            KeyCode::Char('~') if self.filter.is_empty() => {
                if let Ok(home) = std::env::var("HOME") {
                    self.dir = PathBuf::from(home);
                    self.read(None);
                }
            }
            KeyCode::Enter | KeyCode::Right => match self.current() {
                None => return Outcome::Ignored,
                Some(Entry::Up) => self.go_up(),
                Some(Entry::Dir(name)) => {
                    let name = name.clone();
                    self.dir.push(name);
                    self.read(None);
                }
                Some(Entry::File(..)) if k.code == KeyCode::Enter => return self.take_here(),
                Some(Entry::File(..)) => return Outcome::Ignored,
            },
            KeyCode::Char(c) => {
                let on = self.current().map(|e| e.name().to_string());
                self.filter.push(c);
                self.follow();
                let shown = self.shown();
                // Stay where the cursor was if it still shows, else on the
                // best file, else on the first entry after `..`.
                let at = |name: &str| shown.iter().position(|e| e.name() == name);
                self.row = on
                    .as_deref()
                    .filter(|n| *n != "..")
                    .and_then(at)
                    .or_else(|| self.best.as_deref().and_then(at))
                    .unwrap_or(usize::from(shown.len() > 1));
            }
            _ => return Outcome::Ignored,
        }
        Outcome::Moved
    }

    /// Space marks or unmarks the file under the cursor and moves on;
    /// ctrl-a marks every file shown; enter on a file takes the marked
    /// ones, or this one when none is. On a folder enter still opens it, so
    /// files marked here and there can be taken together.
    fn mark_key(&mut self, k: KeyEvent) -> Option<Outcome> {
        let ctrl = k.modifiers.contains(KeyModifiers::CONTROL);
        match k.code {
            KeyCode::Char(' ') => {
                if let Some(why) = self.current().and_then(|e| self.refusal(e)) {
                    self.refused = Some(why);
                } else if let Some(path) = self.file_here() {
                    if !self.marked.remove(&path) {
                        self.marked.insert(path);
                    }
                    self.row = (self.row + 1).min(self.shown().len().saturating_sub(1));
                }
                Some(Outcome::Moved)
            }
            KeyCode::Char('a') if ctrl => {
                // Refused files are left unmarked, the first reason shown.
                let mut files = Vec::new();
                let mut refused = Vec::new();
                for e in self.shown().into_iter().filter(|e| e.is_file()) {
                    match self.refusal(e) {
                        Some(why) => refused.push(why),
                        None => files.push(self.dir.join(e.name())),
                    }
                }
                if let Some(first) = refused.first() {
                    self.refused = Some(match refused.len() {
                        1 => first.clone(),
                        n => format!("{n} files not marked; {first}"),
                    });
                }
                self.marked.extend(files);
                Some(Outcome::Moved)
            }
            KeyCode::Enter => {
                self.current().filter(|e| e.is_file())?;
                let mut marked = std::mem::take(&mut self.marked).into_iter();
                Some(match marked.next() {
                    Some(first) => Outcome::Chosen(Chosen {
                        first,
                        rest: marked.collect(),
                    }),
                    None => self.take_here(),
                })
            }
            _ => None,
        }
    }

    /// The lines of a popup at most `height` rows tall, borders included:
    /// what is left after the header and footer goes to the entries.
    pub fn lines(&self, height: usize) -> Vec<Line<'static>> {
        let h = self.want.header();
        let mut head = vec![Line::from(Span::styled(format!(" {}", h.title), bold()))];
        head.extend(
            h.notes
                .iter()
                .map(|n| Line::from(Span::styled(format!(" {n}"), hint()))),
        );
        head.push(Line::from(Span::styled(
            format!(" {}", super::shown(&self.dir)),
            hint(),
        )));
        if self.completed() {
            head.push(Line::from(format!(
                " go to “{}” → {}/",
                self.filter,
                super::shown(&self.dir)
            )));
        } else if self.typed_path().is_some() {
            head.push(Line::from(format!(" go to “{}”", self.filter)));
        } else if !self.filter.is_empty() {
            head.push(Line::from(format!(" names with “{}”", self.filter)));
        }
        head.push(Line::from(""));

        let shown = self.shown();
        let many = self.want.many();
        let mut foot = Vec::new();
        if !shown.iter().any(|e| e.is_file()) {
            foot.push(Line::from(Span::styled(
                format!("  no {} here", h.what),
                hint(),
            )));
        }
        foot.push(Line::from(""));
        if let Some(why) = &self.refused {
            foot.push(Line::from(Span::styled(format!(" {why}"), bold())));
        }
        if let (Some(s), Some(_)) = (h.star, &self.best) {
            foot.push(Line::from(Span::styled(s, hint())));
        }
        if many {
            if !self.marked.is_empty() {
                foot.push(Line::from(format!(" {} marked", self.marked.len())));
            }
            foot.push(Line::from(Span::styled(
                " space mark   ctrl-a mark all shown   enter take marked (or the one here)",
                hint(),
            )));
        }
        foot.push(Line::from(Span::styled(
            format!(
                " ↑ ↓ choose   enter open / {}   ← or backspace up   type to narrow   ~ home   esc cancel",
                h.verb
            ),
            hint(),
        )));

        let rows = height.saturating_sub(head.len() + foot.len() + 2).max(3);
        let width = shown
            .iter()
            .map(|e| e.name().chars().count() + 1)
            .max()
            .unwrap_or(0)
            .min(36);
        let mut out = head;
        for (i, e) in shown
            .iter()
            .enumerate()
            .skip(first_row(self.row, rows, shown.len()))
            .take(rows)
        {
            let text = match e {
                Entry::Up => " ../".to_string(),
                Entry::Dir(n) => format!(" {n}/"),
                Entry::File(n, about) => {
                    let mark = if many && self.marked.contains(&self.dir.join(n)) {
                        "●"
                    } else if h.star.is_some() && self.best.as_deref() == Some(n) {
                        "★"
                    } else {
                        " "
                    };
                    let about = self.want.describe(about);
                    if about.is_empty() {
                        format!("{mark}{n:<width$}")
                    } else {
                        format!("{mark}{n:<width$}  {about}")
                    }
                }
            };
            let style = if i == self.row {
                selected()
            } else if e.is_file() {
                Style::default()
            } else {
                hint()
            };
            out.push(Line::from(Span::styled(text, style)));
        }
        out.extend(foot);
        out
    }
}

/// Whether `name` is a backend senna reads (`.zarr`, `.zarr.zip`, `.h5`).
pub(crate) fn is_data(name: &str) -> bool {
    data_beans::hdf5_io::strip_backend_suffix(name) != name
}

/// A file's size, `12.3 MB`.
pub(crate) fn size_of(path: &Path) -> String {
    let bytes = std::fs::metadata(path).map_or(0, |m| m.len()) as f64;
    let (v, unit) = [("kB", 1e3), ("MB", 1e6), ("GB", 1e9)]
        .iter()
        .rev()
        .find(|(_, s)| bytes >= *s)
        .map_or((bytes, "B"), |(u, s)| (bytes / s, *u));
    if unit == "B" {
        format!("{v:.0} B")
    } else {
        format!("{v:.1} {unit}")
    }
}

/// Folders (not a `.zarr` store) and the wanted files, `..` first, folders
/// before files, each group by name. Hidden entries are kept; `shown`
/// lists them only on request.
pub(crate) fn list_dir<W: Wanted>(dir: &Path, want: &W) -> Vec<Entry<W::About>> {
    let mut dirs = Vec::new();
    let mut found = Vec::new();
    for e in std::fs::read_dir(dir).into_iter().flatten().flatten() {
        let name = e.file_name().to_string_lossy().into_owned();
        let path = e.path();
        let about = if !path.is_dir() {
            want.file(&path, &name)
        } else if name.ends_with(".zarr") {
            want.store(&path, &name)
        } else {
            dirs.push(Entry::Dir(name));
            continue;
        };
        found.extend(about.map(|a| Entry::File(name, a)));
    }
    dirs.sort_by(|a, b| a.name().cmp(b.name()));
    found.sort_by(|a, b| a.name().cmp(b.name()));
    let mut out = vec![Entry::Up];
    out.extend(dirs);
    out.extend(found);
    out
}

#[cfg(test)]
#[path = "tests/browse.rs"]
mod tests;
