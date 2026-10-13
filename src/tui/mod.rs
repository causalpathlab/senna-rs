//! What the terminal front ends, `senna view` and `senna run`, share: the
//! page's look and popups, a file browser, and the senna and lupin
//! processes they start and follow.

pub(crate) mod browse;
pub(crate) mod busy;
pub(crate) mod child;
pub(crate) mod style;

use std::path::Path;

/// Frames of the spinners the workspace draws.
pub(crate) const SPINNER: [char; 8] = ['⠁', '⠂', '⠄', '⡀', '⢀', '⠠', '⠐', '⠈'];

/// The key that runs what a screen set up, in both front ends: ctrl+r,
/// which every terminal passes on as it is.
pub(crate) const RUN_KEY: &str = "ctrl+r";

/// The answer of work running on a worker thread.
pub(crate) struct Pending<T>(pub(crate) std::sync::mpsc::Receiver<Result<T, String>>);

impl<T: Send + 'static> Pending<T> {
    pub(crate) fn spawn(work: impl FnOnce() -> Result<T, String> + Send + 'static) -> Self {
        let (tx, pending) = Self::pair();
        std::thread::spawn(move || {
            let _ = tx.send(work());
        });
        pending
    }

    /// A pending answer and where to send it, for a worker that answers
    /// more than one waiter.
    pub(crate) fn pair() -> (std::sync::mpsc::Sender<Result<T, String>>, Self) {
        let (tx, rx) = std::sync::mpsc::channel();
        (tx, Self(rx))
    }

    /// The answer once it has come; `stopped` when the worker died first.
    pub(crate) fn poll(&self, stopped: &str) -> Option<Result<T, String>> {
        match self.0.try_recv() {
            Ok(r) => Some(r),
            Err(std::sync::mpsc::TryRecvError::Empty) => None,
            Err(std::sync::mpsc::TryRecvError::Disconnected) => Some(Err(stopped.into())),
        }
    }
}

/// What an answer not come yet says ([`Loading::answer`]); an action that
/// got it is done again when the answer comes.
pub(crate) const LOADING: &str = "loading…";

/// What a [`Loading`] does when started.
type Work<T> = Box<dyn FnOnce() -> Result<T, String> + Send>;

/// Work started once on a worker thread and read by anyone holding it:
/// waited for where the answer is needed now ([`Self::get`]), looked at
/// where it is not ([`Self::try_get`]).
pub(crate) struct Loading<T> {
    work: std::sync::Mutex<Option<Work<T>>>,
    value: std::sync::Arc<std::sync::OnceLock<Result<T, String>>>,
    since: std::sync::OnceLock<std::time::Instant>,
}

impl<T: Send + Sync + 'static> Loading<T> {
    /// `work`, not started yet.
    pub(crate) fn new(work: impl FnOnce() -> Result<T, String> + Send + 'static) -> Self {
        Self {
            work: std::sync::Mutex::new(Some(Box::new(work))),
            value: std::sync::Arc::default(),
            since: std::sync::OnceLock::new(),
        }
    }

    /// An answer known already.
    #[cfg(test)]
    pub(crate) fn ready(value: Result<T, String>) -> Self {
        let done = Self::new(|| Err(String::new()));
        done.work.lock().map(|mut w| w.take()).ok();
        let _ = done.value.set(value);
        done
    }

    /// Start the work on a worker thread, unless it was started already.
    pub(crate) fn start(&self) {
        let Some(work) = self.work.lock().ok().and_then(|mut w| w.take()) else {
            return;
        };
        let _ = self.since.set(std::time::Instant::now());
        let value = self.value.clone();
        std::thread::spawn(move || {
            // A panic is an answer too, so a waiter never waits for ever.
            let out = std::panic::catch_unwind(std::panic::AssertUnwindSafe(work))
                .unwrap_or_else(|_| Err("the work stopped with a panic".into()));
            let _ = value.set(out);
        });
    }

    /// The answer, if it has come.
    pub(crate) fn try_get(&self) -> Option<&Result<T, String>> {
        self.value.get()
    }

    /// The answer: waited for when `wait`, else [`LOADING`] until it
    /// comes (the work started meanwhile).
    pub(crate) fn answer(&self, wait: bool) -> Result<&T, String> {
        let read = if wait {
            self.get()
        } else {
            self.start();
            self.try_get().ok_or_else(|| LOADING.to_string())?
        };
        read.as_ref().map_err(Clone::clone)
    }

    /// How long the work has been running, while it runs.
    pub(crate) fn busy(&self) -> Option<std::time::Duration> {
        self.value
            .get()
            .is_none()
            .then(|| self.since.get().map(std::time::Instant::elapsed))
            .flatten()
    }

    /// The answer, starting the work if need be and waiting for it.
    pub(crate) fn get(&self) -> &Result<T, String> {
        self.start();
        self.value.wait()
    }
}

/// Whether `k` is [`RUN_KEY`].
#[must_use]
pub(crate) fn is_run_key(k: &ratatui::crossterm::event::KeyEvent) -> bool {
    use ratatui::crossterm::event::{KeyCode, KeyModifiers};
    k.code == KeyCode::Char('r') && k.modifiers.contains(KeyModifiers::CONTROL)
}

/// The last component of `path`, or an empty string.
#[must_use]
pub fn name(path: &Path) -> String {
    path.file_name()
        .map_or_else(String::new, |n| n.to_string_lossy().into_owned())
}

/// `text` with a leading `~/` (or a lone `~`) as the home folder.
#[must_use]
pub fn home(text: &str) -> String {
    let rest = match text {
        "~" => "",
        _ => match text.strip_prefix("~/") {
            Some(rest) => rest,
            None => return text.to_string(),
        },
    };
    match std::env::var_os("HOME") {
        Some(h) => format!("{}/{rest}", Path::new(&h).display()),
        None => text.to_string(),
    }
}

/// Edit a one-line text field with key `k`: type, backspace, ctrl+u clears.
/// Returns whether it was an editing key.
pub fn edit_line(text: &mut String, k: &ratatui::crossterm::event::KeyEvent) -> bool {
    use ratatui::crossterm::event::{KeyCode, KeyModifiers};
    let ctrl = k.modifiers.contains(KeyModifiers::CONTROL);
    match k.code {
        KeyCode::Char('u') if ctrl => text.clear(),
        KeyCode::Backspace => {
            text.pop();
        }
        KeyCode::Char(c) if !ctrl => text.push(c),
        _ => return false,
    }
    true
}

/// Key hints `text`, its entries three spaces apart, broken into lines of at
/// most `width` between entries, so a narrow terminal cuts none in half.
#[must_use]
pub fn wrap_keys(text: &str, width: usize) -> Vec<String> {
    let mut lines = vec![String::new()];
    for entry in text.split("   ").filter(|e| !e.is_empty()) {
        let line = lines.last_mut().expect("a line");
        let room = width.saturating_sub(1);
        if !line.is_empty() && line.chars().count() + 3 + entry.chars().count() > room {
            lines.push(entry.to_string());
        } else {
            if !line.is_empty() {
                line.push_str("   ");
            }
            line.push_str(entry);
        }
    }
    lines
}

/// `text` broken between words into lines of at most `width` columns.
#[must_use]
pub fn wrap_words(text: &str, width: usize) -> Vec<String> {
    let mut lines = vec![String::new()];
    for word in text.split(' ') {
        let line = lines.last_mut().expect("a line");
        if !line.is_empty() && line.chars().count() + 1 + word.chars().count() > width.max(1) {
            lines.push(word.to_string());
        } else {
            if !line.is_empty() {
                line.push(' ');
            }
            line.push_str(word);
        }
    }
    lines
}

/// Rows `text` takes word-wrapped to `width` columns, as ratatui wraps a
/// paragraph: whole words to a row, a word wider than a row broken.
#[must_use]
pub fn wrapped_rows(text: &str, width: usize) -> usize {
    let width = width.max(1);
    let (mut rows, mut used) = (1, 0usize);
    for word in text.split(' ') {
        let n = word.chars().count();
        let need = if used == 0 { n } else { used + 1 + n };
        if need <= width {
            used = need;
        } else if n <= width {
            rows += 1;
            used = n;
        } else {
            // Broken across rows of its own.
            rows += usize::from(used > 0) + (n - 1) / width;
            used = n % width;
        }
    }
    rows
}

/// A key and what it does, as lines of at most `width` columns: the key in
/// a column `key_w` wide after `indent`, the words after it wrapped under
/// themselves.
#[must_use]
pub fn key_lines(indent: usize, key: &str, key_w: usize, what: &str, width: usize) -> Vec<String> {
    let lead = indent + key_w + 1;
    let pad = key_w.saturating_sub(key.chars().count());
    wrap_words(what, width.saturating_sub(lead).max(8))
        .into_iter()
        .enumerate()
        .map(|(i, l)| {
            if i == 0 {
                format!("{}{key}{} {l}", " ".repeat(indent), " ".repeat(pad))
            } else {
                format!("{}{l}", " ".repeat(lead))
            }
        })
        .collect()
}

/// Rows a page key moves in a list.
pub const PAGE: usize = 10;

/// How far a list key moves: ↑↓ or k/j by one, PgUp/PgDn by a page,
/// Home/End to either end (as far as any list goes). `None` when `k` moves
/// nothing, so the caller can give it its own meaning.
#[must_use]
pub fn list_delta(k: &ratatui::crossterm::event::KeyEvent) -> Option<i64> {
    use ratatui::crossterm::event::{KeyCode, KeyModifiers};
    if k.modifiers
        .intersects(KeyModifiers::CONTROL | KeyModifiers::ALT)
    {
        return None;
    }
    Some(match k.code {
        KeyCode::Up | KeyCode::Char('k') => -1,
        KeyCode::Down | KeyCode::Char('j') => 1,
        KeyCode::PageUp => -(PAGE as i64),
        KeyCode::PageDown => PAGE as i64,
        KeyCode::Home => i64::MIN / 2,
        KeyCode::End => i64::MAX / 2,
        _ => return None,
    })
}

/// The row [`list_delta`] leads to from `row` in a list of `len` rows.
#[must_use]
pub fn list_move(k: &ratatui::crossterm::event::KeyEvent, row: usize, len: usize) -> Option<usize> {
    let last = len.saturating_sub(1) as i64;
    list_delta(k).map(|d| (row as i64).saturating_add(d).clamp(0, last) as usize)
}

/// A path as shown: relative to the working directory when it is under it.
#[must_use]
pub fn shown(p: &Path) -> String {
    std::env::current_dir()
        .ok()
        .and_then(|cwd| p.strip_prefix(cwd).ok().map(Path::to_path_buf))
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or_else(|| p.to_path_buf())
        .to_string_lossy()
        .into_owned()
}

#[cfg(test)]
mod loading_tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

    /// A Loading that answers `v` once `gate` sends, counting its starts.
    fn gated(
        v: Result<u32, String>,
    ) -> (Loading<u32>, std::sync::mpsc::Sender<()>, Arc<AtomicUsize>) {
        let (gate, wait) = std::sync::mpsc::channel::<()>();
        let starts = Arc::new(AtomicUsize::new(0));
        let n = starts.clone();
        let l = Loading::new(move || {
            n.fetch_add(1, Ordering::SeqCst);
            let _ = wait.recv();
            v
        });
        (l, gate, starts)
    }

    #[test]
    fn a_loading_starts_once_and_answers_loading_until_done() {
        let (l, gate, starts) = gated(Ok(7));
        assert!(l.try_get().is_none() && l.busy().is_none(), "not started");
        assert_eq!(l.answer(false), Err(LOADING.to_string()));
        l.start();
        l.start();
        assert!(l.busy().is_some());
        gate.send(()).unwrap();
        assert_eq!(l.get(), &Ok(7));
        assert_eq!(l.answer(false), Ok(&7));
        assert!(l.busy().is_none());
        assert_eq!(starts.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn a_panic_in_the_work_is_an_answer() {
        let l: Loading<u32> = Loading::new(|| panic!("boom"));
        assert!(l.get().as_ref().is_err_and(|e| e.contains("panic")));
    }
}

#[cfg(test)]
mod layout_tests {
    use super::*;
    use ratatui::crossterm::event::{KeyCode, KeyEvent, KeyModifiers};

    #[test]
    fn rows_count_as_words_wrap() {
        assert_eq!(wrapped_rows("", 10), 1);
        assert_eq!(wrapped_rows("aaaa bbbb", 9), 1);
        // A word that does not fit the row goes whole to the next.
        assert_eq!(wrapped_rows("aaaa bbbbb", 9), 2);
        // Wider than a row: broken across rows.
        assert_eq!(wrapped_rows("aaaaaaaaaaaaaaaaaaaa", 8), 3);
    }

    #[test]
    fn hints_and_keys_wrap_whole() {
        let lines = wrap_keys("a one   b two   c three", 14);
        assert_eq!(lines, ["a one   b two", "c three"]);
        let lines = key_lines(2, "x", 4, "one two three four", 16);
        assert!(lines.iter().all(|l| l.chars().count() <= 16), "{lines:?}");
        assert!(lines[0].starts_with("  x    one"));
        assert!(lines[1].starts_with("       "));
    }

    #[test]
    fn list_keys_move_and_clamp() {
        let k = |c| KeyEvent::new(c, KeyModifiers::NONE);
        assert_eq!(list_move(&k(KeyCode::Char('j')), 0, 5), Some(1));
        assert_eq!(list_move(&k(KeyCode::Up), 0, 5), Some(0));
        assert_eq!(list_move(&k(KeyCode::End), 1, 5), Some(4));
        assert_eq!(list_move(&k(KeyCode::Home), 3, 5), Some(0));
        assert_eq!(list_move(&k(KeyCode::Char('x')), 3, 5), None);
        let ctrl_j = KeyEvent::new(KeyCode::Char('j'), KeyModifiers::CONTROL);
        assert_eq!(list_move(&ctrl_j, 0, 5), None);
    }
}
