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

/// Edit a one-line text field with key `k`: type, backspace, ctrl-u clears.
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
