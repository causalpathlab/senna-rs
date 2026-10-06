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

    /// Wait for the answer; `stopped` when the worker died first.
    pub(crate) fn wait(self, stopped: &str) -> Result<T, String> {
        self.0.recv().unwrap_or_else(|_| Err(stopped.into()))
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
