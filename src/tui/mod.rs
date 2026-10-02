//! What the terminal front ends, `senna view` and `senna run`, share: the
//! page's look and popups, a file browser, and the senna and lupin
//! processes they start and follow.

pub(crate) mod browse;
pub(crate) mod child;
pub(crate) mod style;

use std::path::Path;

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
