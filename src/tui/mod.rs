//! What the terminal front ends, `senna view` and `senna run`, share: the
//! page's look and popups, and how they start the senna and mung
//! processes they follow (`legume_tui::child` runs them). The file
//! browser is `legume_tui::browse`.

pub(crate) mod style;

use std::path::Path;

/// The tool `senna run` and `senna view` sessions belong to: they are
/// named `senna-run-1`, `senna-view-2`, and `senna attach` finds them.
pub(crate) const TOOL: legume_tui::Tool = legume_tui::Tool::new("senna");

/// The detach and switch keys for a `?` page, each with what it does, in
/// lower case as the page shows keys; none when the screen runs plain.
pub(crate) fn session_keys() -> Vec<(String, String)> {
    legume_tui::help_keys()
        .into_iter()
        .map(|k| (k.key.to_lowercase(), k.long))
        .collect()
}

/// How this process's session starts its children; a plain one's outside
/// a session (in tests).
pub(crate) fn child_env() -> legume_tui::ChildEnv {
    legume_tui::current()
        .map(legume_tui::Session::child_env)
        .unwrap_or_default()
}

/// `command` set up as the session starts what
/// [`legume_tui::child::run_one`] runs (a fit, a recompute step, a lupin
/// annotation): in a process group of its own, so a stop reaches what it
/// starts too. A hangup of senna (its host gone, or the terminal of a
/// `--no-detach` run closed) is passed on to that group while it runs,
/// ending a fit there and then; a save ([`legume_tui::child::Stopper::saving`])
/// is spared it and finishes.
pub(crate) fn own_group(command: &mut std::process::Command) -> &mut std::process::Command {
    child_env().apply(command, legume_tui::Group::Own)
}

/// The key that runs what a screen set up, in both front ends, and
/// whether a key is it.
pub(crate) use legume_tui::{is_run_key, RUN_KEY};

/// The command at `path` under `cli` (`["layout", "tsne"]` for a nested
/// one), as both front ends find senna's commands; `cli` built, so its
/// global flags are in it.
#[must_use]
pub(crate) fn subcommand<'a>(cli: &'a clap::Command, path: &[&str]) -> Option<&'a clap::Command> {
    path.iter()
        .try_fold(cli, |at, word| at.find_subcommand(word))
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

/// A path as shown: relative to the working directory when it is under it.
pub use legume_tui::browse::shown;

/// Whether `name` is a backend senna reads (`.zarr`, `.zarr.zip`, `.h5`).
pub(crate) fn is_data(name: &str) -> bool {
    data_beans::hdf5_io::strip_backend_suffix(name) != name
}

#[cfg(test)]
mod session_tests {
    use super::*;

    #[test]
    fn a_plain_screen_is_always_attached_and_never_hosted() {
        let session = legume_tui::Session::plain(&TOOL);
        assert!(!session.is_hosted() && !session.offers_keys());
        assert!(session.attached() && !session.stop_requested());
        assert_eq!(session.name(), None);
    }
}
