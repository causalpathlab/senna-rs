//! The record of a run: `{out}.cmd.sh`, the exact command `senna run`
//! starts, as a script that refuses to run over an earlier result.
//!
//! Paths in the command are relative to the script's folder and the run
//! starts there, so the script and the run read the same files.

use super::jobs::Tool;
use legume_tui::form::script::{self, Script};
use std::path::{Path, PathBuf};

/// A word or a whole command line as the shell reads it.
pub use legume_tui::form::script::{line, quote};

/// `path` absolute, resolved as the system resolves it (links where they
/// are met, `..` after them), so two paths into one folder compare alike.
#[must_use]
pub fn resolved(path: &Path) -> PathBuf {
    senna::run_manifest::canonical_so_far(&senna::run_manifest::from_cwd(path))
}

/// `path` with `.` and `..` resolved by its words alone.
#[must_use]
pub fn lexical(path: &Path) -> PathBuf {
    senna::run_manifest::normalize(path)
}

/// More `..` than this and a path is written whole: from far away a
/// climb up to the root and back down only hides where the file is.
const MAX_UP: usize = 2;

/// `path` as written in a command run in the folder `base`: relative to
/// it when that stays near, else absolute. Both absolute.
#[must_use]
pub fn relative(path: &Path, base: &Path) -> PathBuf {
    let (real, base) = (resolved(path), resolved(base));
    let common = real
        .components()
        .zip(base.components())
        .take_while(|(x, y)| x == y)
        .count();
    if base.components().count() - common > MAX_UP {
        // Whole, as the user reached it, not through its symlinks.
        return lexical(path);
    }
    let out = senna::run_manifest::relative_to(&real, &base);
    if out.as_os_str().is_empty() {
        PathBuf::from(".")
    } else {
        out
    }
}

/// The log level a run is started with, in the script and from the
/// screen alike, unless `RUST_LOG` says otherwise.
pub const LOG_LEVEL: &str = "info";

/// The script of a run of `tool` whose `--out` is `out`, named `name`
/// and written beside its out: it runs from the folder `out` is relative
/// to, and refuses to run over the run's result.
fn script<'a>(out: &'a str, name: &'a str, tool: Tool) -> Script<'a> {
    const ENV: &[(&str, &str)] = &[("RUST_LOG", LOG_LEVEL)];
    // The head names the program and its version: mung's is not known
    // here, senna's is.
    let version = match tool {
        Tool::Senna => env!("CARGO_PKG_VERSION"),
        Tool::Mung => concat!("via senna ", env!("CARGO_PKG_VERSION")),
    };
    let s = Script::new(tool.name(), version, "senna run", name)
        .env(ENV)
        .head(1)
        .out(out)
        .result(tool.result());
    // An out above where the run started: the script runs from where it
    // is started.
    match script::depth(out) {
        Some(up) => s.up(up),
        None => s,
    }
}

/// The command as the script and the confirm popup write it: the program
/// and method, each data file on its own line, then one flag and its
/// values per line. The value of `--out`, `out`, is the script's `$out`.
#[must_use]
pub fn command_lines(argv: &[String], out: &str, tool: Tool) -> Vec<String> {
    script(out, "", tool).command_lines(argv)
}

/// Write the script for `argv` to `path`, never over an existing file.
pub fn write(path: &Path, out: &str, argv: &[String], tool: Tool) -> std::io::Result<()> {
    let name = format!("{out}.cmd.sh");
    script::write(path, &script(out, &name, tool), argv)
}

#[cfg(test)]
#[path = "tests/script.rs"]
mod tests;
