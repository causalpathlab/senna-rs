//! The record of a run: `{out}.cmd.sh`, the exact command `senna run`
//! starts, as a script that refuses to run over an earlier result.
//!
//! Paths in the command are relative to the script's folder and the run
//! starts there, so the script and the run read the same files.

use super::jobs::Tool;
use std::io::Write;
use std::path::{Path, PathBuf};

/// `word` as one shell word: as it is when nothing in it is special, else
/// in single quotes.
#[must_use]
pub fn quote(word: &str) -> String {
    let plain = !word.is_empty()
        && word
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || "_@%+=:,./-".contains(c));
    if plain {
        word.to_string()
    } else {
        format!("'{}'", word.replace('\'', r"'\''"))
    }
}

/// A command line as one shell line.
#[must_use]
pub fn line(argv: &[String]) -> String {
    argv.iter().map(|w| quote(w)).collect::<Vec<_>>().join(" ")
}

/// `path` with `.` and `..` resolved: through symlinks as far as it
/// exists, so two paths into one folder compare alike, then by its words.
#[must_use]
pub fn normalize(path: &Path) -> PathBuf {
    let words = lexical(path);
    let mut head = words.clone();
    let mut tail = Vec::new();
    loop {
        if let Ok(p) = head.canonicalize() {
            return tail.iter().rev().fold(p, |p, w| p.join(w));
        }
        match (
            head.file_name().map(std::ffi::OsStr::to_os_string),
            head.parent(),
        ) {
            (Some(w), Some(up)) => {
                tail.push(w);
                head = up.to_path_buf();
            }
            _ => return words,
        }
    }
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
    let (real, base) = (normalize(path), normalize(base));
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

/// The command as the script and the confirm popup write it: the program
/// and method, each data file on its own line, then one flag and its
/// values per line. The value of `--out` is the script's `$out`.
#[must_use]
pub fn command_lines(argv: &[String], tool: Tool) -> Vec<String> {
    let mut lines: Vec<String> = vec![tool.word()];
    let mut in_flags = false;
    for (k, w) in argv.iter().enumerate() {
        let word = if k > 0 && argv[k - 1] == "--out" {
            "\"$out\"".to_string()
        } else {
            quote(w)
        };
        in_flags |= w.starts_with("--");
        let new_line = k > 0 && (!in_flags || w.starts_with("--"));
        match lines.last_mut() {
            Some(last) if !new_line => {
                last.push(' ');
                last.push_str(&word);
            }
            _ => lines.push(word),
        }
    }
    lines
}

/// The script for `argv` (without the program `tool`), whose `--out` is
/// `out`.
#[must_use]
pub fn text(argv: &[String], out: &str, tool: Tool) -> String {
    let result = format!("${{out}}.{}", tool.result());
    let mut s = String::new();
    s.push_str("#!/usr/bin/env bash\n");
    s.push_str(&format!(
        "# Made by `senna run` (senna {}). Run it again with: bash {}.cmd.sh\n",
        env!("CARGO_PKG_VERSION"),
        quote(out)
    ));
    s.push_str("set -euo pipefail\n");
    s.push_str("cd \"$(dirname \"$0\")\"\n");
    s.push_str(&format!("export RUST_LOG=\"${{RUST_LOG:-{LOG_LEVEL}}}\"\n"));
    s.push_str(&format!("out={}\n", quote(out)));
    s.push_str(&format!("if [ -f \"{result}\" ]; then\n"));
    s.push_str(&format!(
        "  echo \"{result} exists; move it away to run again\" >&2\n"
    ));
    s.push_str("  exit 1\n");
    s.push_str("fi\n");
    s.push_str(&command_lines(argv, tool).join(" \\\n  "));
    s.push('\n');
    s
}

/// Write the script for `argv` to `path`, never over an existing file.
pub fn write(path: &Path, out: &str, argv: &[String], tool: Tool) -> anyhow::Result<()> {
    let mut f = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .map_err(|e| anyhow::anyhow!("{}: {e}", path.display()))?;
    f.write_all(text(argv, out, tool).as_bytes())?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o755))?;
    }
    Ok(())
}

#[cfg(test)]
#[path = "tests/script.rs"]
mod tests;
