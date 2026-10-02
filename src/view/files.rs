//! Small filesystem helpers shared by the view's reload, round and decision
//! code.

use std::path::{Path, PathBuf};
use std::time::SystemTime;

pub use crate::tui::name;

/// Modification time of `path`, if it can be read.
#[must_use]
pub fn modified(path: &Path) -> Option<SystemTime> {
    std::fs::metadata(path).and_then(|m| m.modified()).ok()
}

/// Whether two paths name the same existing file.
#[must_use]
pub fn same_file(a: &Path, b: &Path) -> bool {
    match (a.canonicalize(), b.canonicalize()) {
        (Ok(x), Ok(y)) => x == y,
        _ => false,
    }
}

/// Files next to `path` whose name ends with `suffix`, sorted.
#[must_use]
pub fn siblings(path: &Path, suffix: &str) -> Vec<PathBuf> {
    let dir = senna::run_manifest::manifest_dir(path);
    let mut out: Vec<PathBuf> = std::fs::read_dir(dir)
        .map(|rd| {
            rd.filter_map(Result::ok)
                .map(|e| e.path())
                .filter(|p| p.to_string_lossy().ends_with(suffix))
                .collect()
        })
        .unwrap_or_default();
    out.sort();
    out
}

/// The JSON document at `path`, if it reads and parses.
#[must_use]
pub fn read_json<T: serde::de::DeserializeOwned>(path: &Path) -> Option<T> {
    serde_json::from_str(&std::fs::read_to_string(path).ok()?).ok()
}

/// Whether `name` ends in `.pdf`, in any case.
#[must_use]
pub fn has_pdf(name: &str) -> bool {
    name.trim().to_ascii_lowercase().ends_with(".pdf")
}

/// `name`, trimmed, with `.pdf` added when it lacks it.
#[must_use]
pub fn with_pdf(name: &str) -> PathBuf {
    let name = name.trim();
    if has_pdf(name) {
        name.into()
    } else {
        format!("{name}.pdf").into()
    }
}
