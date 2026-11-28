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

pub use senna::run_manifest::same_file;

/// A file as it was read: its path (absolute, so a watcher elsewhere opens
/// it too) and when it was last changed.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FileStamp {
    pub file: PathBuf,
    modified: Option<SystemTime>,
}

impl FileStamp {
    /// `path`'s stamp; taken before the file is read, so a rewrite in
    /// between is read again.
    pub fn of(path: &Path) -> std::io::Result<Self> {
        let modified = modified(path);
        Ok(Self {
            file: std::path::absolute(path)?,
            modified,
        })
    }

    /// Whether `path` is this file, unchanged since it was stamped.
    #[must_use]
    pub fn is(&self, path: &Path) -> bool {
        same_file(path, &self.file) && modified(path) == self.modified
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

/// `a.zarr`, or `a.zarr + 2 more` for several files.
#[must_use]
pub fn names(paths: &[PathBuf]) -> String {
    match paths {
        [] => String::new(),
        [one] => name(one),
        [first, rest @ ..] => format!("{} + {} more", name(first), rest.len()),
    }
}

/// The JSON document at `path`, if it reads and parses.
#[must_use]
pub fn read_json<T: serde::de::DeserializeOwned>(path: &Path) -> Option<T> {
    serde_json::from_str(&std::fs::read_to_string(path).ok()?).ok()
}
