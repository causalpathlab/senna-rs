//! A first annotation from the view: ask for a markers file, then run
//! `lupin annotate` on this run and open the round it writes.

use super::*;
use crate::view::decide;
use std::path::{Path, PathBuf};

impl App {
    /// Start asking for the markers file, filled in with the run's own when
    /// it records one, else a file nearby whose name mentions markers.
    pub(super) fn ask_markers(&mut self) {
        if self.relabeling.is_some() {
            self.message = Some("lupin is still busy with the last request".into());
            return;
        }
        if self.scene.review.is_some() {
            self.message = Some("leave relabel mode first (R)".into());
            return;
        }
        let guess = self.markers_guess().map_or_else(String::new, |p| shown(&p));
        self.markers_input = Some(guess);
    }

    fn markers_guess(&self) -> Option<PathBuf> {
        if let Some((m, dir)) = &self.scene.data.run {
            if let Some(p) = &m.annotate.markers {
                let p = senna::run_manifest::resolve(dir, p);
                if p.is_file() {
                    return Some(p);
                }
            }
        }
        // Nearby files whose names mention markers; the plainest name wins
        // (a base panel over the ones rounds write, such as `x.r1.markers.tsv`).
        let run_dir = self
            .from
            .parent()
            .map(Path::to_path_buf)
            .unwrap_or_default();
        let dirs = [
            run_dir.clone(),
            run_dir.parent().map(Path::to_path_buf).unwrap_or_default(),
            PathBuf::from("."),
        ];
        dirs.iter()
            .filter_map(|d| {
                std::fs::read_dir(if d.as_os_str().is_empty() {
                    Path::new(".")
                } else {
                    d
                })
                .ok()
            })
            .flatten()
            .filter_map(|e| Some(e.ok()?.path()))
            .filter(|p| {
                let name = p
                    .file_name()
                    .map_or_else(String::new, |n| n.to_string_lossy().to_lowercase());
                p.is_file()
                    && name.contains("marker")
                    && [".tsv", ".txt", ".csv"].iter().any(|x| name.ends_with(x))
            })
            .min_by_key(|p| (p.file_name().map_or(usize::MAX, |n| n.len()), p.clone()))
    }

    pub(super) fn markers_key(&mut self, k: KeyEvent) {
        let Some(input) = self.markers_input.as_mut() else {
            return;
        };
        match k.code {
            KeyCode::Esc => {
                self.markers_input = None;
                self.message = Some("annotation cancelled".into());
            }
            KeyCode::Backspace => {
                input.pop();
            }
            KeyCode::Tab => {
                if let Some(done) = complete_path(input) {
                    *input = done;
                }
            }
            KeyCode::Enter => {
                let path = PathBuf::from(expand_home(input.trim()));
                if !path.is_file() {
                    self.message = Some(format!("no file at {}", path.display()));
                    return;
                }
                self.markers_input = None;
                self.run_annotate(path);
            }
            KeyCode::Char(c) => input.push(c),
            _ => {}
        }
    }

    /// The status line while typing the markers file.
    pub(super) fn markers_line(&self) -> Option<String> {
        let input = self.markers_input.as_ref()?;
        let options = path_options(input);
        let hint = if options.len() > 1 {
            let names: Vec<&str> = options.iter().take(4).map(|(n, _)| n.as_str()).collect();
            format!("   tab: {}", names.join(" · "))
        } else {
            String::new()
        };
        Some(format!("markers file for lupin annotate: {input}▏{hint}"))
    }

    fn run_annotate(&mut self, markers: PathBuf) {
        let run = self
            .from
            .canonicalize()
            .unwrap_or_else(|_| self.from.clone());
        let out = decide::annotate_out(&run);
        let lupin = self.lupin.clone();
        let progress: std::sync::Arc<std::sync::Mutex<String>> = Default::default();
        let shared = progress.clone();
        let (tx, rx) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            let _ = tx.send(decide::annotate(&lupin, &run, &markers, &out, &shared));
        });
        self.relabeling = Some(Relabeling {
            job: RelabelJob::Annotate,
            started: std::time::Instant::now(),
            sent: Vec::new(),
            progress,
            done: rx,
        });
    }
}

/// `~/…` as a home path.
fn expand_home(s: &str) -> String {
    match (s.strip_prefix("~/"), std::env::var("HOME")) {
        (Some(rest), Ok(home)) => format!("{home}/{rest}"),
        _ => s.to_string(),
    }
}

/// A path as typed: relative to the working directory when it is under it.
fn shown(p: &Path) -> String {
    std::env::current_dir()
        .ok()
        .and_then(|cwd| p.strip_prefix(cwd).ok().map(Path::to_path_buf))
        .unwrap_or_else(|| p.to_path_buf())
        .to_string_lossy()
        .into_owned()
}

/// Entries of the typed path's directory that start with its last part:
/// (name, is a directory).
fn path_options(input: &str) -> Vec<(String, bool)> {
    let (dir, prefix) = match input.rfind('/') {
        Some(i) => (&input[..=i], &input[i + 1..]),
        None => ("", input),
    };
    let read = if dir.is_empty() {
        ".".to_string()
    } else {
        expand_home(dir)
    };
    let mut out: Vec<(String, bool)> = std::fs::read_dir(read)
        .map(|rd| {
            rd.filter_map(|e| {
                let e = e.ok()?;
                let name = e.file_name().to_string_lossy().into_owned();
                (name.starts_with(prefix) && !name.starts_with('.'))
                    .then(|| (name, e.path().is_dir()))
            })
            .collect()
        })
        .unwrap_or_default();
    out.sort();
    out
}

/// The typed path completed as far as its directory's entries agree.
fn complete_path(input: &str) -> Option<String> {
    let options = path_options(input);
    let dir = input.rfind('/').map_or("", |i| &input[..=i]);
    match options.as_slice() {
        [] => None,
        [(name, is_dir)] => Some(format!("{dir}{name}{}", if *is_dir { "/" } else { "" })),
        [(first, _), rest @ ..] => {
            let common = rest.iter().fold(first.len(), |n, (o, _)| {
                first
                    .bytes()
                    .zip(o.bytes())
                    .take(n)
                    .take_while(|(a, b)| a == b)
                    .count()
            });
            Some(format!("{dir}{}", &first[..common]))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_path_completes_as_far_as_its_entries_agree() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("markers_a.tsv"), "").unwrap();
        std::fs::write(dir.path().join("markers_b.tsv"), "").unwrap();
        std::fs::create_dir(dir.path().join("sub")).unwrap();
        let base = format!("{}/", dir.path().display());
        assert_eq!(
            complete_path(&format!("{base}ma")).unwrap(),
            format!("{base}markers_")
        );
        assert_eq!(
            complete_path(&format!("{base}markers_b")).unwrap(),
            format!("{base}markers_b.tsv")
        );
        assert_eq!(
            complete_path(&format!("{base}su")).unwrap(),
            format!("{base}sub/")
        );
        assert!(complete_path(&format!("{base}zz")).is_none());
    }
}
