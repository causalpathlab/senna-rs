//! Path conventions: manifest-relative paths, normalisation, and where a
//! run's manifest lives.

use super::RunManifest;
use std::fs;
use std::path::{Path, PathBuf};

/// Inverse of [`resolve`]: turn a path given on the command line (relative
/// to the working directory) into the manifest-relative form for storage.
/// Always relative, with `..` for a path outside the manifest's directory
/// (an input file elsewhere, or `-o /tmp/foo` while the manifest is in
/// `~/work/`): an absolute path names this machine's tree and is wrong
/// wherever the run is copied. Both sides are canonicalized (as far as they
/// exist) so symlinks and `./`-anchored paths agree.
#[must_use]
pub fn rel_to_manifest(manifest_dir: &Path, written_path: &str) -> String {
    match std::env::current_dir() {
        Ok(cwd) => rel_to_manifest_from(&cwd, manifest_dir, written_path),
        Err(_) => written_path.to_string(),
    }
}

/// [`rel_to_manifest`] with the working directory given explicitly.
pub(super) fn rel_to_manifest_from(cwd: &Path, manifest_dir: &Path, written_path: &str) -> String {
    let manifest_abs = canonical_so_far(&cwd.join(manifest_dir));
    let written_abs = canonical_so_far(&cwd.join(written_path));
    relative_to(&written_abs, &manifest_abs)
        .to_string_lossy()
        .into_owned()
}

/// `p` resolved as the system resolves it, one part at a time: a link is
/// followed where it is met (even to a folder not made yet), `..` steps
/// back from what is resolved so far, and a part not there yet is taken as
/// written. For a path that exists this is its canonical path; a file not
/// written yet still gets its folder's real path. Folding `..` before
/// following links would put `link/../x` beside the link, not where the
/// link leads.
#[must_use]
pub fn canonical_so_far(p: &Path) -> PathBuf {
    use std::collections::VecDeque;
    use std::ffi::OsString;
    use std::path::Component;
    /// Links followed before a loop is given up on, as the system does.
    const MAX_LINKS: usize = 40;
    /// A part of a path, owned: a root (or drive) to start from, `..`, or
    /// a name.
    enum Part {
        Root(OsString),
        Up,
        Name(OsString),
    }
    let parts = |p: &Path| -> Vec<Part> {
        p.components()
            .filter_map(|c| match c {
                Component::CurDir => None,
                Component::ParentDir => Some(Part::Up),
                Component::Normal(n) => Some(Part::Name(n.to_os_string())),
                c => Some(Part::Root(c.as_os_str().to_os_string())),
            })
            .collect()
    };
    let mut todo: VecDeque<Part> = parts(p).into();
    let mut out = PathBuf::new();
    let mut links = 0;
    while let Some(part) = todo.pop_front() {
        match part {
            Part::Root(root) => out.push(root),
            Part::Up => step_back(&mut out),
            Part::Name(name) => {
                let next = out.join(&name);
                match fs::read_link(&next) {
                    Ok(target) if links < MAX_LINKS => {
                        links += 1;
                        let target = unverbatim(target);
                        // An absolute target starts again from its root; a
                        // relative one goes on from the link's folder.
                        if target.is_absolute() {
                            out = PathBuf::new();
                        }
                        for p in parts(&target).into_iter().rev() {
                            todo.push_front(p);
                        }
                    }
                    _ => out = next,
                }
            }
        }
    }
    out
}

/// `out` one `..` further: a name before it is cancelled; after nothing, a
/// root or another `..`, the `..` stays (or, at a root, goes).
fn step_back(out: &mut PathBuf) {
    use std::path::Component;
    match out.components().next_back() {
        Some(Component::Normal(_)) => {
            out.pop();
        }
        Some(Component::RootDir | Component::Prefix(_)) => {}
        _ => out.push(".."),
    }
}

/// A link's target as the rest of the path spells it: on Windows without
/// its verbatim prefix, which the working directory never has (`\\?\UNC\`
/// for a share becomes `\\`; `\\?\` before a drive goes).
fn unverbatim(target: PathBuf) -> PathBuf {
    #[cfg(windows)]
    if let Some(t) = target.to_str() {
        if let Some(share) = t.strip_prefix(r"\\?\UNC\") {
            return PathBuf::from(format!(r"\\{share}"));
        }
        if let Some(plain) = t.strip_prefix(r"\\?\") {
            return PathBuf::from(plain);
        }
    }
    target
}

/// `p` from the working directory when it is relative, its `.`, `..` and
/// links as written (unlike `std::path::absolute`, which on Windows folds
/// `..` first); `p` itself when there is no working directory.
#[must_use]
pub fn from_cwd(p: &Path) -> PathBuf {
    if p.is_absolute() {
        return p.to_path_buf();
    }
    std::env::current_dir().map_or_else(|_| p.to_path_buf(), |d| d.join(p))
}

/// The directory a manifest (or any file) sits in, `.` for a bare file name.
/// Relative paths recorded in a manifest resolve against this.
#[must_use]
pub fn manifest_dir(path: &Path) -> &Path {
    path.parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."))
}

/// Resolve a path listed in a manifest against the manifest's own
/// directory. Absolute paths pass through unchanged. Convert the result
/// to a `String` via `.to_string_lossy().into_owned()` at call sites
/// that already hold string-typed paths.
#[must_use]
pub fn resolve(manifest_dir: &Path, rel: &str) -> PathBuf {
    let p = Path::new(rel);
    if p.is_absolute() {
        p.to_path_buf()
    } else {
        manifest_dir.join(p)
    }
}

/// `p` with `.` and `..` folded away, without touching the file system.
#[must_use]
pub fn normalize(p: &Path) -> PathBuf {
    use std::path::Component;
    let mut out = PathBuf::new();
    for c in p.components() {
        match c {
            Component::CurDir => {}
            Component::ParentDir => step_back(&mut out),
            c => out.push(c),
        }
    }
    out
}

/// `p` as a path from `base`, both absolute and normalized.
#[must_use]
pub fn relative_to(p: &Path, base: &Path) -> PathBuf {
    let (pc, bc): (Vec<_>, Vec<_>) = (p.components().collect(), base.components().collect());
    let common = pc.iter().zip(&bc).take_while(|(a, b)| a == b).count();
    let mut out: PathBuf = bc[common..].iter().map(|_| "..").collect();
    out.extend(&pc[common..]);
    out
}

/// Where `path` is after the tree it is in moved from `from_dir` to `to_dir`:
/// the same place relative to `to_dir` as it was to `from_dir`. Lexical, so
/// it works for paths that no longer exist.
#[must_use]
pub fn rebase(path: &Path, from_dir: &Path, to_dir: &Path) -> PathBuf {
    let abs = |p: &Path| normalize(&from_cwd(p));
    normalize(&abs(to_dir).join(relative_to(&abs(path), &abs(from_dir))))
}

/// A command's output prefix: `out` when given, else beside the manifest
/// `from` and named after it ([`derive_out_prefix`]). Not the manifest's
/// recorded `prefix`: one copied from another machine names that machine's
/// tree.
#[must_use]
pub fn out_prefix(out: Option<&str>, from: &str) -> String {
    out.map_or_else(|| derive_out_prefix(from), String::from)
}

/// Derive an output prefix from a `--from` manifest path when `--out` was
/// omitted: strip `.senna.json` / `.pinto.json` (preferred) or any trailing
/// `.json`, leaving the same basename the training run used.
#[must_use]
pub fn derive_out_prefix(from: &str) -> String {
    from.strip_suffix(".senna.json")
        .or_else(|| from.strip_suffix(".pinto.json"))
        .or_else(|| from.strip_suffix(".json"))
        .unwrap_or(from)
        .to_string()
}

/// Write a minimal `{prefix}.senna.json` stating only the kind.
///
/// Test-only. Downstream resolution (which θ table, which annotate mode) keys on
/// `kind` alone, so a fixture that carried the full output map would be asserting
/// on fields the code under test never reads.
#[cfg(test)]
pub fn write_kind_only(prefix: &str, kind: super::RunKind) -> anyhow::Result<()> {
    RunManifest::new(kind, prefix).save(Path::new(&default_path(prefix)))
}

/// Load the manifest for a `--from` that may be given either as the manifest
/// path (`out/run.senna.json`) or as the run's `--out` prefix (`out/run`).
///
/// Some commands were written against a prefix and the rest of senna
/// against a manifest path; accepting both means neither group's users have to
/// re-learn the flag, and a prefix that happens to end in `.json` still resolves
/// through [`derive_out_prefix`].
pub fn load_for(from: &str) -> anyhow::Result<(RunManifest, PathBuf)> {
    let direct = Path::new(from);
    if direct.is_file() {
        return RunManifest::load(direct);
    }
    let derived = manifest_path_for(from);
    RunManifest::load(Path::new(&derived)).map_err(|e| {
        anyhow::anyhow!(
            "{e}\n\nNeither `{from}` nor `{derived}` is a readable senna manifest. \
             Re-run the producer to get `{derived}`."
        )
    })
}

/// The manifest a `--from`-like flag names: `from` itself when it is a file,
/// else the manifest of the run prefix it is (see [`load_for`]).
#[must_use]
pub fn manifest_path_for(from: &str) -> String {
    if Path::new(from).is_file() {
        from.to_string()
    } else {
        default_path(&derive_out_prefix(from))
    }
}

/// Default manifest filename given a run `--out` prefix.
#[must_use]
pub fn default_path(prefix: &str) -> String {
    format!("{prefix}.senna.json")
}
