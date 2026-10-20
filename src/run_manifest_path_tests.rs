//! Resolving a path as the system does: links where they are met, `..`
//! after them, parts not there yet as written.
#![cfg(unix)]

use super::*;
use std::os::unix::fs::symlink;

#[test]
fn an_existing_path_resolves_to_its_canonical_path() {
    let dir = tempfile::tempdir().unwrap();
    fs::create_dir_all(dir.path().join("a/b")).unwrap();
    symlink(dir.path().join("a/b"), dir.path().join("link")).unwrap();
    let p = dir.path().join("link/../b/./");
    assert_eq!(canonical_so_far(&p), p.canonicalize().unwrap());
}

#[test]
fn dot_dot_after_a_link_steps_back_from_where_the_link_leads() {
    let dir = tempfile::tempdir().unwrap();
    fs::create_dir_all(dir.path().join("deep/sub")).unwrap();
    symlink("deep/sub", dir.path().join("link")).unwrap();
    let real = dir.path().canonicalize().unwrap();
    // Not beside the link: `link/..` is `deep`.
    assert_eq!(
        canonical_so_far(&dir.path().join("link/../x")),
        real.join("deep/x")
    );
    assert_eq!(canonical_so_far(&dir.path().join("a/../x")), real.join("x"));
}

#[test]
fn a_link_to_a_folder_not_made_yet_is_followed_and_agrees_once_it_is() {
    let dir = tempfile::tempdir().unwrap();
    symlink("later", dir.path().join("link")).unwrap();
    let real = dir.path().canonicalize().unwrap();
    let before = canonical_so_far(&dir.path().join("link/run.senna.json"));
    assert_eq!(before, real.join("later/run.senna.json"));
    fs::create_dir(dir.path().join("later")).unwrap();
    assert_eq!(
        canonical_so_far(&dir.path().join("link/run.senna.json")),
        before
    );
}

#[test]
fn a_link_loop_ends() {
    let dir = tempfile::tempdir().unwrap();
    symlink("b", dir.path().join("a")).unwrap();
    symlink("a", dir.path().join("b")).unwrap();
    // Given up on after the system's limit; it returns rather than spins.
    let _ = canonical_so_far(&dir.path().join("a/x"));
}
