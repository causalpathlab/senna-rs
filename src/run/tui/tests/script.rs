use super::*;

fn argv(v: &[&str]) -> Vec<String> {
    v.iter().map(ToString::to_string).collect()
}

#[test]
fn words_are_quoted_only_when_the_shell_would_split_them() {
    assert_eq!(quote("data/d1.zarr.zip"), "data/d1.zarr.zip");
    assert_eq!(quote("a b"), "'a b'");
    assert_eq!(quote("it's"), r"'it'\''s'");
    assert_eq!(quote(""), "''");
    assert_eq!(quote("$HOME"), "'$HOME'");
}

#[test]
fn paths_are_made_relative_to_the_script() {
    let r = |p: &str, b: &str| relative(Path::new(p), Path::new(b));
    assert_eq!(
        r("/w/data/d.zarr", "/w/out"),
        PathBuf::from("../data/d.zarr")
    );
    assert_eq!(r("/w/out/d.zarr", "/w/out"), PathBuf::from("d.zarr"));
    assert_eq!(r("/w", "/w"), PathBuf::from("."));
}

#[test]
fn a_dotdot_in_the_out_folder_is_resolved_first() {
    let r = |p: &str, b: &str| relative(Path::new(p), Path::new(b));
    // The out folder /a/b/../res is /a/res: the data is one up and over.
    assert_eq!(
        r("/a/b/d.zarr", "/a/b/../res"),
        PathBuf::from("../b/d.zarr")
    );
    assert_eq!(r("/a/b/../b/d.zarr", "/a/b"), PathBuf::from("d.zarr"));
    assert_eq!(normalize(Path::new("/a/./b/../c")), PathBuf::from("/a/c"));
}

#[test]
fn far_away_data_is_written_whole() {
    let r = |p: &str, b: &str| relative(Path::new(p), Path::new(b));
    assert_eq!(
        r("/w/x/y/d.zarr", "/w/a/b"),
        PathBuf::from("../../x/y/d.zarr")
    );
    assert_eq!(
        r("/u/data/d.zarr", "/t/a/b/c"),
        PathBuf::from("/u/data/d.zarr")
    );
}

#[test]
fn the_script_refuses_to_run_over_a_result() {
    let s = text(
        &argv(&[
            "svd",
            "d 1.zarr",
            "--batch-files",
            "b.tsv",
            "--out",
            "r1",
            "--fast",
        ]),
        "r1",
        Tool::Senna,
    );
    assert!(s.starts_with("#!/usr/bin/env bash\n"));
    assert!(s.contains("if [ -f \"${out}.senna.json\" ]; then\n"));
    assert!(s.contains("out=r1\n"));
    assert!(s.ends_with(
        "\"${SENNA:-senna}\" svd \\\n  'd 1.zarr' \\\n  --batch-files b.tsv \\\n  --out \"$out\" \\\n  --fast\n"
    ));
}

#[test]
fn a_script_is_never_written_over() {
    let dir = tempfile::tempdir().unwrap();
    let a = argv(&["svd", "d.zarr", "--out", "r"]);
    let path = dir.path().join("r.cmd.sh");
    write(&path, "r", &a, Tool::Senna).unwrap();
    assert!(std::fs::read_to_string(&path)
        .unwrap()
        .contains("svd \\\n  d.zarr"));
    std::fs::write(&path, "kept").unwrap();
    assert!(write(&path, "r", &a, Tool::Senna).is_err());
    assert_eq!(std::fs::read_to_string(&path).unwrap(), "kept");
}

#[cfg(unix)]
#[test]
fn the_script_runs_the_command_once_and_then_refuses() {
    let dir = tempfile::tempdir().unwrap();
    // A stand-in senna that writes the manifest its --out names.
    let fake = dir.path().join("fake-senna");
    std::fs::write(
        &fake,
        "#!/bin/sh\nwhile [ $# -gt 0 ]; do [ \"$1\" = --out ] && touch \"$2.senna.json\"; shift; done\n",
    )
    .unwrap();
    use std::os::unix::fs::PermissionsExt;
    std::fs::set_permissions(&fake, std::fs::Permissions::from_mode(0o755)).unwrap();
    let path = dir.path().join("r.cmd.sh");
    write(
        &path,
        "r",
        &argv(&["svd", "d.zarr", "--out", "r"]),
        Tool::Senna,
    )
    .unwrap();
    let run = || {
        std::process::Command::new("bash")
            .arg(&path)
            .env("SENNA", &fake)
            .output()
            .unwrap()
    };
    assert!(run().status.success());
    assert!(dir.path().join("r.senna.json").exists());
    let again = run();
    assert!(!again.status.success());
    assert!(String::from_utf8_lossy(&again.stderr).contains("exists"));
}

#[test]
fn the_script_logs_as_the_run_on_screen_did() {
    let s = text(&argv(&["svd", "d.zarr", "--out", "r"]), "r", Tool::Senna);
    assert!(s.contains(&format!("export RUST_LOG=\"${{RUST_LOG:-{LOG_LEVEL}}}\"\n")));
}

#[test]
fn a_mung_script_runs_mung_and_guards_its_clones() {
    let s = text(
        &argv(&["clones", "d.zarr", "--out", "cnv", "--gff", "g.gtf"]),
        "cnv",
        Tool::Mung,
    );
    assert!(s.contains("if [ -f \"${out}.clones.parquet\" ]; then\n"));
    assert!(s.ends_with(
        "\"${MUNG:-mung}\" clones \\\n  d.zarr \\\n  --out \"$out\" \\\n  --gff g.gtf\n"
    ));
}
