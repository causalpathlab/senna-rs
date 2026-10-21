//! `senna mcp`'s background fits run in a detached session of `senna run`:
//! a person attaches to it and sees the Run screen, it outlives the server
//! that started it, the next server takes it up and can cancel it, and a
//! fit that succeeds is recorded.

#![cfg(all(unix, feature = "view"))]

use legume_tui::testing::{Private, Term};
use legume_tui::Size;
use serde_json::{json, Value};
use std::io::{BufRead, BufReader, Write};
use std::path::Path;
use std::process::{Command, Stdio};
use std::time::Duration;

/// The run screen's tab of flags: drawn on every full paint.
const PAINTED: &str = "3 Parameters";

fn private() -> Private {
    Private::new(env!("CARGO_BIN_EXE_senna")).patience(Duration::from_secs(60))
}

/// A `senna mcp` in `project`, its sessions in `sessions`, given `calls`
/// (tool, arguments) after the handshake: its replies, in order.
fn server(sessions: &Private, project: &Path, calls: &[(&str, Value)]) -> Vec<Value> {
    let mut server = Command::new(env!("CARGO_BIN_EXE_senna"));
    server
        .arg("mcp")
        .current_dir(project)
        .env(legume_tui::DIR_VAR, sessions.path())
        .env(legume_mcp::watch::ENV, project.join("watch"))
        .env(legume_mcp::project::ROOT_ENV, project)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null());
    let mut child = server.spawn().unwrap();
    let mut stdin = child.stdin.take().unwrap();
    let hello = json!({"jsonrpc": "2.0", "id": 0, "method": "initialize", "params": {}});
    writeln!(stdin, "{hello}").unwrap();
    for (id, (tool, args)) in calls.iter().enumerate() {
        let msg = json!({"jsonrpc": "2.0", "id": id + 1, "method": "tools/call",
                         "params": {"name": tool, "arguments": args}});
        writeln!(stdin, "{msg}").unwrap();
    }
    drop(stdin);
    let replies = BufReader::new(child.stdout.take().unwrap())
        .lines()
        .map(|l| serde_json::from_str::<Value>(&l.unwrap()).unwrap())
        .filter(|m| m.get("id").is_some_and(|id| id != 0))
        .map(|m| m["result"].clone())
        .collect();
    assert!(child.wait().unwrap().success());
    replies
}

fn structured(r: &Value) -> &Value {
    &r["structuredContent"]
}

fn term(sessions: &Private, args: &[&str]) -> Term {
    let size = Size {
        rows: 40,
        cols: 120,
        width: 1200,
        height: 800,
    };
    sessions.term_sized(args, &[], size)
}

#[path = "../src/tests/planted.rs"]
mod planted;

#[test]
fn a_background_fit_is_watched_outlives_its_server_and_is_cancelled_by_the_next() {
    let sessions = private();
    let tmp = tempfile::tempdir().unwrap();
    let project = tmp.path();
    // A data file nobody writes: the fit blocks opening it.
    let data = project.join("data.zarr.zip");
    assert!(Command::new("mkfifo")
        .arg(&data)
        .status()
        .unwrap()
        .success());
    let fit =
        json!({"method": "svd", "data": [&data], "out": project.join("fit"), "background": true});
    let replies = server(&sessions, project, &[("run_fit", fit)]);
    let started = structured(&replies[0]);
    let session = started["session"]
        .as_str()
        .unwrap_or_else(|| panic!("{}", replies[0]))
        .to_string();
    assert_eq!(session, "senna-fit-fit", "{started}");

    // Its server is gone; the fit runs on, and a person attaches to it.
    sessions.wait_listed(&session);
    let mut t = term(&sessions, &["attach", &session]);
    t.wait_screen(PAINTED);
    t.wait_screen("svd");
    t.type_(b"\x1b\x04");
    assert_eq!(t.exit_code(), 0);

    // The next server takes it up, and cancels it.
    let replies = server(
        &sessions,
        project,
        &[
            ("job_status", json!({})),
            ("job_cancel", json!({"id": "job-1"})),
            ("job_wait", json!({"id": "job-1", "seconds": 60})),
        ],
    );
    let jobs = &structured(&replies[0])["jobs"];
    assert_eq!(jobs[0]["session"], session.as_str(), "{jobs}");
    assert_eq!(jobs[0]["state"], "running", "{jobs}");
    assert_eq!(
        structured(&replies[2])["state"],
        "cancelled",
        "{}",
        replies[2]
    );
    sessions.wait_said(&["sessions"], "no live senna sessions");
    let index = std::fs::read_to_string(project.join("results/INDEX.tsv")).unwrap_or_default();
    assert!(!index.contains("fit.senna.json"), "{index}");
}

#[test]
fn a_background_fit_that_succeeds_is_recorded() {
    let sessions = private();
    let tmp = tempfile::tempdir().unwrap();
    let project = tmp.path();
    let data = planted::planted_zarr(project);
    let fit =
        json!({"method": "svd", "data": [data], "out": project.join("fit"), "background": true});
    let replies = server(
        &sessions,
        project,
        &[
            ("run_fit", fit),
            ("job_wait", json!({"id": "job-1", "seconds": 120})),
        ],
    );
    let done = structured(&replies[1]);
    assert_eq!(done["state"], "done", "{}", replies[1]);
    // `senna run` ends with its fit, not at its next idle wake.
    assert!(done["seconds"].as_u64().unwrap() < 30, "{done}");
    assert!(project.join("fit.senna.json").exists());
    let index = std::fs::read_to_string(project.join("results/INDEX.tsv")).unwrap();
    assert!(index.contains("fit.senna.json"), "{index}");
    let day = std::fs::read_dir(project.join("results"))
        .unwrap()
        .flatten()
        .find(|e| e.path().is_dir())
        .unwrap()
        .path();
    let script = std::fs::read_to_string(day.join("runall.sh")).unwrap();
    assert!(script.contains("senna svd"), "{script}");
    // The run's own script beside it, as `senna run` writes one.
    assert!(project.join("fit.cmd.sh").exists());
}
