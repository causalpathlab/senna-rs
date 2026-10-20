//! `senna mcp` over stdio: the handshake, the tool list, a failing call
//! answered as a result the agent reads, with nothing else on stdout, and a
//! background fit.
#![cfg(feature = "view")]

use serde_json::{json, Value};
use std::io::{BufRead, BufReader, Write};
use std::process::{Command, ExitStatus, Stdio};

/// A `senna mcp` session given `calls` (id, tool, arguments) after the
/// handshake: every message it wrote, and how it ended.
fn session(calls: &[(u64, &str, Value)]) -> (Vec<Value>, ExitStatus) {
    let mut child = Command::new(env!("CARGO_BIN_EXE_senna"))
        .arg("mcp")
        // No records in the repository, no live feed left behind.
        .env(legume_mcp::project::OFF_ENV, "off")
        .env(legume_mcp::watch::ENV, "off")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .unwrap();
    let mut stdin = child.stdin.take().unwrap();
    let hello = json!({"jsonrpc": "2.0", "id": 0, "method": "initialize", "params": {"protocolVersion": "2025-06-18"}});
    writeln!(stdin, "{hello}").unwrap();
    for (id, method, params) in calls {
        let msg = json!({"jsonrpc": "2.0", "id": id, "method": method, "params": params});
        writeln!(stdin, "{msg}").unwrap();
    }
    drop(stdin);
    let messages = BufReader::new(child.stdout.take().unwrap())
        .lines()
        .map(|l| serde_json::from_str(&l.unwrap()).expect("only protocol on stdout"))
        .collect();
    (messages, child.wait().unwrap())
}

fn call(id: u64, tool: &str, arguments: Value) -> (u64, &str, Value) {
    (
        id,
        "tools/call",
        json!({"name": tool, "arguments": arguments}),
    )
}

#[test]
fn senna_mcp_lists_its_tools_and_answers_a_failure_as_a_result() {
    let (replies, status) = session(&[
        (1, "tools/list", json!({})),
        call(2, "view_map", json!({})),
        call(
            3,
            "run_fit",
            json!({"method": "nope", "data": ["x"], "out": "y"}),
        ),
        call(4, "run_help", json!({"method": "topic"})),
    ]);
    assert!(status.success());
    assert_eq!(replies.len(), 5);

    assert_eq!(replies[0]["result"]["serverInfo"]["name"], "senna");
    assert!(replies[0]["result"]["instructions"]
        .as_str()
        .unwrap()
        .contains("lupin"));
    let names: Vec<&str> = replies[1]["result"]["tools"]
        .as_array()
        .unwrap()
        .iter()
        .map(|t| t["name"].as_str().unwrap())
        .collect();
    assert_eq!(
        names,
        [
            "run_fit",
            "run_help",
            "view_open",
            "view_map",
            "view_markers",
            "view_cell",
            "view_export",
            "job_status",
            "job_wait",
            "job_cancel"
        ]
    );
    for r in &replies[2..4] {
        assert_eq!(r["result"]["isError"], true, "{r}");
    }
    assert!(replies[2]["result"]["content"][0]["text"]
        .as_str()
        .unwrap()
        .contains("view_open"));
    let help = replies[4]["result"]["content"][0]["text"].as_str().unwrap();
    assert!(help.contains("--out"), "{help}");
}

#[cfg(unix)]
#[test]
fn a_background_fit_returns_a_job_and_the_same_out_again_is_refused_naming_it() {
    let dir = tempfile::tempdir().unwrap();
    // A data file nobody writes: senna blocks opening it, so the job still
    // runs when the same out is asked for again (and ends with the server).
    let data = dir.path().join("data.zarr.zip");
    assert!(Command::new("mkfifo")
        .arg(&data)
        .status()
        .unwrap()
        .success());
    let out = dir.path().join("fit").to_string_lossy().into_owned();
    let fit = json!({"method": "svd", "data": [data], "out": out, "background": true});
    let (messages, status) = session(&[call(1, "run_fit", fit.clone()), call(2, "run_fit", fit)]);
    // Should the fit outlive its server, opening the pipe frees it: read
    // and write, so the open itself never waits.
    drop(
        std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .open(&data),
    );
    assert!(
        status.success(),
        "the server stops its job and ends cleanly"
    );
    // Replies only: a job's progress may come between them.
    let replies: Vec<&Value> = messages.iter().filter(|m| m.get("id").is_some()).collect();
    let job = replies[1]["result"]["structuredContent"]["job"]
        .as_str()
        .unwrap_or_else(|| panic!("{}", replies[1]));
    assert!(job.starts_with("job-"));
    let again = &replies[2]["result"];
    assert_eq!(again["isError"], true, "{}", replies[2]);
    let said = again["content"][0]["text"].as_str().unwrap_or_default();
    assert!(said.contains(&format!("by {job}:")), "{said}");
}
