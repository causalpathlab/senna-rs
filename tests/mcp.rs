//! `senna mcp` over stdio: the handshake, the tool list, and a failing call
//! answered as a result the agent reads, with nothing else on stdout.
#![cfg(feature = "view")]

use serde_json::{json, Value};
use std::io::{BufRead, BufReader, Write};
use std::process::{Command, Stdio};

#[test]
fn senna_mcp_lists_its_tools_and_answers_a_failure_as_a_result() {
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
    for (id, method, params) in [
        (1, "initialize", json!({"protocolVersion": "2025-06-18"})),
        (2, "tools/list", json!({})),
        (
            3,
            "tools/call",
            json!({"name": "view_map", "arguments": {}}),
        ),
        (
            4,
            "tools/call",
            json!({"name": "run_fit", "arguments": {"method": "nope", "data": ["x"], "out": "y"}}),
        ),
        (
            5,
            "tools/call",
            json!({"name": "run_help", "arguments": {"method": "topic"}}),
        ),
    ] {
        let msg = json!({"jsonrpc": "2.0", "id": id, "method": method, "params": params});
        writeln!(stdin, "{msg}").unwrap();
    }
    drop(stdin);
    let replies: Vec<Value> = BufReader::new(child.stdout.take().unwrap())
        .lines()
        .map(|l| serde_json::from_str(&l.unwrap()).expect("only protocol on stdout"))
        .collect();
    assert!(child.wait().unwrap().success());
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
            "view_export"
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
