//! `senna run` as a detachable session, driven through pseudo-terminals
//! standing in for the user's. Opt-in only (`cargo test -- --ignored`,
//! Linux): it waits on what a real terminal paints, so its timing follows
//! the machine's speed, which a default test run must not depend on. It
//! goes through a hosted run with the client lost instead of detached, and
//! checks `--no-detach`.

#![cfg(all(unix, feature = "view"))]

use legume_tui::testing::{alive, Private, Term};
use legume_tui::Size;
use std::time::Duration;

/// The run screen's tab of flags: drawn on every full paint.
const PAINTED: &str = "3 Parameters";

/// The size every test terminal has but where one says otherwise.
const SIZE: Size = Size {
    rows: 40,
    cols: 120,
    width: 1200,
    height: 800,
};

/// A private directory for senna's sessions; a session a failed test left
/// running is killed with it.
fn private() -> Private {
    Private::new(env!("CARGO_BIN_EXE_senna")).patience(Duration::from_secs(30))
}

fn term(dir: &Private, args: &[&str], env: &[(&str, &str)]) -> Term {
    dir.term_sized(args, env, SIZE)
}

/// A process's parent and session, as `/proc` has them.
#[cfg(target_os = "linux")]
fn parent_and_session(pid: u32) -> (u32, u32) {
    let stat = std::fs::read_to_string(format!("/proc/{pid}/stat")).unwrap();
    // After the command's name in parentheses: state, parent, group, session.
    let rest = &stat[stat.rfind(')').unwrap() + 1..];
    let fields: Vec<&str> = rest.split_whitespace().collect();
    (fields[1].parse().unwrap(), fields[3].parse().unwrap())
}

/// One variable of a running process's environment.
#[cfg(target_os = "linux")]
fn environ_of(pid: u32, key: &str) -> Option<String> {
    let all = std::fs::read(format!("/proc/{pid}/environ")).unwrap();
    all.split(|b| *b == 0)
        .filter_map(|kv| std::str::from_utf8(kv).ok())
        .find_map(|kv| kv.strip_prefix(&format!("{key}=")).map(str::to_string))
}

/// The size of the terminal a running process reads from.
#[cfg(target_os = "linux")]
fn winsize_of(pid: u32) -> rustix::termios::Winsize {
    let tty = rustix::fs::open(
        format!("/proc/{pid}/fd/0"),
        rustix::fs::OFlags::RDWR | rustix::fs::OFlags::NOCTTY,
        rustix::fs::Mode::empty(),
    )
    .unwrap();
    rustix::termios::tcgetwinsize(&tty).unwrap()
}

/// The whole life of a hosted `senna run` in an empty folder, its client
/// lost rather than detached: run with `cargo test -- --ignored`.
#[cfg(target_os = "linux")]
#[test]
#[ignore = "end to end smoke test; cargo test -- --ignored"]
fn smoke_hosted_run_survives_a_lost_client() {
    let dir = private();
    let work = tempfile::tempdir().unwrap();
    let work = work.path().to_string_lossy().into_owned();

    // Hosted by default: the program is not the process started here, it
    // runs under a host in a session of its own, and one socket is there.
    let mut first = term(&dir, &["run", &work], &[]);
    first.wait_for(PAINTED);
    dir.wait_listed("senna-run-1\tattached");
    let pid = dir.program_pid();
    let client = first.child.id();
    assert_ne!(pid, client, "the program runs apart from its client");
    let (host, sid) = parent_and_session(pid);
    assert_ne!(host, client, "the host is not the client");
    assert_ne!(sid, client, "a hangup on the client's terminal misses it");
    assert_eq!(dir.sockets(), ["senna-run-1"]);

    // The client lost without a word (as on a dropped connection): the
    // program and its host run on, detached.
    first.child.kill().unwrap();
    first.child.wait().unwrap();
    drop(first);
    dir.wait_listed("senna-run-1\tdetached");
    assert!(alive(pid) && alive(host));

    // Attached from a terminal of another size: its cells and pixels reach
    // the program, and the screen arrives whole after the client's reset.
    let size = Size {
        rows: 50,
        cols: 150,
        width: 1500,
        height: 1000,
    };
    let mut second = dir.term_sized(&["attach", "run-1"], &[], size);
    for tab in ["1 Data", "2 Methods", PAINTED] {
        second.wait_screen(tab);
    }
    assert!(second.alternate());
    let w = winsize_of(pid);
    assert_eq!(
        (w.ws_row, w.ws_col, w.ws_xpixel, w.ws_ypixel),
        (50, 150, 1500, 1000)
    );
    let listed = dir.wait_listed("senna-run-1\tattached");
    assert!(listed.contains(&format!("pid {pid}")), "{listed:?}");
    assert!(listed.contains("setting up fits"), "{listed:?}");

    // A TUI started with the program's own environment, marker and all,
    // runs plain: no second host, no second socket.
    let marker = environ_of(pid, legume_tui::SESSION_VAR);
    assert_eq!(marker.as_deref(), Some("senna-run-1"));
    let mut nested = term(
        &dir,
        &["view"],
        &[(legume_tui::SESSION_VAR, marker.as_deref().unwrap())],
    );
    nested.wait_for("Run to view");
    assert_eq!(dir.sockets(), ["senna-run-1"]);
    nested.type_(b"\x03");
    assert_eq!(nested.exit_code(), 0);
    assert_eq!(dir.sockets(), ["senna-run-1"]);

    // Killed while attached: the program quits as on ctrl+c, the client
    // ends with it, and nothing is left: no process, socket or file.
    let said = dir.run(&["kill", "run-1"]);
    assert!(said.contains("senna-run-1 stopped"), "{said:?}");
    assert_eq!(second.exit_code(), 0);
    assert!(second.cooked(), "terminal restored after the program ended");
    dir.gone(pid);
    dir.gone(host);
    assert!(dir.is_empty(), "left behind in {:?}", dir.path());
    assert!(dir.run(&["sessions"]).contains("no live senna sessions"));

    // Opted out, by the flag or the variable, it runs plain as before: in
    // this process, no client between, no socket, and ctrl+c ends it with
    // the terminal restored.
    for (args, env) in [
        (vec!["run", &work, "--no-detach"], vec![]),
        (vec!["run", &work], vec![(legume_tui::NO_DETACH_VAR, "1")]),
    ] {
        let mut plain = term(&dir, &args, &env);
        plain.wait_for(PAINTED);
        assert!(!plain.cooked());
        assert!(dir.sockets().is_empty(), "{args:?} {env:?}");
        assert!(dir.run(&["sessions"]).contains("no live senna sessions"));
        assert!(!plain.text().contains("\x1b_Ga=d,d=A"), "no client's reset");
        plain.type_(b"\x03");
        assert_eq!(plain.exit_code(), 0);
        assert!(plain.cooked(), "terminal restored");
    }
}
