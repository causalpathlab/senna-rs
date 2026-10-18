//! `senna run` as a detachable session, driven through pseudo-terminals
//! standing in for the user's: it starts hosted, a detach leaves it
//! running and asks for a name, an attach from another terminal repaints
//! it, a TUI started under a session's marker runs plain, `senna rename`
//! relabels it, and `senna kill` stops it through its own quit path,
//! leaving no process, socket or label behind. An ignored
//! smoke test (`cargo test -- --ignored`, Linux) goes through it all once
//! more with the client lost instead of detached, and checks `--no-detach`.

#![cfg(all(unix, feature = "view"))]

use legume_tui::testing::{alive, Private, Term, PROMPT};
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

#[test]
fn run_detaches_attaches_and_is_killed_through_its_quit_path() {
    let dir = private();
    let work = tempfile::tempdir().unwrap();
    let work = work.path().to_string_lossy().into_owned();

    // Started in a terminal, it is hosted: this terminal is its first client.
    let mut first = term(&dir, &["run", &work], &[]);
    first.wait_for(PAINTED);
    assert!(!first.cooked(), "the client's terminal is raw");
    let listed = dir.wait_listed("setting up fits");
    assert!(listed.contains("senna-run-1\tattached"), "{listed:?}");
    let pid = dir.program_pid();

    // Detached (ctrl+alt+d): the client ends, its terminal restored, and
    // the program runs on; Enter leaves it unnamed.
    first.type_(b"\x1b\x04");
    first.answer(1, b"\r");
    assert_eq!(first.exit_code(), 0);
    assert!(first.cooked(), "terminal restored after detach");
    first.wait_for("Detached senna-run-1.");
    dir.wait_listed("senna-run-1\tdetached");
    assert!(alive(pid));

    // Attached from another terminal: painted whole there.
    let mut second = term(&dir, &["attach"], &[]);
    second.wait_for(PAINTED);
    let before = second.seen_len();
    // Keys reach the program: typed into the output header it asks first.
    second.type_(b"z");
    second.wait_for_after(before, "z▏");
    // Out of the field and the file browser after it, its `?` page lists
    // the detach key.
    for _ in 0..2 {
        second.type_(b"\x1b");
        std::thread::sleep(Duration::from_millis(300));
    }
    second.type_(b"?");
    second.wait_for("ctrl+alt+d");
    second.wait_for("detach:");
    second.wait_for("ctrl+alt+s");
    second.wait_for("switch:");

    // A TUI started under the session's marker, as `senna run` starts
    // `senna view` when its fits are done, runs plain: no second host, no
    // second socket, drawn while the session is attached.
    let mut nested = term(&dir, &["view"], &[(legume_tui::SESSION_VAR, "senna-run-1")]);
    nested.wait_for("Run to view");
    assert_eq!(dir.sockets(), ["senna-run-1"]);
    nested.type_(b"\x03");
    assert_eq!(nested.exit_code(), 0);
    second.type_(b"\x1b\x04");
    second.answer(1, b"\r");
    assert_eq!(second.exit_code(), 0);
    dir.wait_listed("senna-run-1\tdetached");

    // Renamed from a shell: listed and found by its label.
    let said = dir.run(&["rename", "run-1", "fits"]);
    assert!(said.contains("senna-run-1 is now fits"), "{said:?}");
    dir.wait_listed("fits (senna-run-1)\tdetached");

    // Killed: it quits as ctrl+c does, and leaves nothing behind.
    let said = dir.run(&["kill", "fits"]);
    assert!(said.contains("senna-run-1 stopped"), "{said:?}");
    assert!(dir.sockets().is_empty());
    assert!(dir.is_empty(), "no label or status left");
    dir.gone(pid);
    assert!(dir.run(&["sessions"]).contains("no live senna sessions"));
}

#[test]
fn view_comes_back_on_attach_and_stops_on_kill() {
    let dir = private();

    // With no run named it browses for one, hosted.
    let mut first = term(&dir, &["view"], &[]);
    first.wait_for("Run to view");
    dir.wait_listed("senna-view-1\tattached");
    let pid = dir.program_pid();
    // Named as it is left.
    first.type_(b"\x1b\x04");
    first.answer(1, b"browse\r");
    assert_eq!(first.exit_code(), 0);
    first.wait_for("detached: browse");
    dir.wait_listed("browse (senna-view-1)\tdetached");

    // Back by its label on another terminal, the browser is drawn there
    // whole; it is not asked for a name again.
    let mut second = term(&dir, &["attach", "browse"], &[]);
    second.wait_for("Run to view");
    second.type_(b"\x1b\x04");
    assert_eq!(second.exit_code(), 0);
    second.wait_for("detached: browse");
    assert!(!second.text().contains(PROMPT));

    // Its id still finds it. Killed while detached: the browser gives up
    // as on ctrl+c.
    let said = dir.run(&["kill", "view-1"]);
    assert!(said.contains("senna-view-1 stopped"), "{said:?}");
    assert!(dir.sockets().is_empty());
    dir.gone(pid);
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
    second.wait_for("\x1b[?1049h\x1b[H\x1b[2J");
    let reset = second.text().find("\x1b[?1049h\x1b[H\x1b[2J").unwrap();
    for tab in ["1 Data", "2 Methods", PAINTED] {
        second.wait_for_after(reset, tab);
    }
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

#[test]
fn an_early_failure_stays_in_sight_on_the_normal_screen() {
    let dir = private();

    // Hosted, the run named is not there: the program's error reaches
    // this terminal, which never left the normal screen, and the launcher
    // ends with the program's failure, leaving nothing behind.
    let missing = dir.path().join("none").join("x.senna.json");
    let mut t = term(&dir, &["view", "-f", &missing.to_string_lossy()], &[]);
    t.wait_for("Error");
    assert_ne!(t.exit_code(), 0);
    let text = t.text();
    assert!(text.contains("x.senna.json"), "{text:?}");
    assert!(text.contains("senna-view-1 exited"), "hosted: {text:?}");
    assert!(!text.contains("\x1b[?1049h"), "alternate screen: {text:?}");
    assert!(t.cooked(), "terminal restored");
    assert!(dir.sockets().is_empty());
    assert!(dir.is_empty(), "files left");
}
