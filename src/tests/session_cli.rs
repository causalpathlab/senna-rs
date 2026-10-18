//! The session flags and subcommands of `senna run` and `senna view`.

use super::*;

fn parse(argv: &[&str]) -> Commands {
    Cli::try_parse_from(argv).unwrap().commands
}

#[test]
fn rename_parses() {
    match parse(&["senna", "rename", "run-1", "work"]) {
        Commands::Rename(r) => {
            assert_eq!(r.session, "run-1");
            assert_eq!(r.label, "work");
        }
        other => panic!("{other:?}"),
    }
    assert!(Cli::try_parse_from(["senna", "rename", "run-1"]).is_err());
}

#[test]
fn attach_sessions_and_kill_parse() {
    match parse(&["senna", "attach"]) {
        Commands::Attach(a) => assert_eq!(a.name, None),
        other => panic!("{other:?}"),
    }
    match parse(&["senna", "attach", "run-1"]) {
        Commands::Attach(a) => assert_eq!(a.name.as_deref(), Some("run-1")),
        other => panic!("{other:?}"),
    }
    assert!(matches!(parse(&["senna", "sessions"]), Commands::Sessions));
    match parse(&["senna", "kill", "--force", "view-2"]) {
        Commands::Kill(k) => {
            assert!(k.force);
            assert_eq!(k.name.as_deref(), Some("view-2"));
        }
        other => panic!("{other:?}"),
    }
}

#[test]
fn a_target_named_skips_the_resume_picker() {
    // `senna run` alone may resume; a folder named starts a new session.
    let Commands::Run(r) = parse(&["senna", "run"]) else {
        panic!("run")
    };
    assert!(r.dir.is_none() && !r.session().target && !r.session().no_detach);
    let Commands::Run(r) = parse(&["senna", "run", "data", "--no-detach"]) else {
        panic!("run")
    };
    assert!(r.session().target && r.session().no_detach);
    let Commands::View(v) = parse(&["senna", "view", "--session", "x", "-f", "a.json"]) else {
        panic!("view")
    };
    assert_eq!(v.session.session.as_deref(), Some("x"));
    assert!(v.session.options(!v.from.is_empty()).target);
}

/// The long help of `name`, as `senna <name> --help` prints it.
fn long_help(name: &str) -> String {
    use clap::CommandFactory;
    let mut cli = Cli::command();
    cli.build();
    let sub = cli.find_subcommand(name).unwrap();
    sub.get_long_about().unwrap().to_string()
}

/// `text` on one line, its words a space apart.
fn flat(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

#[test]
fn the_session_help_is_the_shared_text_said_as_senna_says_it() {
    let run = flat(&long_help("run"));
    for said in [
        "the fits go on when the connection drops",
        "`senna run` with no folder offers the live sessions",
        "`senna attach`, `sessions`, `kill` and `rename`",
        "`?` shows the keys in effect.",
        "--no-detach",
    ] {
        assert!(run.contains(said), "{said:?} in {run}");
    }
    let view = flat(&long_help("view"));
    assert!(
        view.contains("`senna view` with no -f offers the live sessions"),
        "{view}"
    );
    assert!(view.contains("`?` shows the keys in effect."), "{view}");
    let attach = flat(&long_help("attach"));
    for said in [
        "a detached `senna run` or `senna view`",
        "sets another; `?` shows it)",
        "`senna sessions` lists them",
        "--no-detach on either starts it plain.",
    ] {
        assert!(attach.contains(said), "{said:?} in {attach}");
    }
    // Wrapped again as the rest of senna's help, a paragraph apart.
    for help in [long_help("run"), long_help("view"), long_help("attach")] {
        let ours = help.rsplit("\n\n").next().unwrap();
        assert!(ours.lines().all(|l| l.chars().count() <= 70), "{help}");
        assert!(ours.lines().count() > 1, "{help}");
    }
}
