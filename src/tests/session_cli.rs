//! The session flags and subcommands of `senna run` and `senna view`.

use super::*;

fn parse(argv: &[&str]) -> Commands {
    Cli::try_parse_from(argv).unwrap().commands
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
    assert!(v.session.options(!v.runs().is_empty()).target);
    // A manifest is taken bare, as well as after -f; bare ones come first.
    let Commands::View(v) = parse(&["senna", "view", "a.json", "-f", "b.json"]) else {
        panic!("view")
    };
    assert_eq!(
        v.runs(),
        ["a.json".into(), "b.json".into()] as [Box<str>; 2]
    );
    let Commands::View(v) = parse(&["senna", "view"]) else {
        panic!("view")
    };
    assert!(v.runs().is_empty());
}
