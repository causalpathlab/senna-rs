use super::*;
use senna::run_manifest::{Decision, DecisionOption, RunKind};

/// A run tool's why comes with who chose it, and is taken out of its
/// arguments; run_help takes none.
#[test]
fn a_why_needs_its_by_and_leaves_the_arguments() {
    let mut args = json!({"method": "ode", "why": "the HSCs are least differentiated", "by": "user", "cites": ["lupin r2 annotate_why"]});
    let why = take("run_fit", &mut args).unwrap().unwrap();
    assert_eq!((why.by(), why.cites().len()), (By::User, 1));
    assert_eq!(args, json!({"method": "ode"}));
    for bad in [
        json!({"why": "because"}),
        json!({"by": "user"}),
        json!({"why": "because", "by": "someone"}),
    ] {
        assert!(take("run_fit", &mut bad.clone()).is_err(), "{bad}");
    }
    let mut help = json!({"method": "ode", "why": "x"});
    assert!(take("run_help", &mut help).unwrap().is_none());
    assert!(help.get("why").is_some());
}

/// Flags that answer a recorded decision need a why; others do not.
#[test]
fn an_answer_to_a_decision_is_refused_without_a_why() {
    let dir = tempfile::tempdir().unwrap();
    let list = dir.path().join("r.root_0.txt");
    std::fs::write(&list, "a\n").unwrap();
    let other = dir.path().join("mine.txt");
    std::fs::write(&other, "a\n").unwrap();
    let mut m = RunManifest::new(RunKind::Ode, "r");
    let list_s = list.to_string_lossy().into_owned();
    m.decisions.push(Decision {
        topic: "root".into(),
        question: "Which root?".into(),
        hint: String::new(),
        options: vec![DecisionOption {
            label: "root 0".into(),
            rerun_args: vec!["--root-cells".into(), list_s.clone()],
            objective: None,
            cells: Some(list_s.clone()),
        }],
    });
    m.save(&dir.path().join("r.senna.json")).unwrap();
    let flag = |p: &std::path::Path| vec![format!("--root-cells={}", p.display())];
    let e = justified(&flag(&list), None).unwrap_err();
    assert!(format!("{e}").contains("Which root?"), "{e}");
    let spaced = ["--root-cells".to_string(), list_s];
    assert!(justified(&spaced, None).is_err());
    let why = Why::new("the user picked root 0", By::User);
    justified(&flag(&list), Some(&why)).unwrap();
    justified(&flag(&other), None).unwrap();
}
