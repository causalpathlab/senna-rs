use super::*;

#[test]
fn every_command_has_the_arguments_its_tool_takes_itself() {
    for c in super::super::steps::COMMANDS {
        assert!(own(c).is_ok(), "run_step's {c}");
    }
    for path in helped() {
        let cmd = command(&path).unwrap();
        for (id, _) in own(&path).unwrap() {
            // A layout without a target lays out cells only.
            if *id == "target" && path.starts_with("layout ") {
                continue;
            }
            assert!(
                cmd.get_arguments().any(|a| a.get_id() == *id),
                "{path} has no {id}"
            );
        }
    }
}

#[test]
fn flags_become_words_and_the_tool_s_own_or_an_unknown_one_are_refused() {
    let map = |v: Value| v.as_object().unwrap().clone();
    let argv = words(
        "predict",
        &map(json!({"batch-files": ["b1.txt", "b2.txt"], "scoring_delta": "train", "preload-data": true})),
    )
    .unwrap();
    assert!(
        argv.contains(&"--scoring-delta=train".to_string()),
        "{argv:?}"
    );
    assert!(argv.contains(&"--preload-data".to_string()), "{argv:?}");
    for (bad, said) in [
        (json!({"out": "x"}), "as `out`"),
        (json!({"decoder-only": true}), "as `latent`"),
        (json!({"data-files": ["x.zarr"]}), "as `data`"),
        (json!({"no-such-flag": 1}), "run_help"),
    ] {
        let e = words("predict", &map(bad.clone())).unwrap_err().to_string();
        assert!(e.contains(said), "{bad}: {e}");
    }
    let listed = listed("predict").unwrap();
    let names: Vec<&str> = listed["flags"]
        .as_array()
        .unwrap()
        .iter()
        .map(|f| f["flag"].as_str().unwrap())
        .collect();
    assert!(names.contains(&"--refine-steps"), "{names:?}");
    assert!(!names.contains(&"--out") && !names.contains(&"--decoder-only"));
}
