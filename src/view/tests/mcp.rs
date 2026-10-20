use super::*;

fn open() -> Open {
    Open::of(crate::view::tests::scene(), PathBuf::from("r.senna.json"))
}

fn args(v: Value) -> MapArgs {
    serde_json::from_value(v).unwrap()
}

/// The JSON piece of a result.
fn state(out: &[Content]) -> &Value {
    out.iter()
        .find_map(|c| match c {
            Content::Json(v) => Some(v),
            _ => None,
        })
        .expect("a JSON piece")
}

#[test]
fn a_map_is_a_png_with_its_legend_and_its_settings_stay() {
    let mut o = open();
    let out = o
        .map(args(json!({"colour_by": "cluster", "width": 64})))
        .unwrap();
    let Content::Png(png) = &out[0] else {
        panic!("the map first")
    };
    assert_eq!(&png[1..4], b"PNG");
    let s = state(&out);
    assert_eq!(s["colour_by"], "cluster");
    assert_eq!(s["image"][0], 64);
    let legend = s["legend"].as_array().unwrap();
    assert_eq!(legend.len(), 2);
    assert_eq!(legend[1]["points"], 2);
    assert!(legend[0]["colour"].as_str().unwrap().starts_with('#'));

    let again = o.map(MapArgs::default()).unwrap();
    assert_eq!(state(&again)["colour_by"], "cluster");
}

#[test]
fn focus_takes_a_name_or_a_cluster_id_and_empty_clears_it() {
    let mut o = open();
    o.map(args(
        json!({"colour_by": "cluster", "focus": "K1", "width": 8}),
    ))
    .unwrap();
    assert_eq!(o.scene.focused_name().as_deref(), Some("C1"));
    o.map(args(json!({"focus": "c0", "width": 8}))).unwrap();
    assert_eq!(o.scene.focused_name().as_deref(), Some("C0"));
    o.map(args(json!({"focus": "", "width": 8}))).unwrap();
    assert_eq!(o.scene.focus, None);
    assert!(o.map(args(json!({"focus": "C9"}))).is_err());
}

#[test]
fn focus_takes_pinto_s_list_of_one_and_a_bare_number() {
    let mut o = open();
    o.map(args(
        json!({"colour_by": "cluster", "focus": ["C1"], "width": 8}),
    ))
    .unwrap();
    assert_eq!(o.scene.focused_name().as_deref(), Some("C1"));
    o.map(args(json!({"focus": 0, "width": 8}))).unwrap();
    assert_eq!(o.scene.focused_name().as_deref(), Some("C0"));
    o.map(args(json!({"focus": [], "width": 8}))).unwrap();
    assert_eq!(o.scene.focus, None);
    let e = o.map(args(json!({"focus": ["C0", "C1"]}))).unwrap_err();
    assert!(format!("{e}").contains("one group"), "{e}");
}

#[test]
fn a_grouping_the_map_lacks_is_an_error_naming_those_it_has() {
    let mut o = open();
    let e = o.map(args(json!({"colour_by": "topic"}))).unwrap_err();
    assert!(format!("{e}").contains("cluster"), "{e}");
}

#[test]
fn another_map_fits_anew_and_an_unknown_one_is_an_error() {
    let mut o = open();
    o.map(args(json!({"zoom": 2.0, "width": 8}))).unwrap();
    assert!(o.window.is_some());
    o.map(args(json!({"map": "phate", "width": 8}))).unwrap();
    assert_eq!(o.scene.current().method, "phate");
    assert!(o.window.is_none());
    assert!(o.map(args(json!({"map": "tsne"}))).is_err());
}

#[test]
fn zoom_halves_the_window_about_its_centre() {
    let mut o = open();
    o.map(args(
        json!({"bbox": [0.0, 0.0, 4.0, 2.0], "zoom": 2.0, "width": 8}),
    ))
    .unwrap();
    assert_eq!(o.window, Some([1.0, 0.5, 3.0, 1.5]));
    o.map(args(json!({"fit": true, "width": 8}))).unwrap();
    assert_eq!(o.window, None);
}

#[test]
fn a_cell_is_found_by_world_point_pixel_or_name_with_its_groups() {
    let mut o = open();
    let at = |o: &mut Open, a: Value| {
        let out = o.cell(serde_json::from_value(a).unwrap()).unwrap();
        state(&out).clone()
    };
    let c = at(&mut o, json!({"at": [2.1, 1.9]}));
    assert_eq!(c["name"], "c3");
    assert_eq!(c["groups"]["cluster"], "C1");
    assert!(c["groups"].get("annotation").is_none());

    assert_eq!(
        at(&mut o, json!({"name": "c1"}))["groups"]["annotation"],
        "CT1"
    );
    assert!(o
        .cell(serde_json::from_value(json!({"px": [0, 0]})).unwrap())
        .is_err());

    o.map(args(json!({"width": 64}))).unwrap();
    let vp = o.drawn.unwrap();
    let (px, py) = vp.to_px([3.0, 0.0]);
    assert_eq!(at(&mut o, json!({"px": [px, py]}))["name"], "c4");
}

#[test]
fn export_refuses_a_file_that_exists_unless_told() {
    let dir = tempfile::tempdir().unwrap();
    let png = dir.path().join("map.png");
    std::fs::write(&png, b"keep").unwrap();
    let mut o = open();
    let ask = |overwrite: bool| -> ExportArgs {
        serde_json::from_value(json!({"png": png, "width": 32, "overwrite": overwrite})).unwrap()
    };
    assert!(o.export(ask(false)).is_err());
    let notes = dir.path().join("notes.txt");
    let other = serde_json::from_value(json!({"png": notes, "overwrite": true})).unwrap();
    assert!(o.export(other).is_err());
    assert!(!notes.exists());
    assert_eq!(std::fs::read(&png).unwrap(), b"keep");
    o.export(ask(true)).unwrap();
    assert_eq!(&std::fs::read(&png).unwrap()[1..4], b"PNG");
}

#[test]
fn unknown_arguments_are_refused() {
    assert!(parse::<MapArgs>(json!({"colour": "cluster"})).is_err());
}

#[test]
fn a_call_that_fails_changes_nothing() {
    let mut o = open();
    o.map(args(json!({"colour_by": "cluster", "width": 8})))
        .unwrap();
    let drawn = o.drawn;
    assert!(o
        .map(args(json!({"map": "phate", "focus": "C9", "width": 8})))
        .is_err());
    assert!(o
        .map(args(json!({"colour_by": "annotation", "width": 0})))
        .is_err());
    assert_eq!(o.scene.current().method, "umap");
    assert_eq!(
        o.scene.colour.map(|i| o.scene.data.labels[i].title()),
        Some("cluster")
    );
    assert_eq!(o.drawn.map(|v| v.window()), drawn.map(|v| v.window()));
}

#[test]
fn the_grouping_shown_asked_for_again_keeps_its_focus() {
    let mut o = open();
    o.map(args(
        json!({"colour_by": "cluster", "focus": "C1", "width": 8}),
    ))
    .unwrap();
    o.map(args(json!({"colour_by": "cluster", "width": 8})))
        .unwrap();
    assert_eq!(o.scene.focused_name().as_deref(), Some("C1"));
    o.map(args(json!({"colour_by": "annotation", "width": 8})))
        .unwrap();
    assert_eq!(o.scene.focus, None);
}

#[test]
fn focus_on_another_map_is_read_in_the_grouping_it_will_show() {
    let mut o = open();
    o.map(args(json!({"colour_by": "annotation", "width": 8})))
        .unwrap();
    o.map(args(
        json!({"map": "phate", "colour_by": "cluster", "focus": "C1", "width": 8}),
    ))
    .unwrap();
    assert_eq!(o.scene.current().method, "phate");
    assert_eq!(o.scene.focused_name().as_deref(), Some("C1"));
}

#[test]
fn a_bbox_must_run_low_to_high_and_be_finite() {
    let mut o = open();
    for bad in [
        json!([0, 0, 0, 1]),
        json!([2, 0, 1, 1]),
        json!([0, 1, 1, 0]),
    ] {
        assert!(o.map(args(json!({"bbox": bad}))).is_err(), "{bad}");
    }
    assert_eq!(o.window, None);
}

#[test]
fn zoom_on_a_fitted_map_starts_from_the_window_drawn() {
    let mut o = open();
    let fitted = o.viewport(64).window();
    o.map(args(json!({"zoom": 2.0, "width": 64}))).unwrap();
    let w = o.window.unwrap();
    let half = |b: [f32; 4]| b[2] - b[0];
    assert!(
        (half(w) * 2.0 - half(fitted)).abs() < 1e-4,
        "{w:?} {fitted:?}"
    );
}

#[test]
fn a_published_view_applied_to_a_fresh_scene_shows_the_same() {
    let mut agent = open();
    agent
        .map(args(json!({"map": "phate", "colour_by": "cluster", "focus": "C1", "zoom": 2.0, "width": 32})))
        .unwrap();
    let (shown, a) = agent_view(&agent.view_state()).unwrap();
    assert_eq!(shown, agent.run);
    assert_eq!(a.bbox, agent.drawn.map(|v| v.window()));

    let mut watcher = crate::view::tests::scene();
    watcher.colour_by(LabelKind::Annotation);
    assert!(apply(&mut watcher, &a).unwrap());
    assert_eq!(watcher.current().method, "phate");
    assert_eq!(watcher.focused_name().as_deref(), Some("C1"));
    // Applied again, nothing moves.
    assert!(!apply(&mut watcher, &a).unwrap());
}

#[test]
fn a_published_view_names_the_run_and_exactly_the_keys_a_watcher_reads() {
    let o = open();
    let state = o.view_state();
    let mut keys: Vec<&str> = state
        .as_object()
        .unwrap()
        .keys()
        .map(String::as_str)
        .collect();
    keys.sort_unstable();
    let mut want: Vec<&str> = VIEW_KEYS.iter().copied().chain(["run"]).collect();
    want.sort_unstable();
    assert_eq!(keys, want);
}

#[test]
fn the_readme_gives_the_instructions_word_for_word() {
    let readme = include_str!("../../../README.md");
    let quoted: Vec<&str> = readme
        .split("### What the agent does")
        .nth(1)
        .expect("the README's agent section")
        .lines()
        .skip_while(|l| !l.starts_with("> "))
        .take_while(|l| l.starts_with('>'))
        .map(|l| l.trim_start_matches('>').trim())
        .collect();
    let words = |s: &str| s.split_whitespace().collect::<Vec<_>>().join(" ");
    assert_eq!(words(&quoted.join(" ")), words(INSTRUCTIONS));
}
