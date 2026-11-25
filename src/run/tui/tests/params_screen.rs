//! The parameters screen: the shared page across queued fits, flags that
//! pick a file, a partition shared by fits, and per-fit defaults.

use super::*;

/// The row index of method `name`.
fn row_of(a: &App, name: &str) -> usize {
    a.rows.iter().position(|r| r.label == name).unwrap()
}

#[test]
fn the_shared_page_sets_a_flag_in_every_queued_fit_that_has_it() {
    let dir = tempfile::tempdir().unwrap();
    let mut a = app(dir.path());
    let (topic, simba) = (row_of(&a, "topic"), row_of(&a, "simba"));
    a.rows[topic].on = true;
    a.rows[simba].on = true;
    a.show_params(topic);
    // The shared page is listed first: `[` from the first method.
    key(&mut a, KeyCode::Char('['));
    assert!(a.shared_page);
    // `--steps` and `--sort-dim` are shared; `--cnv-clones` is topic's alone.
    let shared: Vec<String> = a
        .shared_form()
        .fields
        .iter()
        .map(|f| f.long.clone())
        .collect();
    assert_eq!(shared, ["steps", "sort-dim"]);

    key(&mut a, KeyCode::Char('e'));
    key(&mut a, KeyCode::Backspace);
    key(&mut a, KeyCode::Backspace);
    typed(&mut a, "7");
    key(&mut a, KeyCode::Enter);
    let steps = |a: &App, r: usize| {
        a.rows[r]
            .form
            .fields
            .iter()
            .find(|f| f.long == "steps")
            .unwrap()
            .value
            .clone()
    };
    assert_eq!(
        (steps(&a, topic), steps(&a, simba)),
        ("7".into(), "7".into())
    );

    // Set apart on one method, the shared page says they differ, and an
    // edit with nothing typed leaves each its own.
    a.rows[simba].form.fields[0].value = "3".into();
    assert!(a.differs("steps"));
    assert_eq!(a.shared_form().fields[0].value, "");
    key(&mut a, KeyCode::Char('e'));
    key(&mut a, KeyCode::Enter);
    assert_eq!(
        (steps(&a, topic), steps(&a, simba)),
        ("7".into(), "3".into())
    );
    key(&mut a, KeyCode::Char('r'));
    assert_eq!(
        (steps(&a, topic), steps(&a, simba)),
        ("10".into(), "10".into())
    );

    // With one fit queued there is nothing to share.
    a.rows[simba].on = false;
    a.settle();
    assert!(!a.shared_page);

    // A flag the fits hold at different values is listed even when
    // advanced, and a choice offers only the values every fit takes.
    let mut a = app(dir.path());
    let (topic, vae) = (row_of(&a, "topic"), row_of(&a, "vae"));
    a.rows[topic].on = true;
    a.rows[vae].on = true;
    fn kind_of(a: &mut App, r: usize) -> &mut Field {
        a.rows[r]
            .form
            .fields
            .iter_mut()
            .find(|f| f.long == "feature-name-kind")
            .unwrap()
    }
    kind_of(&mut a, topic).advanced = true;
    kind_of(&mut a, vae).kind = Kind::Choice(vec!["auto".into(), "gene".into()]);
    let shared = |a: &App| {
        a.shared_form()
            .fields
            .into_iter()
            .find(|f| f.long == "feature-name-kind")
            .unwrap()
    };
    assert_eq!(
        shared(&a).kind,
        Kind::Choice(vec!["auto".into(), "gene".into()])
    );
    assert!(shared(&a).advanced, "the same in both: advanced as it is");

    // Cycling on the shared page steps through those values only.
    let target = Target::Shared("feature-name-kind".into());
    a.change_field(&target, |f| f.cycle(1));
    assert_eq!(kind_of(&mut a, topic).value, "gene");
    assert_eq!(kind_of(&mut a, vae).value, "gene");

    kind_of(&mut a, vae).value = "auto".into();
    let f = shared(&a);
    assert_eq!(f.value, "");
    assert!(!f.advanced, "they differ: listed");
}

/// With two fits or more queued the parameters open on what they share:
/// on arriving by Tab or number, and on queuing the second fit. Enter on a
/// fit already queued, or on a third, opens that fit's own page.
#[test]
fn the_parameters_open_on_the_shared_page() {
    let dir = tempfile::tempdir().unwrap();
    let mut a = app(dir.path());
    let (topic, simba, vae) = (row_of(&a, "topic"), row_of(&a, "simba"), row_of(&a, "vae"));
    a.screen = Screen::Methods;
    a.methods_cursor.0 = topic;
    key(&mut a, KeyCode::Enter);
    assert_eq!(a.screen, Screen::Params);
    assert!(!a.shared_page, "one fit: its own page");

    a.screen = Screen::Methods;
    a.methods_cursor.0 = simba;
    key(&mut a, KeyCode::Enter);
    assert_eq!(a.screen, Screen::Params);
    assert!(a.shared_page, "the second fit: the shared page");
    key(&mut a, KeyCode::Char(']'));
    assert!(!a.shared_page, "] goes on to each fit's own");

    a.screen = Screen::Methods;
    key(&mut a, KeyCode::Enter);
    assert!(!a.shared_page, "a fit already queued: its own page");
    assert_eq!(a.param_method, simba);
    a.screen = Screen::Methods;
    a.methods_cursor.0 = vae;
    key(&mut a, KeyCode::Enter);
    assert!(!a.shared_page, "a third fit: its own page");
    assert_eq!(a.param_method, vae);

    for by in [KeyCode::Tab, KeyCode::Char('3')] {
        a.screen = Screen::Methods;
        key(&mut a, by);
        assert_eq!(a.screen, Screen::Params);
        assert!(a.shared_page, "arriving by {by:?}");
        key(&mut a, KeyCode::Char(']'));
    }
    // Already on the parameters, a number key stays on the page shown.
    key(&mut a, KeyCode::Char('3'));
    assert!(!a.shared_page);
}

#[test]
fn enter_on_a_feature_table_flag_picks_the_newest_table() {
    let dir = tempfile::tempdir().unwrap();
    let mut a = app(dir.path());
    std::fs::write(dir.path().join("old.feature_embedding.parquet"), b"").unwrap();
    std::fs::write(dir.path().join("notes.txt"), b"").unwrap();
    // Written later, none of these is what the flag takes: a topic run's β,
    // and an svd run's manifest and signed loadings, which are not ρ.
    std::fs::write(dir.path().join("topic.dictionary.parquet"), b"").unwrap();
    let svd = dir.path().join("svd").to_string_lossy().into_owned();
    senna::run_manifest::RunManifest::new(senna::run_manifest::RunKind::Svd, &svd)
        .save(&dir.path().join("svd.senna.json"))
        .unwrap();
    let genes: Vec<Box<str>> = vec!["A".into(), "B".into()];
    use legume_numeric::matrix::traits::IoOps;
    nalgebra::DMatrix::<f32>::from_row_slice(2, 2, &[0.5, -0.5, -0.5, 0.5])
        .to_parquet_with_names(
            &format!("{svd}.dictionary.parquet"),
            (Some(&genes), Some("gene")),
            None,
        )
        .unwrap();
    let t0 = std::time::SystemTime::now() - std::time::Duration::from_secs(60);
    for (i, n) in [
        "old.feature_embedding.parquet",
        "topic.dictionary.parquet",
        "svd.senna.json",
        "svd.dictionary.parquet",
    ]
    .iter()
    .enumerate()
    {
        std::fs::File::options()
            .write(true)
            .open(dir.path().join(n))
            .unwrap()
            .set_modified(t0 + std::time::Duration::from_secs(10 * i as u64))
            .unwrap();
    }
    let bge = row_of(&a, "bge");
    a.rows[bge].on = true;
    a.show_params(bge);
    let i = a.rows[bge]
        .form
        .fields
        .iter()
        .position(|f| f.long == "lora-feature-embedding")
        .unwrap();
    a.field_row = a.visible().iter().position(|&v| v == i).unwrap();
    key(&mut a, KeyCode::Enter);
    let b = a.browser.as_ref().expect("a browser opens");
    let listed: Vec<&str> = b.shown().iter().map(|e| e.name()).collect();
    assert!(listed.contains(&"old.feature_embedding.parquet"));
    assert!(!listed.contains(&"notes.txt"));
    key(&mut a, KeyCode::Enter);
    assert!(a.browser.is_none());
    // In full: the fit runs in its own output folder.
    assert_eq!(
        a.rows[bge].form.fields[i].value,
        dir.path()
            .join("old.feature_embedding.parquet")
            .to_string_lossy()
    );
}

#[test]
fn fits_with_the_same_collapse_share_one_partition() {
    let dir = tempfile::tempdir().unwrap();
    let mut a = app(dir.path());
    a.pairs = data(dir.path(), &["a.zarr"]);
    let at = |a: &App, name: &str| a.rows.iter().position(|r| r.label == name).unwrap();
    let (topic, vae, masked, svd) = (
        at(&a, "topic"),
        at(&a, "vae"),
        at(&a, "masked-topic"),
        at(&a, "svd"),
    );
    for r in [topic, vae, masked, svd] {
        a.rows[r].on = true;
    }
    // Training flags differ; the collapse does not, but for masked-topic's.
    a.rows[vae].form.fields[0].value = "99".into();
    let sort_dim = |a: &mut App, r: usize| {
        a.rows[r]
            .form
            .fields
            .iter_mut()
            .find(|f| f.long == "sort-dim")
            .unwrap()
            .value = "8".into();
    };
    sort_dim(&mut a, masked);
    let plan = a.plan();
    let by = |name: &str| plan.iter().find(|p| p.job.method == name).unwrap();
    assert_eq!(by("topic").job.pb_from, None);
    assert_eq!(by("vae").job.pb_from.as_deref(), Some("topic.senna.json"));
    assert!(by("vae")
        .job
        .command()
        .ends_with(&["--pb-from".into(), "topic.senna.json".into()]));
    assert!(by("vae")
        .warning
        .as_deref()
        .unwrap()
        .contains("topic's pseudobulks"));
    assert_eq!(by("masked-topic").job.pb_from, None);
    // svd builds no partition.
    assert_eq!(by("svd").job.pb_from, None);
    assert!(
        plan.iter().all(|p| p.problem.is_none()),
        "{:?}",
        plan.iter().map(|p| &p.problem).collect::<Vec<_>>()
    );

    // Row names matched another way make another feature axis: no sharing.
    a.rows[vae]
        .form
        .fields
        .iter_mut()
        .find(|f| f.long == "feature-name-kind")
        .unwrap()
        .value = "gene".into();
    let plan = a.plan();
    let by = |name: &str| plan.iter().find(|p| p.job.method == name).unwrap();
    assert_eq!(by("vae").job.pb_from, None);
}

#[test]
fn a_flag_whose_default_differs_between_fits_stays_on_each_fits_page() {
    let dir = tempfile::tempdir().unwrap();
    let mut a = app(dir.path());
    let (vae, svd) = (row_of(&a, "vae"), row_of(&a, "svd"));
    a.rows[vae].on = true;
    a.rows[svd].on = true;
    let shared: Vec<String> = a.shared_form().fields.into_iter().map(|f| f.long).collect();
    assert!(shared.contains(&"steps".to_string()), "{shared:?}");
    assert!(!shared.contains(&"depth".to_string()), "{shared:?}");
    for m in [vae, svd] {
        assert!(a.rows[m].form.fields.iter().any(|f| f.long == "depth"));
    }
}
