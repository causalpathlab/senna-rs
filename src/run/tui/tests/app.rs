use super::*;

fn cli() -> clap::Command {
    // A stand-in for senna's command: every method takes data files,
    // --out and batch files, and one flag of its own.
    let method = |name: &'static str| {
        clap::Command::new(name)
            .about("fit")
            .arg(
                clap::Arg::new("data_files")
                    .num_args(1..)
                    .value_delimiter(','),
            )
            .arg(clap::Arg::new("out").long("out").required(true))
            .arg(
                clap::Arg::new("batch_files")
                    .long("batch-files")
                    .value_delimiter(','),
            )
            .arg(
                clap::Arg::new("steps")
                    .long("steps")
                    .default_value("10")
                    .value_parser(clap::value_parser!(usize)),
            )
    };
    let mut c = clap::Command::new("senna").subcommands(METHODS.map(method));
    c.build();
    c
}

fn app(dir: &Path) -> App {
    let mut a = App {
        here: dir.to_path_buf(),
        ..App::new(cli(), dir.to_path_buf()).unwrap()
    };
    a.browser = None;
    for m in &mut a.rows {
        m.out = free_out(dir, &m.form.name);
    }
    a
}

fn key(a: &mut App, c: KeyCode) {
    a.key(KeyEvent::new(c, KeyModifiers::NONE));
}

fn data(dir: &Path, names: &[&str]) -> Vec<Pair> {
    names
        .iter()
        .map(|n| {
            let p = dir.join(n);
            std::fs::write(&p, "").unwrap();
            Pair {
                data: p,
                batch: None,
                info: String::new(),
                gene_counts: None,
            }
        })
        .collect()
}

#[test]
fn an_out_already_used_gets_the_next_free_name() {
    let dir = tempfile::tempdir().unwrap();
    assert_eq!(free_out(dir.path(), "svd"), "svd");
    std::fs::write(dir.path().join("svd.senna.json"), "{}").unwrap();
    std::fs::write(dir.path().join("svd-2.cmd.sh"), "").unwrap();
    assert_eq!(free_out(dir.path(), "svd"), "svd-3");
}

#[test]
fn queued_methods_share_the_data_and_write_their_own_out() {
    let dir = tempfile::tempdir().unwrap();
    let mut a = app(dir.path());
    a.pairs = data(dir.path(), &["d1.zarr.zip", "d2.zarr.zip"]);
    a.pairs[0].batch = Some(dir.path().join("b1.tsv"));
    a.pairs[1].batch = Some(dir.path().join("b2.tsv"));
    a.screen = Screen::Methods;
    a.method_row = METHODS.iter().position(|m| *m == "svd").unwrap();
    key(&mut a, KeyCode::Char(' '));
    a.method_row = METHODS.iter().position(|m| *m == "bge").unwrap();
    key(&mut a, KeyCode::Char(' '));
    let planned = a.plan();
    assert_eq!(planned.len(), 2);
    for p in &planned {
        assert_eq!(p.problem, None);
        assert_eq!(
            p.job.argv,
            [
                p.job.method.as_str(),
                "d1.zarr.zip",
                "d2.zarr.zip",
                "--batch-files",
                "b1.tsv,b2.tsv",
                "--out",
                p.job.method.as_str()
            ]
        );
    }
}

#[test]
fn what_would_overwrite_or_misparse_is_stopped_before_running() {
    let dir = tempfile::tempdir().unwrap();
    let mut a = app(dir.path());
    let svd = METHODS.iter().position(|m| *m == "svd").unwrap();
    a.rows[svd].on = true;
    assert!(a.plan()[0].problem.as_ref().unwrap().contains("no data"));

    a.pairs = data(dir.path(), &["d1.zarr", "d2.zarr"]);
    a.pairs[0].batch = Some(dir.path().join("b1.tsv"));
    assert!(a.plan()[0].problem.as_ref().unwrap().contains("batch"));
    a.pairs[0].batch = None;

    std::fs::write(dir.path().join("svd.senna.json"), "{}").unwrap();
    assert!(a.plan()[0].problem.as_ref().unwrap().contains("exists"));
    a.rows[svd].out = "sub/r".into();
    assert!(a.plan()[0]
        .problem
        .as_ref()
        .unwrap()
        .contains("not a folder"));
    std::fs::create_dir(dir.path().join("sub")).unwrap();
    let p = &a.plan()[0];
    assert_eq!(p.problem, None);
    assert_eq!(p.job.dir, script::normalize(&dir.path().join("sub")));
    assert_eq!(p.job.argv[1], "../d1.zarr");

    let steps = a.rows[svd]
        .form
        .fields
        .iter()
        .position(|f| f.long == "steps")
        .unwrap();
    a.rows[svd].form.fields[steps].value = "many".into();
    let p = &a.plan()[0];
    assert!(p.problem.is_some());
    assert_eq!(p.blamed.as_deref(), Some("steps"));
}

#[test]
fn enter_on_a_problem_goes_to_the_flag_clap_blamed() {
    let dir = tempfile::tempdir().unwrap();
    let mut a = app(dir.path());
    a.pairs = data(dir.path(), &["d.zarr"]);
    let svd = METHODS.iter().position(|m| *m == "svd").unwrap();
    a.rows[svd].on = true;
    a.rows[svd].form.fields[0].value = "many".into();
    key(&mut a, KeyCode::Char('g'));
    assert!(a.confirm.is_some());
    key(&mut a, KeyCode::Enter);
    assert!(a.confirm.is_none() && a.queue.is_none());
    assert_eq!(a.screen, Screen::Params);
    assert_eq!(a.param_method, svd);
}

#[test]
fn flags_are_changed_and_typed_on_the_parameters_screen() {
    let dir = tempfile::tempdir().unwrap();
    let mut a = app(dir.path());
    a.screen = Screen::Methods;
    key(&mut a, KeyCode::Enter);
    assert_eq!(a.screen, Screen::Params);
    assert!(a.rows[0].on);
    key(&mut a, KeyCode::Enter);
    key(&mut a, KeyCode::Backspace);
    key(&mut a, KeyCode::Backspace);
    key(&mut a, KeyCode::Char('5'));
    key(&mut a, KeyCode::Enter);
    assert_eq!(a.rows[0].form.fields[0].value, "5");
    assert_eq!(a.rows[0].form.changed(), 1);
    key(&mut a, KeyCode::Char('r'));
    assert_eq!(a.rows[0].form.changed(), 0);
}

#[test]
fn the_filter_narrows_the_flags() {
    let dir = tempfile::tempdir().unwrap();
    let mut a = app(dir.path());
    a.screen = Screen::Params;
    assert_eq!(a.visible().len(), 1);
    key(&mut a, KeyCode::Char('/'));
    key(&mut a, KeyCode::Char('z'));
    assert!(a.visible().is_empty());
    key(&mut a, KeyCode::Esc);
    assert_eq!(a.visible().len(), 1);
}

#[test]
fn every_method_of_senna_parses_from_an_untouched_form() {
    use clap::CommandFactory;
    let mut cli = crate::Cli::command();
    cli.build();
    for m in METHODS {
        let form = Method::new(&cli, m).unwrap();
        assert!(!form.fields.is_empty(), "{m} has no flags");
        assert!(form.takes_batches(), "{m} takes no batch files");
        let argv = form.argv(
            &["d1.zarr".into(), "d2.zarr".into()],
            &["b1.tsv".into(), "b2.tsv".into()],
            "o",
        );
        assert_eq!(form::check(&cli, &argv), Ok(()), "{m}: {argv:?}");
        // A switch flipped on reaches the command line as clap takes it.
        let mut form = form;
        if let Some(f) = form
            .fields
            .iter_mut()
            .find(|f| matches!(f.kind, Kind::Flag { .. }) && !f.advanced)
        {
            f.toggle();
            let argv = form.argv(&["d.zarr".into()], &[], "o");
            assert_eq!(form::check(&cli, &argv), Ok(()), "{m}: {argv:?}");
        }
    }
}

#[test]
fn an_out_folder_given_with_dotdot_still_finds_the_data() {
    let dir = tempfile::tempdir().unwrap();
    let here = dir.path().join("b");
    std::fs::create_dir_all(&here).unwrap();
    std::fs::create_dir_all(dir.path().join("res")).unwrap();
    let mut a = app(&here);
    a.pairs = data(&here, &["d.zarr"]);
    let svd = METHODS.iter().position(|m| *m == "svd").unwrap();
    a.rows[svd].on = true;
    a.rows[svd].out = "../res/r".into();
    let p = &a.plan()[0];
    assert_eq!(p.problem, None);
    assert_eq!(p.job.argv[1], "../b/d.zarr");
    assert_eq!(p.job.dir, script::normalize(&dir.path().join("res")));
}

#[test]
fn a_parent_start_folder_is_resolved() {
    let a = App::new(cli(), PathBuf::from("..")).unwrap();
    let here = std::env::current_dir().unwrap();
    assert_eq!(a.browse_dir, here.parent().unwrap());
}

#[test]
fn a_flag_with_a_default_cannot_be_cleared_to_unset() {
    let dir = tempfile::tempdir().unwrap();
    let mut a = app(dir.path());
    a.screen = Screen::Params;
    key(&mut a, KeyCode::Enter);
    key(&mut a, KeyCode::Backspace);
    key(&mut a, KeyCode::Backspace);
    key(&mut a, KeyCode::Enter);
    let f = &a.rows[0].form.fields[0];
    assert_eq!(f.value, "10");
    assert!(a.message.as_deref().unwrap().contains("default 10"));
}

#[test]
fn the_confirm_popup_scrolls_no_further_than_its_end() {
    let dir = tempfile::tempdir().unwrap();
    let mut a = app(dir.path());
    a.confirm = Some(Vec::new());
    a.confirm_max.set(3);
    for _ in 0..10 {
        key(&mut a, KeyCode::Down);
    }
    assert_eq!(a.confirm_scroll, 3);
    key(&mut a, KeyCode::Up);
    assert_eq!(a.confirm_scroll, 2);
}

#[test]
fn batch_files_are_described_as_paired() {
    let dir = tempfile::tempdir().unwrap();
    let mut a = app(dir.path());
    a.pairs = data(dir.path(), &["s1.zarr", "s2.zarr"]);
    a.take_batches(&[dir.path().join("x.tsv"), dir.path().join("y.tsv")]);
    assert!(a
        .message
        .as_deref()
        .unwrap()
        .contains("in the order listed"));
}
