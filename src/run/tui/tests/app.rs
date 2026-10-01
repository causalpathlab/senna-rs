use super::*;

fn cli() -> clap::Command {
    // A stand-in for senna's command: every method takes data files,
    // --out and batch files, and one flag of its own.
    let method = |name: &'static str| {
        let c = clap::Command::new(name)
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
            );
        // All but one take the clones of `mung clones`.
        if name == "simba" {
            c
        } else {
            c.arg(clap::Arg::new("cnv_clones").long("cnv-clones"))
        }
    };
    let mut c = clap::Command::new("senna").subcommands(METHODS.map(method));
    c.build();
    c
}

/// An app in `dir`, with mung or not.
fn app_from(dir: &Path, mung: Result<clap::Command, String>) -> App {
    let mut a = App {
        here: dir.to_path_buf(),
        ..App::new(cli(), mung, dir.to_path_buf()).unwrap()
    };
    a.browser = None;
    for m in &mut a.rows {
        let name = if m.tool == Tool::Mung {
            "cnv"
        } else {
            m.form.name.as_str()
        };
        m.out = free_out(dir, name);
    }
    a
}

fn app(dir: &Path) -> App {
    app_from(dir, Err("no mung".into()))
}

/// A stand-in for `mung` as `mung describe clones` tells it.
fn mung() -> clap::Command {
    let mut c = clap::Command::new("mung").subcommand(
        clap::Command::new("clones")
            .about("clones")
            .arg(clap::Arg::new("query").num_args(1..))
            .arg(clap::Arg::new("out").long("out").required(true))
            .arg(clap::Arg::new("gff").long("gff").required(true)),
    );
    c.build();
    c
}

/// An app whose first row is the `mung clones` step.
fn app_with_mung(dir: &Path) -> App {
    app_from(dir, Ok(mung()))
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
                batch: Batch::Own,
                info: String::new(),
                cells: Some(3),
                tags: None,
                gene_counts: None,
            }
        })
        .collect()
}

/// A label file `name` in `dir` with `labels`, one per line, as a batch.
fn labels(dir: &Path, name: &str, labels: &[&str]) -> Batch {
    let path = dir.join(name);
    std::fs::write(&path, labels.join("\n") + "\n").unwrap();
    Batch::labels(path).unwrap()
}

fn typed(a: &mut App, text: &str) {
    for c in text.chars() {
        key(a, KeyCode::Char(c));
    }
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
    a.pairs[0].batch = labels(dir.path(), "b1.tsv", &["x", "x", "y"]);
    a.pairs[1].batch = labels(dir.path(), "b2.tsv", &["x", "z", "z"]);
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
    a.pairs[0].batch = Batch::Named("b".into());
    a.pairs[1].cells = None;
    assert!(a.plan()[0]
        .problem
        .as_ref()
        .unwrap()
        .contains("not counted yet"));
    a.pairs[0].batch = Batch::Own;

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
    key(&mut a, KeyCode::Char('G'));
    assert!(a.confirm.is_some());
    key(&mut a, KeyCode::Char('G'));
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
    assert_eq!(a.visible().len(), 2);
    key(&mut a, KeyCode::Char('/'));
    key(&mut a, KeyCode::Char('s'));
    key(&mut a, KeyCode::Char('t'));
    assert_eq!(a.visible().len(), 1);
    key(&mut a, KeyCode::Char('z'));
    assert!(a.visible().is_empty());
    key(&mut a, KeyCode::Esc);
    assert_eq!(a.visible().len(), 2);
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
    let a = App::new(cli(), Err("no mung".into()), PathBuf::from("..")).unwrap();
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
    for f in ["x.tsv", "y.tsv"] {
        std::fs::write(dir.path().join(f), "a\nb\nc\n").unwrap();
    }
    a.take_batches(&[dir.path().join("x.tsv"), dir.path().join("y.tsv")]);
    assert!(a
        .message
        .as_deref()
        .unwrap()
        .contains("in the order listed"));
}

#[test]
fn files_named_alike_are_one_batch_written_beside_the_script() {
    let dir = tempfile::tempdir().unwrap();
    let mut a = app(dir.path());
    a.pairs = data(dir.path(), &["d1.zarr", "d2.zarr", "d3.zarr"]);
    let svd = METHODS.iter().position(|m| *m == "svd").unwrap();
    a.rows[svd].on = true;
    // Every file on its own name: senna's rule, no label files.
    let p = &a.plan()[0];
    assert!(!p.job.argv.contains(&"--batch-files".to_string()));
    assert!(p.job.labels.is_empty());

    a.screen = Screen::Data;
    for row in [0, 1] {
        a.pair_row = row;
        key(&mut a, KeyCode::Char('n'));
        for _ in 0..10 {
            key(&mut a, KeyCode::Backspace);
        }
        typed(&mut a, "b1");
        key(&mut a, KeyCode::Enter);
    }
    assert_eq!(a.pairs[0].batch, Batch::Named("b1".into()));
    assert_eq!(
        batches::summary(&a.pairs),
        [
            ("b1".to_string(), 2, Some(6)),
            ("d3".to_string(), 1, Some(3))
        ]
    );
    let p = &a.plan()[0];
    assert_eq!(p.problem, None);
    let at = p
        .job
        .argv
        .iter()
        .position(|w| w == "--batch-files")
        .unwrap();
    assert_eq!(
        p.job.argv[at + 1],
        "svd.batches/d1.txt,svd.batches/d2.txt,svd.batches/d3.txt"
    );
    assert_eq!(p.job.labels.len(), 3);

    // An empty name goes back to the file's own.
    a.pair_row = 1;
    key(&mut a, KeyCode::Char('n'));
    for _ in 0..10 {
        key(&mut a, KeyCode::Backspace);
    }
    key(&mut a, KeyCode::Enter);
    assert_eq!(a.pairs[1].batch, Batch::Own);
}

#[test]
fn labels_are_renamed_from_their_list() {
    let dir = tempfile::tempdir().unwrap();
    let mut a = app(dir.path());
    a.pairs = data(dir.path(), &["d1.zarr"]);
    a.pairs[0].batch = labels(dir.path(), "b1.tsv", &["x", "y", "x"]);
    a.screen = Screen::Data;
    key(&mut a, KeyCode::Char('e'));
    assert_eq!(a.relabel, Some((0, 0)));
    key(&mut a, KeyCode::Down);
    key(&mut a, KeyCode::Enter);
    key(&mut a, KeyCode::Backspace);
    typed(&mut a, "x");
    key(&mut a, KeyCode::Enter);
    key(&mut a, KeyCode::Esc);
    assert!(a.relabel.is_none());
    assert_eq!(batches::summary(&a.pairs), [("x".to_string(), 1, Some(3))]);
    let svd = METHODS.iter().position(|m| *m == "svd").unwrap();
    a.rows[svd].on = true;
    let p = &a.plan()[0];
    assert!(matches!(
        p.job.labels[0].content,
        batches::Content::Renamed { .. }
    ));
}

#[test]
fn the_parameters_screen_shows_a_queued_method_however_it_is_reached() {
    let dir = tempfile::tempdir().unwrap();
    let mut a = app(dir.path());
    a.screen = Screen::Methods;
    let bge = METHODS.iter().position(|m| *m == "bge").unwrap();
    a.method_row = bge;
    key(&mut a, KeyCode::Char(' '));
    key(&mut a, KeyCode::Char('3'));
    assert_eq!(a.screen, Screen::Params);
    assert_eq!(a.param_method, bge);
}

#[test]
fn without_mung_there_is_no_clones_step_and_the_screen_says_why() {
    let dir = tempfile::tempdir().unwrap();
    let a = app(dir.path());
    assert!(a.rows.iter().all(|r| r.tool == Tool::Senna));
    assert_eq!(a.mung.as_ref().unwrap_err(), "no mung");
}

#[test]
fn queued_clones_go_to_every_fit_that_takes_them() {
    let dir = tempfile::tempdir().unwrap();
    let mut a = app_with_mung(dir.path());
    a.pairs = data(dir.path(), &["d.zarr"]);
    assert_eq!(a.rows[0].tool, Tool::Mung);
    assert_eq!(a.rows[0].label, "mung clones");
    assert_eq!(a.rows[0].out, "cnv");
    let row = |name: &str| a.rows.iter().position(|r| r.form.name == name).unwrap();
    let (svd, simba) = (row("svd"), row("simba"));
    a.rows[svd].on = true;
    a.rows[simba].on = true;
    // Without the step, nothing is added.
    assert!(a.plan().iter().all(|p| p.job.clones.is_none()));

    a.rows[0].on = true;
    let gff = a.rows[0]
        .form
        .fields
        .iter()
        .position(|f| f.long == "gff")
        .unwrap();
    let p = a.plan();
    assert_eq!(p[0].job.tool, Tool::Mung);
    assert_eq!(
        p[0].blamed.as_deref(),
        Some("gff"),
        "clap checks mung's line"
    );
    a.rows[0].form.fields[gff].value = "g.gtf".into();
    let p = a.plan();
    assert_eq!(p[0].problem, None);
    assert_eq!(p[0].job.argv[..3], ["clones", "d.zarr", "--out"]);
    assert!(p[0].job.result().ends_with("cnv.clones.parquet"));
    let svd_job = &p.iter().find(|p| p.job.method == "svd").unwrap().job;
    assert_eq!(svd_job.clones.as_deref(), Some("cnv.clones.parquet"));
    assert!(svd_job
        .command()
        .ends_with(&["--cnv-clones".to_string(), "cnv.clones.parquet".to_string()]));
    let simba_plan = p.iter().find(|p| p.job.method == "simba").unwrap();
    assert!(simba_plan.job.clones.is_none());
    assert!(simba_plan
        .warning
        .as_ref()
        .unwrap()
        .contains("runs without"));
    assert!(p.iter().all(|p| p.problem.is_none()));

    // A table given by hand and the step both: one has to go.
    let by_hand = a.rows[svd]
        .form
        .fields
        .iter()
        .position(|f| f.long == "cnv-clones")
        .unwrap();
    a.rows[svd].form.fields[by_hand].value = "x.parquet".into();
    let p = a.plan();
    let svd_plan = p.iter().find(|p| p.job.method == "svd").unwrap();
    assert!(svd_plan.problem.as_ref().unwrap().contains("reset one"));
}

#[test]
fn an_existing_clone_table_is_not_run_over() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("cnv.clones.parquet"), "").unwrap();
    let mut a = app_with_mung(dir.path());
    assert_eq!(a.rows[0].out, "cnv-2", "a free prefix is offered");
    a.rows[0].out = "cnv".into();
    a.rows[0].on = true;
    a.pairs = data(dir.path(), &["d.zarr"]);
    assert!(a.plan()[0].problem.as_ref().unwrap().contains("exists"));
}
