//! The run screen: a fit given whole, the queue log, progress bars, reruns
//! and a finished fit's open choice.

use super::*;

fn words(w: &[&str]) -> Vec<String> {
    w.iter().map(|s| (*s).to_string()).collect()
}

#[test]
fn a_fit_given_whole_runs_from_here_as_given_its_script_beside_its_out() {
    let tmp = tempfile::tempdir().unwrap();
    let here = tmp.path();
    // Every word as given, a flag's value too: it runs from here.
    std::fs::write(here.join("5"), "").unwrap();
    std::fs::create_dir(here.join("runs")).unwrap();
    let a = app(here);
    for out in [
        &["-o", "runs/fit"][..],
        &["--out", "runs/fit"],
        &["--out=runs/fit"],
    ] {
        let mut argv = words(&["svd", "data.zarr"]);
        argv.extend(words(out));
        argv.extend(words(&["--steps", "5", "--batch-files", "b.txt"]));
        let job = a.fit_job(&argv).unwrap();
        assert_eq!(job.tool, Tool::Senna);
        assert_eq!(job.method, a.rows[job.row].label);
        assert_eq!(a.rows[job.row].form.name, "svd");
        // Run from here, as it was given; its script beside its out.
        assert_eq!(job.dir, here);
        assert_eq!(job.out, "runs/fit");
        assert_eq!(job.script(), here.join("runs/fit.cmd.sh"));
        // As given: it runs from here.
        let data = "data.zarr";
        assert_eq!(
            job.argv,
            words(&[
                "svd",
                data,
                "--out",
                "runs/fit",
                "--steps",
                "5",
                "--batch-files",
                "b.txt"
            ]),
            "{out:?}"
        );
        assert_eq!(job.result(), here.join("runs/fit.senna.json"));
        assert_eq!(job.lane, 0);
    }
    // What it cannot be: no method, one senna has not, or no --out.
    assert!(a.fit_job(&[]).is_err());
    assert!(a
        .fit_job(&words(&["nope", "data.zarr", "-o", "x"]))
        .is_err());
    assert!(a.fit_job(&words(&["svd", "data.zarr"])).is_err());
    // Its out out of here, the data is named whole.
    let away = tempfile::tempdir().unwrap();
    let out = away.path().join("fit").to_string_lossy().into_owned();
    let job = a
        .fit_job(&words(&["svd", "data.zarr", "--out", &out]))
        .unwrap();
    assert_eq!(job.out, out);
    let data = here.join("data.zarr").to_string_lossy().into_owned();
    assert_eq!(job.argv, words(&["svd", &data, "--out", &out]));
}

#[test]
fn a_fit_given_whole_starts_on_the_run_screen_and_ends_as_its_job_did() {
    let tmp = tempfile::tempdir().unwrap();
    let mut a = app(tmp.path());
    // Not this test binary: a program that ends at once.
    a.senna_program = PathBuf::from("true");
    a.start_fit(&words(&["svd", "data.zarr", "-o", "fit"]))
        .unwrap();
    assert!(a.screen == Screen::Run);
    assert_eq!(a.queue.as_ref().map(|q| q.jobs.len()), Some(1));
    a.queue
        .as_ref()
        .unwrap()
        .stop_within(Duration::from_secs(5));
    use legume_mcp::job::End;
    let manifest = tmp.path().join("fit.senna.json");
    assert_eq!(
        fit_end(&jobs::State::Done, &manifest),
        End::Done(manifest.display().to_string())
    );
    assert_eq!(fit_end(&jobs::State::Stopped, &manifest), End::Cancelled);
    assert_eq!(
        fit_end(&jobs::State::Failed("diverged".into()), &manifest),
        End::Failed("diverged".into())
    );
}

#[test]
fn the_queue_log_goes_to_the_job_once_each_line() {
    let mut shared = jobs::Shared::default();
    let mut sent = Sent::default();
    shared.log.extend(["a".to_string(), "b".to_string()]);
    assert_eq!(sent.new_lines(&shared), ["a", "b"]);
    assert!(sent.new_lines(&shared).is_empty());
    // Lines that went off the front unsent are lost, not sent twice.
    shared
        .log
        .extend(["c".to_string(), "d".to_string(), "e".to_string()]);
    shared.log.drain(..3);
    shared.dropped += 3;
    assert_eq!(sent.new_lines(&shared), ["d", "e"]);
}

/// A fit's bar becomes a stage with the times its frame drew; a bar with no
/// message of its own is labelled by the fit's newest log line, and a
/// spinner has no end and no time left.
#[test]
fn a_bar_is_a_stage_with_the_times_its_frame_drew() {
    use legume_tui::child::Progress;
    let secs = Duration::from_secs;
    let bar = Progress::new(200, 1000, "fit gap=0.41").timed(Some(secs(10)), Some(secs(40)));
    let s = stage_of(&bar, "epoch 200/1000");
    assert_eq!(s.label, "fit gap=0.41");
    assert_eq!(
        (s.pos, s.len, s.elapsed_s, s.eta_s),
        (200, Some(1000), 10, Some(40))
    );

    let bare = Progress::new(5, 10, "");
    assert_eq!(stage_of(&bare, "  epoch 200/1000 ").label, "epoch 200/1000");

    let spin = Progress::new(37, 0, "streamed").timed(Some(secs(3)), Some(secs(0)));
    let s = stage_of(&spin, "");
    assert_eq!((s.len, s.eta_s), (None, None));
}

#[test]
fn a_rerun_sets_its_out_and_the_options_flags() {
    let argv = words(&["ode", "d.zarr", "--out", "fit", "--root-cells", "old.txt"]);
    assert_eq!(
        rerun_argv(
            &argv,
            "fit_root1",
            &words(&["--root-cells", "r1.txt", "--trunk", "x"])
        ),
        words(&[
            "ode",
            "d.zarr",
            "--out",
            "fit_root1",
            "--root-cells",
            "r1.txt",
            "--trunk",
            "x"
        ])
    );
}

/// A fit that ends with a choice left open puts it to the user on the Run
/// screen; enter reruns with the option picked, as its own out.
#[test]
fn a_finished_fits_open_choice_is_put_and_reruns_with_the_pick() {
    use senna::run_manifest::{Decision, DecisionOption, RunKind, RunManifest};
    let tmp = tempfile::tempdir().unwrap();
    let mut a = app(tmp.path());
    a.senna_program = PathBuf::from("true");
    a.start_fit(&words(&["ode", "data.zarr", "-o", "fit"]))
        .unwrap();
    a.queue.as_ref().unwrap().join();
    // What the fit left: two roots to choose from.
    let mut m = RunManifest::new(RunKind::Ode, "fit");
    m.decisions.push(Decision {
        topic: "root".into(),
        question: "Which root's cells start it?".into(),
        hint: String::new(),
        options: (0..2)
            .map(|i| DecisionOption {
                label: format!("root {i}"),
                rerun_args: words(&["--root-cells", &format!("fit.root_{i}.txt")]),
                objective: None,
                cells: None,
            })
            .collect(),
    });
    m.save(&tmp.path().join("fit.senna.json")).unwrap();
    a.poll();
    assert!(a.choice.is_some());
    let shown = drawn(&a, 110, 30);
    assert!(shown.contains("Which root's cells start it?"), "{shown}");
    assert!(shown.contains("root 1"), "{shown}");
    key(&mut a, KeyCode::Down);
    key(&mut a, KeyCode::Enter);
    assert!(a.choice.is_none());
    let q = a.queue.as_ref().unwrap();
    assert_eq!(q.jobs[0].out, "fit_root1");
    assert_eq!(
        q.jobs[0].argv,
        words(&[
            "ode",
            "data.zarr",
            "--out",
            "fit_root1",
            "--root-cells",
            "fit.root_1.txt"
        ])
    );
    q.join();
    // Esc leaves a choice open, as the manifest keeps it.
    std::fs::copy(
        tmp.path().join("fit.senna.json"),
        tmp.path().join("fit_root1.senna.json"),
    )
    .unwrap();
    a.poll();
    assert!(a.choice.is_some());
    key(&mut a, KeyCode::Esc);
    assert!(a.choice.is_none());
}
