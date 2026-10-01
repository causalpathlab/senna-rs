use super::*;

#[cfg(unix)]
#[test]
fn jobs_run_in_turn_and_leave_their_scripts() {
    use std::os::unix::fs::PermissionsExt;
    let dir = tempfile::tempdir().unwrap();
    let fake = dir.path().join("fake-senna");
    std::fs::write(
        &fake,
        "#!/bin/sh\necho \"[t INFO x] working on $1\" >&2\nwhile [ $# -gt 0 ]; do [ \"$1\" = --out ] && touch \"$2.senna.json\"; shift; done\n",
    )
    .unwrap();
    std::fs::set_permissions(&fake, std::fs::Permissions::from_mode(0o755)).unwrap();
    let job = |m: &str| Job {
        tool: Tool::Senna,
        method: m.into(),
        dir: dir.path().to_path_buf(),
        out: m.into(),
        argv: vec![m.into(), "d.zarr".into(), "--out".into(), m.into()],
        labels: Vec::new(),
        clones: false,
    };
    let q = Queue::start(vec![job("svd"), job("bge")], fake.clone(), PathBuf::new());
    while !q.finished() {
        std::thread::sleep(std::time::Duration::from_millis(10));
    }
    let s = q.shared.lock().unwrap();
    assert_eq!(s.states, [State::Done, State::Done]);
    assert_eq!(s.last, ["working on svd", "working on bge"]);
    drop(s);
    assert!(dir.path().join("svd.cmd.sh").exists());
    assert_eq!(q.done().len(), 2);
}

#[cfg(unix)]
#[test]
fn a_job_whose_result_exists_does_not_run() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("r.senna.json"), "{}").unwrap();
    let job = Job {
        tool: Tool::Senna,
        method: "svd".into(),
        dir: dir.path().to_path_buf(),
        out: "r".into(),
        argv: vec!["svd".into(), "--out".into(), "r".into()],
        labels: Vec::new(),
        clones: false,
    };
    let q = Queue::start(
        vec![job],
        PathBuf::from("/bin/false").clone(),
        PathBuf::new(),
    );
    while !q.finished() {
        std::thread::sleep(std::time::Duration::from_millis(10));
    }
    assert!(matches!(
        q.shared.lock().unwrap().states[0],
        State::Failed(_)
    ));
    assert!(!dir.path().join("r.cmd.sh").exists());
}

#[cfg(unix)]
#[test]
fn a_stopped_queue_is_waited_for_and_its_fit_killed() {
    use std::os::unix::fs::PermissionsExt;
    let dir = tempfile::tempdir().unwrap();
    let slow = dir.path().join("slow-senna");
    std::fs::write(&slow, "#!/bin/sh\nexec sleep 30\n").unwrap();
    std::fs::set_permissions(&slow, std::fs::Permissions::from_mode(0o755)).unwrap();
    let job = |m: &str| Job {
        tool: Tool::Senna,
        method: m.into(),
        dir: dir.path().to_path_buf(),
        out: m.into(),
        argv: vec![m.into(), "--out".into(), m.into()],
        labels: Vec::new(),
        clones: false,
    };
    let q = Queue::start(vec![job("a"), job("b")], slow.clone(), PathBuf::new());
    while q.shared.lock().unwrap().states[0] != State::Running {
        std::thread::sleep(std::time::Duration::from_millis(5));
    }
    std::thread::sleep(std::time::Duration::from_millis(50));
    let t = std::time::Instant::now();
    q.stop();
    q.join();
    assert!(t.elapsed() < std::time::Duration::from_secs(5));
    assert!(q.finished());
    assert_eq!(
        q.shared.lock().unwrap().states,
        [State::Stopped, State::Stopped]
    );
    assert!(!dir.path().join("b.cmd.sh").exists(), "b never started");
}

/// A stand-in program in `dir` that logs its argv and touches
/// `{out}.<end>`.
#[cfg(unix)]
fn fake(dir: &std::path::Path, name: &str, end: &str) -> PathBuf {
    use std::os::unix::fs::PermissionsExt;
    let p = dir.join(name);
    std::fs::write(
        &p,
        format!(
            "#!/bin/sh\necho \"$*\" >> argv.log\nwhile [ $# -gt 0 ]; do [ \"$1\" = --out ] && touch \"$2.{end}\"; shift; done\n"
        ),
    )
    .unwrap();
    std::fs::set_permissions(&p, std::fs::Permissions::from_mode(0o755)).unwrap();
    p
}

#[cfg(unix)]
fn clones_then_fit(dir: &std::path::Path) -> Vec<Job> {
    let job = |tool, m: &str, argv: &[&str], clones| Job {
        tool,
        method: m.into(),
        dir: dir.to_path_buf(),
        out: m.into(),
        argv: argv.iter().map(ToString::to_string).collect(),
        labels: Vec::new(),
        clones,
    };
    vec![
        job(
            Tool::Mung,
            "cnv",
            &["clones", "d.zarr", "--out", "cnv"],
            false,
        ),
        job(
            Tool::Senna,
            "svd",
            &[
                "svd",
                "d.zarr",
                "--out",
                "svd",
                CLONES_FLAG,
                "cnv.clones.parquet",
            ],
            true,
        ),
    ]
}

#[cfg(unix)]
fn wait(q: &Queue, until: impl Fn(&Queue) -> bool) {
    let t = std::time::Instant::now();
    while !until(q) {
        assert!(t.elapsed() < std::time::Duration::from_secs(10));
        std::thread::sleep(std::time::Duration::from_millis(10));
    }
}

#[cfg(unix)]
#[test]
fn the_queue_waits_after_mung_and_drops_the_clones_when_told() {
    let dir = tempfile::tempdir().unwrap();
    let d = dir.path();
    let (senna, mung) = (fake(d, "s", "senna.json"), fake(d, "m", "clones.parquet"));
    let q = Queue::start(clones_then_fit(d), senna, mung);
    wait(&q, |q| q.asking() == Some(0));
    assert_eq!(
        q.states()[1],
        State::Waiting,
        "the fit waits for the answer"
    );
    q.answer(Keep::Without);
    wait(&q, Queue::finished);
    assert_eq!(q.states(), [State::Done, State::Done]);
    let log = std::fs::read_to_string(d.join("argv.log")).unwrap();
    assert!(!log.contains(CLONES_FLAG), "{log}");
    let script = std::fs::read_to_string(d.join("svd.cmd.sh")).unwrap();
    assert!(!script.contains(CLONES_FLAG));
    assert_eq!(q.done(), [d.join("svd.senna.json")]);
}

#[cfg(unix)]
#[test]
fn kept_clones_reach_the_fit_and_its_script() {
    let dir = tempfile::tempdir().unwrap();
    let d = dir.path();
    let (senna, mung) = (fake(d, "s", "senna.json"), fake(d, "m", "clones.parquet"));
    let q = Queue::start(clones_then_fit(d), senna, mung);
    wait(&q, |q| q.asking() == Some(0));
    q.answer(Keep::Use);
    wait(&q, Queue::finished);
    let log = std::fs::read_to_string(d.join("argv.log")).unwrap();
    assert!(log.contains("--cnv-clones cnv.clones.parquet"), "{log}");
    assert!(std::fs::read_to_string(d.join("svd.cmd.sh"))
        .unwrap()
        .contains("--cnv-clones cnv.clones.parquet"));
}

#[cfg(unix)]
#[test]
fn a_failed_mung_blocks_the_fits_that_wanted_its_clones() {
    let dir = tempfile::tempdir().unwrap();
    let d = dir.path();
    let senna = fake(d, "s", "senna.json");
    let q = Queue::start(clones_then_fit(d), senna, PathBuf::from("/bin/false"));
    wait(&q, Queue::finished);
    let states = q.states();
    assert!(matches!(states[0], State::Failed(_)));
    assert!(matches!(&states[1], State::Failed(why) if why.contains("cnv")));
    assert!(!d.join("svd.cmd.sh").exists());
}

#[cfg(unix)]
#[test]
fn stopping_answers_the_question() {
    let dir = tempfile::tempdir().unwrap();
    let d = dir.path();
    let (senna, mung) = (fake(d, "s", "senna.json"), fake(d, "m", "clones.parquet"));
    let q = Queue::start(clones_then_fit(d), senna, mung);
    wait(&q, |q| q.asking() == Some(0));
    q.stop();
    q.join();
    assert_eq!(q.states(), [State::Done, State::Stopped]);
}
