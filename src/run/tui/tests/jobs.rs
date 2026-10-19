use super::*;
use std::path::Path;

/// Job `m` in `dir` starting `program` with `argv`.
fn job(
    dir: &std::path::Path,
    tool: Tool,
    program: &std::path::Path,
    m: &str,
    argv: &[&str],
) -> Job {
    Job {
        tool,
        program: program.to_path_buf(),
        row: 0,
        method: m.into(),
        dir: dir.to_path_buf(),
        out: m.into(),
        argv: argv.iter().map(ToString::to_string).collect(),
        labels: Vec::new(),
        clones: None,
        pb_from: None,
        lane: 0,
    }
}

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
    let job = |m: &str| {
        job(
            dir.path(),
            Tool::Senna,
            &fake,
            m,
            &[m, "d.zarr", "--out", m],
        )
    };
    let q = Queue::start(vec![job("svd"), job("bge")]);
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
    let mut j = job(
        dir.path(),
        Tool::Senna,
        Path::new("/bin/false"),
        "svd",
        &["svd", "--out", "r"],
    );
    j.out = "r".into();
    let q = Queue::start(vec![j]);
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
    let job = |m: &str| job(dir.path(), Tool::Senna, &slow, m, &[m, "--out", m]);
    let q = Queue::start(vec![job("a"), job("b")]);
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

#[cfg(unix)]
#[test]
fn a_stop_reaches_what_a_fit_starts() {
    use std::os::unix::fs::PermissionsExt;
    let dir = tempfile::tempdir().unwrap();
    // Not exec'd: the shell waits on its own child, which only a stop sent
    // to the fit's whole process group interrupts.
    let slow = dir.path().join("slow-senna");
    std::fs::write(&slow, "#!/bin/sh\nsleep 30\necho done >&2\n").unwrap();
    std::fs::set_permissions(&slow, std::fs::Permissions::from_mode(0o755)).unwrap();
    let q = Queue::start(vec![job(
        dir.path(),
        Tool::Senna,
        &slow,
        "a",
        &["--out", "a"],
    )]);
    while q.shared.lock().unwrap().states[0] != State::Running {
        std::thread::sleep(std::time::Duration::from_millis(5));
    }
    std::thread::sleep(std::time::Duration::from_millis(200));
    let t = std::time::Instant::now();
    assert!(!q.stop(), "one stop interrupts");
    q.join();
    assert!(t.elapsed() < std::time::Duration::from_secs(5));
    assert_eq!(q.shared.lock().unwrap().states, [State::Stopped]);
}

#[cfg(unix)]
#[test]
fn a_failed_fit_says_how_it_ended_and_what_it_was_doing() {
    use std::os::unix::fs::PermissionsExt;
    let dir = tempfile::tempdir().unwrap();
    let write = |name: &str, body: &str| {
        let p = dir.path().join(name);
        std::fs::write(&p, format!("#!/bin/sh\n{body}\n")).unwrap();
        std::fs::set_permissions(&p, std::fs::Permissions::from_mode(0o755)).unwrap();
        p
    };
    let quiet = write("quiet", "echo \"[t INFO x] epoch 3\" >&2\nexit 1");
    let explained = write("explained", "echo \"Error: no such layer\" >&2\nexit 2");
    let q = Queue::start(vec![
        job(dir.path(), Tool::Senna, &quiet, "a", &["--out", "a"]),
        job(dir.path(), Tool::Senna, &explained, "b", &["--out", "b"]),
    ]);
    q.join();
    assert_eq!(
        q.shared.lock().unwrap().states,
        [
            State::Failed("exit status: 1, after: epoch 3".into()),
            State::Failed("no such layer".into()),
        ]
    );
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

/// `mung clones` with `mung`, then an svd with `senna` that wants its
/// clones.
#[cfg(unix)]
fn clones_then_fit(
    d: &std::path::Path,
    senna: &std::path::Path,
    mung: &std::path::Path,
) -> Vec<Job> {
    let mut fit = job(
        d,
        Tool::Senna,
        senna,
        "svd",
        &["svd", "d.zarr", "--out", "svd"],
    );
    fit.clones = Some("cnv.clones.parquet".into());
    vec![
        job(
            d,
            Tool::Mung,
            mung,
            "cnv",
            &["clones", "d.zarr", "--out", "cnv"],
        ),
        fit,
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
    let q = Queue::start(clones_then_fit(d, &senna, &mung));
    wait(&q, Queue::asking);
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
    let q = Queue::start(clones_then_fit(d, &senna, &mung));
    wait(&q, Queue::asking);
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
    let q = Queue::start(clones_then_fit(d, &senna, Path::new("/bin/false")));
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
    let q = Queue::start(clones_then_fit(d, &senna, &mung));
    wait(&q, Queue::asking);
    q.stop();
    q.join();
    assert_eq!(q.states(), [State::Done, State::Stopped]);
}

#[cfg(unix)]
#[test]
fn a_folder_not_there_yet_is_made_for_the_run() {
    let dir = tempfile::tempdir().unwrap();
    let d = dir.path();
    let senna = fake(d, "s", "senna.json");
    let into = d.join("results/exp1");
    let q = Queue::start(vec![job(
        &into,
        Tool::Senna,
        &senna,
        "svd",
        &["svd", "--out", "svd"],
    )]);
    wait(&q, Queue::finished);
    assert_eq!(q.states(), [State::Done]);
    assert!(into.join("svd.cmd.sh").exists());
    assert!(into.join("svd.senna.json").exists());
}

#[cfg(unix)]
#[test]
fn a_fit_collapsing_on_an_unfinished_fit_does_not_start() {
    let dir = tempfile::tempdir().unwrap();
    let builder = job(
        dir.path(),
        Tool::Senna,
        Path::new("/bin/false"),
        "topic",
        &["topic", "--out", "topic"],
    );
    let mut user = job(
        dir.path(),
        Tool::Senna,
        Path::new("/bin/true"),
        "vae",
        &["vae", "--out", "vae"],
    );
    user.pb_from = Some("topic.senna.json".into());
    let q = Queue::start(vec![builder, user]);
    while !q.finished() {
        std::thread::sleep(std::time::Duration::from_millis(10));
    }
    let s = q.shared.lock().unwrap();
    assert!(matches!(s.states[0], State::Failed(_)));
    match &s.states[1] {
        State::Failed(why) => assert!(why.contains("topic did not finish"), "{why}"),
        other => panic!("vae should not start: {other:?}"),
    }
    drop(s);
    assert!(!dir.path().join("vae.cmd.sh").exists());
}

/// Two lanes run side by side; a fit collapsing on another's partition in
/// the other lane waits for it.
#[cfg(unix)]
#[test]
fn lanes_run_side_by_side_and_a_partition_is_waited_for() {
    use std::os::unix::fs::PermissionsExt;
    let dir = tempfile::tempdir().unwrap();
    let fake = dir.path().join("fake-senna");
    // Sleeps, then marks its result.
    std::fs::write(
        &fake,
        "#!/bin/sh
sleep 1
while [ $# -gt 0 ]; do [ \"$1\" = --out ] && touch \"$2.senna.json\"; shift; done\n",
    )
    .unwrap();
    std::fs::set_permissions(&fake, std::fs::Permissions::from_mode(0o755)).unwrap();
    let in_lane = |m: &str, lane: usize| {
        let mut j = job(dir.path(), Tool::Senna, &fake, m, &[m, "--out", m]);
        j.lane = lane;
        j
    };
    let started = std::time::Instant::now();
    let q = Queue::start(vec![in_lane("vae", 0), in_lane("svd", 1)]);
    while !q.finished() {
        std::thread::sleep(std::time::Duration::from_millis(10));
    }
    assert!(
        started.elapsed() < std::time::Duration::from_millis(1900),
        "side by side"
    );
    assert_eq!(q.states(), [State::Done, State::Done]);

    let dir2 = tempfile::tempdir().unwrap();
    let job2 = |m: &str, lane: usize| {
        let mut j = job(dir2.path(), Tool::Senna, &fake, m, &[m, "--out", m]);
        j.dir = dir2.path().to_path_buf();
        j.lane = lane;
        j
    };
    let mut after = job2("rbm", 1);
    after.pb_from = Some("topic.senna.json".into());
    let started = std::time::Instant::now();
    let q = Queue::start(vec![job2("topic", 0), after]);
    while !q.finished() {
        std::thread::sleep(std::time::Duration::from_millis(10));
    }
    assert!(
        started.elapsed() >= std::time::Duration::from_millis(1900),
        "waited"
    );
    assert_eq!(q.states(), [State::Done, State::Done]);
}
