use super::*;

#[test]
fn log_lines_lose_their_lead_colours_and_redraws() {
    assert_eq!(log_line("[2026-01-01 INFO senna] fitting"), "fitting");
    assert_eq!(log_line("\u{1b}[32mok\u{1b}[0m  "), "ok");
    let mut seen = Vec::new();
    let log = "step 1\rstep 2\n[t INFO x] done\n".as_bytes();
    let last = follow_log(Some(log), |l| seen.push(l.to_string()));
    assert_eq!(seen, ["step 1", "step 2", "done"]);
    assert_eq!(last, "done");
}

#[test]
fn bar_frames_are_progress_and_the_rest_is_log() {
    assert_eq!(
        progress_of("#####----- 3/10 (2s) HVG blocks"),
        Some(Progress {
            pos: 3,
            len: 10,
            what: "HVG blocks".into()
        })
    );
    assert_eq!(
        progress_of("---------- 0/140 (1 minute) sweeps")
            .unwrap()
            .what,
        "sweeps"
    );
    assert_eq!(progress_of("⠁ streamed 12 fragments").unwrap().len, 0);
    assert_eq!(progress_of("Selected 300 / 300 features"), None);
    assert_eq!(progress_of("-- not a bar"), None);

    // As a terminal writes it: frames after `\r`, no new line between.
    let raw = "[t INFO x] start\r\n[00:00:01] \u{1b}[36m##--\u{1b}[0m 1/2 (1s) epochs\r\u{1b}[2K\
               [00:00:02] ####-- 2/2 (0s) epochs\r\u{1b}[2K[t INFO x] done\r\n";
    let mut said = Vec::new();
    let last = follow(Some(raw.as_bytes()), |s| said.push(s));
    assert_eq!(last, "done");
    assert_eq!(
        said,
        [
            Said::Line("start".into()),
            Said::Progress(Progress {
                pos: 1,
                len: 2,
                what: "epochs".into()
            }),
            Said::Progress(Progress {
                pos: 2,
                len: 2,
                what: "epochs".into()
            }),
            Said::Line("done".into()),
        ]
    );
}

#[cfg(unix)]
#[test]
fn a_child_runs_on_a_terminal_so_its_bars_draw() {
    // A bar library draws only on a terminal: so does this stand-in.
    let script = "if [ -t 2 ]; then printf '[00:00:00] #--- 1/4 (3s) steps\\r' >&2; fi; echo '[t INFO x] ok' >&2";
    let mut command = Command::new("sh");
    command.args(["-c", script]);
    let mut said = Vec::new();
    let stopper = Stopper::default();
    run_one(command, &stopper, |s| said.push(s)).unwrap();
    assert_eq!(
        said,
        [
            Said::Progress(Progress {
                pos: 1,
                len: 4,
                what: "steps".into()
            }),
            Said::Line("ok".into()),
        ]
    );
}
