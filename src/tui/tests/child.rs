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
