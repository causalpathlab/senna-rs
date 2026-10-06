//! A spinner on the bottom line while a key or click takes long to answer
//! (a run read from disk, a model's tables, a large folder), so a front end
//! never sits frozen without a word.

use std::sync::mpsc::{channel, RecvTimeoutError};
use std::time::{Duration, Instant};

/// How long the work may take before the spinner shows, so quick work
/// never flickers the screen.
const GRACE: Duration = Duration::from_millis(150);

/// Frames of the spinner.
const FRAMES: [char; 8] = ['⠁', '⠂', '⠄', '⡀', '⢀', '⠠', '⠐', '⠈'];

/// Run `work`; when it takes longer than a moment, show a spinner, `what`
/// it does and the seconds on the terminal's bottom line until it is done.
/// Returns its result and whether the spinner showed: it is written past
/// the front end's own frame, so the screen then wants drawing afresh.
pub(crate) fn during<T: Send>(what: &str, work: impl FnOnce() -> T + Send) -> (T, bool) {
    waiting(work, |elapsed| show(what, elapsed))
}

/// [`during`] before a front end has the screen: the spinner on stderr's
/// own line, rewritten in place and wiped when done; nothing when stderr
/// is not a terminal.
pub(crate) fn during_inline<T: Send>(what: &str, work: impl FnOnce() -> T + Send) -> T {
    use std::io::{IsTerminal, Write};
    if !std::io::stderr().is_terminal() {
        return work();
    }
    let (out, shown) = waiting(work, |elapsed| {
        let width = ratatui::crossterm::terminal::size().map_or(80, |(w, _)| usize::from(w));
        let mut err = std::io::stderr();
        let _ = write!(err, "\r{}", line(what, elapsed, width.saturating_sub(1)));
        let _ = err.flush();
    });
    if shown {
        eprint!("\r\x1b[K");
    }
    out
}

/// Run `work` on a thread; after [`GRACE`], call `tick` with the time
/// taken, again every tenth of a second, until it is done. Returns its
/// result and whether `tick` was called.
fn waiting<T: Send>(work: impl FnOnce() -> T + Send, mut tick: impl FnMut(Duration)) -> (T, bool) {
    let (tx, rx) = channel();
    std::thread::scope(|s| {
        let worker = s.spawn(move || {
            let out = work();
            let _ = tx.send(());
            out
        });
        let start = Instant::now();
        let mut shown = false;
        let mut wait = GRACE;
        // Ends when the work answers, or dies (its sender dropped).
        while let Err(RecvTimeoutError::Timeout) = rx.recv_timeout(wait) {
            tick(start.elapsed());
            shown = true;
            wait = Duration::from_millis(100);
        }
        match worker.join() {
            Ok(out) => (out, shown),
            Err(panic) => std::panic::resume_unwind(panic),
        }
    })
}

/// The bottom line, across the terminal: ` ⠂ loading… 3s`.
fn line(what: &str, elapsed: Duration, width: usize) -> String {
    let spin = FRAMES[(elapsed.as_millis() / 100) as usize % FRAMES.len()];
    let text = format!(" {spin} {what}… {}s", elapsed.as_secs());
    let pad = width.saturating_sub(text.chars().count());
    let mut out: String = text.chars().take(width).collect();
    out.extend(std::iter::repeat_n(' ', pad));
    out
}

fn show(what: &str, elapsed: Duration) {
    use ratatui::crossterm::{
        cursor::MoveTo,
        queue,
        style::{Attribute, Print, SetAttribute},
        terminal::size,
    };
    use std::io::Write;
    let (w, h) = size().unwrap_or((80, 24));
    let mut out = std::io::stdout();
    let _ = queue!(
        out,
        MoveTo(0, h.saturating_sub(1)),
        SetAttribute(Attribute::Reverse),
        Print(line(what, elapsed, usize::from(w))),
        SetAttribute(Attribute::Reset),
    );
    let _ = out.flush();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quick_work_shows_nothing() {
        let (out, shown) = during("loading", || 7);
        assert_eq!(out, 7);
        assert!(!shown);
    }

    #[test]
    fn slow_work_ticks_until_it_answers() {
        let mut ticks = 0;
        let (out, shown) = waiting(
            || {
                std::thread::sleep(Duration::from_millis(400));
                "done"
            },
            |_| ticks += 1,
        );
        assert_eq!(out, "done");
        assert!(shown && ticks >= 1);
    }

    #[test]
    fn the_line_fills_the_width_and_counts_seconds() {
        let l = line("loading", Duration::from_millis(2300), 30);
        assert_eq!(l.chars().count(), 30);
        assert!(l.contains("loading… 2s"), "{l}");
        // Narrower than the text: cut, never wider.
        assert_eq!(line("loading", Duration::ZERO, 5).chars().count(), 5);
    }
}
