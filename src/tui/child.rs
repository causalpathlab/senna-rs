//! The senna and lupin processes a terminal front end starts: run one to
//! its end where it can be stopped, and follow its log a line at a time.

use std::process::{Command, Stdio};

/// Stops a run of child commands: the one running is killed, and none
/// after it starts.
#[derive(Default)]
pub(crate) struct Stopper {
    stopped: std::sync::atomic::AtomicBool,
    child: std::sync::Mutex<Option<std::process::Child>>,
}

impl Stopper {
    pub fn stop(&self) {
        self.stopped
            .store(true, std::sync::atomic::Ordering::SeqCst);
        if let Ok(mut c) = self.child.lock() {
            if let Some(c) = c.as_mut() {
                let _ = c.kill();
            }
        }
    }

    pub fn is_stopped(&self) -> bool {
        self.stopped.load(std::sync::atomic::Ordering::SeqCst)
    }
}

/// Why [`run_one`] did not finish well.
#[derive(Debug, PartialEq, Eq)]
pub(crate) enum Failed {
    /// `stopper` was stopped.
    Stopped,
    /// The command did not start.
    Start(String),
    /// It ended badly: the reason it gave last.
    Exit(String),
}

/// Run `command` to its end where `stopper` can kill it, each line of its
/// log (its stderr) to `each`.
pub(crate) fn run_one(
    mut command: Command,
    stopper: &Stopper,
    each: impl FnMut(&str),
) -> Result<(), Failed> {
    let mut child = command
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| {
            let program = command.get_program().to_string_lossy();
            Failed::Start(format!("cannot run {program}: {e}"))
        })?;
    let log = child.stderr.take();
    // Where `stop` can reach it; stopped meanwhile, it goes at once.
    if let Ok(mut c) = stopper.child.lock() {
        *c = Some(child);
    }
    if stopper.is_stopped() {
        stopper.stop();
    }
    let last = follow_log(log, each);
    let status = stopper
        .child
        .lock()
        .ok()
        .and_then(|mut c| c.take())
        .map(|mut c| c.wait());
    if stopper.is_stopped() {
        return Err(Failed::Stopped);
    }
    match status {
        Some(Ok(s)) if s.success() => Ok(()),
        Some(Err(e)) => Err(Failed::Exit(e.to_string())),
        _ => Err(Failed::Exit(
            last.strip_prefix("Error: ").unwrap_or(&last).to_string(),
        )),
    }
}

/// Read a child's log (its stderr) to the end a line at a time, handing
/// every non-empty one to `each` as [`log_line`] trims it. A progress bar's
/// `\r` redraws count as lines of their own. Returns the last.
pub(crate) fn follow_log(log: Option<impl std::io::Read>, mut each: impl FnMut(&str)) -> String {
    use std::io::BufRead;
    let mut last = String::new();
    if let Some(err) = log {
        let mut r = std::io::BufReader::new(err);
        let mut buf = Vec::new();
        while r.read_until(b'\n', &mut buf).is_ok_and(|n| n > 0) {
            for part in String::from_utf8_lossy(&buf).split('\r') {
                let line = log_line(part);
                if !line.is_empty() {
                    each(&line);
                    last = line;
                }
            }
            buf.clear();
        }
    }
    last
}

/// A log line without terminal colours or its "[time LEVEL module] "
/// prefix: the message is what matters on a status line.
pub(crate) fn log_line(line: &str) -> String {
    let mut plain = String::with_capacity(line.len());
    let mut chars = line.chars();
    while let Some(c) = chars.next() {
        if c != '\u{1b}' {
            plain.push(c);
        } else if chars.next() == Some('[') {
            // A CSI sequence runs to its final byte.
            for c in chars.by_ref() {
                if ('@'..='~').contains(&c) {
                    break;
                }
            }
        }
    }
    let t = plain.trim();
    match t.split_once("] ") {
        Some((head, msg)) if head.starts_with('[') => msg.trim().to_string(),
        _ => t.to_string(),
    }
}

#[cfg(test)]
#[path = "tests/child.rs"]
mod tests;
