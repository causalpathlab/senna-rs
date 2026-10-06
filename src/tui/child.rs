//! The senna, mung and lupin processes a terminal front end starts: run
//! one to its end where it can be stopped, and follow its log a line at a
//! time.
//!
//! A child's stderr is a pseudo-terminal where there is one, so the
//! progress bars it draws (indicatif hides them from a pipe) reach us:
//! each frame becomes a [`Progress`], the rest are lines of its log.

use std::process::{Command, Stdio};

/// Stops a run of child commands: the one running is interrupted, as
/// Ctrl+C would, so a fit can wrap up; asked again it is killed. None
/// after it starts.
#[derive(Default)]
pub(crate) struct Stopper {
    stopped: std::sync::atomic::AtomicBool,
    /// Whether the child was interrupted already: the next stop kills.
    interrupted: std::sync::atomic::AtomicBool,
    child: std::sync::Mutex<Option<std::process::Child>>,
}

impl Stopper {
    /// Interrupt the child the first time, kill it after. Returns whether
    /// this stop kills.
    pub fn stop(&self) -> bool {
        use std::sync::atomic::Ordering::SeqCst;
        self.stopped.store(true, SeqCst);
        let kill = self.interrupted.swap(true, SeqCst);
        if let Ok(mut c) = self.child.lock() {
            if let Some(c) = c.as_mut() {
                signal(c, kill);
            }
        }
        kill
    }

    /// Kill the child now, interrupted or not.
    pub fn kill(&self) {
        self.interrupted
            .store(true, std::sync::atomic::Ordering::SeqCst);
        self.stop();
    }

    pub fn is_stopped(&self) -> bool {
        self.stopped.load(std::sync::atomic::Ordering::SeqCst)
    }
}

/// Interrupt (SIGINT) or kill (SIGKILL) `child` and every process it
/// started: it leads its own process group, and anything left holding its
/// stderr would keep the log open.
fn signal(child: &mut std::process::Child, kill: bool) {
    #[cfg(unix)]
    {
        use rustix::process::{kill_process_group, Pid, Signal};
        // Not waited on yet: the group is still ours, even with the child
        // gone and what it started holding the log open.
        let sig = if kill { Signal::KILL } else { Signal::INT };
        let _ = kill_process_group(Pid::from_child(child), sig);
    }
    if kill || cfg!(not(unix)) {
        let _ = child.kill();
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

/// What a child said on its stderr.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum Said {
    /// A line of its log.
    Line(String),
    /// A frame of a progress bar.
    Progress(Progress),
}

/// Where a progress bar stands: `pos` of `len` (0 for a spinner), and
/// what it counts.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct Progress {
    pub pos: u64,
    pub len: u64,
    pub what: String,
}

/// The columns a child's terminal has: wide enough that a bar and its
/// label are not cut.
#[cfg(unix)]
const COLUMNS: u16 = 160;

/// A pseudo-terminal: our end to read, and the child's to write its
/// stderr to.
#[cfg(unix)]
fn terminal() -> std::io::Result<(std::fs::File, std::os::fd::OwnedFd)> {
    use rustix::fs::{Mode, OFlags};
    use rustix::io::{fcntl_setfd, FdFlags};
    use rustix::pty::{grantpt, openpt, ptsname, unlockpt, OpenptFlags};
    let ours = openpt(OpenptFlags::RDWR | OpenptFlags::NOCTTY)?;
    fcntl_setfd(&ours, FdFlags::CLOEXEC)?;
    grantpt(&ours)?;
    unlockpt(&ours)?;
    let name = ptsname(&ours, Vec::new())?;
    let theirs = rustix::fs::open(
        name.as_c_str(),
        OFlags::RDWR | OFlags::NOCTTY | OFlags::CLOEXEC,
        Mode::empty(),
    )?;
    rustix::termios::tcsetwinsize(
        &theirs,
        rustix::termios::Winsize {
            ws_row: 50,
            ws_col: COLUMNS,
            ws_xpixel: 0,
            ws_ypixel: 0,
        },
    )?;
    Ok((std::fs::File::from(ours), theirs))
}

/// Where the child writes its stderr, and how we read it: a terminal
/// where we can make one, else a pipe.
fn stderr_for(command: &mut Command) -> Option<std::fs::File> {
    #[cfg(unix)]
    if let Ok((ours, theirs)) = terminal() {
        command.stderr(Stdio::from(theirs));
        return Some(ours);
    }
    command.stderr(Stdio::piped());
    None
}

/// Run `command` to its end where `stopper` can kill it, what it says on
/// its stderr to `each`.
pub(crate) fn run_one(
    mut command: Command,
    stopper: &Stopper,
    each: impl FnMut(Said),
) -> Result<(), Failed> {
    let program = super::name(std::path::Path::new(command.get_program()));
    let terminal = stderr_for(&mut command);
    // Its own group, so a stop reaches what it starts too.
    #[cfg(unix)]
    std::os::unix::process::CommandExt::process_group(&mut command, 0);
    let mut child = command
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .spawn()
        .map_err(|e| Failed::Start(format!("cannot run {program}: {e}")))?;
    // Our copy of the child's end closes, so its end is the end of the log.
    drop(command);
    let log: Option<Box<dyn std::io::Read + Send>> = match terminal {
        Some(t) => Some(Box::new(t)),
        None => child
            .stderr
            .take()
            .map(|e| Box::new(e) as Box<dyn std::io::Read + Send>),
    };
    // Where `stop` can reach it; stopped meanwhile, it goes at once.
    if let Ok(mut c) = stopper.child.lock() {
        *c = Some(child);
    }
    if stopper.is_stopped() {
        stopper.stop();
    }
    let last = follow(log, each);
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

/// Read a child's stderr to the end, in pieces split at `\r` and `\n`
/// (a progress bar redraws after a `\r`, with no new line), each as
/// [`log_line`] trims it: a bar frame as a [`Said::Progress`], anything
/// else as a [`Said::Line`]. Returns the last line.
pub(crate) fn follow(log: Option<impl std::io::Read>, mut each: impl FnMut(Said)) -> String {
    let mut last = String::new();
    let Some(mut log) = log else {
        return last;
    };
    let mut said = |piece: &[u8]| {
        let line = log_line(&String::from_utf8_lossy(piece));
        if line.is_empty() {
            return;
        }
        match progress_of(&line) {
            Some(p) => each(Said::Progress(p)),
            None => {
                last.clone_from(&line);
                each(Said::Line(line));
            }
        }
    };
    let mut pending: Vec<u8> = Vec::new();
    let mut buf = [0u8; 8192];
    loop {
        let n = match log.read(&mut buf) {
            Ok(0) => break,
            Ok(n) => n,
            Err(e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
            // A terminal whose child is gone reads as an error.
            Err(_) => break,
        };
        for &b in &buf[..n] {
            if b == b'\r' || b == b'\n' {
                said(&pending);
                pending.clear();
            } else {
                pending.push(b);
            }
        }
    }
    said(&pending);
    last
}

/// Read a child's log to the end a line at a time, every line to `each`;
/// progress bars left out. Returns the last line.
pub(crate) fn follow_log(log: Option<impl std::io::Read>, mut each: impl FnMut(&str)) -> String {
    follow(log, |s| {
        if let Said::Line(l) = s {
            each(&l);
        }
    })
}

/// A progress bar's frame, its elapsed time already gone: the bar of `#`
/// and `-`, `pos/len`, the time left in brackets, then what it counts. A
/// spinner's frame starts with one of its ticks.
pub(crate) fn progress_of(line: &str) -> Option<Progress> {
    let (head, rest) = line.split_once(char::is_whitespace)?;
    if head.chars().all(|c| super::SPINNER.contains(&c)) {
        return Some(Progress {
            what: rest.trim().to_string(),
            ..Progress::default()
        });
    }
    if !head.chars().all(|c| c == '#' || c == '-') {
        return None;
    }
    let rest = rest.trim_start();
    let (count, rest) = rest.split_once(char::is_whitespace).unwrap_or((rest, ""));
    let (pos, len) = count.split_once('/')?;
    let what = rest.trim_start();
    let what = match what.strip_prefix('(') {
        Some(w) => w.split_once(')').map_or("", |(_, after)| after),
        None => what,
    };
    Some(Progress {
        pos: pos.parse().ok()?,
        len: len.parse().ok()?,
        what: what.trim().to_string(),
    })
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
