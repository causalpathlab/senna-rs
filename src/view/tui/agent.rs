//! `senna view --watch`: the view follows what an agent does through
//! `senna mcp`. Each view the agent publishes is applied as `view_map`
//! applies it, and a pane lists its calls, progress and the commands it
//! runs. Changing the view by hand stops following; `W` follows again,
//! from the agent's latest view.

use super::*;
use crate::view::mcp;
use legume_mcp::watch::{self, Body};
use legume_tui::map::viewport::YAxis;
use serde_json::Value;
use std::collections::VecDeque;
use std::path::Path;
use std::sync::mpsc::{channel, Receiver, TryRecvError};

/// Lines of the agent's activity the pane keeps.
const KEPT: usize = 200;

/// Characters of an event the pane shows on its line.
const LINE: usize = 120;

/// What the follower hears from the agent's server.
pub(crate) enum Heard {
    /// The view the agent shows now.
    View(Value),
    /// A line of its activity: a call, its progress or result, a command.
    Line(String),
    /// The server ended.
    Ended,
}

/// The agent being followed.
pub(crate) struct Agent {
    heard: Receiver<Heard>,
    /// Its activity, oldest first.
    lines: VecDeque<String>,
    /// Whether the view follows it; off once the human changes the view.
    pub following: bool,
    /// Its latest view, applied again when following resumes.
    latest: Option<Value>,
    ended: bool,
}

/// What the server in `dir` does, as it does it: its feed from the
/// start, so the pane has its history and the view its latest state.
pub(crate) fn listen(dir: &Path) -> anyhow::Result<Receiver<Heard>> {
    let feed = watch::follow(dir, watch::From::Start)?;
    let (tx, rx) = channel();
    std::thread::spawn(move || {
        for event in feed {
            let heard = match event.body {
                Body::State { state, .. } => Heard::View(state),
                body => match line(&event.t, body) {
                    Some(l) => Heard::Line(l),
                    None => continue,
                },
            };
            if tx.send(heard).is_err() {
                return;
            }
        }
        let _ = tx.send(Heard::Ended);
    });
    Ok(rx)
}

/// An event as a line of the pane, at time `t`.
fn line(t: &str, body: Body) -> Option<String> {
    // The time of day, from RFC 3339.
    let at = t.get(11..19).unwrap_or(t);
    let text = match body {
        Body::Call { tool, args, .. } => format!("→ {tool} {args}"),
        Body::Progress { message, .. } => format!("  {message}"),
        Body::Result { ok, ms, text, .. } => {
            // Its words on one line, past an image's placeholder.
            let first = text
                .lines()
                .filter(|l| !l.trim_start().starts_with("[image/"))
                .flat_map(str::split_whitespace)
                .collect::<Vec<_>>()
                .join(" ");
            let mark = if ok { "✓" } else { "✗" };
            format!("{mark} {:.1}s {first}", ms as f64 / 1000.0)
        }
        Body::Cancelled { by, .. } => format!("✗ cancelled by the {by}"),
        // The command itself, not the comments and guards around it.
        Body::Record { line } => {
            let scaffold = ["#", "if ", "then", "fi", "mkdir "];
            let command = line
                .lines()
                .map(str::trim)
                .find(|l| !l.is_empty() && !scaffold.iter().any(|s| l.starts_with(s)))?;
            format!("$ {command}")
        }
        Body::Notebook { entry, text } => format!("✎ {entry}: {text}"),
        Body::Note { text } => format!("you: {text}"),
        Body::Server(_) | Body::State { .. } => return None,
    };
    Some(format!("{at} {text}").chars().take(LINE).collect())
}

impl Agent {
    pub(crate) fn new(heard: Receiver<Heard>) -> Self {
        Self {
            heard,
            lines: VecDeque::new(),
            following: true,
            latest: None,
            ended: false,
        }
    }

    /// Everything heard since the last look. Returns whether anything was,
    /// and the view to apply, when following and the agent showed one.
    fn drain(&mut self) -> (bool, Option<Value>) {
        let (mut heard, mut view) = (false, None);
        loop {
            match self.heard.try_recv() {
                Ok(Heard::View(v)) => {
                    self.latest = Some(v.clone());
                    view = Some(v);
                }
                Ok(Heard::Line(l)) => {
                    if self.lines.len() == KEPT {
                        self.lines.pop_front();
                    }
                    self.lines.push_back(l);
                }
                Ok(Heard::Ended) | Err(TryRecvError::Disconnected) if !self.ended => {
                    self.ended = true;
                    self.lines.push_back("the agent's server ended".into());
                }
                Ok(Heard::Ended) | Err(_) => break,
            }
            heard = true;
        }
        (heard, view.filter(|_| self.following))
    }

    /// The pane's rows: a title saying whether the view follows, then the
    /// newest `n` lines.
    pub(super) fn rows(&self, n: usize) -> Vec<String> {
        let title = match (self.ended, self.following) {
            (true, _) => "agent · ended",
            (false, true) => "agent · following",
            (false, false) => "agent · paused · W follows again",
        };
        let skip = self.lines.len().saturating_sub(n);
        std::iter::once(title.to_string())
            .chain(self.lines.iter().skip(skip).cloned())
            .collect()
    }
}

impl App {
    /// The agent's news, its latest view applied when following. Returns
    /// whether the view changed.
    pub(super) fn hear_agent(&mut self) -> bool {
        let Some(agent) = self.agent.as_mut() else {
            return false;
        };
        let (heard, view) = agent.drain();
        if let Some(view) = view {
            self.show_agent_view(&view);
        }
        heard
    }

    /// `W`: follow the agent again, from its latest view.
    pub(super) fn follow_agent(&mut self) {
        let Some(agent) = self.agent.as_mut() else {
            return;
        };
        agent.following = true;
        match agent.latest.clone() {
            Some(view) => self.show_agent_view(&view),
            None => self.message = Some("following the agent · no view shown yet".into()),
        }
    }

    /// Stop following: the human took the view.
    pub(super) fn pause_agent(&mut self) {
        if let Some(agent) = self.agent.as_mut().filter(|a| a.following) {
            agent.following = false;
            self.message = Some("following paused · W follows the agent again".into());
        }
    }

    /// Show `view`: the manifest the agent shows, its map and groups as
    /// `view_map` applies them, and its window.
    fn show_agent_view(&mut self, view: &Value) {
        let (shown, args) = match mcp::agent_view(view) {
            Ok(v) => v,
            Err(e) => {
                self.message = Some(format!("the agent's view: {e:#}"));
                return;
            }
        };
        if shown != self.from && !senna::run_manifest::same_file(&shown, &self.from) {
            self.open_round(&shown, "the agent opened");
        }
        match mcp::apply(&mut self.scene, &args) {
            Ok(true) => self.vp = None,
            Ok(false) => {}
            Err(e) => self.message = Some(format!("the agent's view: {e:#}")),
        }
        let (w, h) = self.map_px();
        self.vp = match args.bbox {
            Some(window) if w > 0 && h > 0 => Some(Viewport::fit(window, w, h, 1.0, YAxis::Up)),
            Some(_) => self.vp,
            None => None,
        };
        self.restart();
    }
}

#[cfg(test)]
#[path = "../tests/agent.rs"]
mod tests;
