//! `senna view --watch`: the view follows what an agent does through
//! `senna mcp`. Each view the agent publishes is applied as `view_map`
//! applies it, and a pane lists its calls, progress and the commands it
//! runs. Changing the view by hand stops following; `W` follows again,
//! from the agent's latest view.

use super::*;
use crate::view::mcp;
use legume_mcp::watch::{self, Body, Follow};
use legume_tui::map::viewport::YAxis;
use serde_json::Value;
use std::collections::VecDeque;
use std::path::Path;

/// Lines of the agent's activity the pane shows.
const SHOWN: usize = 8;

/// Characters of a line: the widest sidebar's, so none wraps there.
const LINE: usize = 44;

/// The agent being followed: its server's feed, read a little at every
/// tick of the view.
pub(crate) struct Agent {
    feed: Follow,
    /// Its newest activity, oldest first.
    lines: VecDeque<String>,
    /// Whether the view follows it; off once the human changes the view.
    pub following: bool,
    /// Its latest view, and whether it is still to be applied: one heard
    /// while the view was busy waits for it.
    latest: Option<Value>,
    fresh: bool,
    /// The window of the agent's view last applied: what the map frames
    /// while following, through resizes and reattaches.
    window: Option<[f32; 4]>,
    ended: bool,
}

impl Agent {
    /// Follow the server in `dir`: its history read for the pane, and its
    /// latest view, read whole, for the map.
    pub(crate) fn open(dir: &Path) -> anyhow::Result<Self> {
        let mut agent = Self {
            feed: watch::follow(dir, watch::From::Start)?,
            lines: VecDeque::new(),
            following: true,
            latest: None,
            fresh: false,
            window: None,
            ended: false,
        };
        while let Some(event) = agent.feed.try_next() {
            if !matches!(event.body, Body::State { .. }) {
                agent.push(event);
            }
        }
        agent.latest = watch::state(dir);
        agent.fresh = agent.latest.is_some();
        Ok(agent)
    }

    fn push(&mut self, event: watch::Event) {
        if let Some(l) = line(&event.t, event.body) {
            if self.lines.len() == SHOWN {
                self.lines.pop_front();
            }
            self.lines.push_back(l);
        }
    }

    /// What the server wrote since the last look. Returns whether it wrote
    /// anything, or ended.
    fn read(&mut self) -> bool {
        let mut heard = false;
        while let Some(event) = self.feed.try_next() {
            heard = true;
            match event.body {
                Body::State { state, .. } => {
                    self.latest = Some(state);
                    self.fresh = true;
                }
                _ => self.push(event),
            }
        }
        if !self.ended && self.feed.ended() {
            self.ended = true;
            heard = true;
        }
        heard
    }

    /// The pane's rows: a title saying whether the view follows, then the
    /// newest lines.
    pub(super) fn rows(&self) -> Vec<String> {
        let title = match (self.ended, self.following) {
            (true, _) => "agent · ended",
            (false, true) => "agent · following",
            (false, false) => "agent · paused · W follows",
        };
        std::iter::once(title.to_string())
            .chain(self.lines.iter().cloned())
            .collect()
    }
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
        // A step that ran; comments and failed steps have no command.
        Body::Record { command, .. } => format!("$ {}", command?),
        Body::Notebook { entry, text } => format!("✎ {entry}: {text}"),
        Body::Note { text } => format!("you: {text}"),
        // The server, its views and its end are said elsewhere; events
        // legume-mcp adds later are left out until read here.
        _ => return None,
    };
    Some(format!("{at} {text}").chars().take(LINE).collect())
}

impl App {
    /// The agent's news, its latest view applied when following and the
    /// view is not in the middle of something of its own. Returns whether
    /// anything changed.
    pub(super) fn hear_agent(&mut self) -> bool {
        let busy = self.busy_for_agent();
        let Some(agent) = self.agent.as_mut() else {
            return false;
        };
        let heard = agent.read();
        if !(agent.following && agent.fresh) || busy {
            return heard;
        }
        agent.fresh = false;
        if let Some(view) = agent.latest.clone() {
            self.show_agent_view(&view);
        }
        true
    }

    /// A menu, a box being typed in, a relabel round or work under way
    /// (a zoom, lupin, senna) holds the view as it is.
    fn busy_for_agent(&self) -> bool {
        self.menu.is_some()
            || self.modal.is_some()
            || self.label_edit.is_some()
            || self.scene.review.is_some()
            || self.zooming.is_some()
            || self.relabeling.is_some()
            || self.recomputing.is_some()
    }

    /// `W`: follow the agent again, from its latest view at the next tick.
    pub(super) fn follow_agent(&mut self) {
        let Some(agent) = self.agent.as_mut() else {
            return;
        };
        agent.following = true;
        agent.fresh = agent.latest.is_some();
        if !agent.fresh {
            self.message = Some("following the agent · no view shown yet".into());
        }
    }

    /// Stop following: the human took the view.
    pub(super) fn pause_agent(&mut self) {
        if let Some(agent) = self.agent.as_mut().filter(|a| a.following) {
            agent.following = false;
            self.message = Some("following paused · W follows the agent again".into());
        }
    }

    /// The window the map frames while following the agent.
    pub(super) fn agent_window(&self) -> Option<[f32; 4]> {
        self.agent
            .as_ref()
            .filter(|a| a.following)
            .and_then(|a| a.window)
    }

    /// Show `view`: the manifest the agent shows, its map and groups as
    /// `view_map` applies them, and its window, framed at the next layout.
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
        let source = self.scene.source;
        if let Err(e) = mcp::apply(&mut self.scene, &args) {
            self.message = Some(format!("the agent's view: {e:#}"));
        }
        if self.scene.source != source {
            // Searches list the features of the source shown.
            self.names = None;
        }
        if let Some(note) = self.scene.note.take() {
            self.message = Some(note);
        }
        if let Some(agent) = self.agent.as_mut() {
            agent.window = args.bbox;
        }
        self.vp = None;
        self.restart();
    }
}

/// The camera for a map `w × h` pixels showing `window`.
pub(super) fn framing(window: [f32; 4], w: usize, h: usize) -> Viewport {
    Viewport::fit(window, w, h, 1.0, YAxis::Up)
}

#[cfg(test)]
#[path = "../tests/agent.rs"]
mod tests;
