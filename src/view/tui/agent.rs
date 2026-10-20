//! `senna view --watch`: the view follows what an agent does through
//! `senna mcp`. legume-mcp's `watch::Agent` follows the server; each view
//! the agent publishes is applied here as `view_map` applies it, and the
//! sidebar lists its calls. Changing the view by hand stops following; `W`
//! follows again, from the agent's latest view.

use super::*;
use crate::view::mcp;
use legume_mcp::watch;
use serde_json::Value;
use std::path::Path;

/// Calls the sidebar lists.
const CALL_ROWS: usize = 6;

/// Characters of a row: the widest sidebar's, so none wraps there.
const WIDTH: usize = 44;

/// The agent being followed, and what senna keeps of it for its view.
pub(crate) struct Agent {
    watch: watch::Agent,
    /// Whether the map follows the agent; off once the human changes the
    /// view. The call list goes on following either way.
    following: bool,
    /// Following again: the agent's latest view is applied even when it is
    /// the one applied before the human changed the view.
    resumed: bool,
    /// The view last applied: the same one again is not applied again.
    applied: Option<Value>,
    /// The window of the agent's view last applied: what the map frames
    /// while following, through resizes and reattaches.
    window: Option<[f32; 4]>,
}

impl Agent {
    /// Follow the server in `dir`; senna's view does not steer it.
    pub(crate) fn open(dir: &Path) -> anyhow::Result<Self> {
        Ok(Self {
            watch: watch::Agent::new(dir, false)?,
            following: true,
            resumed: false,
            applied: None,
            window: None,
        })
    }

    pub(crate) fn following(&self) -> bool {
        self.following
    }

    /// The view to apply now, if any: the agent's latest after `W`, else
    /// a new one. Taken only when it is applied, so one that comes while
    /// the view is paused or busy waits in the feed's reader.
    fn next_view(&mut self) -> Option<Value> {
        if std::mem::take(&mut self.resumed) {
            // Whatever the human did to the view, the agent's comes back.
            self.watch.take_new_state();
            return self.watch.state().cloned();
        }
        self.watch
            .take_new_state()
            .filter(|v| self.applied.as_ref() != Some(v))
    }

    /// The sidebar's rows: whether the map follows, the latest calls, and
    /// under them what the latest call is doing (its progress, or why it
    /// failed) and the last command the agent ran.
    pub(super) fn rows(&self) -> Vec<String> {
        let w = &self.watch;
        let title = match (w.ended(), self.following) {
            (true, _) => "agent · ended",
            (false, true) => "agent · following",
            (false, false) => "agent · paused · W follows",
        };
        let mut rows = w.lines(CALL_ROWS + 1, WIDTH);
        rows[0] = title.into();
        let last = w.calls().last();
        let doing = last.and_then(|c| match c.status {
            watch::Status::Running => c.progress.back().cloned(),
            watch::Status::Failed { .. } => {
                let words: Vec<&str> = c.text.split_whitespace().collect();
                Some(words.join(" "))
            }
            _ => None,
        });
        let command = w.records().last().map(|c| format!("$ {c}"));
        rows.extend(
            [doing.map(|d| format!("  {d}")), command]
                .into_iter()
                .flatten()
                .map(|r| r.chars().take(WIDTH).collect()),
        );
        rows
    }
}

impl App {
    /// The agent's news, its latest view applied when following and the
    /// view is not in the middle of something of its own. Returns whether
    /// anything changed.
    pub(super) fn hear_agent(&mut self) -> bool {
        let held = self.holds_outside_changes();
        let Some(agent) = self.agent.as_mut() else {
            return false;
        };
        let heard = agent.watch.poll();
        if !agent.following || held {
            return heard;
        }
        let Some(view) = agent.next_view() else {
            return heard;
        };
        match mcp::agent_view(&view) {
            Ok((shown, args)) => {
                agent.applied = Some(view);
                agent.window = args.bbox;
                self.show_agent_view(&shown, &args);
            }
            Err(e) => self.message = Some(format!("the agent's view: {e:#}")),
        }
        true
    }

    /// A menu, a box being typed in, a relabel round or work under way
    /// (a zoom, lupin, senna) holds the view as it is: no reload, newer
    /// round or agent's view changes it underneath them.
    pub(super) fn holds_outside_changes(&self) -> bool {
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
        agent.resumed = true;
        if agent.watch.state().is_none() {
            self.message = Some("following the agent · no view shown yet".into());
        }
    }

    /// Stop following: the human took the view.
    pub(super) fn pause_agent(&mut self) {
        if let Some(agent) = self.agent.as_mut() {
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

    /// Show the agent's view: the manifest it shows, its map and groups as
    /// `view_map` applies them, and its window, framed at the next layout.
    fn show_agent_view(&mut self, shown: &Path, args: &mcp::MapArgs) {
        // Framed afresh at the next layout, so a new run is drawn once.
        self.vp = None;
        if shown != self.from && !senna::run_manifest::same_file(shown, &self.from) {
            self.open_round(shown, "the agent opened");
        }
        let source = self.scene.source;
        if let Err(e) = mcp::apply(&mut self.scene, args) {
            self.scene.note = Some(format!("the agent's view: {e:#}"));
        }
        if self.scene.source != source {
            // Searches list the features of the source shown.
            self.names = None;
        }
        self.settle();
    }
}

#[cfg(test)]
#[path = "../tests/agent.rs"]
mod tests;
