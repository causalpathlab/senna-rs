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

/// Rows of the agent's activity the sidebar shows, its title included.
const ROWS: usize = 9;

/// Characters of a row: the widest sidebar's, so none wraps there.
const WIDTH: usize = 44;

/// The agent being followed, and what senna keeps of it for its view.
pub(crate) struct Agent {
    watch: watch::Agent,
    /// A view taken from the feed while the view was busy, to apply once
    /// it is not.
    pending: Option<Value>,
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
            pending: None,
            applied: None,
            window: None,
        })
    }

    pub(crate) fn following(&self) -> bool {
        self.watch.following()
    }

    /// What the server wrote since the last look. Returns whether it wrote
    /// anything, or ended.
    fn read(&mut self) -> bool {
        let heard = self.watch.poll();
        if let Some(view) = self.watch.take_new_state() {
            self.pending = Some(view);
        }
        heard
    }

    /// The sidebar's rows: the server and whether it is followed, then its
    /// latest calls.
    pub(super) fn rows(&self) -> Vec<String> {
        self.watch.lines(ROWS, WIDTH)
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
        let heard = agent.read();
        if !agent.following() {
            // Following again starts from the latest view, not this one.
            agent.pending = None;
        }
        if held || agent.pending.is_none() {
            return heard;
        }
        let view = agent.pending.take();
        if view == agent.applied {
            return heard;
        }
        agent.applied.clone_from(&view);
        match view.as_ref().map(mcp::agent_view) {
            Some(Ok((shown, args))) => {
                agent.window = args.bbox;
                self.show_agent_view(&shown, &args);
            }
            Some(Err(e)) => self.message = Some(format!("the agent's view: {e:#}")),
            None => {}
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
        agent.watch.resume();
        agent.pending = agent.watch.state().cloned();
        if agent.pending.is_none() {
            self.message = Some("following the agent · no view shown yet".into());
        }
    }

    /// Stop following: the human took the view.
    pub(super) fn pause_agent(&mut self) {
        if let Some(agent) = self.agent.as_mut() {
            agent.watch.pause();
            // The view is the human's now: following again applies the
            // agent's view even when it has not changed.
            agent.applied = None;
            self.message = Some("following paused · W follows the agent again".into());
        }
    }

    /// The window the map frames while following the agent.
    pub(super) fn agent_window(&self) -> Option<[f32; 4]> {
        self.agent
            .as_ref()
            .filter(|a| a.following())
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
