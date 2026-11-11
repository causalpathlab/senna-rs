//! `senna mcp`: the viewer and senna's fits as Model Context Protocol tools,
//! served over stdio with legume-mcp.
//!
//! The view tools hold one run and one view between calls, as the terminal
//! viewer holds them: `view_map` changes only what it is given, then draws
//! the map. The run tools start `senna` as a child, as `senna run` does, so
//! stdout carries only the protocol and the child's log comes back as
//! progress. lupin rounds are read as they are on disk; making them is
//! lupin's own server.
//!
//! `mod.rs` holds the server and hands each call to its family's module:
//! `fit` runs fits; `predict` scores new data against a trained model,
//! `model` continues one (and critiques fits for its rounds), and
//! `reference` reads new data through a reference; `layout` lays a run
//! out again and `steps` runs the other analysis commands; `view` holds a
//! run open and draws it. `jobs` builds, checks and starts the senna
//! commands they run.

mod fit;
mod jobs;
mod layout;
mod model;
mod predict;
mod reference;
mod steps;
mod view;

use anyhow::{bail, Context, Result};
use legume_mcp::{Call, Content, Handler, ToolSpec};
use senna::run_manifest;
use serde::Deserialize;
use serde_json::Value;
use std::path::{Path, PathBuf};

pub(super) use view::{agent_view, apply, MapArgs};

/// What the agent is told when it connects, word for word the block the
/// README gives for clients that do not pass it on.
pub const INSTRUCTIONS: &str = "\
Use senna's MCP tools to embed single-cell data and look at the result; \
annotation is lupin's own server. Open the day with `notebook` (kind `plan`): \
the question, the data, and how the result will be checked. `run_fit` runs a \
fit to its end (`run_help` lists a method's flags) and returns its run \
manifest; an existing run is not fitted again. A long fit runs with \
`background`, as a job to follow with `job_wait`, in a session of `senna run` the user can watch with `senna attach`. \
`run_predict` places new cells in a trained model's latent, `run_update` \
continues a trained model over new data, and `run_layout` lays a run out \
again with chosen settings, umap's min_dist and spread among them. `view_open` opens a run (or a \
lupin round) and computes a missing layout or clustering first. `view_map` \
draws the map with its legend; its arguments stay set until changed, so pass \
only what changes. `view_markers` lists what sets a group apart, `view_cell` \
names a point and its groups, and `view_export` writes figures. World \
coordinates are the layout's, y up. Close with `notebook` (kind \
`conclusion`) and tell the user where the records are.";

/// The tools, and the run they have open.
#[derive(Default)]
pub struct Tools {
    open: Option<view::Open>,
}

impl Handler for Tools {
    fn tools(&self) -> Vec<ToolSpec> {
        let mut specs = fit::specs();
        specs.extend(predict::specs());
        specs.extend(model::specs());
        specs.extend(reference::specs());
        specs.extend(layout::specs());
        specs.extend(steps::specs());
        specs.extend(view::specs());
        specs
    }

    fn call(&mut self, name: &str, args: Value, cx: &Call) -> Result<Vec<Content>> {
        for family in [
            fit::call,
            predict::call,
            model::call,
            reference::call,
            layout::call,
            steps::call,
        ] {
            if let Some(out) = family(name, &args, cx)? {
                return Ok(out);
            }
        }
        match view::call(&mut self.open, name, &args, cx)? {
            Some(out) => Ok(out),
            None => bail!("unknown tool {name}"),
        }
    }
}

fn parse<T: for<'de> Deserialize<'de>>(args: Value) -> Result<T> {
    serde_json::from_value(args).context("reading the arguments")
}

fn absolute(path: &Path) -> PathBuf {
    run_manifest::from_cwd(path)
}

/// The server's name, as `initialize` reports it and watchers find it.
pub(crate) const SERVER: &str = "senna";
