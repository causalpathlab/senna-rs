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
//! commands they run, and `flags` reads each command's flags off senna's
//! command line: a run tool takes its own arguments by name and every
//! other flag in `flags`.

mod cell_colour;
mod fit;
mod flags;
mod jobs;
mod layout;
mod model;
mod predict;
mod prompts;
mod reference;
mod stage2;
mod steps;
mod view;
mod view_cells;
mod view_chart;
mod view_stage2;
mod why;

use anyhow::{bail, Context, Result};
use legume_mcp::{Call, Content, Handler, Prompt, ToolSpec};
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
fit to its end and returns its run manifest; an existing run is not fitted \
again. Every run tool takes its inputs and out by name and any other flag of \
its senna command in `flags` (flag name to value); `run_help` lists a \
command's flags. A long fit runs with \
`background`, as a job to follow with `job_wait`, in a session of `senna run` the user can watch with `senna attach`. \
`run_predict` places new cells in a trained model's latent, `run_update` \
continues a trained model over new data, and `run_layout` lays a run out \
again with chosen settings, umap's min_dist and spread (flags umap-min-dist, umap-spread) among them. `view_open` opens a run (or a \
lupin round) and computes a missing layout or clustering first; in a terminal the same \
run opens with `senna view {run}.senna.json`. `view_map` \
draws the map with its legend; its arguments stay set until changed, so pass \
only what changes. `view_markers` lists what sets a group apart, `view_cell` \
names a point and its groups, and `view_export` writes figures. World \
coordinates are the layout's, y up. When `run_fit` or `view_open` returns \
`needs_input`, the fit could not settle something from its data (an ODE's root, \
say): put each question to the user before reading the result as settled, and \
rerun with the option they choose. Close with `notebook` (kind \
`conclusion`) and tell the user where the records are.

With lupin: cell types come in order, a fit, then `run_step` clustering from \
its manifest, then lupin's `annotate_run` on that manifest; results carry the \
`next` calls toward them, and the prompts `annotate` and `trajectory` give \
the whole recipes. Open the round here (`view_open` the round, or `view_map` \
`round`) so its cell types are a grouping. To root an ODE, look at the types \
(`view_markers`; lupin's `annotate_why` on which type is least \
differentiated), write the root type's cells with `view_cells`, and `run_fit` \
ode with flag root-cells set to that list and a new out. An ODE grown from a \
root says so in its result's `root`: the cells and what they are, by the \
groups `view_cells` wrote them from or by a lupin round of the run; a named \
root is grown from unless no tree can be, then said to be unused, with why. A \
`needs_input` from an ODE labels its options by lupin's types only when the \
run has a round. A run tool's call that carries a choice (a root, a setting \
the user picked) says why with `why`, `by` (user, agent or agent_accepted) \
and `cites`, kept with the step; an answer to a `needs_input` is refused \
without them.";

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
        specs.extend(stage2::specs());
        specs.extend(layout::specs());
        specs.extend(steps::specs());
        specs.extend(view::specs());
        why::add_properties(&mut specs);
        specs
    }

    fn prompts(&self) -> Vec<Prompt> {
        prompts::prompts()
    }

    fn call(&mut self, name: &str, mut args: Value, cx: &Call) -> Result<Vec<Content>> {
        let why = why::take(name, &mut args)?;
        for family in [
            fit::call,
            predict::call,
            model::call,
            reference::call,
            stage2::call,
            layout::call,
            steps::call,
        ] {
            if let Some(out) = family(name, &args, cx, why.as_ref())? {
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
