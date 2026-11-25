//! `run_fit` and `run_help`: senna's fits, run to their end, and the
//! flags of every command the run tools start.

use super::flags::{self, helped};
use super::jobs::{one_of, planned, senna, start, Task};
use super::parse;
use super::why::justified;
use anyhow::{ensure, Context, Result};
use legume_mcp::{Call, Content, Next, ToolSpec, Why};
use senna::run_manifest::{self, RunKind, RunManifest};
use serde::Deserialize;
use serde_json::{json, Map, Value};
use std::path::{Path, PathBuf};

// The fits `run_fit` runs: those `senna run` queues.
use crate::run::tui::METHODS;

/// A fit tool's call; `None` for another family's.
pub(super) fn call(
    name: &str,
    args: &Value,
    cx: &Call,
    why: Option<&Why>,
) -> Result<Option<Vec<Content>>> {
    Ok(Some(match name {
        "run_fit" => run_fit(parse(args.clone())?, cx, why)?,
        "run_help" => run_help(parse(args.clone())?)?,
        _ => return Ok(None),
    }))
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct FitArgs {
    method: String,
    data: Vec<String>,
    out: Option<String>,
    #[serde(default)]
    flags: Map<String, Value>,
    #[serde(default)]
    background: bool,
}

/// `senna {method} {data} -o {out} {flags}`, making the run's manifest.
fn fit_task(a: &FitArgs, flags: &[String], out: &str) -> Task {
    let mut argv = vec![a.method.clone()];
    argv.extend(a.data.iter().cloned());
    argv.extend(["-o".into(), out.into()]);
    argv.extend(flags.iter().cloned());
    Task {
        argv,
        made: vec![("run", run_manifest::default_path(out).into())],
        summary: format!("senna {} fit of {}", a.method, a.data.join(", ")),
    }
}

/// `senna {method} {data} -o {out} {flags}` as a child, unless the run is
/// there already. Without an out, the run goes in today's results folder,
/// named by its first data file and method.
fn run_fit(a: FitArgs, cx: &Call, why: Option<&Why>) -> Result<Vec<Content>> {
    check_method(&a.method)?;
    ensure!(!a.data.is_empty(), "no data files");
    let flags = flags::words(&a.method, &a.flags)?;
    justified(&flags, why)?;
    let (out, task) = planned(cx, a.out.as_deref(), Some(&a.data[0]), &a.method, |out| {
        Ok(fit_task(&a, &flags, out))
    })?;
    // legume-mcp refuses an out a background fit is still writing (naming
    // its job) until that fit has written the manifest, and fails a fit
    // that ends without it.
    let mut done = start(task, a.background, cx, why, |t, outputs| {
        background_fit(cx, &out, t, outputs, why)
    })?;
    // A background fit's run is not there yet: the agent opens it once
    // the job is done.
    if done.get("job").is_some() {
        return Ok(vec![Content::Json(done)]);
    }
    let run = PathBuf::from(done["run"].as_str().unwrap_or_default());
    let (m, _) = RunManifest::load(&run)?;
    done["kind"] = serde_json::to_value(m.kind)?;
    decision_fields(&m, &run, &mut done);
    Ok(with_next(done, Some(&m), &run))
}

/// The run at `run`'s (`m` its manifest) objective, what its fit left to
/// its user ([`needs_input`]) and the root an ODE grew from
/// ([`root_field`]), set in `out`.
pub(super) fn decision_fields(m: &RunManifest, run: &Path, out: &mut Value) {
    if let Some(objective) = m.objective {
        out["objective"] = json!(objective);
    }
    if let Some(ask) = needs_input(m, run) {
        out["needs_input"] = ask;
    }
    if let Some(root) = root_field(m, run) {
        out["root"] = root;
    }
}

/// The root an ODE run grew from (`root_cells` in its manifest `m`): the
/// list, its count and what its cells mostly are by the run's newest lupin
/// round ([`crate::view::cells_by_round`]), or how to have them typed;
/// `None` when the run records no root.
fn root_field(m: &RunManifest, run: &Path) -> Option<Value> {
    let list = m.outputs.root_cells.as_deref()?;
    let path = run_manifest::resolve(run_manifest::manifest_dir(run), list);
    let cells = match legume_numeric::matrix::common_io::read_name_list(&path.to_string_lossy()) {
        Ok(cells) => cells,
        Err(e) => return Some(json!({"cells": path, "error": format!("{e:#}")})),
    };
    let mut root = json!({"cells": path, "n": cells.len()});
    match crate::view::cells_by_round(run, &cells) {
        Some(mostly) => root["mostly"] = mostly.into(),
        None => {
            root["label_cells"] = "the run has no lupin round to say what these cells are: \
                 annotate it with lupin's annotate_run (from: this run) and view_open it again"
                .into();
        }
    }
    Some(root)
}

/// `said`, the result about the run at `run` (`m` its manifest), and the
/// calls that could come next on it ([`next_of`]), when there are any.
pub(super) fn with_next(said: Value, m: Option<&RunManifest>, run: &Path) -> Vec<Content> {
    let next = m.map(|m| next_of(m, run, &said)).unwrap_or_default();
    let mut out = vec![Content::Json(said)];
    if !next.is_empty() {
        out.push(Content::Next(next));
    }
    out
}

/// The calls that could follow a result `said` about the run at `run` (`m`
/// its manifest), toward its cell types: a cell fit's clustering or its
/// annotation with lupin ([`annotate_next`]); an ODE's too while its root
/// or options name no type, and once they do, a root of the user's
/// choosing written with view_cells.
fn next_of(m: &RunManifest, run: &Path, said: &Value) -> Vec<Next> {
    let untyped = said.pointer("/root/label_cells").is_some()
        || said.pointer("/needs_input/label_cells").is_some();
    if m.kind != RunKind::Ode {
        // A cell fit lupin typed needs nothing more toward its types.
        if crate::view::has_round(run) {
            return Vec::new();
        }
        return vec![annotate_next(m, run)];
    }
    if untyped {
        return vec![annotate_next(m, run)];
    }
    if said.pointer("/root/mostly").is_none() && said.get("needs_input").is_none() {
        return Vec::new();
    }
    vec![Next::new(
        "view_cells",
        json!({}),
        "the root is the user's to choose: with the run's lupin round shown (view_map round), \
         write the cells of the type they name as the root (groups), then run_fit ode on the \
         same data with flags root-cells set to that list and a new out",
    )]
}

/// The next step toward the cell types of the run at `run` (`m` its
/// manifest): with no clustering, `run_step` clustering into a new out
/// (lupin annotates clusters); with one, lupin's `annotate_run` on it.
fn annotate_next(m: &RunManifest, run: &Path) -> Next {
    let run = run.to_string_lossy();
    if m.cluster.clusters.is_none() {
        let out = format!("{}_clusters", run_manifest::derive_out_prefix(&run));
        return Next::new(
            "run_step",
            json!({"command": "clustering", "from": run, "out": out}),
            "lupin annotates clusters and the run has none: cluster it first, if the user \
             wants cell types (the method and its resolution are theirs to choose; run_help \
             clustering lists them)",
        );
    }
    lupin_annotate(&run)
}

/// lupin's `annotate_run` on the run manifest `run`.
pub(super) fn lupin_annotate(run: &str) -> Next {
    Next::new(
        "lupin annotate_run",
        json!({"from": run}),
        "name the clusters' cell types, if the user wants them: it needs a marker panel the \
         user names (markers), or lupin annotate_ask without one; then view_open the round \
         here to look at the types",
    )
}

/// What the fit of the run at `run` (`m` its manifest) left to its user
/// ([`RunManifest::decisions`]), as an agent is to put it to them: every
/// option that stands for cells labelled with what they are by the run's
/// newest lupin round ([`crate::view::typed_decisions`]); `None` when
/// nothing is open.
pub(super) fn needs_input(m: &RunManifest, run: &Path) -> Option<Value> {
    if m.decisions.is_empty() {
        return None;
    }
    let (decisions, typed) = crate::view::typed_decisions(run, &m.decisions);
    let mut ask = json!({
        "decisions": decisions,
        "note": "the fit could not settle these from its data: every option's objective is how \
                 far its fit got (higher is better), and gaps this small are within the data's \
                 noise, so judge by what the options are (a decision's hint says how). Put each \
                 question to the user with that evidence and let them choose; rerun with \
                 run_fit, the chosen option's rerun_args (flag, then value) set in its flags \
                 and a new out (its objective comes back with it).",
    });
    let for_cells = m
        .decisions
        .iter()
        .flat_map(|d| &d.options)
        .any(|o| o.cells.is_some());
    if for_cells && !typed {
        ask["label_cells"] = json!(
            "the options name no cell types: the run has no lupin round. Annotate it with \
             lupin's annotate_run (from: this run, with a marker panel the user names), then \
             view_open this run again: every option then says what its cells mostly are."
        );
    }
    Some(ask)
}

/// The fit `senna {argv}` in the background, from this server's folder:
/// on `senna run`'s Run screen in a session of its own, which a
/// person can attach to and which outlives this server; where there are
/// no sessions, a job of this server's. The job, as the agent is told it.
fn background_fit(
    cx: &Call,
    out: &str,
    task: &Task,
    outputs: &[PathBuf],
    why: Option<&Why>,
) -> Result<Value> {
    #[cfg(unix)]
    {
        let (argv, summary) = (&task.argv, &task.summary);
        let mut watch = legume_mcp::own_command(["run", "--"])?;
        watch.args(argv);
        let label = Path::new(out)
            .file_name()
            .map(|n| n.to_string_lossy().into_owned());
        let started = cx.start_session_job(legume_mcp::SessionJob {
            tool: crate::tui::TOOL,
            screen: "fit".into(),
            label,
            watch,
            record: senna(argv)?,
            outputs: outputs.to_vec(),
            summary: summary.into(),
        })?;
        let session = started.session;
        let mut said = json!({
            "job": started.job,
            "session": session,
            "note": format!(
                "the fit runs as this job, on senna run's screen in session {session} (`senna attach {session}` shows it to the user), and goes on if this server ends; follow it with job_status or job_wait, then open the run with view_open once the job is done"
            ),
        });
        // A session job does not take the step's reason yet.
        if why.is_some() {
            said["why"] = "not recorded with the step: a fit in a session does not take one; \
                           put it in the notebook (kind choice)"
                .into();
        }
        Ok(said)
    }
    #[cfg(not(unix))]
    {
        let _ = out;
        let mut said = super::jobs::server_job(task, outputs, cx, why)?;
        said["note"] = "the fit runs as this job until it ends or the server does; follow it with job_status or job_wait, then open the run with view_open once the job is done".into();
        Ok(said)
    }
}

fn check_method(method: &str) -> Result<()> {
    one_of(method, &METHODS).context("senna's fits")
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct HelpArgs {
    method: String,
}

/// The flags of one command a run tool starts, read here from senna's
/// command line: nothing runs, so the session script records no step.
fn run_help(a: HelpArgs) -> Result<Vec<Content>> {
    one_of(&a.method, &helped())?;
    Ok(vec![Content::Json(flags::listed(&a.method)?)])
}

pub(super) fn specs() -> Vec<ToolSpec> {
    let spec = ToolSpec::new;
    vec![
        spec(
            "run_fit",
            "Fit a senna embedding as `senna {method} {data} -o {out} {flags}` and return \
             the absolute path of its run manifest. An out whose manifest exists is not \
             fitted again; one a background fit is still writing is refused, naming its \
             job (job_wait for it). Fits can take long; progress carries the log.",
            json!({
                "type": "object",
                "properties": {
                    "method": {"enum": METHODS, "description": "The fit"},
                    "data": {"type": "array", "items": {"type": "string"}, "minItems": 1, "description": "Data files (.zarr, .zarr.zip, .h5)"},
                    "out": {"type": "string", "description": "Output prefix; the run is {out}.senna.json [default: today's results folder, by data file and method]"},
                    "flags": flags::property("the method", r#"{"batch-files": ["batch.txt"], "epochs": 200}"#),
                    "background": {"type": "boolean", "description": "Run the fit as a job and return its id at once, for a fit longer than a call can wait; follow it with job_status or job_wait [default: false]"},
                },
                "required": ["method", "data"],
                "additionalProperties": false,
            }),
        ),
        spec(
            "run_help",
            "The flags a run tool takes in its `flags` for one fit, or for the command \
             another run tool starts, read from senna's command line: each flag's name, \
             help, default and choices, or that it is a switch; and `own`, the arguments \
             the tool takes by name instead.",
            json!({
                "type": "object",
                "properties": {"method": {"enum": helped()}},
                "required": ["method"],
                "additionalProperties": false,
            }),
        ),
    ]
}

#[cfg(test)]
#[path = "../tests/mcp_fit.rs"]
mod tests;
