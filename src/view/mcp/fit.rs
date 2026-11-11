//! `run_fit` and `run_help`: senna's fits, run to their end, and the
//! flags of every command the run tools start.

use super::jobs::{default_out, one_of, senna, start, Task};
use super::parse;
use crate::view::recompute::Step;
use anyhow::{ensure, Context, Result};
use legume_mcp::{Call, Content, ToolSpec};
use senna::run_manifest::{self, RunManifest};
use serde::Deserialize;
use serde_json::{json, Value};
use std::path::{Path, PathBuf};

// The fits `run_fit` runs: those `senna run` queues.
use crate::run::tui::METHODS;

/// A fit tool's call; `None` for another family's.
pub(super) fn call(name: &str, args: &Value, cx: &Call) -> Result<Option<Vec<Content>>> {
    Ok(Some(match name {
        "run_fit" => run_fit(parse(args.clone())?, cx)?,
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
    args: Vec<String>,
    #[serde(default)]
    background: bool,
}

/// `senna {method} {data} -o {out} {args}` as a child, unless the run is
/// there already. Without an out, the run goes in today's results folder,
/// named by its first data file and method.
fn run_fit(a: FitArgs, cx: &Call) -> Result<Vec<Content>> {
    check_method(&a.method)?;
    ensure!(!a.data.is_empty(), "no data files");
    let out = match a.out {
        Some(out) => out,
        None => default_out(cx, &a.data[0], &a.method, run_manifest::default_path)?,
    };
    let mut argv = vec![a.method.clone()];
    argv.extend(a.data.iter().cloned());
    argv.extend(["-o".into(), out.clone()]);
    argv.extend(a.args);
    let task = Task {
        argv,
        made: vec![("run", run_manifest::default_path(&out).into())],
        summary: format!("senna {} fit of {}", a.method, a.data.join(", ")),
    };
    // legume-mcp refuses an out a background fit is still writing (naming
    // its job) until that fit has written the manifest, and fails a fit
    // that ends without it.
    let mut done = start(task, a.background, cx, |t, outputs| {
        background_fit(cx, &out, &t.argv, outputs, &t.summary)
    })?;
    // A background fit's run is not there yet: the agent opens it once
    // the job is done.
    if done.get("job").is_none() {
        done["kind"] = json!(kind(Path::new(done["run"].as_str().unwrap_or_default()))?);
    }
    Ok(vec![Content::Json(done)])
}

/// The fit `senna {argv}` in the background, from this server's folder:
/// on `senna run`'s Run screen in a session of its own, which a
/// person can attach to and which outlives this server; where there are
/// no sessions, a job of this server's. The job, as the agent is told it.
fn background_fit(
    cx: &Call,
    out: &str,
    argv: &[String],
    outputs: &[PathBuf],
    summary: &str,
) -> Result<Value> {
    #[cfg(unix)]
    {
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
        Ok(json!({
            "job": started.job,
            "session": session,
            "note": format!(
                "the fit runs as this job, on senna run's screen in session {session} (`senna attach {session}` shows it to the user), and goes on if this server ends; follow it with job_status or job_wait, then open the run with view_open once the job is done"
            ),
        }))
    }
    #[cfg(not(unix))]
    {
        let _ = out;
        Ok(json!({
            "job": cx.spawn_job(senna(argv)?, outputs, summary)?,
            "note": "the fit runs as this job until it ends or the server does; follow it with job_status or job_wait, then open the run with view_open once the job is done",
        }))
    }
}

fn check_method(method: &str) -> Result<()> {
    one_of(method, &METHODS).context("senna's fits")
}

/// The kind of fit the run `manifest` records.
fn kind(manifest: &Path) -> Result<Value> {
    let (m, _) = RunManifest::load(manifest)?;
    Ok(serde_json::to_value(m.kind)?)
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct HelpArgs {
    method: String,
}

/// The commands `run_help` covers: the fits, and those the other run
/// tools start.
fn helped() -> Vec<String> {
    let others = ["predict", "update"].into_iter().map(String::from);
    let layouts = Step::CellLayout
        .settings()
        .iter()
        .map(|m| format!("layout {m}"));
    METHODS
        .iter()
        .map(|m| m.to_string())
        .chain(others)
        .chain(layouts)
        .collect()
}

/// The command's help, read here: nothing runs, so the session script
/// records no step.
fn run_help(a: HelpArgs) -> Result<Vec<Content>> {
    use clap::CommandFactory;
    one_of(&a.method, &helped())?;
    let mut cli = crate::Cli::command();
    cli.build();
    let mut sub = &mut cli;
    for word in a.method.split(' ') {
        sub = sub
            .find_subcommand_mut(word)
            .with_context(|| format!("no subcommand {}", a.method))?;
    }
    Ok(vec![Content::Text(sub.render_long_help().to_string())])
}

pub(super) fn specs() -> Vec<ToolSpec> {
    let spec = ToolSpec::new;
    vec![
        spec(
            "run_fit",
            "Fit a senna embedding as `senna {method} {data} -o {out} {args}` and return \
             the absolute path of its run manifest. An out whose manifest exists is not \
             fitted again; one a background fit is still writing is refused, naming its \
             job (job_wait for it). Fits can take long; progress carries the log.",
            json!({
                "type": "object",
                "properties": {
                    "method": {"enum": METHODS, "description": "The fit"},
                    "data": {"type": "array", "items": {"type": "string"}, "minItems": 1, "description": "Data files (.zarr, .zarr.zip, .h5)"},
                    "out": {"type": "string", "description": "Output prefix; the run is {out}.senna.json [default: today's results folder, by data file and method]"},
                    "args": {"type": "array", "items": {"type": "string"}, "description": "Further flags, as run_help lists them, e.g. [\"-b\", \"batch.txt\"]"},
                    "background": {"type": "boolean", "description": "Run the fit as a job and return its id at once, for a fit longer than a call can wait; follow it with job_status or job_wait [default: false]"},
                },
                "required": ["method", "data"],
                "additionalProperties": false,
            }),
        ),
        spec(
            "run_help",
            "Every flag of one fit, or of the command another run tool starts, with its \
             help, as `senna {method} --help` prints it.",
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
