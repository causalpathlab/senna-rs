//! The senna commands the run tools start: built, checked as senna
//! parses them, and run in the call or as a job.

use super::absolute;
use anyhow::{ensure, Context, Result};
use legume_mcp::Call;
use serde_json::{json, Value};
use std::path::{Path, PathBuf};
use std::process::Command;

/// An out in today's results folder, named by the data file `data` and
/// `tag`: `{stem}_{tag}`, then `_2`, `_3`, …, the first whose made file
/// (`made` of the out) no running job is writing. Not legume-mcp's
/// `free_prefix`, which also passes over a name whose files exist: here
/// that name is taken, so a result already made is not made again.
pub(super) fn default_out(
    cx: &Call,
    data: &str,
    tag: &str,
    made: fn(&str) -> String,
) -> Result<String> {
    let dir = cx
        .results_dir()
        .context("no results folder (records are off): pass out")?;
    let data = Path::new(data).file_name().unwrap_or_default();
    let stem = data.to_string_lossy();
    let stem = stem.split('.').next().unwrap_or("run");
    // As many as there could be jobs, and more.
    (1..=1000)
        .map(|k| match k {
            1 => format!("{stem}_{tag}"),
            k => format!("{stem}_{tag}_{k}"),
        })
        .map(|name| dir.join(name).to_string_lossy().into_owned())
        .find(|out| cx.ensure_free(&[absolute(Path::new(&made(out)))]).is_ok())
        .context("every default name is being written by a running job: pass out")
}

/// A senna command a run tool starts, and the files it makes, each under
/// the name the result gives it; the first shows it ran.
#[derive(Debug)]
pub(super) struct Task {
    pub(super) argv: Vec<String>,
    pub(super) made: Vec<(&'static str, PathBuf)>,
    pub(super) summary: String,
}

impl Task {
    pub(super) fn outputs(&self) -> Vec<PathBuf> {
        self.made.iter().map(|(_, p)| absolute(p)).collect()
    }

    /// The made files by name, as the result gives them.
    pub(super) fn named(&self) -> Value {
        let named = self
            .made
            .iter()
            .map(|(k, p)| (k.to_string(), json!(absolute(p))));
        Value::Object(named.collect())
    }

    /// The command line, as an error names it.
    pub(super) fn line(&self) -> String {
        format!("senna {}", legume_mcp::command_line(&self.argv))
    }

    /// `senna {argv}` to its end, verbose, its log the call's progress;
    /// `guarded`, recorded with its outputs as its guard on replay, else
    /// recorded to run every time.
    pub(super) fn run(&self, guarded: bool, cx: &Call) -> Result<()> {
        let outputs = if guarded { self.outputs() } else { vec![] };
        cx.progress(&self.summary);
        cx.run_making(senna(&self.argv)?, &outputs, &self.summary)
            .with_context(|| self.line())?;
        Ok(())
    }
}

/// `task` run, or with `background` started by `spawn` (given the task
/// and its outputs, giving the job as the agent is told it). A task whose
/// first file is there already is not run.
pub(super) fn start(
    task: Task,
    background: bool,
    cx: &Call,
    spawn: impl FnOnce(&Task, &[PathBuf]) -> Result<Value>,
) -> Result<Value> {
    senna_parses(&task.argv)?;
    let mut done = task.named();
    let outputs = task.outputs();
    if outputs[0].exists() {
        done["ran"] = json!(false);
        done["note"] = json!("it is there already; another out runs anew");
    } else if background {
        let job = spawn(&task, &outputs).with_context(|| task.line())?;
        done.as_object_mut()
            .expect("named() is an object")
            .extend(job.as_object().cloned().unwrap_or_default());
    } else {
        task.run(true, cx)?;
        done["ran"] = json!(true);
    }
    Ok(done)
}

/// `task` as this server's job, which ends with the server.
pub(super) fn server_job(task: &Task, outputs: &[PathBuf], cx: &Call) -> Result<Value> {
    Ok(json!({
        "job": cx.spawn_job(senna(&task.argv)?, outputs, &task.summary)?,
        "note": "it runs as this job until it ends or the server does; follow it with job_status or job_wait",
    }))
}

/// `argv` as senna parses it: a call senna would refuse is refused here,
/// before senna is started, with clap's error on one line.
pub(super) fn senna_parses(argv: &[String]) -> Result<crate::Cli> {
    let cli = <crate::Cli as clap::CommandFactory>::command();
    let matches = legume_mcp::cli::check(&cli, argv)?;
    Ok(<crate::Cli as clap::FromArgMatches>::from_arg_matches(
        &matches,
    )?)
}

/// `senna {argv}` as a child, verbose: its log is the call's progress.
pub(super) fn senna(argv: &[String]) -> Result<Command> {
    // This server's own binary, found even when it was reinstalled since.
    let mut senna = legume_mcp::own_command(argv)?;
    if !argv.iter().any(|f| f == "-v" || f == "--verbose") {
        senna.arg("-v");
    }
    Ok(senna)
}

/// `name`, if it is one of `list`.
pub(super) fn one_of(name: &str, list: &[impl AsRef<str>]) -> Result<()> {
    let list: Vec<&str> = list.iter().map(AsRef::as_ref).collect();
    ensure!(
        list.contains(&name),
        "{name} is not one of {}",
        list.join(", ")
    );
    Ok(())
}

/// The `background` property of a tool that runs as this server's job.
pub(super) fn job_property() -> Value {
    json!({"type": "boolean", "description": "Run as a job of this server and return its id at once; follow it with job_status or job_wait [default: false]"})
}
