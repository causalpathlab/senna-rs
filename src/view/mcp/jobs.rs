//! The senna commands the run tools start: built, checked as senna
//! parses them, and run in the call or as a job.

use super::absolute;
use anyhow::{ensure, Context, Result};
use legume_mcp::{Call, Step, Why};
use senna::run_manifest;
use serde_json::{json, Value};
use std::path::{Path, PathBuf};
use std::process::Command;

/// The out given, or one in today's results folder named by the data
/// file `data` and `tag`: `{stem}_{tag}`, then `_2`, `_3`, …, the first
/// whose files no running job is writing and are not half there; with
/// the task `plan` makes of it. Not legume-mcp's `free_prefix`, which
/// also passes over a name whose files exist: here that name is taken, so
/// a result already made is not made again.
pub(super) fn planned(
    cx: &Call,
    out: Option<&str>,
    data: Option<&str>,
    tag: &str,
    plan: impl Fn(&str) -> Result<Task>,
) -> Result<(String, Task)> {
    if let Some(out) = out {
        return Ok((out.to_string(), plan(out)?));
    }
    let data = data.context("no data files to name the out by: pass out")?;
    let dir = cx
        .results_dir()
        .context("no results folder (records are off): pass out")?;
    let data = Path::new(data).file_name().unwrap_or_default();
    let stem = data.to_string_lossy();
    let stem = stem.split('.').next().unwrap_or("run");
    // As many as there could be jobs, and more.
    for k in 1..=1000 {
        let name = match k {
            1 => format!("{stem}_{tag}"),
            k => format!("{stem}_{tag}_{k}"),
        };
        let out = dir.join(name).to_string_lossy().into_owned();
        // What the plan refuses, it refuses for every name.
        let task = plan(&out)?;
        let outputs = task.outputs();
        if cx.ensure_free(&outputs).is_ok() && already(&outputs).is_ok() {
            return Ok((out, task));
        }
    }
    anyhow::bail!("every default name is being written by a running job or half there: pass out")
}

/// `{model}_{what}`, by the model run's file name: the tag of a default
/// out, as the same cells under another model are another result.
pub(super) fn model_tag(model: &str, what: &str) -> String {
    let prefix = run_manifest::derive_out_prefix(model);
    let name = Path::new(&prefix).file_name().unwrap_or_default();
    format!("{}_{what}", name.to_string_lossy())
}

/// A senna command a run tool starts, and the files it makes, each under
/// the name the result gives it; all of them there show it ran.
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
    /// recorded to run every time; with the reason for its choices, `why`.
    pub(super) fn run(&self, guarded: bool, cx: &Call, why: Option<&Why>) -> Result<()> {
        let outputs = if guarded { self.outputs() } else { vec![] };
        cx.progress(&self.summary);
        cx.run_step(senna(&self.argv)?, self.step(&outputs, why))
            .with_context(|| self.line())?;
        Ok(())
    }

    /// The step as the records keep it: its outputs, summary and `why`.
    fn step(&self, outputs: &[PathBuf], why: Option<&Why>) -> Step {
        let step = Step::new(outputs, &self.summary);
        match why {
            Some(why) => step.why(why.clone()),
            None => step,
        }
    }
}

/// `task` run, with the reason for its choices `why`, or with
/// `background` started by `spawn` (given the task and its outputs, giving
/// the job as the agent is told it). A task whose files are there already
/// is not run; one with only some of them there is refused, as running it
/// would write over them.
pub(super) fn start(
    task: Task,
    background: bool,
    cx: &Call,
    why: Option<&Why>,
    spawn: impl FnOnce(&Task, &[PathBuf]) -> Result<Value>,
) -> Result<Value> {
    senna_parses(&task.argv)?;
    let mut done = task.named();
    let outputs = task.outputs();
    // A job still writing them is named before what it has written so
    // far is taken for a result.
    cx.ensure_free(&outputs)?;
    if already(&outputs)? {
        done["ran"] = json!(false);
        done["note"] = json!("it is there already; another out runs anew");
    } else if background {
        let job = spawn(&task, &outputs).with_context(|| task.line())?;
        done.as_object_mut()
            .expect("named() is an object")
            .extend(job.as_object().cloned().unwrap_or_default());
    } else {
        task.run(true, cx, why)?;
        done["ran"] = json!(true);
    }
    Ok(done)
}

/// [`start`], a background task as this server's job.
pub(super) fn start_server(
    task: Task,
    background: bool,
    cx: &Call,
    why: Option<&Why>,
) -> Result<Value> {
    start(task, background, cx, why, |t, o| server_job(t, o, cx, why))
}

/// `task` as this server's job, which ends with the server.
pub(super) fn server_job(
    task: &Task,
    outputs: &[PathBuf],
    cx: &Call,
    why: Option<&Why>,
) -> Result<Value> {
    Ok(json!({
        "job": cx.spawn_step(senna(&task.argv)?, task.step(outputs, why))?,
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

/// Whether every one of `outputs` is there: a result made already. Some
/// there and not the rest is an error, as making them would write over
/// those that are.
pub(super) fn already(outputs: &[PathBuf]) -> Result<bool> {
    let (there, missing): (Vec<_>, Vec<_>) = outputs.iter().partition(|p| p.exists());
    match (there.first(), missing.first()) {
        (_, None) => Ok(true),
        (None, _) => Ok(false),
        (Some(there), Some(missing)) => anyhow::bail!(
            "{} is there already, without {}; not written over: pass another out",
            there.display(),
            missing.display()
        ),
    }
}

/// A senna command line, built word by word; what is not given adds
/// nothing.
pub(super) struct Argv(Vec<String>);

impl Argv {
    pub(super) fn new(command: &str) -> Self {
        Self(vec![command.into()])
    }

    /// Words as they are: the command's positionals, or a flag's.
    pub(super) fn args<S: Into<String>>(mut self, words: impl IntoIterator<Item = S>) -> Self {
        self.0.extend(words.into_iter().map(Into::into));
        self
    }

    /// `flag value`, when there is a value.
    pub(super) fn flag(self, flag: &str, value: Option<impl ToString>) -> Self {
        match value {
            Some(v) => self.args([flag.to_string(), v.to_string()]),
            None => self,
        }
    }

    /// `flag` alone, when `on`.
    pub(super) fn switch(self, flag: &str, on: bool) -> Self {
        if on {
            self.args([flag])
        } else {
            self
        }
    }

    /// `flag v1 v2 …` for a flag that takes several values, when there
    /// are any.
    pub(super) fn many(self, flag: &str, values: &[String]) -> Self {
        if values.is_empty() {
            self
        } else {
            self.args([flag]).args(values.iter().cloned())
        }
    }

    /// The command line, ending with the words of the agent's flags.
    pub(super) fn rest(self, flags: &[String]) -> Vec<String> {
        self.args(flags.iter().cloned()).0
    }
}

#[cfg(test)]
#[path = "../tests/mcp_jobs.rs"]
mod tests;
