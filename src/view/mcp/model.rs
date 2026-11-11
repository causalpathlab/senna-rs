//! `run_predict` and `run_update`: a trained model on new data.

use super::jobs::{default_out, job_property, server_job, start, Task};
use super::parse;
use anyhow::{Context, Result};
use legume_mcp::{Call, Content, ToolSpec};
use senna::run_manifest;
use serde::Deserialize;
use serde_json::{json, Value};
use std::path::Path;

/// A model tool's call; `None` for another family's.
pub(super) fn call(name: &str, args: &Value, cx: &Call) -> Result<Option<Vec<Content>>> {
    Ok(Some(match name {
        "run_predict" => run_predict(parse(args.clone())?, cx)?,
        "run_update" => run_update(parse(args.clone())?, cx)?,
        _ => return Ok(None),
    }))
}

/// `run_predict`'s and `run_update`'s arguments.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ModelArgs {
    model: String,
    #[serde(default)]
    data: Vec<String>,
    out: Option<String>,
    #[serde(default)]
    args: Vec<String>,
    #[serde(default)]
    background: bool,
}

/// `senna {command} {data} --model {model} -o {out} {args}`.
fn model_argv(command: &str, a: &ModelArgs, out: &str) -> Vec<String> {
    let mut argv = vec![command.to_string()];
    argv.extend(a.data.iter().cloned());
    argv.extend(["--model".into(), a.model.clone(), "-o".into(), out.into()]);
    argv.extend(a.args.iter().cloned());
    argv
}

fn latent_path(out: &str) -> String {
    format!("{out}.latent.parquet")
}

/// `senna predict` into `out`: the latent and the per-cell predictive fit.
fn predict_task(a: &ModelArgs, out: &str) -> Task {
    Task {
        argv: model_argv("predict", a, out),
        made: vec![
            ("latent", latent_path(out).into()),
            ("predictive", format!("{out}.predictive.parquet").into()),
        ],
        summary: format!("senna predict of {} by {}", a.data.join(", "), a.model),
    }
}

/// The new cells in a trained model's latent, as `senna predict` puts them.
fn run_predict(a: ModelArgs, cx: &Call) -> Result<Vec<Content>> {
    let out = match &a.out {
        Some(out) => out.clone(),
        None => {
            let data = a
                .data
                .first()
                .context("no data files to name the out by: pass out")?;
            // By model too: the same cells in another model's latent are
            // another result.
            let model = Path::new(&a.model).file_name().unwrap_or_default();
            let tag = format!("{}_predict", model.to_string_lossy());
            default_out(cx, data, &tag, latent_path)?
        }
    };
    let done = start(predict_task(&a, &out), a.background, cx, |t, o| {
        server_job(t, o, cx)
    })?;
    Ok(vec![Content::Json(done)])
}

/// `senna update` into `out`: a new run.
fn update_task(a: &ModelArgs, out: &str) -> Task {
    let summary = match a.data.is_empty() {
        true => format!("senna update round of {}", a.model),
        false => format!("senna update of {} with {}", a.model, a.data.join(", ")),
    };
    Task {
        argv: model_argv("update", a, out),
        made: vec![("run", run_manifest::default_path(out).into())],
        summary,
    }
}

/// A trained model continued over new data, or a round on its own cells:
/// a new run.
fn run_update(a: ModelArgs, cx: &Call) -> Result<Vec<Content>> {
    let out = a.out.as_deref().context("pass out: the new run's prefix")?;
    let done = start(update_task(&a, out), a.background, cx, |t, o| {
        server_job(t, o, cx)
    })?;
    Ok(vec![Content::Json(done)])
}

pub(super) fn specs() -> Vec<ToolSpec> {
    let spec = ToolSpec::new;
    let job = job_property();
    vec![
        spec(
            "run_predict",
            "Place new cells in a trained model's latent as `senna predict {data} --model \
             {model} -o {out} {args}`, and return the latent's path (and the per-cell \
             predictive fit's). The model is a topic, masked-topic, masked-vae, \
             masked-sbp, vae, svd, bge or simba run's prefix. An out whose latent exists is \
             not run again. Writes no run to view_open and no layout.",
            json!({
                "type": "object",
                "properties": {
                    "model": {"type": "string", "description": "Trained run's prefix (its {out}, without .senna.json)"},
                    "data": {"type": "array", "items": {"type": "string"}, "description": "New data files (.zarr, .zarr.zip, .h5); none with --bulk in args"},
                    "out": {"type": "string", "description": "Output prefix: {out}.latent.parquet, {out}.predictive.parquet [default: today's results folder, by data file]"},
                    "args": {"type": "array", "items": {"type": "string"}, "description": "Further flags, as run_help (predict) lists them, e.g. [\"-b\", \"batch.txt\"]"},
                    "background": job,
                },
                "required": ["model"],
                "additionalProperties": false,
            }),
        ),
        spec(
            "run_update",
            "Continue a trained model as `senna update {data} --model {model} -o {out} \
             {args}`: it absorbs the new data into a new run, or with no data runs a \
             round on its own cells. Returns the new run's manifest, for view_open. An \
             out whose manifest exists is not run again.",
            json!({
                "type": "object",
                "properties": {
                    "model": {"type": "string", "description": "Parent run's prefix (its {out}, without .senna.json)"},
                    "data": {"type": "array", "items": {"type": "string"}, "description": "New data files to absorb; none makes a round"},
                    "out": {"type": "string", "description": "The new run's prefix, not the parent's"},
                    "args": {"type": "array", "items": {"type": "string"}, "description": "Further flags, as run_help (update) lists them; -b names the new data's batch files when the parent had batches"},
                    "background": job,
                },
                "required": ["model", "out"],
                "additionalProperties": false,
            }),
        ),
    ]
}

#[cfg(test)]
#[path = "../tests/mcp_model.rs"]
mod tests;
