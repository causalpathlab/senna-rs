//! `run_predict` and `run_probe`: new data scored against a trained model,
//! which neither changes.

use super::jobs::{job_property, model_tag, planned, start_server, Argv, Task};
use super::parse;
use crate::topic::model_metadata::resolve_run_kind;
use crate::topic::predict_eval::mean_finite;
use anyhow::{bail, ensure, Context, Result};
use legume_mcp::{Call, Content, ToolSpec};
use senna::run_manifest::{self, RunKind};
use serde::Deserialize;
use serde_json::{json, Value};
use std::path::{Path, PathBuf};

/// A scoring tool's call; `None` for another family's.
pub(super) fn call(name: &str, args: &Value, cx: &Call) -> Result<Option<Vec<Content>>> {
    Ok(Some(match name {
        "run_predict" => run_predict(parse(args.clone())?, cx)?,
        "run_probe" => run_probe(parse(args.clone())?, cx)?,
        _ => return Ok(None),
    }))
}

/// How `senna predict` finds a new cell's latent.
#[derive(Deserialize, Default, Clone, Copy, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
enum Latent {
    /// The encoder's pass, or the projection of a run with no encoder.
    #[default]
    Encoder,
    /// The encoder's latent refined by `--refine-steps` decoder-side steps.
    Refine,
    /// `--decoder-only`: no encoder, the latent fitted to the frozen decoder.
    Decoder,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct PredictCall {
    model: String,
    #[serde(default)]
    data: Vec<String>,
    #[serde(default)]
    bulk: Vec<String>,
    out: Option<String>,
    #[serde(default)]
    batch_files: Vec<String>,
    #[serde(default)]
    null_from: Vec<String>,
    eval_features: Option<String>,
    scoring_delta: Option<String>,
    #[serde(default)]
    latent: Latent,
    refine_steps: Option<usize>,
    #[serde(default)]
    args: Vec<String>,
    #[serde(default)]
    background: bool,
}

/// `senna predict` of `a` into `out`: the latent and the per-cell
/// predictive fit; its flags checked first.
fn predict_task(a: &PredictCall, out: &str) -> Result<Task> {
    ensure!(
        a.data.is_empty() != a.bulk.is_empty(),
        "give data files or bulk tables, not both or neither"
    );
    ensure!(
        a.batch_files.is_empty() || !a.bulk.is_empty() || a.batch_files.len() == a.data.len(),
        "{} batch files for {} data files: one per data file",
        a.batch_files.len(),
        a.data.len()
    );
    let steps = match (a.latent, a.refine_steps) {
        (Latent::Encoder, Some(_)) => bail!("refine_steps is for latent refine or decoder"),
        (Latent::Refine, None | Some(0)) => bail!("latent refine needs refine_steps (1 or more)"),
        (_, steps) => steps,
    };
    let what = a.data.iter().chain(&a.bulk).cloned().collect::<Vec<_>>();
    Ok(Task {
        argv: Argv::new("predict")
            .args(a.data.iter().cloned())
            .many("--bulk", &a.bulk)
            .flag("--model", Some(run_manifest::derive_out_prefix(&a.model)))
            .flag("-o", Some(out))
            .each("-b", &a.batch_files)
            .many("--null-from", &a.null_from)
            .flag("--eval-features", a.eval_features.as_deref())
            .flag("--scoring-delta", a.scoring_delta.as_deref())
            .switch("--decoder-only", a.latent == Latent::Decoder)
            .flag("--refine-steps", steps)
            .rest(&a.args),
        made: vec![
            ("latent", format!("{out}.latent.parquet").into()),
            ("predictive", format!("{out}.predictive.parquet").into()),
        ],
        summary: format!("senna predict of {} by {}", what.join(", "), a.model),
    })
}

/// Only the dense topic path has a decoder-side latent fit; the other
/// runs refuse the flags (bge, simba, svd) or ignore them (vae, masked).
fn check_latent(a: &PredictCall, kind: RunKind) -> Result<()> {
    ensure!(
        a.latent == Latent::Encoder || matches!(kind, RunKind::Topic | RunKind::JointTopic),
        "latent refine and decoder are for topic runs; a {kind} run is predicted from its \
         encoder or projection only (latent encoder)"
    );
    Ok(())
}

/// The cells of a predictive table and, when it has them, the means over
/// cells of the held-out and the null log-likelihood per count.
fn predict_summary(predictive: &Path) -> Result<Value> {
    use legume_numeric::matrix::traits::IoOps;
    let path = predictive.to_string_lossy();
    let t = senna::embed_common::Mat::from_parquet_with_row_names(&path, Some(0))
        .with_context(|| format!("reading {path}"))?;
    let mean = |name: &str| {
        let j = t.cols.iter().position(|c| &**c == name)?;
        // NaN marks a cell with no counts on the scored genes.
        Some(mean_finite(t.mat.column(j).iter().copied()))
    };
    let mut s = json!({ "cells": t.mat.nrows() });
    match (
        mean("eval_llik_per_count"),
        mean("eval_null_llik_per_count"),
    ) {
        (Some(held_out), Some(floor)) => {
            s["mean_eval_llik_per_count"] = json!(held_out);
            s["mean_eval_null_llik_per_count"] = json!(floor);
        }
        _ => {
            s["note"] = json!(
                "no eval_llik_per_count and eval_null_llik_per_count columns (an svd run, or \
                 no genes scored); the table's own llik is not comparable across families"
            )
        }
    }
    Ok(s)
}

/// `done`, with what `read` makes of the file it names `key`, once the
/// step has run (a job's has not yet).
fn with_summary(
    mut done: Value,
    key: &str,
    read: fn(&Path) -> Result<Value>,
) -> Result<Vec<Content>> {
    if done.get("job").is_none() {
        let path = PathBuf::from(done[key].as_str().unwrap_or_default());
        // Unreadable, the files are still the result: say why, beside them.
        match read(&path) {
            Ok(summary) => done
                .as_object_mut()
                .expect("named() is an object")
                .extend(summary.as_object().cloned().unwrap_or_default()),
            Err(e) => done["summary_error"] = json!(format!("{e:#}")),
        }
    }
    Ok(vec![Content::Json(done)])
}

/// The kind of the trained run `model`, a prefix or its manifest.
fn model_kind(model: &str) -> Result<RunKind> {
    resolve_run_kind(&run_manifest::derive_out_prefix(model))
}

/// New cells in a trained model's latent, as `senna predict` puts them,
/// with the held-out fit summed up.
fn run_predict(a: PredictCall, cx: &Call) -> Result<Vec<Content>> {
    // The run is read only when the latent mode depends on its kind.
    if a.latent != Latent::Encoder {
        check_latent(&a, model_kind(&a.model)?)?;
    }
    let data = a.data.first().or(a.bulk.first()).map(String::as_str);
    let tag = model_tag(&a.model, "predict");
    let (_, task) = planned(cx, a.out.as_deref(), data, &tag, |o| predict_task(&a, o))?;
    let done = start_server(task, a.background, cx)?;
    with_summary(done, "predictive", predict_summary)
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ProbeCall {
    model: String,
    data: Vec<String>,
    calibration: String,
    out: Option<String>,
    alpha: Option<f64>,
    #[serde(default)]
    counterfactual: bool,
    #[serde(default)]
    args: Vec<String>,
    #[serde(default)]
    background: bool,
}

/// `senna probe` of `a` into `out`: the verdict, then each cell's fit.
fn probe_task(a: &ProbeCall, out: &str) -> Task {
    Task {
        argv: Argv::new("probe")
            .args(a.data.iter().cloned())
            .flag("--model", Some(run_manifest::derive_out_prefix(&a.model)))
            .flag("--calibration", Some(&a.calibration))
            .flag("-o", Some(out))
            .flag("--alpha", a.alpha)
            .switch("--counterfactual", a.counterfactual)
            .rest(&a.args),
        made: vec![
            ("verdict", format!("{out}.probe.json").into()),
            ("cells", format!("{out}.probe.tsv").into()),
        ],
        summary: format!("senna probe of {} by {}", a.data.join(", "), a.model),
    }
}

/// The counterfactual is the masked models' only: theirs is the one small
/// gene-side block a batch can move.
fn check_probe(a: &ProbeCall, kind: RunKind) -> Result<()> {
    ensure!(
        !a.counterfactual || kind.is_masked_family(),
        "counterfactual is for masked-topic, masked-sbp and masked-vae runs; {} is a {kind} run",
        a.model
    );
    Ok(())
}

/// The probe's verdict and counterfactual effects, as it wrote them.
fn probe_summary(verdict: &Path) -> Result<Value> {
    let text = std::fs::read_to_string(verdict)
        .with_context(|| format!("reading {}", verdict.display()))?;
    Ok(serde_json::from_str(&text)?)
}

/// Whether a trained model explains new cells, as `senna probe` judges.
fn run_probe(a: ProbeCall, cx: &Call) -> Result<Vec<Content>> {
    if a.counterfactual {
        check_probe(&a, model_kind(&a.model)?)?;
    }
    let data = a.data.first().map(String::as_str);
    // The counterfactual is another result than the fit alone.
    let tag = model_tag(
        &a.model,
        if a.counterfactual {
            "probe_cf"
        } else {
            "probe"
        },
    );
    let (_, task) = planned(cx, a.out.as_deref(), data, &tag, |o| Ok(probe_task(&a, o)))?;
    let done = start_server(task, a.background, cx)?;
    with_summary(done, "verdict", probe_summary)
}

pub(super) fn specs() -> Vec<ToolSpec> {
    let spec = ToolSpec::new;
    let job = job_property();
    let files =
        |what: &str| json!({"type": "array", "items": {"type": "string"}, "description": what});
    vec![
        spec(
            "run_predict",
            "Place new cells in a trained model's latent as `senna predict {data} --model \
             {model} -o {out}`. Writes {out}.latent.parquet and {out}.predictive.parquet, and \
             returns both with the means over cells of the held-out and the null \
             log-likelihood per count when the table has both. Serves topic, masked-topic, \
             masked-sbp, masked-vae and vae runs through their encoder, an svd run by Nyström \
             projection onto its frozen dictionary, and a bge or simba run by projecting each \
             cell onto its frozen gene table. A model trained with batches needs the held-out \
             cells' batches (batch_files, one per data file). A run fitted with --adj-method \
             batch then reads its trained per-batch effect delta for batches it was trained on \
             (one batch is enough), and a bge run divides their counts by its trained fold; \
             otherwise the encoder's null is rebuilt from the held-out cells, which needs two \
             or more batches. The score of a topic, masked, vae, bge or simba run takes a \
             delta as scoring_delta says; by default the trained one for a training batch and \
             a cross-fitted one for other batches, never one fitted on the counts it scores, \
             and the null takes the same batch term. An svd run is scored with no delta. An out with a prediction is not written over: its \
             summary is returned instead, so a background prediction's is read by calling \
             again with its out.",
            json!({
                "type": "object",
                "properties": {
                    "model": {"type": "string", "description": "Trained run: its prefix, or its manifest {prefix}.senna.json"},
                    "data": files("New data files (.zarr, .zarr.zip, .h5); or bulk, not both"),
                    "bulk": files("Dense bulk count tables (parquet, or tab/comma text), genes × samples, instead of data (--bulk)"),
                    "out": {"type": "string", "description": "Output prefix [default: today's results folder, by data file and model]"},
                    "batch_files": files("Batch membership of the data (-b), one file per data file"),
                    "null_from": files("Training data whose gene composition is the null (--null-from); pass the train half"),
                    "scoring_delta": {"enum": ["auto", "train", "crossfit", "none", "heldout"], "description": "Per-batch delta the score is taken under (--scoring-delta). auto: trained delta for a batch named like a training batch, cross-fitted for other batches when there are two or more, none for a model trained without batches; train: trained only, refuses a new batch; crossfit: each fold of a batch under the delta of its other folds; none: the batch-free rate; heldout: fitted on the scored cells, optimistic [default: auto]"},
                    "eval_features": {"type": "string", "description": "Score on these features only, one name per line (--eval-features); the same file for every model compared"},
                    "latent": {"enum": ["encoder", "refine", "decoder"], "description": "encoder: the encoder's pass, or the projection; refine: then refine_steps decoder-side steps (--refine-steps); decoder: no encoder, fitted to the frozen decoder (--decoder-only). refine and decoder are for topic runs [default: encoder]"},
                    "refine_steps": {"type": "integer", "minimum": 1, "description": "Steps for refine (required) or decoder [decoder default: 100]"},
                    "args": {"type": "array", "items": {"type": "string"}, "description": "Further flags, as run_help (predict) lists them"},
                    "background": job,
                },
                "required": ["model"],
                "additionalProperties": false,
            }),
        ),
        spec(
            "run_probe",
            "Ask whether a frozen model explains new cells, as `senna probe {data} --model \
             {model} --calibration {calibration} -o {out}`: each cell's fit against a null \
             calibrated on cells the model already explains, and a batch verdict (covered or \
             novel). With counterfactual (masked-topic, masked-sbp and masked-vae runs only) \
             it also estimates what updating on this batch would do: benefit, the fit it \
             would gain on the new cells (would it teach the model), and forgetting, the fit \
             it would lose on the reference (would it damage it). Writes {out}.probe.json \
             (returned: verdict, flagged cells, the counterfactual effects) and \
             {out}.probe.tsv (per cell). Trains and writes no model. Serves masked, topic, \
             vae, bge and simba runs.",
            json!({
                "type": "object",
                "properties": {
                    "model": {"type": "string", "description": "Trained run: its prefix, or its manifest"},
                    "data": {"type": "array", "items": {"type": "string"}, "minItems": 1, "description": "Query data files (.zarr, .h5)"},
                    "calibration": {"type": "string", "description": "A backend of cells the model already explains (held-out training cells), which sets the null"},
                    "out": {"type": "string", "description": "Output prefix [default: today's results folder, by data file and model]"},
                    "alpha": {"type": "number", "exclusiveMinimum": 0, "exclusiveMaximum": 1, "description": "Null tail probability, the per-cell false-positive rate (--alpha) [default: 0.05]"},
                    "counterfactual": {"type": "boolean", "description": "Masked runs only: also estimate benefit and forgetting (--counterfactual) [default: false]"},
                    "args": {"type": "array", "items": {"type": "string"}, "description": "Further flags, as run_help (probe) lists them"},
                    "background": job,
                },
                "required": ["model", "data", "calibration"],
                "additionalProperties": false,
            }),
        ),
    ]
}

#[cfg(test)]
#[path = "../tests/mcp_predict.rs"]
mod tests;
