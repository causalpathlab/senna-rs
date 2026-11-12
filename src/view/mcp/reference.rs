//! `run_impute` and `run_deconvolve`: new data read through a reference
//! of annotated or fully measured cells.

use super::jobs::{job_property, model_tag, planned, start_server, Argv, Task};
use super::parse;
use anyhow::{ensure, Result};
use legume_mcp::{Call, Content, ToolSpec};
use senna::run_manifest;
use serde::Deserialize;
use serde_json::{json, Value};

/// A reference tool's call; `None` for another family's.
pub(super) fn call(name: &str, args: &Value, cx: &Call) -> Result<Option<Vec<Content>>> {
    Ok(Some(match name {
        "run_impute" => run_impute(parse(args.clone())?, cx)?,
        "run_deconvolve" => run_deconvolve(parse(args.clone())?, cx)?,
        _ => return Ok(None),
    }))
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ImputeCall {
    model: String,
    data: Vec<String>,
    out: Option<String>,
    reference: Option<String>,
    #[serde(default)]
    batch_files: Vec<String>,
    knn: Option<usize>,
    #[serde(default)]
    args: Vec<String>,
    #[serde(default)]
    background: bool,
}

/// `senna impute` of `a` into `out`: every reference feature on each new
/// cell.
fn impute_task(a: &ImputeCall, out: &str) -> Result<Task> {
    ensure!(!a.data.is_empty(), "no data files");
    Ok(Task {
        argv: Argv::new("impute")
            .args(a.data.iter().cloned())
            .flag("--model", Some(run_manifest::derive_out_prefix(&a.model)))
            .flag("-o", Some(out))
            .flag("--reference", a.reference.as_deref())
            .each("-b", &a.batch_files)
            .flag("--knn", a.knn)
            .rest(&a.args),
        made: vec![("imputed", format!("{out}.imputed.parquet").into())],
        summary: format!("senna impute of {} by {}", a.data.join(", "), a.model),
    })
}

/// Full-feature counts on new cells, pooled from their nearest reference
/// cells in the model's latent.
fn run_impute(a: ImputeCall, cx: &Call) -> Result<Vec<Content>> {
    let data = a.data.first().map(String::as_str);
    let tag = model_tag(&a.model, "impute");
    let (_, task) = planned(cx, a.out.as_deref(), data, &tag, |o| impute_task(&a, o))?;
    let done = start_server(task, a.background, cx)?;
    Ok(vec![Content::Json(done)])
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct DeconvolveCall {
    from: String,
    bulk: Vec<String>,
    out: Option<String>,
    annotation: Option<String>,
    #[serde(default)]
    args: Vec<String>,
    #[serde(default)]
    background: bool,
}

/// `senna deconvolve` of the bulk against the run `from`: the fractions
/// first, then their intervals, the abundances and the residual.
fn deconvolve_task(a: &DeconvolveCall) -> Result<Task> {
    ensure!(!a.bulk.is_empty(), "no bulk tables");
    // senna's own default, named here so the files can be checked.
    let out = match &a.out {
        Some(out) => out.clone(),
        None => crate::deconvolve::default_out(&a.from),
    };
    let made = ["fractions", "fractions_ci", "abundance", "residual"]
        .map(|t| (t, format!("{out}.{t}.tsv").into()));
    Ok(Task {
        argv: Argv::new("deconvolve")
            .flag("--from", Some(&a.from))
            .many("--bulk", &a.bulk)
            .flag("-o", Some(&out))
            .flag("--annotation", a.annotation.as_deref())
            .rest(&a.args),
        made: made.to_vec(),
        summary: format!("senna deconvolve of {} by {}", a.bulk.join(", "), a.from),
    })
}

/// Bulk samples split into cell-type fractions against the run's cells.
fn run_deconvolve(a: DeconvolveCall, cx: &Call) -> Result<Vec<Content>> {
    let done = start_server(deconvolve_task(&a)?, a.background, cx)?;
    Ok(vec![Content::Json(done)])
}

pub(super) fn specs() -> Vec<ToolSpec> {
    let spec = ToolSpec::new;
    let job = job_property();
    vec![
        spec(
            "run_impute",
            "Impute full-feature counts on new (often sparse-panel) cells as `senna impute \
             {data} --model {model} -o {out}`: each cell is placed in the model's latent and \
             pools the counts of its nearest reference cells (the model's own training cells \
             unless reference names another run). Writes {out}.imputed.parquet (new cells × \
             reference features), returned. Batches adjust only the query's latent (batch_files, \
             two or more batches); the imputed values are the neighbours' raw reference counts \
             and carry their batch effects and depths. An out whose table exists is not run \
             again.",
            json!({
                "type": "object",
                "properties": {
                    "model": {"type": "string", "description": "Trained run: its prefix, or its manifest (topic family, masked, vae, bge, simba or svd)"},
                    "data": {"type": "array", "items": {"type": "string"}, "minItems": 1, "description": "New data files (.zarr, .h5)"},
                    "out": {"type": "string", "description": "Output prefix [default: today's results folder, by data file and model]"},
                    "reference": {"type": "string", "description": "Run whose training cells are the reference (--reference) [default: the model's]"},
                    "batch_files": {"type": "array", "items": {"type": "string"}, "description": "Batch membership of the data (-b), one per data file; for the query's latent only"},
                    "knn": {"type": "integer", "minimum": 1, "description": "Reference neighbours pooled per new cell (--knn) [default: 25]"},
                    "args": {"type": "array", "items": {"type": "string"}, "description": "Further flags, as run_help (impute) lists them"},
                    "background": job,
                },
                "required": ["model", "data"],
                "additionalProperties": false,
            }),
        ),
        spec(
            "run_deconvolve",
            "Deconvolve bulk samples into cell-type fractions against a run's annotated cells, \
             as `senna deconvolve --from {from} --bulk {bulk} -o {out}` (a Gibbs sampler over \
             measured archetype profiles). Writes {out}.fractions.tsv (returned as fractions; \
             mRNA shares, not cell shares), {out}.fractions_ci.tsv, {out}.abundance.tsv, \
             {out}.residual.tsv and per-type expression. No batch effect is modelled: \
             reference profiles are raw sums with batches ignored, and nothing absorbs a \
             bulk-versus-single-cell platform shift. An out whose fractions exist is not run \
             again.",
            json!({
                "type": "object",
                "properties": {
                    "from": {"type": "string", "description": "Run manifest of a `senna bge` run, which supplies the gene axis, cells and by default the annotation"},
                    "bulk": {"type": "array", "items": {"type": "string"}, "minItems": 1, "description": "Bulk count tables (parquet, or tab/comma text)"},
                    "out": {"type": "string", "description": "Output prefix [default: from's prefix with .deconv]"},
                    "annotation": {"type": "string", "description": "Per-cell annotation, cells × cell types (--annotation) [default: the run's]"},
                    "args": {"type": "array", "items": {"type": "string"}, "description": "Further flags, as run_help (deconvolve) lists them"},
                    "background": job,
                },
                "required": ["from", "bulk"],
                "additionalProperties": false,
            }),
        ),
    ]
}

#[cfg(test)]
#[path = "../tests/mcp_reference.rs"]
mod tests;
