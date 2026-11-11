//! `run_update` and `run_critique`: a trained model continued into a new
//! run, and fits that question each other for its rounds.

use super::jobs::{job_property, start_server, Argv, Task};
use super::parse;
use anyhow::{ensure, Context, Result};
use legume_mcp::{Call, Content, ToolSpec};
use senna::run_manifest;
use serde::Deserialize;
use serde_json::{json, Value};

/// A model tool's call; `None` for another family's.
pub(super) fn call(name: &str, args: &Value, cx: &Call) -> Result<Option<Vec<Content>>> {
    Ok(Some(match name {
        "run_update" => run_update(parse(args.clone())?, cx)?,
        "run_critique" => run_critique(parse(args.clone())?, cx)?,
        _ => return Ok(None),
    }))
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct UpdateCall {
    model: String,
    #[serde(default)]
    data: Vec<String>,
    out: Option<String>,
    #[serde(default)]
    batch_files: Vec<String>,
    pb_from: Option<String>,
    peer_labels: Option<String>,
    #[serde(default)]
    args: Vec<String>,
    #[serde(default)]
    background: bool,
}

/// `senna update` into `out`: a new run, from new data or, with none, a
/// round on the parent's own cells; its flags checked first.
fn update_task(a: &UpdateCall, out: &str) -> Result<Task> {
    let model = run_manifest::derive_out_prefix(&a.model);
    ensure!(
        out != model,
        "out must differ from model: write a new version and keep the parent"
    );
    let round = a.data.is_empty();
    ensure!(
        !round || a.batch_files.is_empty(),
        "a round replays the parent's batches: no batch_files"
    );
    ensure!(
        round || a.pb_from.is_none(),
        "pb_from is for a round: a partition of the parent's cells cannot cover new ones"
    );
    let summary = match round {
        true => format!("senna update round of {}", a.model),
        false => format!("senna update of {} with {}", a.model, a.data.join(", ")),
    };
    Ok(Task {
        argv: Argv::new("update")
            .args(a.data.iter().cloned())
            .flag("--model", Some(&model))
            .flag("-o", Some(out))
            .each("-b", &a.batch_files)
            .flag("--pb-from", a.pb_from.as_deref())
            .flag("--peer-labels", a.peer_labels.as_deref())
            .rest(&a.args),
        made: vec![("run", run_manifest::default_path(out).into())],
        summary,
    })
}

/// A trained model continued over new data, or a round on its own cells:
/// a new run.
fn run_update(a: UpdateCall, cx: &Call) -> Result<Vec<Content>> {
    let out = a.out.as_deref().context("pass out: the new run's prefix")?;
    let done = start_server(update_task(&a, out)?, a.background, cx)?;
    Ok(vec![Content::Json(done)])
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct CritiqueCall {
    runs: Vec<String>,
    out: String,
    partition: Option<String>,
    questions: Option<usize>,
    cell_labels: Option<String>,
    label_column: Option<String>,
    #[serde(default)]
    args: Vec<String>,
    #[serde(default)]
    background: bool,
}

/// `senna critique` of the runs into `out`: its pair and summary tables
/// and its record.
fn critique_task(a: &CritiqueCall) -> Result<Task> {
    ensure!(a.runs.len() >= 2, "critique compares two runs or more");
    let out = &a.out;
    Ok(Task {
        argv: Argv::new("critique")
            .args(a.runs.iter().cloned())
            .flag("-o", Some(out))
            .flag("--partition", a.partition.as_deref())
            .flag("--questions", a.questions)
            .flag("--cell-labels", a.cell_labels.as_deref())
            .flag("--label-column", a.label_column.as_deref())
            .rest(&a.args),
        made: vec![
            ("summary", format!("{out}.critique.summary.parquet").into()),
            ("pairs", format!("{out}.critique.pairs.parquet").into()),
            ("record", format!("{out}.critique.json").into()),
        ],
        summary: format!("senna critique of {}", a.runs.join(", ")),
    })
}

/// Where fits disagree on which pseudobulks are alike.
fn run_critique(a: CritiqueCall, cx: &Call) -> Result<Vec<Content>> {
    let mut done = start_server(critique_task(&a)?, a.background, cx)?;
    // Labels a critique there already made without questions lack.
    if a.questions.is_some_and(|q| q > 0) && done["ran"] != json!(false) {
        done["labels"] = json!(format!(
            "{}.critique.labels.{{model}}.parquet, one per run: run_update's peer_labels",
            a.out
        ));
    }
    Ok(vec![Content::Json(done)])
}

pub(super) fn specs() -> Vec<ToolSpec> {
    let spec = ToolSpec::new;
    let job = job_property();
    vec![
        spec(
            "run_update",
            "Continue a trained model into a new run as `senna update {data} --model {model} \
             -o {out}`, in one of two modes. With data, it absorbs the new samples: the \
             parent's fit re-run over its recorded data plus these, warm-started (batch_files \
             for the new data when the parent had batches). With no data, it runs a round on \
             the parent's own cells, taking peer_labels from run_critique (questions) and \
             optionally pb_from's partition. Writes the new run, {out}.senna.json (returned, \
             for view_open); the parent is kept. An out whose run exists is not run again.",
            json!({
                "type": "object",
                "properties": {
                    "model": {"type": "string", "description": "Parent run: its prefix, or its manifest"},
                    "data": {"type": "array", "items": {"type": "string"}, "description": "New data files to absorb; none makes a round"},
                    "out": {"type": "string", "description": "The new run's prefix, not the parent's"},
                    "batch_files": {"type": "array", "items": {"type": "string"}, "description": "Absorbing: batch membership of the new data (-b), one per data file; needed when the parent had batches"},
                    "pb_from": {"type": "string", "description": "Round only: collapse on this run's partition (--pb-from), so every model in a critique round shares one"},
                    "peer_labels": {"type": "string", "description": "Round: pairs to push apart, from run_critique with questions (--peer-labels)"},
                    "args": {"type": "array", "items": {"type": "string"}, "description": "Further flags, as run_help (update) lists them"},
                    "background": job,
                },
                "required": ["model", "out"],
                "additionalProperties": false,
            }),
        ),
        spec(
            "run_critique",
            "Report where fits disagree on which pseudobulks are alike, as `senna critique \
             {runs} -o {out}`: each fit judged against the median rank of the others. Reads \
             only the runs' latents. Writes {out}.critique.summary.parquet (merges and merge \
             rate per model and level), {out}.critique.pairs.parquet and {out}.critique.json; \
             with questions, also {out}.critique.labels.{model}.parquet, the pairs each model \
             is to push apart in a run_update round.",
            json!({
                "type": "object",
                "properties": {
                    "runs": {"type": "array", "items": {"type": "string"}, "minItems": 2, "description": "Run manifests to compare (topic, masked-*, vae, svd, bge, simba)"},
                    "out": {"type": "string", "description": "Output prefix"},
                    "partition": {"type": "string", "description": "A run whose pseudobulks every fit is averaged over (--partition) [default: the first that has one]"},
                    "questions": {"type": "integer", "minimum": 0, "description": "Questions to draw per model, written as its peer labels (--questions)"},
                    "cell_labels": {"type": "string", "description": "Known cell labels to check the merges against (--cell-labels); with label_column"},
                    "label_column": {"type": "string", "description": "The cell_labels column that holds the labels (--label-column)"},
                    "args": {"type": "array", "items": {"type": "string"}, "description": "Further flags, as run_help (critique) lists them"},
                    "background": job,
                },
                "required": ["runs", "out"],
                "additionalProperties": false,
            }),
        ),
    ]
}

#[cfg(test)]
#[path = "../tests/mcp_model.rs"]
mod tests;
