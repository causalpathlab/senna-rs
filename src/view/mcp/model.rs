//! `run_update` and `run_critique`: a trained model continued into a new
//! run, and fits that question each other for its rounds.

use super::flags::{self, given, parsed};
use super::jobs::{job_property, start_server, Argv, Task};
use super::parse;
use anyhow::{ensure, Context, Result};
use legume_mcp::{Call, Content, ToolSpec};
use senna::run_manifest;
use serde::Deserialize;
use serde_json::{json, Map, Value};

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
    flags: Map<String, Value>,
    #[serde(default)]
    background: bool,
}

/// `senna update` into `out`: a new run, from new data or, with none, a
/// round on the parent's own cells; its flags checked first, as senna
/// parses them and for what senna would refuse only once it ran.
fn update_task(a: &UpdateCall, out: &str) -> Result<Task> {
    let model = run_manifest::derive_out_prefix(&a.model);
    ensure!(
        out != model,
        "out must differ from model: write a new version and keep the parent"
    );
    let argv = Argv::new("update")
        .args(a.data.iter().cloned())
        .flag("--model", Some(&model))
        .flag("-o", Some(out))
        .rest(&flags::words("update", &a.flags)?);
    let m = parsed(&argv)?;
    let round = a.data.is_empty();
    ensure!(
        !round || given(&m, "batch_files").is_empty(),
        "a round replays the parent's batches: no batch-files"
    );
    ensure!(
        round || given(&m, "pb_from").is_empty(),
        "pb-from is for a round: a partition of the parent's cells cannot cover new ones"
    );
    let summary = match round {
        true => format!("senna update round of {}", a.model),
        false => format!("senna update of {} with {}", a.model, a.data.join(", ")),
    };
    Ok(Task {
        argv,
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
    #[serde(default)]
    flags: Map<String, Value>,
    #[serde(default)]
    background: bool,
}

/// `senna critique` of the runs into `out`: its pair and summary tables
/// and its record; and whether it draws questions, which write the peer
/// labels too.
fn critique_task(a: &CritiqueCall) -> Result<(Task, bool)> {
    ensure!(a.runs.len() >= 2, "critique compares two runs or more");
    let out = &a.out;
    let argv = Argv::new("critique")
        .args(a.runs.iter().cloned())
        .flag("-o", Some(out))
        .rest(&flags::words("critique", &a.flags)?);
    let questions = given(&parsed(&argv)?, "questions")
        .first()
        .is_some_and(|q| q.parse::<usize>().is_ok_and(|q| q > 0));
    let task = Task {
        argv,
        made: vec![
            ("summary", format!("{out}.critique.summary.parquet").into()),
            ("pairs", format!("{out}.critique.pairs.parquet").into()),
            ("record", format!("{out}.critique.json").into()),
        ],
        summary: format!("senna critique of {}", a.runs.join(", ")),
    };
    Ok((task, questions))
}

/// Where fits disagree on which pseudobulks are alike.
fn run_critique(a: CritiqueCall, cx: &Call) -> Result<Vec<Content>> {
    let (task, questions) = critique_task(&a)?;
    let mut done = start_server(task, a.background, cx)?;
    // Labels a critique there already made without questions lack.
    if questions && done["ran"] != json!(false) {
        done["labels"] = json!(format!(
            "{}.critique.labels.{{model}}.parquet, one per run: run_update's flags peer-labels",
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
             parent's fit re-run over its recorded data plus these, warm-started (flags \
             batch-files for the new data when the parent had batches). With no data, it runs \
             a round on the parent's own cells, taking flags peer-labels from run_critique \
             (questions) and optionally pb-from's partition. Writes the new run, {out}.senna.json (returned, \
             for view_open); the parent is kept. An out whose run exists is not run again.",
            json!({
                "type": "object",
                "properties": {
                    "model": {"type": "string", "description": "Parent run: its prefix, or its manifest"},
                    "data": {"type": "array", "items": {"type": "string"}, "description": "New data files to absorb; none makes a round"},
                    "out": {"type": "string", "description": "The new run's prefix, not the parent's"},
                    "flags": flags::property("senna update", r#"{"batch-files": ["b.txt"]} when absorbing; {"peer-labels": "c.critique.labels.m1.parquet", "pb-from": "p.senna.json"} for a round"#),
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
             with flags questions, also {out}.critique.labels.{model}.parquet, the pairs each \
             model is to push apart in a run_update round.",
            json!({
                "type": "object",
                "properties": {
                    "runs": {"type": "array", "items": {"type": "string"}, "minItems": 2, "description": "Run manifests to compare (topic, masked-*, vae, svd, bge, simba)"},
                    "out": {"type": "string", "description": "Output prefix"},
                    "flags": flags::property("senna critique", r#"{"partition": "a.senna.json", "questions": 50}"#),
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
