//! `run_step`: the analysis commands that share one shape, a senna
//! command on a run (or on data files) with its own flags.

use super::jobs::{job_property, senna, senna_parses, start_server, Argv, Task};
use super::parse;
use anyhow::{bail, ensure, Context, Result};
use legume_mcp::{Call, Content, ToolSpec};
use senna::run_manifest;
use serde::Deserialize;
use serde_json::{json, Value};

/// A step tool's call; `None` for another family's.
pub(super) fn call(name: &str, args: &Value, cx: &Call) -> Result<Option<Vec<Content>>> {
    Ok(Some(match name {
        "run_step" => run_step(parse(args.clone())?, cx)?,
        _ => return Ok(None),
    }))
}

/// The commands `run_step` runs, as senna names them.
#[derive(Deserialize, Clone, Copy, PartialEq, Eq)]
#[serde(rename_all = "kebab-case")]
enum Command {
    Clustering,
    ResolveTopics,
    ResolveEmbeddingSpace,
    EmbedDiag,
    JointTopic,
    JointSvd,
    Fne,
}

/// Every command of [`Command`], as senna names them.
pub(super) const COMMANDS: [&str; 7] = [
    Command::Clustering.name(),
    Command::ResolveTopics.name(),
    Command::ResolveEmbeddingSpace.name(),
    Command::EmbedDiag.name(),
    Command::JointTopic.name(),
    Command::JointSvd.name(),
    Command::Fne.name(),
];

impl Command {
    /// senna's name for it, which serde reads too.
    const fn name(self) -> &'static str {
        match self {
            Self::Clustering => "clustering",
            Self::ResolveTopics => "resolve-topics",
            Self::ResolveEmbeddingSpace => "resolve-embedding-space",
            Self::EmbedDiag => "embed-diag",
            Self::JointTopic => "joint-topic",
            Self::JointSvd => "joint-svd",
            Self::Fne => "fne",
        }
    }

    /// Whether it reads a finished run (`from`) rather than data files.
    fn on_run(self) -> bool {
        matches!(
            self,
            Self::Clustering | Self::ResolveTopics | Self::ResolveEmbeddingSpace | Self::EmbedDiag
        )
    }
}

#[derive(Deserialize, Clone, Copy, Default, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
enum Target {
    #[default]
    Cells,
    Features,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct StepCall {
    command: Command,
    from: Option<String>,
    #[serde(default)]
    data: Vec<String>,
    out: Option<String>,
    target: Option<Target>,
    #[serde(default)]
    args: Vec<String>,
    #[serde(default)]
    background: bool,
}

/// The command of `a`, and the files it makes (none for `embed-diag`,
/// which only reports); its inputs checked first.
fn step_task(a: &StepCall) -> Result<Task> {
    let c = a.command;
    let name = c.name();
    ensure!(
        a.target.is_none() || c == Command::Clustering,
        "target is for clustering"
    );
    let (argv, out) = if c.on_run() {
        let from = a
            .from
            .as_deref()
            .with_context(|| format!("{name} needs from, a run"))?;
        ensure!(
            a.data.is_empty(),
            "{name} reads the run from, not data files"
        );
        if c == Command::EmbedDiag {
            ensure!(a.out.is_none(), "embed-diag writes nothing: no out");
            let task = Task {
                argv: Argv::new(name).args([from]).rest(&a.args),
                made: vec![],
                summary: format!("senna embed-diag of {from}"),
            };
            return Ok(task);
        }
        let run = run_manifest::derive_out_prefix(from);
        // A clustering of the run's own name may be view_open's, made
        // with other flags: another needs a name of its own.
        ensure!(
            c != Command::Clustering || a.out.is_some(),
            "clustering needs out: the run's own clusters may be there with other flags"
        );
        let out = a.out.clone().unwrap_or_else(|| run.clone());
        ensure!(
            c != Command::ResolveEmbeddingSpace || out != run,
            "resolve-embedding-space makes a run of its own: pass an out other than from's"
        );
        (Argv::new(name).flag("--from", Some(from)), out)
    } else {
        // fne may take its edges through flags alone.
        ensure!(
            !a.data.is_empty() || c == Command::Fne,
            "{name} needs data files"
        );
        ensure!(a.from.is_none(), "{name} reads data files, not a run");
        let out = a.out.clone().with_context(|| format!("{name} needs out"))?;
        (Argv::new(name).args(a.data.iter().cloned()), out)
    };
    let features = a.target == Some(Target::Features);
    let argv = argv
        .flag("-o", Some(&out))
        .flag("--target", features.then_some("features"))
        .rest(&a.args);
    let made: Vec<(&'static str, String)> = match c {
        Command::Clustering if features => {
            vec![("clusters", format!("{out}.feature_clusters.parquet"))]
        }
        Command::Clustering => vec![("clusters", format!("{out}.clusters.parquet"))],
        Command::ResolveTopics => vec![
            ("latent", format!("{out}.latent.parquet")),
            ("dictionary", format!("{out}.softmax_dictionary.parquet")),
            ("topics", format!("{out}.topic_embedding.parquet")),
        ],
        Command::ResolveEmbeddingSpace | Command::JointTopic | Command::JointSvd | Command::Fne => {
            vec![("run", run_manifest::default_path(&out))]
        }
        Command::EmbedDiag => unreachable!("returned above"),
    };
    let on = a.from.clone().unwrap_or_else(|| a.data.join(", "));
    Ok(Task {
        argv,
        made: made.into_iter().map(|(k, p)| (k, p.into())).collect(),
        summary: format!("senna {name} of {on}"),
    })
}

/// One analysis command run on a run or on data files.
fn run_step(a: StepCall, cx: &Call) -> Result<Vec<Content>> {
    let task = step_task(&a)?;
    if task.made.is_empty() {
        // It reads and reports: its report is the result.
        if a.background {
            bail!("{} reports at once: no background", a.command.name());
        }
        senna_parses(&task.argv)?;
        cx.progress(&task.summary);
        let text = cx
            .run_reading(senna(&task.argv)?)
            .with_context(|| task.line())?;
        return Ok(vec![Content::Text(text)]);
    }
    let done = start_server(task, a.background, cx)?;
    Ok(vec![Content::Json(done)])
}

pub(super) fn specs() -> Vec<ToolSpec> {
    vec![ToolSpec::new(
        "run_step",
        "Run one analysis command as `senna {command} …`. On a finished run (from): \
         clustering (-m kmeans|leiden|hsblock in args; writes {out}.clusters.parquet, or with \
         target features {out}.feature_clusters.parquet, and records it in the run), \
         resolve-topics (one topic per cell cluster for a run with none; writes \
         {out}.latent, softmax_dictionary and topic_embedding .parquet), \
         resolve-embedding-space (a topic run recast as a shared cell and gene embedding; \
         writes a new run {out}.senna.json) and embed-diag (reports the embedding tables' \
         geometry; writes nothing). On data files: joint-topic and joint-svd (fits across \
         modalities; --modalities in args) and fne (a feature-graph embedding from edge \
         lists); each writes a run {out}.senna.json. Returns the files made; one whose \
         files exist is not run again, and one with only some of them is refused.",
        json!({
            "type": "object",
            "properties": {
                "command": {"enum": COMMANDS},
                "from": {"type": "string", "description": "Run manifest the command reads (clustering, resolve-*, embed-diag)"},
                "data": {"type": "array", "items": {"type": "string"}, "description": "Input files (joint-topic, joint-svd: data files; fne: edge lists)"},
                "out": {"type": "string", "description": "Output prefix [default for resolve-topics: from's own prefix; required otherwise, and for resolve-embedding-space not from's]"},
                "target": {"enum": ["cells", "features"], "description": "clustering only: the cells, or the run's features [default: cells]"},
                "args": {"type": "array", "items": {"type": "string"}, "description": "Further flags, as run_help ({command}) lists them"},
                "background": job_property(),
            },
            "required": ["command"],
            "additionalProperties": false,
        }),
    )]
}

#[cfg(test)]
#[path = "../tests/mcp_steps.rs"]
mod tests;
