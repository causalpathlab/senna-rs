//! Why a run tool's call is made as it is: its `why`, `by` and `cites`,
//! kept with the step in the records (legume-mcp's [`Why`]); and the
//! answer to a question a fit left to the user, refused without one.

use anyhow::{bail, ensure, Context, Result};
use legume_mcp::{By, ToolSpec, Why};
use senna::run_manifest::{self, RunManifest};
use serde_json::{json, Value};
use std::path::Path;

/// Whether the tool `name` takes a why: every run tool but `run_help`,
/// which runs nothing.
fn takes_why(name: &str) -> bool {
    name.starts_with("run_") && name != "run_help"
}

/// `why`, `by` and `cites` taken out of the arguments `args` of the tool
/// `name`, as the reason its step keeps; `None` without a `why`. `by` is
/// required with a `why`, and neither it nor `cites` comes without one.
pub(super) fn take(name: &str, args: &mut Value) -> Result<Option<Why>> {
    if !takes_why(name) {
        return Ok(None);
    }
    let Some(args) = args.as_object_mut() else {
        return Ok(None);
    };
    let (why, by, cites) = (args.remove("why"), args.remove("by"), args.remove("cites"));
    let Some(why) = why else {
        ensure!(
            by.is_none() && cites.is_none(),
            "by and cites go with a why: pass why too"
        );
        return Ok(None);
    };
    let text = why.as_str().context("why is the reason, as text")?.trim();
    ensure!(!text.is_empty(), "why is empty: give the reason");
    let by = match by.as_ref().and_then(Value::as_str) {
        Some("user") => By::User,
        Some("agent") => By::Agent,
        Some("agent_accepted") => By::AgentAccepted,
        Some(other) => bail!("by is user, agent or agent_accepted, not {other}"),
        None => bail!(
            "with a why, by says who chose: user, agent, or agent_accepted (the agent \
             proposed it and the user accepted)"
        ),
    };
    let cites: Vec<String> = match cites {
        None => Vec::new(),
        Some(c) => serde_json::from_value(c).context("cites is a list of text")?,
    };
    Ok(Some(Why::new(text, by).citing(&cites)))
}

/// `why`, `by` and `cites` in the schema of every run tool among `specs`.
pub(super) fn add_properties(specs: &mut [ToolSpec]) {
    for spec in specs.iter_mut().filter(|s| takes_why(&s.name)) {
        if let Some(p) = spec.input_schema["properties"].as_object_mut() {
            p.insert(
                "why".into(),
                json!({"type": "string", "description": "Why the call is made as it is (its root cells, its settings): kept with the step in the records and the notebook. Required when the flags answer a question a fit left to the user (a needs_input option's rerun_args)"}),
            );
            p.insert(
                "by".into(),
                json!({"enum": ["user", "agent", "agent_accepted"], "description": "Who chose, with why: the user, the agent on its own, or the agent with the user's assent"}),
            );
            p.insert(
                "cites".into(),
                json!({"type": "array", "items": {"type": "string"}, "description": "With why: the evidence it rests on (a tool's result, a lupin annotate_why cite, a figure)"}),
            );
        }
    }
}

/// Refused when the words `flags` of a fit answer a question a fit left to
/// the user (an option's `rerun_args` in the manifest of a run beside the
/// file it names) and no `why` says why.
pub(super) fn justified(flags: &[String], why: Option<&Why>) -> Result<()> {
    if why.is_some() {
        return Ok(());
    }
    if let Some(question) = answered(flags) {
        bail!(
            "these flags answer a question a fit left to the user ({question}): the choice must \
             be justified; pass why, with by (user, agent or agent_accepted) and cites"
        );
    }
    Ok(())
}

/// The question the words `flags` answer: a decision recorded in a run
/// manifest beside a file a flag names, one of whose options reruns with
/// that flag and file.
fn answered(flags: &[String]) -> Option<String> {
    let pairs = flags
        .iter()
        .enumerate()
        .filter_map(|(i, w)| match w.split_once('=') {
            Some((flag, value)) => Some((flag, value)),
            None => Some((w.as_str(), flags.get(i + 1)?.as_str())),
        });
    for (flag, value) in pairs.filter(|(f, _)| f.starts_with("--")) {
        let file = Path::new(value);
        if !file.is_file() {
            continue;
        }
        let Ok(entries) = std::fs::read_dir(run_manifest::manifest_dir(file)) else {
            continue;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if !path.to_string_lossy().ends_with(".senna.json") {
                continue;
            }
            let Ok((m, _)) = RunManifest::load(&path) else {
                continue;
            };
            let hit = m.decisions.iter().find(|d| {
                d.options.iter().any(|o| match o.rerun_args.as_slice() {
                    [f, v] => f == flag && run_manifest::same_file(Path::new(v), file),
                    _ => false,
                })
            });
            if let Some(d) = hit {
                return Some(d.question.clone());
            }
        }
    }
    None
}

#[cfg(test)]
#[path = "../tests/mcp_why.rs"]
mod tests;
