//! senna's command lines as the run tools set them: each tool takes its
//! own arguments (inputs, out, what it resolves or checks itself) by name,
//! and every other flag of its command from a `flags` map, read off senna's
//! clap command line so the tools never drift from the CLI. `run_help`
//! lists a command's flags from the same place.

use anyhow::{Context, Result};
use clap::parser::ValueSource;
use clap::ArgMatches;
use legume_mcp::cli::{check, describe, settable, to_argv};
use serde_json::{json, Map, Value};
use std::sync::OnceLock;

use crate::run::tui::METHODS;

/// The tool that lists the flags of every command a run tool starts.
pub(super) const LISTER: &str = "run_help";

/// senna's command line, built once, its global flags (`-v`, `--threads`)
/// in every command.
fn built() -> &'static clap::Command {
    static BUILT: OnceLock<clap::Command> = OnceLock::new();
    BUILT.get_or_init(|| {
        let mut cli = <crate::Cli as clap::CommandFactory>::command();
        cli.build();
        cli
    })
}

/// senna's command `path` (`predict`, or `layout umap` for a nested one)
/// as senna's command line builds it.
pub(super) fn command(path: &str) -> Result<&'static clap::Command> {
    let words: Vec<&str> = path.split(' ').collect();
    crate::tui::subcommand(built(), &words).with_context(|| format!("senna has no command {path}"))
}

/// A run tool's own arguments, by clap id, each with the name the tool
/// gives it: never set through `flags`.
type Own = &'static [(&'static str, &'static str)];

/// A fit's own arguments.
const DATA_OUT: Own = &[("data_files", "data"), ("out", "out")];

/// A layout's own arguments.
const LAYOUT: Own = &[("from", "run"), ("out", "run"), ("target", "target")];

/// The commands the run tools start besides the fits and the layouts, in
/// the order `run_help` lists them, with their own arguments.
const OTHERS: &[(&str, Own)] = &[
    (
        "predict",
        &[
            ("data_files", "data"),
            ("bulk", "bulk"),
            ("model", "model"),
            ("out", "out"),
            ("decoder_only", "latent"),
        ],
    ),
    (
        "probe",
        &[
            ("data_files", "data"),
            ("model", "model"),
            ("calibration", "calibration"),
            ("out", "out"),
        ],
    ),
    (
        "update",
        &[("data_files", "data"), ("model", "model"), ("out", "out")],
    ),
    ("critique", &[("runs", "runs"), ("out", "out")]),
    (
        "impute",
        &[("data_files", "data"), ("model", "model"), ("out", "out")],
    ),
    (
        "deconvolve",
        &[("from", "from"), ("bulk", "bulk"), ("out", "out")],
    ),
    ("donor-deviation", &[("from", "from"), ("out", "out")]),
    (
        "clustering",
        &[("from", "from"), ("out", "out"), ("target", "target")],
    ),
    (
        "resolve-embedding-space",
        &[("from", "from"), ("out", "out")],
    ),
    ("joint-topic", DATA_OUT),
    ("joint-svd", DATA_OUT),
    ("fne", &[("networks", "data"), ("out", "out")]),
];

/// The layout methods a run tool starts.
fn layouts() -> &'static [&'static str] {
    crate::view::recompute::Step::CellLayout.settings()
}

/// The arguments of `path` its run tool takes itself; refused when no run
/// tool starts `path`.
pub(super) fn own(path: &str) -> Result<Own> {
    if METHODS.contains(&path) {
        return Ok(DATA_OUT);
    }
    if path
        .strip_prefix("layout ")
        .is_some_and(|m| layouts().contains(&m))
    {
        return Ok(LAYOUT);
    }
    OTHERS
        .iter()
        .find(|(p, _)| *p == path)
        .map(|(_, own)| *own)
        .with_context(|| format!("no run tool starts senna {path}: {LISTER} lists those that do"))
}

/// `flags` (flag name → value) as the words of `path`'s command line,
/// refused when one is not `path`'s, is positional, or is the tool's own.
pub(super) fn words(path: &str, flags: &Map<String, Value>) -> Result<Vec<String>> {
    to_argv(command(path)?, flags, own(path)?, LISTER)
}

/// The flags of `path` an agent sets through `flags`: the visible ones,
/// without the positionals and the tool's own.
pub(super) fn listed(path: &str) -> Result<Value> {
    let cmd = command(path)?;
    let own = own(path)?;
    let flags: Vec<Value> = settable(cmd)
        .filter(|a| {
            !a.is_hide_set() && !a.is_positional() && !own.iter().any(|(id, _)| a.get_id() == *id)
        })
        .map(describe)
        .collect();
    let mut names: Vec<&str> = own.iter().map(|(_, name)| *name).collect();
    names.dedup();
    Ok(json!({
        "command": format!("senna {path}"),
        "about": cmd.get_long_about().or(cmd.get_about()).map(ToString::to_string),
        "own": names,
        "flags": flags,
    }))
}

/// The `flags` property of a run tool's schema, for the command(s) `of`.
pub(super) fn property(of: &str, example: &str) -> Value {
    json!({
        "type": "object",
        "description": format!(
            "Every other flag of {of}, by name, as run_help lists them (a switch takes true or \
             false, a list gives several values, null leaves a flag at its default): {example}. \
             Checked as senna parses it before anything runs"
        ),
    })
}

/// `argv` parsed as senna parses it: the matches of the (innermost)
/// command it runs.
pub(super) fn parsed(argv: &[String]) -> Result<ArgMatches> {
    let all = check(built(), argv)?;
    let mut m = &all;
    while let Some((_, sub)) = m.subcommand() {
        m = sub;
    }
    Ok(m.clone())
}

/// The values the command line gave the argument `id` (none when it is
/// left at its default), as written.
pub(super) fn given(m: &ArgMatches, id: &str) -> Vec<String> {
    if m.value_source(id) != Some(ValueSource::CommandLine) {
        return Vec::new();
    }
    m.try_get_raw(id)
        .ok()
        .flatten()
        .into_iter()
        .flatten()
        .map(|v| v.to_string_lossy().into_owned())
        .collect()
}

/// Every command a run tool starts and `run_help` lists, as senna names
/// it: the fits, those of the other run tools, and the layouts.
pub(super) fn helped() -> Vec<String> {
    let others = OTHERS.iter().map(|(p, _)| p.to_string());
    let layouts = layouts().iter().map(|m| format!("layout {m}"));
    METHODS
        .iter()
        .map(|m| m.to_string())
        .chain(others)
        .chain(layouts)
        .collect()
}

#[cfg(test)]
#[path = "../tests/mcp_flags.rs"]
mod tests;
