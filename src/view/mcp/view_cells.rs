//! `view_cells`: the cells of some groups of the open run written as a
//! list, one name a line, for a run tool's flag (an ODE's `root-cells`).

use super::view::{group_name, group_of, grouping_named, Open};
use crate::view::data::{Axis, Labels};
use anyhow::{anyhow, ensure, Context, Result};
use legume_mcp::{Call, Content, ToolSpec};
use senna::run_manifest;
use serde::Deserialize;
use serde_json::{json, Value};
use std::path::{Path, PathBuf};

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct CellsArgs {
    groups: Vec<Value>,
    colour_by: Option<String>,
    out: Option<PathBuf>,
}

/// The cells of some groups of one grouping, on the map shown.
#[derive(Debug, PartialEq)]
pub(super) struct Picked {
    /// The grouping's title.
    pub(super) grouping: String,
    /// Each group asked for, by its name, and its cells on the map.
    pub(super) groups: Vec<(String, usize)>,
    /// The cells, in the map's order.
    pub(super) names: Vec<Box<str>>,
}

impl Open {
    pub(super) fn cells(&self, a: CellsArgs, cx: &Call) -> Result<Vec<Content>> {
        let groups: Vec<String> = a.groups.iter().map(group_name).collect::<Result<_>>()?;
        let picked = self.pick_cells(&groups, a.colour_by.as_deref())?;
        let path = match &a.out {
            Some(out) => {
                ensure!(!out.exists(), "{} exists; pass another out", out.display());
                out.clone()
            }
            None => self.free_list(&picked, cx)?,
        };
        let mut text = picked.names.join("\n");
        text.push('\n');
        cx.write_file(&path, &text)?;
        // Where the list came from, for a fit given it to say what it is.
        let source = self.source(&picked);
        cx.write_file(&source_path(&path), &serde_json::to_string_pretty(&source)?)?;
        Ok(vec![Content::Json(json!({
            "cells": super::absolute(&path),
            "grouping": source["grouping"],
            "groups": source["groups"],
            "total": source["total"],
            "use": "a list of cell names, one a line, for a run tool's flags: root-cells of \
                    run_fit ode (the trajectory's root), drop-cells of run_layout, \
                    archetype-cells of run_deconvolve",
        }))])
    }

    /// The cells of `groups` of the grouping shown, or of `colour_by`'s,
    /// on the map of cells shown.
    pub(super) fn pick_cells(&self, groups: &[String], colour_by: Option<&str>) -> Result<Picked> {
        ensure!(!groups.is_empty(), "name the groups whose cells to write");
        let s = &self.scene;
        ensure!(
            s.current().axis() == Axis::Cells,
            "the cells of a map of cells: view_map space cells first"
        );
        let labels = self.grouping(colour_by)?;
        let mut asked: Vec<u32> = Vec::with_capacity(groups.len());
        for name in groups {
            let g = group_of(labels, name).map_err(|e| {
                let levels: Vec<&str> = labels.levels.iter().map(AsRef::as_ref).collect();
                anyhow!("{e}; it has {}", levels.join(", "))
            })?;
            if !asked.contains(&g) {
                asked.push(g);
            }
        }
        let mut counts = vec![0usize; asked.len()];
        let mut names = Vec::new();
        for name in &s.current().points.names {
            let Some(g) = labels.by_name.get(name) else {
                continue;
            };
            if let Some(k) = asked.iter().position(|a| a == g) {
                counts[k] += 1;
                names.push(name.clone());
            }
        }
        ensure!(!names.is_empty(), "no cell of those groups is on this map");
        Ok(Picked {
            grouping: labels.title().to_string(),
            groups: asked
                .iter()
                .zip(counts)
                .map(|(&g, n)| (labels.levels[g as usize].to_string(), n))
                .collect(),
            names,
        })
    }

    /// Where `picked` came from: the run and the round shown, the
    /// grouping, each group's count.
    pub(super) fn source(&self, picked: &Picked) -> Value {
        let groups: Vec<Value> = picked
            .groups
            .iter()
            .map(|(g, n)| json!({"group": g, "cells": n}))
            .collect();
        let round = self
            .scene
            .data
            .round
            .as_ref()
            .map(|r| super::absolute(&r.path));
        json!({
            "run": self.run,
            "round": round,
            "grouping": picked.grouping,
            "groups": groups,
            "total": picked.names.len(),
        })
    }

    /// The grouping of the map's cells named `colour_by`, else the one
    /// shown.
    fn grouping(&self, colour_by: Option<&str>) -> Result<&Labels> {
        let s = &self.scene;
        let li = match colour_by {
            Some(by) => grouping_named(&s.data, &s.colour_choices(), by)?,
            None => s.colour.context("no grouping shown: pass colour_by")?,
        };
        Ok(&s.data.labels[li])
    }

    /// A list's name in today's results folder, by the run and the groups,
    /// that no file has.
    fn free_list(&self, picked: &Picked, cx: &Call) -> Result<PathBuf> {
        let run = run_manifest::derive_out_prefix(&self.run.to_string_lossy());
        let run = Path::new(&run).file_name().unwrap_or_default();
        let groups: Vec<&str> = picked.groups.iter().map(|(g, _)| g.as_str()).collect();
        let base: String = format!("{}.cells_{}", run.to_string_lossy(), groups.join("+"))
            .chars()
            .map(|c| match c {
                c if c.is_alphanumeric() || "+-_.".contains(c) => c,
                _ => '_',
            })
            .take(100)
            .collect();
        let list = |out: &str| PathBuf::from(format!("{out}.txt"));
        let (out, _) = cx
            .free_prefix(&base, |out| Ok(vec![list(out)]), |_| false)
            .context("no results folder to write in: pass out")?;
        Ok(list(&out))
    }
}

/// `{list}.source.json`, beside a list view_cells writes: the run and
/// round it was read from, the grouping, and each group's count.
pub(super) fn source_path(list: &Path) -> PathBuf {
    let mut name = list.as_os_str().to_owned();
    name.push(".source.json");
    PathBuf::from(name)
}

/// What the `n` cells of the list at `list` are, by the groups view_cells
/// wrote it from ([`source_path`]), as a round's types say it
/// ([`crate::view::mostly`]), and the grouping and round (or run) they are
/// of; `None` when no source is beside it or it counts other cells than
/// the list holds (the list changed since).
pub(super) fn source_of(list: &Path, n: usize) -> Option<(String, Value)> {
    let source: Value = crate::view::files::read_json(&source_path(list))?;
    if source["total"].as_u64() != Some(n as u64) {
        return None;
    }
    let groups: Vec<(&str, usize)> = source["groups"]
        .as_array()?
        .iter()
        .filter_map(|g| Some((g["group"].as_str()?, g["cells"].as_u64()? as usize)))
        .collect();
    let mostly = crate::view::mostly(groups, n)?;
    let of = if source["round"].is_string() {
        &source["round"]
    } else {
        &source["run"]
    };
    let by = json!({"grouping": source["grouping"], "of": of});
    Some((mostly, by))
}

pub(super) fn spec() -> ToolSpec {
    ToolSpec::new(
        "view_cells",
        "Write the cells of some groups of the open run, one name a line, for a run tool \
         that takes a list of cells: an ODE's root (run_fit ode, flag root-cells), cells to \
         leave out of a layout (run_layout, drop-cells), or archetypes (run_deconvolve, \
         archetype-cells). The groups are of the grouping view_map shows (colour_by): with a \
         lupin round shown (view_map round, or view_open on a round), its labels are the \
         cell types, so a type's name gives its cells. The cells are those of the map of \
         cells shown. Returns the list's path, each group's count, the total and the \
         grouping; a group the grouping lacks is an error that lists those it has.",
        json!({
            "type": "object",
            "properties": {
                "groups": {"type": "array", "items": {"type": ["string", "integer"]}, "minItems": 1, "description": "Group names, or cluster ids (3, C3, K3), of the grouping"},
                "colour_by": {"type": "string", "description": "Another grouping of cells, as view_open lists them, for this call only [default: the one view_map shows]"},
                "out": {"type": "string", "description": "The list's path; never replaced [default: today's results folder, by the run and the groups]"},
            },
            "required": ["groups"],
            "additionalProperties": false,
        }),
    )
}

#[cfg(test)]
#[path = "../tests/mcp_view_cells.rs"]
mod tests;
