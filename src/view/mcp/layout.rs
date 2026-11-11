//! `run_layout`: a run laid out again with the settings given.

use super::jobs::{one_of, senna_parses, Task};
use super::parse;
use crate::postprocess::{MIN_DIST, SPREAD};
use crate::view::recompute::{Step, Target};
use anyhow::{ensure, Context, Result};
use legume_mcp::{Call, Content, ToolSpec};
use serde::Deserialize;
use serde_json::{json, Value};

/// A layout tool's call; `None` for another family's.
pub(super) fn call(name: &str, args: &Value, cx: &Call) -> Result<Option<Vec<Content>>> {
    Ok(Some(match name {
        "run_layout" => run_layout(parse(args.clone())?, cx)?,
        _ => return Ok(None),
    }))
}

#[derive(Deserialize, Clone, Copy, Default)]
#[serde(rename_all = "lowercase")]
enum LayoutTarget {
    #[default]
    Cells,
    Features,
}

impl LayoutTarget {
    fn step(self) -> Step {
        match self {
            Self::Cells => Step::CellLayout,
            Self::Features => Step::FeatureLayout,
        }
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct LayoutArgs {
    run: String,
    method: String,
    #[serde(default)]
    target: LayoutTarget,
    min_dist: Option<f32>,
    spread: Option<f32>,
    #[serde(default)]
    args: Vec<String>,
}

/// `senna layout {method} …` as `view_open` lays out the run `t`, with
/// the flags given.
fn layout_task(a: &LayoutArgs, t: &Target) -> Result<Task> {
    let step = a.target.step();
    one_of(&a.method, step.settings())
        .with_context(|| format!("senna's {} methods", step.label()))?;
    let mut argv = step.argv(t, &a.method);
    for (flag, v) in [("--umap-min-dist", a.min_dist), ("--umap-spread", a.spread)] {
        if let Some(v) = v {
            argv.extend([flag.into(), v.to_string()]);
        }
    }
    argv.extend(a.args.iter().cloned());
    // The kernel's range too, which clap leaves to the layout.
    if let crate::Commands::Layout {
        cmd: crate::LayoutCmd::Umap(umap),
    } = senna_parses(&argv)?.commands
    {
        umap.tumap_params()?;
    }
    let made = step.outputs(t, &a.method).remove(0);
    Ok(Task {
        argv,
        made: vec![("coords", made.into())],
        summary: format!("senna layout {} of {}", a.method, a.run),
    })
}

/// The run laid out again with the flags given, replacing that method's
/// layout. Always run, as the flags are the point, and recorded with no
/// outputs, so without a guard: a replay redoes it after any earlier
/// layout of the method. Its file is held and checked here instead, and
/// gets no row in results/INDEX.tsv (legume-mcp records no step that
/// replaces its outputs).
fn run_layout(a: LayoutArgs, cx: &Call) -> Result<Vec<Content>> {
    let task = layout_task(&a, &Target::load(&a.run)?)?;
    let outputs = task.outputs();
    cx.ensure_free(&outputs)?;
    task.run(false, cx)?;
    ensure!(
        outputs[0].exists(),
        "senna layout ended without {}",
        outputs[0].display()
    );
    let mut done = task.named();
    done["ran"] = json!(true);
    done["note"] = json!(format!("view_open the run with map {} to see it", a.method));
    Ok(vec![Content::Json(done)])
}

pub(super) fn specs() -> Vec<ToolSpec> {
    let spec = ToolSpec::new;
    vec![spec(
        "run_layout",
        "Lay out a run again with the settings given, as `senna layout {method} --from \
             {run} {args}`, replacing that method's layout in the run (e.g. umap with \
             min_dist 0.5 and [\"--knn\", \"30\"]; phate or tsne with \
             [\"--knn\", \"30\", \"--n-landmarks\", \"300\"]). Always runs. Then \
             view_open the run with that map to see it.",
        json!({
            "type": "object",
            "properties": {
                "run": {"type": "string", "description": "Run manifest ({out}.senna.json)"},
                "method": {"enum": Step::CellLayout.settings(), "description": "Layout method; the feature embedding takes umap or phate"},
                "target": {"enum": ["cells", "features"], "description": "Lay out the cells, or the feature embedding on its own [default: cells]"},
                "min_dist": {"type": "number", "minimum": 0, "description": format!("umap only: umap-learn's min_dist, in [0, spread] [default {MIN_DIST} when spread is given]; larger loosens groups into continua, smaller packs them. With neither this nor spread, t-UMAP")},
                "spread": {"type": "number", "exclusiveMinimum": 0, "description": format!("umap only: umap-learn's spread, the scale of the map's distances [default {SPREAD} when min_dist is given]")},
                "args": {"type": "array", "items": {"type": "string"}, "description": "Further flags, as run_help (layout {method}) lists them"},
            },
            "required": ["run", "method"],
            "additionalProperties": false,
        }),
    )]
}

#[cfg(test)]
#[path = "../tests/mcp_layout.rs"]
mod tests;
