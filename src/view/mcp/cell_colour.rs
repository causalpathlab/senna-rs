//! view_map's numbers per cell: an extra data file, a per-cell statistic
//! of the counts (over some rows, per the run's own totals), and a table's
//! values per cell.

use super::view::MapArgs;
use crate::view::activity::{read_extra, ExtraFiles, Stat};
use crate::view::cell_colour::{cell_stat, run_cells, CellValues};
use crate::view::data::Dataset;
use crate::view::{CellStat, Pick, Scene};
use anyhow::{ensure, Result};
use serde::Deserialize;
use serde_json::{json, Value};
use std::path::PathBuf;
use std::sync::Arc;

/// A per-cell statistic of the counts shown.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(in crate::view) struct CellStatArg {
    stat: StatArg,
    rows: Option<String>,
    per_run_umi: Option<u32>,
}

#[derive(Deserialize, Clone, Copy, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
enum StatArg {
    Total,
    Detected,
    Off,
}

/// A number per cell from a table.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(in crate::view) struct ValuesArg {
    file: PathBuf,
    column: Option<String>,
}

/// What the arguments ask for, read and checked before the view changes.
pub(super) struct CellColouring {
    extra: Option<Arc<ExtraFiles>>,
    /// `Some(None)`: off.
    stat: Option<Option<Stat>>,
    values: Option<Arc<CellValues>>,
}

/// Read and check `a`'s extra file, statistic and table against `data`,
/// the run as it will be shown.
pub(super) fn check(a: &MapArgs, data: &Dataset) -> Result<CellColouring> {
    let colouring =
        a.cell_stat.as_ref().is_some_and(|c| c.stat != StatArg::Off) || a.values.is_some();
    ensure!(
        !colouring || (a.feature.as_deref().is_none_or(str::is_empty) && a.time != Some(true)),
        "colour by a feature, the time, a cell statistic or values: one of them"
    );
    let cells = run_cells(data);
    let extra = match a.extra.as_deref() {
        Some(paths) => Some(Arc::new(ExtraFiles::ready(Ok(read_extra(paths, cells)?)))),
        None => None,
    };
    let stat = a.cell_stat.as_ref().map(CellStatArg::stat).transpose()?;
    ensure!(
        !(matches!(stat, Some(Some(_))) && a.values.is_some()),
        "colour by a cell statistic or by values, not both"
    );
    Ok(CellColouring {
        extra,
        stat,
        values: values_of(a, cells)?,
    })
}

impl CellStatArg {
    /// The statistic asked for; `None`: off.
    fn stat(&self) -> Result<Option<Stat>> {
        let measure = match self.stat {
            StatArg::Total => CellStat::Total,
            StatArg::Detected => CellStat::Detected,
            StatArg::Off => {
                ensure!(
                    self.rows.is_none() && self.per_run_umi.is_none(),
                    "cell_stat off takes no rows or per_run_umi"
                );
                return Ok(None);
            }
        };
        cell_stat(measure, self.rows.as_deref(), self.per_run_umi)
            .map(Some)
            .map_err(anyhow::Error::msg)
    }
}

fn values_of(a: &MapArgs, cells: &[Box<str>]) -> Result<Option<Arc<CellValues>>> {
    a.values
        .as_ref()
        .map(|v| CellValues::read(&v.file, v.column.as_deref(), cells).map(Arc::new))
        .transpose()
}

impl CellColouring {
    /// Open the extra file: before the source and feature apply, so they
    /// may be the extra file's.
    pub(super) fn open_extra(&mut self, scene: &mut Scene) {
        if let Some(extra) = self.extra.take() {
            scene.show_extra(extra);
        }
    }

    /// Colour the cells as asked, after the rest applies.
    pub(super) fn apply(self, scene: &mut Scene) {
        match self.stat {
            Some(Some(stat)) => scene.show_stat(stat),
            Some(None) if matches!(scene.pick, Some(Pick::Stat(_) | Pick::Values)) => {
                scene.clear_pick();
            }
            _ => {}
        }
        if let Some(values) = self.values {
            scene.show_values(values);
        }
    }
}

/// The colouring drawn when it is a number per cell: its key, and how
/// many of the points in view have a value.
pub(super) fn state(s: &Scene) -> Value {
    match (&s.pick, s.current_shown()) {
        (Some(Pick::Stat(_) | Pick::Values), Some(shown)) => json!({
            "key": shown.title,
            "source": matches!(s.pick, Some(Pick::Stat(_))).then(|| s.source.name()),
            "with_values": shown.levels.values.iter().filter(|v| v.is_finite()).count(),
            "points": shown.levels.values.len(),
        }),
        _ => Value::Null,
    }
}

/// view_map's property `name`, one of these arguments.
pub(super) fn spec(name: &str) -> Value {
    match name {
        "extra" => {
            json!({"type": "array", "items": {"type": "string"}, "description": "Data files the run was not fitted on (.zarr, .zarr.zip, .h5), read together, their cells matched to the run's by name or barcode: the extra source, its feature levels and cell statistics (observed: true shows them)"})
        }
        "cell_stat" => json!({"type": "object", "properties": {
                "stat": {"enum": ["total", "detected", "off"], "description": "Each cell's total count (log1p, or the raw count with per_run_umi) or features detected, of the counts shown (the extra file's when one is open); off shows the groups again"},
                "rows": {"type": "string", "description": "Over the features whose names match this regular expression only, as \"/methylated$\""},
                "per_run_umi": {"type": "integer", "minimum": 1, "description": "Divided by each cell's total count in the run's own data, times this: 1000 is per 1k UMIs"}
            }, "required": ["stat"], "additionalProperties": false, "description": "Colour the cells by a per-cell statistic of the counts; the reply's cell_colour gives its key and how many points have a value"}),
        "values" => json!({"type": "object", "properties": {
                "file": {"type": "string", "description": "A table (TSV, CSV or parquet) whose first column names the cell, matched to the run's cells by name or barcode"},
                "column": {"type": "string", "description": "The value column [default: the last numeric one]"}
            }, "required": ["file"], "additionalProperties": false, "description": "Colour the cells by a number each from a table, unmatched cells grey; the note says how many cells matched; feature \"\" shows the groups again"}),
        _ => unreachable!("no view_map argument {name} here"),
    }
}
