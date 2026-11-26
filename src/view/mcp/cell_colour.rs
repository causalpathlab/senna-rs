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
    per_rows: Option<String>,
}

#[derive(Deserialize, Clone, Copy, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
enum StatArg {
    Total,
    Detected,
    Mean,
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
/// What is open already is not read again: a watcher applies the agent's
/// whole view each time it changes.
pub(super) fn check(a: &MapArgs, s: &Scene, data: &Dataset) -> Result<CellColouring> {
    if let Some(cs) = &a.colour_scale {
        cs.check()
            .map_err(|e| anyhow::anyhow!("colour_scale: {e}"))?;
    }
    let colouring =
        a.cell_stat.as_ref().is_some_and(|c| c.stat != StatArg::Off) || a.values.is_some();
    ensure!(
        !colouring || (a.feature.as_deref().is_none_or(str::is_empty) && a.time != Some(true)),
        "colour by a feature, the time, a cell statistic or values: one of them"
    );
    let cells = run_cells(data);
    let open = s
        .data
        .loads
        .extra
        .as_ref()
        .and_then(|x| x.try_get()?.as_ref().ok());
    let extra = match a.extra.as_deref() {
        Some(paths) if open.is_some_and(|x| x.paths == paths) => None,
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
        values: values_of(a, s, cells)?,
    })
}

impl CellStatArg {
    /// The statistic asked for; `None`: off.
    fn stat(&self) -> Result<Option<Stat>> {
        let measure = match self.stat {
            StatArg::Total => CellStat::Total,
            StatArg::Detected => CellStat::Detected,
            StatArg::Mean => CellStat::Mean,
            StatArg::Off => {
                ensure!(
                    self.rows.is_none() && self.per_run_umi.is_none() && self.per_rows.is_none(),
                    "cell_stat off takes no rows, per_run_umi or per_rows"
                );
                return Ok(None);
            }
        };
        cell_stat(
            measure,
            self.rows.as_deref(),
            self.per_run_umi,
            self.per_rows.as_deref(),
        )
        .map(Some)
        .map_err(anyhow::Error::msg)
    }
}

fn values_of(a: &MapArgs, s: &Scene, cells: &[Box<str>]) -> Result<Option<Arc<CellValues>>> {
    let Some(v) = &a.values else {
        return Ok(None);
    };
    let open = s
        .cell_colour
        .values
        .as_ref()
        .filter(|o| o.file == v.file && v.column.as_deref().is_none_or(|c| *c == *o.column));
    match open {
        Some(o) => Ok(Some(o.clone())),
        None => Ok(Some(Arc::new(CellValues::read(
            &v.file,
            v.column.as_deref(),
            cells,
        )?))),
    }
}

/// `state`'s (a published view's) extra files, cell statistic and table,
/// as view_map takes them.
pub(super) fn publish(s: &Scene, state: &mut Value) {
    let extra = s
        .data
        .loads
        .extra
        .as_ref()
        .and_then(|x| x.try_get()?.as_ref().ok());
    state["extra"] = json!(extra.map(|x| &x.paths));
    let measure = |m: CellStat| match m {
        CellStat::Total => "total",
        CellStat::Detected => "detected",
        CellStat::Mean => "mean",
    };
    state["cell_stat"] = match &s.pick {
        Some(Pick::Stat(Stat::Total)) => json!({"stat": "total"}),
        Some(Pick::Stat(Stat::Detected)) => json!({"stat": "detected"}),
        Some(Pick::Stat(Stat::Rows(r))) => {
            let mut v =
                json!({"stat": measure(r.measure), "rows": r.rows, "per_run_umi": r.per_run});
            if let Some(p) = &r.per_rows {
                v["per_rows"] = json!(p);
            }
            v
        }
        _ => Value::Null,
    };
    state["colour_scale"] = json!(s.colour_scale);
    state["outliers"] = json!(s.outliers);
    state["values"] = match (&s.pick, &s.cell_colour.values) {
        (Some(Pick::Values), Some(v)) => json!({"file": v.file, "column": v.column}),
        _ => Value::Null,
    };
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
            "with_values": (0..shown.levels.values.len()).filter(|&i| shown.levels.t(i) != crate::view::activity::MISSING).count(),
            "range": shown.levels.range(),
            "scale": shown.levels.transform.name(),
            "diverging": shown.levels.diverging,
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
                "stat": {"enum": ["total", "detected", "mean", "off"], "description": "Each cell's total count (on a log1p scale unless colour_scale says), features detected, or mean count over the rows, of the counts shown (the extra file's when one is open); off shows the groups again"},
                "rows": {"type": "string", "description": "Over the features whose names match this regular expression only (case-insensitive, as data-beans' row_pattern), as \"/methylated$\""},
                "per_run_umi": {"type": "integer", "minimum": 1, "description": "Divided by each cell's total count in the run's own data, times this: 1000 is per 1k UMIs"},
                "per_rows": {"type": "string", "description": "Divided by the same statistic over the features matching this regular expression: rows \"/methylated$\" with per_rows \"/(un)?methylated$\" is each cell's methylated fraction; not with per_run_umi"}
            }, "required": ["stat"], "additionalProperties": false, "description": "Colour the cells by a per-cell statistic of the counts; the reply's cell_colour gives its key and how many points have a value"}),
        "values" => json!({"type": "object", "properties": {
                "file": {"type": "string", "description": "A table (TSV, CSV or parquet) whose first column names the cell, matched to the run's cells by name or barcode"},
                "column": {"type": "string", "description": "The value column [default: the last numeric one]"}
            }, "required": ["file"], "additionalProperties": false, "description": "Colour the cells by a number each from a table, unmatched cells grey; the note says how many cells matched; feature \"\" shows the groups again"}),
        "colour_scale" => colour_scale_spec(),
        "outliers" => json!({"type": "object", "properties": {
                "mode": {"enum": ["off", "fit", "hide"], "description": "fit: the camera (and an export) fits the rest and outliers are drawn pinned to its edge; hide: they are not drawn either; off: every point counts"},
                "rule": {"enum": ["isolated", "range"], "description": "isolated: the distance to the k-th nearest neighbour in the layout far above the rest (strays; tight small clusters kept); range: outside a quantile range of x or y (spectral spikes on a few cells) [default: range on diffusion, LLE and PCA maps, else isolated]"},
                "k": {"type": "integer", "minimum": 1, "maximum": 100, "description": "isolated's neighbour [default: 10]"},
                "mad": {"type": "number", "exclusiveMinimum": 0, "description": "Out past median + mad × MAD (of the distance, or of x and y for range) [default: 5]"},
                "quantile": {"type": "number", "exclusiveMinimum": 0.5, "exclusiveMaximum": 1, "description": "Out past this quantile instead, as 0.995 (range: outside 0.5%-99.5% of x or y) [range's default: 0.995]"}
            }, "required": ["mode"], "additionalProperties": false, "description": "Outlying points of the layout, so a few strays or a spectral spike do not shrink the map; kept until changed (O in the viewer). The reply's outliers says the rule, how many points and the bounds fitted"}),
        _ => unreachable!("no view_map argument {name} here"),
    }
}

/// view_map's `colour_scale`: how every number per cell is coloured.
fn colour_scale_spec() -> Value {
    use legume_tui::map::palette::Palette;
    let palettes: Vec<&str> = Palette::ALL.iter().map(|p| p.name()).collect();
    json!({"type": "object", "properties": {
            "scale": {"enum": ["linear", "log10", "log1p", "sqrt"], "description": "The scale values are put on before the ramp spans them [default: log1p for total counts, else linear]; the key's ticks stay in the values' own units"},
            "clip": {"type": "array", "items": {"type": "number", "minimum": 0, "maximum": 100}, "minItems": 2, "maxItems": 2, "description": "The ramp spans these percentiles of the values, [low, high] as [1, 99] [default: [2, 99], counts from 0]; values beyond are drawn at the ends"},
            "min": {"type": "number", "description": "The ramp's low end in the values' units, overriding clip's low; pass the reply's cell_colour.range to another map so panels share one range"},
            "max": {"type": "number", "description": "The ramp's high end in the values' units, overriding clip's high"},
            "palette": {"enum": palettes, "description": "The ramp [default: activity, or blue_red for signed scores]; blue_red and purple_green are diverging, centred at midpoint"},
            "reverse": {"type": "boolean", "description": "Run the palette high to low"},
            "midpoint": {"type": "number", "description": "The centre of a diverging ramp, in the values' units [default: 0]; given alone it asks for blue_red. The range is symmetric about it"},
            "missing": {"enum": ["grey", "hidden"], "description": "Cells with no value (unmatched, NaN): muted grey under the rest, or not drawn [default: grey]"},
            "order": {"enum": ["high", "low", "random"], "description": "Which points are drawn on top: high values (diverging: the largest departures), low values, or the map's fixed random order [default: high]"},
            "key": {"type": "object", "properties": {
                "place": {"enum": ["top_left", "top_right", "bottom_left", "bottom_right", "right", "below"], "description": "Where the key goes [default: top_left]; in a view_export panel right and below put it outside the map"},
                "length": {"type": "number", "exclusiveMinimum": 0, "description": "The bar's length in line heights [default: 8]"},
                "title": {"type": "string", "description": "The key's title [default: what is coloured]"},
                "hidden": {"type": "boolean", "description": "No key"}
            }, "additionalProperties": false, "description": "The key: the ramp as a bar, ticks labelled with round values and the ends, and an NA swatch when some cells have no value"}
        }, "additionalProperties": false,
        "description": "How every number per cell is coloured (feature, time, cell_stat, values, stage 2), from now on until changed: a whole object each time, {} for the defaults. The reply's cell_colour gives the range, scale and key drawn"})
}
