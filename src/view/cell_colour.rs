//! Colouring the cells by a number of each: a per-cell statistic of a
//! count source (over some of its rows, per the run's own totals), or a
//! value per cell from a table. The flags, the table, and the `u` cycle
//! through them.

use super::activity::{RowStat, Source, Stat};
use super::data::Dataset;
use super::{Levels, Pick, Scene};
use clap::Args;
use legume_numeric::matrix::table::TableReader;
use std::path::{Path, PathBuf};
use std::sync::Arc;

/// A per-cell statistic to start on.
#[derive(Clone, Copy, Debug, PartialEq, Eq, clap::ValueEnum)]
pub enum CellStat {
    /// `ln(1 + total count)`.
    Total,
    /// How many features have a count.
    Detected,
}

/// What the cells start coloured by, as a number of each.
#[derive(Args, Debug, Clone, Default)]
pub struct CellArgs {
    #[arg(
        long,
        value_enum,
        help = "Start coloured by a per-cell statistic of the counts shown (the extra file's, with --extra; `u` in the view)"
    )]
    pub cell_stat: Option<CellStat>,

    #[arg(
        long,
        value_name = "REGEX",
        requires = "cell_stat",
        help = "With --cell-stat: over the features whose names match REGEX only, as '/methylated$'"
    )]
    pub rows: Option<String>,

    #[arg(
        long,
        value_name = "SCALE",
        requires = "cell_stat",
        help = "With --cell-stat: divided by each cell's total count in the run's own data, times SCALE (1000: per 1k UMIs)"
    )]
    pub per_run_umi: Option<u32>,

    #[arg(
        long,
        value_name = "FILE[:COLUMN]",
        help = "Start coloured by a number per cell from a table (TSV, CSV or parquet): its first column names the cell (matched to the run's as --extra matches), COLUMN (default: the last numeric one) the value; `u` in the view"
    )]
    pub values: Option<String>,
}

impl CellArgs {
    /// The statistic the flags ask for: the plain total or count of
    /// features, else one over rows or per the run's totals.
    pub(crate) fn stat(&self) -> Result<Option<Stat>, String> {
        self.cell_stat
            .map(|m| cell_stat(m, self.rows.as_deref(), self.per_run_umi))
            .transpose()
    }
}

/// `measure` over the rows matching `rows`, per `per_run` of the run's own
/// counts: the plain total or count of features when neither is set.
pub(crate) fn cell_stat(
    measure: CellStat,
    rows: Option<&str>,
    per_run: Option<u32>,
) -> Result<Stat, String> {
    Ok(match (rows, per_run, measure) {
        (None, None, CellStat::Total) => Stat::Total,
        (None, None, CellStat::Detected) => Stat::Detected,
        _ => Stat::Rows(RowStat::new(measure, rows, per_run)?),
    })
}

/// `FILE[:COLUMN]`: the column after the last `:`, unless the whole is a
/// file.
#[must_use]
pub fn file_and_column(arg: &str) -> (PathBuf, Option<&str>) {
    match arg.rsplit_once(':') {
        Some((file, column)) if !Path::new(arg).exists() && !column.is_empty() => {
            (file.into(), Some(column))
        }
        _ => (arg.into(), None),
    }
}

/// A number per cell of the run, read from a table.
pub(crate) struct CellValues {
    /// The column and the file, as the key says them.
    pub title: String,
    pub by_cell: std::collections::HashMap<Box<str>, f32>,
    /// How many cells the run has.
    pub of: usize,
}

impl CellValues {
    /// `column` (else the last numeric column) of the table at `path`, its
    /// first column matched to `cells` as an extra file's cells are
    /// ([`senna::senna_input::match_cells`]). Missing values (empty, `NA`,
    /// `NaN`) are left out; a table with none of the cells is refused.
    pub(crate) fn read(
        path: &Path,
        column: Option<&str>,
        cells: &[Box<str>],
    ) -> anyhow::Result<Self> {
        let shown = path.to_string_lossy();
        let table = TableReader::open(&shown)?;
        let width = table.header().len();
        anyhow::ensure!(
            width >= 2,
            "{shown}: a cell column and a value column needed"
        );
        let rows: Vec<Vec<Box<str>>> = table
            .rows(&(0..width).collect::<Vec<_>>())?
            .collect::<anyhow::Result<_>>()?;
        let number = |v: &str| match v {
            "" | "NA" | "NaN" => Some(None),
            v => v.parse::<f32>().ok().map(Some),
        };
        let numeric = |c: usize| {
            rows.iter().all(|r| number(&r[c]).is_some())
                && rows.iter().any(|r| number(&r[c]).flatten().is_some())
        };
        let at = match column {
            Some(c) => table.select(&[c])?[0],
            None => (1..width)
                .rev()
                .find(|&c| numeric(c))
                .ok_or_else(|| anyhow::anyhow!("{shown}: no numeric column"))?,
        };
        anyhow::ensure!(
            numeric(at),
            "{shown}: column {} is not numeric",
            table.header()[at]
        );
        let names: Vec<Box<str>> = rows.iter().map(|r| r[0].clone()).collect();
        let by_cell: std::collections::HashMap<Box<str>, f32> =
            senna::senna_input::match_cells(cells, &names)
                .into_iter()
                .filter_map(|(cell, row)| Some((cell, number(&rows[row as usize][at])??)))
                .collect();
        let file = path
            .file_name()
            .map_or(shown.clone(), |f| f.to_string_lossy());
        anyhow::ensure!(
            !by_cell.is_empty(),
            "none of the run's {} cells have a value in {file}",
            cells.len()
        );
        Ok(Self {
            title: format!("{} · {file}", table.header()[at]),
            by_cell,
            of: cells.len(),
        })
    }

    /// How the table matched the run's cells.
    #[must_use]
    pub fn said(&self) -> String {
        format!(
            "values: {} · {} of the run's {} cells matched",
            self.title,
            self.by_cell.len(),
            self.of
        )
    }
}

/// What the flags and the agent set for the `u` cycle: a statistic over
/// rows (or per the run's totals), and a table's values per cell.
#[derive(Clone, Default)]
pub(crate) struct CellColour {
    pub stat: Option<RowStat>,
    pub values: Option<Arc<CellValues>>,
}

/// The run's cells, as its first cell map names them.
pub(crate) fn run_cells(data: &Dataset) -> &[Box<str>] {
    data.spaces
        .iter()
        .find(|s| s.axis() == super::data::Axis::Cells)
        .map_or(&[], |s| s.points.names.as_slice())
}

impl Scene {
    /// Start as the flags say: a per-cell statistic, or a table's values.
    pub(super) fn apply_cell_args(&mut self, args: &CellArgs) {
        match args.stat() {
            Ok(Some(stat)) => self.show_stat(stat),
            Ok(None) => {}
            Err(e) => log::warn!("view: {e}"),
        }
        if let Some(arg) = &args.values {
            let (file, column) = file_and_column(arg);
            match CellValues::read(&file, column, run_cells(&self.data)) {
                Ok(v) => self.show_values(Arc::new(v)),
                Err(e) => log::warn!("view: {e}"),
            }
        }
    }

    /// Colour the cells by `stat`, of the counts (never the model's
    /// level); one over rows or per the run's totals joins the `u` cycle.
    pub(crate) fn show_stat(&mut self, stat: Stat) {
        if let Stat::Rows(r) = &stat {
            self.cell_colour.stat = Some(r.clone());
        }
        if self.source == Source::Expected {
            self.source = self.count_source();
        }
        self.set_pick(Pick::Stat(stat));
    }

    /// Colour the cells by `values`, from now on in the `u` cycle too.
    pub(crate) fn show_values(&mut self, values: Arc<CellValues>) {
        self.note = Some(values.said());
        self.cell_colour.values = Some(values);
        self.set_pick(Pick::Values);
    }

    /// The table's value of every point of `space` on the ramp, unmatched
    /// points grey.
    pub(super) fn values_levels(&self, space: usize) -> Result<(Levels, String), String> {
        let v = self
            .cell_colour
            .values
            .as_ref()
            .ok_or("no values per cell are open (--values FILE[:COLUMN])")?;
        let values = self.data.spaces[space]
            .points
            .names
            .iter()
            .map(|n| v.by_cell.get(n).copied().unwrap_or(f32::NAN))
            .collect();
        Ok((Levels::from_values(values, false), v.title.clone()))
    }

    /// Colour the cells by the next per-cell number: the counts' total,
    /// features detected, the focused group's markers' share (when it has
    /// markers), the statistic the flags set, stage 2's exclusivity (a run
    /// with one), a table's values (when one is open), then back to group
    /// colours. The model's expectation has no counts, so the counts come
    /// up in its place.
    pub fn cycle_stat(&mut self) {
        // The share shown stays in the ring, its group unfocused since.
        let shown_share = matches!(self.pick, Some(Pick::Stat(Stat::Share(_))));
        let share = self
            .focused_name()
            .filter(|g| !self.marker_features(g).is_empty())
            .map(|g| Pick::Stat(Stat::Share(g)))
            .or_else(|| self.pick.clone().filter(|_| shown_share));
        let ring: Vec<Pick> = [
            Some(Pick::Stat(Stat::Total)),
            Some(Pick::Stat(Stat::Detected)),
            share,
            self.cell_colour
                .stat
                .clone()
                .map(|r| Pick::Stat(Stat::Rows(r))),
            self.has_stage2().then_some(Pick::Exclusive),
            self.cell_colour.values.is_some().then_some(Pick::Values),
        ]
        .into_iter()
        .flatten()
        .collect();
        // A share of another group's markers is where this group's is.
        let same = |p: &Pick| {
            Some(p) == self.pick.as_ref()
                || matches!(
                    (p, &self.pick),
                    (Pick::Stat(Stat::Share(_)), Some(Pick::Stat(Stat::Share(_))))
                )
        };
        let next = match ring.iter().position(same) {
            Some(at) => ring.get(at + 1).cloned(),
            None => ring.first().cloned(),
        };
        match next {
            Some(Pick::Stat(stat)) => self.show_stat(stat),
            Some(pick) => self.set_pick(pick),
            None => {
                self.clear_pick();
            }
        }
    }
}

#[cfg(test)]
#[path = "tests/cell_colour.rs"]
mod tests;
