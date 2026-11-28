//! Colouring the cells by a number of each: a per-cell statistic of a
//! count source (over some of its rows, per the run's own totals), or a
//! value per cell from a table. The flags, the table, and the `u` cycle
//! through them.

use super::activity::{RowStat, Source, Stat};
use super::data::Dataset;
use super::group_table::GroupTable;
use super::{Levels, Pick, Scene};
use crate::critique::read_cell_labels;
use clap::Args;
use legume_numeric::matrix::common_io::open_buf_reader;
use legume_numeric::matrix::table::{is_parquet_path, Delimiter, TableReader};
use rustc_hash::FxHashMap as HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

/// A per-cell statistic to start on.
#[derive(Clone, Copy, Debug, PartialEq, Eq, clap::ValueEnum)]
pub enum CellStat {
    /// The total count (on ln(1 + count) unless a colour scale says).
    Total,
    /// How many features have a count.
    Detected,
    /// The mean count over the features (those matching `--rows`).
    Mean,
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
        help = "With --cell-stat: over the features whose names match REGEX only (case-insensitive, as data-beans statistics --row-name-pattern), as '/methylated$'"
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
        value_name = "REGEX",
        requires = "cell_stat",
        conflicts_with = "per_run_umi",
        help = "With --cell-stat: divided by the same statistic over the features matching REGEX, as --rows '/methylated$' --per-rows '/(un)?methylated$' for a methylated fraction"
    )]
    pub per_rows: Option<String>,

    #[arg(
        long,
        value_name = "FILE[:COLUMN]",
        conflicts_with = "cell_stat",
        help = "Start coloured by a number per cell from a table (TSV, CSV or parquet): its first column names the cell (matched to the run's as --extra matches), COLUMN (default: the last numeric one) the value; `u` in the view"
    )]
    pub values: Option<String>,

    #[arg(
        long,
        value_name = "FILE[:COLUMN]",
        help = "Start coloured by groups from a table (TSV, CSV or parquet): its first column names the cell (matched to the run's by name or barcode), COLUMN (default: the second) the group; shown as the `groups` grouping, unmatched cells grey"
    )]
    pub groups: Option<String>,

    #[arg(
        long,
        value_name = "FILE|NAME=#RRGGBB,...",
        requires = "groups",
        help = "With --groups: each group's colour, from a table of group and #rrggbb (its first two columns) or as NAME=#RRGGBB pairs joined by commas"
    )]
    pub group_colours: Option<String>,
}

impl CellArgs {
    /// The statistic the flags ask for: the plain total or count of
    /// features, else one over rows or per the run's totals.
    pub(crate) fn stat(&self) -> Result<Option<Stat>, String> {
        self.cell_stat
            .map(|m| {
                cell_stat(
                    m,
                    self.rows.as_deref(),
                    self.per_run_umi,
                    self.per_rows.as_deref(),
                )
            })
            .transpose()
    }

    /// Refuse what cannot work before anything opens: a statistic that is
    /// not one, a table with no such column.
    pub fn check(&self) -> anyhow::Result<()> {
        self.stat().map_err(anyhow::Error::msg)?;
        if let Some(arg) = &self.values {
            let (file, column) = file_and_column(arg);
            read_numbers(&file, column)?;
        }
        if let Some(arg) = &self.groups {
            let (file, column) = file_and_column(arg);
            check_table(&file, column)?;
        }
        if let Some(c) = &self.group_colours {
            super::group_table::parse_colours(c)?;
        }
        Ok(())
    }
}

/// `measure` over the rows matching `rows`, per `per_run` of the run's own
/// counts: the plain total or count of features when neither is set.
pub(crate) fn cell_stat(
    measure: CellStat,
    rows: Option<&str>,
    per_run: Option<u32>,
    per_rows: Option<&str>,
) -> Result<Stat, String> {
    Ok(match (rows, per_run, per_rows, measure) {
        (None, None, None, CellStat::Total) => Stat::Total,
        (None, None, None, CellStat::Detected) => Stat::Detected,
        _ => Stat::Rows(RowStat::new(measure, rows, per_run)?.per_rows(per_rows)?),
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

/// A table's numbers by cell.
type CellNumbers = Vec<(Box<str>, f32)>;

/// `column` (else the last numeric column) of the table at `path`, by its
/// first column: its name and the finite numbers (a missing or non-finite
/// value is left out). A column with any other text is not numeric.
pub(crate) fn read_numbers(
    path: &Path,
    column: Option<&str>,
) -> anyhow::Result<(Box<str>, CellNumbers)> {
    let shown = path.to_string_lossy();
    let header = check_table(path, None)?;
    let numbers = |c: &str| -> anyhow::Result<Option<CellNumbers>> {
        let read: Vec<(Box<str>, Box<str>)> = read_cell_labels(&shown, Some(c))?;
        let parsed: Option<Vec<_>> = read
            .into_iter()
            .map(|(cell, v)| Some((cell, v.parse::<f32>().ok()?)))
            .collect();
        Ok(parsed
            .map(|v| {
                v.into_iter()
                    .filter(|(_, x)| x.is_finite())
                    .collect::<CellNumbers>()
            })
            .filter(|v| !v.is_empty()))
    };
    match column {
        Some(c) => Ok((
            c.into(),
            numbers(c)?.ok_or_else(|| anyhow::anyhow!("{shown}: column {c} is not numeric"))?,
        )),
        None => header
            .iter()
            .skip(1)
            .rev()
            .find_map(|c| Some((c.clone(), numbers(c).ok()??)))
            .ok_or_else(|| anyhow::anyhow!("{shown}: no numeric column")),
    }
}

/// The header of the table at `path`, with `column` among its names (else
/// a column after the cell's); a text table whose header is one name
/// short is refused, as its names would sit one column off. Nothing past
/// the first rows is read.
pub(crate) fn check_table(path: &Path, column: Option<&str>) -> anyhow::Result<Vec<Box<str>>> {
    let shown = path.to_string_lossy();
    let table = TableReader::open(&shown)?;
    anyhow::ensure!(
        is_parquet_path(&shown) || !header_one_short(&shown)?,
        "{shown}: its rows have one more field than its header names (R's write.table without col.names = NA?): name the cell column too"
    );
    match column {
        Some(c) => {
            table.select(&[c])?;
        }
        None => anyhow::ensure!(
            table.header().len() > 1,
            "{shown}: no column after the cell's"
        ),
    }
    Ok(table.header().to_vec())
}

/// The rows of a table read from `path` whose cell (its first column) is
/// one of `cells`, by name or barcode as an extra file's cells match
/// ([`senna::senna_input::match_cells`]), keyed by the run's name. A table
/// with none of them is refused.
pub(crate) fn match_rows<T: Default>(
    cells: &[Box<str>],
    rows: Vec<(Box<str>, T)>,
    path: &Path,
) -> anyhow::Result<Vec<(Box<str>, T)>> {
    let (names, mut values): (Vec<Box<str>>, Vec<T>) = rows.into_iter().unzip();
    let matched: Vec<(Box<str>, T)> = senna::senna_input::match_cells(cells, &names)
        .into_iter()
        .map(|(cell, row)| (cell, std::mem::take(&mut values[row as usize])))
        .collect();
    let file = path
        .file_name()
        .map_or(path.to_string_lossy(), |f| f.to_string_lossy());
    anyhow::ensure!(
        !matched.is_empty(),
        "none of the run's {} cells are named in {file}",
        cells.len()
    );
    Ok(matched)
}

/// Whether a text table's first row has one field more than its header:
/// R's write.table names every column but the row names', so the names
/// would sit one column off.
fn header_one_short(path: &str) -> anyhow::Result<bool> {
    use std::io::BufRead;
    let mut lines = open_buf_reader(path)?
        .lines()
        .map_while(Result::ok)
        .filter(|l| !l.trim().is_empty() && !l.starts_with("##"));
    let (Some(header), Some(row)) = (lines.next(), lines.next()) else {
        return Ok(false);
    };
    let header = header.strip_prefix('#').unwrap_or(&header);
    let delim = Delimiter::detect(header);
    Ok(delim.split(&row).len() == delim.split(header).len() + 1)
}

/// A number per cell of the run, read from a table.
pub(crate) struct CellValues {
    /// The table and its column, as asked again.
    pub file: PathBuf,
    pub column: Box<str>,
    /// The column and the file, as the key says them.
    pub title: String,
    pub by_cell: HashMap<Box<str>, f32>,
    /// How many cells the run has.
    pub of: usize,
}

impl CellValues {
    /// [`read_numbers`] of the table at `path`, its first column matched
    /// to `cells` as an extra file's cells are
    /// ([`senna::senna_input::match_cells`]). A table with none of the
    /// cells is refused.
    pub(crate) fn read(
        path: &Path,
        column: Option<&str>,
        cells: &[Box<str>],
    ) -> anyhow::Result<Self> {
        let (column, read) = read_numbers(path, column)?;
        let by_cell: HashMap<Box<str>, f32> = match_rows(cells, read, path)?.into_iter().collect();
        let file = path
            .file_name()
            .map_or(path.to_string_lossy(), |f| f.to_string_lossy());
        Ok(Self {
            title: format!("{column} · {file}"),
            file: path.to_path_buf(),
            column,
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
        // Checked at the start ([`CellArgs::check`]); what is left to fail
        // is how the table matches the run, said in the note.
        if let Ok(Some(stat)) = args.stat() {
            self.show_stat(stat);
        }
        if let Some(arg) = &args.values {
            let (file, column) = file_and_column(arg);
            match CellValues::read(&file, column, run_cells(&self.data)) {
                Ok(v) => self.show_values(Arc::new(v)),
                Err(e) => self.note = Some(format!("values: {e}")),
            }
        }
        if let Some(arg) = &args.groups {
            let (file, column) = file_and_column(arg);
            let colours = args
                .group_colours
                .as_deref()
                .map(super::group_table::parse_colours)
                .transpose();
            match colours.and_then(|c| {
                GroupTable::read(&file, column, c.unwrap_or_default(), run_cells(&self.data))
            }) {
                // Coloured by a number per cell asked for too, the table
                // is one grouping more.
                Ok(t) => {
                    let numbers = matches!(self.pick, Some(Pick::Stat(_) | Pick::Values));
                    self.open_groups(Arc::new(t), !numbers);
                }
                Err(e) => self.note = Some(format!("groups: {e}")),
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
        // Another table is another colouring, though the pick is the same.
        self.shown = None;
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
        Ok((self.named_levels(space, &v.by_cell), v.title.clone()))
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
