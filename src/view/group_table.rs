//! Cell groups from a table: a categorical column of a TSV, CSV or parquet
//! file, its first column naming the cell, shown as the `groups` grouping
//! beside the run's own (colours, legend, counts, focus and export as any
//! grouping has), with optional colours per group.

use super::cell_colour::{check_table, match_rows};
use super::data::{Axis, LabelKind, Labels};
use super::files::modified;
use super::Scene;
use anyhow::{ensure, Context, Result};
use std::path::{Path, PathBuf};
use std::sync::Arc;

/// A group's name and its colour (8-bit sRGB).
pub(crate) type GroupColour = (Box<str>, [u8; 3]);

/// A table's groups matched to the run's cells.
pub(crate) struct GroupTable {
    /// The table (absolute, so a watcher elsewhere opens it too), when it
    /// was last changed, and its column, as asked again.
    pub file: PathBuf,
    modified: Option<std::time::SystemTime>,
    pub column: Option<Box<str>>,
    /// Each group's colour given, over the palette's (a colour chosen in
    /// the style menu wins); never saved beside the run.
    pub colours: Vec<GroupColour>,
    pub labels: Labels,
    /// How many cells the run has.
    of: usize,
}

impl GroupTable {
    /// `column` (else the second) of the table at `path`, by its first
    /// column, matched to `cells` as `--values` matches them
    /// ([`match_rows`]). Empty and `NA` labels are left out; a colour for a
    /// group the table does not have is refused.
    pub(crate) fn read(
        path: &Path,
        column: Option<&str>,
        colours: Vec<GroupColour>,
        cells: &[Box<str>],
    ) -> Result<Self> {
        let shown = path.to_string_lossy();
        check_table(path, column)?;
        let read: Vec<(Box<str>, Box<str>)> = crate::critique::read_cell_labels(&shown, column)?;
        let labels = Labels::new(LabelKind::Groups, match_rows(cells, read, path)?, &[]);
        if let Some((name, _)) = colours
            .iter()
            .find(|(n, _)| !labels.levels.iter().any(|l| l == n))
        {
            anyhow::bail!(
                "a colour for group {name}, which none of the run's cells in {shown} have; it has {}",
                labels.levels.join(", ")
            );
        }
        Ok(Self {
            file: std::path::absolute(path)?,
            modified: modified(path),
            column: column.map(Into::into),
            colours,
            labels,
            of: cells.len(),
        })
    }

    /// How the table matched the run's cells.
    #[must_use]
    pub fn said(&self) -> String {
        let file = self
            .file
            .file_name()
            .map_or(self.file.to_string_lossy(), |f| f.to_string_lossy());
        format!(
            "groups: {} groups from {file} · {} of the run's {} cells matched",
            self.labels.levels.len(),
            self.labels.by_name.len(),
            self.of
        )
    }

    /// Whether `self` is the table `file`/`column` with `colours`, the
    /// file unchanged since it was read.
    #[must_use]
    pub fn is(&self, file: &Path, column: Option<&str>, colours: &[GroupColour]) -> bool {
        std::path::absolute(file).is_ok_and(|f| f == self.file)
            && modified(file) == self.modified
            && self.column.as_deref() == column
            && self.colours == colours
    }
}

/// Group names and `#rrggbb` (or `rrggbb`) colours, as 8-bit sRGB.
pub(crate) fn colours<'a>(
    pairs: impl IntoIterator<Item = (&'a str, &'a str)>,
) -> Result<Vec<GroupColour>> {
    pairs
        .into_iter()
        .map(|(name, s)| {
            let h = s.trim().trim_start_matches('#');
            ensure!(
                h.len() == 6 && h.chars().all(|c| c.is_ascii_hexdigit()),
                "{name}: {s} is not a #rrggbb colour"
            );
            let byte = |i: usize| u8::from_str_radix(&h[i..i + 2], 16).expect("checked hex digits");
            Ok((name.trim().into(), [byte(0), byte(2), byte(4)]))
        })
        .collect()
}

/// `--group-colours`: a table of group and colour (its first two columns),
/// else `NAME=#RRGGBB` pairs joined by commas.
pub(crate) fn parse_colours(arg: &str) -> Result<Vec<GroupColour>> {
    if Path::new(arg).is_file() {
        let read: Vec<(Box<str>, Box<str>)> = crate::critique::read_cell_labels(arg, None)?;
        return colours(read.iter().map(|(n, c)| (&**n, &**c)));
    }
    let pairs = arg
        .split(',')
        .filter(|p| !p.trim().is_empty())
        .map(|p| {
            p.rsplit_once('=')
                .with_context(|| format!("{p}: give NAME=#RRGGBB, or a table of them"))
        })
        .collect::<Result<Vec<_>>>()?;
    colours(pairs)
}

impl Scene {
    /// Open `table` as the `groups` grouping, in place of any table open
    /// before (whose groups are not focused any more), and with `colour`
    /// colour by it.
    pub(crate) fn open_groups(&mut self, table: Arc<GroupTable>, colour: bool) {
        self.note = Some(table.said());
        match self.label_index(LabelKind::Groups) {
            Some(at) => {
                self.data.labels[at] = table.labels.clone();
                if self.colour == Some(at) {
                    self.focus = None;
                }
            }
            None => self.data.labels.push(table.labels.clone()),
        }
        self.group_table = Some(table);
        self.groups = None;
        self.refresh_groups();
        if colour {
            self.colour_by_groups();
        }
    }

    /// Colour by the open table's groups, on a map of cells; a focus kept
    /// when they are shown already.
    pub(crate) fn colour_by_groups(&mut self) {
        let at = self.label_index(LabelKind::Groups);
        if self.current().axis() == Axis::Cells && at.is_some() && at != self.colour {
            self.set_colour(at);
        }
    }

    /// The open table's colours, when `labels` are its groups.
    pub(super) fn given_colours(&self, labels: &Labels) -> &[GroupColour] {
        match &self.group_table {
            Some(t) if labels.kind == LabelKind::Groups => &t.colours,
            _ => &[],
        }
    }
}

#[cfg(test)]
#[path = "tests/group_table.rs"]
mod tests;
