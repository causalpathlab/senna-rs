//! `{out}.clones.parquet`, the clone table `mung clones` writes: one row
//! per cell with its `donor` and `stratum` (0 mixes freely; each donor-
//! private clone has its own). The file is the whole contract with mung:
//! senna reads it and never links mung.

use legume_numeric::matrix::parquet::read_table_columns;
use log::warn;
use rustc_hash::FxHashMap;

/// A clone table, one entry per cell.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct CloneTable {
    pub cell: Vec<Box<str>>,
    pub donor: Vec<Box<str>>,
    pub stratum: Vec<usize>,
    /// Donor purity of each cell's clone, when the table has it.
    pub purity: Option<Vec<f32>>,
    /// Posterior malignancy of each cell, when the table has it.
    pub p_malig: Option<Vec<f32>>,
}

/// Read the clone table at `path`.
pub fn read(path: &str) -> anyhow::Result<CloneTable> {
    let (strs, nums) = read_table_columns(path, &["cell", "donor"], &["stratum"])?;
    let mut strs = strs.into_iter();
    let (cell, donor) = (
        strs.next().unwrap_or_default(),
        strs.next().unwrap_or_default(),
    );
    let stratum = nums
        .into_iter()
        .next()
        .unwrap_or_default()
        .into_iter()
        .enumerate()
        .map(|(i, s)| {
            anyhow::ensure!(
                s >= 0.0 && s.fract() == 0.0,
                "{path}: row {i}: stratum {s} is not a count"
            );
            Ok(s as usize)
        })
        .collect::<anyhow::Result<Vec<_>>>()?;
    anyhow::ensure!(
        cell.len() == stratum.len() && donor.len() == stratum.len(),
        "{path}: columns of different lengths"
    );
    // Optional: older or other tables may lack them.
    let optional = |name: &str| {
        read_table_columns(path, &[], &[name])
            .ok()
            .and_then(|(_, n)| n.into_iter().next())
            .filter(|v| v.len() == cell.len())
            .map(|v| v.into_iter().map(|x| x as f32).collect())
    };
    Ok(CloneTable {
        purity: optional("purity"),
        p_malig: optional("p_malig"),
        cell,
        donor,
        stratum,
    })
}

/// Each of `cells`' stratum in `table`; cells it does not list go to
/// stratum 0, with a warning.
#[must_use]
pub fn align(table: &CloneTable, cells: &[Box<str>]) -> Vec<usize> {
    let by_cell: FxHashMap<&str, usize> = table
        .cell
        .iter()
        .zip(&table.stratum)
        .map(|(c, &s)| (c.as_ref(), s))
        .collect();
    let mut missing = 0usize;
    let out: Vec<usize> = cells
        .iter()
        .map(|c| {
            by_cell.get(c.as_ref()).copied().unwrap_or_else(|| {
                missing += 1;
                0
            })
        })
        .collect();
    if missing > 0 {
        warn!(
            "{missing} / {} cells missing from the clone table; treating as stratum 0",
            cells.len()
        );
    }
    out
}

#[cfg(test)]
#[path = "tests/clone_strata.rs"]
mod tests;
