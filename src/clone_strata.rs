//! `{out}.clones.parquet`, the clone table `mung clones` writes: one row
//! per cell with its `donor` and `stratum` (0 mixes freely; each donor-
//! private clone has its own). The file is the whole contract with mung:
//! senna reads it and never links mung.

use legume_numeric::matrix::parquet::{peek_parquet_field_names, read_table_columns};
use log::warn;
use rustc_hash::FxHashMap;

/// A clone table as `senna run` summarizes it, one entry per cell.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct CloneTable {
    pub donor: Vec<Box<str>>,
    pub stratum: Vec<usize>,
    /// Donor purity of each cell's clone, when the table has it.
    pub purity: Option<Vec<f32>>,
    /// Posterior malignancy of each cell, when the table has it.
    pub p_malig: Option<Vec<f32>>,
}

/// The `stratum` column as counts.
fn strata(path: &str, column: Vec<f64>) -> anyhow::Result<Vec<usize>> {
    column
        .into_iter()
        .enumerate()
        .map(|(i, s)| {
            anyhow::ensure!(
                s >= 0.0 && s.fract() == 0.0,
                "{path}: row {i}: stratum {s} is not a count"
            );
            Ok(s as usize)
        })
        .collect()
}

/// A string column and numeric columns of one table.
type Columns = (Vec<Box<str>>, Vec<Vec<f64>>);

/// String column `name` and numeric columns `numeric` of `path`, all of
/// one length.
fn columns(path: &str, name: &str, numeric: &[&str]) -> anyhow::Result<Columns> {
    let (strs, nums) = read_table_columns(path, &[name], numeric)?;
    let strs = strs.into_iter().next().unwrap_or_default();
    anyhow::ensure!(
        nums.len() == numeric.len() && nums.iter().all(|c| c.len() == strs.len()),
        "{path}: columns of different lengths"
    );
    Ok((strs, nums))
}

/// Read the clone table at `path`: donors, strata, and purity and
/// malignancy where the table has them.
pub fn read(path: &str) -> anyhow::Result<CloneTable> {
    let fields = peek_parquet_field_names(path)?;
    let has = |c: &str| fields.iter().any(|f| &**f == c);
    let optional: Vec<&str> = ["purity", "p_malig"]
        .into_iter()
        .filter(|c| has(c))
        .collect();
    let numeric: Vec<&str> = std::iter::once("stratum")
        .chain(optional.iter().copied())
        .collect();
    let (donor, nums) = columns(path, "donor", &numeric)?;
    let mut nums = nums.into_iter();
    let stratum = strata(path, nums.next().unwrap_or_default())?;
    let mut t = CloneTable {
        donor,
        stratum,
        ..CloneTable::default()
    };
    for (c, v) in optional.into_iter().zip(nums) {
        let v = Some(v.into_iter().map(|x| x as f32).collect());
        match c {
            "purity" => t.purity = v,
            _ => t.p_malig = v,
        }
    }
    Ok(t)
}

/// Each of `cells`' stratum in the clone table at `path`; cells it does
/// not list go to stratum 0, with a warning.
pub fn strata_of(path: &str, cells: &[Box<str>]) -> anyhow::Result<Vec<usize>> {
    let (cell, nums) = columns(path, "cell", &["stratum"])?;
    let stratum = strata(path, nums.into_iter().next().unwrap_or_default())?;
    let by_cell: FxHashMap<&str, usize> = cell.iter().map(AsRef::as_ref).zip(stratum).collect();
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
    Ok(out)
}

#[cfg(test)]
#[path = "tests/clone_strata.rs"]
mod tests;
