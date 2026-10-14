//! A fit's preprocessing, kept for another fit of the same data: the cells
//! QC kept and those it writes out, the rows, the HVG selection and the
//! random projection. A fit collapsing on this one's partition (`--pb-from`,
//! built with the same shaping settings) reads them instead of computing them
//! again: no QC pass, no HVG pass, no projection pass.

use senna::embed_common::*;

/// `{prefix}.prep.parquet`: every training cell's projection.
fn proj_path(prefix: &str) -> String {
    format!("{prefix}.prep.parquet")
}

/// `{prefix}.prep.json`: the rows, the HVG selection, the cells written out.
fn meta_path(prefix: &str) -> String {
    format!("{prefix}.prep.json")
}

#[derive(serde::Serialize, serde::Deserialize)]
struct Meta {
    /// The rows after QC.
    rows: Vec<Box<str>>,
    /// The HVG selection's row names, when one ran.
    hvg: Option<Vec<Box<str>>>,
    /// The training cells written out (QC's near-empty ones left out), when
    /// QC ran.
    output: Option<Vec<usize>>,
}

/// A fit's preprocessing.
pub struct Prep {
    /// The training cells, in order.
    pub cells: Vec<Box<str>>,
    pub rows: Vec<Box<str>>,
    pub hvg: Option<Vec<Box<str>>>,
    pub output: Option<Vec<usize>>,
    /// `[K × N]` the random projection of the training cells.
    pub proj_kn: Mat,
}

impl Prep {
    pub fn write(&self, prefix: &str) -> anyhow::Result<()> {
        let names: Vec<Box<str>> = (0..self.proj_kn.nrows())
            .map(|i| format!("p{i}").into())
            .collect();
        self.proj_kn.transpose().to_parquet_with_names(
            &proj_path(prefix),
            (Some(&self.cells), Some("cell")),
            Some(&names),
        )?;
        let meta = Meta {
            rows: self.rows.clone(),
            hvg: self.hvg.clone(),
            output: self.output.clone(),
        };
        std::fs::write(meta_path(prefix), serde_json::to_string(&meta)?)?;
        Ok(())
    }

    /// The preprocessing the run at `prefix` kept, if it kept any.
    pub fn read(prefix: &str) -> anyhow::Result<Option<Self>> {
        let (proj, meta) = (proj_path(prefix), meta_path(prefix));
        if !std::path::Path::new(&proj).exists() || !std::path::Path::new(&meta).exists() {
            return Ok(None);
        }
        let Meta { rows, hvg, output } = serde_json::from_str(&std::fs::read_to_string(&meta)?)?;
        let table = Mat::from_parquet(&proj)?;
        Ok(Some(Self {
            cells: table.rows,
            rows,
            hvg,
            output,
            proj_kn: table.mat.transpose(),
        }))
    }
}

#[cfg(test)]
#[path = "prep_tests.rs"]
mod tests;
