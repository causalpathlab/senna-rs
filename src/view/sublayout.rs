//! A fresh layout of one group's cells, computed inside the viewer.
//!
//! The group's rows are taken from the run's geometry table (the one
//! `senna layout` uses) and laid out on their own with the same transform and
//! the same t-UMAP, so a sub-population's structure gets the whole screen
//! instead of the corner the global layout gave it.

use crate::postprocess::{latent_layout_features, tumap_on_columns, DirectUmap};
use rustc_hash::FxHashMap as HashMap;
use senna::embed_common::*;
use senna::run_manifest::{self, RunKind, RunManifest};
use std::path::Path;

/// Fewer cells than this do not make a layout worth looking at.
const MIN_CELLS: usize = 10;
/// Winsorization applied before the layout, as `senna layout` does by default.
const TRIM_MADS: f32 = 5.0;

/// Cells that were laid out, and their coordinates.
pub type Laid = (Vec<Box<str>>, Vec<[f32; 2]>);

/// The run's per-cell geometry table, read once and shared with the worker
/// threads that lay groups out.
pub struct Geometry {
    kind: RunKind,
    latent_nk: Mat,
    index: HashMap<Box<str>, usize>,
}

impl Geometry {
    pub fn load(m: &RunManifest, dir: &Path) -> anyhow::Result<Self> {
        let rel = m
            .outputs
            .geometry_latent()
            .ok_or_else(|| anyhow::anyhow!("the run has no cell embedding or latent to lay out"))?;
        let path = run_manifest::resolve(dir, rel);
        let MatWithNames { rows, mat, .. } =
            Mat::from_parquet_with_row_names(&path.to_string_lossy(), Some(0))?;
        let index = rows.into_iter().enumerate().map(|(i, n)| (n, i)).collect();
        Ok(Self {
            kind: m.kind,
            latent_nk: mat,
            index,
        })
    }

    /// Lay out the cells named in `names`. Returns the cells that were found,
    /// with their coordinates.
    pub fn layout(&self, names: &[Box<str>]) -> anyhow::Result<Laid> {
        let (found, rows): (Vec<Box<str>>, Vec<usize>) = names
            .iter()
            .filter_map(|n| self.index.get(n).map(|&i| (n.clone(), i)))
            .unzip();
        anyhow::ensure!(
            rows.len() >= MIN_CELLS,
            "only {} of these cells are in the run's embedding; too few to lay out",
            rows.len()
        );
        let sub = self.latent_nk.select_rows(&rows);
        let (feat_kn, _) = latent_layout_features(self.kind, &sub, 1.0, TRIM_MADS);
        let params = DirectUmap {
            knn: DirectUmap::default().knn.min(rows.len() - 1),
            ..DirectUmap::default()
        };
        let coords = tumap_on_columns(&feat_kn, &params)?;
        let xy = (0..coords.nrows())
            .map(|i| [coords[(i, 0)], coords[(i, 1)]])
            .collect();
        Ok((found, xy))
    }
}
