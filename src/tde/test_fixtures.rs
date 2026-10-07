//! Shared synthetic fixtures for tde's own tests and `senna predict`'s tde
//! test.

use data_beans::sparse_io::{create_sparse_from_dmatrix, SparseIoBackend};
use nalgebra::DMatrix;

pub(crate) fn boxes(names: &[&str]) -> Vec<Box<str>> {
    names.iter().map(|&s| s.into()).collect()
}

/// A tiny synthetic zarr backend at `dir/{stem}.zarr`, one small positive
/// integer per cell so nothing is near-empty.
pub(crate) fn synth(dir: &std::path::Path, stem: &str, rows: &[&str], cols: &[&str]) -> Box<str> {
    let path = dir.join(format!("{stem}.zarr"));
    let path: Box<str> = path.to_string_lossy().into_owned().into();
    let m = DMatrix::<f32>::from_fn(rows.len(), cols.len(), |r, c| {
        (((r * 7 + c * 11 + 3) % 9) + 1) as f32
    });
    let mut b = create_sparse_from_dmatrix(&m, Some(&path), Some(&SparseIoBackend::Zarr))
        .expect("create synthetic backend");
    b.register_row_names_vec(&boxes(rows));
    b.register_column_names_vec(&boxes(cols));
    path
}
