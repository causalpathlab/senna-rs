//! A small planted data set for tests that fit end to end.

use senna::sparse_output::{write_backend, Entries};
use std::path::Path;

/// Two groups of cells, each with its own block of high genes and sparse
/// background elsewhere, large enough for the pseudobulk tree to build.
pub(crate) fn planted_zarr(dir: &Path) -> String {
    planted_batched_zarr(dir, 200, 1).0
}

/// [`planted_zarr`] with `n_cells` cells in `n_batches` batches (cells take
/// them in turn), where every batch after the first reads a third of the
/// genes three times higher: a batch effect a null has to take out. Returns
/// the backend and its batch file (`B0`, `B1`, ... one line per cell).
pub(crate) fn planted_batched_zarr(
    dir: &Path,
    n_cells: usize,
    n_batches: usize,
) -> (String, String) {
    let n_genes = 60usize;
    let mut triplets: Vec<(u64, u64, f32)> = Vec::new();
    for c in 0..n_cells {
        let grp = usize::from(c >= n_cells / 2);
        let batch = c % n_batches;
        for g in 0..n_genes {
            let own = usize::from(g >= n_genes / 2) == grp;
            let base = if own {
                3 + (c + g) % 4
            } else {
                usize::from((c * 7 + g) % 5 == 0)
            };
            let x = if batch > 0 && g % 3 == 0 {
                3 * base
            } else {
                base
            };
            if x > 0 {
                triplets.push((g as u64, c as u64, x as f32));
            }
        }
    }
    let path = dir.join("planted.zarr.zip").to_string_lossy().into_owned();
    let rows: Vec<Box<str>> = (0..n_genes).map(|g| format!("GENE{g}").into()).collect();
    let cols: Vec<Box<str>> = (0..n_cells).map(|c| format!("c{c}").into()).collect();
    write_backend(&path, Entries::Triplets(triplets), &rows, &cols).expect("backend");
    let batches = dir.join("planted.batch.txt");
    let labels: String = (0..n_cells)
        .map(|c| format!("B{}\n", c % n_batches))
        .collect();
    std::fs::write(&batches, labels).expect("batch file");
    (path, batches.to_string_lossy().into_owned())
}
