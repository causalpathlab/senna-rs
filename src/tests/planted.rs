//! A small planted data set for tests that fit end to end.

use data_beans::sparse_io::{create_sparse_from_triplets, SparseIoBackend};
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
    let nnz = triplets.len();
    let path = dir.join("planted.zarr").to_string_lossy().into_owned();
    let mut b = create_sparse_from_triplets(
        &triplets,
        (n_genes, n_cells, nnz),
        Some(&path),
        Some(&SparseIoBackend::Zarr),
    )
    .expect("backend");
    b.register_row_names_vec(
        &(0..n_genes)
            .map(|g| format!("GENE{g}").into_boxed_str())
            .collect::<Vec<_>>(),
    );
    b.register_column_names_vec(
        &(0..n_cells)
            .map(|c| format!("c{c}").into_boxed_str())
            .collect::<Vec<_>>(),
    );
    let batches = dir.join("planted.batch.txt");
    let labels: String = (0..n_cells)
        .map(|c| format!("B{}\n", c % n_batches))
        .collect();
    std::fs::write(&batches, labels).expect("batch file");
    (path, batches.to_string_lossy().into_owned())
}
