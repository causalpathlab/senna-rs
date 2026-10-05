//! A small planted data set for tests that fit end to end.

use data_beans::sparse_io::{create_sparse_from_triplets, SparseIoBackend};
use std::path::Path;

/// Two groups of cells, each with its own block of high genes and sparse
/// background elsewhere, large enough for the pseudobulk tree to build.
pub(crate) fn planted_zarr(dir: &Path) -> String {
    let (n_genes, n_cells) = (60usize, 200usize);
    let mut triplets: Vec<(u64, u64, f32)> = Vec::new();
    for c in 0..n_cells {
        let grp = usize::from(c >= n_cells / 2);
        for g in 0..n_genes {
            let own = usize::from(g >= n_genes / 2) == grp;
            let x = if own {
                3 + (c + g) % 4
            } else if (c * 7 + g) % 5 == 0 {
                1
            } else {
                0
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
    path
}
