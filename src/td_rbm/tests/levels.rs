use super::*;
use data_beans::sparse_io::{create_sparse_from_dmatrix, SparseIoBackend};
use nalgebra::DMatrix;

/// Two groups of cells expressing different genes: every level keeps them
/// apart, levels get no coarser toward the fine end, and every cell has a
/// pseudobulk.
#[test]
fn levels_keep_apart_cells_of_different_expression() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("x.zarr").to_string_lossy().into_owned();
    let (g, n) = (40, 200);
    let m = DMatrix::<f32>::from_fn(g, n, |j, c| {
        let on = (j < g / 2) == (c < n / 2);
        let base = if on { 20.0 } else { 1.0 };
        base + ((j * 13 + c * 7) % 5) as f32
    });
    let mut b = create_sparse_from_dmatrix(&m, Some(&path), Some(&SparseIoBackend::Zarr)).unwrap();
    let rows: Vec<Box<str>> = (0..g).map(|j| format!("G{j}").into()).collect();
    let cols: Vec<Box<str>> = (0..n).map(|c| format!("c{c}").into()).collect();
    b.register_row_names_vec(&rows);
    b.register_column_names_vec(&cols);
    let levels = pb_levels(
        &super::super::counts::open_tracks(&[path.into()]).unwrap(),
        3,
    )
    .unwrap();
    assert_eq!(levels.len(), 3);
    let count = |l: &[Option<usize>]| l.iter().flatten().max().map_or(0, |&m| m + 1);
    for w in levels.windows(2) {
        assert!(
            count(&w[0]) <= count(&w[1]),
            "coarse {} vs fine {}",
            count(&w[0]),
            count(&w[1])
        );
    }
    for level in &levels {
        assert!(
            level.iter().all(Option::is_some),
            "a cell without a pseudobulk"
        );
        for a in 0..n / 2 {
            for b in n / 2..n {
                assert_ne!(level[a], level[b], "cells {a} and {b} share a pseudobulk");
            }
        }
    }
}
