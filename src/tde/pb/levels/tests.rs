use super::*;
use data_beans::sparse_io::{create_sparse_from_dmatrix, SparseIoBackend};

#[test]
fn parents_take_the_majority_of_their_cells() {
    let fine = [Some(0), Some(0), Some(0), Some(1), Some(1), None];
    let coarse = [Some(4), Some(4), Some(7), Some(7), Some(7), Some(4)];
    assert_eq!(parents(&fine, &coarse), vec![4, 7]);
}

/// Two well-separated groups of cells: every level keeps them apart, levels
/// get no coarser toward the fine end, and a cell without a state is left out.
#[test]
fn levels_follow_the_states_and_skip_cells_without_one() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("x.zarr").to_string_lossy().into_owned();
    let n = 120;
    let m = DMatrix::<f32>::from_fn(6, n, |g, c| ((g + c) % 4 + 1) as f32);
    let mut b = create_sparse_from_dmatrix(&m, Some(&path), Some(&SparseIoBackend::Zarr)).unwrap();
    let rows: Vec<Box<str>> = (0..6).map(|g| format!("G{g}").into()).collect();
    let cols: Vec<Box<str>> = (0..n).map(|c| format!("c{c}").into()).collect();
    b.register_row_names_vec(&rows);
    b.register_column_names_vec(&cols);
    let states = DMatrix::<f32>::from_fn(4, n, |k, c| {
        let side = if c < n / 2 { 1.0 } else { -1.0 };
        side * [3.0, 1.0, 0.5, 0.2][k] + 0.01 * ((c * 7 + k) % 11) as f32
    });
    let mut has = vec![true; n];
    has[5] = false;
    let levels = pb_levels(&path, &states, &has, 3).unwrap();
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
        assert!(level[5].is_none());
        for a in 0..n / 2 {
            for b in n / 2..n {
                if let (Some(x), Some(y)) = (level[a], level[b]) {
                    assert_ne!(x, y, "cells {a} and {b} share a pseudobulk");
                }
            }
        }
    }
}
