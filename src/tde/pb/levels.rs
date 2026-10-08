//! Multilevel pseudobulks of the base fit's cells, from `data-beans`'
//! multilevel collapse with the cells' base states as the projection: the
//! levels the kinetics are fitted on, coarse to fine.
//!
//! Provisional: these are not the base fit's own pseudobulks (that fit keeps
//! its levels to itself), but cells are grouped by where the base fit placed
//! them, at its default collapse settings.

use data_beans::alg::collapse_data::{
    collapse_columns_multilevel_with_hierarchy, MultilevelParams,
};
use data_beans::sparse_io::open_sparse_matrix_by_path;
use data_beans::sparse_io_vector::SparseIoVec;
use nalgebra::DMatrix;
use std::sync::Arc;

/// `senna`'s default collapse: sort dimension, levels, neighbours, iterations.
const SORT_DIM: usize = 10;
const KNN: usize = 10;
const ITER_OPT: usize = 30;

/// Every cell's pseudobulk at every level, coarsest → finest; `None` for a
/// cell without a base state. `states` `[H × N]` are the cells' base states in
/// the backend's column order, `has_state` which columns have one.
pub fn pb_levels(
    path: &str,
    states: &DMatrix<f32>,
    has_state: &[bool],
    num_levels: usize,
) -> anyhow::Result<Vec<Vec<Option<usize>>>> {
    let mut data = SparseIoVec::new();
    data.push(Arc::from(open_sparse_matrix_by_path(path)?), None)?;
    anyhow::ensure!(
        has_state.len() == data.num_columns() && states.ncols() == has_state.len(),
        "{} states for {} columns",
        states.ncols(),
        data.num_columns()
    );
    data.mask_columns(has_state)?;
    let kept: Vec<usize> = (0..has_state.len()).filter(|&c| has_state[c]).collect();
    let proj = DMatrix::<f32>::from_fn(states.nrows(), kept.len(), |k, j| states[(k, kept[j])]);
    let batch = vec![0u32; kept.len()];
    let mut params = MultilevelParams::new(states.nrows());
    params.num_levels = num_levels;
    params.sort_dim = SORT_DIM;
    params.knn_pb_samples = KNN;
    params.num_opt_iter = ITER_OPT;
    let out = collapse_columns_multilevel_with_hierarchy(&mut data, &proj, &batch, &params)?;
    // The collapse lists its levels finest first.
    Ok(out
        .cell_to_pb_per_level
        .iter()
        .rev()
        .map(|level| {
            let mut full = vec![None; has_state.len()];
            for (j, &pb) in level.iter().enumerate() {
                full[kept[j]] = Some(pb);
            }
            full
        })
        .collect())
}

/// Each pseudobulk's parent at the next coarser level: the parent of most of
/// its cells.
#[must_use]
pub fn parents(fine: &[Option<usize>], coarse: &[Option<usize>]) -> Vec<usize> {
    let n_fine = fine.iter().flatten().max().map_or(0, |&m| m + 1);
    let mut votes: Vec<std::collections::HashMap<usize, usize>> = vec![Default::default(); n_fine];
    for (f, c) in fine.iter().zip(coarse) {
        if let (Some(f), Some(c)) = (f, c) {
            *votes[*f].entry(*c).or_default() += 1;
        }
    }
    votes
        .iter()
        .map(|v| {
            v.iter()
                .max_by_key(|(c, n)| (**n, std::cmp::Reverse(**c)))
                .map_or(0, |(c, _)| *c)
        })
        .collect()
}

#[cfg(test)]
#[path = "levels/tests.rs"]
mod tests;
