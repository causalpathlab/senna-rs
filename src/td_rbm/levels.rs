//! Multilevel pseudobulks of the backends' cells, from `data-beans`'
//! multilevel collapse at `senna`'s default settings: a random sketch of
//! the reads, then the refined pseudobulk tree, every file a batch. The
//! levels the kinetics are fitted on, coarse to fine; no other fit is needed
//! for them.

use data_beans::alg::collapse_data::{
    collapse_columns_multilevel_with_hierarchy, MultilevelParams, PbTreeParams, ReassignCellsParams,
};
use data_beans::alg::random_projection::RandProjOps;
use data_beans::sparse_io_vector::SparseIoVec;

/// `senna`'s default collapse: sketch rank, sort dimension, neighbours,
/// iterations.
const PROJ_DIM: usize = 50;
const SORT_DIM: usize = 10;
const KNN: usize = 10;
const ITER_OPT: usize = 30;

/// Every cell's pseudobulk at every level, coarsest → finest, in `data`'s
/// column order. The file a cell came from is its batch.
pub fn pb_levels(data: &SparseIoVec, num_levels: usize) -> anyhow::Result<Vec<Vec<Option<usize>>>> {
    let mut data = data.clone_for_collapse();
    let batch: Vec<u32> = (0..data.num_columns())
        .map(|c| data.column_source(c).map_or(0, |b| b as u32))
        .collect();
    let proj = data
        .project_columns_with_batch_correction(PROJ_DIM, None, Some(&batch))?
        .proj;
    let mut params = MultilevelParams::new(PROJ_DIM);
    params.num_levels = num_levels;
    params.sort_dim = SORT_DIM;
    params.knn_pb_samples = KNN;
    params.num_opt_iter = ITER_OPT;
    params.pb_tree = Some(PbTreeParams {
        reassign_cells: Some(ReassignCellsParams::default()),
        ..PbTreeParams::default()
    });
    let out = collapse_columns_multilevel_with_hierarchy(&mut data, &proj, &batch, &params)?;
    // The collapse lists its levels finest first.
    Ok(out
        .cell_to_pb_per_level
        .iter()
        .rev()
        .map(|level| level.iter().map(|&pb| Some(pb)).collect())
        .collect())
}

#[cfg(test)]
#[path = "tests/levels.rs"]
mod tests;
