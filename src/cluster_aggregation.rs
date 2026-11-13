//! Streaming per-cluster aggregation over `SparseIoVec`.
//!
//! Walks raw count columns in parallel column-blocks, accumulating
//! `T[k, g] = Σ_{n ∈ k} y[g, n]`, the per-group gene-sum tensor laid out
//! row-major as `Vec<f64>` of length `k · m`. Used by:
//!   - `cluster_bhc::run_cluster_bhc` for Dirichlet-Multinomial BHC sufficient
//!     stats;
//!   - `annotate` to derive NB-Fisher-adjusted per-cluster mean expression.
//!
//! Cells with `labels[n] >= k` (or `usize::MAX` from `remove_small_clusters`)
//! are silently skipped — callers handle "unassigned" out of band.

use crate::embed_common::*;
use rayon::prelude::*;

/// Streaming per-group gene-sum aggregator. Returns a `Vec<f64>` of length
/// `k * m` laid out row-major (`gene_sum[k * m + g]` = total count of gene
/// `g` over cells with `label == k`).
pub fn accumulate_gene_sum(
    data_vec: &SparseIoVec,
    labels: &[usize],
    k: usize,
    m: usize,
    block_size: usize,
) -> anyhow::Result<Vec<f64>> {
    let mut out = accumulate_gene_sum_multi(data_vec, &[(labels, k)], m, block_size, None)?;
    Ok(out.pop().expect("one grouping in, one out"))
}

/// Accumulate any number of independent groupings in one pass over the columns.
///
/// Reading the columns is the expensive part — it is the raw matrix off disk —
/// so a caller that needs several partitions of the same cells (deconvolve pools
/// over archetype granularities) must not re-stream the matrix per partition.
/// Each grouping is `(labels, k)`; cells whose label is outside `0..k` are
/// skipped, which is how callers park cells they mean to exclude.
///
/// With `fold` (`[m, columns]` and each cell's column), a cell's count of
/// gene `g` is divided by `fold[g, column]` as it is added: the per-batch
/// fold of that cell's batch, so the sums are of batch-adjusted counts.
pub fn accumulate_gene_sum_multi(
    data_vec: &SparseIoVec,
    groupings: &[(&[usize], usize)],
    m: usize,
    block_size: usize,
    fold: Option<CellFold<'_>>,
) -> anyhow::Result<Vec<Vec<f64>>> {
    anyhow::ensure!(!groupings.is_empty(), "no groupings to accumulate");
    let n = groupings[0].0.len();
    for (labels, _) in groupings {
        anyhow::ensure!(
            labels.len() == n,
            "grouping length mismatch: {} vs {n}",
            labels.len()
        );
    }
    let blocks: Vec<(usize, usize)> = (0..n)
        .step_by(block_size.max(1))
        .map(|lb| (lb, (lb + block_size).min(n)))
        .collect();

    // One accumulator per worker, and workers are capped by what an accumulator
    // costs. This is the peak allocation of the whole aggregation: every worker
    // holds `Σ_i k_i · m` f64 at once, which for a thousand groups over 30k
    // genes is ~250 MB each — enough to exhaust memory on a machine with plenty
    // of cores before any of the count data is even touched.
    let acc_elems: usize = groupings.iter().map(|&(_, k)| k * m).sum();
    let acc_bytes = acc_elems * std::mem::size_of::<f64>();
    let affordable = (ACCUMULATOR_BUDGET_BYTES / acc_bytes.max(1)).max(1);
    let workers = rayon::current_num_threads()
        .min(affordable)
        .min(blocks.len().max(1));
    if workers < rayon::current_num_threads() {
        log::info!(
            "gene-sum aggregation: {workers} workers (not {}) — each holds {} MiB of group sums",
            rayon::current_num_threads(),
            acc_bytes >> 20
        );
    }

    let chunk = blocks.len().div_ceil(workers.max(1));
    blocks
        .par_chunks(chunk.max(1))
        .map(|chunk| {
            // Accumulate every block of this chunk into ONE buffer. Giving each
            // block its own would double the peak for no benefit.
            let mut acc: Vec<Vec<f64>> = groupings
                .iter()
                .map(|&(_, k)| vec![0.0f64; k * m])
                .collect();
            for &(lb, ub) in chunk {
                add_block_gene_sum(&mut acc, data_vec, groupings, (lb, ub), m, fold)?;
            }
            anyhow::Ok(acc)
        })
        .try_reduce_with(|mut a, b| {
            for (x, y) in a.iter_mut().zip(b) {
                for (xi, yi) in x.iter_mut().zip(y) {
                    *xi += yi;
                }
            }
            anyhow::Ok(a)
        })
        .unwrap_or_else(|| {
            Ok(groupings
                .iter()
                .map(|&(_, k)| vec![0.0f64; k * m])
                .collect())
        })
}

/// Peak group-sum memory allowed across all aggregation workers.
///
/// Parallelism here is bounded by memory, not by cores: the column reads are
/// cheap next to holding one full group-sum tensor per worker.
const ACCUMULATOR_BUDGET_BYTES: usize = 1 << 30;

/// A per-batch gene fold `[genes, columns]` and each cell's column of it.
pub type CellFold<'a> = (&'a Mat, &'a [u32]);

/// Add one column block into an existing per-grouping accumulator.
fn add_block_gene_sum(
    sums: &mut [Vec<f64>],
    data_vec: &SparseIoVec,
    groupings: &[(&[usize], usize)],
    (lb, ub): (usize, usize),
    m: usize,
    fold: Option<CellFold<'_>>,
) -> anyhow::Result<()> {
    let csc = data_vec.read_columns_csc(lb..ub)?;
    for j in 0..csc.ncols() {
        let global = lb + j;
        let col = csc.col(j);
        let rows = col.row_indices();
        let vals = col.values();

        for (sum, &(labels, k)) in sums.iter_mut().zip(groupings) {
            let kk = labels[global];
            if kk >= k {
                continue;
            }
            let row = &mut sum[kk * m..(kk + 1) * m];
            match fold {
                Some((delta, cols)) => {
                    let d = delta.column(cols[global] as usize);
                    for (&g, &v) in rows.iter().zip(vals.iter()) {
                        row[g] += f64::from(v / d[g]);
                    }
                }
                None => {
                    for (&g, &v) in rows.iter().zip(vals.iter()) {
                        row[g] += f64::from(v);
                    }
                }
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod fold_tests {
    use super::*;
    use data_beans::sparse_io::{create_sparse_from_triplets, SparseIoBackend};

    // Bar: with a fold, a cell's counts are divided by its batch's per-gene
    // fold as they are summed; without one the sums are the raw counts.
    #[test]
    fn gene_sums_divide_each_cell_by_its_batch_fold() -> anyhow::Result<()> {
        let dir = tempfile::tempdir()?;
        let path = dir.path().join("x.zarr");
        // Two genes × three cells, all in one group; cell 2 is in batch 1.
        let triplets: Vec<(u64, u64, f32)> = vec![
            (0, 0, 2.0),
            (1, 0, 4.0),
            (0, 1, 6.0),
            (0, 2, 9.0),
            (1, 2, 5.0),
        ];
        let mut b = create_sparse_from_triplets(
            &triplets,
            (2, 3, triplets.len()),
            Some(path.to_str().unwrap()),
            Some(&SparseIoBackend::Zarr),
        )?;
        b.register_row_names_vec(&["g0".into(), "g1".into()]);
        b.register_column_names_vec(&["c0".into(), "c1".into(), "c2".into()]);
        let mut data = SparseIoVec::new();
        data.push(std::sync::Arc::from(b), None)?;
        let labels = [0usize, 0, 0];
        let raw = accumulate_gene_sum_multi(&data, &[(&labels, 1)], 2, 2, None)?;
        assert_eq!(raw[0], [17.0, 9.0]);
        // Batch 0 has no fold; batch 1 reads gene 0 three times too high.
        let delta = Mat::from_row_slice(2, 2, &[1.0, 3.0, 1.0, 1.0]);
        let cols = [0u32, 0, 1];
        let adjusted =
            accumulate_gene_sum_multi(&data, &[(&labels, 1)], 2, 2, Some((&delta, &cols)))?;
        assert_eq!(adjusted[0], [11.0, 9.0]);
        Ok(())
    }
}
