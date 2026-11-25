//! Running a fit's work over row blocks: how many in flight, block
//! membership, and the per-block batch δ.

use crate::logging::new_progress_bar;
use senna::embed_common::*;

use candle_core::{Device, Tensor};
use indicatif::ParallelProgressIterator;
use rayon::prelude::*;

/// How many blocks may be in flight, given the device the closure computes on.
///
/// **One off the CPU.** A CUDA device is a single stream behind a single
/// cuBLAS handle, and driving it from several threads at once raced: some runs
/// died with `CUBLAS_STATUS_EXECUTION_FAILED`, others hung, and the same work
/// succeeded every time on one thread. The GPU is already parallel *inside* a
/// block, so the pool only ever paid off for CPU blocks.
///
/// Takes the two facts rather than a `Device` so it can be tested without one.
#[must_use]
pub(crate) fn device_concurrency(is_cpu: bool, threads: usize) -> usize {
    if is_cpu {
        threads.max(1)
    } else {
        1
    }
}

/// Run `f` over `jobs` with at most `max_conc` blocks in flight, collecting
/// the results in job order.
///
/// The one place the device rule above is applied. Every block runner in senna
/// goes through it — [`process_blocks`] here, and the dense `predict` drivers
/// — so "one block at a time off the CPU" is stated once instead of being
/// re-derived wherever a scoring loop is written.
pub(crate) fn map_blocks<T, F>(
    jobs: &[(usize, usize)],
    max_conc: usize,
    f: F,
) -> anyhow::Result<Vec<T>>
where
    T: Send,
    F: Fn((usize, usize)) -> anyhow::Result<T> + Send + Sync,
{
    let bar = new_progress_bar(jobs.len() as u64);
    // Bound before `?`, so an error still clears the bar.
    let out = if max_conc <= 1 {
        // On the calling thread — the one that owns the device's tensors —
        // rather than a one-worker pool, which was pure overhead.
        jobs.iter()
            .map(|&block| {
                let r = f(block);
                bar.inc(1);
                r
            })
            .collect::<anyhow::Result<Vec<_>>>()
    } else if max_conc >= rayon::current_num_threads() {
        jobs.par_iter()
            .progress_with(bar.clone())
            .map(|&block| f(block))
            .collect::<anyhow::Result<Vec<_>>>()
    } else {
        // A pool sized to the cap rather than waves of `par_iter`: the cap has
        // to bound how many blocks hold their dense tensors at once, and a wave
        // stalls on its slowest block before the next starts.
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(max_conc)
            .build()?;
        pool.install(|| {
            jobs.par_iter()
                .progress_with(bar.clone())
                .map(|&block| f(block))
                .collect::<anyhow::Result<Vec<_>>>()
        })
    };
    bar.finish_and_clear();
    out
}

/// Run a block-processing closure over `ntot` items in blocks of `block_size`,
/// dispatching in parallel on CPU or sequentially on GPU.
///
/// The closure receives `(lb, ub)` and must return `(lb, Mat)`.
/// Results are reassembled into a single `ntot × kk` matrix.
pub(crate) fn process_blocks<F>(
    ntot: usize,
    kk: usize,
    block_size: usize,
    dev: &Device,
    eval_block: F,
) -> anyhow::Result<Mat>
where
    F: Fn((usize, usize)) -> anyhow::Result<(usize, Mat)> + Send + Sync,
{
    let max_conc = device_concurrency(dev.is_cpu(), rayon::current_num_threads());
    process_blocks_at(ntot, kk, block_size, max_conc, eval_block)
}

/// [`process_blocks`] with the number of blocks in flight given rather than
/// taken from the device alone.
///
/// A caller whose block holds a dense `[n, D]` working set needs the second
/// ceiling the device rule does not know about: every extra block in flight is
/// another copy of that set, and the encoder chain is a dozen-odd tensors of it.
/// See `predict::dense_block_concurrency`, which computes the cap.
pub(crate) fn process_blocks_at<F>(
    ntot: usize,
    kk: usize,
    block_size: usize,
    max_conc: usize,
    eval_block: F,
) -> anyhow::Result<Mat>
where
    F: Fn((usize, usize)) -> anyhow::Result<(usize, Mat)> + Send + Sync,
{
    let jobs = create_jobs(ntot, 0, Some(block_size));
    let mut chunks: Vec<(usize, Mat)> = map_blocks(&jobs, max_conc, eval_block)?;

    chunks.sort_by_key(|&(lb, _)| lb);

    let mut ret = Mat::zeros(ntot, kk);
    let mut lb = 0;
    for (_, z) in chunks {
        let ub = lb + z.nrows();
        ret.rows_range_mut(lb..ub).copy_from(&z);
        lb = ub;
    }
    Ok(ret)
}

/// Per-cell batch (`Batch`) or pseudobulk-group (`Residual`) id for the
/// block `lb..ub`, selecting the membership axis by `adj_method`.
pub(crate) fn block_membership(
    data_vec: &SparseIoVec,
    adj_method: &AdjMethod,
    lb: usize,
    ub: usize,
) -> anyhow::Result<Vec<usize>> {
    Ok(match adj_method {
        AdjMethod::Batch => data_vec.get_batch_membership(lb..ub),
        AdjMethod::Residual => data_vec.get_group_membership(lb..ub)?,
    })
}

/// Expand a precomputed delta tensor `[B, D]` to `[N, D]` using per-sample
/// batch or group membership indices for the block `lb..ub`.
pub(crate) fn expand_delta_for_block(
    data_vec: &SparseIoVec,
    delta_bd: &Tensor,
    adj_method: &AdjMethod,
    lb: usize,
    ub: usize,
    dev: &Device,
) -> anyhow::Result<Tensor> {
    let membership = block_membership(data_vec, adj_method, lb, ub)?;
    let indices = Tensor::from_iter(membership.into_iter().map(|x| x as u32), dev)?;
    Ok(delta_bd.index_select(&indices, 0)?)
}
