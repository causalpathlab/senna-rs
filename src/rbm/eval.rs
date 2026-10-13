//! Every cell read out exactly: its hidden units' posterior mean, its free
//! energy and its reconstruction per read, with no encoder in between.
//!
//! With batches, a cell's reads are first divided by its batch's `δ` per
//! feature, then every block is scaled back to the cell's own depth: the
//! machine sees the batch-adjusted shares at the depth it was read at.

use super::blocks::Blocks;
use super::machine::Machine;
use crate::topic::common::{expand_delta_for_block, process_blocks};
use legume_numeric::candle::candle_core::{Device, Tensor, D};
use senna::embed_common::*;

/// Per row: `E[h | v]` (`K` columns), and the free energy and the
/// reconstruction, both per read.
pub fn score(m: &Machine, x: &Tensor) -> anyhow::Result<Mat> {
    let mean = m.posterior_mean(x)?;
    let fe = m.free_energy_per_read(x)?.unsqueeze(D::Minus1)?;
    let recon = m.reconstruction(x)?.unsqueeze(D::Minus1)?;
    let all = Tensor::cat(&[mean, fe, recon], 1)?.to_device(&Device::Cpu)?;
    Mat::from_tensor(&all)
}

/// [`score`] over every cell of `data_vec`. `delta` is the batch null
/// `[B × F]` in the machine's feature order.
pub fn score_cells(
    data_vec: &SparseIoVec,
    m: &Machine,
    blocks: &Blocks,
    delta: Option<&Tensor>,
    block_size: usize,
) -> anyhow::Result<Mat> {
    let dev = m.w.device().clone();
    let perm = Tensor::from_iter(blocks.perm.iter().map(|&i| i as u32), &dev)?;
    let width = m.units()? + 2;
    process_blocks(
        data_vec.num_columns(),
        width,
        block_size,
        &dev,
        |(lb, ub)| {
            let x_dn = data_vec.read_columns_csc(lb..ub)?.to_tensor(&dev)?;
            let x = x_dn
                .contiguous()?
                .index_select(&perm, 0)?
                .t()?
                .contiguous()?;
            let x = match delta {
                Some(d) => {
                    let d = expand_delta_for_block(data_vec, d, &AdjMethod::Batch, lb, ub, &dev)?;
                    m.keep_depth(&(&x / d.maximum(1e-8)?)?, &x)?
                }
                None => x,
            };
            Ok((lb, score(m, &x)?))
        },
    )
}

/// [`score`] over `rows` (`[P × F]` row-major) in chunks of `chunk`.
pub fn score_rows(m: &Machine, rows: &[f32], chunk: usize) -> anyhow::Result<Mat> {
    let f = m.features()?;
    let dev = m.w.device();
    let parts = rows
        .chunks(chunk * f)
        .map(|c| score(m, &Tensor::from_slice(c, (c.len() / f, f), dev)?))
        .collect::<anyhow::Result<Vec<_>>>()?;
    let n: usize = parts.iter().map(|p| p.nrows()).sum();
    let mut out = Mat::zeros(n, m.units()? + 2);
    let mut lb = 0;
    for p in parts {
        out.rows_mut(lb, p.nrows()).copy_from(&p);
        lb += p.nrows();
    }
    Ok(out)
}
