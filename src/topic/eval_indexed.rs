use super::common::{expand_delta_for_block, process_blocks_at};
use senna::embed_common::*;

use candle_core::{Device, Tensor};
use legume_numeric::candle::decoder::masked_etm::ModuleTarget;
use legume_numeric::candle::decoder::EmbeddedNbTopicDecoder;
use legume_numeric::candle::traits::*;
use legume_numeric::candle::vae::masked_topic::{
    decoder_log_theta, dense_module_targets, masked_encode_dense, LatentHead, MaskedDenseInput,
    MaskedLikelihood,
};
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};

/// A sparse `[D_query, n]` block as the dense `[n, D_train]` rows the
/// window-free encoder reads.
///
/// `gene_remap` (`Some(new_to_train)`) sends held-out gene ids to training ids;
/// several query genes can land on one training gene, so the fill ADDS.
pub(crate) fn csc_block_to_dense(
    x_dn: &nalgebra_sparse::CscMatrix<f32>,
    n_train_features: usize,
    gene_remap: Option<&[Option<usize>]>,
    dev: &Device,
) -> anyhow::Result<Tensor> {
    anyhow::ensure!(
        gene_remap.is_some() || x_dn.nrows() <= n_train_features,
        "a {}-gene block cannot be read onto a {n_train_features}-gene training axis without a \
         gene_remap: `None` means the columns are already on that axis",
        x_dn.nrows()
    );
    let n = x_dn.ncols();
    let mut buf = vec![0f32; n * n_train_features];
    for j in 0..n {
        let col = x_dn.col(j);
        for (&r, &v) in col.row_indices().iter().zip(col.values().iter()) {
            let Some(g) = (match gene_remap {
                Some(map) => map.get(r).copied().flatten(),
                None => Some(r),
            }) else {
                continue;
            };
            buf[j * n_train_features + g] += v;
        }
    }
    Ok(Tensor::from_vec(buf, (n, n_train_features), dev)?)
}

/// Live `[n, D]` f32 tensors one **dense** encoder block holds at its peak.
///
/// The window-free read densifies the block to `[n, D]`, and the encoder chain
/// over it is a dozen-odd more of that shape: the Anscombe residual alone
/// (clean, `a`, its row mean, `r`, the column-mean difference, its square, the
/// scaled value and its `tanh`) plus the input, the null, the all-visible mask,
/// the scores, the `-inf` mask, the softmax and the pooling weights. Rounded
/// up, as `predict`'s `NB_CHAIN_TENSORS` is: over-counting only holds fewer
/// blocks in flight, under-counting is the thing that bites.
const ENCODER_CHAIN_TENSORS: usize = 16;

/// Dense encoder blocks in flight, whatever the machine.
///
/// A block's own matmul asks rayon for every core it can see — candle's
/// `get_num_threads()` reads `num_cpus`, not the pool it is running inside — so
/// `B` blocks at once ask for `B × cores` workers, and the elementwise chain
/// around it competes for the same memory bandwidth. Measured on the latent
/// write of a 7943-cell fit at `D = 34008` (64 cores), as summed block wall /
/// blocks in flight: 64 → ~7 s, 39 → 4.9 s, 19 → 4.2 s, **9 → 2.8 s**, 4 → 4.1 s.
/// The optimum is a handful; below it the cores go idle, above it they fight.
///
/// Not a memory number — that ceiling is `dense_block_concurrency`, and both
/// apply.
const DENSE_BLOCKS_IN_FLIGHT: usize = 8;

/// How many blocks may be in flight: the dense-`predict` memory budget
/// (`LEGUME_PREDICT_BUDGET_BYTES` to raise) and [`DENSE_BLOCKS_IN_FLIGHT`].
/// Off the CPU the first is already 1, so neither lets the GPU be driven from
/// several threads.
fn masked_block_concurrency(dev: &Device, n: usize, d: usize) -> usize {
    crate::predict::dense_block_concurrency(
        dev,
        crate::predict::dense_bytes(n, d, ENCODER_CHAIN_TENSORS),
    )
    .clamp(1, DENSE_BLOCKS_IN_FLIGHT)
}

/// Config for [`evaluate_latent_masked`] — encoder-only inference for the
/// masked-imputation embedded topic model (no decoder / no refinement).
pub(crate) struct EvaluateLatentMaskedConfig<'a> {
    pub dev: &'a Device,
    pub adj_method: &'a AdjMethod,
    pub minibatch_size: usize,
    pub feature_mean: &'a [f32],
    /// Latent head to run at inference — must match the head the model was
    /// trained with. The eval path (all genes visible, no decoder) is identical
    /// across heads; only the final simplex/latent map differs. The Gaussian
    /// arm returns the posterior mean `z` (train=false → reparam returns mean).
    pub head: LatentHead,
}

/// Encoder-only latent inference for the masked-imputation topic model.
///
/// Calls the deterministic masked-encoder forward with **all real genes
/// visible** (`visible = value>0`, no masking at inference) — matching the
/// training-time pooling (pads excluded) — and uses no decoder.
pub(crate) fn evaluate_latent_masked(
    data_vec: &SparseIoVec,
    encoder: &legume_numeric::candle::encoder::IndexedEmbeddingEncoder,
    config: &EvaluateLatentMaskedConfig,
    delta: Option<&Tensor>,
    gene_remap: Option<&[Option<usize>]>,
) -> anyhow::Result<Mat> {
    evaluate_latent_masked_blocks(
        data_vec.num_columns(),
        encoder,
        config,
        gene_remap,
        |(lb, ub)| {
            let x_dn = data_vec.read_columns_csc(lb..ub)?;
            let x0_nd = delta
                .map(|delta_bm| {
                    expand_delta_for_block(
                        data_vec,
                        delta_bm,
                        config.adj_method,
                        lb,
                        ub,
                        config.dev,
                    )
                })
                .transpose()?;
            Ok((x_dn, x0_nd))
        },
    )
}

/// [`evaluate_latent_masked`] on **dense rate rows** — the pseudobulk matrix the
/// model was trained on, `[D × P]` with an optional per-row null `[P × D]`.
///
/// This is the diagnostic that separates a collapsed latent from a saturated
/// encoder. Training rows are pseudobulk rates; inference rows are single-cell
/// counts. If the same encoder is multi-topic on the rows it trained on and
/// one-hot on cells, the model is fine and the *input* is out of distribution
/// (see `--poisson-thin`); if it is one-hot on both, training collapsed and the
/// anchor penalty is the lever.
pub(crate) fn evaluate_latent_masked_rows(
    x_dp: &Mat,
    null_pd: Option<&Mat>,
    encoder: &legume_numeric::candle::encoder::IndexedEmbeddingEncoder,
    config: &EvaluateLatentMaskedConfig,
) -> anyhow::Result<Mat> {
    if let Some(n) = null_pd {
        anyhow::ensure!(
            n.nrows() == x_dp.ncols() && n.ncols() == x_dp.nrows(),
            "row null is {}×{}, expected {}×{} (rows × genes)",
            n.nrows(),
            n.ncols(),
            x_dp.ncols(),
            x_dp.nrows()
        );
    }
    evaluate_latent_masked_blocks(x_dp.ncols(), encoder, config, None, |(lb, ub)| {
        let block = x_dp.columns(lb, ub - lb).into_owned();
        let x_dn = nalgebra_sparse::CscMatrix::from(&block);
        let x0_nd = null_pd
            .map(|n| n.rows(lb, ub - lb).into_owned().to_tensor(config.dev))
            .transpose()?;
        Ok((x_dn, x0_nd))
    })
}

/// The shared block loop: `read` yields each block's `[D, n]` counts and its
/// optional `[n, D]` null; everything after that is identical for every source.
fn evaluate_latent_masked_blocks<R>(
    ntot: usize,
    encoder: &legume_numeric::candle::encoder::IndexedEmbeddingEncoder,
    config: &EvaluateLatentMaskedConfig,
    gene_remap: Option<&[Option<usize>]>,
    read: R,
) -> anyhow::Result<Mat>
where
    R: Fn((usize, usize)) -> anyhow::Result<(nalgebra_sparse::CscMatrix<f32>, Option<Tensor>)>
        + Send
        + Sync,
{
    let kk = IndexedEncoderT::dim_latent(encoder);
    let d_train = config.feature_mean.len();
    let mean_1d = Tensor::from_vec(config.feature_mean.to_vec(), (1, d_train), config.dev)?;

    // Where a block's wall clock goes, summed over blocks and reported once.
    // Read by eye, the dense read looks like a densify cost; measured, the
    // densify is 3% of it and the encoder forward is 95%. Summed across the
    // blocks in flight, so these are work, not wall clock.
    let t_read = std::sync::atomic::AtomicU64::new(0);
    let t_block = std::sync::atomic::AtomicU64::new(0);
    let t_fwd = std::sync::atomic::AtomicU64::new(0);
    let t_cpu = std::sync::atomic::AtomicU64::new(0);
    let add = |a: &std::sync::atomic::AtomicU64, t: std::time::Instant| {
        a.fetch_add(
            t.elapsed().as_micros() as u64,
            std::sync::atomic::Ordering::Relaxed,
        );
    };

    let max_conc = masked_block_concurrency(config.dev, config.minibatch_size, d_train);
    let out = process_blocks_at(ntot, kk, config.minibatch_size, max_conc, |block| {
        let (lb, _ub) = block;
        let t0 = std::time::Instant::now();
        let (x_dn, x0_nd) = read(block)?;
        add(&t_read, t0);

        // Encoder-only inference: `train = false`; the latent alone is
        // written out. Nothing is masked at inference — window-free, that means
        // every gene is visible, zeros included, which is the distribution the
        // dense encoder trained on.
        let t1 = std::time::Instant::now();
        let x_nd = csc_block_to_dense(&x_dn, d_train, gene_remap, config.dev)?;
        let visible = Tensor::ones(x_nd.shape(), candle_core::DType::F32, config.dev)?;
        add(&t_block, t1);
        let t2 = std::time::Instant::now();
        let latent_nk = masked_encode_dense(
            encoder,
            config.head,
            &MaskedDenseInput {
                x_nd: &x_nd,
                x0_nd: x0_nd.as_ref(),
                mean_1d: Some(&mean_1d),
                visible_nd: &visible,
            },
            false,
        )?;
        add(&t_fwd, t2);
        let t3 = std::time::Instant::now();
        let z_nk = latent_nk.to_device(&candle_core::Device::Cpu)?;
        let r = Mat::from_tensor(&z_nk)?;
        add(&t_cpu, t3);
        Ok((lb, r))
    });
    let ms = |a: &std::sync::atomic::AtomicU64| {
        a.load(std::sync::atomic::Ordering::Relaxed) as f64 / 1000.0
    };
    log::debug!(
        "masked eval: {} blocks, {max_conc} in flight; summed block wall — read {:.0} ms, \
         densify {:.0} ms, forward {:.0} ms, to host {:.0} ms",
        ntot.div_ceil(config.minibatch_size),
        ms(&t_read),
        ms(&t_block),
        ms(&t_fwd),
        ms(&t_cpu)
    );
    out
}

/// The seeded hold-out draw: `1` on a held-out slot, `0` everywhere else.
///
/// Only a position that carries a count is a candidate — holding out a zero
/// would move the metric without adding information, since the score asks
/// whether the model can put back what was taken away and a zero that stays
/// zero is not an imputation. A zero therefore consumes NO draw.
///
/// `seed` is the already-combined key (`config.seed ^ lb`): the combination is
/// the caller's, so that old evaluation numbers keep their positions.
fn seeded_holdout_mask(values: &[f32], seed: u64, rate: f64) -> Vec<f32> {
    let mut rng = StdRng::seed_from_u64(seed);
    values
        .iter()
        .map(|&v| {
            if v > 0.0 && rng.random::<f64>() < rate {
                1.0
            } else {
                0.0
            }
        })
        .collect()
}

/// Config for [`evaluate_holdout_imputation`].
pub(crate) struct HoldoutEvalConfig<'a> {
    pub dev: &'a Device,
    pub adj_method: &'a AdjMethod,
    pub minibatch_size: usize,
    pub feature_mean: &'a [f32],
    /// Latent head the model was trained with (matches the persisted model).
    pub head: LatentHead,
    /// Per-gene likelihood — must match training for a comparable number.
    pub likelihood: MaskedLikelihood,
    /// Topic smoothing applied to the simplex heads, mirroring training.
    pub topic_smoothing: f64,
    /// Fraction of each cell's observed genes to hold out and score.
    pub mask_fraction: f64,
    /// Seed for the hold-out mask. A fixed seed masks the **same** per-cell
    /// positions across heads, so the comparison is apples-to-apples.
    pub seed: u64,
}

/// Held-out masked-imputation likelihood: for each cell, hide `mask_fraction`
/// of its observed genes, encode from the **visible** genes, impute the hidden
/// ones with the trained NB/multinomial ETM decoder, and return the mean
/// log-likelihood per held-out gene.
///
/// This is the honest generalization metric the per-epoch training likelihood
/// can't provide: the scored positions are never encoder inputs on their own
/// forward pass, and individual cells enter training only via their pseudobulk
/// aggregate — so a model that merely memorizes the training pseudobulks cannot
/// score well here. Runs one no-gradient pass; the decoder's `β = α·ρᵀ` (and
/// its log-partition) are fixed, so they are computed once.
pub(crate) fn evaluate_holdout_imputation(
    data_vec: &SparseIoVec,
    encoder: &legume_numeric::candle::encoder::IndexedEmbeddingEncoder,
    decoder: &EmbeddedNbTopicDecoder,
    config: &HoldoutEvalConfig,
    delta: Option<&Tensor>,
) -> anyhow::Result<f32> {
    let ntot = data_vec.num_columns();
    let full_kd = decoder.full_logits_kd()?;
    let d_train = config.feature_mean.len();
    let mean_1d = Tensor::from_vec(config.feature_mean.to_vec(), (1, d_train), config.dev)?;

    let mut llik_sum = 0f64;
    let mut mask_cnt = 0f64;
    for (lb, ub) in create_jobs(ntot, 0, Some(config.minibatch_size)) {
        let x_dn = data_vec.read_columns_csc(lb..ub)?;
        let x0_nd = delta
            .map(|delta_bm| {
                expand_delta_for_block(data_vec, delta_bm, config.adj_method, lb, ub, config.dev)
            })
            .transpose()?;
        let n = ub - lb;
        let map = decoder.coarsening();

        // Hold out a fraction of each cell's OBSERVED genes. Holding
        // out zeros as well would move the metric without adding
        // information: the score asks whether the model can put back
        // what was taken away, and a zero that stays zero is not an
        // imputation.
        let x_nd = csc_block_to_dense(&x_dn, d_train, None, config.dev)?;
        let observed: Vec<f32> = x_nd.flatten_all()?.to_vec1()?;
        let held = seeded_holdout_mask(&observed, config.seed ^ (lb as u64), config.mask_fraction);
        let vis: Vec<f32> = held.iter().map(|m| 1.0 - m).collect();
        let visible_nd = Tensor::from_vec(vis, (n, d_train), config.dev)?;
        let raw_z = masked_encode_dense(
            encoder,
            config.head,
            &MaskedDenseInput {
                x_nd: &x_nd,
                x0_nd: x0_nd.as_ref(),
                mean_1d: Some(&mean_1d),
                visible_nd: &visible_nd,
            },
            false,
        )?;
        // The same aggregation the trainer uses, so the held-out number
        // is on the training trace's scale.
        let t = dense_module_targets(map, &x_nd, &visible_nd)?;
        // Must match the training-time decoder coupling exactly, or the
        // held-out number is not comparable to the training trace.
        let log_z = decoder_log_theta(raw_z, config.head, config.topic_smoothing)?;

        // Score what the encoder did not see, matching the training law: every
        // module's unseen counts (every gene's, under the identity map) against
        // the module rate scaled by the unseen share, the full library, and
        // `residual` because these are cells, whose counts still carry the batch
        // effect.
        let residual_nm = x0_nd
            .as_ref()
            .map(|x0| map.aggregate_columns(&x0.broadcast_mul(&map.log_share_1d().exp()?)?))
            .transpose()?;
        let target = ModuleTarget {
            values: &t.values_nm,
            visible_counts: &t.visible_counts_nm,
            visible_share: &t.visible_share_nm,
            residual: residual_nm.as_ref(),
            lib: &t.lib_n1,
        };
        let (llik, units) = match config.likelihood {
            MaskedLikelihood::Nb => decoder.score_unseen_modules_nb(&log_z, &target, &full_kd)?,
            MaskedLikelihood::Multinomial => {
                decoder.score_unseen_modules_multinomial(&log_z, &target, &full_kd)?
            }
        };
        llik_sum += f64::from(llik.sum_all()?.to_scalar::<f32>()?);
        mask_cnt += f64::from(units.sum_all()?.to_scalar::<f32>()?);
    }

    Ok(if mask_cnt > 0.0 {
        (llik_sum / mask_cnt) as f32
    } else {
        f32::NAN
    })
}

#[cfg(test)]
#[path = "eval_indexed_tests.rs"]
mod eval_indexed_tests;
