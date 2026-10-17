//! Persistent contrastive divergence (Tieleman, *ICML* (2008)) on pseudobulk
//! samples drawn at every block's depth `D_m`. The loss is the gap between
//! the data's and the fantasies' mean free energy per read; candle's backward
//! gives its gradient through `W = ρ αᵀ`, and Adam takes the step.

use super::blocks::Blocks;
use super::hidden::Hidden;
use super::machine::{Chains, Machine};
use super::weights::Weights;
use crate::rbm_core::{adam, multinomial};
use legume_numeric::candle::candle_core::{DType, Device, Result, Tensor, Var};
use legume_numeric::candle::candle_nn::Optimizer;
use log::info;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::SeedableRng;
use rayon::prelude::*;
use std::sync::atomic::{AtomicBool, Ordering};

/// One level's pseudobulks: every row's shares within each block
/// (`[P × F]` row-major, machine order; a block with no reads is all zero).
pub struct Level {
    pub shares: Vec<f32>,
    pub rows: usize,
}

impl Level {
    /// Rows of rates `[F × P]` (column-major `[D × P]` of the data's row
    /// order) as shares within every block, in the machine's order.
    pub fn from_rates(rates_fp: &nalgebra::DMatrix<f32>, blocks: &Blocks) -> Self {
        let (f, p) = (blocks.features(), rates_fp.ncols());
        let mut shares = vec![0f32; p * f];
        shares.par_chunks_mut(f).enumerate().for_each(|(j, row)| {
            let col = rates_fp.column(j);
            for r in &blocks.ranges {
                let total: f32 = r.clone().map(|i| col[blocks.perm[i]].max(0.0)).sum();
                if total > 0.0 {
                    for i in r.clone() {
                        row[i] = col[blocks.perm[i]].max(0.0) / total;
                    }
                }
            }
        });
        Self { shares, rows: p }
    }

    /// Every row drawn at `depth[m]` reads in block `m` (none where it has
    /// no share), `[P × F]` row-major.
    pub fn draw(&self, blocks: &Blocks, depth: &[f32], seed: u64) -> Vec<f32> {
        let f = blocks.features();
        let mut out = vec![0f32; self.shares.len()];
        out.par_chunks_mut(f)
            .zip(self.shares.par_chunks(f))
            .enumerate()
            .for_each(|(j, (o, p))| {
                let mut rng = StdRng::seed_from_u64(seed ^ (j as u64).wrapping_mul(0x9E37_79B9));
                for (m, r) in blocks.ranges.iter().enumerate() {
                    if p[r.clone()].iter().any(|&x| x > 0.0) {
                        multinomial(
                            depth[m].round() as u64,
                            &p[r.clone()],
                            &mut o[r.clone()],
                            &mut rng,
                        );
                    }
                }
            });
        out
    }

    /// The rows' expected reads at `depth`, `[P × F]` row-major.
    pub fn expected(&self, blocks: &Blocks, depth: &[f32]) -> Vec<f32> {
        let mut out = self.shares.clone();
        for row in out.chunks_mut(blocks.features()) {
            at_depth(row, blocks, depth);
        }
        out
    }
}

/// One row of shares within every block scaled to `depth[m]` reads in block `m`.
fn at_depth(row: &mut [f32], blocks: &Blocks, depth: &[f32]) {
    for (m, r) in blocks.ranges.iter().enumerate() {
        row[r.clone()].iter_mut().for_each(|x| *x *= depth[m]);
    }
}

pub struct TrainConfig<'a> {
    pub epochs: usize,
    pub batch: usize,
    pub chains: usize,
    pub sweeps: usize,
    pub lr: f64,
    /// The LoRA shared factor's rate over `lr` (LoRA+).
    pub lora_lr_ratio: f64,
    /// Ridge per given row on the LoRA residual.
    pub lora_ridge: f64,
    /// L2 on the embeddings `ρ` and `α`, on their rows' mean square norm.
    pub embedding_l2: f64,
    /// The data term at a fresh draw of `D` reads per pseudobulk instead of
    /// its expected reads (Hinton's advice: probabilities, not samples,
    /// wherever the statistics allow).
    pub draw_data: bool,
    /// Hinton's sparsity target: the mean activation `p` every unit is held
    /// near, by a cross-entropy at `sparsity_cost` against a running estimate.
    pub sparsity_target: f64,
    pub sparsity_cost: f64,
    /// Feature modules: `bge`'s load-balance prior and residual ridge, and the
    /// epochs the membership is held at its start.
    pub module_balance: f64,
    pub module_ridge: f64,
    pub module_warmup: usize,
    /// Share of the finest pseudobulks held out of training to watch the
    /// free energy for overfitting.
    pub holdout: f64,
    pub seed: u64,
    /// Set to stop after the current minibatch.
    pub stop: &'a AtomicBool,
}

/// The fitted machine's variables.
pub struct Fitted {
    pub weights: Weights,
    pub b: Var,
    pub c: Var,
    /// The centres `μ` `[1 × F]` and `λ` `[1 × K]`.
    pub mu: Tensor,
    pub lambda: Tensor,
    pub hidden: Hidden,
    pub depth: Vec<f32>,
    pub trace: Trace,
}

impl Fitted {
    pub fn machine(&self, blocks: &Blocks) -> Result<Machine> {
        Machine::new(
            self.b.as_tensor().clone(),
            self.c.as_tensor().clone(),
            self.weights.w(&blocks.ranges)?,
            self.hidden,
            blocks.ranges.clone(),
            self.depth.clone(),
        )
        .map(|m| m.centred(self.mu.clone(), self.lambda.clone()))
    }
}

/// Per epoch: the free-energy gap per read, the data's reconstruction per
/// read, the held-out pseudobulks' mean free energy per read minus the
/// training ones' (rising: overfitting), and the units in use.
#[derive(Default)]
pub struct Trace {
    pub gap: Vec<f32>,
    pub reconstruction: Vec<f32>,
    pub heldout: Vec<f32>,
    pub live_units: Vec<f32>,
}

/// The module layer's learning rate over the rest's.
const MODULE_LR_RATIO: f64 = 0.1;

/// The sliding average's weight on the past for the sparsity estimate.
const SPARSITY_DECAY: f64 = 0.9;
/// A unit is in use while its activation varies by more than this across
/// pseudobulks (on the unit's own 0..1 scale).
const LIVE_SD: f32 = 0.01;

/// Every feature's mean share within its block over the finest level's
/// training rows (`out` marks the held-out ones).
fn mean_shares(levels: &[Level], blocks: &Blocks, out: &[bool]) -> Vec<f32> {
    let f = blocks.features();
    let finest = levels.last().expect("a level");
    let mut mean = vec![0f64; f];
    let mut n = 0usize;
    for (row, _) in finest.shares.chunks(f).zip(out).filter(|(_, &o)| !o) {
        for (m, &x) in mean.iter_mut().zip(row) {
            *m += f64::from(x);
        }
        n += 1;
    }
    mean.iter().map(|&m| (m / n.max(1) as f64) as f32).collect()
}

/// The sliding average's weight on the past for `λ`.
const LAMBDA_DECAY: f64 = 0.99;

/// Fit the machine on the levels (coarse to fine).
pub fn fit(
    levels: &[Level],
    blocks: &Blocks,
    weights: Weights,
    hidden: Hidden,
    depth: Vec<f32>,
    cfg: &TrainConfig,
    dev: &Device,
) -> anyhow::Result<Fitted> {
    anyhow::ensure!(!levels.is_empty(), "no pseudobulk level to train on");
    let f = blocks.features();
    let k = weights.alpha[0].dim(0)?;
    let mut rng = StdRng::seed_from_u64(cfg.seed);
    let finest = levels.last().expect("a level");
    let finest_i = levels.len() - 1;

    // Held-out finest pseudobulks, and as many training ones to compare. They
    // are chosen first so nothing below (the biases, the centre, the start of
    // the weights, the chains) sees them.
    let mut idx: Vec<usize> = (0..finest.rows).collect();
    idx.shuffle(&mut rng);
    let n_out = ((finest.rows as f64 * cfg.holdout).round() as usize).min(finest.rows / 2);
    let heldout: Vec<u32> = idx[..n_out].iter().map(|&i| i as u32).collect();
    let probe: Vec<u32> = idx[n_out..2 * n_out].iter().map(|&i| i as u32).collect();
    let mut is_out = vec![false; finest.rows];
    heldout.iter().for_each(|&i| is_out[i as usize] = true);
    let kept: Vec<usize> = (0..finest.rows).filter(|&i| !is_out[i]).collect();

    // `b` at every block's log mean share, so `softmax(b)` starts at the data;
    // `μ` the mean reads at depth `D`.
    let shares = mean_shares(levels, blocks, &is_out);
    let log_shares: Vec<f32> = shares.iter().map(|&p| p.max(1e-8).ln()).collect();
    let mut mu = shares.clone();
    at_depth(&mut mu, blocks, &depth);
    let mu = Tensor::from_vec(mu, (1, f), dev)?;
    let b = Var::from_tensor(&Tensor::from_vec(log_shares, (1, f), dev)?)?;
    let c = Var::zeros((1, k), DType::F32, dev)?;
    let mut lambda = hidden.mean(&c.as_tensor().detach())?;
    // The centred reads of the finest level's training rows start the weights.
    let finest_rows = finest.expected(blocks, &depth);
    {
        let mu_host: Vec<f32> = mu.flatten_all()?.to_vec1()?;
        let mut x = Vec::with_capacity(kept.len() * f);
        for &i in &kept {
            x.extend(
                finest_rows[i * f..(i + 1) * f]
                    .iter()
                    .zip(&mu_host)
                    .map(|(v, m)| v - m),
            );
        }
        let x = nalgebra::DMatrix::from_row_slice(kept.len(), f, &x);
        weights.start_from(&x, &blocks.ranges, cfg.seed.wrapping_add(3))?;
    }
    let mut vars = weights.vars();
    vars.push(b.clone());
    vars.push(c.clone());
    let mut opt = adam(vars, cfg.lr)?;
    let mut opt_shared = weights
        .lora_shared()
        .map(|v| adam(vec![v], cfg.lr * cfg.lora_lr_ratio))
        .transpose()?;
    let module_vars = weights.module_vars();
    let mut opt_modules = (!module_vars.is_empty())
        .then(|| adam(module_vars, cfg.lr * MODULE_LR_RATIO))
        .transpose()?;
    let template = Machine::new(
        b.as_tensor().clone(),
        c.as_tensor().clone(),
        weights.w(&blocks.ranges)?,
        hidden,
        blocks.ranges.clone(),
        depth.clone(),
    )?;
    let machine = |w: Tensor, lambda: &Tensor| {
        template
            .with(b.as_tensor().clone(), c.as_tensor().clone(), w)
            .centred(mu.clone(), lambda.clone())
    };
    let n_given = match weights.given.as_ref() {
        Some((_, mask)) => f64::from(mask.sum_all()?.to_scalar::<f32>()?),
        None => 0.0,
    };

    let start = finest.draw(blocks, &depth, cfg.seed.wrapping_add(1));
    let x0: Vec<f32> = (0..cfg.chains)
        .flat_map(|j| {
            let i = kept[j % kept.len().max(1)];
            start[i * f..(i + 1) * f].to_vec()
        })
        .collect();
    let mut chains = Chains::new(x0, f, &blocks.ranges, cfg.seed.wrapping_add(2), dev)?;
    let mut trace = Trace::default();
    let finest_x = Tensor::from_vec(finest_rows, (finest.rows, f), dev)?;
    // Without draws every level's reads are the same every epoch.
    let fixed: Vec<Option<Tensor>> = levels
        .iter()
        .enumerate()
        .map(|(li, lv)| -> Result<Option<Tensor>> {
            match (cfg.draw_data, li == finest_i) {
                (true, _) => Ok(None),
                (false, true) => Ok(Some(finest_x.clone())),
                (false, false) => {
                    Tensor::from_vec(lv.expected(blocks, &depth), (lv.rows, f), dev).map(Some)
                }
            }
        })
        .collect::<Result<_>>()?;
    let pick = |ids: &[u32]| -> Result<Option<Tensor>> {
        if ids.is_empty() {
            return Ok(None);
        }
        finest_x
            .index_select(&Tensor::from_slice(ids, ids.len(), dev)?, 0)
            .map(Some)
    };
    let (heldout_x, probe_x) = (pick(&heldout)?, pick(&probe)?);
    let unit_scale = hidden.unit_scale();
    let mut q = Tensor::full(cfg.sparsity_target as f32, (1, k), dev)?;

    let bar = crate::logging::new_progress_bar(cfg.epochs as u64).with_message("rbm");
    'epochs: for epoch in 0..cfg.epochs {
        bar.inc(1);
        weights.set_frozen(epoch < cfg.module_warmup);
        // Summed on the device and read once an epoch.
        let mut gap = Tensor::zeros((), DType::F32, dev)?;
        let mut recon = gap.clone();
        let mut steps = 0usize;
        for (li, lv) in levels.iter().enumerate() {
            let seed = cfg
                .seed
                .wrapping_add(1_000_003 * (epoch as u64 + 1))
                .wrapping_add(li as u64);
            let all = match &fixed[li] {
                Some(t) => t.clone(),
                None => Tensor::from_vec(lv.draw(blocks, &depth, seed), (lv.rows, f), dev)?,
            };
            let mut order: Vec<u32> = (0..lv.rows as u32)
                .filter(|&i| li != finest_i || !is_out[i as usize])
                .collect();
            order.shuffle(&mut rng);
            for chunk in order.chunks(cfg.batch) {
                if cfg.stop.load(Ordering::Relaxed) {
                    info!("rbm: stopping at epoch {epoch}");
                    break 'epochs;
                }
                let batch = all.index_select(&Tensor::from_slice(chunk, chunk.len(), dev)?, 0)?;
                let m = machine(weights.w(&blocks.ranges)?, &lambda);
                let fantasy = chains.reads().clone();
                let (free_batch, mean_batch) = m.free_energy_and_mean(&batch)?;
                let mut loss =
                    (free_batch.mean_all()? - m.free_energy_per_read(&fantasy)?.mean_all()?)?;
                gap = (gap + loss.detach())?;
                loss = (loss
                    + weights.penalty(cfg.embedding_l2, cfg.module_balance, cfg.module_ridge)?)?;
                if cfg.sparsity_cost > 0.0 {
                    // Cross-entropy of the target against the running estimate,
                    // the current minibatch's share live (Hinton's recipe).
                    let now = (mean_batch.mean_keepdim(0)? / unit_scale)?;
                    let est = ((q.detach() * SPARSITY_DECAY)? + (now * (1.0 - SPARSITY_DECAY))?)?
                        .clamp(1e-6, 1.0 - 1e-6)?;
                    let p = cfg.sparsity_target;
                    let ce = ((est.log()? * -p)? - (est.affine(-1.0, 1.0)?.log()? * (1.0 - p))?)?;
                    loss = (loss + (ce.sum_all()? * cfg.sparsity_cost)?)?;
                    q = est.detach();
                }
                if let Some(pen) = weights.lora_penalty()? {
                    loss = (loss + (pen * (cfg.lora_ridge / n_given.max(1.0)))?)?;
                }
                let grads = loss.backward()?;
                opt.step(&grads)?;
                if let Some(o) = opt_shared.as_mut() {
                    o.step(&grads)?;
                }
                if let Some(o) = opt_modules.as_mut() {
                    o.step(&grads)?;
                }
                // Move the hidden centre toward the data's mean activation and
                // the visible bias with it (`b += W Δλ`), so the machine itself
                // is unchanged on rows at the training depth: centring is a
                // reparametrisation, not a step of its own.
                let w_now = weights.w(&blocks.ranges)?.detach();
                // `E[h | v]` does not depend on `λ` or `b`, so the batch's
                // posterior serves both the centre and the reconstruction.
                let h = machine(w_now.clone(), &lambda)
                    .detach()
                    .posterior_mean(&batch)?;
                let seen = h.mean_keepdim(0)?;
                let moved = ((&lambda * LAMBDA_DECAY)? + (seen * (1.0 - LAMBDA_DECAY))?)?;
                b.set(&(b.as_tensor() + (&moved - &lambda)?.matmul(&w_now.t()?)?)?)?;
                lambda = moved;
                let after = machine(w_now, &lambda).detach();
                recon = (recon + after.reconstruction_given(&batch, &h)?.mean_all()?)?;
                steps += 1;
                for _ in 0..cfg.sweeps {
                    chains.sweep(&after)?;
                }
            }
        }
        let steps = steps.max(1) as f64;
        let gap = f64::from(gap.to_scalar::<f32>()?);
        let recon = f64::from(recon.to_scalar::<f32>()?);
        let (gap, recon) = (gap / steps, recon / steps);
        bar.set_message(format!("rbm gap={gap:.4} recon={recon:.4}"));
        trace.gap.push(gap as f32);
        trace.reconstruction.push(recon as f32);
        let now = machine(weights.w(&blocks.ranges)?, &lambda).detach();
        let per_read =
            |x: &Tensor| -> Result<f32> { now.free_energy_per_read(x)?.mean_all()?.to_scalar() };
        let heldout_gap = match (heldout_x.as_ref(), probe_x.as_ref()) {
            (Some(h), Some(t)) => per_read(h)? - per_read(t)?,
            _ => f32::NAN,
        };
        let act = (now.posterior_mean(&finest_x)? / unit_scale)?;
        let sd: Vec<f32> = act
            .broadcast_sub(&act.mean_keepdim(0)?)?
            .sqr()?
            .mean(0)?
            .sqrt()?
            .to_vec1()?;
        let live = sd.iter().filter(|&&v| v > LIVE_SD).count();
        trace.heldout.push(heldout_gap);
        trace.live_units.push(live as f32);
        if epoch % 10 == 0 || epoch + 1 == cfg.epochs {
            info!(
                "rbm epoch {epoch}: free-energy gap per read {:.5}, reconstruction per read {:.4}, \
                 held-out − training free energy per read {heldout_gap:.5}, {live}/{k} units in use",
                gap / steps,
                recon / steps
            );
        }
    }
    bar.finish_and_clear();
    Ok(Fitted {
        weights,
        b,
        c,
        mu,
        lambda,
        hidden,
        depth,
        trace,
    })
}

#[cfg(test)]
#[path = "tests/train.rs"]
mod tests;
