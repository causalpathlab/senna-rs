//! Dynamic feature and unit embedding: `senna gem`'s one softmax over every
//! (gene, track) row, where a unit's state is a function of its time.
//!
//! Every module `m` has a direction `w_m` in the embedding space, and a unit
//! at time `τ` sits where the modules' time courses put it:
//!
//! ```text
//! θ^k(τ) = Σ_m φ^k_m(τ) w_m = W φ^k(τ),    φ^k_m(τ) = log y^k_m(τ),  y^s = s_m, y^u = u_m
//! score(g, k | p) = ⟨θ^k(τ_p), ρ_g⟩ + b_g + [k=u](b^u_g + κ_p)
//! P(g, k | p) = softmax over all 2G rows
//! ```
//!
//! - `θ^s(τ_p)` is the unit's dynamic embedding, a point on one curve; its
//!   velocity `dθ^s/dτ = W (β u/s − γ)` is exact, and its unspliced state
//!   `θ^u(τ)` is displaced from it by `W (log u − log s)`.
//! - `ρ_g` is the gene's, in the same space as the units and the module
//!   directions; `⟨w_m, ρ_g⟩` is how the gene follows module `m`, up or down.
//! - With flat kinetics the unspliced-minus-spliced score is one number per
//!   gene: gem's static track offset.
//!
//! There is no state apart from time, so nothing competes with it. One curve
//! cannot branch; that is for later. Times move by Adam and by a per-unit grid
//! search, as in [`super::fit`]; the direction is the full model's call
//! ([`fit_dynamic`]).

use super::fit::{
    fit_oriented, logit, reversed, FitConfig, FreeKinetics, PbCounts, PreparedPrior, TauPrior,
    GRID_ELEMS,
};
use super::kinetics::{curves, spliced_log_rate, Kinetics, ModuleKinetics};
use legume_numeric::candle::candle_core::{DType, Device, Result as CResult, Tensor, Var, D};
use legume_numeric::candle::candle_nn::ops::sigmoid;
use legume_numeric::candle::candle_nn::{AdamW, Optimizer, ParamsAdamW};
use legume_numeric::matrix::traits::ConvertMatOps;
use nalgebra::DMatrix;

/// Where the joint fit starts: a stage-1 fit's times and kinetics, and the
/// base fit's gene rows, biases and unit states.
pub struct JointInit<'a> {
    pub tau: &'a [f32],
    pub modules: &'a [ModuleKinetics],
    /// `[G × H]`, `[G]`.
    pub rho: &'a DMatrix<f32>,
    pub bias: &'a [f32],
    /// `[P × H]` the units' base states: the module directions start where
    /// `W φ^s(τ_p)` best matches them.
    pub theta: &'a DMatrix<f32>,
}

#[derive(Clone, Debug)]
pub struct JointConfig {
    pub rounds: usize,
    pub adam_steps: usize,
    pub learning_rate: f64,
    /// Grid points over `τ ∈ (0, 1)`.
    pub grid: usize,
}

impl Default for JointConfig {
    fn default() -> Self {
        Self {
            rounds: 30,
            adam_steps: 50,
            learning_rate: 0.05,
            grid: 64,
        }
    }
}

pub struct JointResult {
    pub tau: Vec<f32>,
    pub kappa: Vec<f32>,
    /// `[P × H]` every unit's state `θ^s(τ_p)` and velocity `dθ^s/dτ`.
    pub theta: DMatrix<f32>,
    pub velocity: DMatrix<f32>,
    /// `[G × H]` gene rows and `[M × H]` module directions.
    pub rho: DMatrix<f32>,
    pub directions: DMatrix<f32>,
    pub bias: Vec<f32>,
    pub bias_u: Vec<f32>,
    pub modules: Vec<ModuleKinetics>,
    /// Loss per read, prior included, after every round.
    pub loss: Vec<f64>,
}

struct Joint<'a> {
    cfg: &'a JointConfig,
    xs: Tensor,
    xu: Tensor,
    /// `[P × 1]` reads per unit.
    n: Tensor,
    reads: f64,
    free: FreeKinetics,
    /// `[M × H]` module directions, `[G × H]` gene rows.
    w: Var,
    rho: Var,
    /// `[1 × G]`.
    b: Var,
    bu: Var,
    /// `[P × 1]`.
    kappa: Var,
    /// `[P]`, `τ = σ(z)`.
    z: Var,
    prior: PreparedPrior,
    loss: Vec<f64>,
}

/// Log-sum-exp over the last dimension, kept.
fn logsumexp(x: &Tensor) -> CResult<Tensor> {
    let m = x.max_keepdim(D::Minus1)?.detach();
    x.broadcast_sub(&m)?
        .exp()?
        .sum_keepdim(D::Minus1)?
        .log()?
        .broadcast_add(&m)
}

impl<'a> Joint<'a> {
    fn new(
        counts: &PbCounts,
        init: &JointInit,
        prior: &TauPrior,
        cfg: &'a JointConfig,
    ) -> anyhow::Result<Self> {
        let xs = counts.spliced.to_dtype(DType::F32)?;
        let xu = counts.unspliced.to_dtype(DType::F32)?;
        let dev = xs.device().clone();
        let (p, g) = xs.dims2()?;
        let m = counts.n_modules;
        anyhow::ensure!(
            init.tau.len() == p,
            "{} times for {p} units",
            init.tau.len()
        );
        anyhow::ensure!(
            init.rho.nrows() == g && init.bias.len() == g,
            "ρ, b not over {g} genes"
        );
        anyhow::ensure!(init.theta.nrows() == p, "θ not over {p} units");
        anyhow::ensure!(
            init.modules.len() == m,
            "{} modules, {m} expected",
            init.modules.len()
        );
        let free = FreeKinetics::new(init.modules, &dev)?;
        let n = (&xs + &xu)?.sum_keepdim(1)?;
        let reads = f64::from(n.sum_all()?.to_scalar::<f32>()?).max(1.0);
        let tot_s = (xs.sum_keepdim(0)? + 0.5)?;
        let tot_u = (xu.sum_keepdim(0)? + 0.5)?;
        let w = start_directions(&free.kinetics()?, init.tau, init.theta, &dev)?;
        let z: Vec<f32> = init
            .tau
            .iter()
            .map(|&t| logit(f64::from(t)) as f32)
            .collect();
        Ok(Self {
            cfg,
            n,
            reads,
            free,
            w: Var::from_tensor(&w)?,
            rho: Var::from_tensor(&init.rho.to_tensor(&dev)?.contiguous()?)?,
            b: Var::from_tensor(&Tensor::from_slice(init.bias, (1, g), &dev)?)?,
            // The gene's pooled log ratio of unspliced to spliced reads.
            bu: Var::from_tensor(&(tot_u.log()? - tot_s.log()?)?)?,
            kappa: Var::zeros((p, 1), DType::F32, &dev)?,
            z: Var::from_tensor(&Tensor::from_vec(z, p, &dev)?)?,
            prior: PreparedPrior::new(prior, p, &dev)?,
            loss: Vec::new(),
            xs,
            xu,
        })
    }

    fn tau(&self) -> CResult<Tensor> {
        sigmoid(self.z.as_tensor())
    }

    /// Both tracks' gene scores without biases at times `tau` `[T]`:
    /// `⟨W φ^s(τ), ρ_g⟩` and `⟨W φ^u(τ), ρ_g⟩`, each `[T × G]`.
    fn time_scores(&self, tau: &Tensor) -> CResult<(Tensor, Tensor)> {
        let (u, s) = curves(&self.free.kinetics()?, tau)?;
        // [M × G]: how every gene follows every module.
        let follow = self.w.as_tensor().matmul(&self.rho.as_tensor().t()?)?;
        Ok((s.log()?.matmul(&follow)?, u.log()?.matmul(&follow)?))
    }

    /// Per unit, the negative log-likelihood of its reads, `[P]`.
    fn nll(&self) -> CResult<Tensor> {
        let (ts, tu) = self.time_scores(&self.tau()?)?;
        let ss = ts.broadcast_add(self.b.as_tensor())?;
        let su = tu
            .broadcast_add(&(self.b.as_tensor() + self.bu.as_tensor())?)?
            .broadcast_add(self.kappa.as_tensor())?;
        let lse = logsumexp(&Tensor::cat(&[&ss, &su], 1)?)?;
        let fit = ((&self.xs * &ss)?.sum(1)? + (&self.xu * &su)?.sum(1)?)?;
        self.n.broadcast_mul(&lse)?.squeeze(1)? - fit
    }

    /// The objective in nats: the reads and the prior on times.
    fn objective(&self) -> CResult<Tensor> {
        self.nll()?.sum_all()? + self.prior.nats(&self.tau()?)?
    }

    /// Move every unit to the grid time that fits its reads and the prior
    /// best, its neighbours where they are, when that beats its own.
    fn grid_step(&mut self) -> CResult<()> {
        let kg = self.cfg.grid;
        let (p, g) = self.xs.dims2()?;
        let dev = self.xs.device().clone();
        let grid: Vec<f32> = (0..kg).map(|i| (i as f32 + 0.5) / kg as f32).collect();
        let (ts, tu) = self.time_scores(&Tensor::from_vec(grid.clone(), kg, &dev)?)?;
        // Every unit shares the grid's states; only κ_p differs, on the
        // unspliced rows.
        let ss = ts
            .broadcast_add(self.b.as_tensor())?
            .detach()
            .unsqueeze(0)?; // [1 × K × G]
        let su = tu
            .broadcast_add(&(self.b.as_tensor() + self.bu.as_tensor())?)?
            .detach()
            .unsqueeze(0)?;
        let now = self.nll()?.detach().to_vec1::<f32>()?;
        let mut tau = self.tau()?.to_vec1::<f32>()?;
        let current = tau.clone();
        let chunk = (GRID_ELEMS / (kg * 2 * g)).max(1);
        for start in (0..p).step_by(chunk) {
            let w = chunk.min(p - start);
            let kappa = self
                .kappa
                .as_tensor()
                .narrow(0, start, w)?
                .unsqueeze(2)?
                .detach(); // [w × 1 × 1]
            let su_w = su.broadcast_add(&kappa)?; // [w × K × G]
            let ss_w = ss.broadcast_as((w, kg, g))?;
            let lse = logsumexp(&Tensor::cat(&[&ss_w, &su_w], 2)?)?.squeeze(2)?; // [w × K]
            let xs = self.xs.narrow(0, start, w)?.unsqueeze(1)?;
            let xu = self.xu.narrow(0, start, w)?.unsqueeze(1)?;
            let fit = (ss_w.broadcast_mul(&xs)?.sum(2)? + su_w.broadcast_mul(&xu)?.sum(2)?)?;
            let nll = (self.n.narrow(0, start, w)?.broadcast_mul(&lse)? - fit)?.to_vec2::<f32>()?;
            for (i, row) in nll.iter().enumerate() {
                let pb = start + i;
                let score = |k: usize| row[k] + self.prior.at(pb, grid[k], &current);
                let best = (0..kg)
                    .min_by(|&x, &y| score(x).total_cmp(&score(y)))
                    .expect("grid");
                if score(best) < now[pb] + self.prior.at(pb, current[pb], &current) - 1e-4 {
                    tau[pb] = grid[best];
                }
            }
        }
        let z: Vec<f32> = tau.iter().map(|&t| logit(f64::from(t)) as f32).collect();
        self.z.set(&Tensor::from_vec(z, p, &dev)?)
    }

    fn rounds(&mut self) -> CResult<()> {
        let mut vars = self.free.vars();
        vars.extend([
            self.w.clone(),
            self.rho.clone(),
            self.b.clone(),
            self.bu.clone(),
            self.kappa.clone(),
            self.z.clone(),
        ]);
        let mut adam = AdamW::new(
            vars,
            ParamsAdamW {
                lr: self.cfg.learning_rate,
                weight_decay: 0.0,
                ..Default::default()
            },
        )?;
        for _ in 0..self.cfg.rounds {
            self.grid_step()?;
            for _ in 0..self.cfg.adam_steps {
                adam.backward_step(&(self.objective()? / self.reads)?)?;
            }
            let l = f64::from(self.objective()?.to_scalar::<f32>()?) / self.reads;
            self.loss.push(l);
        }
        Ok(())
    }

    fn finish(self) -> anyhow::Result<JointResult> {
        let row =
            |v: &Var| -> anyhow::Result<Vec<f32>> { Ok(v.as_tensor().flatten_all()?.to_vec1()?) };
        let k = self.free.kinetics()?;
        let tau = self.tau()?;
        let (_, s) = curves(&k, &tau)?;
        let w = self.w.as_tensor();
        Ok(JointResult {
            tau: tau.to_vec1()?,
            kappa: row(&self.kappa)?,
            theta: DMatrix::from_tensor(&s.log()?.matmul(w)?)?,
            velocity: DMatrix::from_tensor(&spliced_log_rate(&k, &tau)?.matmul(w)?)?,
            rho: DMatrix::from_tensor(self.rho.as_tensor())?,
            directions: DMatrix::from_tensor(w)?,
            bias: row(&self.b)?,
            bias_u: row(&self.bu)?,
            modules: k.to_modules()?,
            loss: self.loss,
        })
    }
}

/// `[M × H]` module directions whose curve best matches the base states:
/// least squares of `θ_p` on `[1, log s(τ_p)]`, the intercept dropped (the
/// biases carry it).
fn start_directions(
    k: &Kinetics,
    tau: &[f32],
    theta: &DMatrix<f32>,
    dev: &Device,
) -> anyhow::Result<Tensor> {
    let p = tau.len();
    let (_, s) = curves(k, &Tensor::from_slice(tau, p, dev)?)?;
    let phi = DMatrix::<f32>::from_tensor(&s.log()?)?.map(f64::from);
    let m = phi.ncols();
    let x = DMatrix::<f64>::from_fn(p, m + 1, |i, j| if j == 0 { 1.0 } else { phi[(i, j - 1)] });
    let y = theta.map(f64::from);
    let coef = x
        .svd(true, true)
        .solve(&y, 1e-9)
        .map_err(|e| anyhow::anyhow!("module directions: {e}"))?;
    let w = coef.rows(1, m).map(|v| v as f32);
    Ok(w.to_tensor(dev)?.contiguous()?)
}

/// The joint fit from a stage-1 solution.
pub fn fit_joint(
    counts: &PbCounts,
    init: &JointInit,
    prior: &TauPrior,
    cfg: &JointConfig,
) -> anyhow::Result<JointResult> {
    let mut j = Joint::new(counts, init, prior, cfg)?;
    j.rounds()?;
    j.finish()
}

/// The joint fit and its orientation: for the starting order `tau0` and its
/// reverse, a one-orientation stage-1 fit, then the joint fit from it; the
/// lower final loss is kept.
#[allow(clippy::too_many_arguments)]
pub fn fit_dynamic(
    counts: &PbCounts,
    tau0: &[f32],
    rho: &DMatrix<f32>,
    bias: &[f32],
    theta: &DMatrix<f32>,
    prior: &TauPrior,
    stage1: &FitConfig,
    cfg: &JointConfig,
) -> anyhow::Result<(JointResult, [f64; 2], bool)> {
    let (reverse, reverse_prior) = reversed(tau0, prior);
    let mut outs = Vec::with_capacity(2);
    for (start, prior) in [(tau0, prior), (reverse.as_slice(), &reverse_prior)] {
        let s1 = fit_oriented(counts, start, prior, stage1)?;
        let init = JointInit {
            tau: &s1.tau,
            modules: &s1.modules,
            rho,
            bias,
            theta,
        };
        outs.push(fit_joint(counts, &init, prior, cfg)?);
    }
    let last = |r: &JointResult| r.loss.last().copied().unwrap_or(f64::INFINITY);
    let losses = [last(&outs[0]), last(&outs[1])];
    let reversed = losses[1] < losses[0];
    Ok((outs.swap_remove(usize::from(reversed)), losses, reversed))
}

#[cfg(test)]
#[path = "joint/tests.rs"]
mod tests;
