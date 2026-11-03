//! The posterior mode by Adam on the exact marginal likelihood.
//!
//! Every parameter a candle variable, the latents summed out on the grid:
//!
//! ```text
//! log p(x) = Σ_p log Σ_{ℓ,k} π_ℓ (1/n) p(x_p | ℓ, k) + log p₀(θ)
//! ```
//!
//! the levels by the exact recursion ([`super::kinetics`]) as tensor
//! operations, so the gradient runs through it. Paths are whitened,
//! `z_ℓ = C η_ℓ` (`C` fixed: `κ/T` held during the climb); lineage 1's
//! duration is the unit. The priors are [`super::blocks::GenePrior`]'s on the
//! sampler's coordinates, `N(0, I)` on `η`, `N(0, 1)` on `log T`.

use super::blocks::GenePrior;
use super::kernel::GridPrior;
use super::likelihood::Reads;
use super::model::Params;
use crate::rbm_core::adam;
use legume_numeric::candle::candle_core::{DType, Device, Result, Tensor, Var, D};
use legume_numeric::candle::candle_nn::ops::log_softmax;
use legume_numeric::candle::candle_nn::Optimizer;
use std::sync::atomic::{AtomicBool, Ordering};

/// Below this `|x|`, `expm1(x)/x` by its series (the division is lost).
const SMALL: f64 = 1e-4;

/// Pseudobulks' reads on the device, `[P × G]` per track.
pub struct DeviceReads {
    u: Tensor,
    s: Tensor,
    total: f64,
}

pub struct Fit {
    log_beta: Var,
    log_gamma: Var,
    rho: Var,
    base: Var,
    /// `[L × n × H]` whitened paths.
    eta: Var,
    /// `[L − 1]` the other lineages' log durations (lineage 1's is 0).
    log_t: Option<Var>,
    logit_pi: Var,
    /// `[n × n]` the prior's Cholesky factor.
    chol: Tensor,
    prior: GenePrior,
    dims: (usize, usize, usize, usize),
    /// Positions per grid step ([`Params::sub`]).
    sub: usize,
}

fn var(v: Vec<f64>, shape: &[usize], dev: &Device) -> Result<Var> {
    let t = Tensor::from_vec(v.iter().map(|&x| x as f32).collect::<Vec<_>>(), shape, dev)?;
    Var::from_tensor(&t)
}

impl Fit {
    /// Start at `params`, the paths whitened by `prior` (whose `κ/T` is held).
    pub fn from_params(params: &Params, prior: &GridPrior, dev: &Device) -> Result<Self> {
        let (l, g) = (params.path.len(), params.beta.len());
        let n = params.path[0].len();
        let h = params.rho[0].len();
        let mut eta = Vec::with_capacity(l * n * h);
        for path in &params.path {
            let dims: Vec<_> = (0..h)
                .map(|d| prior.whiten(&nalgebra::DVector::from_fn(n, |k, _| path[k][d])))
                .collect();
            for k in 0..n {
                eta.extend(dims.iter().map(|e| e[k]));
            }
        }
        let c = prior.chol();
        let log_t = if l > 1 {
            let t0 = params.t_len[0];
            Some(var(
                params.t_len[1..].iter().map(|t| (t / t0).ln()).collect(),
                &[l - 1],
                dev,
            )?)
        } else {
            None
        };
        Ok(Self {
            log_beta: var(params.beta.iter().map(|b| b.ln()).collect(), &[1, g], dev)?,
            log_gamma: var(params.gamma.iter().map(|y| y.ln()).collect(), &[1, g], dev)?,
            rho: var(params.rho.iter().flatten().copied().collect(), &[g, h], dev)?,
            base: var(params.base.clone(), &[1, g], dev)?,
            eta: var(eta, &[l, n, h], dev)?,
            log_t,
            logit_pi: var(vec![0.0; l], &[1, l], dev)?,
            chol: Tensor::from_vec(c.iter().map(|&x| x as f32).collect::<Vec<_>>(), (n, n), dev)?,
            prior: GenePrior::default(),
            dims: (l, n, g, h),
            sub: params.sub.max(1),
        })
    }

    pub fn reads(&self, reads: &Reads) -> Result<DeviceReads> {
        let dev = self.eta.device();
        let (p, g) = (reads.u.nrows(), reads.u.ncols());
        let host = |m: &nalgebra::DMatrix<f64>| {
            Tensor::from_vec(
                (0..p)
                    .flat_map(|i| (0..g).map(move |j| m[(i, j)] as f32))
                    .collect::<Vec<_>>(),
                (p, g),
                dev,
            )
        };
        let total = (0..p).map(|i| reads.total(i)).sum();
        Ok(DeviceReads {
            u: host(&reads.u)?,
            s: host(&reads.s)?,
            total,
        })
    }

    fn vars(&self) -> Vec<Var> {
        let mut v = vec![
            self.log_beta.clone(),
            self.log_gamma.clone(),
            self.rho.clone(),
            self.base.clone(),
            self.eta.clone(),
            self.logit_pi.clone(),
        ];
        v.extend(self.log_t.clone());
        v
    }

    /// `[L × 1]` every lineage's duration.
    fn t_len(&self) -> Result<Tensor> {
        let zero = Tensor::zeros((1,), DType::F32, self.eta.device())?;
        let log_t = match &self.log_t {
            Some(t) => Tensor::cat(&[&zero, t.as_tensor()], 0)?,
            None => zero,
        };
        log_t.exp()?.unsqueeze(1)
    }

    /// `[L × n × H]` the paths.
    fn paths(&self) -> Result<Tensor> {
        // `chol` holds Cᵀ, so a row-vector product: z[k] = Σ_j C[k, j] η[j].
        let (l, n, h) = (self.dims.0, self.dims.1, self.dims.3);
        let eta = self.eta.as_tensor().transpose(1, 2)?.reshape((l * h, n))?;
        eta.matmul(&self.chol)?.reshape((l, h, n))?.transpose(1, 2)
    }

    /// `([L·n × G], [L·n × G])` the levels.
    pub fn levels(&self) -> Result<(Tensor, Tensor)> {
        let (l, n, g, h) = self.dims;
        let z = self.paths()?.reshape((l * n, h))?;
        let alpha = z
            .matmul(&self.rho.as_tensor().t()?)?
            .broadcast_add(self.base.as_tensor())?
            .exp()?
            .reshape((l, n, g))?;
        let (beta, gamma) = (
            self.log_beta.as_tensor().exp()?,
            self.log_gamma.as_tensor().exp()?,
        );
        let step = (self.t_len()? / (n - 1) as f64)?; // [L × 1]
        let m = self.sub;
        // `e^{−βδ}`, `e^{−γδ}` and `E(δ) = δ e^{−γδ} expm1((γ−β)δ)/((γ−β)δ)`
        // for a step `δ` `[L × 1]`, each `[L × G]`.
        let factors = |delta: &Tensor| -> Result<(Tensor, Tensor, Tensor)> {
            let bh = delta.broadcast_mul(&beta)?;
            let gh = delta.broadcast_mul(&gamma)?;
            let (eb, eg) = (bh.neg()?.exp()?, gh.neg()?.exp()?);
            let x = (&gh - &bh)?;
            let small = x.abs()?.lt(SMALL)?;
            let safe = small.where_cond(&x.ones_like()?, &x)?;
            let ratio =
                small.where_cond(&(x.affine(0.5, 1.0))?, &((safe.exp()? - 1.0)? / &safe)?)?;
            let e = (delta.broadcast_mul(&eg)? * ratio)?;
            Ok((eb, eg, e))
        };
        let full = factors(&step)?;
        let part: Vec<(Tensor, Tensor, Tensor)> = (1..m)
            .map(|j| factors(&(&step * (j as f64 / m as f64))?))
            .collect::<Result<_>>()?;
        let (inv_b, inv_g) = (beta.recip()?, gamma.recip()?);
        // One exact step of the ODE from `(u, s)` by `(eb, eg, e)`, `α = a`.
        let advance = |u: &Tensor,
                       s: &Tensor,
                       a: &Tensor,
                       (eb, eg, e): &(Tensor, Tensor, Tensor)|
         -> Result<(Tensor, Tensor)> {
            let a_b = a.broadcast_mul(&inv_b)?;
            let next_s = ((s * eg)?
                + (a.broadcast_mul(&inv_g)? * (1.0 - eg)?)?
                + ((u.broadcast_mul(&beta)? - a)? * e)?)?;
            let next_u = (&a_b + ((u - &a_b)? * eb)?)?;
            Ok((next_u, next_s))
        };
        let a0 = alpha.narrow(1, 0, 1)?.squeeze(1)?;
        let (mut u, mut s) = (a0.broadcast_mul(&inv_b)?, a0.broadcast_mul(&inv_g)?);
        let (mut us, mut ss) = (Vec::new(), Vec::new());
        for k in 0..n - 1 {
            let a = alpha.narrow(1, k, 1)?.squeeze(1)?;
            us.push(u.clone());
            ss.push(s.clone());
            for f in &part {
                let (uj, sj) = advance(&u, &s, &a, f)?;
                us.push(uj);
                ss.push(sj);
            }
            (u, s) = advance(&u, &s, &a, &full)?;
        }
        us.push(u);
        ss.push(s);
        let positions = (n - 1) * m + 1;
        let stack = |v: &[Tensor]| Tensor::stack(v, 1)?.reshape((l * positions, g));
        Ok((stack(&us)?, stack(&ss)?))
    }

    /// `Σ_p log Σ_{ℓ,k} π_ℓ (1/n) p(x_p | ℓ, k)`, the multinomial (no
    /// coefficient).
    pub fn log_marginal(&self, reads: &DeviceReads) -> Result<Tensor> {
        let (l, n) = (self.dims.0, self.positions());
        let (u, s) = self.levels()?;
        let z = (u.sum_keepdim(1)? + s.sum_keepdim(1)?)?.log()?;
        let lu = u.log()?.broadcast_sub(&z)?;
        let ls = s.log()?.broadcast_sub(&z)?;
        let tab = (reads.u.matmul(&lu.t()?)? + reads.s.matmul(&ls.t()?)?)?; // [P × L·n]
        let log_pi = log_softmax(self.logit_pi.as_tensor(), D::Minus1)?; // [1 × L]
        let prior = (log_pi
            .unsqueeze(2)?
            .broadcast_as((1, l, n))?
            .reshape((1, l * n))?
            - (n as f64).ln())?;
        tab.broadcast_add(&prior)?.log_sum_exp(1)?.sum_all()
    }

    /// The log priors.
    fn log_prior(&self) -> Result<Tensor> {
        let (_, _, _, h) = self.dims;
        let pr = &self.prior;
        let gauss = |x: &Tensor, (m, s): (f64, f64)| -> Result<Tensor> {
            ((x - m)? / s)?.sqr()?.sum_all()? * -0.5
        };
        let (lb, lg) = (self.log_beta.as_tensor(), self.log_gamma.as_tensor());
        let z_bar = self
            .paths()?
            .mean_keepdim(1)?
            .mean_keepdim(0)?
            .reshape((h, 1))?;
        let spliced = ((self.rho.as_tensor().matmul(&z_bar)?.t()? + self.base.as_tensor())? - lg)?;
        let mut total = (gauss(lg, pr.log_gamma)? + gauss(&(lg - lb)?, pr.log_ratio)?)?;
        total = (total + gauss(self.rho.as_tensor(), (0.0, pr.rho_sd))?)?;
        total = (total + gauss(&spliced, pr.log_spliced)?)?;
        total = (total + gauss(self.eta.as_tensor(), (0.0, 1.0))?)?;
        if let Some(t) = &self.log_t {
            total = (total + gauss(t.as_tensor(), (0.0, 1.0))?)?;
        }
        Ok(total)
    }

    pub fn objective(&self, reads: &DeviceReads) -> Result<Tensor> {
        self.log_marginal(reads)? + self.log_prior()?
    }

    /// `steps` of Adam at `lr` on the objective per read; the objective after.
    pub fn adam(&self, reads: &Reads, steps: usize, lr: f64) -> Result<f32> {
        self.adam_until(reads, steps, lr, None)
    }

    /// [`Self::adam`], ended early once `stop` is set (as Ctrl+C sets it),
    /// with a progress bar as senna's other fits show.
    pub fn adam_until(
        &self,
        reads: &Reads,
        steps: usize,
        lr: f64,
        stop: Option<&AtomicBool>,
    ) -> Result<f32> {
        let dev = self.reads(reads)?;
        let mut opt = adam(self.vars(), lr)?;
        let bar = crate::logging::new_progress_bar(steps as u64).with_message("ode-mixture climb");
        for _ in 0..steps {
            if stop.is_some_and(|s| s.load(Ordering::Relaxed)) {
                log::info!("ode-mixture climb: stopped");
                break;
            }
            let loss = (self.objective(&dev)? / -dev.total)?;
            opt.backward_step(&loss)?;
            bar.inc(1);
        }
        bar.finish_and_clear();
        self.objective(&dev)?.to_scalar::<f32>()
    }

    /// `Σ_p log π_ℓ p(x_p | ℓ, k)` with every pseudobulk held at its row
    /// `rows[p] = ℓ n + k`, the multinomial (no coefficient), plus the log
    /// priors.
    pub fn held_objective(&self, reads: &DeviceReads, rows: &Tensor) -> Result<Tensor> {
        let (l, n) = (self.dims.0, self.positions());
        let (u, s) = self.levels()?;
        let z = (u.sum_keepdim(1)? + s.sum_keepdim(1)?)?.log()?;
        let lu = u.log()?.broadcast_sub(&z)?.index_select(rows, 0)?;
        let ls = s.log()?.broadcast_sub(&z)?.index_select(rows, 0)?;
        let ll = ((&reads.u * lu)? + (&reads.s * ls)?)?.sum_all()?;
        let log_pi = log_softmax(self.logit_pi.as_tensor(), D::Minus1)?
            .unsqueeze(2)?
            .broadcast_as((1, l, n))?
            .reshape((l * n,))?
            .index_select(rows, 0)?
            .sum_all()?;
        (ll + log_pi)? + self.log_prior()?
    }

    /// `steps` of Adam at `lr` on [`Self::held_objective`] per read; the
    /// objective after.
    pub fn adam_held(&self, reads: &Reads, rows: &[u32], steps: usize, lr: f64) -> Result<f32> {
        let dev = self.reads(reads)?;
        let rows = Tensor::from_vec(rows.to_vec(), (rows.len(),), self.eta.device())?;
        let mut opt = adam(self.vars(), lr)?;
        for _ in 0..steps {
            let loss = (self.held_objective(&dev, &rows)? / -dev.total)?;
            opt.backward_step(&loss)?;
        }
        self.held_objective(&dev, &rows)?.to_scalar::<f32>()
    }

    /// Every pseudobulk's posterior over lineages `[P][L]` and its mean
    /// time on its lineages, `[P]` (lineage time, `k T_ℓ/(n − 1)`).
    pub fn posterior(&self, reads: &DeviceReads) -> Result<(Vec<Vec<f64>>, Vec<f64>)> {
        let (l, n) = (self.dims.0, self.positions());
        let (u, s) = self.levels()?;
        let z = (u.sum_keepdim(1)? + s.sum_keepdim(1)?)?.log()?;
        let lu = u.log()?.broadcast_sub(&z)?;
        let ls = s.log()?.broadcast_sub(&z)?;
        let tab = (reads.u.matmul(&lu.t()?)? + reads.s.matmul(&ls.t()?)?)?;
        let log_pi = log_softmax(self.logit_pi.as_tensor(), D::Minus1)?;
        let prior = log_pi
            .unsqueeze(2)?
            .broadcast_as((1, l, n))?
            .reshape((1, l * n))?;
        let post = legume_numeric::candle::candle_nn::ops::softmax(
            &tab.broadcast_add(&prior)?,
            D::Minus1,
        )?
        .to_vec2::<f32>()?;
        let t = self.t_len()?.flatten_all()?.to_vec1::<f32>()?;
        let mut lineage = Vec::with_capacity(post.len());
        let mut time = Vec::with_capacity(post.len());
        for row in &post {
            let mut w = vec![0.0; l];
            let mut mean = 0.0;
            for (r, &q) in row.iter().enumerate() {
                let (li, k) = (r / n, r % n);
                w[li] += f64::from(q);
                mean += f64::from(q) * k as f64 * f64::from(t[li]) / (n - 1) as f64;
            }
            lineage.push(w);
            time.push(mean);
        }
        Ok((lineage, time))
    }

    /// Positions per lineage, `(n − 1) sub + 1`.
    pub fn positions(&self) -> usize {
        (self.dims.1 - 1) * self.sub + 1
    }

    /// The parameters on the host.
    pub fn params(&self) -> Result<Params> {
        let (l, n, _, _) = self.dims;
        let row = |v: &Var| -> Result<Vec<f64>> {
            Ok(v.as_tensor()
                .flatten_all()?
                .to_vec1::<f32>()?
                .iter()
                .map(|&x| f64::from(x))
                .collect())
        };
        let paths = self.paths()?.to_vec3::<f32>()?;
        let t = self.t_len()?.flatten_all()?.to_vec1::<f32>()?;
        Ok(Params {
            beta: row(&self.log_beta)?.iter().map(|x| x.exp()).collect(),
            gamma: row(&self.log_gamma)?.iter().map(|x| x.exp()).collect(),
            rho: self
                .rho
                .as_tensor()
                .to_vec2::<f32>()?
                .iter()
                .map(|r| r.iter().map(|&x| f64::from(x)).collect())
                .collect(),
            base: row(&self.base)?,
            t_len: t.iter().map(|&x| f64::from(x)).collect(),
            path: (0..l)
                .map(|i| {
                    (0..n)
                        .map(|k| paths[i][k].iter().map(|&x| f64::from(x)).collect())
                        .collect()
                })
                .collect(),
            sub: self.sub,
        })
    }
}
