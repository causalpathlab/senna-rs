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
//!
//! With [`Fit::with_share`], every pseudobulk's unspliced share jitters on
//! its own, `σ` fitted: [`super::likelihood::share_term`] added at every
//! point, its mode `ω̂` found with the graph detached (at the mode the
//! gradient through `ω̂` vanishes from the first part; the Laplace
//! determinant's share of it is dropped). The term is a marginal likelihood
//! for `σ` with `σ → 0` the shared model, so `σ` has no prior.
//!
//! With [`Fit::with_trunk`], every other lineage branches off lineage 0 at
//! its own time `t_b` on one clock (every duration the unit):
//! `z_ℓ = z_0 + √2 r(t) w_ℓ`, `w_ℓ = C η_ℓ`, `r = 1 − e^{−((t − t_b)₊/κ)²}`,
//! so before `t_b` the lineages share path, clock and levels, past it they
//! part smoothly and far past it are as far apart as two independent
//! paths; `t_b` uniform on `[0, 1]`.

use super::blocks::GenePrior;
use super::kernel::GridPrior;
use super::likelihood::Reads;
use super::model::{branch_ramp, clamped_exp, off_the_edges, phase, positions, Params, LOG_CLAMP};
use crate::rbm_core::adam;
use legume_numeric::candle::candle_core::{DType, Device, Result, Tensor, Var, D};
use legume_numeric::candle::candle_nn::ops::log_softmax;
use legume_numeric::candle::candle_nn::Optimizer;
use std::sync::atomic::{AtomicBool, Ordering};

/// The departure's misfit scale when a start is put on a trunk
/// ([`Fit::with_trunk`]).
const TRUNK_FIT: f64 = 0.05;

/// `σ` a jitter starts from: small, so the early climb leaves the share's
/// signal to the kinetics; Adam on `log σ` grows it where the reads jitter.
pub const SHARE_START: f64 = 0.01;

/// Newton steps to the jitter's mode, from its Gaussian guess (as
/// [`super::likelihood::share_mode`]).
const SHARE_NEWTON: usize = 8;

/// Below this `|x|`, `x = (γ − β)δ`, `E(δ)` by its series: in f32 the
/// difference of exponentials loses `ε/|x|` of `E` (`6·10⁻⁶` here), the
/// second-order series `x³/24` (`4·10⁻⁸`).
const SMALL: f64 = 1e-2;

/// Pseudobulks' reads on the device, `[P × G]` per track.
pub struct DeviceReads {
    u: Tensor,
    s: Tensor,
    /// `[P × 1]` every pseudobulk's unspliced and all reads.
    n_u: Tensor,
    n: Tensor,
    total: f64,
}

pub struct Fit {
    /// `[1 × G]` each rate's log, read through its bounds
    /// ([`GenePrior::log_rate`], [`Fit::log_rates`]).
    log_beta: Var,
    log_gamma: Var,
    rho: Var,
    base: Var,
    /// `[L × n × H]` whitened paths.
    eta: Var,
    /// `[L − 1]` the other lineages' log durations (lineage 1's is 0).
    log_t: Option<Var>,
    logit_pi: Var,
    /// `[L − 1]` every other lineage's branch time off lineage 0, as a
    /// logit; `None`: lineages apart, each its own duration.
    branch: Option<Var>,
    /// The prior's length scale over the duration, the branch's too.
    kappa: f64,
    /// `[1]` the log sd of the unspliced share's jitter, `None` for none.
    log_share_sd: Option<Var>,
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
        let gene_prior = GenePrior::bounded(params);
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
            log_gamma: var(params.gamma.iter().map(|r| r.ln()).collect(), &[1, g], dev)?,
            rho: var(params.rho.iter().flatten().copied().collect(), &[g, h], dev)?,
            base: var(params.base.clone(), &[1, g], dev)?,
            eta: var(eta, &[l, n, h], dev)?,
            log_t,
            logit_pi: var(vec![0.0; l], &[1, l], dev)?,
            log_share_sd: None,
            branch: None,
            kappa: prior.kappa(),
            chol: Tensor::from_vec(c.iter().map(|&x| x as f32).collect::<Vec<_>>(), (n, n), dev)?,
            prior: gene_prior,
            dims: (l, n, g, h),
            sub: params.sub.max(1),
        })
    }

    /// With every pseudobulk's unspliced share jittering on its own from
    /// sd `sd` on (`None`: shared, no jitter).
    pub fn with_share(mut self, sd: Option<f64>) -> Result<Self> {
        self.log_share_sd = sd
            .map(|sd| var(vec![sd.ln()], &[1], self.eta.device()))
            .transpose()?;
        Ok(self)
    }

    /// With every other lineage on lineage 0's trunk until `branch_at`, one
    /// clock for all: the paths of `params` taken as the trunk (lineage 0's)
    /// and every other lineage's departure `d = z_ℓ − z_0` from it, its
    /// whitened `η_ℓ` the most probable under the prior given `d`
    /// ([`GridPrior::whiten_scaled`] to within [`TRUNK_FIT`]).
    pub fn with_trunk(
        mut self,
        params: &Params,
        prior: &GridPrior,
        branch_at: f64,
    ) -> Result<Self> {
        use nalgebra::DVector;
        let (l, n, _, h) = self.dims;
        let dev = self.eta.device().clone();
        let tb = off_the_edges(branch_at);
        let r = branch_ramp(n, tb, self.kappa);
        let mut eta = Vec::with_capacity(l * n * h);
        for (li, path) in params.path.iter().enumerate() {
            let dims: Vec<DVector<f64>> = (0..h)
                .map(|d| {
                    if li == 0 {
                        prior.whiten(&DVector::from_fn(n, |k, _| path[k][d]))
                    } else {
                        let dep = DVector::from_fn(n, |k, _| path[k][d] - params.path[0][k][d]);
                        prior.whiten_scaled(&r, &dep, TRUNK_FIT)
                    }
                })
                .collect();
            for k in 0..n {
                eta.extend(dims.iter().map(|e| e[k]));
            }
        }
        self.eta = var(eta, &[l, n, h], &dev)?;
        self.log_t = None;
        self.branch = (l > 1)
            .then(|| var(vec![(tb / (1.0 - tb)).ln(); l - 1], &[l - 1], &dev))
            .transpose()?;
        Ok(self)
    }

    /// Every other lineage's branch time off lineage 0, `None` with the
    /// lineages apart.
    pub fn branch_at(&self) -> Result<Option<Vec<f64>>> {
        self.branch
            .as_ref()
            .map(|b| {
                Ok(b.as_tensor()
                    .to_vec1::<f32>()?
                    .iter()
                    .map(|&x| 1.0 / (1.0 + (-f64::from(x)).exp()))
                    .collect())
            })
            .transpose()
    }

    /// The unspliced share's jitter sd, `None` with no jitter.
    pub fn share_sd(&self) -> Result<Option<f64>> {
        self.log_share_sd
            .as_ref()
            .map(|v| Ok(clamped_exp(f64::from(v.as_tensor().to_vec1::<f32>()?[0]))))
            .transpose()
    }

    /// [`super::likelihood::share_term`] for every pseudobulk at every point
    /// `a` holds, `a` the levels' `log(Z_u/Z_s)` as `[1 × R]` (every point
    /// for every pseudobulk, `[P × R]` out) or `[P × 1]` (a point each); `None`
    /// with no jitter.
    fn share(&self, reads: &DeviceReads, a: &Tensor) -> Result<Option<Tensor>> {
        let Some(log_sd) = &self.log_share_sd else {
            return Ok(None);
        };
        let v = clamped(&(log_sd.as_tensor() * 2.0)?)?
            .exp()?
            .reshape((1, 1))?;
        let (n_u, n) = (&reads.n_u, &reads.n);
        let sigmoid = |x: &Tensor| -> Result<Tensor> { (x.neg()?.exp()? + 1.0)?.recip() };
        let softplus =
            |x: &Tensor| -> Result<Tensor> { x.relu()? + (x.abs()?.neg()?.exp()? + 1.0)?.log()? };
        // The mode, detached: from the Gaussian guess, Newton.
        let (a_d, v_d) = (a.detach(), v.detach());
        let q0 = ((n_u + 0.5)? / (n + 1.0)?)?;
        let h0 = (n * (&q0 * (1.0 - &q0)?)?)?;
        let shrink = h0.broadcast_mul(&v_d)?;
        let shrink = (&shrink / (&shrink + 1.0)?)?;
        let logit0 = (q0.log()? - (1.0 - &q0)?.log()?)?;
        let mut w = logit0.broadcast_sub(&a_d)?.broadcast_mul(&shrink)?;
        for _ in 0..SHARE_NEWTON {
            let q = sigmoid(&w.broadcast_add(&a_d)?)?;
            let grad = (n_u.broadcast_sub(&n.broadcast_mul(&q)?)? - w.broadcast_div(&v_d)?)?;
            let hess = n
                .broadcast_mul(&(&q * (1.0 - &q)?)?)?
                .broadcast_add(&v_d.recip()?)?;
            w = (w + (grad / hess)?)?;
        }
        let w = w.detach();
        let x = w.broadcast_add(a)?;
        let q = sigmoid(&x)?;
        let fit = ((n_u.broadcast_mul(&w)?
            - n.broadcast_mul(&(softplus(&x)? - softplus(a)?.broadcast_as(x.shape())?)?)?)?
            - w.sqr()?.broadcast_div(&(&v * 2.0)?)?)?;
        let det =
            ((n.broadcast_mul(&(&q * (1.0 - &q)?)?)?.broadcast_mul(&v)? + 1.0)?.log()? * 0.5)?;
        Ok(Some((fit - det)?))
    }

    /// `[R]` the levels' `log(Z_u/Z_s)` at every point.
    fn log_share_ratio(u: &Tensor, s: &Tensor) -> Result<Tensor> {
        u.sum(1)?.log()? - s.sum(1)?.log()?
    }

    /// `[P × R]` every pseudobulk's log-likelihood at every point, the
    /// multinomial (no coefficient) plus the share term.
    fn table(&self, reads: &DeviceReads, u: &Tensor, s: &Tensor) -> Result<Tensor> {
        let z = (u.sum_keepdim(1)? + s.sum_keepdim(1)?)?.log()?;
        let lu = u.log()?.broadcast_sub(&z)?;
        let ls = s.log()?.broadcast_sub(&z)?;
        let tab = (reads.u.matmul(&lu.t()?)? + reads.s.matmul(&ls.t()?)?)?;
        let a = Self::log_share_ratio(u, s)?.unsqueeze(0)?;
        match self.share(reads, &a)? {
            Some(share) => tab + share,
            None => Ok(tab),
        }
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
        let column = |f: &dyn Fn(usize) -> f64| {
            Tensor::from_vec((0..p).map(|i| f(i) as f32).collect::<Vec<_>>(), (p, 1), dev)
        };
        Ok(DeviceReads {
            u: host(&reads.u)?,
            s: host(&reads.s)?,
            n_u: column(&|i| reads.u.row(i).sum())?,
            n: column(&|i| reads.total(i))?,
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
        v.extend(self.log_share_sd.clone());
        v.extend(self.branch.clone());
        v
    }

    /// `([1 × G], [1 × G])` `log β` and `log γ` as the model reads them:
    /// clamped to their bounds (unbounded, to `±LOG_CLAMP`). The climb seeks the mode in the rates' own
    /// coordinates, so inside the bounds nothing changes (the sampler's
    /// link, [`GenePrior::link`], carries a Jacobian; a mode sought through
    /// it would move toward the bounds' middle); the priors, on the
    /// unclamped logs, pull a rate that strays past back in.
    fn log_rates(&self) -> Result<(Tensor, Tensor)> {
        let (lb, lg) = (self.log_beta.as_tensor(), self.log_gamma.as_tensor());
        match self.prior.log_rate {
            None => Ok((clamped(lb)?, clamped(lg)?)),
            Some((lo, hi)) => Ok((lb.clamp(lo, hi)?, lg.clamp(lo, hi)?)),
        }
    }

    /// `[L × 1]` every lineage's duration.
    fn t_len(&self) -> Result<Tensor> {
        let log_t = match &self.log_t {
            Some(t) => {
                let zero = Tensor::zeros((1,), DType::F32, self.eta.device())?;
                Tensor::cat(&[&zero, t.as_tensor()], 0)?
            }
            None => Tensor::zeros((self.dims.0,), DType::F32, self.eta.device())?,
        };
        clamped(&log_t)?.exp()?.unsqueeze(1)
    }

    /// `[L × n × H]` the paths.
    fn paths(&self) -> Result<Tensor> {
        // `chol` holds Cᵀ, so a row-vector product: z[k] = Σ_j C[k, j] η[j].
        let (l, n, h) = (self.dims.0, self.dims.1, self.dims.3);
        let eta = self.eta.as_tensor().transpose(1, 2)?.reshape((l * h, n))?;
        let z = eta
            .matmul(&self.chol)?
            .reshape((l, h, n))?
            .transpose(1, 2)?;
        let Some(branch) = &self.branch else {
            return Ok(z);
        };
        // z_ℓ = z_0 + √2 r_ℓ(t) w_ℓ, r_ℓ = 1 − exp(−((t − t_b)₊/κ)²).
        let dev = z.device();
        let t = Tensor::from_vec(
            (0..n).map(|k| phase(k, n) as f32).collect::<Vec<_>>(),
            (1, n),
            dev,
        )?;
        let tb = (branch.as_tensor().neg()?.exp()? + 1.0)?
            .recip()?
            .unsqueeze(1)?; // [L−1 × 1]
        let x = (t.broadcast_sub(&tb)?.relu()? / self.kappa)?;
        let r = ((x.sqr()?.neg()?.exp()?.neg()? + 1.0)? * std::f64::consts::SQRT_2)?; // [L−1 × n]
        let trunk = z.narrow(0, 0, 1)?;
        let parted =
            trunk.broadcast_add(&z.narrow(0, 1, l - 1)?.broadcast_mul(&r.unsqueeze(2)?)?)?;
        Tensor::cat(&[&trunk, &parted], 0)
    }

    /// `([L·n × G], [L·n × G])` the levels.
    pub fn levels(&self) -> Result<(Tensor, Tensor)> {
        let (l, n, g, h) = self.dims;
        let z = self.paths()?.reshape((l * n, h))?;
        let alpha = clamped(
            &z.matmul(&self.rho.as_tensor().t()?)?
                .broadcast_add(self.base.as_tensor())?,
        )?
        .exp()?
        .reshape((l, n, g))?;
        let (log_beta, log_gamma) = self.log_rates()?;
        let (beta, gamma) = (log_beta.exp()?, log_gamma.exp()?);
        let step = (self.t_len()? / (n - 1) as f64)?; // [L × 1]
        let m = self.sub;
        // `e^{−βδ}`, `e^{−γδ}` and `E(δ) = (e^{−βδ} − e^{−γδ})/(γ − β)`
        // (`δ e^{−γδ} (1 + x/2 + x²/6)` near `x = (γ − β)δ = 0`, where the
        // difference cancels) for a step `δ` `[L × 1]`, each `[L × G]`. Both
        // exponentials are at most 1, so `E` cannot overflow however far
        // apart the rates are.
        let factors = |delta: &Tensor| -> Result<(Tensor, Tensor, Tensor)> {
            let bh = delta.broadcast_mul(&beta)?;
            let gh = delta.broadcast_mul(&gamma)?;
            let (eb, eg) = (bh.neg()?.exp()?, gh.neg()?.exp()?);
            let x = (&gh - &bh)?;
            let small = x.abs()?.lt(SMALL)?;
            let gap =
                small.where_cond(&x.ones_like()?, &(&gamma - &beta)?.broadcast_as(x.shape())?)?;
            let series = ((x.affine(1.0 / 6.0, 0.5)? * &x)? + 1.0)?;
            let near = (delta.broadcast_mul(&eg)? * series)?;
            let e = small.where_cond(&near, &((&eb - &eg)? / gap)?)?;
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
        let positions = positions(n, m);
        let stack = |v: &[Tensor]| Tensor::stack(v, 1)?.reshape((l * positions, g));
        Ok((stack(&us)?, stack(&ss)?))
    }

    /// `Σ_p log Σ_{ℓ,k} π_ℓ (1/n) p(x_p | ℓ, k)`, the multinomial (no
    /// coefficient).
    pub fn log_marginal(&self, reads: &DeviceReads) -> Result<Tensor> {
        let (l, n) = (self.dims.0, self.positions());
        let (u, s) = self.levels()?;
        let tab = self.table(reads, &u, &s)?; // [P × L·n]
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
        if let Some(b) = &self.branch {
            // t_b uniform on [0, 1]: on its logit, log σ(x) + log σ(−x).
            let x = b.as_tensor();
            let softplus = |x: &Tensor| -> Result<Tensor> {
                x.relu()? + (x.abs()?.neg()?.exp()? + 1.0)?.log()?
            };
            total = (total - (softplus(x)? + softplus(&x.neg()?)?)?.sum_all()?)?;
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

    /// Every pseudobulk's posterior over lineages `[P][L]` and its mean
    /// time on its lineages, `[P]` (lineage time, `k T_ℓ/(n − 1)`).
    pub fn posterior(&self, reads: &DeviceReads) -> Result<(Vec<Vec<f64>>, Vec<f64>)> {
        let (l, n) = (self.dims.0, self.positions());
        let (u, s) = self.levels()?;
        let tab = self.table(reads, &u, &s)?;
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
        positions(self.dims.1, self.sub)
    }

    /// The parameters on the host.
    pub fn params(&self) -> Result<Params> {
        let (l, n, _, _) = self.dims;
        let row = |x: &Tensor| -> Result<Vec<f64>> {
            Ok(x.flatten_all()?
                .to_vec1::<f32>()?
                .iter()
                .map(|&v| f64::from(v))
                .collect())
        };
        let paths = self.paths()?.to_vec3::<f32>()?;
        let (lb, lg) = self.log_rates()?;
        let rates =
            |x: &Tensor| -> Result<Vec<f64>> { Ok(row(x)?.into_iter().map(clamped_exp).collect()) };
        Ok(Params {
            beta: rates(&lb)?,
            gamma: rates(&lg)?,
            rho: self
                .rho
                .as_tensor()
                .to_vec2::<f32>()?
                .iter()
                .map(|r| r.iter().map(|&x| f64::from(x)).collect())
                .collect(),
            base: row(self.base.as_tensor())?,
            t_len: row(&self.t_len()?)?,
            path: (0..l)
                .map(|i| {
                    (0..n)
                        .map(|k| paths[i][k].iter().map(|&x| f64::from(x)).collect())
                        .collect()
                })
                .collect(),
            sub: self.sub,
            branch: self.branch_at()?,
        })
    }
}

/// `x` clamped to `±`[`LOG_CLAMP`] before an `exp`.
fn clamped(x: &Tensor) -> Result<Tensor> {
    x.clamp(-LOG_CLAMP, LOG_CLAMP)
}
