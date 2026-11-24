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
//! gene's coordinates, `N(0, I)` on `η`, `N(0, 1)` on `log T`.
//!
//! With [`Fit::with_share`], every pseudobulk's unspliced share jitters on
//! its own, `σ` fitted: [`super::likelihood::share_term`] added at every
//! point, its mode `ω̂` found with the graph detached (at the mode the
//! gradient through `ω̂` vanishes from the first part; the Laplace
//! determinant's share of it is dropped). The term is a marginal likelihood
//! for `σ` with `σ → 0` the shared model, so `σ` has no prior.
//!
//! The unspliced levels carry one global capture `e^c` against the
//! spliced ([`Params::log_capture`]), `c` fitted with the rest under
//! [`CAPTURE_PRIOR`]; the jitter is every pseudobulk's departure from it.
//!
//! With [`Fit::with_trunks`], every other lineage branches off lineage 0 at
//! its own time `t_b` on one clock (every duration the unit):
//! `z_ℓ = z_0 + √2 r(t) w_ℓ`, `w_ℓ = C η_ℓ`, `r = 1 − e^{−((t − t_b)₊/κ)²}`,
//! so before `t_b` the lineages share path, clock and levels, past it they
//! part smoothly and far past it are as far apart as two independent
//! paths; `t_b` uniform on `[0, 1]`.
//!
//! A fit holds a batch of starts, every variable with a leading start axis
//! and every start's objective its own: Adam's steps are elementwise, so
//! the starts climb together exactly as each would alone, in one call per
//! step for all (a step costs the GPU about the same at any size).

use super::blocks::GenePrior;
use super::kernel::GridPrior;
use super::likelihood::Reads;
use super::model::{branch_ramp, clamped_exp, Params, CAPTURE_PRIOR, LOG_CLAMP};
use crate::rbm_core::{adam, softplus};
use legume_numeric::candle::candle_core::{bail, DType, Device, Result, Tensor, Var, D};
use legume_numeric::candle::candle_nn::ops::log_softmax;
use legume_numeric::candle::candle_nn::Optimizer;
use std::sync::atomic::{AtomicBool, Ordering};

/// The departure's misfit scale when a start is put on a trunk
/// ([`Fit::with_trunks`]).
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

/// [`SMALL`] in f64: the difference loses `2·10⁻¹²` of `E`, the series
/// `4·10⁻¹⁴`.
const SMALL_F64: f64 = 1e-4;

/// Pseudobulks' reads on the device: `[G × P]` per track (transposed, for
/// the starts' levels to multiply), every pseudobulk's unspliced and all
/// reads `[1 × 1 × P]`.
pub struct DeviceReads {
    u_t: Tensor,
    s_t: Tensor,
    n_u: Tensor,
    n: Tensor,
    /// `[1 × 1 × P]` the observed share's logit and its binomial weight,
    /// `q̂ = (n_u + ½)/(n + 1)`: `logit q̂` and `n q̂(1 − q̂)`, where the
    /// jitter's Newton starts ([`Fit::share`]).
    logit0: Tensor,
    h0: Tensor,
    total: f64,
}

impl DeviceReads {
    /// `(P, G)`: the pseudobulks and the genes.
    pub fn shape(&self) -> (usize, usize) {
        let (g, p) = self.u_t.dims2().expect("[G × P]");
        (p, g)
    }

    /// `reads` on `dev`, once for every climb on them.
    pub fn new(reads: &Reads, dev: &Device) -> Result<Self> {
        let (p, g) = (reads.u.nrows(), reads.u.ncols());
        // nalgebra is column-major: a column a gene, so `[G × P]` row-major.
        let host = |m: &nalgebra::DMatrix<f64>| {
            Tensor::from_vec(m.iter().map(|&x| x as f32).collect::<Vec<_>>(), (g, p), dev)
        };
        let total = (0..p).map(|i| reads.total(i)).sum();
        let row = |f: &dyn Fn(usize) -> f64| {
            Tensor::from_vec(
                (0..p).map(|i| f(i) as f32).collect::<Vec<_>>(),
                (1, 1, p),
                dev,
            )
        };
        let (n_u, n) = (row(&|i| reads.u.row(i).sum())?, row(&|i| reads.total(i))?);
        let q0 = ((&n_u + 0.5)? / (&n + 1.0)?)?;
        let h0 = (&n * (&q0 * (1.0 - &q0)?)?)?;
        let logit0 = (q0.log()? - (1.0 - &q0)?.log()?)?;
        Ok(Self {
            u_t: host(&reads.u)?,
            s_t: host(&reads.s)?,
            n_u,
            n,
            logit0,
            h0,
            total,
        })
    }
}

/// A batch of `B` starts on one model ([`Fit::from_starts`]); with one
/// start, that start's fit.
pub struct Fit {
    /// `[B × 1 × G]` each rate's log, read through its bounds
    /// ([`GenePrior::log_rate`], [`Fit::log_rates`]).
    log_beta: Var,
    log_gamma: Var,
    /// `[B × G × H]`.
    rho: Var,
    /// `[B × 1 × G]`.
    base: Var,
    /// `[B × L × n × H]` whitened paths.
    eta: Var,
    /// `[B × (L − 1)]` the other lineages' log durations (lineage 1's is 0).
    log_t: Option<Var>,
    /// `[B × L]`.
    logit_pi: Var,
    /// `[B × (L − 1)]` every other lineage's branch time off lineage 0, as
    /// a logit; `None`: lineages apart, each its own duration.
    branch: Option<Var>,
    /// The prior's length scale over the duration, the branch's too.
    kappa: f64,
    /// `[B]` the log sd of the unspliced share's jitter, `None` for none.
    log_share_sd: Option<Var>,
    /// `[B]` the unspliced reads' log capture ([`Params::log_capture`]).
    log_capture: Var,
    /// `[n × n]` the prior's Cholesky factor.
    chol: Tensor,
    prior: GenePrior,
    /// `(B, L, n, G, H)`.
    dims: (usize, usize, usize, usize, usize),
    /// Positions per grid step ([`Params::sub`]).
    sub: usize,
}

fn var(v: Vec<f64>, shape: &[usize], dev: &Device) -> Result<Var> {
    let t = Tensor::from_vec(v.iter().map(|&x| x as f32).collect::<Vec<_>>(), shape, dev)?;
    Var::from_tensor(&t)
}

/// `[B]` the sum of every start's entries of `x` (`[B × …]`).
fn per_start(x: &Tensor) -> Result<Tensor> {
    let b = x.dim(0)?;
    x.reshape((b, x.elem_count() / b))?.sum(1)
}

/// The most entries of a batch's widest table climbed at once
/// ([`starts_per_batch`]): past it the starts climb in batches. The
/// backward pass keeps a few dozen such tables (at f32, 64 MB each).
const BATCH_ENTRIES: usize = 1 << 24;

/// How many starts of `lineages` lineages, `positions` each, on the reads
/// of `p` pseudobulks over `g` genes climb in one batch: a start's widest
/// tables are its points by the pseudobulks and by the genes.
pub fn starts_per_batch(lineages: usize, positions: usize, p: usize, g: usize) -> usize {
    (BATCH_ENTRIES / (lineages * positions * p.max(g)).max(1)).max(1)
}

/// A start's whitened paths, `[L·n·H]` (lineage, point, dimension).
fn whitened(params: &Params, prior: &GridPrior) -> Vec<f64> {
    let (n, h) = (params.path[0].len(), params.rho[0].len());
    let mut eta = Vec::with_capacity(params.path.len() * n * h);
    for path in &params.path {
        let dims: Vec<_> = (0..h)
            .map(|d| prior.whiten(&nalgebra::DVector::from_fn(n, |k, _| path[k][d])))
            .collect();
        for k in 0..n {
            eta.extend(dims.iter().map(|e| e[k]));
        }
    }
    eta
}

impl Fit {
    /// Start at `params`, the paths whitened by `prior` (whose `κ/T` is held).
    #[cfg(test)]
    pub fn from_params(params: &Params, prior: &GridPrior, dev: &Device) -> Result<Self> {
        Self::from_starts(std::slice::from_ref(params), prior, dev)
    }

    /// Every one of `starts` (one model: lineages, points, genes, paths'
    /// dimension and positions all alike, and so the gene prior's bounds)
    /// a start of the batch, in order.
    pub fn from_starts(starts: &[Params], prior: &GridPrior, dev: &Device) -> Result<Self> {
        let Some(first) = starts.first() else {
            bail!("ode-mixture: a fit of no start");
        };
        let b = starts.len();
        let (l, g) = (first.path.len(), first.beta.len());
        let (n, h) = (first.path[0].len(), first.rho[0].len());
        let gene_prior = GenePrior::bounded(first);
        if starts.iter().any(|s| {
            s.path.len() != l
                || s.beta.len() != g
                || s.path[0].len() != n
                || s.rho[0].len() != h
                || s.sub != first.sub
                || GenePrior::bounded(s).log_rate != gene_prior.log_rate
        }) {
            bail!("ode-mixture: the starts of one fit differ in their model");
        }
        let all = |f: &dyn Fn(&Params) -> Vec<f64>| starts.iter().flat_map(f).collect::<Vec<_>>();
        let log_t = if l > 1 {
            Some(var(
                all(&|s| s.t_len[1..].iter().map(|t| (t / s.t_len[0]).ln()).collect()),
                &[b, l - 1],
                dev,
            )?)
        } else {
            None
        };
        let c = prior.chol();
        Ok(Self {
            log_beta: var(
                all(&|s| s.beta.iter().map(|x| x.ln()).collect()),
                &[b, 1, g],
                dev,
            )?,
            log_gamma: var(
                all(&|s| s.gamma.iter().map(|x| x.ln()).collect()),
                &[b, 1, g],
                dev,
            )?,
            rho: var(
                all(&|s| s.rho.iter().flatten().copied().collect()),
                &[b, g, h],
                dev,
            )?,
            base: var(all(&|s| s.base.clone()), &[b, 1, g], dev)?,
            eta: var(all(&|s| whitened(s, prior)), &[b, l, n, h], dev)?,
            log_t,
            logit_pi: var(vec![0.0; b * l], &[b, l], dev)?,
            log_share_sd: None,
            log_capture: var(all(&|s| vec![s.log_capture]), &[b], dev)?,
            branch: None,
            kappa: prior.kappa(),
            chol: Tensor::from_vec(c.iter().map(|&x| x as f32).collect::<Vec<_>>(), (n, n), dev)?,
            prior: gene_prior,
            dims: (b, l, n, g, h),
            sub: first.sub.max(1),
        })
    }

    /// Start `b` alone, its own fit (its variables copied).
    pub fn start(&self, b: usize) -> Result<Self> {
        let one = |v: &Var| Var::from_tensor(&v.as_tensor().narrow(0, b, 1)?.contiguous()?);
        let maybe = |v: &Option<Var>| v.as_ref().map(one).transpose();
        let (_, l, n, g, h) = self.dims;
        Ok(Self {
            log_beta: one(&self.log_beta)?,
            log_gamma: one(&self.log_gamma)?,
            rho: one(&self.rho)?,
            base: one(&self.base)?,
            eta: one(&self.eta)?,
            log_t: maybe(&self.log_t)?,
            logit_pi: one(&self.logit_pi)?,
            branch: maybe(&self.branch)?,
            kappa: self.kappa,
            log_share_sd: maybe(&self.log_share_sd)?,
            log_capture: one(&self.log_capture)?,
            chol: self.chol.clone(),
            prior: self.prior.clone(),
            dims: (1, l, n, g, h),
            sub: self.sub,
        })
    }

    /// The batch's one start, for what reads a single fit.
    fn single(&self) -> Result<()> {
        if self.dims.0 != 1 {
            bail!(
                "ode-mixture: {} starts where one is read (Fit::start)",
                self.dims.0
            );
        }
        Ok(())
    }

    /// With every pseudobulk's unspliced share jittering on its own from
    /// sd `sd` on, in every start (`None`: shared, no jitter).
    pub fn with_share(mut self, sd: Option<f64>) -> Result<Self> {
        let b = self.dims.0;
        self.log_share_sd = sd
            .map(|sd| var(vec![sd.ln(); b], &[b], self.eta.device()))
            .transpose()?;
        Ok(self)
    }

    /// [`Self::with_trunks`] for a fit of one start.
    #[cfg(test)]
    pub fn with_trunk(self, params: &Params, prior: &GridPrior, branch_at: &[f64]) -> Result<Self> {
        self.with_trunks(std::slice::from_ref(params), prior, &[branch_at.to_vec()])
    }

    /// With every other lineage on lineage 0's trunk until its branch time
    /// (`branch_at`, a list per start, one per other lineage), one clock for
    /// all: every start's paths (`starts`, as [`Self::from_starts`] took
    /// them) taken as the trunk (lineage 0's) and every other lineage's
    /// departure `d = z_ℓ − z_0` from it, its whitened `η_ℓ` the most
    /// probable under the prior given `d` ([`GridPrior::whiten_scaled`] to
    /// within [`TRUNK_FIT`]).
    pub fn with_trunks(
        mut self,
        starts: &[Params],
        prior: &GridPrior,
        branch_at: &[Vec<f64>],
    ) -> Result<Self> {
        use nalgebra::DVector;
        let (b, l, n, _, h) = self.dims;
        if starts.len() != b || branch_at.len() != b || branch_at.iter().any(|t| t.len() + 1 != l) {
            bail!("ode-mixture: branch times for {b} starts of {l} lineages wanted");
        }
        let dev = self.eta.device().clone();
        let mut eta = Vec::with_capacity(b * l * n * h);
        let mut logits = Vec::with_capacity(b * l.saturating_sub(1));
        for (params, at) in starts.iter().zip(branch_at) {
            let tb: Vec<f64> = at.iter().map(|t| t.clamp(0.02, 0.98)).collect();
            for (li, path) in params.path.iter().enumerate() {
                let ramp = (li > 0).then(|| branch_ramp(n, tb[li - 1], self.kappa));
                let dims: Vec<DVector<f64>> = (0..h)
                    .map(|d| match &ramp {
                        None => prior.whiten(&DVector::from_fn(n, |k, _| path[k][d])),
                        Some(r) => {
                            let dep = DVector::from_fn(n, |k, _| path[k][d] - params.path[0][k][d]);
                            prior.whiten_scaled(r, &dep, TRUNK_FIT)
                        }
                    })
                    .collect();
                for k in 0..n {
                    eta.extend(dims.iter().map(|e| e[k]));
                }
            }
            logits.extend(tb.iter().map(|t| (t / (1.0 - t)).ln()));
        }
        self.eta = var(eta, &[b, l, n, h], &dev)?;
        self.log_t = None;
        self.branch = (l > 1)
            .then(|| var(logits, &[b, l - 1], &dev))
            .transpose()?;
        Ok(self)
    }

    /// Every other lineage's branch time off lineage 0, `None` with the
    /// lineages apart.
    pub fn branch_at(&self) -> Result<Option<Vec<f64>>> {
        self.single()?;
        self.branch
            .as_ref()
            .map(|b| {
                Ok(b.as_tensor()
                    .flatten_all()?
                    .to_vec1::<f32>()?
                    .iter()
                    .map(|&x| 1.0 / (1.0 + (-f64::from(x)).exp()))
                    .collect())
            })
            .transpose()
    }

    /// The unspliced reads' log capture.
    pub fn log_capture(&self) -> Result<f64> {
        self.single()?;
        Ok(f64::from(self.log_capture.as_tensor().to_vec1::<f32>()?[0]))
    }

    /// The unspliced share's jitter sd, `None` with no jitter.
    pub fn share_sd(&self) -> Result<Option<f64>> {
        self.single()?;
        self.log_share_sd
            .as_ref()
            .map(|v| Ok(clamped_exp(f64::from(v.as_tensor().to_vec1::<f32>()?[0]))))
            .transpose()
    }

    /// `[B × R × P]` [`super::likelihood::share_term`] for every pseudobulk
    /// at every point, `a` `[B × R × 1]` the levels' `log(Z_u/Z_s)` at every
    /// point; `None` with no jitter.
    fn share(&self, reads: &DeviceReads, a: &Tensor) -> Result<Option<Tensor>> {
        let Some(log_sd) = &self.log_share_sd else {
            return Ok(None);
        };
        let v = clamped(&(log_sd.as_tensor() * 2.0)?)?
            .exp()?
            .reshape((self.dims.0, 1, 1))?;
        let (n_u, n) = (&reads.n_u, &reads.n);
        let sigmoid = |x: &Tensor| -> Result<Tensor> { (x.neg()?.exp()? + 1.0)?.recip() };
        // The mode, detached: from the Gaussian guess, Newton.
        let (a_d, v_d) = (a.detach(), v.detach());
        let v_inv = v_d.recip()?;
        let shrink = reads.h0.broadcast_mul(&v_d)?;
        let shrink = (&shrink / (&shrink + 1.0)?)?;
        let mut w = reads.logit0.broadcast_sub(&a_d)?.broadcast_mul(&shrink)?;
        for _ in 0..SHARE_NEWTON {
            let q = sigmoid(&w.broadcast_add(&a_d)?)?;
            let grad = (n_u.broadcast_sub(&n.broadcast_mul(&q)?)? - w.broadcast_mul(&v_inv)?)?;
            let hess = n
                .broadcast_mul(&(&q * (1.0 - &q)?)?)?
                .broadcast_add(&v_inv)?;
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

    /// `[B × R × P]` every pseudobulk's log-likelihood at every point of
    /// every start, the multinomial (no coefficient) plus the share term;
    /// `u`, `s` the levels `[B × R × G]`.
    fn table(&self, reads: &DeviceReads, u: &Tensor, s: &Tensor) -> Result<Tensor> {
        let (b, r, g) = u.dims3()?;
        let p = reads.u_t.dim(1)?;
        let (zu, zs) = (u.sum_keepdim(2)?, s.sum_keepdim(2)?);
        let z = (&zu + &zs)?.log()?;
        let lu = u.log()?.broadcast_sub(&z)?.reshape((b * r, g))?;
        let ls = s.log()?.broadcast_sub(&z)?.reshape((b * r, g))?;
        let tab = (lu.matmul(&reads.u_t)? + ls.matmul(&reads.s_t)?)?.reshape((b, r, p))?;
        // The levels' log(Z_u/Z_s) at every point, [B × R × 1].
        let a = (zu.log()? - zs.log()?)?;
        match self.share(reads, &a)? {
            Some(share) => tab + share,
            None => Ok(tab),
        }
    }

    #[cfg(test)]
    pub fn reads(&self, reads: &Reads) -> Result<DeviceReads> {
        DeviceReads::new(reads, self.eta.device())
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
        v.push(self.log_capture.clone());
        v.extend(self.branch.clone());
        v
    }

    /// `([B × 1 × G], [B × 1 × G])` `log β` and `log γ` as the model reads
    /// them: clamped to their bounds (unbounded, to `±LOG_CLAMP`). The climb
    /// seeks the mode in the rates' own coordinates, so inside the bounds
    /// nothing changes (the gene's link, [`GenePrior::link`], carries a
    /// Jacobian; a mode sought through it would move toward the bounds'
    /// middle); the priors, on the unclamped logs, pull a rate that strays
    /// past back in.
    fn log_rates(&self) -> Result<(Tensor, Tensor)> {
        let (lb, lg) = (self.log_beta.as_tensor(), self.log_gamma.as_tensor());
        match self.prior.log_rate {
            None => Ok((clamped(lb)?, clamped(lg)?)),
            Some((lo, hi)) => Ok((lb.clamp(lo, hi)?, lg.clamp(lo, hi)?)),
        }
    }

    /// `[B × L]` every lineage's duration.
    fn t_len(&self) -> Result<Tensor> {
        let (b, l) = (self.dims.0, self.dims.1);
        let dev = self.eta.device();
        let log_t = match &self.log_t {
            Some(t) => {
                let zero = Tensor::zeros((b, 1), DType::F32, dev)?;
                Tensor::cat(&[&zero, t.as_tensor()], 1)?
            }
            None => Tensor::zeros((b, l), DType::F32, dev)?,
        };
        clamped(&log_t)?.exp()
    }

    /// `[B × L × n × H]` the paths.
    fn paths(&self) -> Result<Tensor> {
        // `chol` holds Cᵀ, so a row-vector product: z[k] = Σ_j C[k, j] η[j].
        let (b, l, n, _, h) = self.dims;
        let eta = self
            .eta
            .as_tensor()
            .transpose(2, 3)?
            .reshape((b * l * h, n))?;
        let z = eta
            .matmul(&self.chol)?
            .reshape((b, l, h, n))?
            .transpose(2, 3)?;
        let Some(branch) = &self.branch else {
            return Ok(z);
        };
        // z_ℓ = z_0 + √2 r_ℓ(t) w_ℓ, r_ℓ = 1 − exp(−((t − t_b)₊/κ)²).
        let dev = z.device();
        let t = Tensor::from_vec(
            (0..n)
                .map(|k| k as f32 / (n - 1) as f32)
                .collect::<Vec<_>>(),
            (1, 1, n),
            dev,
        )?;
        let tb = (branch.as_tensor().neg()?.exp()? + 1.0)?
            .recip()?
            .unsqueeze(2)?; // [B × (L−1) × 1]
        let x = (t.broadcast_sub(&tb)?.relu()? / self.kappa)?;
        let r = ((x.sqr()?.neg()?.exp()?.neg()? + 1.0)? * std::f64::consts::SQRT_2)?; // [B × (L−1) × n]
        let trunk = z.narrow(1, 0, 1)?;
        let parted =
            trunk.broadcast_add(&z.narrow(1, 1, l - 1)?.broadcast_mul(&r.unsqueeze(3)?)?)?;
        Tensor::cat(&[&trunk, &parted], 1)
    }

    /// `([B × L·n × G], [B × L·n × G])` every start's levels on its
    /// `paths` ([`Self::paths`], [`closed_form_levels`]), the unspliced
    /// captured at `e^c`.
    fn batch_levels(&self, paths: &Tensor) -> Result<(Tensor, Tensor)> {
        let (log_beta, log_gamma) = self.log_rates()?;
        let capture = clamped(self.log_capture.as_tensor())?
            .exp()?
            .reshape((self.dims.0, 1, 1))?;
        let (u, s) = closed_form_levels(
            paths,
            &self.t_len()?,
            self.sub,
            &log_beta,
            &log_gamma,
            self.rho.as_tensor(),
            self.base.as_tensor(),
        )?;
        Ok((u.broadcast_mul(&capture)?, s))
    }

    /// `([L·n × G], [L·n × G])` the levels of a fit of one start.
    #[cfg(test)]
    pub fn levels(&self) -> Result<(Tensor, Tensor)> {
        self.single()?;
        let (u, s) = self.batch_levels(&self.paths()?)?;
        Ok((u.squeeze(0)?, s.squeeze(0)?))
    }

    /// `[B × L·n × 1]` every point's log prior weight, `log π_ℓ − log n`.
    fn point_prior(&self) -> Result<Tensor> {
        let (b, l) = (self.dims.0, self.dims.1);
        let n = self.positions();
        let log_pi = log_softmax(self.logit_pi.as_tensor(), D::Minus1)?; // [B × L]
        log_pi
            .unsqueeze(2)?
            .broadcast_as((b, l, n))?
            .reshape((b, l * n, 1))?
            - (n as f64).ln()
    }

    /// `[B]` every start's `Σ_p log Σ_{ℓ,k} π_ℓ (1/n) p(x_p | ℓ, k)`, the
    /// multinomial (no coefficient).
    fn log_marginals(&self, reads: &DeviceReads, paths: &Tensor) -> Result<Tensor> {
        let (u, s) = self.batch_levels(paths)?;
        let tab = self.table(reads, &u, &s)?; // [B × R × P]
        tab.broadcast_add(&self.point_prior()?)?
            .log_sum_exp(1)?
            .sum(1)
    }

    /// The starts' log marginals ([`Self::log_marginals`]) summed: with one
    /// start, its own.
    #[cfg(test)]
    pub fn log_marginal(&self, reads: &DeviceReads) -> Result<Tensor> {
        self.log_marginals(reads, &self.paths()?)?.sum_all()
    }

    /// `[B]` every start's log priors, on its `paths`.
    fn log_prior(&self, paths: &Tensor) -> Result<Tensor> {
        let (b, _, _, _, h) = self.dims;
        let pr = &self.prior;
        let gauss = |x: &Tensor, (m, s): (f64, f64)| -> Result<Tensor> {
            per_start(&((x - m)? / s)?.sqr()?)? * -0.5
        };
        let (lb, lg) = (self.log_beta.as_tensor(), self.log_gamma.as_tensor());
        let z_bar = paths.mean_keepdim(2)?.mean_keepdim(1)?.reshape((b, h, 1))?;
        let spliced = ((self.rho.as_tensor().matmul(&z_bar)?.transpose(1, 2)?
            + self.base.as_tensor())?
            - lg)?;
        let mut total = (gauss(lg, pr.log_gamma)? + gauss(&(lg - lb)?, pr.log_ratio)?)?;
        total = (total + gauss(self.rho.as_tensor(), (0.0, pr.rho_sd))?)?;
        total = (total + gauss(&spliced, pr.log_spliced)?)?;
        total = (total + gauss(self.eta.as_tensor(), (0.0, 1.0))?)?;
        total = (total + gauss(self.log_capture.as_tensor(), CAPTURE_PRIOR)?)?;
        if let Some(t) = &self.log_t {
            total = (total + gauss(t.as_tensor(), (0.0, 1.0))?)?;
        }
        if let Some(br) = &self.branch {
            // t_b uniform on [0, 1]: on its logit, log σ(x) + log σ(−x).
            let x = br.as_tensor();
            total = (total - per_start(&(softplus(x)? + softplus(&x.neg()?)?)?)?)?;
        }
        Ok(total)
    }

    /// `[B]` every start's objective.
    fn objectives(&self, reads: &DeviceReads) -> Result<Tensor> {
        let paths = self.paths()?;
        self.log_marginals(reads, &paths)? + self.log_prior(&paths)?
    }

    /// The starts' objectives summed (each start's gradient its own): with
    /// one start, its own.
    pub fn objective(&self, reads: &DeviceReads) -> Result<Tensor> {
        self.objectives(reads)?.sum_all()
    }

    /// `steps` of Adam at `lr` on the objective per read, every start at
    /// once, ended early once `stop` is set, with a progress bar as senna's
    /// other fits show; every start's objective after.
    pub fn adam(
        &self,
        dev: &DeviceReads,
        steps: usize,
        lr: f64,
        stop: Option<&AtomicBool>,
    ) -> Result<Vec<f32>> {
        let mut ascent = adam(self.vars(), lr)?;
        let bar = crate::logging::new_progress_bar(steps as u64).with_message("ode-mixture climb");
        for _ in 0..steps {
            if stop.is_some_and(|s| s.load(Ordering::Relaxed)) {
                log::info!("ode-mixture climb: stopped");
                break;
            }
            let loss = (self.objective(dev)? / -dev.total)?;
            ascent.backward_step(&loss)?;
            bar.inc(1);
        }
        bar.finish_and_clear();
        self.objectives(dev)?.to_vec1::<f32>()
    }

    /// Every pseudobulk's posterior over lineages `[P][L]` and its mean
    /// time on its lineages, `[P]` (lineage time, `k T_ℓ/(n − 1)`).
    #[cfg(test)]
    pub fn posterior(&self, reads: &DeviceReads) -> Result<(Vec<Vec<f64>>, Vec<f64>)> {
        self.single()?;
        let (l, n) = (self.dims.1, self.positions());
        let (u, s) = self.batch_levels(&self.paths()?)?;
        let tab = self
            .table(reads, &u, &s)?
            .broadcast_add(&self.point_prior()?)?;
        let post =
            legume_numeric::candle::candle_nn::ops::softmax(&tab.squeeze(0)?.t()?, D::Minus1)?
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

    /// `[L]` the lineages' log weights.
    pub fn log_weights(&self) -> Result<Vec<f64>> {
        self.single()?;
        Ok(log_softmax(self.logit_pi.as_tensor(), D::Minus1)?
            .flatten_all()?
            .to_vec1::<f32>()?
            .iter()
            .map(|&x| f64::from(x))
            .collect())
    }

    /// Positions per lineage, `(n − 1) sub + 1`.
    pub fn positions(&self) -> usize {
        (self.dims.2 - 1) * self.sub + 1
    }

    /// The parameters on the host.
    pub fn params(&self) -> Result<Params> {
        self.single()?;
        let (_, l, n, _, _) = self.dims;
        let row = |x: &Tensor| -> Result<Vec<f64>> {
            Ok(x.flatten_all()?
                .to_vec1::<f32>()?
                .iter()
                .map(|&v| f64::from(v))
                .collect())
        };
        let paths = self.paths()?.squeeze(0)?.to_vec3::<f32>()?;
        let (lb, lg) = self.log_rates()?;
        let rates =
            |x: &Tensor| -> Result<Vec<f64>> { Ok(row(x)?.into_iter().map(clamped_exp).collect()) };
        Ok(Params {
            beta: rates(&lb)?,
            gamma: rates(&lg)?,
            rho: self
                .rho
                .as_tensor()
                .squeeze(0)?
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
            log_capture: self.log_capture()?.clamp(-LOG_CLAMP, LOG_CLAMP),
        })
    }
}

/// `([B × L·n × G], [B × L·n × G])` the levels of every start: paths
/// `[B × L × n × H]` of durations `[B × L]`, `sub` positions a step, every
/// gene's `log β`, `log γ` and `b` `[B × 1 × G]` and `ρ` `[B × G × H]`, in
/// the paths' dtype.
///
/// `α` is constant over a step and every step is as long, so one step
/// is `x_{k+1} = A x_k + c α_k`, `x = (ū, s̄)`, the same `A` throughout:
/// `A^m` is `[[e^{−βmδ}, 0], [β E(mδ), e^{−γmδ}]]` and
/// `x_k = A^k x_0 + Σ_{j<k} A^{k−1−j} c α_j`, a scan solved for every gene
/// at once by doubling (`⌈log₂ n⌉` rounds, no loop over the path); the
/// positions inside a step are one exact part-step from its node, all at
/// once. Every start's every lineage is one row of the scan.
pub(super) fn closed_form_levels(
    paths: &Tensor,
    t_len: &Tensor,
    sub: usize,
    log_beta: &Tensor,
    log_gamma: &Tensor,
    rho: &Tensor,
    base: &Tensor,
) -> Result<(Tensor, Tensor)> {
    let (b, l0, n, h) = paths.dims4()?;
    let g = rho.dim(1)?;
    // Every start's every lineage a row of the scan: `l` rows.
    let l = b * l0;
    let (dev, dtype) = (paths.device(), paths.dtype());
    let small = if dtype == DType::F64 {
        SMALL_F64
    } else {
        SMALL
    };
    let alpha = clamped(
        &paths
            .reshape((b, l0 * n, h))?
            .matmul(&rho.transpose(1, 2)?.contiguous()?)?
            .broadcast_add(base)?,
    )?
    .exp()?
    .reshape((l, n, g))?;
    // A start's rates on every row of its lineages, `[l × 1 × G]`.
    let rows = |x: &Tensor| -> Result<Tensor> {
        clamped(x)?
            .exp()?
            .unsqueeze(1)?
            .broadcast_as((b, l0, 1, g))?
            .reshape((l, 1, g))
    };
    let (beta, gamma) = (rows(log_beta)?, rows(log_gamma)?);
    let step = (t_len.reshape((l, 1, 1))? / (n - 1) as f64)?; // [l × 1 × 1]
    let m = sub.max(1);
    // `e^{−βδ}`, `e^{−γδ}` and `E(δ) = (e^{−βδ} − e^{−γδ})/(γ − β)`
    // (`δ e^{−γδ} (1 + x/2 + x²/6)` near `x = (γ − β)δ = 0`, where the
    // difference cancels) for steps `δ` `[l × K × 1]`, each `[l × K × G]`.
    // Both exponentials are at most 1, so `E` cannot overflow however far
    // apart the rates are.
    let factors = |delta: &Tensor| -> Result<(Tensor, Tensor, Tensor)> {
        let bh = delta.broadcast_mul(&beta)?;
        let gh = delta.broadcast_mul(&gamma)?;
        let (eb, eg) = (bh.neg()?.exp()?, gh.neg()?.exp()?);
        let x = (&gh - &bh)?;
        let small = x.abs()?.lt(small)?;
        let gap = small.where_cond(&x.ones_like()?, &(&gamma - &beta)?.broadcast_as(x.shape())?)?;
        let series = ((x.affine(1.0 / 6.0, 0.5)? * &x)? + 1.0)?;
        let near = (delta.broadcast_mul(&eg)? * series)?;
        let e = small.where_cond(&near, &((&eb - &eg)? / gap)?)?;
        Ok((eb, eg, e))
    };
    // Every step length needed, in one call: the doublings' `dδ`
    // (`d = 1, 2, 4, …`; `d = 1` gives `c` too) and the part-steps' `jδ/m`.
    let doublings: Vec<usize> = std::iter::successors(Some(1), |d| Some(d * 2))
        .take_while(|&d| d < n.max(2))
        .collect();
    let multiples: Vec<f64> = doublings
        .iter()
        .map(|&d| d as f64)
        .chain((1..m).map(|j| j as f64 / m as f64))
        .collect();
    let k = multiples.len();
    let multiples = Tensor::from_vec(multiples, (1, k, 1), dev)?.to_dtype(dtype)?;
    let (eb_all, eg_all, e_all) = factors(&multiples.broadcast_mul(&step)?)?; // [l × K × G]
    let at = |t: &Tensor, i: usize, len: usize| t.narrow(1, i, len);
    // One step from rest, `c`, and the inputs `b_0 = x_0` (at rest),
    // `b_k = c α_{k−1}`, `[l × n × G]` per track.
    let (eb1, eg1, e1) = (at(&eb_all, 0, 1)?, at(&eg_all, 0, 1)?, at(&e_all, 0, 1)?);
    let c_u = (1.0 - &eb1)?.div(&beta)?;
    let c_s = ((1.0 - &eg1)?.div(&gamma)? - &e1)?;
    let (a0, a_in) = (alpha.narrow(1, 0, 1)?, alpha.narrow(1, 0, n - 1)?);
    let mut u = Tensor::cat(&[&a0.div(&beta)?, &a_in.broadcast_mul(&c_u)?], 1)?;
    let mut s = Tensor::cat(&[&a0.div(&gamma)?, &a_in.broadcast_mul(&c_s)?], 1)?;
    // `x_k = Σ_{j≤k} A^{k−j} b_j` by doubling: `x_k += A^d x_{k−d}` for
    // `d = 1, 2, 4, …`, `A^d x = (e^{−βdδ} ū, β E(dδ) ū + e^{−γdδ} s̄)`.
    for (i, &d) in doublings.iter().enumerate().filter(|_| n > 1) {
        let (eb, eg) = (at(&eb_all, i, 1)?, at(&eg_all, i, 1)?);
        let be = at(&e_all, i, 1)?.mul(&beta)?;
        let (up, sp) = (u.narrow(1, 0, n - d)?, s.narrow(1, 0, n - d)?);
        let du = up.broadcast_mul(&eb)?;
        let ds = (up.broadcast_mul(&be)? + sp.broadcast_mul(&eg)?)?;
        let pad = Tensor::zeros((l, d, g), dtype, dev)?;
        u = (u + Tensor::cat(&[&pad, &du], 1)?)?;
        s = (s + Tensor::cat(&[&pad, &ds], 1)?)?;
    }
    // Inside every step at once: from its node `x_k`, `α_k` held, every
    // part-step `jδ/m` along a new axis, `[l × (n − 1) × (m − 1) × G]`.
    let (uk, sk, ak) = (
        u.narrow(1, 0, n - 1)?.unsqueeze(2)?,
        s.narrow(1, 0, n - 1)?.unsqueeze(2)?,
        alpha.narrow(1, 0, n - 1)?.unsqueeze(2)?,
    );
    let (us, ss) = if m > 1 {
        let part = |t: &Tensor| at(t, doublings.len(), m - 1)?.unsqueeze(1); // [l × 1 × (m−1) × G]
        let (eb, eg, e) = (part(&eb_all)?, part(&eg_all)?, part(&e_all)?);
        let (beta4, gamma4) = (beta.unsqueeze(1)?, gamma.unsqueeze(1)?); // [l × 1 × 1 × G]
        let (a_b, a_g) = (ak.broadcast_div(&beta4)?, ak.broadcast_div(&gamma4)?);
        let up = a_b.broadcast_add(&uk.broadcast_sub(&a_b)?.broadcast_mul(&eb)?)?;
        let sp = ((sk.broadcast_mul(&eg)? + a_g.broadcast_mul(&(1.0 - &eg)?)?)?
            + (uk.broadcast_mul(&beta4)? - &ak)?.broadcast_mul(&e)?)?;
        (Tensor::cat(&[&uk, &up], 2)?, Tensor::cat(&[&sk, &sp], 2)?)
    } else {
        (uk, sk)
    };
    let positions = (n - 1) * m + 1;
    let stack = |v: &Tensor, last: &Tensor| -> Result<Tensor> {
        let inner = v.reshape((l, (n - 1) * m, g))?;
        Tensor::cat(&[&inner, &last.narrow(1, n - 1, 1)?], 1)?.reshape((b, l0 * positions, g))
    };
    Ok((stack(&us, &u)?, stack(&ss, &s)?))
}

/// `x` clamped to `±`[`LOG_CLAMP`] before an `exp`.
fn clamped(x: &Tensor) -> Result<Tensor> {
    x.clamp(-LOG_CLAMP, LOG_CLAMP)
}
