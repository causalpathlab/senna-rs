//! The sampler's blocks, each a draw from its conditional.
//!
//! Every random stream is keyed by `(seed, item, sweep)`, so a draw does
//! not depend on how the items were split across threads.

use super::band::{dot, Band, Bordered, Factor};
use super::kernel::{GridPrior, BAND};
use super::kinetics::fine_trajectory;
use super::likelihood::{share_mode, softplus, Depth, GeneRows, Reads};
use super::model::{alpha_on, branch_ramp, clamped_exp, Params, LOG_CLAMP};
use legume_numeric::mcmc::engine::elliptical_slice_step;
use nalgebra::{DMatrix, DVector};
use rand::rngs::StdRng;
use rand::{Rng, RngExt, SeedableRng};
use rand_distr::{Distribution, Gamma, StandardNormal};
use rayon::prelude::*;

/// A stream for one item of one sweep.
pub fn stream(seed: u64, item: u64, sweep: u64) -> StdRng {
    let mix = |x: u64| {
        let x = (x ^ (x >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        let x = (x ^ (x >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        x ^ (x >> 31)
    };
    StdRng::seed_from_u64(mix(seed ^ mix(item ^ mix(sweep))))
}

/// Pseudobulk `p`'s log posterior over every row `ℓ n + k`, `tab + log π_ℓ`
/// normalised (time uniform on the grid), and the log normaliser
/// `log Σ_r e^{tab + log π}` (the grid's `1/n` aside). A row with every
/// entry `−∞` gives `−∞` throughout, not NaN.
pub fn log_posterior_row(
    tab: &DMatrix<f64>,
    log_w: &[f64],
    points: usize,
    p: usize,
) -> (Vec<f64>, f64) {
    let logit: Vec<f64> = (0..tab.ncols())
        .map(|r| tab[(p, r)] + log_w[r / points])
        .collect();
    let top = logit.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    if !top.is_finite() {
        return (vec![f64::NEG_INFINITY; logit.len()], f64::NEG_INFINITY);
    }
    let norm = top + logit.iter().map(|x| (x - top).exp()).sum::<f64>().ln();
    (logit.iter().map(|x| x - norm).collect(), norm)
}

/// Every pseudobulk's row `ℓ n + k` drawn from `table + log π_ℓ` (time
/// uniform on the grid), in parallel over pseudobulks.
pub fn draw_latents(
    tab: &DMatrix<f64>,
    log_w: &[f64],
    points: usize,
    seed: u64,
    sweep: u64,
) -> Vec<usize> {
    (0..tab.nrows())
        .into_par_iter()
        .map(|p| {
            let (log_post, _) = log_posterior_row(tab, log_w, points, p);
            let w: Vec<f64> = log_post.iter().map(|x| x.exp()).collect();
            let mut rng = stream(seed, p as u64, sweep);
            let mut at = rng.random::<f64>() * w.iter().sum::<f64>();
            for (r, &x) in w.iter().enumerate() {
                if at < x {
                    return r;
                }
                at -= x;
            }
            w.len() - 1
        })
        .collect()
}

/// The lineages' log weights drawn from `Dirichlet(a0 + n_ℓ)`.
pub fn draw_weights(
    rows: &[usize],
    lineages: usize,
    points: usize,
    a0: f64,
    seed: u64,
    sweep: u64,
) -> Vec<f64> {
    let mut n = vec![0f64; lineages];
    for &r in rows {
        n[r / points] += 1.0;
    }
    let mut rng = stream(seed, u64::MAX, sweep);
    let g: Vec<f64> = n
        .iter()
        .map(|&c| {
            Gamma::new(a0 + c, 1.0)
                .expect("a positive shape")
                .sample(&mut rng)
        })
        .collect();
    let total: f64 = g.iter().sum();
    g.iter().map(|x| (x / total).ln()).collect()
}

/// A gene's parameters as the sampler moves them, each with a meaning of
/// its own:
///
/// ```text
/// θ = (log γ, log γ/β, ρ, b̃),   b̃ = b + ρᵀz̄ − log γ
/// ```
///
/// the speed, the steady unspliced share, the embedding and the mean log
/// spliced level (`z̄` the paths' mean over every grid point). Two ridges
/// are taken out so: scaling both rates and the transcription alike (a gene
/// near steady state stays as it is) is `log γ` alone, and the paths' mean
/// trading against the baseline is gone (`ρ` shapes the change over time
/// only).
pub fn to_theta(beta: f64, gamma: f64, rho: &[f64], base: f64, z_bar: &[f64]) -> Vec<f64> {
    let mut t = vec![gamma.ln(), (gamma / beta).ln()];
    t.extend_from_slice(rho);
    t.push(base + dot(rho, z_bar) - gamma.ln());
    t
}

/// `(β, γ, ρ, b)` from [`to_theta`]'s coordinates.
pub fn from_theta<'a>(theta: &'a [f64], z_bar: &[f64]) -> (f64, f64, &'a [f64], f64) {
    let h = theta.len() - 3;
    let gamma = clamped_exp(theta[0]);
    let rho = &theta[2..2 + h];
    let base = theta[2 + h] + theta[0] - dot(rho, z_bar);
    (gamma / clamped_exp(theta[1]), gamma, rho, base)
}

/// The normal prior on a gene's [`to_theta`] coordinates, `(mean, sd)` each.
#[derive(Clone, Debug)]
pub struct GenePrior {
    pub log_gamma: (f64, f64),
    pub log_ratio: (f64, f64),
    pub rho_sd: f64,
    pub log_spliced: (f64, f64),
    /// Bounds `(lo, hi)` on `log γ` and `log β` ([`GenePrior::bounded`]),
    /// carried by the sampler's coordinates ([`GenePrior::link`]); `None`:
    /// unbounded.
    pub log_rate: Option<(f64, f64)>,
}

impl Default for GenePrior {
    fn default() -> Self {
        Self {
            log_gamma: (2f64.ln(), 1.5),
            log_ratio: (0.5f64.ln(), 1.0),
            rho_sd: 1.0,
            log_spliced: (0.0, 1.5),
            log_rate: None,
        }
    }
}

/// A rate's half-life is held above one fine step and below this many path
/// durations ([`GenePrior::bounded`]): faster, the reads follow `α` in
/// quasi-steady state; slower, they accumulate `α` with no decay seen
/// (over four durations a level decays by under a sixth). Either way the
/// rate runs along a flat ridge only the weak prior holds.
const SLOW_HALF_LIVES: f64 = 4.0;

/// How near a bound a rate outside it is taken in ([`GenePrior::link`]).
const LINK_EDGE: f64 = 1e-9;

/// Where `x` sits between `lo` and `hi`, as a share kept `edge` off either.
fn share_of(x: f64, lo: f64, hi: f64, edge: f64) -> f64 {
    ((x - lo) / (hi - lo)).clamp(edge, 1.0 - edge)
}

/// `log(p / (1 − p))`, [`sigmoid`]'s inverse.
fn logit(p: f64) -> f64 {
    (p / (1.0 - p)).ln()
}

impl GenePrior {
    /// The default prior with `log γ` and `log β` bounded, fixed for a
    /// run: a half-life of at least one fine step and at most
    /// [`SLOW_HALF_LIVES`] durations, both on the time unit, the first
    /// lineage's duration (every fit holds it at its start's; the others
    /// are carried relative to it). The other lineages' durations move in
    /// the climb and the chains, so they are left out: the bounds stay
    /// the same at every stage.
    pub fn bounded(params: &Params) -> Self {
        let steps = params.positions().saturating_sub(1).max(1) as f64;
        let unit = params.t_len[0];
        let ln2 = std::f64::consts::LN_2;
        Self {
            log_rate: Some((
                (ln2 / (SLOW_HALF_LIVES * unit)).ln(),
                (ln2 * steps / unit).ln(),
            )),
            ..Self::default()
        }
    }

    /// The sampler's coordinates of `θ` ([`to_theta`]): with bounds, `log γ`
    /// and `log β = θ₀ − θ₁` each as `u`, `log r = lo + (hi − lo) σ(u)`, so
    /// both rates stay inside (a rate outside is taken in to the edge);
    /// the rest as they are.
    pub fn link(&self, theta: &[f64]) -> Vec<f64> {
        let mut v = theta.to_vec();
        if let Some((lo, hi)) = self.log_rate {
            let unit = |x: f64| logit(share_of(x, lo, hi, LINK_EDGE));
            (v[0], v[1]) = (unit(theta[0]), unit(theta[0] - theta[1]));
        }
        v
    }

    /// `θ` from the sampler's coordinates ([`Self::link`]).
    pub fn unlink(&self, v: &[f64]) -> Vec<f64> {
        let mut theta = v.to_vec();
        if let Some((lo, hi)) = self.log_rate {
            let rate = |u: f64| lo + (hi - lo) * sigmoid(u);
            let (log_gamma, log_beta) = (rate(v[0]), rate(v[1]));
            (theta[0], theta[1]) = (log_gamma, log_gamma - log_beta);
        }
        theta
    }

    /// The prior's log density in the sampler's coordinates: at `θ` with
    /// the link's log-Jacobian, `Σ log((hi − lo) σ(u)(1 − σ(u)))` over the
    /// two rates (the map from `(log γ, log β)` to `θ`'s has unit Jacobian).
    pub(super) fn linked_log_density(&self, v: &[f64]) -> f64 {
        let jacobian = self.log_rate.map_or(0.0, |(lo, hi)| {
            [v[0], v[1]]
                .iter()
                .map(|&u| (hi - lo).ln() - softplus(u) - softplus(-u))
                .sum()
        });
        self.log_density(&self.unlink(v)) + jacobian
    }

    /// A normal in the sampler's coordinates near the prior, `(mean, sd)`
    /// each (the scalar slice moves' scale): the prior's own unbounded;
    /// with bounds, the rates' prior carried through the link at its mean.
    fn reference(&self, h: usize) -> (Vec<f64>, Vec<f64>) {
        let (mut mean, mut sd) = (self.mean(h), self.sd(h));
        if let Some((lo, hi)) = self.log_rate {
            let (mg, sg) = self.log_gamma;
            let (mr, sr) = self.log_ratio;
            for (i, (m, s)) in [(mg, sg), (mg - mr, sg.hypot(sr))].into_iter().enumerate() {
                let p = share_of(m, lo, hi, 0.05);
                mean[i] = logit(p);
                sd[i] = s / ((hi - lo) * p * (1.0 - p));
            }
        }
        (mean, sd)
    }
}

impl GenePrior {
    /// The prior mean, `[2 + H + 1]`.
    pub fn mean(&self, h: usize) -> Vec<f64> {
        let mut m = vec![self.log_gamma.0, self.log_ratio.0];
        m.extend(std::iter::repeat_n(0.0, h));
        m.push(self.log_spliced.0);
        m
    }

    fn sd(&self, h: usize) -> Vec<f64> {
        let mut s = vec![self.log_gamma.1, self.log_ratio.1];
        s.extend(std::iter::repeat_n(self.rho_sd, h));
        s.push(self.log_spliced.1);
        s
    }
}

/// Gene `g`'s log-likelihood at `θ` from its reads by row, every other
/// parameter from `params` (`z_bar` their paths' mean).
fn gene_ll(params: &Params, z_bar: &[f64], theta: &[f64], by_row: &GeneRows) -> f64 {
    let (beta, gamma, rho, base) = from_theta(theta, z_bar);
    let lv = params.levels_of(beta, gamma, rho, base);
    by_row.log_likelihood((&lv.0, &lv.1))
}

impl GenePrior {
    /// `log p₀(θ)` up to a constant.
    pub(super) fn log_density(&self, theta: &[f64]) -> f64 {
        let h = theta.len() - 3;
        let (m, s) = (self.mean(h), self.sd(h));
        theta
            .iter()
            .zip(m.iter().zip(&s))
            .map(|(t, (m, s))| -0.5 * ((t - m) / s).powi(2))
            .sum()
    }
}

/// Where a gene's conditional is, as a normal in the sampler's
/// coordinates ([`GenePrior::link`]): its mode (`θ`, and linked as
/// `centre`) and the Cholesky factor `L` of the precision `P = L Lᵀ` (the
/// negative Hessian there, over [`WIDEN`]²).
#[derive(Clone)]
pub struct GeneProposal {
    pub mode: Vec<f64>,
    centre: Vec<f64>,
    /// The proposal's precision, factored.
    precision: Bordered,
}

/// The proposal's sd over the Laplace approximation's: heavier than the
/// conditional, so its tails are covered.
const WIDEN: f64 = 1.5;

/// Gradient steps at most on a path's one frozen precision per refit
/// ([`frozen_steps`]): a banded precision is a rougher Newton matrix than
/// the full one, so the mode takes a few more; they stop once a step rises
/// less than [`STEP_GAIN`].
const PATH_STEPS: usize = 8;

/// A rise (nats) under which the frozen steps stop: the mode is then close
/// enough for the proposal (the slice move is exact on any).
const STEP_GAIN: f64 = 1e-2;

/// A failed step is halved this many times (down to about a thousandth of
/// its length) before the search stops: a well-read gene or path's
/// posterior can be far narrower than the capped step.
const HALVINGS: usize = 10;

/// A step's largest move in any coordinate (about the prior's sd): the
/// steps cannot run away where the frozen precision is far off.
const STEP_CAP: f64 = 1.0;

/// An objective's value and gradient at a point.
type Objective<'a> = dyn FnMut(&[f64]) -> (f64, Vec<f64>) + 'a;

/// At most `steps` of `x += P⁻¹ g` from `start` (its value and gradient
/// `at_start`), `solve` applying the frozen `P⁻¹`, `eval` the objective's
/// value and gradient; a step is cut to [`STEP_CAP`] and one that does
/// not rise is halved ([`HALVINGS`]), the point kept if none does. Stops
/// on a rise under [`STEP_GAIN`], or where the value or gradient is not
/// finite (a rate overflowing); the best point met.
fn frozen_steps(
    start: &[f64],
    at_start: (f64, Vec<f64>),
    steps: usize,
    solve: &dyn Fn(&[f64]) -> Vec<f64>,
    eval: &mut Objective,
) -> Vec<f64> {
    let mut x = start.to_vec();
    let (mut f, mut g) = at_start;
    let finite = |f: f64, g: &[f64]| f.is_finite() && g.iter().all(|v| v.is_finite());
    if !finite(f, &g) {
        return x;
    }
    for _ in 0..steps {
        let step = solve(&g);
        let top = step.iter().fold(0.0f64, |m, v| m.max(v.abs()));
        let Some((y, fy, gy, _)) = capped_search(&x, f, &step, top, |y| {
            let (fy, gy) = eval(y);
            finite(fy, &gy).then_some((fy, gy))
        }) else {
            break;
        };
        let gain = fy - f;
        (x, f, g) = (y, fy, gy);
        if gain < STEP_GAIN {
            break;
        }
    }
    x
}

/// A step's line search: from `x` (its value `fx`) along `step`, cut so
/// its largest move `top` is at most [`STEP_CAP`], then halved
/// ([`HALVINGS`]) until `eval` (`None` where not finite) rises; the point,
/// its value and `eval`'s rest, and the scale taken; `None` if none does.
fn capped_search<T>(
    x: &[f64],
    fx: f64,
    step: &[f64],
    top: f64,
    mut eval: impl FnMut(&[f64]) -> Option<(f64, T)>,
) -> Option<(Vec<f64>, f64, T, f64)> {
    let mut scale = if top > STEP_CAP { STEP_CAP / top } else { 1.0 };
    for _ in 0..=HALVINGS {
        let y: Vec<f64> = x.iter().zip(step).map(|(a, b)| a + scale * b).collect();
        if let Some((fy, rest)) = eval(&y) {
            if fy > fx {
                return Some((y, fy, rest, scale));
            }
        }
        scale *= 0.5;
    }
    None
}

/// Newton iterations at most for a gene's mode ([`gene_mode`]).
const GENE_ITERS: usize = 50;

/// The Laplace approximation of a gene's conditional (likelihood from its
/// reads by row, times prior; `z_bar` the paths' mean), by bounded Newton
/// from `anchor`, numerical derivatives (a gene has `H + 3` coordinates):
/// every step's Hessian factored guarded ([`Factor::new`], the prior's
/// precision the fallback), the step cut to [`STEP_CAP`] prior sds in any
/// coordinate and halved ([`HALVINGS`]) until the objective rises, else
/// the search stops (and where a value is not finite). The sampler fits it
/// only during burn-in ([`super::sampler::Options::laplace_every`]) and
/// keeps the last one after, so the kept moves are one fixed kernel.
pub fn gene_mode(
    params: &Params,
    by_row: &GeneRows,
    z_bar: &[f64],
    prior: &GenePrior,
    anchor: &[f64],
) -> GeneProposal {
    // In the sampler's coordinates.
    let f =
        |v: &[f64]| gene_ll(params, z_bar, &prior.unlink(v), by_row) + prior.linked_log_density(v);
    let n = anchor.len();
    let (_, sd) = prior.reference(n - 3);
    let prior_precision = || Band::diagonal_of(&sd.iter().map(|s| s.powi(-2)).collect::<Vec<_>>());
    let factor =
        |p: &DMatrix<f64>| Bordered::new(Factor::new(&Band::from_dense(p), prior_precision), None);
    let eps = 1e-4;
    // At `t`, `f0 = f(t)` known.
    let hessian = |t: &[f64], f0: f64| -> (DVector<f64>, DMatrix<f64>) {
        let at = |d: &[(usize, f64)]| {
            let mut x = t.to_vec();
            for &(i, v) in d {
                x[i] += v;
            }
            f(&x)
        };
        // `f(t ± ε e_i)`, shared by the gradient and the diagonal.
        let (up, down): (Vec<f64>, Vec<f64>) =
            (0..n).map(|i| (at(&[(i, eps)]), at(&[(i, -eps)]))).unzip();
        let grad = DVector::from_fn(n, |i, _| (up[i] - down[i]) / (2.0 * eps));
        let mut hess = DMatrix::zeros(n, n);
        for i in 0..n {
            hess[(i, i)] = (up[i] - 2.0 * f0 + down[i]) / (eps * eps);
            for j in 0..i {
                let v = (at(&[(i, eps), (j, eps)])
                    - at(&[(i, eps), (j, -eps)])
                    - at(&[(i, -eps), (j, eps)])
                    + at(&[(i, -eps), (j, -eps)]))
                    / (4.0 * eps * eps);
                hess[(i, j)] = v;
                hess[(j, i)] = v;
            }
        }
        (grad, hess)
    };
    let mut t = prior.link(anchor);
    let mut ft = f(&t);
    let (mut grad, mut hess) = hessian(&t, ft);
    for _ in 0..GENE_ITERS {
        let finite = ft.is_finite() && grad.iter().chain(hess.iter()).all(|v| v.is_finite());
        if !finite {
            break;
        }
        let step = factor(&(-&hess)).solve(grad.as_slice());
        let top = step
            .iter()
            .zip(&sd)
            .fold(0.0f64, |m, (v, s)| m.max(v.abs() / s));
        let found = capped_search(&t, ft, &step, top, |x| {
            let fx = f(x);
            fx.is_finite().then_some((fx, ()))
        });
        let Some((x, fx, (), scale)) = found else {
            break;
        };
        let size = scale * step.iter().map(|v| v * v).sum::<f64>().sqrt();
        (t, ft) = (x, fx);
        (grad, hess) = hessian(&t, ft);
        if size < 1e-7 {
            break;
        }
    }
    let precision = factor(&(-hess / (WIDEN * WIDEN)));
    GeneProposal {
        mode: prior.unlink(&t),
        centre: t,
        precision,
    }
}

impl GeneProposal {
    /// `log N(v; centre, P⁻¹)` up to a constant, `v` linked.
    fn log_density(&self, v: &[f64]) -> f64 {
        let d: Vec<f64> = v.iter().zip(&self.centre).map(|(a, b)| a - b).collect();
        -0.5 * self.precision.quad(&d)
    }

    /// A draw of `θ − mode`: `L⁻ᵀ z`, `z ~ N(0, I)`.
    fn draw(&self, rng: &mut impl Rng) -> DVector<f64> {
        let z: Vec<f64> = (0..self.centre.len())
            .map(|_| StandardNormal.sample(rng))
            .collect();
        DVector::from_vec(self.precision.draw(&z))
    }
}

/// Degrees of freedom of the Student-t ellipse in [`gene_step`].
const T_DF: f64 = 4.0;

/// One move of gene `g`'s `θ`: a scalar slice move per coordinate on the
/// prior's scale (what mixes where the prior shapes a direction), then one
/// by generalized elliptical slice sampling
/// (Nishihara, Murray and Adams, *JMLR* (2014)): a Student-t centred on the
/// conditional's mode and shaped like it (`prop`), as a scale mixture of
/// normals. The scale `s` is drawn given `θ`,
///
/// ```text
/// s ~ InvGamma((ν + d)/2, (ν + Δ²)/2),   Δ² = (θ − m)ᵀ P (θ − m)
/// ```
///
/// then one elliptical slice move on `N(m, s P⁻¹)` for `π(θ) / t_ν(θ)`. The
/// t's tails are heavier than the conditional's, so that ratio stays bounded
/// and a chain anywhere, far in the tails too, comes in; near the mode it
/// moves along the conditional's ridges. The latents and depths are held.
/// `cur_ll` may be `−∞` (computed then). Returns the new `θ` and its
/// log-likelihood.
#[allow(clippy::too_many_arguments)]
pub fn gene_step(
    params: &Params,
    theta: &[f64],
    cur_ll: f64,
    by_row: &GeneRows,
    z_bar: &[f64],
    prior: &GenePrior,
    prop: &GeneProposal,
    rng: &mut impl Rng,
) -> (Vec<f64>, f64) {
    let n = theta.len();
    let d = n as f64;
    // In the sampler's coordinates ([`GenePrior::link`]); the scalar moves'
    // normal is the reference ([`GenePrior::reference`]), the prior's
    // difference from it (none unbounded) taken with the reads.
    let (mean, sd) = prior.reference(n - 3);
    let log_ref = |v: &[f64]| -> f64 {
        v.iter()
            .zip(mean.iter().zip(&sd))
            .map(|(t, (m, s))| -0.5 * ((t - m) / s).powi(2))
            .sum()
    };
    let bounded = prior.log_rate.is_some();
    let ll = |v: &[f64]| {
        let extra = if bounded {
            prior.linked_log_density(v) - log_ref(v)
        } else {
            0.0
        };
        gene_ll(params, z_bar, &prior.unlink(v), by_row) + extra
    };
    let theta = &prior.link(theta)[..];
    let mut cur = theta.to_vec();
    let mut cur_ll = if cur_ll.is_finite() {
        cur_ll
    } else {
        ll(theta)
    };

    // A scalar slice move per coordinate on the prior's scale first: where
    // the prior, not the reads, shapes a direction, this is what mixes.
    for j in 0..n {
        let x = DVector::from_element(1, (cur[j] - mean[j]) as f32);
        let e: f64 = StandardNormal.sample(rng);
        let nu = DVector::from_element(1, (sd[j] * e) as f32);
        let base = cur_ll;
        let lnpdf = |y: &DVector<f32>| {
            let mut t = cur.clone();
            t[j] = f64::from(y[0]) + mean[j];
            (ll(&t) - base) as f32
        };
        let (next, rel) = elliptical_slice_step(&x, &nu, &lnpdf, 0.0, rng);
        cur[j] = f64::from(next[0]) + mean[j];
        cur_ll = base + f64::from(rel);
    }
    let theta = &cur[..];
    // log t_ν(θ) up to a constant, from the normal's `−Δ²/2`.
    let log_t = |t: &[f64]| -0.5 * (T_DF + d) * (1.0 - 2.0 * prop.log_density(t) / T_DF).ln();
    let tilt = |t: &[f64]| log_ref(t) - log_t(t);

    let delta2 = -2.0 * prop.log_density(theta);
    let shape = 0.5 * (T_DF + d);
    let rate = 0.5 * (T_DF + delta2);
    let scale = rate
        / Gamma::new(shape, 1.0)
            .expect("a positive shape")
            .sample(rng);

    let base = cur_ll + tilt(theta);
    let at = |y: &DVector<f32>| -> Vec<f64> {
        y.iter()
            .zip(&prop.centre)
            .map(|(&v, m)| f64::from(v) + m)
            .collect()
    };
    let x = DVector::from_iterator(
        n,
        theta.iter().zip(&prop.centre).map(|(t, m)| (t - m) as f32),
    );
    let nu = (prop.draw(rng) * scale.sqrt()).map(|v| v as f32);
    // Relative to the current value, so f32 keeps the slice's resolution.
    let lnpdf = |y: &DVector<f32>| {
        let t = at(y);
        (ll(&t) + tilt(&t) - base) as f32
    };
    let (next, _) = elliptical_slice_step(&x, &nu, &lnpdf, 0.0, rng);
    let t = at(&next);
    let new_ll = ll(&t);
    (prior.unlink(&t), new_ll)
}

/// The pseudobulks whose rows are on lineage `l` (`positions` rows a
/// lineage), each with its position there: `(p, k)`.
pub(super) fn on_lineage(rows: &[usize], positions: usize, l: usize) -> Vec<(usize, usize)> {
    rows.iter()
        .enumerate()
        .filter(|&(_, &r)| r / positions == l)
        .map(|(p, &r)| (p, r - l * positions))
        .collect()
}

/// The log-likelihood (no `log x!`) of the pseudobulks on lineage `l`, at
/// their rows and depths, every gene's levels on `l` from `params`; in
/// parallel over genes.
pub fn path_ll(params: &Params, l: usize, reads: &Reads, rows: &[usize], lambda: &[Depth]) -> f64 {
    let (n, positions) = (params.path[l].len(), params.positions());
    let on = on_lineage(rows, positions, l);
    let h = params.t_len[l] / (n - 1) as f64;
    (0..params.beta.len())
        .into_par_iter()
        .map(|g| {
            let alpha = alpha_on(&params.path[l], &params.rho[g], params.base[g]);
            let (u, s) = fine_trajectory(&alpha, params.beta[g], params.gamma[g], h, params.sub);
            on.iter()
                .map(|&(p, k)| {
                    let (mu, ms) = (lambda[p][0] * u[k], lambda[p][1] * s[k]);
                    reads.u[(p, g)] * mu.ln() - mu + reads.s[(p, g)] * ms.ln() - ms
                })
                .sum::<f64>()
        })
        .sum()
}

/// `[n][H]` the path from its whitened coordinates, one `[n]` per dimension.
fn coloured(prior: &GridPrior, eta: &[DVector<f64>]) -> Vec<Vec<f64>> {
    let dims: Vec<DVector<f64>> = eta.iter().map(|e| prior.colour(e)).collect();
    let n = dims.first().map_or(0, |d| d.len());
    (0..n)
        .map(|k| dims.iter().map(|d| d[k]).collect())
        .collect()
}

/// The genes' prior at the current paths, `Σ_g log p₀(θ_g)`: through
/// `b̃ = b + ρᵀz̄ − log γ` it moves with the paths' mean `z̄`.
pub fn genes_log_prior(params: &Params, prior: &GenePrior) -> f64 {
    let z_bar = params.path_mean();
    (0..params.beta.len())
        .map(|g| {
            prior.log_density(&to_theta(
                params.beta[g],
                params.gamma[g],
                &params.rho[g],
                params.base[g],
                &z_bar,
            ))
        })
        .sum()
}

/// The lineages path block `block`'s whitened coordinates move: its own;
/// on a trunk ([`Params::branch`]) the trunk's (block 0) moves every one.
pub fn moved_by(params: &Params, block: usize) -> Vec<usize> {
    match params.branch {
        Some(_) if block == 0 => (0..params.path.len()).collect(),
        _ => vec![block],
    }
}

/// The map `M`, `[n × n]`, from path block `block`'s whitened coordinates
/// (one dimension) to lineage `lineage`'s path, `z = offset + M η`: `C`,
/// or for a departure from the trunk `diag(r) C` ([`branch_ramp`]).
fn path_map(params: &Params, prior: &GridPrior, block: usize) -> DMatrix<f64> {
    let c = prior.chol();
    match &params.branch {
        Some(branch) if block > 0 => {
            let r = branch_ramp(c.nrows(), branch[block - 1], prior.kappa());
            DMatrix::from_fn(c.nrows(), c.ncols(), |i, j| r[i] * c[(i, j)])
        }
        _ => c.clone(),
    }
}

/// Departures from the trunk are read back to within this when a chain
/// starts ([`path_eta`]).
const DEPARTURE_FIT: f64 = 1e-3;

/// Path block `block`'s whitened coordinates at `params`' paths, one `[n]`
/// per dimension: the path whitened, or for a departure from the trunk the
/// most probable `η` that gives it ([`GridPrior::whiten_scaled`]; the
/// trunk's points say nothing of it).
pub fn path_eta(params: &Params, prior: &GridPrior, block: usize) -> Vec<DVector<f64>> {
    let path = &params.path[block];
    let (n, h) = (
        path.len(),
        params.rho.first().map_or(path[0].len(), Vec::len),
    );
    match &params.branch {
        Some(branch) if block > 0 => {
            let r = branch_ramp(n, branch[block - 1], prior.kappa());
            (0..h)
                .map(|d| {
                    let dep = DVector::from_fn(n, |k, _| path[k][d] - params.path[0][k][d]);
                    prior.whiten_scaled(&r, &dep, DEPARTURE_FIT)
                })
                .collect()
        }
        _ => (0..h)
            .map(|d| prior.whiten(&DVector::from_fn(n, |k, _| path[k][d])))
            .collect(),
    }
}

/// Path block `block` set to whitened `eta`, every lineage it moves: its
/// own path `C η`; on a trunk the trunk's moves every lineage with their
/// departures in `base` kept, and a departure's lineage is the trunk plus
/// `diag(r) C η`.
pub fn place_path(
    params: &mut Params,
    base: &Params,
    prior: &GridPrior,
    block: usize,
    eta: &[DVector<f64>],
) {
    match &params.branch {
        Some(_) if block == 0 => {
            let trunk = coloured(prior, eta);
            for l in 1..params.path.len() {
                params.path[l] = trunk
                    .iter()
                    .zip(base.path[l].iter().zip(&base.path[0]))
                    .map(|(z0, (zl, b0))| {
                        z0.iter()
                            .zip(zl.iter().zip(b0))
                            .map(|(a, (x, y))| a + x - y)
                            .collect()
                    })
                    .collect();
            }
            params.path[0] = trunk;
        }
        Some(_) => {
            let m = path_map(params, prior, block);
            let dims: Vec<DVector<f64>> = eta.iter().map(|e| &m * e).collect();
            params.path[block] = (0..m.nrows())
                .map(|k| {
                    dims.iter()
                        .zip(&params.path[0][k])
                        .map(|(d, z0)| z0 + d[k])
                        .collect()
                })
                .collect();
        }
        None => params.path[block] = coloured(prior, eta),
    }
}

/// Path block `block`'s conditional up to a constant, `η`'s own prior
/// aside: [`path_ll`] on every lineage it moves plus [`genes_log_prior`].
pub fn path_log_density(
    params: &Params,
    block: usize,
    reads: &Reads,
    rows: &[usize],
    lambda: &[Depth],
    gene_prior: &GenePrior,
) -> f64 {
    moved_by(params, block)
        .into_iter()
        .map(|l| path_ll(params, l, reads, rows, lambda))
        .sum::<f64>()
        + genes_log_prior(params, gene_prior)
}

/// One elliptical slice move of lineage `l`'s whitened path `η` (every
/// dimension at once, prior `N(0, I)`) for [`path_log_density`], genes,
/// latents and depths held; `params.path[l]` is left at the new path.
/// `cur_ll` (that density) may be `−∞`.
#[allow(clippy::too_many_arguments)]
pub fn path_step(
    params: &mut Params,
    l: usize,
    eta: &[DVector<f64>],
    cur_ll: f64,
    prior: &GridPrior,
    reads: &Reads,
    rows: &[usize],
    lambda: &[Depth],
    gene_prior: &GenePrior,
    rng: &mut impl Rng,
) -> (Vec<DVector<f64>>, f64) {
    let (h, n) = (eta.len(), eta.first().map_or(0, |e| e.len()));
    let base = params.clone();
    place_path(params, &base, prior, l, eta);
    let cur_ll = if cur_ll.is_finite() {
        cur_ll
    } else {
        path_log_density(params, l, reads, rows, lambda, gene_prior)
    };
    let split = |x: &DVector<f32>| -> Vec<DVector<f64>> {
        (0..h)
            .map(|d| DVector::from_fn(n, |k, _| f64::from(x[d * n + k])))
            .collect()
    };
    let x = DVector::from_fn(h * n, |i, _| eta[i / n][i % n] as f32);
    let nu = DVector::from_fn(h * n, |_, _| {
        let e: f64 = StandardNormal.sample(rng);
        e as f32
    });
    let probe = params.clone();
    let lnpdf = |y: &DVector<f32>| {
        let mut p = probe.clone();
        place_path(&mut p, &probe, prior, l, &split(y));
        (path_log_density(&p, l, reads, rows, lambda, gene_prior) - cur_ll) as f32
    };
    let (next, rel) = elliptical_slice_step(&x, &nu, &lnpdf, 0.0, rng);
    let eta = split(&next);
    place_path(params, &probe, prior, l, &eta);
    (eta, cur_ll + f64::from(rel))
}

/// `(A_u, A_s)`, `[positions × n]` each: the trajectory is linear in the
/// transcription at the nodes for fixed rates, `ū = A_u α`, `s̄ = A_s α`
/// ([`fine_trajectory`] from the steady state of `α_0`, at every position);
/// column `j` the response to `α = e_j`.
pub(super) fn response(
    beta: f64,
    gamma: f64,
    h: f64,
    n: usize,
    sub: usize,
) -> (DMatrix<f64>, DMatrix<f64>) {
    let sub = sub.max(1);
    let positions = (n - 1) * sub + 1;
    let (mut a_u, mut a_s) = (DMatrix::zeros(positions, n), DMatrix::zeros(positions, n));
    // The first node sets the start too, the last nothing past it (its
    // column zero); a pulse between them is the first such pulse later by
    // whole steps (exact zeros before it: the system is time-invariant from
    // rest), so two trajectories make every column.
    for j in 0..(n - 1).min(2) {
        let mut e = vec![0.0; n];
        e[j] = 1.0;
        let (u, s) = fine_trajectory(&e, beta, gamma, h, sub);
        a_u.column_mut(j).copy_from_slice(&u);
        a_s.column_mut(j).copy_from_slice(&s);
    }
    for j in 2..n.saturating_sub(1) {
        let shift = (j - 1) * sub;
        for k in shift..positions {
            a_u[(k, j)] = a_u[(k - shift, 1)];
            a_s[(k, j)] = a_s[(k - shift, 1)];
        }
    }
    (a_u, a_s)
}

/// `(A_u, A_s)` ([`response`]) of every gene on lineage `l`: they move with
/// the rates and the duration only, so a path's mode search reuses them.
fn responses(params: &Params, l: usize) -> Vec<(DMatrix<f64>, DMatrix<f64>)> {
    let n = params.path[l].len();
    let step = params.t_len[l] / (n - 1) as f64;
    (0..params.beta.len())
        .into_par_iter()
        .map(|g| response(params.beta[g], params.gamma[g], step, n, params.sub))
        .collect()
}

/// One pass of lineage `l`'s reads ([`lineage_reads`]).
pub(super) struct LineageReads {
    /// [`path_ll`] on `l` but for a constant, `Σ u log λ`.
    pub ll: f64,
    /// `[n × H]` its gradient in `z`.
    pub grad: DMatrix<f64>,
    /// Its Fisher information in `z`, node-major (`k H + d`), banded to
    /// [`BAND`] nodes, when asked for.
    pub fisher: Option<Band>,
}

/// Lineage `l`'s reads in its path `z`, rows and depths held, `resp` its
/// genes' [`responses`]: the log-likelihood, its gradient and (with
/// `fisher`) a banded Fisher information: every position's response cut
/// to the [`BAND`] nodes around it, which keeps it positive semidefinite
/// (cutting the information itself would not).
///
/// The reads on `l` pooled per position, `U_gk = Σ_{p at k} u_pg`,
/// `Λ^u_k = Σ_{p at k} λ^u_p` (the spliced track's likewise):
/// `ℓ = Σ_{g,k} U_gk log ū_gk − Λ^u_k ū_gk (+ s̄)`,
/// `∂ℓ/∂ū_gk = U_gk/ū_gk − Λ^u_k`, Fisher `Λ^u_k/ū_gk`; then through
/// `ū = A_u α` and `α_gj = exp(ρ_gᵀ z_j + b_g)` ([`clamped_exp`]).
#[allow(clippy::too_many_arguments)]
pub(super) fn lineage_reads(
    params: &Params,
    l: usize,
    resp: &[(DMatrix<f64>, DMatrix<f64>)],
    reads: &Reads,
    rows: &[usize],
    lambda: &[Depth],
    fisher: bool,
) -> LineageReads {
    let path = &params.path[l];
    let n = path.len();
    let h = path.first().map_or(0, Vec::len);
    let positions = params.positions();
    let on = on_lineage(rows, positions, l);
    // The depths pooled per position and track.
    let (mut pooled_u, mut pooled_s) = (vec![0.0; positions], vec![0.0; positions]);
    for &(p, k) in &on {
        pooled_u[k] += lambda[p][0];
        pooled_s[k] += lambda[p][1];
    }
    let width = h * (BAND + 1) - 1;
    let zero = || {
        (
            0.0,
            DMatrix::zeros(n, h),
            Band::zeros(if fisher { h * n } else { 0 }, width),
        )
    };
    let (ll, grad, band) = (0..params.beta.len())
        .into_par_iter()
        .fold(zero, |(mut ll, mut gz, mut fz), g| {
            let rho = &params.rho[g];
            let log_alpha: Vec<f64> = path.iter().map(|z| dot(z, rho) + params.base[g]).collect();
            let alpha: Vec<f64> = log_alpha.iter().map(|&x| clamped_exp(x)).collect();
            // `dα/d log α`: none outside the clamp.
            let slope: Vec<f64> = log_alpha
                .iter()
                .zip(&alpha)
                .map(|(x, a)| if x.abs() < LOG_CLAMP { *a } else { 0.0 })
                .collect();
            let (a_u, a_s) = &resp[g];
            let av = DVector::from_column_slice(&alpha);
            let (u, s) = (a_u * &av, a_s * &av);
            let (mut ru, mut rs) = (vec![0.0; positions], vec![0.0; positions]);
            for &(p, k) in &on {
                ru[k] += reads.u[(p, g)];
                rs[k] += reads.s[(p, g)];
            }
            for k in 0..positions {
                ll +=
                    ru[k] * u[k].ln() - pooled_u[k] * u[k] + rs[k] * s[k].ln() - pooled_s[k] * s[k];
            }
            let eu = DVector::from_fn(positions, |k, _| ru[k] / u[k] - pooled_u[k]);
            let es = DVector::from_fn(positions, |k, _| rs[k] / s[k] - pooled_s[k]);
            // ∂ℓ/∂α_j, then ∂ℓ/∂z_jd = α_j ρ_d ∂ℓ/∂α_j.
            let d_alpha = a_u.tr_mul(&eu) + a_s.tr_mul(&es);
            gz += DMatrix::from_fn(n, h, |j, d| slope[j] * rho[d] * d_alpha[j]);
            if fisher {
                let (wu, ws): (Vec<f64>, Vec<f64>) = (0..positions)
                    .map(|i| (pooled_u[i] / u[i], pooled_s[i] / s[i]))
                    .unzip();
                // Wrt α: J̃ᵀ W J̃, `J̃` the response cut to the nodes from
                // [`BAND`] − 1 before a position's to the one after (so
                // positive semidefinite and banded); wrt z, α_i α_j ρ_d ρ_e
                // times it.
                let sub = params.sub.max(1);
                let mut fa = vec![0.0; n * (BAND + 1)];
                for k in 0..positions {
                    let node = k / sub;
                    let (lo, hi) = (node.saturating_sub(BAND - 1), (node + 1).min(n - 1));
                    for i in lo..=hi {
                        for j in lo..=i {
                            fa[i * (BAND + 1) + i - j] += a_u[(k, i)] * a_u[(k, j)] * wu[k]
                                + a_s[(k, i)] * a_s[(k, j)] * ws[k];
                        }
                    }
                }
                for i in 0..n {
                    for j in i.saturating_sub(BAND)..=i {
                        let m = slope[i] * slope[j] * fa[i * (BAND + 1) + i - j];
                        for d in 0..h {
                            for e in 0..h {
                                if i > j || d >= e {
                                    fz.add(i * h + d, j * h + e, rho[d] * rho[e] * m);
                                }
                            }
                        }
                    }
                }
            }
            (ll, gz, fz)
        })
        .reduce(zero, |(la, ga, mut fa), (lb, gb, fb)| {
            fa += &fb;
            (la + lb, ga + gb, fa)
        });
    LineageReads {
        ll,
        grad,
        fisher: fisher.then_some(band),
    }
}

/// The reads' gradient `[n × H]` and banded Fisher information
/// (node-major) in lineage `l`'s path `z` ([`lineage_reads`]).
#[cfg(test)]
pub(super) fn lineage_information(
    params: &Params,
    l: usize,
    reads: &Reads,
    rows: &[usize],
    lambda: &[Depth],
) -> (DMatrix<f64>, Band) {
    let r = lineage_reads(params, l, &responses(params, l), reads, rows, lambda, true);
    (r.grad, r.fisher.expect("asked for"))
}

/// A path block's coordinates, flattened: its whitened path `η` (`d n + k`)
/// and, for a departure from a trunk, the logit `x` of its branch time last
/// (moved with the departure: alone, the two trade off along a ridge no
/// scalar move crosses). Their own prior is `N(η; 0, I)` and the branch
/// time's uniform, `log σ(x) + log σ(−x)`.
pub struct Space<'a> {
    prior: &'a GridPrior,
    block: usize,
    h: usize,
    n: usize,
    branch: bool,
}

/// `1/(1 + e^{−x})`.
fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

impl<'a> Space<'a> {
    pub fn new(params: &Params, prior: &'a GridPrior, block: usize) -> Self {
        let path = &params.path[block];
        Self {
            prior,
            block,
            h: path[0].len(),
            n: path.len(),
            branch: block > 0 && params.branch.is_some(),
        }
    }

    fn dim(&self) -> usize {
        self.h * self.n + usize::from(self.branch)
    }

    /// The whitened path of coordinates `x`, one `[n]` per dimension.
    fn eta(&self, x: &[f64]) -> Vec<DVector<f64>> {
        unflatten(&x[..self.h * self.n], self.h, self.n)
    }

    /// The coordinates of whitened `eta` and `params`' branch time.
    fn coords(&self, params: &Params, eta: &[DVector<f64>]) -> DVector<f64> {
        let hn = self.h * self.n;
        DVector::from_fn(self.dim(), |i, _| {
            if i < hn {
                eta[i / self.n][i % self.n]
            } else {
                let t = params.branch.as_ref().expect("a trunk")[self.block - 1];
                (t / (1.0 - t)).ln()
            }
        })
    }

    /// `p` at coordinates `x`, its other blocks as in `base`.
    fn place(&self, p: &mut Params, base: &Params, x: &[f64]) {
        if self.branch {
            p.branch.as_mut().expect("a trunk")[self.block - 1] = sigmoid(x[self.dim() - 1]);
        }
        place_path(p, base, self.prior, self.block, &self.eta(x));
    }

    /// The coordinates' own log prior.
    fn own(&self, x: &[f64]) -> f64 {
        let hn = self.h * self.n;
        let eta = -0.5 * x[..hn].iter().map(|v| v * v).sum::<f64>();
        if self.branch {
            let b = x[hn];
            eta - softplus(b) - softplus(-b)
        } else {
            eta
        }
    }

    /// The ramp `r` of the block's map `M = diag(r) C` ([`path_map`]):
    /// ones but for a departure.
    fn ramp(&self, params: &Params) -> Vec<f64> {
        match &params.branch {
            Some(branch) if self.branch => {
                branch_ramp(self.n, branch[self.block - 1], self.prior.kappa())
            }
            _ => vec![1.0; self.n],
        }
    }

    /// `∂z_d/∂x` of a departure's branch-time coordinate `x`, one `[n]` per
    /// dimension, at `params` (placed at `x`): `∂r/∂t_b ⊙ C η_d` times
    /// `dt_b/dx`.
    fn branch_column(&self, params: &Params, x: &[f64]) -> Vec<DVector<f64>> {
        let tb = params.branch.as_ref().expect("a trunk")[self.block - 1];
        let kappa = self.prior.kappa();
        let n = self.n;
        self.eta(x)
            .iter()
            .map(|e| {
                let w = self.prior.colour(e);
                DVector::from_fn(n, |k, _| {
                    let u = (k as f64 / (n - 1) as f64 - tb).max(0.0) / kappa;
                    let dr = -std::f64::consts::SQRT_2 * 2.0 * u * (-u * u).exp() / kappa;
                    dr * w[k] * tb * (1.0 - tb)
                })
            })
            .collect()
    }

    /// The genes' prior `−½ Σ_g ((b_g + ρ_gᵀz̄ − log γ_g − m)/s)²`'s
    /// gradient `[H]` in `z̄`, `z̄_d` the mean of every lineage's path at
    /// every node.
    fn genes_prior_in_mean(&self, params: &Params, gene_prior: &GenePrior) -> Vec<f64> {
        let h = self.h;
        let (mean, sd) = gene_prior.log_spliced;
        let z_bar = params.path_mean();
        let mut grad = vec![0.0; h];
        for g in 0..params.beta.len() {
            let rho = &params.rho[g];
            let r = params.base[g] + dot(rho, &z_bar) - params.gamma[g].ln() - mean;
            for d in 0..h {
                grad[d] -= r * rho[d] / (sd * sd);
            }
        }
        grad
    }

    /// The genes' prior's curvature `[H × H]` in `z̄`
    /// ([`Self::genes_prior_in_mean`]): `Σ_g ρ_g ρ_gᵀ / s²`.
    fn genes_prior_curvature(&self, params: &Params, gene_prior: &GenePrior) -> DMatrix<f64> {
        let (_, sd) = gene_prior.log_spliced;
        let mut curv = DMatrix::<f64>::zeros(self.h, self.h);
        for rho in &params.rho {
            for d in 0..self.h {
                for e in 0..self.h {
                    curv[(d, e)] += rho[d] * rho[e] / (sd * sd);
                }
            }
        }
        curv
    }

    /// The share of `z̄` a node of a moved lineage holds (`moved` of them).
    fn share(&self, params: &Params, moved: usize) -> f64 {
        moved as f64 / (params.path.len() * self.n) as f64
    }

    /// The log conditional's gradient in the block's path space, `u_d = C
    /// η_d` node-major (`k H + d`), the branch-time coordinate last: from
    /// the reads' gradient `[n × H]` in `z` of every lineage the block
    /// moves (`parts`; `z = … + r ⊙ u`), with the prior `−½ uᵀK⁻¹u`
    /// (`−C⁻ᵀη`, exact), the genes' prior through `z̄` and the branch
    /// time's own.
    fn gradient(
        &self,
        params: &Params,
        x: &[f64],
        gene_prior: &GenePrior,
        parts: &[&DMatrix<f64>],
    ) -> Vec<f64> {
        let (h, n, hn) = (self.h, self.n, self.h * self.n);
        let r = self.ramp(params);
        let mut grad = vec![0.0; self.dim()];
        for grad_z in parts {
            for k in 0..n {
                for d in 0..h {
                    grad[k * h + d] += r[k] * grad_z[(k, d)];
                }
            }
        }
        let prior_grad = self.genes_prior_in_mean(params, gene_prior);
        let share = self.share(params, parts.len());
        for k in 0..n {
            for d in 0..h {
                grad[k * h + d] += share * r[k] * prior_grad[d];
            }
        }
        let c = self.prior.chol();
        for (d, eta) in self.eta(x).iter().enumerate() {
            let back = c
                .transpose()
                .solve_upper_triangular(eta)
                .expect("a Cholesky factor has a nonzero diagonal");
            for k in 0..n {
                grad[k * h + d] -= back[k];
            }
        }
        if self.branch {
            let col = self.branch_column(params, x);
            for grad_z in parts {
                for (d, c) in col.iter().enumerate() {
                    grad[hn] += (0..n).map(|k| c[k] * grad_z[(k, d)]).sum::<f64>();
                }
            }
            for (d, c) in col.iter().enumerate() {
                grad[hn] += share * c.sum() * prior_grad[d];
            }
            grad[hn] += 1.0 - 2.0 * sigmoid(x[hn]);
        }
        grad
    }

    /// The Gauss–Newton precision in the block's path space ([`Self::gradient`]'s
    /// coordinates), banded: the reads' banded Fisher information of every
    /// lineage the block moves (`parts`) through `diag(r)`, plus the
    /// prior's banded precision ([`GridPrior::markov`]); for a departure a
    /// dense border for its branch time (with the genes' prior's curvature
    /// through `z̄` there). The genes' prior's curvature across the nodes
    /// (dense, rank `H`) is left out. Factored guarded ([`Factor::new`]),
    /// the prior's precision the fallback.
    fn precision(
        &self,
        params: &Params,
        x: &[f64],
        gene_prior: &GenePrior,
        parts: &[&Band],
    ) -> Bordered {
        let (h, n) = (self.h, self.n);
        let r = self.ramp(params);
        let prior = || {
            let mut q = Band::zeros(h * n, h * (BAND + 1) - 1);
            let m = self.prior.markov();
            for i in 0..n {
                for j in i.saturating_sub(BAND)..=i {
                    for d in 0..h {
                        q.add(i * h + d, j * h + d, m.get(i, j));
                    }
                }
            }
            q
        };
        let mut a = prior();
        let width = a.width();
        for f in parts {
            for i in 0..h * n {
                for j in i.saturating_sub(width)..=i {
                    a.add(i, j, r[i / h] * r[j / h] * f.get(i, j));
                }
            }
        }
        let border = self.branch.then(|| {
            let col = self.branch_column(params, x);
            let cz: Vec<f64> = (0..h * n).map(|i| col[i % h][i / h]).collect();
            let mut c = vec![0.0; h * n];
            let mut q = 0.0;
            for f in parts {
                let fc = f.mul(&cz);
                for i in 0..h * n {
                    c[i] += r[i / h] * fc[i];
                }
                q += cz.iter().zip(&fc).map(|(a, b)| a * b).sum::<f64>();
            }
            // The genes' prior on the paths' mean: its border terms (the
            // `u`–`u` block couples every node, dense, and is left out of
            // the band; should the Schur complement then not be positive,
            // [`Bordered::new`] drops the coupling).
            let curv = self.genes_prior_curvature(params, gene_prior);
            let share = self.share(params, parts.len());
            let means: Vec<f64> = col.iter().map(|c| share * c.sum()).collect();
            for k in 0..n {
                for d in 0..h {
                    for e in 0..h {
                        c[k * h + d] += share * r[k] * means[e] * curv[(d, e)];
                    }
                }
            }
            for d in 0..h {
                for e in 0..h {
                    q += means[d] * means[e] * curv[(d, e)];
                }
            }
            let s = sigmoid(x[h * n]);
            (c, q + 2.0 * s * (1.0 - s))
        });
        Bordered::new(Factor::new(&a, prior), border)
    }

    /// Path space (node-major `u`, the branch time last) from the block's
    /// coordinates (`d n + k`): `u_d = C η_d`.
    fn to_path(&self, x: &[f64]) -> Vec<f64> {
        let (h, n, hn) = (self.h, self.n, self.h * self.n);
        let mut out = vec![0.0; self.dim()];
        for (d, eta) in self.eta(x).iter().enumerate() {
            let u = self.prior.colour(eta);
            for k in 0..n {
                out[k * h + d] = u[k];
            }
        }
        if self.branch {
            out[hn] = x[hn];
        }
        out
    }

    /// The block's coordinates from path space: `η_d = C⁻¹ u_d`.
    fn coords_of_path(&self, u: &[f64]) -> Vec<f64> {
        let (h, n, hn) = (self.h, self.n, self.h * self.n);
        let mut out = vec![0.0; self.dim()];
        for d in 0..h {
            let ud = DVector::from_fn(n, |k, _| u[k * h + d]);
            let eta = self
                .prior
                .chol()
                .solve_lower_triangular(&ud)
                .expect("a Cholesky factor has a nonzero diagonal");
            for k in 0..n {
                out[d * n + k] = eta[k];
            }
        }
        if self.branch {
            out[hn] = u[hn];
        }
        out
    }
}

/// The normal approximation of a path's conditional: its mode in its
/// block's coordinates ([`Space`]) and its banded precision in path space
/// ([`Space::precision`]), over [`WIDEN`]².
#[derive(Clone)]
pub struct PathProposal {
    centre: Vec<f64>,
    precision: Bordered,
}

impl PathProposal {
    /// The mode's whitened paths, one `[n]` per dimension.
    pub fn mode(&self, h: usize) -> Vec<DVector<f64>> {
        let n = self.centre.len() / h.max(1);
        unflatten(&self.centre, h, n)
    }

    /// A departure's branch time at the mode (`None` for a block without).
    pub fn branch_time(&self, h: usize, n: usize) -> Option<f64> {
        (self.centre.len() > h * n).then(|| sigmoid(self.centre[h * n]))
    }

    /// `log N(θ; mode, P⁻¹)` up to a constant, `θ` in `space`'s coordinates.
    fn log_density(&self, space: &Space, theta: &[f64]) -> f64 {
        let d: Vec<f64> = theta.iter().zip(&self.centre).map(|(a, b)| a - b).collect();
        -0.5 * self.precision.quad(&space.to_path(&d)) / (WIDEN * WIDEN)
    }

    /// A draw of `θ − mode`.
    fn draw(&self, space: &Space, rng: &mut impl Rng) -> DVector<f64> {
        let z: Vec<f64> = (0..self.precision.size())
            .map(|_| StandardNormal.sample(rng))
            .collect();
        let u: Vec<f64> = self.precision.draw(&z).iter().map(|v| v * WIDEN).collect();
        DVector::from_vec(space.coords_of_path(&u))
    }
}

/// `[H][n]` from flattened `d n + k`.
fn unflatten(x: &[f64], h: usize, n: usize) -> Vec<DVector<f64>> {
    (0..h)
        .map(|d| DVector::from_fn(n, |k, _| x[d * n + k]))
        .collect()
}

/// The Laplace approximation of path block `l`'s conditional (rows,
/// depths, genes held), banded in path space: at `anchor` (whitened; in
/// the sampler the last fit's mode; a departure's branch time from
/// `params`) the precision once ([`Space::precision`]), then a few
/// gradient steps on it ([`frozen_steps`]), the gradient analytic
/// ([`lineage_reads`]). Fitted during burn-in only, as [`gene_mode`].
#[allow(clippy::too_many_arguments)]
pub fn path_mode(
    params: &Params,
    l: usize,
    anchor: &[DVector<f64>],
    prior: &GridPrior,
    reads: &Reads,
    rows: &[usize],
    lambda: &[Depth],
    gene_prior: &GenePrior,
) -> PathProposal {
    let space = Space::new(params, prior, l);
    let lineages = moved_by(params, l);
    let resp: Vec<_> = lineages.iter().map(|&m| responses(params, m)).collect();
    let base = params.clone();
    let mut probe = params.clone();
    // The objective and its gradient in path space at `x` (and the
    // precision, when asked).
    let mut at = |x: &[f64], fisher: bool| {
        space.place(&mut probe, &base, x);
        let parts: Vec<LineageReads> = lineages
            .iter()
            .zip(&resp)
            .map(|(&m, r)| lineage_reads(&probe, m, r, reads, rows, lambda, fisher))
            .collect();
        let grads: Vec<&DMatrix<f64>> = parts.iter().map(|r| &r.grad).collect();
        let f = parts.iter().map(|r| r.ll).sum::<f64>()
            + genes_log_prior(&probe, gene_prior)
            + space.own(x);
        let g = space.gradient(&probe, x, gene_prior, &grads);
        let prec = fisher.then(|| {
            let bands: Vec<&Band> = parts.iter().filter_map(|r| r.fisher.as_ref()).collect();
            space.precision(&probe, x, gene_prior, &bands)
        });
        (f, g, prec)
    };
    let start: Vec<f64> = space.coords(params, anchor).iter().copied().collect();
    let (f0, g0, precision) = at(&start, true);
    let precision = precision.expect("asked for");
    let centre = frozen_steps(
        &start,
        (f0, g0),
        PATH_STEPS,
        &|g: &[f64]| space.coords_of_path(&precision.solve(g)),
        &mut |x: &[f64]| {
            let (f, g, _) = at(x, false);
            (f, g)
        },
    );
    PathProposal { centre, precision }
}

/// One generalized elliptical slice move of path block `l` (its whitened
/// path, and a departure's branch time with it, [`Space`]) on `prop`, a
/// Student-t (`ν` = [`T_DF`]) centred on the Laplace mode and shaped like
/// it, as in [`gene_step`], for the coordinates' own prior times
/// [`path_log_density`]. Genes, latents and depths held; `params` is left
/// at the new path (and branch time). `cur_ll` ([`path_log_density`] at
/// `eta`) may be `−∞`; the new `η` and that density there.
#[allow(clippy::too_many_arguments)]
pub fn path_gess_step(
    params: &mut Params,
    l: usize,
    eta: &[DVector<f64>],
    cur_ll: f64,
    prior: &GridPrior,
    reads: &Reads,
    rows: &[usize],
    lambda: &[Depth],
    gene_prior: &GenePrior,
    prop: &PathProposal,
    rng: &mut impl Rng,
) -> (Vec<DVector<f64>>, f64) {
    let space = Space::new(params, prior, l);
    t_ess(
        &space,
        params,
        eta,
        cur_ll,
        prop,
        &|p: &Params| path_log_density(p, l, reads, rows, lambda, gene_prior),
        rng,
    )
}

/// One generalized elliptical slice move in `space` on `prop` for the
/// coordinates' own prior times `density`; `params` is left at the new
/// coordinates. `cur_ll` (`density` at `eta` and `params`' branch time)
/// may be `−∞`; the new `η` and `density` there.
fn t_ess(
    space: &Space,
    params: &mut Params,
    eta: &[DVector<f64>],
    cur_ll: f64,
    prop: &PathProposal,
    density: &dyn Fn(&Params) -> f64,
    rng: &mut impl Rng,
) -> (Vec<DVector<f64>>, f64) {
    let dim = space.dim();
    let theta: Vec<f64> = space.coords(params, eta).iter().copied().collect();
    let base = params.clone();
    space.place(params, &base, &theta);
    let cur_ll = if cur_ll.is_finite() {
        cur_ll
    } else {
        density(params)
    };
    // log t_ν up to a constant.
    let log_t = |t: &[f64]| {
        -0.5 * (T_DF + dim as f64) * (1.0 - 2.0 * prop.log_density(space, t) / T_DF).ln()
    };
    let delta2 = -2.0 * prop.log_density(space, &theta);
    let scale = 0.5 * (T_DF + delta2)
        / Gamma::new(0.5 * (T_DF + dim as f64), 1.0)
            .expect("a positive shape")
            .sample(rng);
    let offset = cur_ll + space.own(&theta) - log_t(&theta);
    let at = |y: &DVector<f32>| -> Vec<f64> {
        y.iter()
            .zip(&prop.centre)
            .map(|(&v, m)| f64::from(v) + m)
            .collect()
    };
    let x = DVector::from_iterator(
        dim,
        theta.iter().zip(&prop.centre).map(|(t, m)| (t - m) as f32),
    );
    let nu = (prop.draw(space, rng) * scale.sqrt()).map(|v| v as f32);
    let probe = params.clone();
    // Relative to the current value, so f32 keeps the slice's resolution.
    let lnpdf = |y: &DVector<f32>| {
        let t = at(y);
        let mut p = probe.clone();
        space.place(&mut p, &probe, &t);
        (density(&p) + space.own(&t) - log_t(&t) - offset) as f32
    };
    let (next, _) = elliptical_slice_step(&x, &nu, &lnpdf, 0.0, rng);
    let t = at(&next);
    space.place(params, &probe, &t);
    let ll = density(params);
    (space.eta(&t), ll)
}

/// The jitter sd's inverse-gamma prior, `σ² ~ IG(a, b)`: weak, the draws
/// follow the jitters' own spread for tens of pseudobulks or more.
const SHARE_SD_PRIOR: (f64, f64) = (1.0, 1e-4);

/// Spread of the jitter's independence proposal over its Laplace sd.
const SHARE_PROPOSAL_WIDTH: f64 = 1.2;

/// One independence Metropolis–Hastings move of a pseudobulk's jitter `ω`
/// on its conditional `∝ e^{Δ(ω)} N(ω; 0, σ²)` (its row held: `N_u` of `N`
/// reads unspliced, `Z_u`, `Z_s` the levels' totals there), the proposal
/// the conditional's Laplace approximation widened by
/// [`SHARE_PROPOSAL_WIDTH`]. The conditional is near Gaussian at any depth
/// (a logistic likelihood in one variable), so nearly every move is taken.
pub fn share_step(
    n_u: f64,
    n: f64,
    z_u: f64,
    z_s: f64,
    omega: f64,
    sd: f64,
    rng: &mut impl Rng,
) -> f64 {
    let (a, v) = ((z_u / z_s).ln(), sd * sd);
    let log_f = |w: f64| n_u * w - n * softplus(a + w) - w * w / (2.0 * v);
    let mode = share_mode(n_u, n, a, v);
    let q = 1.0 / (1.0 + (-(a + mode)).exp());
    let width = SHARE_PROPOSAL_WIDTH / (n * q * (1.0 - q) + 1.0 / v).sqrt();
    let log_q = |w: f64| -0.5 * ((w - mode) / width).powi(2);
    let e: f64 = StandardNormal.sample(rng);
    let next = mode + width * e;
    let log_ratio = log_f(next) - log_f(omega) - log_q(next) + log_q(omega);
    if rng.random::<f64>().ln() < log_ratio {
        next
    } else {
        omega
    }
}

/// The jitter sd drawn from its conditional given every pseudobulk's
/// jitter: `σ² ~ IG(a + P/2, b + Σ ω²/2)` ([`SHARE_SD_PRIOR`]).
pub fn draw_share_sd(omega: &[f64], seed: u64, sweep: u64) -> f64 {
    let (a, b) = SHARE_SD_PRIOR;
    let shape = a + omega.len() as f64 / 2.0;
    let rate = b + omega.iter().map(|w| w * w).sum::<f64>() / 2.0;
    let mut rng = stream(seed, u64::MAX - 1, sweep);
    let precision = Gamma::new(shape, 1.0 / rate)
        .expect("a positive shape and rate")
        .sample(&mut rng);
    precision.recip().sqrt()
}
