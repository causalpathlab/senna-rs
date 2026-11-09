//! A gene's conditional given the paths, the latents and the depths:
//! the gene's coordinates ([`to_theta`]), the prior on them
//! ([`GenePrior`], the rates bounded) and the Laplace approximation of the
//! conditional at its mode by bounded Newton ([`gene_mode`]), from which
//! every gene's rates and their sds are read; and every pseudobulk's
//! posterior over the rows ([`log_posterior_row`]).

use super::band::{dot, Band, Bordered, Factor};
use super::likelihood::{softplus, GeneRows};
use super::model::{clamped_exp, Params};
use nalgebra::{DMatrix, DVector};

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

/// A gene's parameters as the Laplace approximation takes them, each with a meaning of
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
    /// carried by the gene's coordinates ([`GenePrior::link`]); `None`:
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

/// A rate's half-life is held above one fine step and below this many time
/// units ([`GenePrior::bounded`]): faster, the reads follow `α` in
/// quasi-steady state; slower, they accumulate `α` with no decay seen
/// (over four units a level decays by under a sixth). Either way the
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
    /// the climb, so they are left out: the bounds stay
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

    /// The gene's coordinates of `θ` ([`to_theta`]): with bounds, `log γ`
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

    /// `θ` from the gene's coordinates ([`Self::link`]).
    pub fn unlink(&self, v: &[f64]) -> Vec<f64> {
        let mut theta = v.to_vec();
        if let Some((lo, hi)) = self.log_rate {
            let rate = |u: f64| lo + (hi - lo) * sigmoid(u);
            let (log_gamma, log_beta) = (rate(v[0]), rate(v[1]));
            (theta[0], theta[1]) = (log_gamma, log_gamma - log_beta);
        }
        theta
    }

    /// The prior's log density in the gene's coordinates: at `θ` with
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

    /// A normal in the gene's coordinates near the prior, `(mean, sd)`
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

/// Where a gene's conditional is, as a normal in the gene's
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

/// A failed step is halved this many times (down to about a thousandth of
/// its length) before the search stops: a well-read gene or path's
/// posterior can be far narrower than the capped step.
const HALVINGS: usize = 10;

/// A step's largest move in any coordinate (about the prior's sd): the
/// steps cannot run away where the frozen precision is far off.
const STEP_CAP: f64 = 1.0;

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
/// the search stops (and where a value is not finite).
pub fn gene_mode(
    params: &Params,
    by_row: &GeneRows,
    z_bar: &[f64],
    prior: &GenePrior,
    anchor: &[f64],
) -> GeneProposal {
    // In the gene's coordinates.
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
    /// The Laplace approximation's sds of `log γ` and `log γ/β` at the
    /// mode: the conditional's covariance is `P⁻¹/`[`WIDEN`]`²` in the
    /// gene's coordinates, carried through the link's derivative at the
    /// centre (`log γ/β = log γ − log β`, each rate its own `u`).
    pub fn rate_sds(&self, prior: &GenePrior) -> (f64, f64) {
        let n = self.centre.len();
        let column = |i: usize| {
            let mut e = vec![0.0; n];
            e[i] = 1.0;
            self.precision.solve(&e)
        };
        let (c0, c1) = (column(0), column(1));
        let (v00, v01, v11) = (
            c0[0] / (WIDEN * WIDEN),
            c0[1] / (WIDEN * WIDEN),
            c1[1] / (WIDEN * WIDEN),
        );
        let (j0, j1) = match prior.log_rate {
            Some((lo, hi)) => {
                let slope = |u: f64| {
                    let s = sigmoid(u);
                    (hi - lo) * s * (1.0 - s)
                };
                (slope(self.centre[0]), slope(self.centre[1]))
            }
            None => (1.0, 1.0),
        };
        let var_gamma = j0 * j0 * v00;
        let var_ratio = j0 * j0 * v00 - 2.0 * j0 * j1 * v01 + j1 * j1 * v11;
        (var_gamma.max(0.0).sqrt(), var_ratio.max(0.0).sqrt())
    }
}

/// `1/(1 + e^{−x})`.
fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}
