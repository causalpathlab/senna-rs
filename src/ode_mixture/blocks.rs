//! The sampler's blocks, each a draw from its conditional.
//!
//! Every random stream is keyed by `(seed, item, sweep)`, so a draw does
//! not depend on how the items were split across threads.

use super::likelihood::{gene_log_likelihood, Reads};
use super::model::Params;
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
            let logit: Vec<f64> = (0..tab.ncols())
                .map(|r| tab[(p, r)] + log_w[r / points])
                .collect();
            let top = logit.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let w: Vec<f64> = logit.iter().map(|x| (x - top).exp()).collect();
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
    let gamma = theta[0].exp();
    let rho = &theta[2..2 + h];
    let base = theta[2 + h] + theta[0] - dot(rho, z_bar);
    (gamma / theta[1].exp(), gamma, rho, base)
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// The normal prior on a gene's [`to_theta`] coordinates, `(mean, sd)` each.
#[derive(Clone, Debug)]
pub struct GenePrior {
    pub log_gamma: (f64, f64),
    pub log_ratio: (f64, f64),
    pub rho_sd: f64,
    pub log_spliced: (f64, f64),
}

impl Default for GenePrior {
    fn default() -> Self {
        Self {
            log_gamma: (2f64.ln(), 1.5),
            log_ratio: (0.5f64.ln(), 1.0),
            rho_sd: 1.0,
            log_spliced: (0.0, 1.5),
        }
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

/// Gene `g`'s log-likelihood at `θ`, every other parameter from `params`.
fn gene_ll(
    params: &Params,
    theta: &[f64],
    reads: &Reads,
    g: usize,
    rows: &[usize],
    lambda: &[f64],
) -> f64 {
    let (beta, gamma, rho, base) = from_theta(theta, &params.path_mean());
    let lv = params.levels_of(beta, gamma, rho, base);
    gene_log_likelihood(reads, g, rows, lambda, (&lv.0, &lv.1))
}

impl GenePrior {
    /// `log p₀(θ)` up to a constant.
    fn log_density(&self, theta: &[f64]) -> f64 {
        let h = theta.len() - 3;
        let (m, s) = (self.mean(h), self.sd(h));
        theta
            .iter()
            .zip(m.iter().zip(&s))
            .map(|(t, (m, s))| -0.5 * ((t - m) / s).powi(2))
            .sum()
    }
}

/// Where a gene's conditional is, as a normal: its mode and the Cholesky
/// factor `L` of the precision `P = L Lᵀ` (the negative Hessian there,
/// over [`WIDEN`]²).
pub struct GeneProposal {
    pub mode: Vec<f64>,
    precision_chol: DMatrix<f64>,
}

/// The proposal's sd over the Laplace approximation's: heavier than the
/// conditional, so its tails are covered.
const WIDEN: f64 = 1.5;

/// The Laplace approximation of gene `g`'s conditional (likelihood times
/// prior), by Newton's method from `anchor`, numerical derivatives (a gene
/// has `H + 3` coordinates). `anchor` must not be the chain's current `θ`
/// (the previous sweep's mode is), so the move built on it stays exact.
#[allow(clippy::too_many_arguments)]
pub fn gene_laplace(
    params: &Params,
    g: usize,
    reads: &Reads,
    rows: &[usize],
    lambda: &[f64],
    prior: &GenePrior,
    anchor: &[f64],
) -> GeneProposal {
    let f = |t: &[f64]| gene_ll(params, t, reads, g, rows, lambda) + prior.log_density(t);
    let n = anchor.len();
    let eps = 1e-4;
    let hessian = |t: &[f64]| -> (DVector<f64>, DMatrix<f64>) {
        let f0 = f(t);
        let at = |d: &[(usize, f64)]| {
            let mut x = t.to_vec();
            for &(i, v) in d {
                x[i] += v;
            }
            f(&x)
        };
        let grad = DVector::from_fn(n, |i, _| (at(&[(i, eps)]) - at(&[(i, -eps)])) / (2.0 * eps));
        let mut hess = DMatrix::zeros(n, n);
        for i in 0..n {
            hess[(i, i)] = (at(&[(i, eps)]) - 2.0 * f0 + at(&[(i, -eps)])) / (eps * eps);
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
    // Damped Newton: the negative Hessian plus a ridge until positive
    // definite, the step halved until the objective rises.
    let mut t = anchor.to_vec();
    let mut ft = f(&t);
    for _ in 0..50 {
        let (grad, hess) = hessian(&t);
        let mut ridge = 0.0;
        let step = loop {
            let p = -&hess + DMatrix::identity(n, n) * ridge;
            if let Some(c) = p.cholesky() {
                break c.solve(&grad);
            }
            ridge = if ridge == 0.0 { 1e-3 } else { ridge * 10.0 };
        };
        let mut scale = 1.0;
        let moved = loop {
            let x: Vec<f64> = t
                .iter()
                .zip(step.iter())
                .map(|(a, b)| a + scale * b)
                .collect();
            let fx = f(&x);
            if fx >= ft {
                break Some((x, fx));
            }
            scale *= 0.5;
            if scale < 1e-8 {
                break None;
            }
        };
        let Some((x, fx)) = moved else { break };
        let size = scale * step.norm();
        (t, ft) = (x, fx);
        if size < 1e-7 {
            break;
        }
    }
    let (_, hess) = hessian(&t);
    let mut p = -hess / (WIDEN * WIDEN);
    let mut ridge = 1e-9;
    let precision_chol = loop {
        if let Some(c) = p.clone().cholesky() {
            break c.l();
        }
        p += DMatrix::identity(n, n) * ridge;
        ridge *= 10.0;
    };
    GeneProposal {
        mode: t,
        precision_chol,
    }
}

impl GeneProposal {
    /// `log N(θ; mode, P⁻¹)` up to a constant.
    fn log_density(&self, theta: &[f64]) -> f64 {
        let d = DVector::from_iterator(
            theta.len(),
            theta.iter().zip(&self.mode).map(|(a, b)| a - b),
        );
        -0.5 * (self.precision_chol.transpose() * d).norm_squared()
    }

    /// A draw of `θ − mode`: `L⁻ᵀ z`, `z ~ N(0, I)`.
    fn draw(&self, rng: &mut impl Rng) -> DVector<f64> {
        let z = DVector::from_fn(self.mode.len(), |_, _| StandardNormal.sample(rng));
        self.precision_chol
            .transpose()
            .solve_upper_triangular(&z)
            .expect("a Cholesky factor has a nonzero diagonal")
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
    g: usize,
    theta: &[f64],
    cur_ll: f64,
    reads: &Reads,
    rows: &[usize],
    lambda: &[f64],
    prior: &GenePrior,
    prop: &GeneProposal,
    rng: &mut impl Rng,
) -> (Vec<f64>, f64) {
    let n = theta.len();
    let d = n as f64;
    let ll = |t: &[f64]| gene_ll(params, t, reads, g, rows, lambda);
    let mut cur = theta.to_vec();
    let mut cur_ll = if cur_ll.is_finite() {
        cur_ll
    } else {
        ll(theta)
    };

    // A scalar slice move per coordinate on the prior's scale first: where
    // the prior, not the reads, shapes a direction, this is what mixes.
    let (mean, sd) = (prior.mean(n - 3), prior.sd(n - 3));
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
    let tilt = |t: &[f64]| prior.log_density(t) - log_t(t);

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
            .zip(&prop.mode)
            .map(|(&v, m)| f64::from(v) + m)
            .collect()
    };
    let x = DVector::from_iterator(n, theta.iter().zip(&prop.mode).map(|(t, m)| (t - m) as f32));
    let nu = (prop.draw(rng) * scale.sqrt()).map(|v| v as f32);
    // Relative to the current value, so f32 keeps the slice's resolution.
    let lnpdf = |y: &DVector<f32>| {
        let t = at(y);
        (ll(&t) + tilt(&t) - base) as f32
    };
    let (next, _) = elliptical_slice_step(&x, &nu, &lnpdf, 0.0, rng);
    let t = at(&next);
    let new_ll = ll(&t);
    (t, new_ll)
}
