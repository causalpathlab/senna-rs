//! The sampler's blocks, each a draw from its conditional.
//!
//! Every random stream is keyed by `(seed, item, sweep)`, so a draw does
//! not depend on how the items were split across threads.

use super::kernel::GridPrior;
use super::kinetics::trajectory;
use super::likelihood::{gene_log_likelihood, table, Reads};
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
#[derive(Clone)]
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

/// The log-likelihood (no `log x!`) of the pseudobulks on lineage `l`, at
/// their rows and depths, every gene's levels on `l` from `params`; in
/// parallel over genes.
pub fn path_ll(params: &Params, l: usize, reads: &Reads, rows: &[usize], lambda: &[f64]) -> f64 {
    let n = params.path[l].len();
    let on: Vec<(usize, usize)> = rows
        .iter()
        .enumerate()
        .filter(|&(_, &r)| r / n == l)
        .map(|(p, &r)| (p, r - l * n))
        .collect();
    let h = params.t_len[l] / (n - 1) as f64;
    (0..params.beta.len())
        .into_par_iter()
        .map(|g| {
            let alpha: Vec<f64> = params.path[l]
                .iter()
                .map(|z| (dot(z, &params.rho[g]) + params.base[g]).exp())
                .collect();
            let (u, s) = trajectory(&alpha, params.beta[g], params.gamma[g], h);
            on.iter()
                .map(|&(p, k)| {
                    let (mu, ms) = (lambda[p] * u[k], lambda[p] * s[k]);
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

/// Lineage `l`'s path's conditional up to a constant, `η`'s own prior
/// aside: [`path_ll`] plus [`genes_log_prior`].
pub fn path_log_density(
    params: &Params,
    l: usize,
    reads: &Reads,
    rows: &[usize],
    lambda: &[f64],
    gene_prior: &GenePrior,
) -> f64 {
    path_ll(params, l, reads, rows, lambda) + genes_log_prior(params, gene_prior)
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
    lambda: &[f64],
    gene_prior: &GenePrior,
    rng: &mut impl Rng,
) -> (Vec<DVector<f64>>, f64) {
    let (h, n) = (eta.len(), eta.first().map_or(0, |e| e.len()));
    params.path[l] = coloured(prior, eta);
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
        p.path[l] = coloured(prior, &split(y));
        (path_log_density(&p, l, reads, rows, lambda, gene_prior) - cur_ll) as f32
    };
    let (next, rel) = elliptical_slice_step(&x, &nu, &lnpdf, 0.0, rng);
    let eta = split(&next);
    params.path[l] = coloured(prior, &eta);
    (eta, cur_ll + f64::from(rel))
}

/// `(A_u, A_s)`, `[n × n]` each: the trajectory is linear in the
/// transcription for fixed rates, `ū = A_u α`, `s̄ = A_s α`
/// ([`trajectory`] from the steady state of `α_0`); column `j` the
/// response to `α = e_j`.
fn response(beta: f64, gamma: f64, h: f64, n: usize) -> (DMatrix<f64>, DMatrix<f64>) {
    let (mut a_u, mut a_s) = (DMatrix::zeros(n, n), DMatrix::zeros(n, n));
    for j in 0..n {
        let mut e = vec![0.0; n];
        e[j] = 1.0;
        let (u, s) = trajectory(&e, beta, gamma, h);
        for k in 0..n {
            a_u[(k, j)] = u[k];
            a_s[(k, j)] = s[k];
        }
    }
    (a_u, a_s)
}

/// Lineage `l`'s path's log conditional in its whitened coordinates
/// (flattened `d n + k`): its gradient and the Gauss–Newton precision
/// (`I` from `η`'s prior, the reads' Fisher information, the genes'
/// prior's exact curvature through `z̄`). Rows and depths held.
///
/// The reads on lineage `l` pooled per grid point, `U_gk = Σ_{p at k} u_pg`,
/// `Λ_k = Σ_{p at k} λ_p`: `ℓ = Σ_{g,k} U_gk log ū_gk − Λ_k ū_gk (+ s̄)`,
/// `∂ℓ/∂ū_gk = U_gk/ū_gk − Λ_k`, Fisher `Λ_k/ū_gk`; then through `ū = A_u α`,
/// `α_gj = exp(ρ_gᵀ z_j + b_g)` and `z_d = C η_d`.
fn path_newton(
    params: &Params,
    l: usize,
    prior: &GridPrior,
    reads: &Reads,
    rows: &[usize],
    lambda: &[f64],
    gene_prior: &GenePrior,
) -> (DVector<f64>, DMatrix<f64>) {
    let path = &params.path[l];
    let n = path.len();
    let h = params.rho.first().map_or(0, Vec::len);
    let genes = params.beta.len();
    let step = params.t_len[l] / (n - 1) as f64;
    let on: Vec<(usize, usize)> = rows
        .iter()
        .enumerate()
        .filter(|&(_, &r)| r / n == l)
        .map(|(p, &r)| (p, r - l * n))
        .collect();
    let mut pooled = vec![0.0; n];
    for &(p, k) in &on {
        pooled[k] += lambda[p];
    }

    // Summed over genes: the gradient wrt z `[n × H]`, the Fisher `[Hn × Hn]`.
    let (grad_z, fisher_z) = (0..genes)
        .into_par_iter()
        .map(|g| {
            let rho = &params.rho[g];
            let alpha: Vec<f64> = path
                .iter()
                .map(|z| (dot(z, rho) + params.base[g]).exp())
                .collect();
            let (a_u, a_s) = response(params.beta[g], params.gamma[g], step, n);
            let av = DVector::from_column_slice(&alpha);
            let (u, s) = (&a_u * &av, &a_s * &av);
            let (mut ru, mut rs) = (vec![0.0; n], vec![0.0; n]);
            for &(p, k) in &on {
                ru[k] += reads.u[(p, g)];
                rs[k] += reads.s[(p, g)];
            }
            let eu = DVector::from_fn(n, |k, _| ru[k] / u[k] - pooled[k]);
            let es = DVector::from_fn(n, |k, _| rs[k] / s[k] - pooled[k]);
            // ∂ℓ/∂α_j, then ∂ℓ/∂z_jd = α_j ρ_d ∂ℓ/∂α_j.
            let d_alpha = a_u.transpose() * eu + a_s.transpose() * es;
            let gz = DMatrix::from_fn(n, h, |j, d| alpha[j] * rho[d] * d_alpha[j]);
            // Wrt α: A_uᵀ W_u A_u + A_sᵀ W_s A_s; wrt z, α_i α_j ρ_d ρ_e times it.
            let wu = DMatrix::from_fn(n, n, |i, j| a_u[(i, j)] * pooled[i] / u[i]);
            let ws = DMatrix::from_fn(n, n, |i, j| a_s[(i, j)] * pooled[i] / s[i]);
            let fa = a_u.transpose() * wu + a_s.transpose() * ws;
            let fz = DMatrix::from_fn(h * n, h * n, |a, b| {
                let (d, i, e, j) = (a / n, a % n, b / n, b % n);
                rho[d] * rho[e] * alpha[i] * alpha[j] * fa[(i, j)]
            });
            (gz, fz)
        })
        .reduce(
            || (DMatrix::zeros(n, h), DMatrix::zeros(h * n, h * n)),
            |a, b| (a.0 + b.0, a.1 + b.1),
        );
    whitened(params, l, prior, gene_prior, &grad_z, &fisher_z)
}

/// From the reads' gradient `[n × H]` and information `[Hn × Hn]` in `z`
/// (flattened `d n + k`) of lineage `l`'s path: the gradient and precision
/// in its whitened coordinates, with `η`'s `N(0, I)` and the genes' prior's
/// exact curvature through `z̄`:
/// `−½ Σ_g ((b_g + ρ_gᵀz̄ − log γ_g − m)/s)²`, `∂z̄_d/∂η_di = (Cᵀ1)_i / (L n)`.
fn whitened(
    params: &Params,
    l: usize,
    prior: &GridPrior,
    gene_prior: &GenePrior,
    grad_z: &DMatrix<f64>,
    fisher_z: &DMatrix<f64>,
) -> (DVector<f64>, DMatrix<f64>) {
    let path = &params.path[l];
    let n = path.len();
    let h = params.rho.first().map_or(0, Vec::len);
    let c = prior.chol();
    let (m, sd) = gene_prior.log_spliced;
    let z_bar = params.path_mean();
    let mut prior_grad = vec![0.0; h];
    let mut prior_curv = DMatrix::<f64>::zeros(h, h);
    for g in 0..params.beta.len() {
        let rho = &params.rho[g];
        let r = params.base[g] + dot(rho, &z_bar) - params.gamma[g].ln() - m;
        for d in 0..h {
            prior_grad[d] -= r * rho[d] / (sd * sd);
            for e in 0..h {
                prior_curv[(d, e)] += rho[d] * rho[e] / (sd * sd);
            }
        }
    }
    let points = (params.path.len() * n) as f64;
    let col = DVector::from_fn(n, |i, _| (i..n).map(|k| c[(k, i)]).sum::<f64>() / points);
    let mut grad = DVector::zeros(h * n);
    let mut prec = DMatrix::identity(h * n, h * n);
    for d in 0..h {
        let eta = prior.whiten(&DVector::from_fn(n, |k, _| path[k][d]));
        let gd = c.transpose() * grad_z.column(d) + &col * prior_grad[d] - eta;
        grad.rows_mut(d * n, n).copy_from(&gd);
        for e in 0..h {
            let f = fisher_z.view((d * n, e * n), (n, n));
            let block = c.transpose() * f * c + &col * col.transpose() * prior_curv[(d, e)];
            let mut v = prec.view_mut((d * n, e * n), (n, n));
            v += block;
        }
    }
    (grad, prec)
}

/// The normal approximation of a path's conditional, as [`GeneProposal`]
/// (its mode and the Cholesky factor of its precision over [`WIDEN`]²), in
/// whitened coordinates flattened `d n + k`.
#[derive(Clone)]
pub struct PathProposal {
    normal: GeneProposal,
}

impl PathProposal {
    /// The mode as whitened paths, one `[n]` per dimension.
    pub fn mode(&self, h: usize) -> Vec<DVector<f64>> {
        let n = self.normal.mode.len() / h.max(1);
        (0..h)
            .map(|d| DVector::from_fn(n, |k, _| self.normal.mode[d * n + k]))
            .collect()
    }
}

/// `[H][n]` from flattened `d n + k`.
fn unflatten(x: &[f64], h: usize, n: usize) -> Vec<DVector<f64>> {
    (0..h)
        .map(|d| DVector::from_fn(n, |k, _| x[d * n + k]))
        .collect()
}

/// The Laplace approximation of lineage `l`'s path's conditional (rows,
/// depths, genes held) by damped Gauss–Newton from `anchor` (whitened; not
/// the chain's current path but, in the sampler, the previous sweep's
/// mode), the Fisher information analytic ([`path_newton`]).
#[allow(clippy::too_many_arguments)]
pub fn path_laplace(
    params: &Params,
    l: usize,
    anchor: &[DVector<f64>],
    prior: &GridPrior,
    reads: &Reads,
    rows: &[usize],
    lambda: &[f64],
    gene_prior: &GenePrior,
) -> PathProposal {
    scoring(
        params,
        l,
        anchor,
        prior,
        &|p: &Params| path_log_density(p, l, reads, rows, lambda, gene_prior),
        &|p: &Params| path_newton(p, l, prior, reads, rows, lambda, gene_prior),
        20,
    )
}

/// A gradient and a precision.
type Newton = (DVector<f64>, DMatrix<f64>);

/// Damped Fisher scoring of `density` (`η`'s prior aside) in lineage `l`'s
/// whitened path from `anchor`, `newton` its gradient and precision (`η`'s
/// prior in); stopped when the predicted rise `gᵀP⁻¹g` is under a
/// hundredth of a nat. The normal at the end, as a [`PathProposal`].
fn scoring(
    params: &Params,
    l: usize,
    anchor: &[DVector<f64>],
    prior: &GridPrior,
    density: &dyn Fn(&Params) -> f64,
    newton: &dyn Fn(&Params) -> Newton,
    max_iter: usize,
) -> PathProposal {
    let h = anchor.len();
    let n = anchor.first().map_or(0, |e| e.len());
    let mut probe = params.clone();
    let objective = |p: &mut Params, x: &DVector<f64>| {
        p.path[l] = coloured(prior, &unflatten(x.as_slice(), h, n));
        density(p) - 0.5 * x.norm_squared()
    };
    let mut x = DVector::from_fn(h * n, |i, _| anchor[i / n][i % n]);
    let mut fx = objective(&mut probe, &x);
    let mut prec;
    let mut iter = 0;
    loop {
        let grad;
        (grad, prec) = newton(&probe);
        let step = prec
            .clone()
            .cholesky()
            .expect("I plus a Fisher information is positive definite")
            .solve(&grad);
        iter += 1;
        if grad.dot(&step) < 1e-2 || iter > max_iter {
            break;
        }
        let mut scale = 1.0;
        let moved = loop {
            let y = &x + &step * scale;
            let fy = objective(&mut probe, &y);
            if fy >= fx {
                break Some((y, fy));
            }
            scale *= 0.5;
            if scale < 1e-4 {
                break None;
            }
        };
        let Some((y, fy)) = moved else {
            probe.path[l] = coloured(prior, &unflatten(x.as_slice(), h, n));
            break;
        };
        (x, fx) = (y, fy);
    }
    let precision_chol = (prec / (WIDEN * WIDEN))
        .cholesky()
        .expect("I plus a Fisher information is positive definite")
        .l();
    PathProposal {
        normal: GeneProposal {
            mode: x.iter().copied().collect(),
            precision_chol,
        },
    }
}

/// One generalized elliptical slice move of lineage `l`'s whitened path on
/// `prop`, a Student-t (`ν` = [`T_DF`]) centred on the Laplace mode and
/// shaped like it, as in [`gene_step`], for `N(η; 0, I)` times
/// [`path_log_density`]. Genes, latents and depths held; `params.path[l]`
/// is left at the new path. `cur_ll` ([`path_log_density`] at `eta`) may be
/// `−∞`; the new `η` and that density there.
#[allow(clippy::too_many_arguments)]
pub fn path_gess_step(
    params: &mut Params,
    l: usize,
    eta: &[DVector<f64>],
    cur_ll: f64,
    prior: &GridPrior,
    reads: &Reads,
    rows: &[usize],
    lambda: &[f64],
    gene_prior: &GenePrior,
    prop: &PathProposal,
    rng: &mut impl Rng,
) -> (Vec<DVector<f64>>, f64) {
    t_ess(
        params,
        l,
        eta,
        cur_ll,
        prior,
        prop,
        &|p: &Params| path_log_density(p, l, reads, rows, lambda, gene_prior),
        rng,
    )
}

/// One generalized elliptical slice move of lineage `l`'s whitened path on
/// `prop` for `N(η; 0, I)` times `density`; `params.path[l]` is left at the
/// new path. `cur_ll` (`density` at `eta`) may be `−∞`; the new `η` and
/// `density` there.
#[allow(clippy::too_many_arguments)]
fn t_ess(
    params: &mut Params,
    l: usize,
    eta: &[DVector<f64>],
    cur_ll: f64,
    prior: &GridPrior,
    prop: &PathProposal,
    density: &dyn Fn(&Params) -> f64,
    rng: &mut impl Rng,
) -> (Vec<DVector<f64>>, f64) {
    let (h, n) = (eta.len(), eta.first().map_or(0, |e| e.len()));
    let dim = (h * n) as f64;
    let normal = &prop.normal;
    params.path[l] = coloured(prior, eta);
    let cur_ll = if cur_ll.is_finite() {
        cur_ll
    } else {
        density(params)
    };
    let theta: Vec<f64> = (0..h * n).map(|i| eta[i / n][i % n]).collect();
    // log t_ν up to a constant, and η's own prior.
    let log_t = |t: &[f64]| -0.5 * (T_DF + dim) * (1.0 - 2.0 * normal.log_density(t) / T_DF).ln();
    let own = |t: &[f64]| -0.5 * t.iter().map(|x| x * x).sum::<f64>();
    let delta2 = -2.0 * normal.log_density(&theta);
    let scale = 0.5 * (T_DF + delta2)
        / Gamma::new(0.5 * (T_DF + dim), 1.0)
            .expect("a positive shape")
            .sample(rng);
    let base = cur_ll + own(&theta) - log_t(&theta);
    let at = |y: &DVector<f32>| -> Vec<f64> {
        y.iter()
            .zip(&normal.mode)
            .map(|(&v, m)| f64::from(v) + m)
            .collect()
    };
    let x = DVector::from_iterator(
        h * n,
        theta.iter().zip(&normal.mode).map(|(t, m)| (t - m) as f32),
    );
    let nu = (normal.draw(rng) * scale.sqrt()).map(|v| v as f32);
    let probe = params.clone();
    // Relative to the current value, so f32 keeps the slice's resolution.
    let lnpdf = |y: &DVector<f32>| {
        let t = at(y);
        let mut p = probe.clone();
        p.path[l] = coloured(prior, &unflatten(&t, h, n));
        (density(&p) + own(&t) - log_t(&t) - base) as f32
    };
    let (next, _) = elliptical_slice_step(&x, &nu, &lnpdf, 0.0, rng);
    let eta = unflatten(&at(&next), h, n);
    params.path[l] = coloured(prior, &eta);
    let ll = density(params);
    (eta, ll)
}

/// The paths' log density with every pseudobulk's lineage and grid point
/// summed out (`η`'s own prior aside): the multinomial marginal
/// `Σ_p log Σ_{ℓ,k} π_ℓ (1/n) p(x_p | ℓ, k)` plus [`genes_log_prior`].
pub fn collapsed_path_log_density(
    params: &Params,
    reads: &Reads,
    log_w: &[f64],
    gene_prior: &GenePrior,
) -> f64 {
    let lv = params.levels();
    let n = lv.points;
    let tab = table(reads, &lv);
    let ln_n = (n as f64).ln();
    let marginal: f64 = (0..tab.nrows())
        .map(|p| log_posterior_row(&tab, log_w, n, p).1 - ln_n)
        .sum();
    marginal + genes_log_prior(params, gene_prior)
}

/// Lineage `l`'s path with the latents summed out: the gradient of
/// [`collapsed_path_log_density`] (with `η`'s prior) in its whitened
/// coordinates and the collapsed precision by Louis's identity,
///
/// ```text
/// I_c = Σ_p E_q[I_pk] − Var_q[s_pk]
/// ```
///
/// `s_pk` pseudobulk `p`'s score at grid point `k` of lineage `l`
/// (`∂/∂ū_gk = u_pg/ū_gk − N_p/Z_k`, the multinomial's), `q` its posterior
/// over every lineage's grid points (a score on another lineage is 0),
/// `E_q[I_pk]` the multinomial's information (the Poisson's at the profiled
/// depth `N_p/Z_k`, less its fixed total's `N_p/Z_k² ∂Z ∂Zᵀ`). The variance
/// is the information the latents take away: a shift of the path with every
/// time along. The whitened precision is clipped at `η`'s prior's.
pub fn collapsed_newton(
    params: &Params,
    l: usize,
    prior: &GridPrior,
    reads: &Reads,
    log_w: &[f64],
    gene_prior: &GenePrior,
) -> (DVector<f64>, DMatrix<f64>) {
    let lv = params.levels();
    let n = lv.points;
    let h = params.rho.first().map_or(0, Vec::len);
    let (genes, pbs, hn) = (params.beta.len(), reads.u.nrows(), h * n);
    let tab = table(reads, &lv);
    // Every pseudobulk's posterior weight on lineage `l`'s grid points.
    let q: Vec<Vec<f64>> = (0..pbs)
        .map(|p| {
            let (log_post, _) = log_posterior_row(&tab, log_w, n, p);
            (0..n).map(|k| log_post[l * n + k].exp()).collect()
        })
        .collect();
    let z: Vec<f64> = (0..n)
        .map(|k| lv.u.row(l * n + k).sum() + lv.s.row(l * n + k).sum())
        .collect();
    let total: Vec<f64> = (0..pbs).map(|p| reads.total(p)).collect();
    // Σ_p q_pk N_p/Z_k (the Poisson information's depth) and Σ_p q_pk N_p/Z_k²
    // (the multinomial's correction for the fixed total).
    let pooled: Vec<f64> = (0..n)
        .map(|k| (0..pbs).map(|p| q[p][k] * total[p] / z[k]).sum())
        .collect();
    let pooled2: Vec<f64> = (0..n).map(|k| pooled[k] / z[k]).collect();
    let step = params.t_len[l] / (n - 1) as f64;
    let path = &params.path[l];
    // Per gene, folded in parallel: the score's pieces `W_u, W_s` (rows of
    // `[G × n·Hn]`, to be weighted by the reads in one product), `R = ∂Z/∂z`
    // `[n × Hn]` and the Poisson expected information `[Hn × Hn]`, the last
    // two summed as they go (never one per gene held).
    type Acc = (Vec<(usize, Vec<f64>, Vec<f64>)>, DMatrix<f64>, DMatrix<f64>);
    let (rows_w, r_sum, expected): Acc = (0..genes)
        .into_par_iter()
        .fold(
            || (Vec::new(), DMatrix::zeros(n, hn), DMatrix::zeros(hn, hn)),
            |(mut rows_w, mut r_sum, mut expected), g| {
                let rho = &params.rho[g];
                let alpha: Vec<f64> = path
                    .iter()
                    .map(|zk| (dot(zk, rho) + params.base[g]).exp())
                    .collect();
                let (a_u, a_s) = response(params.beta[g], params.gamma[g], step, n);
                let (u, s): (Vec<f64>, Vec<f64>) = (0..n)
                    .map(|k| (lv.u[(l * n + k, g)], lv.s[(l * n + k, g)]))
                    .unzip();
                let (mut wu, mut ws) = (vec![0.0; n * hn], vec![0.0; n * hn]);
                for k in 0..n {
                    for c in 0..hn {
                        let (d, j) = (c / n, c % n);
                        let lead = rho[d] * alpha[j];
                        wu[k * hn + c] = lead * a_u[(k, j)] / u[k];
                        ws[k * hn + c] = lead * a_s[(k, j)] / s[k];
                        r_sum[(k, c)] += lead * (a_u[(k, j)] + a_s[(k, j)]);
                    }
                }
                rows_w.push((g, wu, ws));
                let wu_f = DMatrix::from_fn(n, n, |i, j| a_u[(i, j)] * pooled[i] / u[i]);
                let ws_f = DMatrix::from_fn(n, n, |i, j| a_s[(i, j)] * pooled[i] / s[i]);
                let fa = a_u.transpose() * wu_f + a_s.transpose() * ws_f;
                for a in 0..hn {
                    for b in 0..hn {
                        let (d, i, e, j) = (a / n, a % n, b / n, b % n);
                        expected[(a, b)] += rho[d] * rho[e] * alpha[i] * alpha[j] * fa[(i, j)];
                    }
                }
                (rows_w, r_sum, expected)
            },
        )
        .reduce(
            || (Vec::new(), DMatrix::zeros(n, hn), DMatrix::zeros(hn, hn)),
            |mut a, b| {
                a.0.extend(b.0);
                (a.0, a.1 + b.1, a.2 + b.2)
            },
        );
    let (mut w_u, mut w_s) = (DMatrix::zeros(genes, n * hn), DMatrix::zeros(genes, n * hn));
    for (g, wu, ws) in &rows_w {
        w_u.row_mut(*g).copy_from_slice(wu);
        w_s.row_mut(*g).copy_from_slice(ws);
    }
    // Every pseudobulk's scores on every grid point at once, `[P × n·Hn]`.
    let weighted = &reads.u * &w_u + &reads.s * &w_s;
    let (grad, second, outer) = (0..pbs)
        .into_par_iter()
        .map(|p| {
            let sc = DMatrix::from_fn(n, hn, |k, c| {
                weighted[(p, k * hn + c)] - r_sum[(k, c)] * total[p] / z[k]
            });
            let w = DVector::from_column_slice(&q[p]);
            let mean = sc.transpose() * &w;
            let root = DMatrix::from_fn(n, hn, |k, c| sc[(k, c)] * w[k].sqrt());
            let second = root.transpose() * &root;
            let outer = &mean * mean.transpose();
            (mean, second, outer)
        })
        .reduce(
            || {
                (
                    DVector::zeros(hn),
                    DMatrix::zeros(hn, hn),
                    DMatrix::zeros(hn, hn),
                )
            },
            |a, b| (a.0 + b.0, a.1 + b.1, a.2 + b.2),
        );
    // The multinomial's fixed total: `− Σ_k (Σ_p q_pk N_p/Z_k²) ∂Z_k ∂Z_kᵀ`.
    let mut fixed_total = DMatrix::<f64>::zeros(hn, hn);
    for (k, &w) in pooled2.iter().enumerate() {
        let rk = r_sum.row(k).transpose();
        fixed_total += &rk * rk.transpose() * w;
    }
    let info = expected - fixed_total - (second - outer);
    let grad_z = DMatrix::from_fn(n, h, |j, d| grad[d * n + j]);
    let (g, prec) = whitened(params, l, prior, gene_prior, &grad_z, &info);
    // The whole whitened precision (data and the genes' prior through z̄)
    // clipped at `η`'s prior: eigenvalues at least 1.
    let eig = prec.symmetric_eigen();
    let clipped = eig.eigenvalues.map(|v| v.max(1.0));
    let prec = &eig.eigenvectors * DMatrix::from_diagonal(&clipped) * eig.eigenvectors.transpose();
    (g, prec)
}

/// The Laplace approximation of lineage `l`'s path with the latents summed
/// out, by at most `max_iter` steps of Fisher scoring from `anchor` (the
/// previous mode).
#[allow(clippy::too_many_arguments)]
pub fn collapsed_laplace(
    params: &Params,
    l: usize,
    anchor: &[DVector<f64>],
    prior: &GridPrior,
    reads: &Reads,
    log_w: &[f64],
    gene_prior: &GenePrior,
    max_iter: usize,
) -> PathProposal {
    scoring(
        params,
        l,
        anchor,
        prior,
        &|p: &Params| collapsed_path_log_density(p, reads, log_w, gene_prior),
        &|p: &Params| collapsed_newton(p, l, prior, reads, log_w, gene_prior),
        max_iter,
    )
}

/// One generalized elliptical slice move of lineage `l`'s whitened path
/// for [`collapsed_path_log_density`] (latents summed out): the path and
/// every time move together. The sampler must draw the latents next.
#[allow(clippy::too_many_arguments)]
pub fn collapsed_path_step(
    params: &mut Params,
    l: usize,
    eta: &[DVector<f64>],
    cur: f64,
    prior: &GridPrior,
    reads: &Reads,
    log_w: &[f64],
    gene_prior: &GenePrior,
    prop: &PathProposal,
    rng: &mut impl Rng,
) -> (Vec<DVector<f64>>, f64) {
    t_ess(
        params,
        l,
        eta,
        cur,
        prior,
        prop,
        &|p: &Params| collapsed_path_log_density(p, reads, log_w, gene_prior),
        rng,
    )
}
