//! The sampler's blocks, each a draw from its conditional.
//!
//! Every random stream is keyed by `(seed, item, sweep)`, so a draw does
//! not depend on how the items were split across threads.

use super::kernel::GridPrior;
use super::kinetics::fine_trajectory;
use super::likelihood::{gene_log_likelihood, share_mode, softplus, Depth, Reads};
use super::model::{branch_ramp, Params};
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
    lambda: &[Depth],
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
    lambda: &[Depth],
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
    lambda: &[Depth],
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
pub fn path_ll(params: &Params, l: usize, reads: &Reads, rows: &[usize], lambda: &[Depth]) -> f64 {
    let (n, positions) = (params.path[l].len(), params.positions());
    let on: Vec<(usize, usize)> = rows
        .iter()
        .enumerate()
        .filter(|&(_, &r)| r / positions == l)
        .map(|(p, &r)| (p, r - l * positions))
        .collect();
    let h = params.t_len[l] / (n - 1) as f64;
    (0..params.beta.len())
        .into_par_iter()
        .map(|g| {
            let alpha: Vec<f64> = params.path[l]
                .iter()
                .map(|z| (dot(z, &params.rho[g]) + params.base[g]).exp())
                .collect();
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
fn response(beta: f64, gamma: f64, h: f64, n: usize, sub: usize) -> (DMatrix<f64>, DMatrix<f64>) {
    let positions = (n - 1) * sub.max(1) + 1;
    let (mut a_u, mut a_s) = (DMatrix::zeros(positions, n), DMatrix::zeros(positions, n));
    for j in 0..n {
        let mut e = vec![0.0; n];
        e[j] = 1.0;
        let (u, s) = fine_trajectory(&e, beta, gamma, h, sub);
        for k in 0..positions {
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
/// The reads on lineage `l` pooled per position, `U_gk = Σ_{p at k} u_pg`,
/// `Λ^u_k = Σ_{p at k} λ^u_p` (and the spliced track's likewise):
/// `ℓ = Σ_{g,k} U_gk log ū_gk − Λ^u_k ū_gk (+ s̄)`,
/// `∂ℓ/∂ū_gk = U_gk/ū_gk − Λ^u_k`, Fisher `Λ^u_k/ū_gk`; then through `ū = A_u α`,
/// `α_gj = exp(ρ_gᵀ z_j + b_g)` and `z_d = C η_d`.
#[allow(clippy::too_many_arguments)]
fn path_newton(
    space: &Space,
    params: &Params,
    x: &DVector<f64>,
    reads: &Reads,
    rows: &[usize],
    lambda: &[Depth],
    gene_prior: &GenePrior,
) -> Newton {
    let parts: Vec<(DMatrix<f64>, DMatrix<f64>)> = moved_by(params, space.block)
        .into_iter()
        .map(|l| lineage_information(params, l, reads, rows, lambda))
        .collect();
    space.information(params, x, gene_prior, &parts)
}

/// The reads' gradient `[n × H]` and Fisher information `[Hn × Hn]`
/// (flattened `d n + k`) in lineage `l`'s path `z`, as [`path_newton`].
fn lineage_information(
    params: &Params,
    l: usize,
    reads: &Reads,
    rows: &[usize],
    lambda: &[Depth],
) -> (DMatrix<f64>, DMatrix<f64>) {
    let path = &params.path[l];
    let n = path.len();
    let h = path.first().map_or(0, Vec::len);
    let genes = params.beta.len();
    let step = params.t_len[l] / (n - 1) as f64;
    let positions = params.positions();
    let on: Vec<(usize, usize)> = rows
        .iter()
        .enumerate()
        .filter(|&(_, &r)| r / positions == l)
        .map(|(p, &r)| (p, r - l * positions))
        .collect();
    // The depths pooled per position and track.
    let (mut pooled_u, mut pooled_s) = (vec![0.0; positions], vec![0.0; positions]);
    for &(p, k) in &on {
        pooled_u[k] += lambda[p][0];
        pooled_s[k] += lambda[p][1];
    }

    // Summed over genes: the gradient wrt z `[n × H]`, the Fisher `[Hn × Hn]`.
    (0..genes)
        .into_par_iter()
        .map(|g| {
            let rho = &params.rho[g];
            let alpha: Vec<f64> = path
                .iter()
                .map(|z| (dot(z, rho) + params.base[g]).exp())
                .collect();
            let (a_u, a_s) = response(params.beta[g], params.gamma[g], step, n, params.sub);
            let av = DVector::from_column_slice(&alpha);
            let (u, s) = (&a_u * &av, &a_s * &av);
            let (mut ru, mut rs) = (vec![0.0; positions], vec![0.0; positions]);
            for &(p, k) in &on {
                ru[k] += reads.u[(p, g)];
                rs[k] += reads.s[(p, g)];
            }
            let eu = DVector::from_fn(positions, |k, _| ru[k] / u[k] - pooled_u[k]);
            let es = DVector::from_fn(positions, |k, _| rs[k] / s[k] - pooled_s[k]);
            // ∂ℓ/∂α_j, then ∂ℓ/∂z_jd = α_j ρ_d ∂ℓ/∂α_j.
            let d_alpha = a_u.transpose() * eu + a_s.transpose() * es;
            let gz = DMatrix::from_fn(n, h, |j, d| alpha[j] * rho[d] * d_alpha[j]);
            // Wrt α: A_uᵀ W_u A_u + A_sᵀ W_s A_s; wrt z, α_i α_j ρ_d ρ_e times it.
            let wu = DMatrix::from_fn(positions, n, |i, j| a_u[(i, j)] * pooled_u[i] / u[i]);
            let ws = DMatrix::from_fn(positions, n, |i, j| a_s[(i, j)] * pooled_s[i] / s[i]);
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
        )
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

    /// `∂z/∂(η, x)` of a lineage the block moves, `[Hn × dim]` (rows
    /// `d n + k`), at `params` (placed at `x`): the map `M` per dimension
    /// ([`path_map`]), and for the branch time `∂r/∂t_b ⊙ C η_d` times
    /// `dt_b/dx`.
    fn jacobian(&self, params: &Params, x: &[f64]) -> DMatrix<f64> {
        let (h, n) = (self.h, self.n);
        let m = path_map(params, self.prior, self.block);
        let mut j = DMatrix::zeros(h * n, self.dim());
        for d in 0..h {
            j.view_mut((d * n, d * n), (n, n)).copy_from(&m);
        }
        if self.branch {
            let tb = params.branch.as_ref().expect("a trunk")[self.block - 1];
            let kappa = self.prior.kappa();
            let eta = self.eta(x);
            for (d, e) in eta.iter().enumerate() {
                let w = self.prior.colour(e);
                for k in 0..n {
                    let u = (k as f64 / (n - 1) as f64 - tb).max(0.0) / kappa;
                    let dr = -std::f64::consts::SQRT_2 * 2.0 * u * (-u * u).exp() / kappa;
                    j[(d * n + k, h * n)] = dr * w[k] * tb * (1.0 - tb);
                }
            }
        }
        j
    }

    /// From the reads' gradient `[n × H]` and information `[Hn × Hn]` in
    /// `z` (flattened `d n + k`) of every lineage the block moves (`parts`):
    /// the gradient and Gauss–Newton precision in the coordinates `x`,
    /// through the Jacobian ([`Self::jacobian`]), with their own prior and
    /// the genes' prior's exact curvature through `z̄`,
    /// `−½ Σ_g ((b_g + ρ_gᵀz̄ − log γ_g − m)/s)²`, `z̄_d` the mean of every
    /// lineage's path at every node.
    fn information(
        &self,
        params: &Params,
        x: &DVector<f64>,
        gene_prior: &GenePrior,
        parts: &[(DMatrix<f64>, DMatrix<f64>)],
    ) -> Newton {
        let (h, n, dim) = (self.h, self.n, self.dim());
        let j = self.jacobian(params, x.as_slice());
        let (mean, sd) = gene_prior.log_spliced;
        let z_bar = params.path_mean();
        let mut prior_grad = vec![0.0; h];
        let mut prior_curv = DMatrix::<f64>::zeros(h, h);
        for g in 0..params.beta.len() {
            let rho = &params.rho[g];
            let r = params.base[g] + dot(rho, &z_bar) - params.gamma[g].ln() - mean;
            for d in 0..h {
                prior_grad[d] -= r * rho[d] / (sd * sd);
                for e in 0..h {
                    prior_curv[(d, e)] += rho[d] * rho[e] / (sd * sd);
                }
            }
        }
        // ∂z̄_d/∂(η, x): every moved lineage's rows of dimension d, over L n.
        let points = (params.path.len() * n) as f64;
        let cols: Vec<DVector<f64>> = (0..h)
            .map(|d| {
                DVector::from_fn(dim, |c, _| {
                    (0..n).map(|k| j[(d * n + k, c)]).sum::<f64>() * parts.len() as f64 / points
                })
            })
            .collect();
        let hn = h * n;
        let mut grad = DVector::zeros(dim);
        let mut prec = DMatrix::zeros(dim, dim);
        for (grad_z, fisher_z) in parts {
            let g = DVector::from_fn(hn, |i, _| grad_z[(i % n, i / n)]);
            grad += j.transpose() * g;
            prec += j.transpose() * fisher_z * &j;
        }
        for d in 0..h {
            grad += &cols[d] * prior_grad[d];
            for e in 0..h {
                prec += &cols[d] * cols[e].transpose() * prior_curv[(d, e)];
            }
        }
        // The coordinates' own prior.
        for i in 0..hn {
            grad[i] -= x[i];
            prec[(i, i)] += 1.0;
        }
        if self.branch {
            let q = sigmoid(x[hn]);
            grad[hn] += 1.0 - 2.0 * q;
            prec[(hn, hn)] += 2.0 * q * (1.0 - q);
        }
        (grad, prec)
    }
}

/// The normal approximation of a path's conditional, as [`GeneProposal`]
/// (its mode and the Cholesky factor of its precision over [`WIDEN`]²), in
/// its block's coordinates ([`Space`]).
#[derive(Clone)]
pub struct PathProposal {
    normal: GeneProposal,
}

impl PathProposal {
    /// The mode's whitened paths, one `[n]` per dimension.
    pub fn mode(&self, h: usize) -> Vec<DVector<f64>> {
        let n = self.normal.mode.len() / h.max(1);
        (0..h)
            .map(|d| DVector::from_fn(n, |k, _| self.normal.mode[d * n + k]))
            .collect()
    }
}

impl PathProposal {
    /// A departure's branch time at the mode (`None` for a block without).
    pub fn branch_time(&self, h: usize, n: usize) -> Option<f64> {
        (self.normal.mode.len() > h * n).then(|| sigmoid(self.normal.mode[h * n]))
    }
}

/// `[H][n]` from flattened `d n + k`.
fn unflatten(x: &[f64], h: usize, n: usize) -> Vec<DVector<f64>> {
    (0..h)
        .map(|d| DVector::from_fn(n, |k, _| x[d * n + k]))
        .collect()
}

/// The Laplace approximation of path block `l`'s conditional (rows,
/// depths, genes held) by damped Gauss–Newton from `anchor` (whitened; not
/// the chain's current path but, in the sampler, the previous sweep's
/// mode; a departure's branch time from `params`), the Fisher information
/// analytic ([`path_newton`]).
#[allow(clippy::too_many_arguments)]
pub fn path_laplace(
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
    scoring(
        &space,
        params,
        &space.coords(params, anchor),
        &|p: &Params| path_log_density(p, l, reads, rows, lambda, gene_prior),
        &|p: &Params, x: &DVector<f64>| path_newton(&space, p, x, reads, rows, lambda, gene_prior),
    )
}

/// A gradient and a precision.
type Newton = (DVector<f64>, DMatrix<f64>);

/// Damped Fisher scoring of `density` plus the coordinates' own prior in
/// `space` from `start`, `newton` its gradient and precision; stopped when
/// the predicted rise `gᵀP⁻¹g` is under a hundredth of a nat. The normal at
/// the end, as a [`PathProposal`].
fn scoring(
    space: &Space,
    params: &Params,
    start: &DVector<f64>,
    density: &dyn Fn(&Params) -> f64,
    newton: &dyn Fn(&Params, &DVector<f64>) -> Newton,
) -> PathProposal {
    let base = params.clone();
    let mut probe = params.clone();
    let objective = |p: &mut Params, x: &DVector<f64>| {
        space.place(p, &base, x.as_slice());
        density(p) + space.own(x.as_slice())
    };
    let mut x = start.clone();
    let mut fx = objective(&mut probe, &x);
    let mut prec;
    let mut iter = 0;
    loop {
        let grad;
        (grad, prec) = newton(&probe, &x);
        let step = prec
            .clone()
            .cholesky()
            .expect("a prior's precision plus a Fisher information is positive definite")
            .solve(&grad);
        iter += 1;
        if grad.dot(&step) < 1e-2 || iter > 20 {
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
            space.place(&mut probe, &base, x.as_slice());
            break;
        };
        (x, fx) = (y, fy);
    }
    let precision_chol = (prec / (WIDEN * WIDEN))
        .cholesky()
        .expect("a prior's precision plus a Fisher information is positive definite")
        .l();
    PathProposal {
        normal: GeneProposal {
            mode: x.iter().copied().collect(),
            precision_chol,
        },
    }
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
    let normal = &prop.normal;
    let theta: Vec<f64> = space.coords(params, eta).iter().copied().collect();
    let base = params.clone();
    space.place(params, &base, &theta);
    let cur_ll = if cur_ll.is_finite() {
        cur_ll
    } else {
        density(params)
    };
    // log t_ν up to a constant.
    let log_t =
        |t: &[f64]| -0.5 * (T_DF + dim as f64) * (1.0 - 2.0 * normal.log_density(t) / T_DF).ln();
    let delta2 = -2.0 * normal.log_density(&theta);
    let scale = 0.5 * (T_DF + delta2)
        / Gamma::new(0.5 * (T_DF + dim as f64), 1.0)
            .expect("a positive shape")
            .sample(rng);
    let offset = cur_ll + space.own(&theta) - log_t(&theta);
    let at = |y: &DVector<f32>| -> Vec<f64> {
        y.iter()
            .zip(&normal.mode)
            .map(|(&v, m)| f64::from(v) + m)
            .collect()
    };
    let x = DVector::from_iterator(
        dim,
        theta.iter().zip(&normal.mode).map(|(t, m)| (t - m) as f32),
    );
    let nu = (normal.draw(rng) * scale.sqrt()).map(|v| v as f32);
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
