//! Every cell's place in the embedding, the genes and the curves fixed:
//!
//! ```text
//! θ_c = argmin_θ  − Σ_g n_cg log softmax_g(θᵀ w_g + b_g) + (1/2σ²) |θ − M_c|²
//! ```
//!
//! the softmax being the Poisson `Σ_g [e^{θᵀw_g + b_g + c} − n_cg (θᵀw_g +
//! b_g + c)]` with its free intercept `c` at its optimum, the pseudobulk
//! fit's own objective for one point. `M_c = Σ_{ℓ,k} q_c(ℓ,k) μ_ℓ(k)`, `q_c`
//! the cell's posterior on the grid ([`grid_prior`]). Solved by Newton's
//! method from `M_c`, a cell at a time within a block (each cell's own
//! `H × H` system), the step halved where it does not lower the objective,
//! the tensors on the cells still moving alone; scores are clamped to
//! `±30`.

use super::model::SCORE_CLAMP;
use legume_numeric::candle::candle_core::{Device, Result, Tensor};
use legume_numeric::candle::candle_nn::ops::log_softmax;
use nalgebra::{DMatrix, DVector};

/// Newton steps a block at most.
const MAX_STEPS: usize = 50;

/// Halvings of a step at most.
const MAX_HALVINGS: usize = 20;

/// A step this small (relative to the point) ends a cell's search.
const STEP_TOL: f64 = 1e-6;

/// A step is taken if the objective rises by no more than this share of
/// it, single precision's own rounding.
const RISE_TOL: f64 = 1e-6;

/// Added to the Hessian's diagonal, for a cell with no prior.
const RIDGE: f64 = 1e-6;

/// The genes, fixed, on the device.
pub struct Placer {
    /// `[G × H]`.
    w: Tensor,
    /// `[1 × G]`.
    b: Tensor,
    /// `[G × H²]` every gene's `w_g w_gᵀ`.
    ww: Tensor,
    genes: usize,
    dim: usize,
}

impl Placer {
    /// From `w [G × H]` row-major and `b [G]`.
    pub fn new(w: &[f32], b: &[f32], dim: usize, dev: &Device) -> anyhow::Result<Self> {
        let genes = b.len();
        anyhow::ensure!(
            dim > 0 && w.len() == genes * dim,
            "{} gene vectors' entries for {genes} genes of dimension {dim}",
            w.len()
        );
        let ww: Vec<f32> = (0..genes)
            .flat_map(|g| {
                let row = &w[g * dim..(g + 1) * dim];
                (0..dim * dim).map(move |e| row[e / dim] * row[e % dim])
            })
            .collect();
        Ok(Self {
            w: Tensor::from_slice(w, (genes, dim), dev)?,
            b: Tensor::from_slice(b, (1, genes), dev)?,
            ww: Tensor::from_vec(ww, (genes, dim * dim), dev)?,
            genes,
            dim,
        })
    }

    /// `[B × G]` scores at `theta [B × H]`, clamped.
    fn scores(&self, theta: &Tensor) -> Result<Tensor> {
        theta
            .matmul(&self.w.t()?)?
            .broadcast_add(&self.b)?
            .clamp(-SCORE_CLAMP, SCORE_CLAMP)
    }

    /// Every row's objective at `theta [B × H]` and its log-softmax `[B ×
    /// G]`, `reads [B × G]` its reads.
    fn objective(
        &self,
        reads: &Tensor,
        theta: &[f32],
        centre: &[f32],
        inv_var: f64,
    ) -> anyhow::Result<(Vec<f64>, Tensor)> {
        let (bn, h) = (theta.len() / self.dim, self.dim);
        let t = Tensor::from_slice(theta, (bn, h), reads.device())?;
        let ls = log_softmax(&self.scores(&t)?, 1)?;
        let fit: Vec<f32> = (&ls * reads)?.sum(1)?.neg()?.to_vec1()?;
        let f = (0..bn)
            .map(|i| {
                let d2: f64 = (0..h)
                    .map(|d| f64::from(theta[i * h + d] - centre[i * h + d]).powi(2))
                    .sum();
                f64::from(fit[i]) + 0.5 * inv_var * d2
            })
            .collect();
        Ok((f, ls))
    }

    /// `[B × H]` row-major, every cell of `reads [B × G]` placed with its
    /// prior centre `centre [B × H]` and precision `inv_var = 1/σ²` (zero:
    /// no prior).
    pub fn place(
        &self,
        reads: &DMatrix<f32>,
        centre: &[f32],
        inv_var: f32,
    ) -> anyhow::Result<Vec<f32>> {
        let (bn, g) = reads.shape();
        let h = self.dim;
        anyhow::ensure!(
            g == self.genes && centre.len() == bn * h,
            "{bn} cells of {g} genes and {} centre entries for {} genes of dimension {h}",
            centre.len(),
            self.genes
        );
        let dev = self.w.device();
        let index = |rows: &[usize]| {
            let v: Vec<u32> = rows.iter().map(|&r| r as u32).collect();
            Tensor::from_vec(v, (rows.len(),), dev)
        };
        // Row-major: the transpose's column-major storage.
        let mut n_t = Tensor::from_slice(reads.transpose().as_slice(), (bn, g), dev)?;
        let depth: Vec<f64> = reads
            .row_iter()
            .map(|r| r.iter().map(|&v| f64::from(v)).sum())
            .collect();
        let mut depth_t = Tensor::from_vec(
            depth.iter().map(|&v| v as f32).collect::<Vec<_>>(),
            (bn, 1),
            dev,
        )?;
        let inv_var = f64::from(inv_var);
        let mut theta = centre.to_vec();
        // `ls` the log-softmax at every live row's θ; `live[r]` row `r`'s
        // cell, the rows cut to the active ones once they are half.
        let (mut f, mut ls) = self.objective(&n_t, &theta, centre, inv_var)?;
        let mut live: Vec<usize> = (0..bn).collect();
        let mut active = vec![true; bn];
        for _ in 0..MAX_STEPS {
            let mut on: Vec<usize> = (0..live.len()).filter(|&r| active[live[r]]).collect();
            if on.is_empty() {
                break;
            }
            if on.len() * 2 < live.len() {
                let idx = index(&on)?;
                n_t = n_t.index_select(&idx, 0)?;
                ls = ls.index_select(&idx, 0)?;
                depth_t = depth_t.index_select(&idx, 0)?;
                live = on.iter().map(|&r| live[r]).collect();
                on = (0..live.len()).collect();
            }
            let m = live.len();
            let p = ls.exp()?;
            let resid = (p.broadcast_mul(&depth_t)? - &n_t)?;
            let grad_fit: Vec<f32> = resid.matmul(&self.w)?.flatten_all()?.to_vec1()?;
            let pw: Vec<f32> = p.matmul(&self.w)?.flatten_all()?.to_vec1()?;
            let pww: Vec<f32> = p.matmul(&self.ww)?.flatten_all()?.to_vec1()?;
            let mut step = vec![0f32; m * h];
            for &r in &on {
                let i = live[r];
                let grad = DVector::from_fn(h, |d, _| {
                    f64::from(grad_fit[r * h + d])
                        + inv_var * f64::from(theta[i * h + d] - centre[i * h + d])
                });
                let hess = DMatrix::from_fn(h, h, |a, c| {
                    let fit = depth[i]
                        * (f64::from(pww[r * h * h + a * h + c])
                            - f64::from(pw[r * h + a]) * f64::from(pw[r * h + c]));
                    fit + if a == c { inv_var + RIDGE } else { 0.0 }
                });
                let dir = match hess.clone().cholesky() {
                    Some(ch) => -ch.solve(&grad),
                    None => -grad / hess.diagonal().max().max(1.0),
                };
                for d in 0..h {
                    step[r * h + d] = dir[d] as f32;
                }
            }
            // Halve every row's step until its objective does not rise; the
            // log-softmax of every one taken, by the rows it is of.
            let mut scale = vec![1f32; m];
            let mut pending = on.clone();
            let mut taken: Vec<(Vec<usize>, Tensor)> = Vec::new();
            for _ in 0..=MAX_HALVINGS {
                if pending.is_empty() {
                    break;
                }
                let (mut cand, mut at) = (Vec::with_capacity(pending.len() * h), Vec::new());
                for &r in &pending {
                    let i = live[r];
                    at.extend_from_slice(&centre[i * h..(i + 1) * h]);
                    cand.extend((0..h).map(|d| theta[i * h + d] + scale[r] * step[r * h + d]));
                }
                let rows = if pending.len() == m {
                    n_t.clone()
                } else {
                    n_t.index_select(&index(&pending)?, 0)?
                };
                let (fc, lc) = self.objective(&rows, &cand, &at, inv_var)?;
                let (mut kept, mut of, mut still) = (Vec::new(), Vec::new(), Vec::new());
                for (j, &r) in pending.iter().enumerate() {
                    let i = live[r];
                    if fc[j].is_finite() && fc[j] <= f[i] + RISE_TOL * (1.0 + f[i].abs()) {
                        theta[i * h..(i + 1) * h].copy_from_slice(&cand[j * h..(j + 1) * h]);
                        f[i] = fc[j];
                        kept.push(j);
                        of.push(r);
                    } else {
                        scale[r] *= 0.5;
                        still.push(r);
                    }
                }
                if !kept.is_empty() {
                    let lc = if kept.len() == pending.len() {
                        lc
                    } else {
                        lc.index_select(&index(&kept)?, 0)?
                    };
                    taken.push((of, lc));
                }
                pending = still;
            }
            ls = match taken.as_slice() {
                [] => ls,
                [(of, lc)] if of.len() == m => lc.clone(),
                _ => {
                    let mut gather: Vec<usize> = (0..m).collect();
                    let mut parts = vec![ls];
                    let mut next = m;
                    for (of, lc) in taken {
                        for (j, &r) in of.iter().enumerate() {
                            gather[r] = next + j;
                        }
                        next += of.len();
                        parts.push(lc);
                    }
                    Tensor::cat(&parts, 0)?.index_select(&index(&gather)?, 0)?
                }
            };
            let mut unmoved = vec![false; m];
            pending.iter().for_each(|&r| unmoved[r] = true);
            for &r in &on {
                let i = live[r];
                let moved: f64 = (0..h)
                    .map(|d| f64::from(scale[r] * step[r * h + d]).powi(2))
                    .sum::<f64>()
                    .sqrt();
                let size: f64 = (0..h)
                    .map(|d| f64::from(theta[i * h + d]).powi(2))
                    .sum::<f64>()
                    .sqrt();
                // Done when it barely moves, or no halving lowered it.
                if unmoved[r] || moved < STEP_TOL * (1.0 + size) {
                    active[i] = false;
                }
            }
        }
        Ok(theta)
    }
}

/// A cell's posterior on the grid, `(lineage, position, mass)`: every
/// lineage's posterior `p_ℓ` spread over its `n` positions as a Gaussian in
/// phase about `τ / T_ℓ`, of sd `√((τ_sd / T_ℓ)² + 1/(n − 1)²)` (a grid step
/// wide at least), `T_ℓ` its duration. `None` when the time or every
/// lineage's posterior is unknown.
#[must_use]
pub fn grid_prior(
    p: &[f32],
    tau: f32,
    tau_sd: f32,
    duration: &[f64],
    n: usize,
) -> Option<Vec<(usize, usize, f32)>> {
    if !tau.is_finite() {
        return None;
    }
    let total: f64 = p
        .iter()
        .filter(|v| v.is_finite() && **v > 0.0)
        .map(|&v| f64::from(v))
        .sum();
    if total <= 0.0 {
        return None;
    }
    let step = 1.0 / (n.max(2) - 1) as f64;
    let sd_tau = if tau_sd.is_finite() {
        f64::from(tau_sd.max(0.0))
    } else {
        0.0
    };
    let mut out = Vec::new();
    for (l, (&pl, &t_l)) in p.iter().zip(duration).enumerate() {
        if !(pl.is_finite() && pl > 0.0 && t_l > 0.0) {
            continue;
        }
        let (u, sd) = (
            f64::from(tau) / t_l,
            ((sd_tau / t_l).powi(2) + step * step).sqrt(),
        );
        // On the log scale, from its peak: a time far off the grid still
        // puts its lineage's mass on the grid's nearest end.
        let logw: Vec<f64> = (0..n)
            .map(|k| -((k as f64 * step - u) / sd).powi(2) / 2.0)
            .collect();
        let peak = logw.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        if !peak.is_finite() {
            continue;
        }
        let w: Vec<f64> = logw.iter().map(|v| (v - peak).exp()).collect();
        let z: f64 = w.iter().sum();
        let mass = f64::from(pl) / total;
        out.extend(
            w.iter()
                .enumerate()
                .map(|(k, &v)| (l, k, (mass * v / z) as f32))
                .filter(|m| m.2 > 1e-8),
        );
    }
    // Every kept mass, back to one over the masses floored away.
    let kept: f32 = out.iter().map(|m| m.2).sum();
    if kept.is_nan() || kept <= 0.0 {
        return None;
    }
    out.iter_mut().for_each(|m| m.2 /= kept);
    Some(out)
}

/// `[H]` `Σ q · table[ℓ, k]` over `q`'s `(lineage, position, mass)`, `table`
/// `[L × n × H]` row-major (the curves, or their velocities).
#[must_use]
pub fn expect_on_grid(
    q: impl IntoIterator<Item = (usize, usize, f32)>,
    table: &[f32],
    n: usize,
    h: usize,
) -> Vec<f32> {
    let mut out = vec![0f32; h];
    for (l, k, m) in q {
        let row = &table[(l * n + k) * h..(l * n + k + 1) * h];
        for (o, v) in out.iter_mut().zip(row) {
            *o += m * v;
        }
    }
    out
}
