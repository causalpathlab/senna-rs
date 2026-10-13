//! The energy's tables: a trajectory in the genes' embedding space, its
//! velocity, and every grid time's weight on every (gene, track).
//!
//! ```text
//! z(τ) = Σ_m B_m(τ) A_m,   ż(τ) = Σ_m B'_m(τ) A_m,   ν(τ) = Σ_m B_m(τ) c_m
//! ψ_k(g, s) = ⟨z(τ_k), ρ_g⟩ + b_g,   π_k = softmax_g ψ_k(·, s),   ρ̄_k = Σ_g π_kg ρ_g
//! m_kg = γ_g + ⟨ρ_g − ρ̄_k, ż(τ_k)⟩ + ν(τ_k)                    = β u/s
//! ψ_k(g, u) = ψ_k(g, s) + log m⁺_kg − log β_g
//! W_(g,u) = W_(g,s) = ρ_g B⊥                  B⊥ = B with the curves' spans removed
//! ```
//!
//! Every lineage `ℓ` has its own curve `z_ℓ` and size change `ν_ℓ`
//! (independent timelines); `ρ`, `b`, `β` and `γ` are shared.
//!
//! `m` is the splicing half of the ODE, `β u = ds/dt + γ s`, with
//! `d log s_g/dτ = ⟨ρ_g − ρ̄, ż⟩ + ν` for a softmax level: the velocity `ż`
//! lives in `ρ`'s space and every gene's unspliced share reads it through its
//! own embedding. `m⁺`, a smooth floor of `m` at `0`, keeps `u ≥ 0` (a gene
//! shuts off no faster than `γ` allows).
//!
//! A cell's state `h ∈ ℝ^d` moves it by `B⊥ h` in the embedding space, so its
//! weights on the features are `W = ρ B⊥`, the same on both tracks: a state
//! moves which genes a cell reads, never the unspliced share of a gene. `B⊥`
//! is `B` with the directions the curve travels removed, so a state cannot
//! stand in for time.

use super::curve::Spline;
use legume_numeric::candle::candle_core::{DType, Device, Result, Tensor, Var};
use legume_numeric::candle::candle_nn::ops::{log_softmax, sigmoid};
use rand::rngs::StdRng;
use rand::SeedableRng;
use rand_distr::{Distribution, Normal};

/// Log range of the splicing and degradation rates.
pub const LOG_RATE: (f64, f64) = (-3.0, 7.0);
/// The floor's softness, as a share of `γ`: `m⁺ = εγ softplus(m/(εγ))`.
const FLOOR_SOFTNESS: f64 = 0.05;
/// Values are floored here before a log.
const TINY: f64 = 1e-30;

/// `log(1 + eˣ)`, stable for any `x`.
pub fn softplus(x: &Tensor) -> Result<Tensor> {
    x.relu()? + x.abs()?.neg()?.exp()?.affine(1.0, 1.0)?.log()?
}

/// `lo + (hi − lo) σ(v)`.
fn bounded(v: &Tensor, (lo, hi): (f64, f64)) -> Result<Tensor> {
    (sigmoid(v)? * (hi - lo))? + lo
}

/// `x` in `(lo, hi)` on the free scale of [`bounded`].
pub fn unbounded(x: f64, (lo, hi): (f64, f64)) -> f64 {
    let p = ((x - lo) / (hi - lo)).clamp(1e-4, 1.0 - 1e-4);
    (p / (1.0 - p)).ln()
}

/// Sizes of the energy.
#[derive(Clone, Copy, Debug)]
pub struct Shape {
    pub genes: usize,
    pub embedding: usize,
    pub knots: usize,
    /// Width of a cell's state.
    pub states: usize,
    /// Lineages: independent timelines, each its own curve.
    pub lineages: usize,
}

/// Where the parameters start.
#[derive(Clone, Copy)]
pub struct Init0<'a> {
    /// `[G × H]` the gene embedding, row-major.
    pub rho: &'a [f32],
    /// `[G]` the spliced baselines.
    pub base: &'a [f32],
    /// `[L·M × H]` every lineage's curve coefficients, row-major, lineage by
    /// lineage.
    pub curve: &'a [f32],
    pub beta: f64,
    pub gamma: f64,
}

/// The energy's parameters.
pub struct Energy {
    /// `[G × H]` the gene embedding.
    rho: Var,
    /// `[L·M × H]` every lineage's curve coefficients, `[L·M × 1]` its size
    /// curve's.
    coef: Var,
    size: Var,
    /// `[1 × G]` the spliced baselines.
    base: Var,
    /// `[1 × G]` free-scale log rates of splicing and degradation, shared by
    /// every lineage.
    beta: Var,
    gamma: Var,
    /// `[L × 1]` every lineage's log speed: how fast its time runs against
    /// the shared rates (a lineage spanning longer in real time moves slower
    /// per unit of its `τ`). Centred over the lineages where read, as a common
    /// scale of the speeds, `β` and `γ` cancels.
    speed: Var,
    /// `[H × d]` the states' loadings in the embedding space.
    load: Var,
    lineages: usize,
    knots: usize,
    /// 1 with the velocity read by the unspliced share, 0 at steady state
    /// (every gene's u/s at γ/β, which runs the same either way in time).
    motion: std::cell::Cell<f64>,
}

/// The tables an energy evaluation needs.
pub struct Tables {
    /// `[L·K × 2G]` every (lineage, grid time)'s log probability of every
    /// feature, unspliced genes first, then spliced; rows `ℓ·K + k`.
    pub psi: Tensor,
    /// `[1 × L·K]` every row's log prior: its lineage's log weight minus
    /// `log K`.
    pub bias: Tensor,
    /// `[2G × d]` the states' weights on every feature.
    pub w: Tensor,
    /// `[H × d]` the states' loadings with the curves' spans removed.
    pub load: Tensor,
    /// `[L·K × H]` every lineage's curve and velocity on the grid, `[L·K × 1]`
    /// its size change.
    pub curve: Tensor,
    pub velocity: Tensor,
    pub size: Tensor,
    pub lineages: usize,
}

impl Tables {
    /// The same tables, cut from the graph.
    pub fn detach(&self) -> Self {
        Self {
            psi: self.psi.detach(),
            bias: self.bias.detach(),
            w: self.w.detach(),
            load: self.load.detach(),
            curve: self.curve.detach(),
            velocity: self.velocity.detach(),
            size: self.size.detach(),
            lineages: self.lineages,
        }
    }

    /// Grid times per lineage.
    pub fn grid_len(&self) -> Result<usize> {
        Ok(self.curve.dim(0)? / self.lineages)
    }
}

impl Energy {
    /// An energy of `shape` starting at `init`, the states' loadings at small
    /// random values (`seed`).
    pub fn new(shape: Shape, init: &Init0, seed: u64, dev: &Device) -> Result<Self> {
        let (g, h, m, d, l) = (
            shape.genes,
            shape.embedding,
            shape.knots,
            shape.states,
            shape.lineages,
        );
        let mut rng = StdRng::seed_from_u64(seed);
        let normal = Normal::new(0.0f32, 0.1).expect("a valid sd");
        let load: Vec<f32> = (0..h * d).map(|_| normal.sample(&mut rng)).collect();
        let constant = |shape: (usize, usize), x: f64| -> Result<Var> {
            Var::from_tensor(&(Tensor::ones(shape, DType::F32, dev)? * x)?)
        };
        let rate = |r: f64| constant((1, g), unbounded(r.ln(), LOG_RATE));
        Ok(Self {
            rho: Var::from_tensor(&Tensor::from_slice(init.rho, (g, h), dev)?)?,
            coef: Var::from_tensor(&Tensor::from_slice(init.curve, (l * m, h), dev)?)?,
            size: constant((l * m, 1), 0.0)?,
            base: Var::from_tensor(&Tensor::from_slice(init.base, (1, g), dev)?)?,
            beta: rate(init.beta)?,
            gamma: rate(init.gamma)?,
            speed: constant((l, 1), 0.0)?,
            load: Var::from_tensor(&Tensor::from_vec(load, (h, d), dev)?)?,
            lineages: l,
            knots: m,
            motion: std::cell::Cell::new(1.0),
        })
    }

    /// Read the curves' velocity in the unspliced share (`true`) or hold
    /// every gene at its steady ratio (`false`), which no direction of time
    /// can tell apart.
    pub fn set_motion(&self, on: bool) {
        self.motion.set(if on { 1.0 } else { 0.0 });
    }

    /// Every parameter the exact phases fit: all but the states' loadings.
    pub fn vars(&self) -> Vec<Var> {
        vec![
            self.rho.clone(),
            self.coef.clone(),
            self.size.clone(),
            self.base.clone(),
            self.beta.clone(),
            self.gamma.clone(),
            self.speed.clone(),
        ]
    }

    /// Set every gene's splicing and degradation rates (`[1 × G]` natural
    /// logs), held inside their bounds.
    pub fn set_log_rates(&self, log_beta: &Tensor, log_gamma: &Tensor) -> Result<()> {
        let free = |x: &Tensor| -> Result<Tensor> {
            let (lo, hi) = LOG_RATE;
            let p = ((x.clamp(lo + 1e-3, hi - 1e-3)? - lo)? / (hi - lo))?;
            p.log()? - p.affine(-1.0, 1.0)?.log()?
        };
        self.beta.set(&free(log_beta)?)?;
        self.gamma.set(&free(log_gamma)?)
    }

    /// The parameters [`Self::snapshot`] and [`Self::restore`] carry.
    #[cfg(test)]
    fn kept(&self) -> [&Var; 7] {
        [
            &self.coef,
            &self.size,
            &self.beta,
            &self.gamma,
            &self.rho,
            &self.base,
            &self.speed,
        ]
    }

    /// The parameters' current values, to restore with [`Self::restore`].
    #[cfg(test)]
    pub fn snapshot(&self) -> Result<Vec<Tensor>> {
        self.kept().iter().map(|v| v.as_tensor().copy()).collect()
    }

    /// Restore what [`Self::snapshot`] took (or its leading part).
    #[cfg(test)]
    pub fn restore(&self, saved: &[Tensor]) -> Result<()> {
        for (v, t) in self.kept().iter().zip(saved) {
            v.set(t)?;
        }
        Ok(())
    }

    /// Run lineage `l` the other way: its curve `z(τ) → z(1 − τ)` and size
    /// change `ν(τ) → −ν(1 − τ)`, exactly, by reversing its coefficients (a
    /// clamped uniform B-spline has `B_m(1 − τ) = B_{M−1−m}(τ)`).
    pub fn mirror(&self, l: usize) -> Result<()> {
        let m = self.knots;
        let rev: Vec<u32> = (0..m as u32).rev().collect();
        let rev = Tensor::from_vec(rev, m, self.coef.device())?;
        let flip = |v: &Var, negate: bool| -> Result<()> {
            let all = v.as_tensor();
            let block = all.narrow(0, l * m, m)?.index_select(&rev, 0)?;
            let block = if negate { block.neg()? } else { block };
            let mut parts = Vec::new();
            if l > 0 {
                parts.push(all.narrow(0, 0, l * m)?);
            }
            parts.push(block);
            let rest = all.dim(0)? - (l + 1) * m;
            if rest > 0 {
                parts.push(all.narrow(0, (l + 1) * m, rest)?);
            }
            v.set(&Tensor::cat(&parts, 0)?)
        };
        flip(&self.coef, false)?;
        flip(&self.size, true)
    }

    /// Set lineage `l`'s curve coefficients (`[M × H]` row-major) and zero
    /// its size change.
    pub fn set_lineage_curve(&self, l: usize, curve: &[f32]) -> Result<()> {
        let (m, h) = (self.knots, self.coef.dim(1)?);
        let replace = |v: &Var, block: Tensor| -> Result<()> {
            let all = v.as_tensor();
            let mut parts = Vec::new();
            if l > 0 {
                parts.push(all.narrow(0, 0, l * m)?);
            }
            parts.push(block);
            let rest = all.dim(0)? - (l + 1) * m;
            if rest > 0 {
                parts.push(all.narrow(0, (l + 1) * m, rest)?);
            }
            v.set(&Tensor::cat(&parts, 0)?)
        };
        replace(
            &self.coef,
            Tensor::from_slice(curve, (m, h), self.coef.device())?,
        )?;
        replace(
            &self.size,
            Tensor::zeros((m, 1), DType::F32, self.size.device())?,
        )
    }

    /// `[G × H]` the gene embedding.
    pub fn rho(&self) -> &Tensor {
        self.rho.as_tensor()
    }

    /// `[1 × G]` the spliced baselines.
    pub fn base(&self) -> &Tensor {
        self.base.as_tensor()
    }

    /// `[1 × G]` the splicing rates `β`.
    pub fn beta(&self) -> Result<Tensor> {
        bounded(self.beta.as_tensor(), LOG_RATE)?.exp()
    }

    /// `[L × 1]` every lineage's speed, `exp` of its centred log speed.
    pub fn speed(&self) -> Result<Tensor> {
        let ls = self.speed.as_tensor();
        ls.broadcast_sub(&ls.mean_keepdim(0)?)?.exp()
    }

    /// The states' loadings.
    pub fn load_var(&self) -> Var {
        self.load.clone()
    }

    /// `[1 × G]` the degradation rates `γ`.
    pub fn gamma(&self) -> Result<Tensor> {
        bounded(self.gamma.as_tensor(), LOG_RATE)?.exp()
    }

    /// The tables on the grid of `spline`, every lineage weighted by
    /// `log_weight` (`[L]`); with the states switched off (their weights zero)
    /// unless `states`.
    pub fn tables(&self, spline: &Spline, states: bool, log_weight: &[f32]) -> Result<Tables> {
        let (l, m) = (self.lineages, self.knots);
        let k = spline.values.dim(0)?;
        let rho = self.rho.as_tensor();
        let (beta, gamma) = (self.beta()?, self.gamma()?);
        let speed = self.speed()?;
        let (mut psi, mut curves, mut velocities, mut sizes) = (vec![], vec![], vec![], vec![]);
        for i in 0..l {
            let coef = self.coef.narrow(0, i * m, m)?;
            let curve = spline.values.matmul(&coef)?; // [K × H]
            let velocity = spline.slopes.matmul(&coef)?; // [K × H]
            let size = spline.values.matmul(&self.size.narrow(0, i * m, m)?)?; // [K × 1]
                                                                               // In the shared rates' time: the lineage's speed per unit `τ`.
            let pace = (speed.narrow(0, i, 1)? * self.motion.get())?; // [1 × 1]
            let raw = time_tables(
                rho,
                &curve,
                &velocity.broadcast_mul(&pace)?,
                &size.broadcast_mul(&pace)?,
                self.base.as_tensor(),
                &beta,
                &gamma,
            )?;
            psi.push(log_softmax(&raw, 1)?);
            curves.push(curve);
            velocities.push(velocity);
            sizes.push(size);
        }
        let bias: Vec<f32> = log_weight
            .iter()
            .flat_map(|&w| std::iter::repeat_n(w - (k as f32).ln(), k))
            .collect();
        let load = if states {
            without_span(&self.load, &self.coef.detach(), log_weight)?
        } else {
            self.load.zeros_like()?
        };
        let half = rho.matmul(&load)?; // [G × d]
        Ok(Tables {
            psi: Tensor::cat(&psi, 0)?,
            bias: Tensor::from_vec(bias, (1, l * k), rho.device())?,
            w: Tensor::cat(&[&half, &half], 0)?,
            load,
            curve: Tensor::cat(&curves, 0)?,
            velocity: Tensor::cat(&velocities, 0)?,
            size: Tensor::cat(&sizes, 0)?,
            lineages: l,
        })
    }
}

impl Energy {
    /// Lineage `l`'s velocity read out per gene on the grid, `[K × G]`:
    /// `x_kg = s_ℓ (⟨ρ_g − ρ̄_k, ż(τ_k)⟩ + ν(τ_k))`, i.e. `d log s_g/dt` in
    /// the shared rates' time, which sets the unspliced excess. Its sign is
    /// the direction of time.
    pub fn velocity_readout(&self, spline: &Spline, l: usize) -> Result<Tensor> {
        let m = self.knots;
        let rho = self.rho.as_tensor();
        let coef = self.coef.narrow(0, l * m, m)?;
        let curve = spline.values.matmul(&coef)?;
        let velocity = spline.slopes.matmul(&coef)?;
        let size = spline.values.matmul(&self.size.narrow(0, l * m, m)?)?;
        let spliced = curve
            .matmul(&rho.t()?)?
            .broadcast_add(self.base.as_tensor())?;
        let rho_bar = log_softmax(&spliced, 1)?.exp()?.matmul(rho)?;
        let along = velocity.matmul(&rho.t()?)?;
        let centre = (&rho_bar * &velocity)?.sum_keepdim(1)?;
        let pace = self.speed()?.narrow(0, l, 1)?;
        along
            .broadcast_sub(&centre)?
            .broadcast_add(&size)?
            .broadcast_mul(&pace)
    }
}

/// `load` `[H × d]` with the span of the live lineages' curve travel
/// removed: every live lineage's (`log_weight` finite) coefficient rows
/// (`coef` `[L·M × H]`) about their mean, as a B-spline curve moves only
/// within them (its basis sums to one). The span is taken by its principal
/// directions on the host, those above a relative floor, and at most `H − d`
/// of them, so the states always keep `d` directions: when the curves span
/// more, the least travelled are left to the states.
pub fn without_span(load: &Tensor, coef: &Tensor, log_weight: &[f32]) -> Result<Tensor> {
    let (h, d) = load.dims2()?;
    let m = coef.dim(0)? / log_weight.len().max(1);
    let mut gram = nalgebra::DMatrix::<f64>::zeros(h, h);
    for (i, w) in log_weight.iter().enumerate() {
        if !w.is_finite() {
            continue;
        }
        let c = coef.narrow(0, i * m, m)?;
        for r in c.broadcast_sub(&c.mean_keepdim(0)?)?.to_vec2::<f32>()? {
            let v = nalgebra::DVector::from_iterator(h, r.iter().map(|&x| f64::from(x)));
            gram += &v * v.transpose();
        }
    }
    let eig = nalgebra::SymmetricEigen::new(gram);
    let mut order: Vec<usize> = (0..h).collect();
    order.sort_by(|&a, &b| eig.eigenvalues[b].total_cmp(&eig.eigenvalues[a]));
    let top = eig.eigenvalues.iter().cloned().fold(0f64, f64::max);
    let kept: Vec<usize> = order
        .into_iter()
        .take(h.saturating_sub(d))
        .filter(|&j| top > 0.0 && eig.eigenvalues[j] > 1e-6 * top)
        .collect();
    if kept.is_empty() {
        return Ok(load.clone());
    }
    let p: Vec<f32> = (0..h)
        .flat_map(|i| {
            kept.iter()
                .map(|&j| eig.eigenvectors[(i, j)] as f32)
                .collect::<Vec<_>>()
        })
        .collect();
    let p = Tensor::from_vec(p, (h, kept.len()), load.device())?; // [H × r]
    load - p.matmul(&p.t()?.matmul(load)?)?
}

/// `[K × 2G]` every grid time's weight on every feature (unspliced genes
/// first): genes embedded at `rho` `[G × H]`, a curve and its velocity
/// `[K × H]`, the size change `[K × 1]`, baselines and rates `[1 × G]`.
pub fn time_tables(
    rho: &Tensor,
    curve: &Tensor,
    velocity: &Tensor,
    size: &Tensor,
    base: &Tensor,
    beta: &Tensor,
    gamma: &Tensor,
) -> Result<Tensor> {
    let spliced = curve.matmul(&rho.t()?)?.broadcast_add(base)?; // [K × G]
    let rho_bar = log_softmax(&spliced, 1)?.exp()?.matmul(rho)?; // [K × H]
    let along = velocity.matmul(&rho.t()?)?; // [K × G]
    let centre = (&rho_bar * velocity)?.sum_keepdim(1)?; // [K × 1]
    let m = along
        .broadcast_sub(&centre)?
        .broadcast_add(size)?
        .broadcast_add(gamma)?;
    let soft = (gamma * FLOOR_SOFTNESS)?;
    let m_plus = softplus(&m.broadcast_div(&soft)?)?.broadcast_mul(&soft)?;
    let eta = m_plus.maximum(TINY)?.log()?.broadcast_sub(&beta.log()?)?;
    Tensor::cat(&[&(&spliced + eta)?, &spliced], 1)
}

#[cfg(test)]
#[path = "tests/model.rs"]
mod tests;
