//! The machine: a row's reads over the `F = 2G` features are the visible
//! units; a categorical unit over (lineage `ℓ`, grid time `k`) and a Gaussian
//! state `h ∈ ℝ^d` the hidden ones, with energy
//!
//! ```text
//! E(v, ℓ, k, h) = −Σ_f v_f ψ_{ℓ,k}(f) − log π_ℓ − (1/N) Σ_f v_f (W h)_f + ½‖h‖²,    N = Σ_f v_f
//! ```
//!
//! — an exponential-family harmonium (Welling, Rosen-Zvi and Hinton, *NIPS*
//! (2004)) with count visible units, a softmax hidden unit for (lineage,
//! time) and Gaussian hidden units for the state. Every `ψ_{ℓ,k}` is a log
//! probability over the features ([`super::model`]), so the time prior is
//! flat within a lineage at any depth; the state reads the reads per unit
//! depth, so it cannot saturate on a deep row. Given the reads, (lineage,
//! time) and state are exact and independent:
//! `p(ℓ, k | v) = softmax(v·ψ_{ℓ,k} + log π_ℓ)`, `h | v ~ N(Wᵀv/N, I)`; given
//! the hidden units the reads are `Multinomial(N, softmax(ψ_{ℓ,k} + W h/N))`.
//! The free energy `F(v) = −log Σ_{ℓ,k} (π_ℓ/K) e^{v·ψ_{ℓ,k}} − ½‖Wᵀv/N‖²` (up
//! to a constant) drives the state fit ([`free_energy`]); [`Chains`] are the
//! persistent Gibbs chains of its negative phase. With the states off the
//! machine is a mixture of multinomials with an exact likelihood
//! ([`lineage_log_likelihood`]).

use super::model::Tables;
pub(crate) use crate::rbm_core::multinomial;
pub use crate::rbm_core::{cumsum_rows, multinomial_rows};
use legume_numeric::candle::candle_core::{Device, Result, Tensor, D};
use legume_numeric::candle::candle_nn::ops::softmax;
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};
use rand_distr::{Distribution, Normal};

/// Rows of reads: `[N × F]` features (unspliced genes first) and `[N × 1]`
/// depths.
pub struct Reads {
    pub x: Tensor,
    pub depth: Tensor,
}

impl Reads {
    /// Rows already joined, `[N × F]`.
    pub fn from_features(x: Tensor) -> Result<Self> {
        let depth = x.sum_keepdim(1)?;
        Ok(Self { x, depth })
    }

    /// The rows `ids` (u32 `[n]`).
    pub fn select(&self, ids: &Tensor) -> Result<Self> {
        Ok(Self {
            x: self.x.index_select(ids, 0)?,
            depth: self.depth.index_select(ids, 0)?,
        })
    }
}

/// The hidden units' input from the reads: `[N × K]` time logits and the
/// state's posterior mean `[N × d]`.
pub fn hidden_input(r: &Reads, t: &Tables) -> Result<(Tensor, Tensor)> {
    let time = r.x.matmul(&t.psi.t()?)?.broadcast_add(&t.bias)?;
    let state = r.x.matmul(&t.w)?.broadcast_div(&r.depth.maximum(1.0)?)?;
    Ok((time, state))
}

/// Every row's free energy `[N]`.
pub fn free_energy(r: &Reads, t: &Tables) -> Result<Tensor> {
    let (time, state) = hidden_input(r, t)?;
    (time.log_sum_exp(1)? + (state.sqr()?.sum(1)? * 0.5)?)?.neg()
}

/// Every row's exact log-likelihood under every lineage with the states
/// off, up to the multinomial coefficient: `log (1/K) Σ_k e^{v·ψ_{ℓ,k}}`,
/// `[N × L]`. Every `ψ_{ℓ,k}` is a log probability over the features, so this
/// is a mixture of multinomials over the grid: no partition function.
pub fn lineage_log_likelihood(r: &Reads, t: &Tables) -> Result<Tensor> {
    let (l, k) = (t.lineages, t.grid_len()?);
    let n = r.x.dim(0)?;
    r.x.matmul(&t.psi.t()?)?
        .reshape((n, l, k))?
        .log_sum_exp(2)?
        - (k as f64).ln()
}

/// Every row's exact posterior: (lineage, time) `[N × L·K]` (rows `ℓ·K + k`)
/// and the state's mean `[N × d]` (its covariance is the identity).
pub fn posterior(r: &Reads, t: &Tables) -> Result<(Tensor, Tensor)> {
    let (time, state) = hidden_input(r, t)?;
    Ok((softmax(&time, D::Minus1)?, state))
}

/// Persistent Gibbs chains over the reads: `S` chains, each holding its
/// reads (on the machine's device) and keeping its depth. A sweep draws on
/// the host on a CPU, where successive binomials are cheapest; on a GPU it
/// stays on the device, only the uniforms and normals drawn on the host
/// (from the chains' own seed, so a run replays).
pub struct Chains {
    x: Tensor,
    depth: Vec<u32>,
    rng: StdRng,
    on_device: bool,
}

impl Chains {
    /// Chains started at the rows `x` `[S × F]` (row-major, whole counts)
    /// on `dev`.
    pub fn new(x: Vec<f32>, features: usize, dev: &Device, seed: u64) -> Result<Self> {
        let depth = x
            .chunks(features)
            .map(|row| row.iter().sum::<f32>().round() as u32)
            .collect::<Vec<_>>();
        Ok(Self {
            x: Tensor::from_vec(x, (depth.len(), features), dev)?,
            depth,
            rng: StdRng::seed_from_u64(seed),
            on_device: !dev.is_cpu(),
        })
    }

    /// Sweep on the device whatever it is (the GPU path, testable on a CPU).
    #[cfg(test)]
    pub fn on_device(mut self) -> Self {
        self.on_device = true;
        self
    }

    /// The chains' reads.
    pub fn reads(&self) -> Result<Reads> {
        Reads::from_features(self.x.clone())
    }

    /// One block Gibbs sweep at the tables `t`: hidden units given the
    /// reads, then reads given the hidden units.
    pub fn sweep(&mut self, t: &Tables) -> Result<()> {
        if self.on_device {
            self.sweep_on_device(t)
        } else {
            self.sweep_on_host(t)
        }
    }

    fn sweep_on_host(&mut self, t: &Tables) -> Result<()> {
        let dev = self.x.device().clone();
        let (p_time, mean) = posterior(&self.reads()?, t)?;
        let (p_time, mean) = (p_time.to_vec2::<f32>()?, mean.to_vec2::<f32>()?);
        let (s, f) = self.x.dims2()?;
        let d = t.w.dim(1)?;
        let normal = Normal::new(0.0f32, 1.0).expect("a valid sd");
        let mut k = Vec::with_capacity(s);
        let mut h = Vec::with_capacity(s * d);
        for j in 0..s {
            k.push(categorical(&p_time[j], &mut self.rng) as u32);
            // The reads see the state per read: W h / N.
            let n = self.depth[j].max(1) as f32;
            h.extend(
                mean[j]
                    .iter()
                    .map(|&m| (m + normal.sample(&mut self.rng)) / n),
            );
        }
        let k = Tensor::from_vec(k, s, &dev)?;
        let h = Tensor::from_vec(h, (s, d), &dev)?;
        let logits = (t.psi.index_select(&k, 0)? + h.matmul(&t.w.t()?)?)?;
        let probs = softmax(&logits, D::Minus1)?.to_vec2::<f32>()?;
        let mut x = vec![0f32; s * f];
        for (j, p) in probs.iter().enumerate() {
            multinomial(
                u64::from(self.depth[j]),
                p,
                &mut x[j * f..(j + 1) * f],
                &mut self.rng,
            );
        }
        self.x = Tensor::from_vec(x, (s, f), &dev)?;
        Ok(())
    }

    fn sweep_on_device(&mut self, t: &Tables) -> Result<()> {
        let dev = self.x.device().clone();
        let (p_time, mean) = posterior(&self.reads()?, t)?;
        let (s, d) = mean.dims2()?;
        // (lineage, time) one-hot by one draw each; its tables by a matmul.
        let hot = multinomial_rows(&cumsum_rows(&p_time)?, &vec![1; s], &mut self.rng)?;
        let psi = hot.matmul(&t.psi)?;
        // The state, read per read: (mean + noise) / N.
        let normal = Normal::new(0.0f32, 1.0).expect("a valid sd");
        let noise: Vec<f32> = (0..s * d).map(|_| normal.sample(&mut self.rng)).collect();
        let n =
            Tensor::from_iter(self.depth.iter().map(|&n| n.max(1) as f32), &dev)?.unsqueeze(1)?;
        let h = (mean + Tensor::from_vec(noise, (s, d), &dev)?)?.broadcast_div(&n)?;
        let logits = (psi + h.matmul(&t.w.t()?)?)?;
        let probs = softmax(&logits, D::Minus1)?;
        self.x = multinomial_rows(&cumsum_rows(&probs)?, &self.depth, &mut self.rng)?;
        Ok(())
    }
}

fn unit(rng: &mut StdRng) -> f32 {
    rng.random::<f32>()
}

/// An index drawn with probabilities `p`.
pub(crate) fn categorical(p: &[f32], rng: &mut StdRng) -> usize {
    let mut u = unit(rng) * p.iter().sum::<f32>();
    for (i, &w) in p.iter().enumerate() {
        if u < w {
            return i;
        }
        u -= w;
    }
    p.len() - 1
}

#[cfg(test)]
#[path = "tests/rbm.rs"]
mod tests;
