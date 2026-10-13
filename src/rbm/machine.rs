//! The machine: an exponential-family harmonium (Welling, Rosen-Zvi and
//! Hinton, *NIPS* (2004)) with count visible units in modality blocks and
//! hidden units of one [`Hidden`] kind. Every block `m` is a multinomial over
//! its own features at its own depth `N_m`, read per unit depth against a
//! fixed depth `D_m` (as in the replicated softmax, Salakhutdinov and Hinton,
//! *NIPS* (2009), so a deep row does not saturate):
//!
//! ```text
//! E(v, h) = −Σ_f v_f b_f − Σ_f s_f v_f (W h)_f − c·h − Σ_k log μ(h_k),   s_f = D_m / N_m  (f ∈ m)
//! a(v)    = c + (s ⊙ v) W
//! F(v)    = −v·b − Σ_k A(a_k(v))
//! p(v | h) = Π_m Multinomial(N_m, softmax_{f∈m}(b_f + s_f (W h)_f))
//! ```
//!
//! The energy is centred (Montavon and Müller, *Neural Networks: Tricks of
//! the Trade* (2012)): the scaled reads by their mean `μ` and the hidden units
//! by theirs, `λ`, as `E = −v·b − (s⊙v − μ)ᵀ W (h − λ) − c·h − Σ log μ(h)`. A
//! reparametrisation of the same machines, it keeps a unit that is off for
//! some cells learning from them; `μ = 0, λ = 0` is the plain energy above.
//!
//! Given the reads, the hidden units are independent with natural
//! parameters `a(v)`. A block a row has no reads in (`N_m = 0`) drops out of
//! `a(v)` and `F(v)` altogether: given `h` the blocks are independent, so this
//! is the exact marginal over the missing block.

use super::hidden::Hidden;
use crate::rbm_core::cumsum_rows;
use legume_numeric::candle::candle_core::{Device, Result, Tensor, D};
use legume_numeric::candle::candle_nn::ops::{log_softmax, softmax};
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};
use std::ops::Range;

/// The machine's parameters as tensors: biases `b` `[1 × F]`, `c` `[1 × K]`
/// and weights `W` `[F × K]`, with the blocks and their depths.
#[derive(Clone)]
pub struct Machine {
    pub b: Tensor,
    pub c: Tensor,
    pub w: Tensor,
    pub hidden: Hidden,
    pub ranges: Vec<Range<usize>>,
    /// `D_m` per block.
    pub depth: Vec<f32>,
    /// `[F × M]` every feature's block, one-hot.
    onehot: Tensor,
    /// `[1 × F]` the centre of the scaled reads.
    pub mu: Tensor,
    /// `[1 × K]` the centre of the hidden units.
    pub lambda: Tensor,
}

impl Machine {
    pub fn new(
        b: Tensor,
        c: Tensor,
        w: Tensor,
        hidden: Hidden,
        ranges: Vec<Range<usize>>,
        depth: Vec<f32>,
    ) -> Result<Self> {
        let onehot = block_onehot(&ranges, w.device())?;
        let mu = b.zeros_like()?;
        let lambda = c.zeros_like()?;
        Ok(Self {
            b,
            c,
            w,
            hidden,
            ranges,
            depth,
            onehot,
            mu,
            lambda,
        })
    }

    /// The same machine centred at `mu` `[1 × F]` and `lambda` `[1 × K]`.
    pub fn centred(self, mu: Tensor, lambda: Tensor) -> Self {
        Self { mu, lambda, ..self }
    }

    /// The same machine at other parameters.
    pub fn with(&self, b: Tensor, c: Tensor, w: Tensor) -> Self {
        Self {
            b,
            c,
            w,
            ..self.clone()
        }
    }

    pub fn detach(&self) -> Self {
        Self {
            b: self.b.detach(),
            c: self.c.detach(),
            w: self.w.detach(),
            mu: self.mu.detach(),
            lambda: self.lambda.detach(),
            ..self.clone()
        }
    }

    pub fn features(&self) -> Result<usize> {
        self.w.dim(0)
    }

    pub fn units(&self) -> Result<usize> {
        self.w.dim(1)
    }

    /// Every row's per-read scale `s` `[N × F]`: `D_m / N_m` on block `m`'s
    /// features (`0` where the row has no reads in the block).
    pub fn scale(&self, x: &Tensor) -> Result<Tensor> {
        let n = x.matmul(&self.onehot)?;
        let d = Tensor::from_slice(&self.depth, (1, self.depth.len()), x.device())?;
        let empty = n.le(0.0)?;
        let s = d.broadcast_div(&n.maximum(1.0)?)?;
        let s = empty.where_cond(&s.zeros_like()?, &s)?;
        s.matmul(&self.onehot.t()?)
    }

    /// `x` with every block scaled to the depth it has in `like`.
    pub fn keep_depth(&self, x: &Tensor, like: &Tensor) -> Result<Tensor> {
        let want = like.matmul(&self.onehot)?;
        let have = x.matmul(&self.onehot)?.maximum(1e-8)?;
        x * (want / have)?.matmul(&self.onehot.t()?)?
    }

    /// `s ⊙ v − μ` on the blocks a row has reads in, `0` on the rest.
    fn centred_reads(&self, x: &Tensor) -> Result<Tensor> {
        let s = self.scale(x)?;
        let present = s.gt(0.0)?.to_dtype(x.dtype())?;
        (x * &s)? - present.broadcast_mul(&self.mu)?
    }

    /// The hidden units' natural parameters `a(v)` `[N × K]`.
    pub fn input(&self, x: &Tensor) -> Result<Tensor> {
        self.centred_reads(x)?
            .matmul(&self.w)?
            .broadcast_add(&self.c)
    }

    /// Every row's free energy `[N]`.
    pub fn free_energy(&self, x: &Tensor) -> Result<Tensor> {
        let r = self.centred_reads(x)?.matmul(&self.w)?;
        let a = r.broadcast_add(&self.c)?;
        let vb = x.broadcast_mul(&self.b)?.sum(D::Minus1)?;
        let centre = r.broadcast_mul(&self.lambda)?.sum(D::Minus1)?;
        ((vb - centre)? + self.hidden.log_partition(&a)?.sum(D::Minus1)?)?.neg()
    }

    /// Every row's free energy divided by its reads `[N]`, comparable across
    /// rows of any depth or block coverage.
    pub fn free_energy_per_read(&self, x: &Tensor) -> Result<Tensor> {
        self.free_energy(x)? / x.sum(D::Minus1)?.maximum(1.0)?
    }

    /// `E[h | v]` `[N × K]`.
    pub fn posterior_mean(&self, x: &Tensor) -> Result<Tensor> {
        self.hidden.mean(&self.input(x)?)
    }

    /// The reads' logits given the hidden units `h` `[N × K]` at the per-read
    /// scales `s` `[N × F]`; a softmax within every block gives its shares.
    pub fn visible_logits(&self, h: &Tensor, s: &Tensor) -> Result<Tensor> {
        (h.broadcast_sub(&self.lambda)?.matmul(&self.w.t()?)? * s)?.broadcast_add(&self.b)
    }

    /// Every row's reconstruction: the multinomial log-likelihood of its reads
    /// (up to the coefficient) at `E[h | v]`, per read `[N]`.
    pub fn reconstruction(&self, x: &Tensor) -> Result<Tensor> {
        let s = self.scale(x)?;
        let logits = self.visible_logits(&self.posterior_mean(x)?, &s)?;
        let mut total = x.zeros_like()?.sum(D::Minus1)?;
        for r in &self.ranges {
            let lp = log_softmax(&logits.narrow(1, r.start, r.len())?, D::Minus1)?;
            let xr = x.narrow(1, r.start, r.len())?;
            total = (total + (xr * lp)?.sum(D::Minus1)?)?;
        }
        total / x.sum(D::Minus1)?.maximum(1.0)?
    }
}

/// `[F × M]` one-hot block membership.
fn block_onehot(ranges: &[Range<usize>], dev: &Device) -> Result<Tensor> {
    let f = ranges.iter().map(|r| r.end).max().unwrap_or(0);
    let m = ranges.len();
    let mut v = vec![0f32; f * m];
    for (b, r) in ranges.iter().enumerate() {
        for i in r.clone() {
            v[i * m + b] = 1.0;
        }
    }
    Tensor::from_vec(v, (f, m), dev)
}

/// Entries of the `[S × c × F]` comparison one chunk of draws may hold.
const COMPARE_BUDGET: usize = 1 << 26;

/// `n_s` draws per row `s` from the categorical whose cumulative shares are
/// `cum` `[S × F]`, as counts `[S × F]`: a feature's count is how many of the
/// row's uniforms fall in its stretch of the cumulative. Only the `S × max n`
/// uniforms come from `rng`; the comparisons run on `cum`'s device.
pub fn multinomial_rows(cum: &Tensor, n: &[u32], rng: &mut StdRng) -> Result<Tensor> {
    let dev = cum.device();
    let (s, f) = cum.dims2()?;
    let top = n.iter().copied().max().unwrap_or(0) as usize;
    if top == 0 || f == 0 {
        return cum.zeros_like();
    }
    // A row with fewer draws pads with uniforms past every share.
    let u: Vec<f32> = n
        .iter()
        .flat_map(|&ns| (0..top).map(move |d| (d, ns)))
        .map(|(d, ns)| {
            if d < ns as usize {
                rng.random::<f32>()
            } else {
                2.0
            }
        })
        .collect();
    let u = Tensor::from_vec(u, (s, top), dev)?;
    let ns = Tensor::from_iter(n.iter().map(|&x| x as f32), dev)?.unsqueeze(1)?;
    if f == 1 {
        return Ok(ns);
    }
    // `below[s, j]`: the row's draws under the cumulative share of feature `j`,
    // compared a chunk of draws at a time to bound the `[S × c × F]` working set.
    let head = cum.narrow(1, 0, f - 1)?;
    let chunk = (COMPARE_BUDGET / (s * f).max(1)).clamp(1, top);
    let head3 = head.unsqueeze(1)?;
    let mut below = head.zeros_like()?;
    for d in (0..top).step_by(chunk) {
        let c = chunk.min(top - d);
        let hits = u
            .narrow(1, d, c)?
            .unsqueeze(2)?
            .broadcast_lt(&head3)?
            .to_dtype(cum.dtype())?
            .sum(1)?;
        below = (below + hits)?;
    }
    let first = below.narrow(1, 0, 1)?;
    let middle = (below.narrow(1, 1, f - 2)? - below.narrow(1, 0, f - 2)?)?;
    // The last feature takes what the rest left, so rounding never loses a draw.
    let last = (ns - below.narrow(1, f - 2, 1)?)?;
    Tensor::cat(&[first, middle, last], 1)
}

/// Persistent Gibbs chains on the machine's device: each holds its reads and
/// keeps every block's depth.
pub struct Chains {
    x: Tensor,
    /// Per block, every chain's depth.
    depth: Vec<Vec<u32>>,
    rng: StdRng,
}

impl Chains {
    /// Chains started at the rows `x` `[S × F]` (row-major, whole counts).
    pub fn new(
        x: Vec<f32>,
        features: usize,
        ranges: &[Range<usize>],
        seed: u64,
        dev: &Device,
    ) -> Result<Self> {
        let depth = ranges
            .iter()
            .map(|r| {
                x.chunks(features)
                    .map(|row| row[r.clone()].iter().sum::<f32>().round() as u32)
                    .collect()
            })
            .collect();
        let s = x.len() / features.max(1);
        Ok(Self {
            x: Tensor::from_vec(x, (s, features), dev)?,
            depth,
            rng: StdRng::seed_from_u64(seed),
        })
    }

    /// The chains' reads `[S × F]`.
    pub fn reads(&self) -> &Tensor {
        &self.x
    }

    /// One block Gibbs sweep at `m`: the hidden units given the reads, then
    /// every block's reads given the hidden units, all on the device.
    pub fn sweep(&mut self, m: &Machine) -> Result<()> {
        let s = m.scale(&self.x)?;
        let h = m.hidden.sample(&m.input(&self.x)?, &mut self.rng)?;
        let logits = m.visible_logits(&h, &s)?;
        let parts = m
            .ranges
            .iter()
            .zip(&self.depth)
            .map(|(r, n)| {
                let p = softmax(&logits.narrow(1, r.start, r.len())?, D::Minus1)?;
                multinomial_rows(&cumsum_rows(&p)?, n, &mut self.rng)
            })
            .collect::<Result<Vec<_>>>()?;
        self.x = Tensor::cat(&parts, 1)?;
        Ok(())
    }
}

#[cfg(test)]
#[path = "tests/machine.rs"]
mod tests;
