//! The hidden units' kinds. Each kind is an exponential family on its own
//! support with base measure `μ(h)`, read through the natural parameter `a`:
//! `p(h | a) ∝ μ(h) e^{a h}`, so its log-partition `A(a)` is all the free
//! energy needs and its derivative the posterior mean.
//!
//! | kind | support | μ(h) | A(a) |
//! |---|---|---|---|
//! | Bernoulli | {0, 1} | 1 | softplus(a) |
//! | Binomial(n) | {0..n} | C(n, h) | n softplus(a) |
//! | Poisson ≤ M | {0..M} | 1/h! | log Σ_{h≤M} e^{a h}/h! |
//!
//! The Poisson unit is truncated at `M` so the partition stays finite for any `a`.

use crate::rbm_core::softplus;
use legume_numeric::candle::candle_core::{Result, Tensor, D};
use legume_numeric::candle::candle_nn::ops::{sigmoid, softmax};
use rand::rngs::StdRng;
use rand::RngExt;

#[derive(
    Clone, Copy, Debug, PartialEq, Eq, clap::ValueEnum, serde::Serialize, serde::Deserialize,
)]
#[serde(rename_all = "kebab-case")]
pub enum HiddenArg {
    Bernoulli,
    Binomial,
    Poisson,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Hidden {
    Bernoulli,
    Binomial { trials: u32 },
    Poisson { max: u32 },
}

impl Hidden {
    pub fn new(kind: HiddenArg, trials: u32, max_count: u32) -> Self {
        match kind {
            HiddenArg::Bernoulli => Self::Bernoulli,
            HiddenArg::Binomial => Self::Binomial { trials },
            HiddenArg::Poisson => Self::Poisson { max: max_count },
        }
    }

    /// `A(a)` elementwise, differentiable.
    pub fn log_partition(&self, a: &Tensor) -> Result<Tensor> {
        match *self {
            Self::Bernoulli => softplus(a),
            Self::Binomial { trials } => softplus(a)? * f64::from(trials),
            Self::Poisson { max } => poisson_logits(a, max)?.1.log_sum_exp(D::Minus1),
        }
    }

    /// `E[h | a] = A'(a)` elementwise.
    pub fn mean(&self, a: &Tensor) -> Result<Tensor> {
        match *self {
            Self::Bernoulli => sigmoid(a),
            Self::Binomial { trials } => sigmoid(a)? * f64::from(trials),
            Self::Poisson { max } => {
                let (h, logits) = poisson_logits(a, max)?;
                softmax(&logits, D::Minus1)?
                    .broadcast_mul(&h)?
                    .sum(D::Minus1)
            }
        }
    }

    /// The scale that puts a unit's mean on `0..1`.
    pub fn unit_scale(&self) -> f64 {
        match *self {
            Self::Bernoulli => 1.0,
            Self::Binomial { trials } => f64::from(trials.max(1)),
            Self::Poisson { max } => f64::from(max.max(1)),
        }
    }

    /// One draw of `h | a` `[N × K]` on `a`'s device. The few uniforms a draw
    /// needs come from `rng`, so a sweep replays from its seed; the rest is
    /// elementwise work on the device.
    pub fn sample(&self, a: &Tensor, rng: &mut StdRng) -> Result<Tensor> {
        let dev = a.device();
        let shape = a.dims().to_vec();
        let n = a.elem_count();
        let uniform = |count: usize, rng: &mut StdRng| -> Vec<f32> {
            (0..count).map(|_| rng.random::<f32>()).collect()
        };
        match *self {
            Self::Bernoulli => {
                let u = Tensor::from_vec(uniform(n, rng), shape, dev)?;
                u.lt(&sigmoid(a)?)?.to_dtype(a.dtype())
            }
            Self::Binomial { trials } => {
                let p = sigmoid(a)?;
                let mut h = a.zeros_like()?;
                for _ in 0..trials {
                    let u = Tensor::from_vec(uniform(n, rng), shape.clone(), dev)?;
                    h = (h + u.lt(&p)?.to_dtype(a.dtype())?)?;
                }
                Ok(h)
            }
            Self::Poisson { max } => {
                // `h` is the number of support points whose cumulative mass
                // the uniform lies at or above.
                let logits = poisson_logits(a, max)?.1;
                let cum = softmax(&logits, D::Minus1)?.cumsum(D::Minus1)?;
                let cum = cum.narrow(D::Minus1, 0, max as usize)?;
                let mut ushape = shape.clone();
                ushape.push(1);
                let u = Tensor::from_vec(uniform(n, rng), ushape, dev)?;
                u.broadcast_ge(&cum)?.to_dtype(a.dtype())?.sum(D::Minus1)
            }
        }
    }
}

/// The truncated Poisson's support `h = 0..=max` `[max + 1]` and every
/// support point's log weight `a h − log h!` `[N × K × (max + 1)]`.
fn poisson_logits(a: &Tensor, max: u32) -> Result<(Tensor, Tensor)> {
    let dev = a.device();
    let h: Vec<f32> = (0..=max).map(|h| h as f32).collect();
    let mut lf = 0f32;
    let base: Vec<f32> = (0..=max)
        .map(|h| {
            if h > 0 {
                lf += (h as f32).ln();
            }
            -lf
        })
        .collect();
    let h = Tensor::from_vec(h, max as usize + 1, dev)?;
    let base = Tensor::from_vec(base, max as usize + 1, dev)?;
    let logits = a
        .unsqueeze(D::Minus1)?
        .broadcast_mul(&h)?
        .broadcast_add(&base)?;
    Ok((h, logits))
}
