//! Pieces every restricted Boltzmann machine here shares: Adam, multinomial draws
//! on the host and on the device, a device cumulative sum that fits at any
//! feature count, and a stable softplus.

use legume_numeric::candle::candle_core::{Result, Tensor, Var};
use legume_numeric::candle::candle_nn::{AdamW, Optimizer, ParamsAdamW};
use rand::rngs::StdRng;
use rand::RngExt;
use rand_distr::{Binomial, Distribution};

/// Adam (AdamW with no weight decay) over `vars` at learning rate `lr`.
pub fn adam(vars: Vec<Var>, lr: f64) -> Result<AdamW> {
    AdamW::new(
        vars,
        ParamsAdamW {
            lr,
            weight_decay: 0.0,
            ..Default::default()
        },
    )
}

/// `log(1 + eˣ)`, stable for any `x`.
pub fn softplus(x: &Tensor) -> Result<Tensor> {
    x.relu()? + x.abs()?.neg()?.exp()?.affine(1.0, 1.0)?.log()?
}

/// The cumulative sum of `t` `[S × F]` along its rows, in blocks of about
/// `√F`: a triangular matmul within every block, then every block's offset.
/// candle's own `cumsum` multiplies by an `F × F` triangle, which at tens of
/// thousands of features does not fit on a device.
pub fn cumsum_rows(t: &Tensor) -> Result<Tensor> {
    let (s, f) = t.dims2()?;
    if f <= 1 {
        return Ok(t.clone());
    }
    let b = ((f as f64).sqrt().ceil() as usize).max(1);
    let nb = f.div_ceil(b);
    let padded = if nb * b > f {
        Tensor::cat(
            &[
                t.clone(),
                Tensor::zeros((s, nb * b - f), t.dtype(), t.device())?,
            ],
            1,
        )?
    } else {
        t.clone()
    };
    let within = padded
        .contiguous()?
        .reshape((s * nb, b))?
        .matmul(&Tensor::triu2(b, t.dtype(), t.device())?)?
        .reshape((s, nb, b))?;
    let totals = within.narrow(2, b - 1, 1)?.squeeze(2)?.contiguous()?;
    // Every block's offset: the totals of the blocks before it.
    let before =
        (Tensor::triu2(nb, t.dtype(), t.device())? - Tensor::eye(nb, t.dtype(), t.device())?)?;
    let offset = totals.matmul(&before)?;
    within
        .broadcast_add(&offset.unsqueeze(2)?)?
        .reshape((s, nb * b))?
        .narrow(1, 0, f)
}

/// Elements per comparison block in [`multinomial_rows`].
const DRAW_BLOCK: usize = 1 << 24;

/// `n_s` draws per row `s` from the categorical whose running shares are
/// `cum` `[S × F]`, as counts `[S × F]`: a feature's count is how many of the
/// row's uniforms fall in its stretch of the running shares. Only the
/// `S × max n` uniforms come from `rng`; the comparisons run on `cum`'s
/// device, a block of draws at a time. The last feature takes what the rest
/// left, so rounding never loses a draw.
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
    // `below[s, j]`: the row's draws under the running share of feature `j`.
    let head = cum.narrow(1, 0, f - 1)?.unsqueeze(1)?.contiguous()?; // [S × 1 × F−1]
    let chunk = (DRAW_BLOCK / (s * f).max(1)).clamp(1, top);
    let mut below = Tensor::zeros((s, f - 1), cum.dtype(), dev)?;
    for start in (0..top).step_by(chunk) {
        let c = chunk.min(top - start);
        let us = u.narrow(1, start, c)?.unsqueeze(2)?; // [S × c × 1]
        let hits = us.broadcast_lt(&head)?.to_dtype(cum.dtype())?.sum(1)?;
        below = (below + hits)?;
    }
    let first = below.narrow(1, 0, 1)?;
    let middle = (below.narrow(1, 1, f - 2)? - below.narrow(1, 0, f - 2)?)?;
    let last = (ns - below.narrow(1, f - 2, 1)?)?;
    Tensor::cat(&[first, middle, last], 1)
}

/// `n` draws over the categories of `p` (summing to one) into `out`, by
/// successive binomials.
pub fn multinomial(n: u64, p: &[f32], out: &mut [f32], rng: &mut StdRng) {
    let mut left = n;
    // The shares' own total, so rounding in them never pushes a draw onto a
    // category with no share.
    let mut mass: f64 = p.iter().map(|&x| f64::from(x)).sum();
    for (o, &pi) in out.iter_mut().zip(p) {
        if left == 0 {
            *o = 0.0;
            continue;
        }
        let share = (f64::from(pi) / mass.max(1e-300)).clamp(0.0, 1.0);
        let d = Binomial::new(left, share).map_or(0, |b| b.sample(rng));
        *o = d as f32;
        left -= d;
        mass -= f64::from(pi);
    }
    // Rounding can leave a draw over; it goes to the last category.
    if left > 0 {
        if let Some(last) = out.last_mut() {
            *last += left as f32;
        }
    }
}

#[cfg(test)]
#[path = "rbm_core/tests.rs"]
mod tests;
