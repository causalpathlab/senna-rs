//! Pieces every restricted Boltzmann machine here shares: a host multinomial
//! draw, a device cumulative sum that fits at any feature count, and a stable
//! softplus.

use legume_numeric::candle::candle_core::{Result, Tensor};
use rand::rngs::StdRng;
use rand_distr::{Binomial, Distribution};

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
