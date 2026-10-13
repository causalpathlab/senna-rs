//! The trajectory's basis: clamped cubic B-splines on `[0, 1]`, their values
//! and first derivatives at given times, as `[T × M]` tables.
//!
//! A curve `z(τ) = Σ_m B_m(τ) A_m` and its velocity `ż(τ) = Σ_m B'_m(τ) A_m`
//! are then two matrix products with the coefficients `A` `[M × H]`.

use legume_numeric::candle::candle_core::{Device, Result, Tensor};

/// The degree of the basis.
const DEGREE: usize = 3;

/// Clamped knots for `m` basis functions of [`DEGREE`] on `[0, 1]`:
/// `DEGREE + 1` repeated at each end, the inner ones evenly spaced.
fn knots(m: usize) -> Vec<f64> {
    let inner = m - DEGREE - 1;
    let mut t = vec![0.0; DEGREE + 1];
    t.extend((1..=inner).map(|i| i as f64 / (inner + 1) as f64));
    t.extend(std::iter::repeat_n(1.0, DEGREE + 1));
    t
}

/// Every basis function of degree `p` at `x` (Cox–de Boor), `knots.len() − p − 1`
/// values. `x = 1` belongs to the last interval.
fn basis(knots: &[f64], p: usize, x: f64) -> Vec<f64> {
    let n0 = knots.len() - 1;
    let last = knots[n0];
    let mut b: Vec<f64> = (0..n0)
        .map(|i| {
            let (a, c) = (knots[i], knots[i + 1]);
            let inside = (a <= x && x < c) || (x >= last && c >= last && a < c);
            f64::from(u8::from(inside))
        })
        .collect();
    for d in 1..=p {
        b = (0..n0 - d)
            .map(|i| {
                let left = knots[i + d] - knots[i];
                let right = knots[i + d + 1] - knots[i + 1];
                let l = if left > 0.0 {
                    (x - knots[i]) / left * b[i]
                } else {
                    0.0
                };
                let r = if right > 0.0 {
                    (knots[i + d + 1] - x) / right * b[i + 1]
                } else {
                    0.0
                };
                l + r
            })
            .collect();
    }
    b
}

/// The `k` grid times `(i + ½)/k` over `(0, 1)`.
pub fn grid_times(k: usize) -> Vec<f32> {
    (0..k).map(|i| (i as f32 + 0.5) / k as f32).collect()
}

/// The basis and its derivative at fixed times.
pub struct Spline {
    /// `[T × M]` values and first derivatives.
    pub values: Tensor,
    pub slopes: Tensor,
}

impl Spline {
    /// `m ≥ DEGREE + 1` basis functions at the times `tau` on `dev`.
    pub fn new(m: usize, tau: &[f32], dev: &Device) -> Result<Self> {
        assert!(
            m > DEGREE,
            "a cubic spline needs at least {} basis functions",
            DEGREE + 1
        );
        let t = knots(m);
        let mut values = Vec::with_capacity(tau.len() * m);
        let mut slopes = Vec::with_capacity(tau.len() * m);
        for &x in tau {
            let x = f64::from(x).clamp(0.0, 1.0);
            values.extend(basis(&t, DEGREE, x).into_iter().map(|v| v as f32));
            // B'_{i,p} = p/(t_{i+p} − t_i) B_{i,p−1} − p/(t_{i+p+1} − t_{i+1}) B_{i+1,p−1}.
            let lower = basis(&t, DEGREE - 1, x);
            let p = DEGREE as f64;
            slopes.extend((0..m).map(|i| {
                let left = t[i + DEGREE] - t[i];
                let right = t[i + DEGREE + 1] - t[i + 1];
                let l = if left > 0.0 { p / left * lower[i] } else { 0.0 };
                let r = if right > 0.0 {
                    p / right * lower[i + 1]
                } else {
                    0.0
                };
                (l - r) as f32
            }));
        }
        Ok(Self {
            values: Tensor::from_vec(values, (tau.len(), m), dev)?,
            slopes: Tensor::from_vec(slopes, (tau.len(), m), dev)?,
        })
    }
}

#[cfg(test)]
#[path = "tests/curve.rs"]
mod tests;
