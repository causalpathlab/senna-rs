//! Splicing kinetics of a gene module, in closed form.
//!
//! A module m switches its genes' transcription on at `t_on` and off at
//! `t_off = t_on + T`. Transcription relaxes toward its target at rate `λ`:
//!
//! ```text
//! a(t) = r                                   t < t_on
//!      = 1 − (1 − r) e^{−λ(t − t_on)}        t_on ≤ t < t_off
//!      = r + (a_off − r) e^{−λ(t − t_off)}   t ≥ t_off
//! du/dt = a − β u,   ds/dt = β u − γ s
//! ```
//!
//! with `r` the basal fraction; before `t_on` the module sits at its basal
//! steady state `(r/β, r/γ)`. The scale of `a` cancels in `u/s`, which is all
//! the track classifier reads.
//!
//! On a piece starting from `(u0, s0)` with input `A + B e^{−λΔ}`,
//!
//! ```text
//! u(Δ) = A g[0,β] + u0 e^{−βΔ} + B g[λ,β]
//! s(Δ) = s0 e^{−γΔ} + A g[0,γ] + (β u0 − A) g[β,γ] + β B g[λ,β,γ]
//! ```
//!
//! where `g[a,b] = (e^{−aΔ} − e^{−bΔ})/(b − a)` and `g[a,b,c]` are the
//! (sign-adjusted) divided differences of `x ↦ e^{−xΔ}`. Both are evaluated
//! stably when rates coincide, so gradients stay finite at `β = γ = λ`.

use legume_numeric::candle::candle_core::{DType, Device, Result, Tensor};

/// Below this `|spread · Δ|` a divided difference takes its Taylor series.
const SMALL: f64 = 1e-2;

/// One set of module kinetics on the constrained scale; every field is `[M]`.
#[derive(Clone, Debug)]
pub struct Kinetics {
    /// Switch-on time.
    pub t_on: Tensor,
    /// Time from switch-on to switch-off, `> 0`.
    pub duration: Tensor,
    /// Rate at which transcription relaxes toward its target, `> 0`.
    pub lambda: Tensor,
    /// Basal transcription as a fraction of the induced level, in `(0, 1)`.
    pub basal: Tensor,
    /// Splicing rate, `> 0`.
    pub beta: Tensor,
    /// Degradation rate of spliced RNA, `> 0`.
    pub gamma: Tensor,
}

/// One module's kinetics as plain numbers.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ModuleKinetics {
    pub t_on: f32,
    pub duration: f32,
    pub lambda: f32,
    pub basal: f32,
    pub beta: f32,
    pub gamma: f32,
}

impl Kinetics {
    /// The modules as `[M]` tensors of `dtype` on `dev`.
    pub fn from_modules(modules: &[ModuleKinetics], dtype: DType, dev: &Device) -> Result<Self> {
        let col = |f: fn(&ModuleKinetics) -> f32| {
            let v: Vec<f32> = modules.iter().map(f).collect();
            Tensor::from_vec(v, modules.len(), dev)?.to_dtype(dtype)
        };
        Ok(Self {
            t_on: col(|m| m.t_on)?,
            duration: col(|m| m.duration)?,
            lambda: col(|m| m.lambda)?,
            basal: col(|m| m.basal)?,
            beta: col(|m| m.beta)?,
            gamma: col(|m| m.gamma)?,
        })
    }

    /// The modules as plain numbers.
    pub fn to_modules(&self) -> Result<Vec<ModuleKinetics>> {
        let col = |t: &Tensor| t.to_dtype(DType::F32)?.to_vec1::<f32>();
        let (t_on, duration, lambda) = (col(&self.t_on)?, col(&self.duration)?, col(&self.lambda)?);
        let (basal, beta, gamma) = (col(&self.basal)?, col(&self.beta)?, col(&self.gamma)?);
        Ok((0..t_on.len())
            .map(|m| ModuleKinetics {
                t_on: t_on[m],
                duration: duration[m],
                lambda: lambda[m],
                basal: basal[m],
                beta: beta[m],
                gamma: gamma[m],
            })
            .collect())
    }
}

/// `(e^{−aΔ} − e^{−bΔ}) / (b − a)`, elementwise on equal shapes; `Δ e^{−aΔ}`
/// at `a = b`.
pub fn dd2(a: &Tensor, b: &Tensor, dt: &Tensor) -> Result<Tensor> {
    let d = (b - a)?;
    let x = (&d * dt)?;
    let small = x.abs()?.lt(SMALL)?;
    // Taylor about the midpoint: Δ e^{−āΔ} (1 + x²/24 + x⁴/1920).
    let base = ((a + b)? * 0.5)?.mul(dt)?.neg()?.exp()?;
    let x2 = x.sqr()?;
    let series = ((&x2 * (1.0 / 24.0))? + (x2.sqr()? * (1.0 / 1920.0))?)?;
    let taylor = (dt * &base)?.mul(&(series + 1.0)?)?;
    // The exact branch sees a unit denominator where it is not taken, so its
    // value and gradient stay finite there too.
    let safe = small.where_cond(&d.ones_like()?, &d)?;
    let exact = ((a * dt)?.neg()?.exp()? - (b * dt)?.neg()?.exp()?)?.div(&safe)?;
    small.where_cond(&taylor, &exact)
}

/// The second divided difference of `x ↦ e^{−xΔ}` at `a, b, c` (symmetric,
/// positive), elementwise on equal shapes; `Δ² e^{−aΔ} / 2` at `a = b = c`.
pub fn dd3(a: &Tensor, b: &Tensor, c: &Tensor, dt: &Tensor) -> Result<Tensor> {
    let lo = a.minimum(b)?.minimum(c)?;
    let hi = a.maximum(b)?.maximum(c)?;
    let sum = ((a + b)? + c)?;
    let mid = ((&sum - &lo)? - &hi)?;
    let spread = (&hi - &lo)?;
    let small = (&spread * dt)?.abs()?.lt(SMALL)?;
    // Taylor about the mean: e^{−x̄Δ} (Δ²/2 + Δ⁴ h₂/24), h₂ the complete
    // symmetric polynomial of degree 2 in the deviations.
    let mean = (&sum * (1.0 / 3.0))?;
    let (ea, eb, ec) = ((a - &mean)?, (b - &mean)?, (c - &mean)?);
    let h2 = (((ea.sqr()? + eb.sqr()?)? + ec.sqr()?)?
        + (((&ea * &eb)? + (&ea * &ec)?)? + (&eb * &ec)?)?)?;
    let dt2 = dt.sqr()?;
    let base = (&mean * dt)?.neg()?.exp()?;
    let taylor = ((&dt2 * 0.5)? + (dt2.sqr()?.mul(&h2)? * (1.0 / 24.0))?)?.mul(&base)?;
    let safe = small.where_cond(&spread.ones_like()?, &spread)?;
    let exact = (dd2(&lo, &mid, dt)? - dd2(&mid, &hi, dt)?)?.div(&safe)?;
    small.where_cond(&taylor, &exact)
}

/// `(u, s)` after `dt` of input `A + B e^{−λΔ}` from `(u0, s0)`; all `[T × M]`.
#[allow(clippy::too_many_arguments)]
fn piece(
    a_in: &Tensor,
    b_in: &Tensor,
    u0: &Tensor,
    s0: &Tensor,
    lambda: &Tensor,
    beta: &Tensor,
    gamma: &Tensor,
    dt: &Tensor,
) -> Result<(Tensor, Tensor)> {
    let zero = dt.zeros_like()?;
    let u = ((a_in * dd2(&zero, beta, dt)?)?
        + (u0 * (beta * dt)?.neg()?.exp()?)?
        + (b_in * dd2(lambda, beta, dt)?)?)?;
    let s = ((s0 * (gamma * dt)?.neg()?.exp()?)?
        + (a_in * dd2(&zero, gamma, dt)?)?
        + (((beta * u0)? - a_in)? * dd2(beta, gamma, dt)?)?
        + ((beta * b_in)? * dd3(lambda, beta, gamma, dt)?)?)?;
    Ok((u, s))
}

/// Every module's `(u, s)` at the times `tau` `[T]`: two `[T × M]` tensors.
pub fn curves(k: &Kinetics, tau: &Tensor) -> Result<(Tensor, Tensor)> {
    let t = tau.dim(0)?;
    let m = k.t_on.dim(0)?;
    let row = |x: &Tensor| x.unsqueeze(0)?.broadcast_as((t, m))?.contiguous();
    let (t_on, dur, lambda, r, beta, gamma) = (
        row(&k.t_on)?,
        row(&k.duration)?,
        row(&k.lambda)?,
        row(&k.basal)?,
        row(&k.beta)?,
        row(&k.gamma)?,
    );
    let tau = tau.unsqueeze(1)?.broadcast_as((t, m))?.contiguous()?;
    let since_on = (&tau - &t_on)?;
    // Piece 1 runs from t_on for at most T; piece 2 takes over after t_off.
    let dt1 = since_on.maximum(0.0)?.minimum(&dur)?;
    let dt2 = (&since_on - &dur)?.maximum(0.0)?;
    let ones = r.ones_like()?;
    let (u1, s1) = piece(
        &ones,
        &(&r - 1.0)?,
        &(&r / &beta)?,
        &(&r / &gamma)?,
        &lambda,
        &beta,
        &gamma,
        &dt1,
    )?;
    // Transcription where piece 1 stops; the off piece relaxes from there.
    let a_end = (1.0 - ((1.0 - &r)? * (&lambda * &dt1)?.neg()?.exp()?)?)?;
    piece(&r, &(a_end - &r)?, &u1, &s1, &lambda, &beta, &gamma, &dt2)
}

/// `d log s / dt = β u/s − γ` of every module at `tau`, `[T × M]`: the rate
/// at which each module's spliced RNA changes, the velocity readout's input.
pub fn spliced_log_rate(k: &Kinetics, tau: &Tensor) -> Result<Tensor> {
    let (u, s) = curves(k, tau)?;
    (u / s)?
        .broadcast_mul(&k.beta.unsqueeze(0)?)?
        .broadcast_sub(&k.gamma.unsqueeze(0)?)
}

/// `log u − log s` of every module at `tau`, `[T × M]`.
pub fn log_ratio(k: &Kinetics, tau: &Tensor) -> Result<Tensor> {
    let (u, s) = curves(k, tau)?;
    u.log()? - s.log()?
}

#[cfg(test)]
#[path = "kinetics/tests.rs"]
mod tests;
