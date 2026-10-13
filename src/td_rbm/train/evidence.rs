//! Log-gamma and log-beta as tensor ops in f32, stable over the ranges the
//! direction's evidence needs (arguments from about 1e-7 to a few thousand):
//! no two large log-gammas are ever subtracted. Candle has no log-gamma.

use legume_numeric::candle::candle_core::{Result, Tensor};

const HALF_LN_2PI: f64 = 0.918_938_533_204_672_8;
/// Stirling's series is used at or above this argument.
const LARGE: f64 = 8.0;

/// Stirling's series' remainder past `(z − ½) ln z − z + ½ ln 2π`.
fn stirling_tail(z: &Tensor) -> Result<Tensor> {
    let inv = z.recip()?;
    let inv2 = inv.sqr()?;
    // 1/(12z) − 1/(360z³) + 1/(1260z⁵)
    let poly = ((inv2.affine(1.0 / 1260.0, 0.0)? - (1.0 / 360.0))?.mul(&inv2)? + (1.0 / 12.0))?;
    poly.mul(&inv)
}

/// `ln(1 + r)` for `r ≥ 0`, by its series where `1 + r` would round to 1.
fn ln_1p(r: &Tensor) -> Result<Tensor> {
    let series = r.mul(
        &r.affine(-0.5, 1.0)?
            .add(&r.sqr()?.affine(1.0 / 3.0, 0.0)?)?,
    )?;
    let direct = r.affine(1.0, 1.0)?.log()?;
    r.lt(1e-3)?.where_cond(&series, &direct)
}

/// `ln Γ(z)` for `z ≥ LARGE` by Stirling's series.
fn ln_gamma_large(z: &Tensor) -> Result<Tensor> {
    let main = (z.affine(1.0, -0.5)?.mul(&z.log()?)? - z)?;
    (main + stirling_tail(z)?)?.affine(1.0, HALF_LN_2PI)
}

/// `ln Γ(z)` for any `z > 0`: Stirling's series at `z + 8`, less
/// `ln(z (z+1) ⋯ (z+7))` (exact to f32 rounding for small `z`, where the
/// values themselves are small).
pub fn ln_gamma(z: &Tensor) -> Result<Tensor> {
    let mut shift = z.log()?;
    for i in 1..8 {
        shift = (shift + z.affine(1.0, i as f64)?.log()?)?;
    }
    ln_gamma_large(&z.affine(1.0, LARGE)?)? - shift
}

/// `ln B(x, y)` for `x, y > 0`, elementwise.
/// - both `≥ LARGE`: `x ln(x/s) + y ln(y/s) − ½ ln(xy/s) + ½ ln 2π` plus
///   Stirling's remainders, `s = x + y` (no large terms cancel);
/// - the smaller below: `ln Γ(small) + [ln Γ(large) − ln Γ(s)]`, the bracket
///   by the difference of Stirling's series;
/// - both below: `ln Γ` directly, all three values small.
pub fn ln_beta(x: &Tensor, y: &Tensor) -> Result<Tensor> {
    let s = (x + y)?;
    let lo = x.minimum(y)?;
    let hi = x.maximum(y)?;
    // Both large.
    let hi_c = hi.maximum(LARGE)?;
    let lo_c = lo.maximum(LARGE)?;
    let s_c = (&hi_c + &lo_c)?;
    let both = {
        // ln(hi/s) = −ln(1 + lo/hi), exact even where lo ≪ hi.
        let a = lo_c.mul(&lo_c.div(&s_c)?.log()?)?;
        let b = hi_c.mul(&ln_1p(&lo_c.div(&hi_c)?)?)?.neg()?;
        let half = lo_c
            .mul(&hi_c)?
            .div(&s_c)?
            .log()?
            .affine(-0.5, HALF_LN_2PI)?;
        let tail = stirling_tail(&lo_c)? + stirling_tail(&hi_c)? - stirling_tail(&s_c)?;
        (((a + b)? + half)? + tail?)?
    };
    // One small, one large: ln Γ(hi) − ln Γ(hi + lo) by Stirling's
    // difference, (hi − ½) ln(hi/s) − lo ln s + lo + tail(hi) − tail(s).
    let one = {
        let diff = (hi_c
            .affine(1.0, -0.5)?
            .mul(&ln_1p(&lo.div(&hi_c)?)?)?
            .neg()?
            - lo.mul(&(&hi_c + &lo)?.log()?)?)?;
        let tail = (stirling_tail(&hi_c)? - stirling_tail(&(&hi_c + &lo)?)?)?;
        ((ln_gamma(&lo)? + diff)? + (&lo + tail)?)?
    };
    // Both small.
    let neither = ((ln_gamma(x)? + ln_gamma(y)?)? - ln_gamma(&s)?)?;
    let lo_large = lo.ge(LARGE)?;
    let hi_large = hi.ge(LARGE)?;
    let small = hi_large.where_cond(&one, &neither)?;
    lo_large.where_cond(&both, &small)
}

#[cfg(test)]
#[path = "../tests/evidence.rs"]
mod tests;
