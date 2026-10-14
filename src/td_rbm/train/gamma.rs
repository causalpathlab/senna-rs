//! A fast, approximate `ln Γ` over a slice: Paul Mineiro's `fastlgamma`
//! (fastapprox), in f32 and branch-free, so the compiler vectorizes it (AVX2
//! and FMA where the processor has them, picked at run time). About `1e-3`
//! relative: speed over the last digits, for evidence integrated over a grid.
//!
//! `ln Γ(x) ≈ −2.0810615 − x + 1/(12(x+3)) − ln(x (x+1) (x+2)) + (x + 2.5) ln(x+3)`,
//! every log by `fastlog` (the float's bits read as its log₂, corrected).

/// Mineiro's `fastlog`: `ln x` for a positive normal `x`, about `1e-4`.
#[inline(always)]
fn fast_ln(x: f32) -> f32 {
    let i = x.to_bits();
    let m = f32::from_bits((i & 0x007F_FFFF) | 0x3f00_0000);
    let y = i as f32 * 1.192_092_9e-7;
    std::f32::consts::LN_2 * (y - 124.225_52 - 1.498_030_3 * m - 1.725_88 / (0.352_088_7 + m))
}

/// Mineiro's `fastlgamma`, for `x > 0`.
#[inline(always)]
fn fast_ln_gamma(x: f32) -> f32 {
    let xp3 = 3.0 + x;
    -2.081_061_5 - x + 0.083_333_3 / xp3 - fast_ln(x * (1.0 + x) * (2.0 + x))
        + (2.5 + x) * fast_ln(xp3)
}

#[inline(always)]
fn each(xs: &mut [f32]) {
    for x in xs {
        *x = fast_ln_gamma(*x);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn each_avx2(xs: &mut [f32]) {
    each(xs);
}

/// Every `x` in place by its approximate `ln Γ(x)`; each must be positive.
pub(crate) fn ln_gamma_in_place(xs: &mut [f32]) {
    #[cfg(target_arch = "x86_64")]
    if std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma") {
        // SAFETY: the processor has the features the function is built for.
        unsafe { each_avx2(xs) };
        return;
    }
    each(xs);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fast_ln_gamma_is_near_the_library() {
        let lg = |z: f64| special::Gamma::ln_gamma(z).0;
        let mut xs = Vec::new();
        let mut v = 1e-5f32;
        while v < 2e5 {
            xs.push(v);
            v *= 1.01;
        }
        let mut got = xs.clone();
        ln_gamma_in_place(&mut got);
        for (&x, &g) in xs.iter().zip(&got) {
            let want = lg(f64::from(x));
            let err = (want - f64::from(g)).abs();
            assert!(
                err <= 2e-3 * want.abs().max(1.0),
                "ln Γ({x}): {g} against {want}"
            );
        }
    }
}
