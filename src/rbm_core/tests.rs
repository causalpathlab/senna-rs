use super::*;
use rand::SeedableRng;

#[test]
fn multinomial_keeps_the_depth_and_the_shares() {
    let mut rng = StdRng::seed_from_u64(1);
    let p = [0.5f32, 0.3, 0.2];
    let mut total = [0f64; 3];
    let mut out = [0f32; 3];
    let reps = 4000;
    for _ in 0..reps {
        multinomial(10, &p, &mut out, &mut rng);
        assert_eq!(out.iter().sum::<f32>(), 10.0);
        for (t, &o) in total.iter_mut().zip(&out) {
            *t += f64::from(o);
        }
    }
    for (t, &pi) in total.iter().zip(&p) {
        assert!((t / (10.0 * reps as f64) - f64::from(pi)).abs() < 0.01);
    }
}

#[test]
fn blocked_cumsum_matches_the_plain_one() {
    use legume_numeric::candle::candle_core::{Device, Tensor};
    for f in [1usize, 2, 7, 16, 37] {
        let v: Vec<f32> = (0..3 * f)
            .map(|i| ((i * 7919) % 13) as f32 / 13.0)
            .collect();
        let t = Tensor::from_vec(v, (3, f), &Device::Cpu).unwrap();
        let got = cumsum_rows(&t).unwrap().to_vec2::<f32>().unwrap();
        let want = t.cumsum(1).unwrap().to_vec2::<f32>().unwrap();
        for (g, w) in got.iter().flatten().zip(want.iter().flatten()) {
            assert!((g - w).abs() < 1e-5, "F = {f}: {g} vs {w}");
        }
    }
}

#[test]
fn multinomial_never_draws_a_category_with_no_share() {
    let mut rng = StdRng::seed_from_u64(9);
    // Shares that sum a little under one, the last ones empty.
    let p = [0.6f32 - 1e-6, 0.4 - 1e-6, 0.0, 0.0];
    let mut out = [0f32; 4];
    for _ in 0..2000 {
        multinomial(20, &p, &mut out, &mut rng);
        assert_eq!(out.iter().sum::<f32>(), 20.0);
        assert_eq!(out[2] + out[3], 0.0, "{out:?}");
    }
}
