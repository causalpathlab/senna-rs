use super::*;

#[test]
fn the_basis_sums_to_one_everywhere() {
    let tau: Vec<f32> = (0..=50).map(|i| i as f32 / 50.0).collect();
    let s = Spline::new(8, &tau, &Device::Cpu).unwrap();
    for row in s.values.to_vec2::<f32>().unwrap() {
        assert!((row.iter().sum::<f32>() - 1.0).abs() < 1e-5, "{row:?}");
        assert!(row.iter().all(|&v| v >= -1e-7));
    }
    // Derivatives of a partition of unity sum to zero.
    for row in s.slopes.to_vec2::<f32>().unwrap() {
        assert!(row.iter().sum::<f32>().abs() < 1e-3, "{row:?}");
    }
}

#[test]
fn slopes_match_finite_differences() {
    let h = 1e-3f32;
    let tau = [0.05f32, 0.3, 0.5, 0.77, 0.95];
    let s = Spline::new(7, &tau, &Device::Cpu).unwrap();
    let up: Vec<f32> = tau.iter().map(|t| t + h).collect();
    let down: Vec<f32> = tau.iter().map(|t| t - h).collect();
    let (a, b) = (
        Spline::new(7, &up, &Device::Cpu).unwrap().values,
        Spline::new(7, &down, &Device::Cpu).unwrap().values,
    );
    let fd = ((a - b).unwrap() / f64::from(2.0 * h))
        .unwrap()
        .to_vec2::<f32>()
        .unwrap();
    let exact = s.slopes.to_vec2::<f32>().unwrap();
    for (r, e) in fd.iter().zip(&exact) {
        for (x, y) in r.iter().zip(e) {
            assert!((x - y).abs() < 2e-2 * (1.0 + y.abs()), "{x} vs {y}");
        }
    }
}
