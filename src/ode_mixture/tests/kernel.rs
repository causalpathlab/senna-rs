use crate::ode_mixture::kernel::{grid_cov, matern52, GridPrior};
use nalgebra::DVector;
use rand::rngs::StdRng;
use rand::SeedableRng;

#[test]
fn the_kernel_is_one_at_zero_and_decays() {
    assert!((matern52(0.0, 0.3) - 1.0).abs() < 1e-12);
    let v: Vec<f64> = (0..6).map(|i| matern52(0.1 * i as f64, 0.3)).collect();
    assert!(v.windows(2).all(|w| w[1] < w[0]), "{v:?}");
    assert!(matern52(3.0, 0.3) < 1e-4);
}

#[test]
fn the_grid_covariance_depends_on_the_length_scale_over_the_duration_only() {
    let a = grid_cov(21, 1.0, 0.25);
    let b = grid_cov(21, 8.0, 2.0);
    assert!((a - b).abs().max() < 1e-12);
}

#[test]
fn a_whitened_draw_has_the_kernel_covariance() {
    let prior = GridPrior::new(9, 2.0, 0.6).unwrap();
    let cov = grid_cov(9, 2.0, 0.6);
    let mut rng = StdRng::seed_from_u64(3);
    let n = 40_000;
    let mut acc = nalgebra::DMatrix::<f64>::zeros(9, 9);
    for _ in 0..n {
        let z = prior.draw(&mut rng);
        acc += &z * z.transpose();
    }
    acc /= n as f64;
    assert!((acc - cov).abs().max() < 0.03);
}

#[test]
fn whitening_round_trips() {
    let prior = GridPrior::new(65, 1.0, 0.05).unwrap();
    let eta = DVector::from_fn(65, |i, _| (i as f64 * 0.37).sin());
    let back = prior.whiten(&prior.colour(&eta));
    assert!((back - eta).abs().max() < 1e-6);
}
