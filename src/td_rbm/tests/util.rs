//! Helpers shared by the td-rbm tests.

use rand::rngs::StdRng;
use rand::SeedableRng;
use rand_distr::{Distribution, Normal};

pub use crate::td_rbm::start::rank_correlation as spearman;

/// The true times with Gaussian noise, clamped into `(0, 1)`: a rough order
/// such as a diffusion pseudotime gives.
pub fn noisy(tau: &[f32], sd: f64, seed: u64) -> Vec<f32> {
    let mut rng = StdRng::seed_from_u64(seed);
    let e = Normal::new(0.0, sd).unwrap();
    tau.iter()
        .map(|&t| (f64::from(t) + e.sample(&mut rng)).clamp(0.02, 0.98) as f32)
        .collect()
}
