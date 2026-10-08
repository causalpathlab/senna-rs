//! Helpers shared by the stage-1 and joint tests.

use rand::rngs::StdRng;
use rand::SeedableRng;
use rand_distr::{Distribution, Normal};

fn ranks(x: &[f32]) -> Vec<f64> {
    let mut idx: Vec<usize> = (0..x.len()).collect();
    idx.sort_by(|&a, &b| x[a].total_cmp(&x[b]));
    let mut r = vec![0f64; x.len()];
    for (k, &i) in idx.iter().enumerate() {
        r[i] = k as f64;
    }
    r
}

fn pearson(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len() as f64;
    let (ma, mb) = (a.iter().sum::<f64>() / n, b.iter().sum::<f64>() / n);
    let cov: f64 = a.iter().zip(b).map(|(x, y)| (x - ma) * (y - mb)).sum();
    let va: f64 = a.iter().map(|x| (x - ma).powi(2)).sum();
    let vb: f64 = b.iter().map(|y| (y - mb).powi(2)).sum();
    cov / (va * vb).sqrt()
}

pub fn spearman(a: &[f32], b: &[f32]) -> f64 {
    pearson(&ranks(a), &ranks(b))
}

/// The true times with Gaussian noise, clamped into `(0, 1)`: a rough order
/// such as a diffusion pseudotime gives.
pub fn noisy(tau: &[f32], sd: f64, seed: u64) -> Vec<f32> {
    let mut rng = StdRng::seed_from_u64(seed);
    let e = Normal::new(0.0, sd).unwrap();
    tau.iter()
        .map(|&t| (f64::from(t) + e.sample(&mut rng)).clamp(0.02, 0.98) as f32)
        .collect()
}
