use super::*;
use rand::rngs::StdRng;
use rand::SeedableRng;
use rand_distr::{Distribution, Normal, Uniform};

fn ranks(x: &[f32]) -> Vec<f64> {
    let mut idx: Vec<usize> = (0..x.len()).collect();
    idx.sort_by(|&a, &b| x[a].total_cmp(&x[b]));
    let mut r = vec![0f64; x.len()];
    for (k, &i) in idx.iter().enumerate() {
        r[i] = k as f64;
    }
    r
}

fn spearman(a: &[f32], b: &[f32]) -> f64 {
    let (ra, rb) = (ranks(a), ranks(b));
    let n = ra.len() as f64;
    let m = (n - 1.0) / 2.0;
    let cov: f64 = ra.iter().zip(&rb).map(|(x, y)| (x - m) * (y - m)).sum();
    let var: f64 = ra.iter().map(|x| (x - m).powi(2)).sum();
    cov / var
}

/// Points along a bent curve in six dimensions, with noise.
#[test]
fn orders_a_noisy_curve_along_its_length() {
    let mut rng = StdRng::seed_from_u64(3);
    let u = Uniform::new(0.0f32, 1.0).unwrap();
    let e = Normal::new(0.0f32, 0.02).unwrap();
    let t: Vec<f32> = (0..300).map(|_| u.sample(&mut rng)).collect();
    let theta = DMatrix::<f32>::from_fn(300, 6, |i, d| {
        let a = std::f32::consts::PI * t[i];
        let x = match d {
            0 => a.cos(),
            1 => a.sin(),
            2 => 0.5 * t[i],
            3 => 1.0,
            _ => 0.0,
        };
        x + e.sample(&mut rng)
    });
    let tau = diffusion_order(&theta, 15);
    assert!(tau.iter().all(|&x| (0.05..=0.95).contains(&x)));
    let rho = spearman(&tau, &t);
    assert!(rho.abs() > 0.95, "Spearman {rho}");
}
