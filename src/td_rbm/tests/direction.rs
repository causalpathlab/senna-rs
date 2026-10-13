use super::*;
use rand::SeedableRng;
use rand_distr::{Distribution, Poisson};

/// The tensor evidence (f32, what a GPU runs) and the host's (f64, exact
/// log-gammas) agree gene by gene, both orientations, on reads with dynamics.
#[test]
fn tensor_and_host_evidence_agree() {
    let (k, g) = (12, 9);
    let mut rng = StdRng::seed_from_u64(5);
    let mut x = vec![vec![0f32; g]; k];
    let (mut u, mut s) = (vec![vec![0f32; g]; k], vec![vec![0f32; g]; k]);
    for kk in 0..k {
        for j in 0..g {
            let t = kk as f64 / k as f64;
            let xv = (j as f64 - 4.0) * 0.3 * (6.0 * t).cos();
            x[kk][j] = xv as f32;
            let (beta, gamma) = (2.0, 1.0 + 0.1 * j as f64);
            let ratio = ((gamma + xv).max(0.05 * gamma)) / beta;
            let total = 5.0 + 40.0 * (j as f64 + 1.0) * (1.0 + t);
            let mean_u = total * ratio / (1.0 + ratio);
            u[kk][j] = Poisson::new(mean_u).unwrap().sample(&mut rng) as f32;
            s[kk][j] = Poisson::new(total - mean_u).unwrap().sample(&mut rng) as f32;
        }
    }
    let rest: Vec<f32> = (0..g)
        .map(|j| {
            let (a, b) = (0..k).fold((0f64, 0f64), |(a, b), kk| {
                (a + f64::from(u[kk][j]), b + f64::from(s[kk][j]))
            });
            ((a + 0.5) / (b + 0.5)).ln() as f32
        })
        .collect();
    let reach: Vec<f32> = (0..g)
        .map(|j| (0..k).map(|kk| x[kk][j].abs()).fold(1e-6f32, f32::max).ln())
        .collect();
    let dev = Device::Cpu;
    let t = |m: &[Vec<f32>]| Tensor::from_vec(m.concat(), (k, g), &dev).unwrap();
    for sign in [1.0, -1.0] {
        let host = evidence_host(&u, &s, &x, &rest, &reach, sign);
        let tens = evidence(&t(&u), &t(&s), &t(&x), &rest, &reach, sign, &dev).unwrap();
        for j in 0..g {
            let (a, b) = (host.log_e[j], tens.log_e[j]);
            assert!(
                (a - b).abs() < 0.05,
                "gene {j}, sign {sign}: host {a}, tensor {b}"
            );
            assert!((host.log_gamma[j] - tens.log_gamma[j]).abs() < 0.05);
            assert!((host.log_beta[j] - tens.log_beta[j]).abs() < 0.05);
        }
    }
}
