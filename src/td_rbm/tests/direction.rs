use super::*;
use rand::SeedableRng;
use rand_distr::{Distribution, Poisson};

/// The host evidence scores tables together as it does alone, and reads
/// made forward favour forward.
#[test]
fn evidence_scores_tables_alike_and_reads_the_direction() {
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
    // Every table scored beside others, as the draws are, gets what it gets
    // alone: the shared per-grid terms do not leak between tables.
    let other: Vec<Vec<f32>> = s
        .iter()
        .map(|r| r.iter().map(|v| v + 3.0).collect())
        .collect();
    let (fwd, bwd) = evidence_host(
        &[(u.clone(), s.clone()), (u.clone(), other)],
        &x,
        &rest,
        &reach,
    );
    let (fwd1, bwd1) = evidence_host(&[(u.clone(), s.clone())], &x, &rest, &reach);
    for (together, alone) in [(&fwd[0], &fwd1[0]), (&bwd[0], &bwd1[0])] {
        for j in 0..g {
            assert!(
                (together.log_e[j] - alone.log_e[j]).abs() < 1e-9,
                "gene {j}"
            );
            assert!((together.log_gamma[j] - alone.log_gamma[j]).abs() < 1e-9);
            assert!((together.log_beta[j] - alone.log_beta[j]).abs() < 1e-9);
        }
    }
    // Reads made with the velocity forward favour forward, gene by gene, for
    // the genes the velocity moves.
    let moved = (0..g).filter(|&j| j != 4);
    let favoured = moved
        .filter(|&j| fwd1[0].log_e[j] > bwd1[0].log_e[j])
        .count();
    assert!(
        favoured >= g - 2,
        "{favoured} of {} genes favour forward",
        g - 1
    );
}
