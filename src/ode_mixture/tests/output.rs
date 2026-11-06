use crate::ode_mixture::output::{chain_stats, MapBasis, RHAT_FLAG};
use nalgebra::DMatrix;
use rand::rngs::StdRng;
use rand::SeedableRng;
use rand_distr::{Distribution, Normal};

/// `chains` chains of `draws` draws of two coordinates, coordinate `j`'s
/// chain `c` centred at `shift[j] * c`.
fn normal_chains(chains: usize, draws: usize, shift: [f64; 2]) -> Vec<Vec<Vec<f64>>> {
    let mut rng = StdRng::seed_from_u64(5);
    let z = Normal::new(0.0, 1.0).unwrap();
    (0..chains)
        .map(|c| {
            (0..draws)
                .map(|_| {
                    (0..2)
                        .map(|j| 1.0 + shift[j] * c as f64 + z.sample(&mut rng))
                        .collect()
                })
                .collect()
        })
        .collect()
}

/// Over well-mixed chains a coordinate's mean, sd and ESS are the
/// draws', its R̂ about 1 and not flagged; chains apart are flagged.
#[test]
fn chain_stats_summarise_and_flag_chains_apart() {
    let stats = chain_stats(&normal_chains(4, 500, [0.0, 3.0]));
    assert_eq!(stats.len(), 2);
    let (mixed, apart) = (&stats[0], &stats[1]);
    assert!((mixed.mean - 1.0).abs() < 0.1, "mean {}", mixed.mean);
    assert!((mixed.sd - 1.0).abs() < 0.1, "sd {}", mixed.sd);
    assert!(mixed.ess > 1500.0, "ess {}", mixed.ess);
    assert!(mixed.rhat < 1.01 && !mixed.flagged(), "R̂ {}", mixed.rhat);
    assert!(
        apart.rhat > RHAT_FLAG && apart.flagged(),
        "R̂ {}",
        apart.rhat
    );
}

/// No draws: every coordinate unknown, and flagged.
#[test]
fn chain_stats_of_no_draws_are_unknown() {
    let stats = chain_stats(&[vec![], vec![]]);
    assert!(stats.is_empty());
    let stats = chain_stats(&[vec![vec![1.0]], vec![]]);
    assert_eq!(stats.len(), 1);
    assert!(stats[0].rhat.is_nan() && stats[0].flagged());
}

/// Rows of reads `[P × G]` with two directions of variation.
fn rows() -> DMatrix<f64> {
    DMatrix::from_fn(40, 6, |i, j| {
        let t = i as f64 / 39.0;
        let a = [5.0, 1.0, 3.0, 0.5, 2.0, 4.0][j];
        let b = [0.0, 2.0, -1.0, 1.0, 0.5, -2.0][j];
        50.0 * (a + b * t + 0.3 * (i % 3) as f64).exp()
    })
}

/// The basis places its own rows at scores of zero mean, the first
/// component the most spread.
#[test]
fn the_map_places_its_rows_centred() {
    let x = rows();
    let map = MapBasis::new(&x, 3).unwrap();
    assert_eq!(map.dim(), 3);
    let placed: Vec<Vec<f64>> = x
        .row_iter()
        .map(|r| map.place(&r.iter().copied().collect::<Vec<_>>()))
        .collect();
    let var = |d: usize| {
        let m = placed.iter().map(|p| p[d]).sum::<f64>() / placed.len() as f64;
        assert!(m.abs() < 1e-6, "component {d} mean {m}");
        placed.iter().map(|p| (p[d] - m).powi(2)).sum::<f64>()
    };
    assert!(var(0) >= var(1) && var(1) >= var(2));
}

/// A cell's velocity on the map is the rate its place moves at when every
/// gene's level changes at its relative rate `d log x_g/dt`.
#[test]
fn the_maps_velocity_is_the_places_rate_of_change() {
    let x = rows();
    let map = MapBasis::new(&x, 3).unwrap();
    let cell: Vec<f64> = x.row(17).iter().copied().collect();
    let dlog: [f64; 6] = [0.4, -0.2, 0.1, 0.9, -0.5, 0.0];
    let h = 1e-6;
    let ahead: Vec<f64> = cell
        .iter()
        .zip(&dlog)
        .map(|(c, v)| c * (v * h).exp())
        .collect();
    let (now, next) = (map.place(&cell), map.place(&ahead));
    let v = map.velocity(&cell, &dlog);
    for d in 0..3 {
        let fd = (next[d] - now[d]) / h;
        assert!(
            (v[d] - fd).abs() < 1e-4 * (1.0 + fd.abs()),
            "{d}: {} vs {fd}",
            v[d]
        );
    }
}
