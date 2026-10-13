use super::*;
use crate::td_rbm::test_util::spearman;
use rand::rngs::StdRng;
use rand::SeedableRng;
use rand_distr::{Distribution, Normal, Uniform};

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

/// Rows' expression states order them along their time: the diffusion
/// order of the states of simulated cells, each its own row, follows their
/// true time.
#[test]
fn expression_states_order_rows_along_time() {
    use crate::td_rbm::sim::{simulate, SimConfig};
    let sim = simulate(&SimConfig {
        n_cells: 600,
        ..SimConfig::default()
    })
    .unwrap();
    let (p, g) = (sim.n_cells, sim.n_genes);
    let x: Vec<f32> = (0..p)
        .flat_map(|i| {
            sim.unspliced[i * g..(i + 1) * g]
                .iter()
                .chain(&sim.spliced[i * g..(i + 1) * g])
                .copied()
                .collect::<Vec<_>>()
        })
        .collect();
    let states = crate::td_rbm::start::row_states(&x, 2 * g, 30);
    assert_eq!(states.shape(), (p, 30));
    let rho = spearman(&diffusion_order(&states, 15), &sim.tau);
    assert!(rho.abs() > 0.9, "Spearman {rho}");
}
