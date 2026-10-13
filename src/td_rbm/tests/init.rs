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
    let tracks = crate::td_rbm::counts::PbTracks {
        genes: (0..g).map(|j| format!("GENE{j}").into()).collect(),
        unspliced: DMatrix::from_row_slice(p, g, &sim.unspliced),
        spliced: DMatrix::from_row_slice(p, g, &sim.spliced),
        cells: vec![1.0; p],
    };
    let genes: Vec<usize> = (0..g).collect();
    let states = expression_states(&tracks, &genes, 30);
    assert_eq!(states.shape(), (p, 30));
    let rho = spearman(&diffusion_order(&states, 15), &sim.tau);
    assert!(rho.abs() > 0.9, "Spearman {rho}");
}
