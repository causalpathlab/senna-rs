use crate::embed_ode::curves::{grid_velocity, CurveSpec, Curves};
use crate::ode_mixture::kernel::GridPrior;
use crate::ode_mixture::model::branch_ramp;
use legume_numeric::candle::candle_core::{Device, Tensor};
use nalgebra::DVector;

/// `[L × m × H]` a fixed, non-trivial `η`, row-major.
fn some_eta(l: usize, m: usize, h: usize) -> Vec<f32> {
    (0..l * m * h)
        .map(|i| ((i as f32) * 0.37 + 0.2).sin())
        .collect()
}

fn mu_of(curves: &Curves, eta: &[f32], h: usize, log_a: f32) -> Vec<f32> {
    let dev = Device::Cpu;
    let (l, m) = (curves.lineages(), curves.basis_size());
    let eta = Tensor::from_vec(eta.to_vec(), (l, m, h), &dev).unwrap();
    let log_a = Tensor::from_vec(vec![log_a], (1,), &dev).unwrap();
    let mu = curves.mu(&eta, &log_a).unwrap();
    assert_eq!(mu.dims(), &[l, curves.positions(), h]);
    mu.flatten_all().unwrap().to_vec1().unwrap()
}

/// With no trunk every lineage's curve is `a C η_ℓ`, `C` the prior's
/// Cholesky factor on the positions.
#[test]
fn without_a_trunk_a_curve_is_the_amplitude_times_the_coloured_eta() {
    let spec = CurveSpec {
        lineages: 2,
        points: 9,
        sub: 4,
        kappa: 0.25,
        branch: None,
    };
    let curves = Curves::new(&spec, &Device::Cpu).unwrap();
    let n = 33;
    assert_eq!((curves.positions(), curves.basis_size()), (n, n));
    assert!(!curves.on_nodes());
    let h = 3;
    let eta = some_eta(2, n, h);
    let a = 1.7f32;
    let mu = mu_of(&curves, &eta, h, a.ln());
    let prior = GridPrior::new(n, 1.0, 0.25).unwrap();
    for l in 0..2 {
        for d in 0..h {
            let e = DVector::from_fn(n, |k, _| f64::from(eta[(l * n + k) * h + d]));
            let z = prior.colour(&e);
            for k in 0..n {
                let got = mu[(l * n + k) * h + d];
                let want = f64::from(a) * z[k];
                assert!((f64::from(got) - want).abs() < 1e-4, "{l} {k} {d}");
            }
        }
    }
}

/// With a trunk every lineage is on lineage 0's curve up to its branch
/// time and leaves it after, by `a r ⊙ C η_ℓ`.
#[test]
fn with_a_trunk_the_lineages_agree_before_the_branch_and_part_after() {
    let tb = 0.4;
    let spec = CurveSpec {
        lineages: 3,
        points: 9,
        sub: 4,
        kappa: 0.25,
        branch: Some(vec![tb, 0.6]),
    };
    let curves = Curves::new(&spec, &Device::Cpu).unwrap();
    let (n, h) = (curves.positions(), 2);
    let eta = some_eta(3, n, h);
    let mu = mu_of(&curves, &eta, h, 0.3);
    let at = |l: usize, k: usize, d: usize| mu[(l * n + k) * h + d];
    for (l, tb) in [(1, 0.4), (2, 0.6)] {
        let mut apart = 0f32;
        for k in 0..n {
            let phase = k as f64 / (n - 1) as f64;
            let gap: f32 = (0..h).map(|d| (at(l, k, d) - at(0, k, d)).abs()).sum();
            if phase <= tb {
                assert!(gap < 1e-6, "lineage {l} off the trunk at {phase}");
            } else {
                apart = apart.max(gap);
            }
        }
        assert!(apart > 0.1, "lineage {l} never leaves the trunk");
    }
    // The departure is exactly `a r ⊙ C η_ℓ`.
    let prior = GridPrior::new(n, 1.0, 0.25).unwrap();
    let r = branch_ramp(n, tb, prior.kappa());
    let a = 0.3f64.exp();
    for d in 0..h {
        let e = DVector::from_fn(n, |k, _| f64::from(eta[(n + k) * h + d]));
        let z = prior.colour(&e);
        for k in 0..n {
            let want = a * r[k] * z[k];
            let got = f64::from(at(1, k, d) - at(0, k, d));
            assert!((got - want).abs() < 1e-4, "{k} {d}: {got} vs {want}");
        }
    }
}

/// Branch times at the grid's edges are pulled in, so a ramp never starts
/// on the last point.
#[test]
fn branch_times_are_clamped_off_the_edges() {
    let spec = CurveSpec {
        lineages: 2,
        points: 9,
        sub: 1,
        kappa: 0.25,
        branch: Some(vec![1.5]),
    };
    let curves = Curves::new(&spec, &Device::Cpu).unwrap();
    assert_eq!(curves.branch_times(), Some(&[0.98][..]));
}

/// On the nodes alone a curve is `a C η` at node `j` (position `j sub`)
/// and the straight line between two nodes at the positions between.
#[test]
fn on_the_nodes_a_curve_is_interpolated_linearly_between_them() {
    let spec = CurveSpec {
        lineages: 1,
        points: 9,
        sub: 4,
        kappa: 0.25,
        branch: None,
    };
    let curves = Curves::on_the_nodes(&spec, &Device::Cpu).unwrap();
    assert!(curves.on_nodes());
    assert_eq!((curves.positions(), curves.basis_size()), (33, 9));
    let h = 2;
    let eta = some_eta(1, 9, h);
    let mu = mu_of(&curves, &eta, h, 0.0);
    let prior = GridPrior::new(9, 1.0, 0.25).unwrap();
    for d in 0..h {
        let z = prior.colour(&DVector::from_fn(9, |j, _| f64::from(eta[j * h + d])));
        for k in 0..33 {
            let (j, f) = (k / 4, (k % 4) as f64 / 4.0);
            let want = if j == 8 {
                z[8]
            } else {
                (1.0 - f) * z[j] + f * z[j + 1]
            };
            assert!((f64::from(mu[k * h + d]) - want).abs() < 1e-4, "{k} {d}");
        }
    }
}

/// A finite difference along a known curve is its derivative in phase, to
/// second order everywhere, the ends too.
#[test]
fn the_grid_velocity_is_the_derivative_along_phase() {
    let (l, n, h) = (2, 177, 2);
    let tau = std::f64::consts::TAU;
    let curve = |l: usize, d: usize, t: f64| match (l, d) {
        (0, 0) => (tau * t).sin(),
        (0, _) => t * t,
        (_, 0) => (2.0 * t).exp(),
        _ => 3.0 * t - 1.0,
    };
    let slope = |l: usize, d: usize, t: f64| match (l, d) {
        (0, 0) => tau * (tau * t).cos(),
        (0, _) => 2.0 * t,
        (_, 0) => 2.0 * (2.0 * t).exp(),
        _ => 3.0,
    };
    let phase = |k: usize| k as f64 / (n - 1) as f64;
    let mut mu = vec![0f32; l * n * h];
    for li in 0..l {
        for k in 0..n {
            for d in 0..h {
                mu[(li * n + k) * h + d] = curve(li, d, phase(k)) as f32;
            }
        }
    }
    let v = grid_velocity(&mu, l, n, h);
    assert_eq!(v.len(), mu.len());
    for li in 0..l {
        for k in 0..n {
            for d in 0..h {
                let (got, want) = (f64::from(v[(li * n + k) * h + d]), slope(li, d, phase(k)));
                assert!(
                    (got - want).abs() < 2e-3 * want.abs().max(1.0),
                    "{li} {k} {d}: {got} vs {want}"
                );
            }
        }
    }
}

/// The ode run's default grid (45 nodes, 4 positions per step) has a
/// Cholesky factor on every position, so its curves need no interpolation.
#[test]
fn the_default_grid_puts_the_curves_on_every_position() {
    let spec = CurveSpec {
        lineages: 2,
        points: 45,
        sub: 4,
        kappa: 0.25,
        branch: Some(vec![0.3]),
    };
    let curves = Curves::new(&spec, &Device::Cpu).unwrap();
    assert_eq!(curves.positions(), 177);
    assert!(!curves.on_nodes());
}
