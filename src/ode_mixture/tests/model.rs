use crate::ode_mixture::model::{lineage_label, lineage_of, lineage_p_column, Params};
use crate::ode_mixture::sim::{simulate, Sim, SimConfig};

/// The simulator's truth as parameters on a grid of `n` points, the paths
/// read off the simulator's fine grid (which `n` must divide into).
pub(super) fn truth_params(sim: &Sim, fine: usize, n: usize) -> Params {
    let t = &sim.truth;
    let every = (fine - 1) / (n - 1);
    assert_eq!(every * (n - 1), fine - 1, "the grids align");
    Params {
        beta: t.beta.clone(),
        gamma: t.gamma.clone(),
        rho: t.rho.clone(),
        base: t.base.clone(),
        t_len: t.t_len.clone(),
        sub: 1,
        branch: None,
        path: t
            .path
            .iter()
            .map(|z| (0..n).map(|k| z[k * every].clone()).collect())
            .collect(),
    }
}

#[test]
fn the_levels_are_the_simulators_on_its_own_grid() {
    let cfg = SimConfig {
        lineages: 2,
        fine: 81,
        ..SimConfig::default()
    };
    let sim = simulate(&cfg).unwrap();
    let lv = truth_params(&sim, cfg.fine, cfg.fine).levels();
    assert_eq!((lv.lineages, lv.points, lv.genes), (2, 81, sim.n_genes));
    let step = cfg.t_len / 80.0;
    for p in [0, 5, 33, 120] {
        let (l, k) = (
            sim.truth.lineage[p],
            (sim.truth.time[p] / step).round() as usize,
        );
        let r = lv.row(l, k);
        let total: f64 = (0..sim.n_genes).map(|g| lv.u[(r, g)] + lv.s[(r, g)]).sum();
        let lambda = sim.depth[p] / total;
        for g in 0..sim.n_genes {
            let want = sim.truth.u_mean[p * sim.n_genes + g];
            assert!((lambda * lv.u[(r, g)] / want - 1.0).abs() < 1e-9);
        }
    }
}

/// With `sub = m`, the levels sit at `(n − 1) m + 1` positions per lineage,
/// every gene's the ODE's between the nodes, and a position's time is its
/// share of the duration.
#[test]
fn the_levels_sit_between_the_nodes_too() {
    use crate::ode_mixture::kinetics::fine_trajectory;
    let cfg = SimConfig {
        lineages: 2,
        fine: 81,
        ..SimConfig::default()
    };
    let sim = simulate(&cfg).unwrap();
    let mut params = truth_params(&sim, 81, 21);
    params.t_len[1] = 1.4;
    params.sub = 4;
    let lv = params.levels();
    assert_eq!((lv.lineages, lv.points), (2, 81));
    for g in [0, 9] {
        for l in 0..2 {
            let alpha: Vec<f64> = params.path[l]
                .iter()
                .map(|z| {
                    (z.iter()
                        .zip(&params.rho[g])
                        .map(|(a, b)| a * b)
                        .sum::<f64>()
                        + params.base[g])
                        .exp()
                })
                .collect();
            let (u, s) = fine_trajectory(
                &alpha,
                params.beta[g],
                params.gamma[g],
                params.t_len[l] / 20.0,
                4,
            );
            for k in 0..81 {
                assert!((lv.u[(lv.row(l, k), g)] / u[k] - 1.0).abs() < 1e-12);
                assert!((lv.s[(lv.row(l, k), g)] / s[k] - 1.0).abs() < 1e-12);
            }
        }
    }
    assert!((params.time_at(1, 40) - 0.7).abs() < 1e-12);
}

/// The levels' velocities, `dū/dt = α − β ū` and `ds̄/dt = β ū − γ s̄`
/// (`α` held over its step), are the trajectory's slope within a step.
#[test]
fn the_velocities_are_the_trajectorys_slope() {
    let sim = simulate(&SimConfig {
        fine: 133,
        ..SimConfig::default()
    })
    .unwrap();
    let mut params = truth_params(&sim, 133, 23);
    params.sub = 8;
    let lv = params.levels();
    let (du, ds) = params.velocities();
    assert_eq!((du.nrows(), du.ncols()), (lv.u.nrows(), lv.u.ncols()));
    let delta = params.t_len[0] / (lv.points - 1) as f64;
    let mut worst = 0f64;
    for k in (0..lv.points - 1).filter(|k| k % 8 >= 1 && k % 8 <= 6) {
        for g in 0..lv.genes {
            for (lev, vel) in [(&lv.u, &du), (&lv.s, &ds)] {
                let fd = (lev[(k + 1, g)] - lev[(k - 1, g)]) / (2.0 * delta);
                let scale = vel.column(g).abs().max().max(1e-9);
                worst = worst.max((fd - vel[(k, g)]).abs() / scale);
            }
        }
    }
    eprintln!("worst relative gap to the slope {worst:.2e}");
    assert!(worst < 1e-2, "{worst}");
}

#[test]
fn a_lineage_label_reads_back_as_its_lineage() {
    for l in 0..12 {
        assert_eq!(lineage_of(&lineage_label(l)), Some(l));
    }
    assert_eq!(lineage_p_column(1).as_ref(), "p_L2");
    assert_eq!(lineage_of("L0"), None);
    assert_eq!(lineage_of("T1"), None);
}
