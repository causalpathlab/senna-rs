use crate::ode_mixture::model::Params;
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
