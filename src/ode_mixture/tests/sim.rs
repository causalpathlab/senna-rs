use crate::ode_mixture::sim::{simulate, SimConfig};

fn ratio_spread(sim: &crate::ode_mixture::sim::Sim, g: usize) -> f64 {
    let t = &sim.truth;
    let steady = t.gamma[g] / t.beta[g];
    (0..sim.n_pb)
        .map(|p| {
            let r = t.u_mean[p * sim.n_genes + g] / t.s_mean[p * sim.n_genes + g];
            (r / steady).ln().abs()
        })
        .fold(0.0, f64::max)
}

#[test]
fn every_pseudobulk_has_a_lineage_a_time_and_its_depth() {
    let cfg = SimConfig {
        lineages: 2,
        ..SimConfig::default()
    };
    let sim = simulate(&cfg).unwrap();
    assert_eq!(sim.u.len(), cfg.n_pb * cfg.n_genes);
    assert_eq!(sim.s.len(), cfg.n_pb * cfg.n_genes);
    for p in 0..cfg.n_pb {
        let l = sim.truth.lineage[p];
        assert!(l < 2);
        let t = sim.truth.time[p];
        assert!((0.0..=sim.truth.t_len[l]).contains(&t));
        let row = p * cfg.n_genes..(p + 1) * cfg.n_genes;
        let total: f64 = sim.truth.u_mean[row.clone()].iter().sum::<f64>()
            + sim.truth.s_mean[row].iter().sum::<f64>();
        assert!((total / sim.depth[p] - 1.0).abs() < 1e-9);
    }
    assert!((0..cfg.n_pb).any(|p| sim.truth.lineage[p] == 1));
}

#[test]
fn the_counts_are_drawn_at_their_means() {
    let sim = simulate(&SimConfig::default()).unwrap();
    let (u, um): (f64, f64) = (sim.u.iter().sum(), sim.truth.u_mean.iter().sum());
    let (s, sm): (f64, f64) = (sim.s.iter().sum(), sim.truth.s_mean.iter().sum());
    assert!((u / um - 1.0).abs() < 0.01 && (s / sm - 1.0).abs() < 0.01);
    assert!(sim
        .u
        .iter()
        .chain(&sim.s)
        .all(|x| x.fract() == 0.0 && *x >= 0.0));
}

#[test]
fn at_steady_state_every_gene_keeps_its_ratio() {
    let sim = simulate(&SimConfig {
        steady: true,
        ..SimConfig::default()
    })
    .unwrap();
    for g in 0..sim.n_genes {
        assert!(ratio_spread(&sim, g) < 1e-9, "gene {g}");
    }
}

#[test]
fn with_kinetics_the_ratios_move_off_steady_state() {
    let sim = simulate(&SimConfig::default()).unwrap();
    let moved = (0..sim.n_genes)
        .filter(|&g| ratio_spread(&sim, g) > 0.1)
        .count();
    assert!(moved * 2 > sim.n_genes, "only {moved} genes move");
}

#[test]
fn a_drift_shifts_every_genes_unspliced_share_alike() {
    let sim = simulate(&SimConfig {
        steady: true,
        unspliced_drift: 1.0,
        ..SimConfig::default()
    })
    .unwrap();
    let t = &sim.truth;
    for p in 0..sim.n_pb {
        let shift = t.time[p] / t.t_len[t.lineage[p]] - 0.5;
        for g in [0, 7, 42] {
            let r = t.u_mean[p * sim.n_genes + g] / t.s_mean[p * sim.n_genes + g];
            assert!(((r * t.beta[g] / t.gamma[g]).ln() - shift).abs() < 1e-9);
        }
    }
}

#[test]
fn a_seed_replays() {
    let (a, b) = (
        simulate(&SimConfig::default()).unwrap(),
        simulate(&SimConfig::default()).unwrap(),
    );
    assert_eq!(a.u, b.u);
    assert_eq!(a.s, b.s);
}

#[test]
fn the_truth_rebuilds_the_means() {
    use crate::ode_mixture::kinetics::trajectory;
    let cfg = SimConfig::default();
    let sim = simulate(&cfg).unwrap();
    let t = &sim.truth;
    let step = cfg.t_len / (cfg.fine - 1) as f64;
    let p = 17;
    let (lin, k) = (t.lineage[p], (t.time[p] / step).round() as usize);
    let levels: Vec<(f64, f64)> = (0..sim.n_genes)
        .map(|g| {
            let alpha: Vec<f64> = t.path[lin]
                .iter()
                .map(|z| {
                    (z.iter().zip(&t.rho[g]).map(|(a, b)| a * b).sum::<f64>() + t.base[g]).exp()
                })
                .collect();
            let (u, s) = trajectory(&alpha, t.beta[g], t.gamma[g], step);
            (u[k], s[k])
        })
        .collect();
    let total: f64 = levels.iter().map(|(u, s)| u + s).sum();
    for (g, (u, s)) in levels.iter().enumerate() {
        let lambda = sim.depth[p] / total;
        assert!((lambda * u / t.u_mean[p * sim.n_genes + g] - 1.0).abs() < 1e-9);
        assert!((lambda * s / t.s_mean[p * sim.n_genes + g] - 1.0).abs() < 1e-9);
    }
}
