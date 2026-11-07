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

/// A jitter moves every pseudobulk's unspliced share by its own `ω_p`,
/// every gene's alike: at steady state `log(ū β/(s̄ γ)) = ω_p` in every
/// gene, those amounts spread as the jitter.
#[test]
fn a_jitter_moves_every_pseudobulks_unspliced_share_its_own_way() {
    let sim = simulate(&SimConfig {
        steady: true,
        share_jitter: 0.5,
        ..SimConfig::default()
    })
    .unwrap();
    let (g_n, t) = (sim.n_genes, &sim.truth);
    let shift = |p: usize, g: usize| {
        let i = p * g_n + g;
        (t.u_mean[i] * t.beta[g] / (t.s_mean[i] * t.gamma[g])).ln()
    };
    let omega: Vec<f64> = (0..sim.n_pb).map(|p| shift(p, 0)).collect();
    for (p, &w) in omega.iter().enumerate() {
        for g in [7, 42] {
            assert!((shift(p, g) - w).abs() < 1e-9, "{p} {g}");
        }
    }
    let sd = (omega.iter().map(|w| w * w).sum::<f64>() / omega.len() as f64).sqrt();
    assert!((sd - 0.5).abs() < 0.08, "sd {sd}");
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

/// Lineages on a shared trunk: lineage 0's path until the branch, then
/// apart, value and slope continuous at the branch; the truth says which
/// pseudobulks are on the trunk.
#[test]
fn branching_lineages_share_a_trunk_and_part_smoothly() {
    let cfg = SimConfig {
        lineages: 3,
        branch_at: Some(0.4),
        ..SimConfig::default()
    };
    let sim = simulate(&cfg).unwrap();
    let t = &sim.truth;
    let step = cfg.t_len / (cfg.fine - 1) as f64;
    let kb = (0..cfg.fine)
        .find(|&k| k as f64 * step > 0.4 * cfg.t_len)
        .unwrap();
    for l in 1..3 {
        let gap = |k: usize| -> f64 {
            t.path[l][k]
                .iter()
                .zip(&t.path[0][k])
                .map(|(a, b)| (a - b).powi(2))
                .sum::<f64>()
                .sqrt()
        };
        // Equal on the trunk.
        assert!((0..kb).all(|k| gap(k) == 0.0), "lineage {l} off the trunk");
        // No kink: the gap grows slower than linearly off the branch.
        let (g1, g4) = (gap(kb), gap(kb + 3));
        assert!(g1 < 1e-3 && g4 < 1e-2, "lineage {l} kinks: {g1} {g4}");
        // Apart after.
        let far = (kb..cfg.fine).map(gap).fold(0.0, f64::max);
        assert!(far > 0.5, "lineage {l} never parts: {far}");
    }
    for p in 0..sim.n_pb {
        assert_eq!(t.trunk[p], t.time[p] <= 0.4 * cfg.t_len, "pseudobulk {p}");
    }
    assert!(t.trunk.iter().any(|&b| b) && t.trunk.iter().any(|&b| !b));
}

#[test]
fn without_a_branch_no_pseudobulk_is_on_a_trunk() {
    let sim = simulate(&SimConfig {
        lineages: 2,
        ..SimConfig::default()
    })
    .unwrap();
    assert!(sim.truth.trunk.iter().all(|&b| !b));
}
