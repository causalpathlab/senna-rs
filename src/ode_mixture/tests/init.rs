use super::model::truth_params;
use crate::ode_mixture::fit::Fit;
use crate::ode_mixture::init::{climb, start_order, Climb};
use crate::ode_mixture::likelihood::Reads;
use crate::ode_mixture::sim::{simulate, Sim, SimConfig};
use legume_numeric::candle::candle_core::Device;

/// Spearman's rank correlation.
pub(super) fn spearman(a: &[f64], b: &[f64]) -> f64 {
    let rank = |v: &[f64]| {
        let mut idx: Vec<usize> = (0..v.len()).collect();
        idx.sort_by(|&i, &j| v[i].total_cmp(&v[j]));
        let mut r = vec![0.0; v.len()];
        for (k, &i) in idx.iter().enumerate() {
            r[i] = k as f64;
        }
        r
    };
    let (ra, rb) = (rank(a), rank(b));
    let n = a.len() as f64;
    let m = (n - 1.0) / 2.0;
    let (mut c, mut va, mut vb) = (0.0, 0.0, 0.0);
    for (x, y) in ra.iter().zip(&rb) {
        c += (x - m) * (y - m);
        va += (x - m).powi(2);
        vb += (y - m).powi(2);
    }
    c / (va * vb).sqrt()
}

/// The first `n` pseudobulks' reads.
pub(super) fn first(sim: &Sim, n: usize) -> Reads {
    let g = sim.n_genes;
    Reads::new(&sim.u[..n * g], &sim.s[..n * g], n, g)
}

#[test]
fn the_start_orders_pseudobulks_along_the_trajectory() {
    let sim = simulate(&SimConfig::default()).unwrap();
    let reads = first(&sim, sim.n_pb);
    let (lineage, order) = start_order(&reads, 1, 3);
    assert!(lineage.iter().all(|&l| l == 0));
    let r = spearman(&order, &sim.truth.time);
    // Rough: a curved trajectory is not straight along its first axis; the
    // climb straightens it.
    assert!(r.abs() > 0.8, "start Spearman {r}");
}

#[test]
fn from_scratch_the_climb_finds_time_and_its_direction() {
    let cfg = SimConfig {
        fine: 133,
        ..SimConfig::default()
    };
    let sim = simulate(&cfg).unwrap();
    let n_train = 240;
    let train = first(&sim, n_train);
    let dev = Device::Cpu;
    let opts = Climb {
        lineages: 1,
        points: 45,
        dim: 3,
        kappa: cfg.kappa,
        steps: 2500,
        lr: 0.03,
    };
    let (fit, objective) = climb(&train, &opts, &dev).unwrap();
    let at_truth = {
        let prior = crate::ode_mixture::kernel::GridPrior::new(45, 1.0, cfg.kappa).unwrap();
        let f = Fit::from_params(&truth_params(&sim, 133, 45), &prior, &dev).unwrap();
        f.objective(&f.reads(&train).unwrap())
            .unwrap()
            .to_scalar::<f32>()
            .unwrap()
    };
    let all = first(&sim, sim.n_pb);
    let (_, time) = fit.posterior(&fit.reads(&all).unwrap()).unwrap();
    let held: Vec<usize> = (n_train..sim.n_pb).collect();
    let r = spearman(
        &held.iter().map(|&p| time[p]).collect::<Vec<_>>(),
        &held.iter().map(|&p| sim.truth.time[p]).collect::<Vec<_>>(),
    );
    eprintln!("objective {objective:.1} (truth {at_truth:.1}); held-out signed Spearman {r:.3}");
    assert!(r > 0.95, "held-out Spearman {r}");
    assert!(f64::from(objective) > f64::from(at_truth) - 50.0);
}

/// Two lineages from scratch: each true lineage in a fitted one of its
/// own, and time right within each.
#[test]
fn from_scratch_two_lineages_come_apart() {
    let cfg = SimConfig {
        lineages: 2,
        n_pb: 600,
        fine: 133,
        ..SimConfig::default()
    };
    let sim = simulate(&cfg).unwrap();
    let n_train = 480;
    let train = first(&sim, n_train);
    let dev = Device::Cpu;
    let opts = Climb {
        lineages: 2,
        points: 45,
        dim: 3,
        kappa: cfg.kappa,
        steps: 2500,
        lr: 0.03,
    };
    let (fit, objective) = climb(&train, &opts, &dev).unwrap();
    let all = first(&sim, sim.n_pb);
    let (weights, time) = fit.posterior(&fit.reads(&all).unwrap()).unwrap();
    let held: Vec<usize> = (n_train..sim.n_pb).collect();
    let fitted: Vec<usize> = held
        .iter()
        .map(|&p| if weights[p][0] > weights[p][1] { 0 } else { 1 })
        .collect();
    let mut table = [[0usize; 2]; 2];
    for (i, &p) in held.iter().enumerate() {
        table[sim.truth.lineage[p]][fitted[i]] += 1;
    }
    let purity =
        (table[0][0] + table[1][1]).max(table[0][1] + table[1][0]) as f64 / held.len() as f64;
    let mut per = Vec::new();
    for lin in 0..2 {
        let rows: Vec<usize> = held
            .iter()
            .copied()
            .filter(|&p| sim.truth.lineage[p] == lin)
            .collect();
        per.push(spearman(
            &rows.iter().map(|&p| time[p]).collect::<Vec<_>>(),
            &rows.iter().map(|&p| sim.truth.time[p]).collect::<Vec<_>>(),
        ));
    }
    eprintln!("objective {objective:.1}; table {table:?}; purity {purity:.3}; per-lineage signed Spearman {per:?}");
    assert!(purity > 0.95, "purity {purity}");
    assert!(
        per.iter().all(|&r| r > 0.95),
        "per-lineage Spearman {per:?}"
    );
}
