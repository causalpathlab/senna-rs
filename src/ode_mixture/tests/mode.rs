//! The readout at the climb's mode: every pseudobulk at
//! its most probable row, every gene's rates with their Laplace sds.

use super::init::{first, spearman};
use super::model::truth_params;
use crate::ode_mixture::readout::mode_snapshot;
use crate::ode_mixture::readout::{read_cells, Snapshot};
use crate::ode_mixture::refine::genes_at_mode;
use crate::ode_mixture::sim::{simulate, Sim, SimConfig};

const FINE: usize = 133;
const GRID: usize = 45;

/// Two lineages apart, with the unspliced reads' capture to fit.
fn sim() -> Sim {
    simulate(&SimConfig {
        lineages: 2,
        n_pb: 400,
        fine: FINE,
        seed: 5,
        ..SimConfig::default()
    })
    .unwrap()
}

/// The snapshot at the truth's parameters, lineages equally weighted.
fn truth_snapshot(sim: &Sim) -> Snapshot {
    let reads = first(sim, sim.n_pb);
    let params = truth_params(sim, FINE, GRID);
    let l = params.path.len();
    mode_snapshot(&reads, &params, &vec![-(l as f64).ln(); l], None)
}

/// At the truth, the mode's rows put every pseudobulk on its lineage (up
/// to the lineages' labels) in the truth's time order, and the cells read
/// out from the one snapshot.
#[test]
fn at_the_truth_the_mode_puts_every_pseudobulk_at_its_row() {
    let sim = sim();
    let snap = truth_snapshot(&sim);
    assert_eq!(snap.rows.len(), sim.n_pb);
    assert_eq!(snap.lambda.len(), sim.n_pb);
    let n = snap.params.positions();
    let lineage: Vec<usize> = snap.rows.iter().map(|r| r / n).collect();
    let time: Vec<f64> = snap
        .rows
        .iter()
        .map(|r| snap.params.time_at(r / n, r % n))
        .collect();
    let same = (0..sim.n_pb)
        .filter(|&p| lineage[p] == sim.truth.lineage[p])
        .count();
    let agree = same.max(sim.n_pb - same) as f64 / sim.n_pb as f64;
    let rho = spearman(&time, &sim.truth.time).abs();
    eprintln!("mode at the truth: lineage agreement {agree:.3}, time Spearman {rho:.3}");
    assert!(agree > 0.95, "lineage agreement {agree}");
    assert!(rho > 0.95, "time Spearman {rho}");

    let reads = first(&sim, sim.n_pb);
    let out = read_cells(&reads, &[snap]);
    assert!(out.time_mean.iter().all(|t| t.is_finite()));
    assert!(out.lineage.iter().all(|w| w.len() == 2));
}

/// At the truth's rows, every gene's Laplace approximation puts the truth's
/// `log γ` and `log γ/β` within two of its sds most of the time.
#[test]
fn at_the_truth_the_genes_laplace_sds_cover_their_rates() {
    let sim = sim();
    let snap = truth_snapshot(&sim);
    let reads = first(&sim, sim.n_pb);
    let rates = genes_at_mode(&reads, None, &snap);
    let g = sim.n_genes;
    assert_eq!(rates.log_gamma.len(), g);
    assert_eq!(rates.log_ratio.len(), g);
    let covered = |est: &[(f64, f64)], truth: &dyn Fn(usize) -> f64| -> f64 {
        (0..g)
            .filter(|&j| {
                let (m, s) = est[j];
                s.is_finite() && s > 0.0 && (truth(j) - m).abs() <= 2.0 * s
            })
            .count() as f64
            / g as f64
    };
    let t = &sim.truth;
    let gamma = covered(&rates.log_gamma, &|j| t.gamma[j].ln());
    let ratio = covered(&rates.log_ratio, &|j| (t.gamma[j] / t.beta[j]).ln());
    let sd = |est: &[(f64, f64)]| {
        let mut s: Vec<f64> = est.iter().map(|e| e.1).collect();
        s.sort_by(f64::total_cmp);
        s[g / 2]
    };
    eprintln!(
        "genes at the truth: log γ covered {gamma:.2} (median sd {:.3}), log γ/β covered {ratio:.2} (median sd {:.3})",
        sd(&rates.log_gamma),
        sd(&rates.log_ratio)
    );
    assert!(gamma > 0.8, "log γ covered {gamma}");
    assert!(ratio > 0.8, "log γ/β covered {ratio}");
}
