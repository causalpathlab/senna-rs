//! The readout at the climb's mode: every pseudobulk at its most probable
//! row, every gene's rates with their Laplace sds.

use super::init::{first, spearman};
use super::model::{coverage, truth_params, uniform_log_w};
use crate::ode_mixture::readout::{grid_posterior, place, read_cells, Mode, Placed};
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

/// The truth as the mode, lineages equally weighted, and every pseudobulk
/// placed at it.
fn truth_mode(sim: &Sim) -> (Mode, Placed) {
    let reads = first(sim, sim.n_pb);
    let params = truth_params(sim, FINE, GRID);
    let mode = Mode {
        log_w: uniform_log_w(params.path.len()),
        params,
        share_sd: None,
    };
    let placed = place(&reads, &mode, &grid_posterior(&reads, &mode));
    (mode, placed)
}

/// At the truth, the mode's rows put every pseudobulk on its lineage (up
/// to the lineages' labels) in the truth's time order, and the cells read
/// out from it.
#[test]
fn at_the_truth_the_mode_puts_every_pseudobulk_at_its_row() {
    let sim = sim();
    let (mode, placed) = truth_mode(&sim);
    assert_eq!(placed.rows.len(), sim.n_pb);
    assert_eq!(placed.lambda.len(), sim.n_pb);
    let rows: Vec<usize> = placed.rows.iter().map(|r| r.expect("placed")).collect();
    let n = mode.params.positions();
    let lineage: Vec<usize> = rows.iter().map(|r| r / n).collect();
    let time: Vec<f64> = rows.iter().map(|r| mode.params.phase_at(r % n)).collect();
    let same = (0..sim.n_pb)
        .filter(|&p| lineage[p] == sim.truth.lineage[p])
        .count();
    let agree = same.max(sim.n_pb - same) as f64 / sim.n_pb as f64;
    let rho = spearman(&time, &sim.truth.time).abs();
    eprintln!("mode at the truth: lineage agreement {agree:.3}, time Spearman {rho:.3}");
    assert!(agree > 0.95, "lineage agreement {agree}");
    assert!(rho > 0.95, "time Spearman {rho}");
    assert!(placed.lambda.iter().all(|l| l[0] > 0.0 && l[1] > 0.0));

    let reads = first(&sim, sim.n_pb);
    let out = read_cells(&reads, &mode);
    assert!(out.time_mean.iter().all(|t| t.is_finite()));
    assert!(out.lineage.iter().all(|w| w.len() == 2));
}

/// At the truth's rows, every gene's Laplace approximation puts the truth's
/// `log γ` and `log γ/β` within two of its sds most of the time.
#[test]
fn at_the_truth_the_genes_laplace_sds_cover_their_rates() {
    let sim = sim();
    let (mode, placed) = truth_mode(&sim);
    let reads = first(&sim, sim.n_pb);
    let rates = genes_at_mode(&reads, None, &mode, &placed);
    let g = sim.n_genes;
    assert_eq!(rates.log_gamma.len(), g);
    assert_eq!(rates.log_ratio.len(), g);
    let t = &sim.truth;
    let gamma = coverage(&rates.log_gamma, |j| t.gamma[j].ln(), 0.0);
    let ratio = coverage(&rates.log_ratio, |j| (t.gamma[j] / t.beta[j]).ln(), 0.0);
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

/// With the jitter in the model, every pseudobulk's depths at the mode
/// carry its own jitter: the unspliced depth differs from the spliced one
/// by the jitter's factor, not fixed to one.
#[test]
fn with_a_jitter_the_depths_carry_each_pseudobulks_own() {
    let sim = simulate(&SimConfig {
        lineages: 2,
        n_pb: 200,
        fine: FINE,
        share_jitter: 0.4,
        seed: 6,
        ..SimConfig::default()
    })
    .unwrap();
    let reads = first(&sim, sim.n_pb);
    let params = truth_params(&sim, FINE, GRID);
    let mode = Mode {
        log_w: uniform_log_w(2),
        params,
        share_sd: Some(0.4),
    };
    let placed = place(&reads, &mode, &grid_posterior(&reads, &mode));
    let ratios: Vec<f64> = placed.lambda.iter().map(|l| (l[0] / l[1]).ln()).collect();
    let spread = {
        let m = ratios.iter().sum::<f64>() / ratios.len() as f64;
        (ratios.iter().map(|r| (r - m).powi(2)).sum::<f64>() / ratios.len() as f64).sqrt()
    };
    eprintln!(
        "jitter at the mode: sd of log(λ_u/λ_s) over pseudobulks {spread:.3} (simulated 0.4)"
    );
    assert!(spread > 0.15, "the depths carry no jitter: spread {spread}");
}
