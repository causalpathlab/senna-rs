//! The plan's gates on one lineage, from scratch: rates in the right unit,
//! time in order, and both survive a jitter of the unspliced share.

use super::init::{first, spearman};
use crate::ode_mixture::fit::Fit;
use crate::ode_mixture::init::{climb, Climb};
use crate::ode_mixture::sim::{simulate, Sim, SimConfig};
use legume_numeric::candle::candle_core::Device;
use std::sync::OnceLock;

const FINE: usize = 133;
const POINTS: usize = 45;

fn opts(cfg: &SimConfig) -> Climb {
    Climb {
        jitter: true,
        trunk: false,
        lineages: 1,
        points: POINTS,
        dim: cfg.dim,
        kappa: cfg.kappa,
        steps: 2500,
        lr: 0.03,
        sub: 1,
        roots: Vec::new(),
    }
}

/// The kinetics lineage's simulation.
fn config() -> SimConfig {
    SimConfig {
        fine: FINE,
        seed: 1,
        ..SimConfig::default()
    }
}

/// A climb from scratch on simulated reads.
struct Climbed {
    sim: Sim,
    fit: Fit,
    /// The fit's signed Spearman of time with the truth.
    spearman: f64,
}

fn climbed(cfg: &SimConfig) -> Climbed {
    let sim = simulate(cfg).unwrap();
    let reads = first(&sim, sim.n_pb);
    let climbed = climb(&reads, &opts(cfg), &Device::Cpu).unwrap();
    let (fit, objective) = (climbed.fit, climbed.objective);
    let (_, time) = fit.posterior(&fit.reads(&reads).unwrap()).unwrap();
    let spearman = spearman(&time, &sim.truth.time);
    eprintln!(
        "jitter {}: climbed {objective:.1} (signed Spearman {spearman:.3})",
        cfg.share_jitter
    );
    Climbed { sim, fit, spearman }
}

/// The kinetics lineage, shared by the gates.
fn kinetics() -> &'static Climbed {
    static CLIMBED: OnceLock<Climbed> = OnceLock::new();
    CLIMBED.get_or_init(|| climbed(&config()))
}

/// Per gene: how far off steady state the truth runs, the mean over
/// pseudobulks of `|log(s̄/ū) − log(β/γ)|` (zero at steady state), and its
/// mean unspliced reads per pseudobulk.
fn off_steady_and_reads(sim: &Sim) -> (Vec<f64>, Vec<f64>) {
    let (p_n, g_n, t) = (sim.n_pb, sim.n_genes, &sim.truth);
    (0..g_n)
        .map(|g| {
            let at_steady = (t.beta[g] / t.gamma[g]).ln();
            let off = (0..p_n)
                .map(|p| {
                    let i = p * g_n + g;
                    ((t.s_mean[i] / t.u_mean[i]).ln() - at_steady).abs()
                })
                .sum::<f64>()
                / p_n as f64;
            let reads = (0..p_n).map(|p| sim.u[p * g_n + g]).sum::<f64>() / p_n as f64;
            (off, reads)
        })
        .unzip()
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(f64::total_cmp);
    v[v.len() / 2]
}

/// Genes whose reads pin their rates: spliced over unspliced on average
/// more than 10% off its steady ratio `β/γ` (a gene at steady state shows
/// only `β/γ`, not the rates), and at least 5 unspliced reads per
/// pseudobulk (the unspliced track is the thinner).
fn pinned(sim: &Sim) -> Vec<usize> {
    let (off, reads) = off_steady_and_reads(sim);
    (0..sim.n_genes)
        .filter(|&g| off[g] > 0.1 && reads[g] >= 5.0)
        .collect()
}

/// `(median |log γ̂/γ|, median |log β̂/β|)` over `genes`.
fn rate_errors(c: &Climbed, genes: &[usize]) -> (f64, f64) {
    let got = c.fit.params().unwrap();
    let t = &c.sim.truth;
    let err = |fit: &[f64], truth: &[f64]| {
        median(
            genes
                .iter()
                .map(|&g| (fit[g] / truth[g]).ln().abs())
                .collect(),
        )
    };
    (err(&got.gamma, &t.gamma), err(&got.beta, &t.beta))
}

/// Lineage 0's duration is the unit and the simulator's is 1, so the
/// fitted rates are in the truth's unit with no map between them.
#[test]
#[cfg_attr(not(feature = "slow-tests"), ignore = "slow: --features slow-tests")]
fn from_scratch_the_rates_are_the_truths_where_the_reads_pin_them() {
    let c = kinetics();
    let genes = pinned(&c.sim);
    let all: Vec<usize> = (0..c.sim.n_genes).collect();
    let (gamma, beta) = rate_errors(c, &genes);
    let (gamma_all, beta_all) = rate_errors(c, &all);
    eprintln!(
        "{} pinned genes: median |log γ̂/γ| {gamma:.3}, |log β̂/β| {beta:.3}; every gene {gamma_all:.3}, {beta_all:.3}",
        genes.len()
    );
    assert!(genes.len() >= c.sim.n_genes / 4, "{} pinned", genes.len());
    assert!(gamma < 1.25f64.ln(), "median |log γ̂/γ| {gamma}");
}

/// With every unspliced read captured at `e^c` and each pseudobulk's share
/// jittering around it, the climb from scratch finds `c` and the jitter's
/// spread (not their sum), and the gates pass: neither is pushed into the
/// rates. (One case: a capture below 1 with a jitter covers both the
/// capture and the jitter the model carries.)
#[test]
#[cfg_attr(not(feature = "slow-tests"), ignore = "slow: --features slow-tests")]
fn the_climb_finds_a_global_capture_apart_from_the_jitter() {
    capture_gate(-1.2, 0.3, 1);
}

/// The climb from scratch on reads captured at `e^{c_true}` and jittering
/// by `j`: the capture, the jitter's sd, the rates and the times.
fn capture_gate(c_true: f64, j: f64, seed: u64) {
    let c = climbed(&SimConfig {
        log_capture: c_true,
        share_jitter: j,
        seed,
        ..config()
    });
    let found = c.fit.log_capture().unwrap();
    let sd = c.fit.share_sd().unwrap().unwrap();
    let (gamma, beta) = rate_errors(&c, &pinned(&c.sim));
    eprintln!(
            "capture {c_true}, jitter {j}: found {found:.3}, sd {sd:.3}; median |log γ̂/γ| {gamma:.3}, |log β̂/β| {beta:.3}"
        );
    assert!(
        (found - c_true).abs() < 0.15,
        "capture {c_true}: found {found}"
    );
    assert!(sd < j + 0.1, "jitter {j}: sd {sd}");
    assert!(gamma < 1.25f64.ln(), "median |log γ̂/γ| {gamma}");
    assert!(beta < 1.25f64.ln(), "median |log β̂/β| {beta}");
    assert!(c.spearman.abs() > 0.95, "Spearman {}", c.spearman);
}
