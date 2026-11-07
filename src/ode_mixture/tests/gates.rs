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
    let (fit, objective) = climb(&reads, &opts(cfg), &Device::Cpu).unwrap();
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

/// With the jitter in the model ([`Climb::jitter`]), a jitter of the
/// unspliced share from pseudobulk to pseudobulk leaves the gates as they
/// are without it: rates and time. (The shared model loses the rates
/// already at the smaller jitter.)
#[test]
fn with_the_jitter_in_the_model_the_gates_survive_a_jitter_of_the_unspliced_share() {
    for j in [0.3, 1.0] {
        let cfg = SimConfig {
            share_jitter: j,
            ..config()
        };
        let c = climbed(&cfg);
        let (gamma, beta) = rate_errors(&c, &pinned(&c.sim));
        let sd = c.fit.share_sd().unwrap().unwrap();
        eprintln!(
            "jitter {j}: fitted sd {sd:.3}; median |log γ̂/γ| {gamma:.3}, |log β̂/β| {beta:.3}"
        );
        assert!(gamma < 1.25f64.ln(), "jitter {j}: median |log γ̂/γ| {gamma}");
        assert!(
            c.spearman.abs() > 0.95,
            "jitter {j}: climbed Spearman {}",
            c.spearman
        );
        assert!((sd / j).ln().abs() < 0.25f64, "jitter {j}: fitted sd {sd}");
    }
}

/// With no jitter in the reads, the jitter in the model costs nothing: the
/// gates pass and the fitted jitter is small.
#[test]
fn with_no_jitter_in_the_reads_the_jitter_in_the_model_stays_small() {
    let c = kinetics();
    let (gamma, _) = rate_errors(c, &pinned(&c.sim));
    let sd = c.fit.share_sd().unwrap().unwrap();
    eprintln!("no jitter: fitted sd {sd:.4}; median |log γ̂/γ| {gamma:.3}");
    assert!(gamma < 1.25f64.ln(), "median |log γ̂/γ| {gamma}");
    assert!(c.spearman.abs() > 0.95, "climbed Spearman {}", c.spearman);
    assert!(sd < 0.05, "fitted sd {sd}");
}
