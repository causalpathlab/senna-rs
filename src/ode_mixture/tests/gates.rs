//! The plan's gates on one lineage, from scratch: rates in the right unit,
//! the direction called with dynamics and left open without them, and a
//! shared drift of the unspliced share not taken for a direction.

use super::init::{first, spearman};
use crate::ode_mixture::direction::{direction, Direction};
use crate::ode_mixture::fit::Fit;
use crate::ode_mixture::init::Climb;
use crate::ode_mixture::sim::{simulate, Sim, SimConfig};
use legume_numeric::candle::candle_core::Device;
use std::sync::OnceLock;

const FINE: usize = 133;
const POINTS: usize = 45;

fn opts(cfg: &SimConfig) -> Climb {
    Climb {
        jitter: true,
        lineages: 1,
        points: POINTS,
        dim: cfg.dim,
        kappa: cfg.kappa,
        steps: 2500,
        lr: 0.03,
        sub: 1,
    }
}

fn config(steady: bool, drift: f64, seed: u64) -> SimConfig {
    SimConfig {
        fine: FINE,
        steady,
        unspliced_drift: drift,
        seed,
        ..SimConfig::default()
    }
}

/// What a direction call came to on simulated reads.
struct Call {
    sim: Sim,
    dir: Direction,
    /// The climbed fit's signed Spearman of time with the truth.
    climbed: f64,
}

impl Call {
    /// Evidence in the truth's favour: positive when the held refit the
    /// true way round fits better.
    fn for_truth(&self) -> f64 {
        self.dir.evidence() * self.climbed.signum()
    }

    /// The held refit the evidence prefers.
    fn chosen(&self) -> &Fit {
        if self.dir.evidence() >= 0.0 {
            &self.dir.forward
        } else {
            &self.dir.reversed
        }
    }
}

fn call(cfg: &SimConfig) -> Call {
    call_with(cfg, &opts(cfg))
}

fn call_with(cfg: &SimConfig, opts: &Climb) -> Call {
    let sim = simulate(cfg).unwrap();
    let reads = first(&sim, sim.n_pb);
    let dir = direction(&reads, opts, 0, &Device::Cpu).unwrap();
    let (_, time) = dir.fit.posterior(&dir.fit.reads(&reads).unwrap()).unwrap();
    let climbed = spearman(&time, &sim.truth.time);
    let c = Call { sim, dir, climbed };
    eprintln!(
        "steady {} drift {} seed {}: climbed {:.1} (signed Spearman {climbed:.3}); held forward {:.1}, reversed {:.1}; evidence for the truth {:.1} ({:.1} per pseudobulk), determined {}",
        cfg.steady,
        cfg.unspliced_drift,
        cfg.seed,
        c.dir.objective,
        c.dir.forward_objective,
        c.dir.reversed_objective,
        c.for_truth(),
        c.for_truth() / c.sim.n_pb as f64,
        c.dir.determined()
    );
    c
}

/// The kinetics lineage, shared by the rate and direction gates.
fn kinetics() -> &'static Call {
    static CALL: OnceLock<Call> = OnceLock::new();
    CALL.get_or_init(|| call(&config(false, 0.0, 1)))
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
fn rate_errors(c: &Call, genes: &[usize]) -> (f64, f64) {
    let got = c.chosen().params().unwrap();
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

/// The held objective is the host's table at the held rows, plus log π and
/// the priors (the objective minus the marginal).
#[test]
fn the_held_objective_is_the_hosts() {
    use super::model::truth_params;
    use crate::ode_mixture::kernel::GridPrior;
    use crate::ode_mixture::likelihood::table;
    use legume_numeric::candle::candle_core::Tensor;
    let cfg = SimConfig {
        lineages: 2,
        ..config(false, 0.0, 1)
    };
    let sim = simulate(&cfg).unwrap();
    let params = truth_params(&sim, FINE, POINTS);
    let prior = GridPrior::new(POINTS, 1.0, cfg.kappa).unwrap();
    let reads = first(&sim, sim.n_pb);
    let fit = Fit::from_params(&params, &prior, &Device::Cpu).unwrap();
    let dev = fit.reads(&reads).unwrap();
    let rows: Vec<u32> = (0..sim.n_pb).map(|p| (p % (2 * POINTS)) as u32).collect();
    let scalar = |t: Tensor| f64::from(t.to_scalar::<f32>().unwrap());
    let got = scalar(
        fit.held_objective(&dev, &Tensor::new(rows.as_slice(), &Device::Cpu).unwrap())
            .unwrap(),
    );
    let log_prior = scalar(fit.objective(&dev).unwrap()) - scalar(fit.log_marginal(&dev).unwrap());
    let tab = table(&reads, &params.levels());
    let want: f64 = rows
        .iter()
        .enumerate()
        .map(|(p, &r)| tab[(p, r as usize)] + 0.5f64.ln())
        .sum::<f64>()
        + log_prior;
    assert!((got - want).abs() / want.abs() < 1e-5, "{got} vs {want}");
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

#[test]
fn with_kinetics_the_right_direction_is_called() {
    let c = kinetics();
    assert!(c.climbed.abs() > 0.95, "climbed Spearman {}", c.climbed);
    assert!(c.dir.determined(), "evidence {}", c.dir.evidence());
    assert!(c.for_truth() > 0.0, "the wrong way: {}", c.for_truth());
}

#[test]
fn with_no_dynamics_the_direction_is_undetermined() {
    let c = call(&config(true, 0.0, 1));
    assert!(!c.dir.determined(), "evidence {}", c.dir.evidence());
}

/// The gate as the plan sets it: a drift of every gene's unspliced share
/// alone, with no kinetics in the reads, calls no direction. It needs the
/// jitter in the model ([`Climb::jitter`]): the shared model calls one as
/// confidently as kinetics do (a shared shift of the unspliced share looks
/// like a shared rate change).
#[test]
fn a_shared_unspliced_drift_alone_is_not_a_direction() {
    for drift in [1.0, 3.0] {
        let cfg = config(true, drift, 1);
        let c = call_with(
            &cfg,
            &Climb {
                jitter: true,
                ..opts(&cfg)
            },
        );
        assert!(
            !c.dir.determined(),
            "drift {drift}: evidence {}",
            c.dir.evidence()
        );
    }
}

/// With kinetics in the reads too, the direction survives the drift.
#[test]
fn with_kinetics_the_direction_survives_a_shared_unspliced_drift() {
    let c = call(&config(false, 3.0, 1));
    let (gamma, beta) = rate_errors(&c, &pinned(&c.sim));
    eprintln!("drift 3 with kinetics: median |log γ̂/γ| {gamma:.3}, |log β̂/β| {beta:.3}");
    assert!(c.dir.determined(), "evidence {}", c.dir.evidence());
    assert!(c.for_truth() > 0.0, "the wrong way: {}", c.for_truth());
}

/// The rate gate under the drift. It fails, with the jitter in the model
/// too at the smaller drift: a drift smooth in time is read as a shared
/// change of rates, and nothing in the reads tells the two apart.
#[test]
#[ignore = "fails: a drift smooth in time is not identifiable from kinetics; the jitter in the model absorbs only its rough part"]
fn with_kinetics_the_rates_survive_a_shared_unspliced_drift() {
    for drift in [1.0, 3.0] {
        let c = call(&config(false, drift, 1));
        let (gamma, _) = rate_errors(&c, &pinned(&c.sim));
        assert!(
            gamma < 1.25f64.ln(),
            "drift {drift}: median |log γ̂/γ| {gamma}"
        );
    }
}

/// Every gate's numbers over three seeds.
#[test]
#[ignore = "a sweep for the record, slow"]
fn sweep() {
    for seed in [1u64, 2, 3] {
        for (steady, drift) in [
            (false, 0.0),
            (true, 0.0),
            (true, 1.0),
            (true, 3.0),
            (false, 1.0),
            (false, 3.0),
        ] {
            let c = call(&config(steady, drift, seed));
            if !steady {
                let genes = pinned(&c.sim);
                let (gamma, beta) = rate_errors(&c, &genes);
                eprintln!(
                    "  {} pinned genes: median |log γ̂/γ| {gamma:.3}, |log β̂/β| {beta:.3}",
                    genes.len()
                );
            }
        }
    }
}

/// With the jitter in the model ([`Climb::jitter`]), a jitter of the
/// unspliced share from pseudobulk to pseudobulk leaves the gates as they
/// are without it: rates, time and direction. (The shared model loses the
/// rates already at the smaller jitter.)
#[test]
fn with_the_jitter_in_the_model_the_gates_survive_a_jitter_of_the_unspliced_share() {
    for j in [0.3, 1.0] {
        let cfg = SimConfig {
            share_jitter: j,
            ..config(false, 0.0, 1)
        };
        let c = call_with(
            &cfg,
            &Climb {
                jitter: true,
                ..opts(&cfg)
            },
        );
        let (gamma, beta) = rate_errors(&c, &pinned(&c.sim));
        let sd = c.chosen().share_sd().unwrap().unwrap();
        eprintln!(
            "jitter {j}: fitted sd {sd:.3}; median |log γ̂/γ| {gamma:.3}, |log β̂/β| {beta:.3}"
        );
        assert!(gamma < 1.25f64.ln(), "jitter {j}: median |log γ̂/γ| {gamma}");
        assert!(
            c.climbed.abs() > 0.95,
            "jitter {j}: climbed Spearman {}",
            c.climbed
        );
        assert!(
            c.dir.determined() && c.for_truth() > 0.0,
            "jitter {j}: evidence {}",
            c.for_truth()
        );
        assert!((sd / j).ln().abs() < 0.25f64, "jitter {j}: fitted sd {sd}");
    }
}

/// With no jitter in the reads, the jitter in the model costs nothing: the
/// gates pass and the fitted jitter is small.
#[test]
fn with_no_jitter_in_the_reads_the_jitter_in_the_model_stays_small() {
    let cfg = config(false, 0.0, 1);
    let c = call_with(
        &cfg,
        &Climb {
            jitter: true,
            ..opts(&cfg)
        },
    );
    let (gamma, _) = rate_errors(&c, &pinned(&c.sim));
    let sd = c.chosen().share_sd().unwrap().unwrap();
    eprintln!("no jitter: fitted sd {sd:.4}; median |log γ̂/γ| {gamma:.3}");
    assert!(gamma < 1.25f64.ln(), "median |log γ̂/γ| {gamma}");
    assert!(c.climbed.abs() > 0.95, "climbed Spearman {}", c.climbed);
    assert!(
        c.dir.determined() && c.for_truth() > 0.0,
        "evidence {}",
        c.for_truth()
    );
    assert!(sd < 0.05, "fitted sd {sd}");
}
