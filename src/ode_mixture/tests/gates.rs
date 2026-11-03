//! The plan's gates on one lineage, from scratch: rates in the right unit,
//! the direction called with dynamics and left open without them, and a
//! shared drift of the unspliced share not taken for a direction.

use super::init::{first, spearman};
use crate::ode_mixture::direction::{orientations, Orientations};
use crate::ode_mixture::init::Climb;
use crate::ode_mixture::sim::{simulate, Sim, SimConfig};
use legume_numeric::candle::candle_core::Device;

const FINE: usize = 133;
const POINTS: usize = 45;

fn opts(cfg: &SimConfig) -> Climb {
    Climb {
        lineages: 1,
        points: POINTS,
        dim: cfg.dim,
        kappa: cfg.kappa,
        steps: 2500,
        lr: 0.03,
    }
}

/// Both orientations of lineage 0 climbed from the start on every
/// pseudobulk, and the better one's signed Spearman of time with the truth.
fn both_ways(cfg: &SimConfig) -> (Sim, Orientations, f64) {
    let sim = simulate(cfg).unwrap();
    let reads = first(&sim, sim.n_pb);
    let dev = Device::Cpu;
    let both = orientations(&reads, &opts(cfg), 0, &dev).unwrap();
    let fit = both.better();
    let (_, time) = fit.posterior(&fit.reads(&reads).unwrap()).unwrap();
    let r = spearman(&time, &sim.truth.time);
    (sim, both, r)
}

fn config(steady: bool, drift: f64) -> SimConfig {
    SimConfig {
        fine: FINE,
        steady,
        unspliced_drift: drift,
        ..SimConfig::default()
    }
}

#[test]
fn with_kinetics_the_right_orientation_wins_clearly() {
    let (_, both, r) = both_ways(&config(false, 0.0));
    eprintln!(
        "kinetics: forward {:.1}, reversed {:.1}, evidence {:.1}; better's signed Spearman {r:.3}",
        both.forward.objective,
        both.reversed.objective,
        both.evidence()
    );
    assert!(r > 0.95, "the better orientation runs backwards: {r}");
    assert!(both.evidence() > 100.0, "evidence {}", both.evidence());
}
