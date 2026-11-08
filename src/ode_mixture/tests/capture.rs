//! One global capture of the unspliced reads, `e^c` on every unspliced
//! level, each pseudobulk's jitter around it: the chains find it (the
//! climb's gate is in `gates.rs`).

use super::init::first;
use super::model::truth_params;
use crate::ode_mixture::init::{climb, Climb};
use crate::ode_mixture::model::Params;
use crate::ode_mixture::sampler::{run, Options};
use crate::ode_mixture::sim::{simulate, Sim, SimConfig};
use legume_numeric::candle::candle_core::Device;
use std::sync::OnceLock;

const POINTS: usize = 45;
const TRAIN: usize = 240;
const KAPPA: f64 = 0.25;
const CAPTURE: f64 = -1.2;
const JITTER: f64 = 0.3;

/// One lineage, every unspliced read captured at `e^{CAPTURE}`, every
/// pseudobulk's share jittering by [`JITTER`] around it.
fn sim() -> &'static Sim {
    static SIM: OnceLock<Sim> = OnceLock::new();
    SIM.get_or_init(|| {
        simulate(&SimConfig {
            fine: 133,
            kappa: KAPPA,
            log_capture: CAPTURE,
            share_jitter: JITTER,
            ..SimConfig::default()
        })
        .unwrap()
    })
}

/// The climb's mode from the reads alone and its jitter sd.
fn mode() -> &'static (Params, Option<f64>) {
    static MODE: OnceLock<(Params, Option<f64>)> = OnceLock::new();
    MODE.get_or_init(|| {
        let opts = Climb {
            jitter: true,
            trunk: false,
            lineages: 1,
            points: POINTS,
            dim: 3,
            kappa: KAPPA,
            steps: 2500,
            lr: 0.03,
            sub: 1,
        };
        let (fit, _) = climb(&first(sim(), TRAIN), &opts, &Device::Cpu).unwrap();
        (fit.params().unwrap(), fit.share_sd().unwrap())
    })
}

fn median_ratio_gap(beta: &[f64], gamma: &[f64]) -> f64 {
    let t = &sim().truth;
    let mut gap: Vec<f64> = (0..beta.len())
        .map(|g| ((gamma[g] / beta[g]).ln() - (t.gamma[g] / t.beta[g]).ln()).abs())
        .collect();
    gap.sort_by(f64::total_cmp);
    gap[gap.len() / 2]
}

/// The sim's levels carry the capture: the truth's unspliced levels are
/// the uncaptured ones scaled by `e^c`.
#[test]
fn the_truths_unspliced_levels_carry_the_capture() {
    let truth = truth_params(sim(), 133, POINTS);
    assert_eq!(truth.log_capture, CAPTURE);
    let mut bare = truth.clone();
    bare.log_capture = 0.0;
    let (lv, lv0) = (truth.levels(), bare.levels());
    for (a, b) in lv.u.iter().zip(lv0.u.iter()) {
        assert!((a - b * CAPTURE.exp()).abs() <= 1e-12 * a.abs().max(1e-300));
    }
    assert_eq!(lv.s, lv0.s);
}

/// Chains from the climb draw the capture around the truth, the genes'
/// `log γ/β` with it.
#[test]
fn the_chains_find_the_capture() {
    let draws = run(
        &first(sim(), TRAIN),
        &mode().0,
        &Options {
            kappa: KAPPA,
            chains: 2,
            sweeps: 200,
            burn: 50,
            share: mode().1,
            ..Options::default()
        },
    )
    .unwrap();
    let c: Vec<f64> = draws
        .iter()
        .flat_map(|d| d.log_capture.iter().copied())
        .collect();
    let mean = c.iter().sum::<f64>() / c.len() as f64;
    let sd = (c.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / c.len() as f64).sqrt();
    let last = &draws[0];
    let s = last.log_ratio.len() - 1;
    let beta: Vec<f64> = (0..last.log_ratio[s].len())
        .map(|g| (last.log_gamma[s][g] - last.log_ratio[s][g]).exp())
        .collect();
    let gamma: Vec<f64> = last.log_gamma[s].iter().map(|y| y.exp()).collect();
    let gap = median_ratio_gap(&beta, &gamma);
    eprintln!("chains: capture {mean:.3} ± {sd:.3} (truth {CAPTURE}); median log γ/β gap {gap:.3}");
    assert!(sd > 0.0, "the capture never moved");
    assert!((mean - CAPTURE).abs() < 0.15, "capture {mean}");
    assert!(gap < 0.3, "median log γ/β gap {gap}");
}
