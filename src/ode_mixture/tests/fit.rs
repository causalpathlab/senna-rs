use super::model::truth_params;
use crate::ode_mixture::fit::Fit;
use crate::ode_mixture::kernel::GridPrior;
use crate::ode_mixture::likelihood::{table, Reads};
use crate::ode_mixture::sim::{simulate, SimConfig};
use legume_numeric::candle::candle_core::Device;

fn rel(a: f64, b: f64) -> f64 {
    (a - b).abs() / b.abs().max(1e-9)
}

#[test]
fn the_levels_are_the_hosts() {
    let cfg = SimConfig {
        lineages: 2,
        fine: 133,
        ..SimConfig::default()
    };
    let sim = simulate(&cfg).unwrap();
    let mut params = truth_params(&sim, 133, 45);
    params.t_len[1] = 1.7;
    params.sub = 4;
    let prior = GridPrior::new(45, 1.0, cfg.kappa).unwrap();
    let fit = Fit::from_params(&params, &prior, &Device::Cpu).unwrap();
    let (u, s) = fit.levels().unwrap();
    let (u, s) = (u.to_vec2::<f32>().unwrap(), s.to_vec2::<f32>().unwrap());
    let lv = params.levels();
    let mut worst = 0f64;
    for r in 0..lv.u.nrows() {
        for g in 0..lv.genes {
            worst = worst.max(rel(f64::from(u[r][g]), lv.u[(r, g)]));
            worst = worst.max(rel(f64::from(s[r][g]), lv.s[(r, g)]));
        }
    }
    assert!(worst < 1e-3, "worst relative error {worst}");
}

#[test]
fn the_marginal_is_the_hosts() {
    let cfg = SimConfig {
        lineages: 2,
        fine: 133,
        ..SimConfig::default()
    };
    let sim = simulate(&cfg).unwrap();
    let params = truth_params(&sim, 133, 45);
    let prior = GridPrior::new(45, 1.0, cfg.kappa).unwrap();
    let reads = Reads::new(&sim.u, &sim.s, sim.n_pb, sim.n_genes);
    let fit = Fit::from_params(&params, &prior, &Device::Cpu).unwrap();
    let got = fit
        .log_marginal(&fit.reads(&reads).unwrap())
        .unwrap()
        .to_scalar::<f32>()
        .unwrap();
    let tab = table(&reads, &params.levels());
    let want: f64 = (0..sim.n_pb)
        .map(|p| {
            let v: Vec<f64> = (0..tab.ncols())
                .map(|r| tab[(p, r)] + 0.5f64.ln() - 45f64.ln())
                .collect();
            let top = v.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            top + v.iter().map(|x| (x - top).exp()).sum::<f64>().ln()
        })
        .sum();
    assert!(rel(f64::from(got), want) < 1e-5, "{got} vs {want}");
}

#[test]
fn adam_climbs_from_the_prior_rates_to_the_truth() {
    let cfg = SimConfig {
        fine: 133,
        ..SimConfig::default()
    };
    let sim = simulate(&cfg).unwrap();
    let truth = truth_params(&sim, 133, 45);
    let prior = GridPrior::new(45, 1.0, cfg.kappa).unwrap();
    let reads = Reads::new(&sim.u, &sim.s, sim.n_pb, sim.n_genes);
    let dev = Device::Cpu;
    let at_truth = {
        let f = Fit::from_params(&truth, &prior, &dev).unwrap();
        f.objective(&f.reads(&reads).unwrap())
            .unwrap()
            .to_scalar::<f32>()
            .unwrap()
    };
    let mut start = truth.clone();
    start.beta = vec![4.0; sim.n_genes];
    start.gamma = vec![2.0; sim.n_genes];
    let fit = Fit::from_params(&start, &prior, &dev).unwrap();
    let before = fit
        .objective(&fit.reads(&reads).unwrap())
        .unwrap()
        .to_scalar::<f32>()
        .unwrap();
    let after = fit.adam(&reads, 600, 0.03).unwrap();
    let got = fit.params().unwrap();
    let mut err: Vec<f64> = (0..sim.n_genes)
        .map(|g| {
            ((got.gamma[g] / got.beta[g]) / (truth.gamma[g] / truth.beta[g]))
                .ln()
                .abs()
        })
        .collect();
    err.sort_by(f64::total_cmp);
    eprintln!(
        "objective: start {before:.1}, after {after:.1}, truth {at_truth:.1}; median |log γ/β error| {:.3}",
        err[err.len() / 2]
    );
    assert!(f64::from(after) > f64::from(at_truth) - 50.0);
    assert!(err[err.len() / 2] < 0.05);
}

/// A stop flag already set ends Adam before its first step: the objective
/// is the start's.
#[test]
fn a_stop_ends_adam_at_once() {
    use std::sync::atomic::AtomicBool;
    let cfg = SimConfig {
        fine: 133,
        ..SimConfig::default()
    };
    let sim = simulate(&cfg).unwrap();
    let prior = GridPrior::new(45, 1.0, cfg.kappa).unwrap();
    let reads = Reads::new(&sim.u, &sim.s, sim.n_pb, sim.n_genes);
    let fit = Fit::from_params(&truth_params(&sim, 133, 45), &prior, &Device::Cpu).unwrap();
    let before = fit
        .objective(&fit.reads(&reads).unwrap())
        .unwrap()
        .to_scalar::<f32>()
        .unwrap();
    let after = fit
        .adam_until(&reads, 1000, 0.03, Some(&AtomicBool::new(true)))
        .unwrap();
    assert_eq!(before, after);
}
