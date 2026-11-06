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

/// A gene whose `γ` is far faster than its `β` (`(γ − β) δ` past f32's
/// exp range a step) still has the host's finite levels.
#[test]
fn a_fast_genes_levels_stay_finite() {
    let sim = simulate(&SimConfig {
        fine: 133,
        ..SimConfig::default()
    })
    .unwrap();
    let mut params = truth_params(&sim, 133, 45);
    params.gamma[0] = 9000.0;
    params.beta[0] = 0.05;
    let prior = GridPrior::new(45, 1.0, 0.25).unwrap();
    let fit = Fit::from_params(&params, &prior, &Device::Cpu).unwrap();
    let (u, s) = fit.levels().unwrap();
    let (u, s) = (u.to_vec2::<f32>().unwrap(), s.to_vec2::<f32>().unwrap());
    let lv = params.levels();
    for r in 0..lv.u.nrows() {
        assert!(
            u[r][0].is_finite() && s[r][0].is_finite(),
            "row {r}: {} {}",
            u[r][0],
            s[r][0]
        );
        assert!(rel(f64::from(s[r][0]), lv.s[(r, 0)]) < 1e-3, "row {r}");
    }
}

/// Genes whose rates are nearly equal, `(γ − β) δ` across the series'
/// switch, have the host's levels to f32's precision, not the cancelling
/// difference's.
#[test]
fn near_equal_rates_keep_their_precision() {
    let sim = simulate(&SimConfig {
        fine: 133,
        ..SimConfig::default()
    })
    .unwrap();
    let mut params = truth_params(&sim, 133, 45);
    params.sub = 1;
    let delta = params.t_len[0] / 44.0;
    for (g, x) in [2e-4, 1e-3, 3e-3, 5e-3, 2e-2].iter().enumerate() {
        params.beta[g] = 2.0;
        params.gamma[g] = 2.0 + x / delta;
    }
    let prior = GridPrior::new(45, 1.0, 0.25).unwrap();
    let fit = Fit::from_params(&params, &prior, &Device::Cpu).unwrap();
    let (_, s) = fit.levels().unwrap();
    let s = s.to_vec2::<f32>().unwrap();
    let lv = params.levels();
    let worst = s
        .iter()
        .enumerate()
        .flat_map(|(r, row)| (0..5).map(move |g| (r, g, row[g])))
        .map(|(r, g, x)| rel(f64::from(x), lv.s[(r, g)]))
        .fold(0f64, f64::max);
    eprintln!("near-equal rates: worst relative error {worst:.2e}");
    assert!(worst < 5e-6, "worst relative error {worst}");
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

/// The fit's share term is the host's: the held objective with a jitter
/// minus without is `Σ_p` [`share_term`] at every pseudobulk's held row.
#[test]
fn the_share_term_is_the_hosts() {
    use crate::ode_mixture::likelihood::share_term;
    use legume_numeric::candle::candle_core::Tensor;
    let sim = simulate(&SimConfig {
        fine: 133,
        share_jitter: 0.4,
        ..SimConfig::default()
    })
    .unwrap();
    let params = truth_params(&sim, 133, 45);
    let prior = GridPrior::new(45, 1.0, 0.25).unwrap();
    let reads = Reads::new(&sim.u, &sim.s, sim.n_pb, sim.n_genes);
    let rows: Vec<u32> = (0..sim.n_pb).map(|p| (p % 45) as u32).collect();
    let held = |sd: Option<f64>| {
        let fit = Fit::from_params(&params, &prior, &Device::Cpu)
            .unwrap()
            .with_share(sd)
            .unwrap();
        let dev = fit.reads(&reads).unwrap();
        let r = Tensor::new(rows.as_slice(), &Device::Cpu).unwrap();
        f64::from(
            fit.held_objective(&dev, &r)
                .unwrap()
                .to_scalar::<f32>()
                .unwrap(),
        )
    };
    let got = held(Some(0.4)) - held(None);
    let lv = params.levels();
    let want: f64 = rows
        .iter()
        .enumerate()
        .map(|(p, &r)| {
            let r = r as usize;
            let n_u = reads.u.row(p).sum();
            share_term(
                n_u,
                reads.total(p),
                lv.u.row(r).sum(),
                lv.s.row(r).sum(),
                0.4,
            )
        })
        .sum();
    assert!(
        (got - want).abs() < 1e-4 * want.abs().max(1.0) + 2.0,
        "{got} vs {want}"
    );
}

/// With a shared trunk every lineage's levels are lineage 0's up to the
/// branch, exactly (the same path and clock from the same steady start),
/// and part after it; the branch time reads back.
#[test]
fn with_a_trunk_the_lineages_levels_agree_before_the_branch() {
    let sim = simulate(&SimConfig {
        lineages: 2,
        fine: 133,
        branch_at: Some(0.4),
        ..SimConfig::default()
    })
    .unwrap();
    let params = truth_params(&sim, 133, 45);
    let prior = GridPrior::new(45, 1.0, 0.25).unwrap();
    let fit = Fit::from_params(&params, &prior, &Device::Cpu)
        .unwrap()
        .with_trunk(&params, &prior, 0.4)
        .unwrap();
    assert!((fit.branch_at().unwrap().unwrap()[0] - 0.4).abs() < 1e-5);
    let (u, _) = fit.levels().unwrap();
    let u = u.to_vec2::<f32>().unwrap();
    let (mut before, mut after) = (0f64, 0f64);
    for k in 0..45 {
        let d = (0..sim.n_genes)
            .map(|g| rel(f64::from(u[45 + k][g]), f64::from(u[k][g])))
            .fold(0.0, f64::max);
        if (k as f64) / 44.0 <= 0.4 {
            before = before.max(d);
        } else {
            after = after.max(d);
        }
    }
    assert!(before < 1e-6, "before the branch {before}");
    assert!(after > 0.05, "after the branch {after}");
}
