//! Two lineages on a shared trunk, from scratch.

use super::init::{first, spearman};
use super::model::truth_params;
use crate::ode_mixture::fit::Fit;
use crate::ode_mixture::init::{climb, Climb};
use crate::ode_mixture::kernel::GridPrior;
use crate::ode_mixture::likelihood::{table, Reads};
use crate::ode_mixture::model::Params;
use crate::ode_mixture::sim::{simulate, SimConfig};
use legume_numeric::candle::candle_core::Device;

/// Every pseudobulk's mean time on each lineage, given that lineage,
/// `[P][L]`.
fn time_on_each(params: &Params, reads: &Reads) -> Vec<Vec<f64>> {
    let lv = params.levels();
    let tab = table(reads, &lv);
    let n = lv.points;
    (0..tab.nrows())
        .map(|p| {
            (0..lv.lineages)
                .map(|l| {
                    let row: Vec<f64> = (0..n).map(|k| tab[(p, lv.row(l, k))]).collect();
                    let top = row.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                    let w: Vec<f64> = row.iter().map(|x| (x - top).exp()).collect();
                    let sum: f64 = w.iter().sum();
                    let step = params.t_len[l] / (n - 1) as f64;
                    w.iter()
                        .enumerate()
                        .map(|(k, q)| q * k as f64 * step)
                        .sum::<f64>()
                        / sum
                })
                .collect()
        })
        .collect()
}

/// `([true][fitted] table, purity)` of the pseudobulks `rows` on the
/// fitted lineage of higher weight, either labelling.
fn purity(rows: &[usize], weights: &[Vec<f64>], truth: &[usize]) -> ([[usize; 2]; 2], f64) {
    let mut tab = [[0usize; 2]; 2];
    for &p in rows {
        tab[truth[p]][usize::from(weights[p][1] > weights[p][0])] += 1;
    }
    let right = (tab[0][0] + tab[1][1]).max(tab[0][1] + tab[1][0]);
    (tab, right as f64 / rows.len() as f64)
}

/// Branch pseudobulks go to their own fitted lineage, time is right on
/// each lineage trunk included, and a trunk pseudobulk reads the same time
/// on either lineage.
///
/// The branches part smoothly (no kink), so just past the branch the two
/// lineages' reads are the same within noise: no fit can call those
/// pseudobulks, the truth's own parameters included. Purity is therefore
/// asked of the branch pseudobulks a quarter of the branch or more past
/// it, and over every branch pseudobulk the fit must call as many as the
/// truth's parameters do.
#[test]
#[cfg_attr(not(feature = "slow-tests"), ignore = "slow: --features slow-tests")]
fn from_scratch_branching_lineages_keep_their_trunk() {
    let tb = 0.4;
    let cfg = SimConfig {
        lineages: 2,
        n_pb: 600,
        fine: 133,
        branch_at: Some(tb),
        ..SimConfig::default()
    };
    let sim = simulate(&cfg).unwrap();
    let n_train = 480;
    let train = first(&sim, n_train);
    let dev = Device::Cpu;
    let opts = Climb {
        trunk: true,
        jitter: true,
        lineages: 2,
        points: 45,
        dim: 3,
        kappa: cfg.kappa,
        steps: 2500,
        lr: 0.03,
        sub: 1,
        roots: Vec::new(),
    };
    let (fit, objective) = climb(&train, &opts, &dev).unwrap();
    let branch_at = fit.branch_at().unwrap().unwrap()[0];
    eprintln!("branch at {branch_at:.3} (truth {tb})");
    let all = first(&sim, sim.n_pb);
    let (weights, time) = fit.posterior(&fit.reads(&all).unwrap()).unwrap();
    let held: Vec<usize> = (n_train..sim.n_pb).collect();
    let truth = &sim.truth;

    let branch: Vec<usize> = held.iter().copied().filter(|&p| !truth.trunk[p]).collect();
    let parted: Vec<usize> = branch
        .iter()
        .copied()
        .filter(|&p| truth.time[p] > (tb + 0.25 * (1.0 - tb)) * cfg.t_len)
        .collect();
    let (tab, all_branch) = purity(&branch, &weights, &truth.lineage);
    let (_, past) = purity(&parted, &weights, &truth.lineage);
    let at_truth = {
        let prior = GridPrior::new(45, 1.0, cfg.kappa).unwrap();
        let f = Fit::from_params(&truth_params(&sim, 133, 45), &prior, &dev).unwrap();
        let (w, _) = f.posterior(&f.reads(&all).unwrap()).unwrap();
        purity(&branch, &w, &truth.lineage).1
    };

    let per: Vec<f64> = (0..2)
        .map(|lin| {
            let rows: Vec<usize> = held
                .iter()
                .copied()
                .filter(|&p| truth.lineage[p] == lin)
                .collect();
            spearman(
                &rows.iter().map(|&p| time[p]).collect::<Vec<_>>(),
                &rows.iter().map(|&p| truth.time[p]).collect::<Vec<_>>(),
            )
        })
        .collect();

    let params = fit.params().unwrap();
    let on_each = time_on_each(&params, &all);
    let trunk: Vec<usize> = held.iter().copied().filter(|&p| truth.trunk[p]).collect();
    let gap: Vec<f64> = trunk
        .iter()
        .map(|&p| (on_each[p][0] - on_each[p][1]).abs())
        .collect();
    let mean_gap = gap.iter().sum::<f64>() / gap.len() as f64;
    let max_gap = gap.iter().copied().fold(0.0, f64::max);
    let trunk_r = spearman(
        &trunk.iter().map(|&p| on_each[p][0]).collect::<Vec<_>>(),
        &trunk.iter().map(|&p| on_each[p][1]).collect::<Vec<_>>(),
    );
    eprintln!(
        "objective {objective:.1}; T {:?}; branch table {tab:?}; purity over every branch \
         pseudobulk {all_branch:.3} (truth's parameters {at_truth:.3}), past a quarter of \
         the branch {past:.3} ({} of {}); per-lineage signed Spearman {per:?}; trunk time \
         across lineages: Spearman {trunk_r:.3}, mean |gap| {mean_gap:.4}, \
         max |gap| {max_gap:.4} ({} trunk)",
        params.t_len,
        parted.len(),
        branch.len(),
        trunk.len()
    );
    assert!(past > 0.95, "purity past the branch {past}");
    assert!(
        all_branch >= at_truth - 0.02,
        "branch purity {all_branch}, truth's {at_truth}"
    );
    assert!(
        per.iter().all(|&r| r > 0.95),
        "per-lineage Spearman {per:?}"
    );
    assert!(trunk_r > 0.95, "trunk Spearman across lineages {trunk_r}");
    // One clock and one trunk: a trunk pseudobulk's time is the same on
    // either lineage but where it sits near the branch.
    assert!(mean_gap < 0.01, "trunk mean gap {mean_gap}");
    assert!((branch_at - tb).abs() < 0.1, "branch at {branch_at}");
}
