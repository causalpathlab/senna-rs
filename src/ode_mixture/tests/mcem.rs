//! Several fates on a shared trunk: the climb from scratch finds every
//! fate as its own lineage.

use super::init::{first, spearman};
use crate::ode_mixture::fit::Fit;
use crate::ode_mixture::init::{climb, tree_start_from_roots, tree_starts, Climb, Places};
use crate::ode_mixture::likelihood::Reads;
use crate::ode_mixture::sim::{simulate, Sim, SimConfig};
use legume_numeric::candle::candle_core::Device;

const BRANCH: f64 = 0.4;

fn fates(l: usize, seed: u64) -> (Sim, Reads) {
    let sim = simulate(&SimConfig {
        lineages: l,
        n_pb: 300 * l,
        fine: 133,
        branch_at: Some(BRANCH),
        part: 1.5,
        seed,
        ..SimConfig::default()
    })
    .unwrap();
    let reads = first(&sim, sim.n_pb);
    (sim, reads)
}

fn opts(l: usize) -> Climb {
    Climb {
        trunk: true,
        jitter: true,
        lineages: l,
        points: 45,
        dim: 3,
        kappa: 0.25,
        steps: 2500,
        lr: 0.03,
        sub: 1,
        roots: Vec::new(),
    }
}

/// Over the pseudobulks well past the branch (a quarter of it or more),
/// the share on the fitted lineage that best matches their true one, the
/// lineages matched one to one (the best of every permutation).
fn purity(fit: &Fit, reads: &Reads, sim: &Sim) -> f64 {
    let (weights, _) = fit.posterior(&fit.reads(reads).unwrap()).unwrap();
    let t = &sim.truth;
    let past: Vec<usize> = (0..sim.n_pb)
        .filter(|&p| t.time[p] > BRANCH * 1.25)
        .collect();
    let l = weights[0].len();
    let pairs: Vec<(usize, usize)> = past
        .iter()
        .map(|&p| {
            let w = &weights[p];
            (
                t.lineage[p],
                (0..l).max_by(|&a, &b| w[a].total_cmp(&w[b])).unwrap(),
            )
        })
        .collect();
    best_match(l, &pairs) as f64 / past.len() as f64
}

/// Over `(true, called)` lineage pairs, how many agree under the best
/// one-to-one matching of the `l` lineages (every permutation tried).
fn best_match(l: usize, pairs: &[(usize, usize)]) -> usize {
    let mut perm: Vec<usize> = (0..l).collect();
    let mut best = 0;
    permute(&mut perm, 0, &mut |perm| {
        let right = pairs
            .iter()
            .filter(|&&(truth, called)| perm[truth] == called)
            .count();
        best = best.max(right);
    });
    best
}

fn permute(v: &mut Vec<usize>, k: usize, f: &mut dyn FnMut(&[usize])) {
    if k == v.len() {
        return f(v);
    }
    for i in k..v.len() {
        v.swap(k, i);
        permute(v, k + 1, f);
        v.swap(k, i);
    }
}

fn check(l: usize, seed: u64) -> (f64, f64) {
    let (sim, reads) = fates(l, seed);
    let (fit, objective) = climb(&reads, &opts(l), &Device::Cpu).unwrap();
    let (_, time) = fit.posterior(&fit.reads(&reads).unwrap()).unwrap();
    let (rho, pure) = (spearman(&time, &sim.truth.time), purity(&fit, &reads, &sim));
    eprintln!("{l} fates, seed {seed}: objective {objective:.1}, time Spearman {rho:.3}, purity {pure:.3}");
    (rho, pure)
}

/// Over the pseudobulks well past the branch placed on one lineage, the
/// share on the lineage matching their fate (the best of every
/// permutation); and over the trunk's, the share placed on more than one.
fn start_purity(places: &Places, sim: &Sim) -> (f64, f64) {
    let t = &sim.truth;
    let l = sim.truth.path.len();
    let past: Vec<(usize, usize)> = (0..sim.n_pb)
        .filter(|&p| t.time[p] > BRANCH * 1.25 && places[p].len() == 1)
        .map(|p| (t.lineage[p], places[p][0].0))
        .collect();
    let trunk: Vec<usize> = (0..sim.n_pb).filter(|&p| t.trunk[p]).collect();
    let shared = trunk.iter().filter(|&&p| places[p].len() > 1).count();
    (
        best_match(l, &past) as f64 / past.len().max(1) as f64,
        shared as f64 / trunk.len().max(1) as f64,
    )
}

/// One of the tree starts puts three fates on three lineages, the trunk
/// shared.
#[test]
fn a_tree_start_puts_three_fates_on_three_lineages() {
    tree_starts_find_the_fates(1..=1);
}

fn tree_starts_find_the_fates(seeds: std::ops::RangeInclusive<u64>) {
    let mut missed = Vec::new();
    for seed in seeds {
        let (sim, reads) = fates(3, seed);
        let starts = tree_starts(&reads, 3, 3);
        let scored: Vec<(f64, f64)> = starts.iter().map(|s| start_purity(s, &sim)).collect();
        eprintln!("seed {seed}: tree starts (purity, trunk shared): {scored:?}");
        assert_eq!(starts.len(), 4, "a start per end");
        if !scored
            .iter()
            .any(|&(pure, shared)| pure > 0.9 && shared > 0.8)
        {
            missed.push(seed);
        }
    }
    assert!(
        missed.is_empty(),
        "seeds {missed:?}: no start finds the fates"
    );
}

/// Given the root (the earliest pseudobulk), the one tree start puts the
/// fates on lineages, the trunk shared: the root is the user's to name, not
/// an end of the tree, which a stem in the middle of a tree never is.
#[test]
fn given_the_root_the_tree_start_puts_the_fates_on_lineages() {
    from_the_root_the_start_finds_the_fates(1..=2);
}

fn from_the_root_the_start_finds_the_fates(seeds: std::ops::RangeInclusive<u64>) {
    from_roots_the_start_finds_the_fates(seeds, 1);
}

/// The root named as a set of pseudobulks (the earliest tenth): the tree
/// grows from the set, every pseudobulk's distance from the root its
/// distance to the nearest member, so no one member decides the tree.
#[test]
fn given_root_pseudobulks_the_tree_start_puts_the_fates_on_lineages() {
    from_roots_the_start_finds_the_fates(1..=2, 90);
}

/// The tree start from the `n_roots` earliest pseudobulks finds the fates.
fn from_roots_the_start_finds_the_fates(seeds: std::ops::RangeInclusive<u64>, n_roots: usize) {
    let mut missed = Vec::new();
    for seed in seeds {
        let (sim, reads) = fates(3, seed);
        let mut by_time: Vec<usize> = (0..sim.n_pb).collect();
        by_time.sort_by(|&a, &b| sim.truth.time[a].total_cmp(&sim.truth.time[b]));
        let roots = &by_time[..n_roots];
        let start = tree_start_from_roots(&reads, 3, 3, roots).expect("a tree from the roots");
        let (pure, shared) = start_purity(&start, &sim);
        eprintln!(
            "seed {seed}: from {n_roots} root(s) (purity, trunk shared) ({pure:.3}, {shared:.3})"
        );
        if !(pure > 0.9 && shared > 0.8) {
            missed.push(seed);
        }
    }
    assert!(
        missed.is_empty(),
        "seeds {missed:?}: the start from the root misses a fate"
    );
}

/// Three fates off one trunk, each its own lineage: the fitted lineages
/// are the fates, time in order.
#[test]
#[cfg_attr(not(feature = "slow-tests"), ignore = "slow: --features slow-tests")]
fn from_scratch_three_fates_are_three_lineages() {
    let (rho, pure) = check(3, 1);
    assert!(rho.abs() > 0.9, "time Spearman {rho}");
    assert!(pure > 0.9, "purity {pure}");
}
