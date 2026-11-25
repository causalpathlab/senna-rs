//! Several fates on a shared trunk: the tree starts put every fate on a
//! lineage of its own (the climb from scratch is checked on branching
//! lineages in `branch`).

use super::init::first;
use crate::ode_mixture::init::{tree_start_from_roots, tree_starts, Places};
use crate::ode_mixture::likelihood::Reads;
use crate::ode_mixture::sim::{simulate, Sim, SimConfig};

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

/// A root the user names is theirs: its tree start alone climbs, raced by
/// no k-means start; with one lineage no tree grows from it, the k-means
/// starts run and the climb says why.
#[test]
fn a_named_root_is_grown_from_or_said_why_not() {
    use crate::ode_mixture::init::{climb, Climb, NoTree};
    let (sim, reads) = fates(2, 1);
    let mut by_time: Vec<usize> = (0..sim.n_pb).collect();
    by_time.sort_by(|&a, &b| sim.truth.time[a].total_cmp(&sim.truth.time[b]));
    let opts = |lineages| Climb {
        lineages,
        points: 12,
        dim: 3,
        kappa: 1.0,
        steps: 5,
        lr: 0.03,
        sub: 1,
        jitter: false,
        trunk: false,
        roots: by_time[..8].to_vec(),
    };
    let dev = legume_numeric::candle::candle_core::Device::Cpu;
    let grown = climb(&reads, &opts(2), &dev).unwrap();
    assert_eq!(
        (grown.roots.len(), grown.root, grown.unrooted),
        (1, Some(0), None)
    );
    let not = climb(&reads, &opts(1), &dev).unwrap();
    assert_eq!((not.root, not.unrooted), (None, Some(NoTree::OneLineage)));
    let note = NoTree::OneLineage.note(8, sim.n_pb, true, 1);
    assert!(
        note.contains("8 of 600 pseudobulks are mostly root cells"),
        "{note}"
    );
}
