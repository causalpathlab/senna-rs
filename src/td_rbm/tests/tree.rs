use super::*;
use rand::rngs::StdRng;
use rand::SeedableRng;
use rand_distr::{Distribution, Normal, Uniform};

/// Points along a Y (a trunk that splits in two) and a separate island in
/// directions of its own, in four dimensions.
fn y_and_island() -> (DMatrix<f32>, Vec<usize>) {
    let mut rng = StdRng::seed_from_u64(2);
    let (u, e) = (
        Uniform::new(0.0f32, 1.0).unwrap(),
        Normal::new(0.0f32, 0.03).unwrap(),
    );
    let mut rows = Vec::new();
    let mut part = Vec::new();
    for i in 0..900 {
        let t = u.sample(&mut rng);
        let (arm, p) = match i % 3 {
            // Trunk from (0,0) to (1,0); arms from (1,0) up and down.
            0 => (0, [t, 0.0, 0.0, 0.0]),
            1 => (1, [1.0 + t, t, 0.0, 0.0]),
            _ => (2, [1.0 + t, -t, 0.0, 0.0]),
        };
        rows.push(p.map(|v| v + e.sample(&mut rng)));
        part.push(arm);
    }
    for _ in 0..200 {
        let t = u.sample(&mut rng);
        rows.push([0.0, 0.0, 4.0 + t, 4.0].map(|v| v + e.sample(&mut rng)));
        part.push(3);
    }
    // Centred, as expression states are (the neighbours are by cosine).
    let n = rows.len() as f32;
    let mean: Vec<f32> = (0..4).map(|j| rows.iter().map(|r| r[j]).sum::<f32>() / n).collect();
    let m = DMatrix::from_fn(rows.len(), 4, |i, j| rows[i][j] - mean[j]);
    (m, part)
}

#[test]
fn a_branching_path_and_an_island_make_two_trees() {
    let (m, part) = y_and_island();
    let forest = Forest::new(&m, 10, 10, 1);
    assert_eq!(forest.trees.len(), 2, "trees {:?}", forest.trees);
    // The island's clusters are a tree of their own.
    let island: std::collections::BTreeSet<usize> = part
        .iter()
        .zip(&forest.cluster)
        .filter(|(&p, _)| p == 3)
        .map(|(_, &c)| forest.tree[c])
        .collect();
    assert_eq!(island.len(), 1);
    let y = 1 - *island.iter().next().unwrap();
    // The Y has three leaves; rooted at one, two lineages.
    let leaves = forest.leaves(y);
    assert_eq!(leaves.len(), 3, "leaves {leaves:?}");
    let mut roots = vec![0; forest.trees.len()];
    for (t, r) in roots.iter_mut().enumerate() {
        *r = forest.far_end(t);
    }
    let paths = forest.paths(&roots);
    assert_eq!(paths.iter().filter(|p| p.tree == y).count(), 2);
    for p in paths.iter().filter(|p| p.tree == y) {
        assert_eq!(p.clusters[0], roots[y]);
    }
}

/// Every edge's genes vote for running away from one leaf: that leaf scores
/// best as the root, whichever leaf the edges were measured from.
#[test]
fn the_root_is_where_every_edge_points_away_from() {
    let (m, _) = y_and_island();
    let forest = Forest::new(&m, 10, 10, 1);
    let y = (0..forest.trees.len())
        .max_by_key(|&t| forest.trees[t].len())
        .unwrap();
    let leaves = forest.leaves(y);
    let truth = leaves[1];
    for &provisional in &leaves {
        let (parent, _) = forest.rooted(truth);
        // An edge measured from `provisional` agrees with the truth when its
        // parent there is its parent under the truth.
        let lbf: Vec<Vec<f64>> = forest
            .edges(provisional)
            .iter()
            .map(|&(a, b)| {
                let agree = parent[b] == Some(a);
                vec![if agree { 2.0 } else { -2.0 }; 50]
            })
            .collect();
        let scores = root_scores(&forest, provisional, &lbf, &leaves);
        let best = leaves[(0..leaves.len())
            .max_by(|&i, &j| scores[i].total_cmp(&scores[j]))
            .unwrap()];
        assert_eq!(best, truth, "from {provisional}: scores {scores:?}");
    }
}

/// An edge counts by its size against its own noise: a total at the noise
/// level leaves the two ways near even, a total far beyond it decides.
#[test]
fn evidence_is_weighed_against_its_noise() {
    let p = |y: f64, noise: f64| {
        let (a, b) = (edge_evidence(&[y], &[noise]), edge_evidence(&[-y], &[noise]));
        1.0 / (1.0 + (b - a).exp())
    };
    let quiet = p(-0.001, 2.0);
    assert!((quiet - 0.5).abs() < 0.05, "noise-level evidence: p {quiet}");
    let loud = p(200.0, 2.0);
    // One edge cannot measure the spread between edges, so it stops short
    // of certainty; it still passes the call.
    assert!(loud > 0.95, "evidence far past its noise: p {loud}");
}
