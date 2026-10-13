//! The lineage structure, after Slingshot (Street et al., *BMC Genomics*
//! (2018)): the finest pseudobulks in clusters, a minimum spanning tree over
//! the clusters' centroids, cut into a forest wherever no pseudobulk's
//! nearest neighbours cross an edge (so an island is a tree of its own).
//! Given a root per tree, every root-to-leaf path is a lineage; lineages of
//! one tree share their trunk and one direction.

use nalgebra::DMatrix;

/// Finest pseudobulks in clusters, and the clusters in a forest.
#[derive(Clone, Debug)]
pub struct Forest {
    /// `[P]` every finest pseudobulk's cluster.
    pub cluster: Vec<usize>,
    pub clusters: usize,
    /// `[C]` every cluster's neighbours in the forest.
    pub adjacent: Vec<Vec<usize>>,
    /// `[C]` every cluster's tree; the trees, each its clusters.
    pub tree: Vec<usize>,
    pub trees: Vec<Vec<usize>>,
    /// `[C × D]` the clusters' centroids in the expression states.
    centroid: DMatrix<f32>,
}

/// One root-to-leaf path: a lineage.
#[derive(Clone, Debug, PartialEq)]
pub struct Path {
    pub tree: usize,
    /// Clusters from the root to the leaf.
    pub clusters: Vec<usize>,
}

impl Forest {
    /// `clusters` k-means clusters of the rows of `states` (`[P × D]`)
    /// (empty ones dropped), the minimum spanning tree of their centroids,
    /// and an edge kept only where some row's `knn` nearest neighbours join
    /// its two clusters.
    pub fn new(states: &DMatrix<f32>, clusters: usize, knn: usize, seed: u64) -> Self {
        let p = states.nrows();
        let raw = super::start::kmeans(states, clusters.clamp(1, p.max(1)), seed);
        // Clusters numbered densely, empty ones gone.
        let mut used = raw.clone();
        used.sort_unstable();
        used.dedup();
        let dense: std::collections::BTreeMap<usize, usize> =
            used.iter().enumerate().map(|(i, &a)| (a, i)).collect();
        let cluster: Vec<usize> = raw.iter().map(|a| dense[a]).collect();
        let c = dense.len();
        let d = states.ncols();
        let mut centroid = DMatrix::<f32>::zeros(c, d);
        let mut size = vec![0f32; c];
        for (r, &a) in cluster.iter().enumerate() {
            size[a] += 1.0;
            for j in 0..d {
                centroid[(a, j)] += states[(r, j)];
            }
        }
        for a in 0..c {
            for j in 0..d {
                centroid[(a, j)] /= size[a].max(1.0);
            }
        }
        // Rows' neighbours crossing between clusters.
        let k = knn.min(p.saturating_sub(1)).max(1);
        let mut crossing = vec![vec![0usize; c]; c];
        for (a, b, _) in super::init::knn_edges(states, k) {
            let (x, y) = (cluster[a], cluster[b]);
            if x != y {
                crossing[x][y] += 1;
                crossing[y][x] += 1;
            }
        }
        let dist = |a: usize, b: usize| -> f32 {
            (0..d)
                .map(|j| (centroid[(a, j)] - centroid[(b, j)]).powi(2))
                .sum::<f32>()
                .sqrt()
        };
        // Prim's tree over the centroids; an edge no neighbour pair crosses
        // is cut. (So a cluster joined to the rest only off the tree's edges
        // stands apart: the tree's shape, not the neighbour graph's, decides.)
        let mut adjacent = vec![Vec::new(); c];
        let mut inside = vec![false; c];
        let mut best = vec![(f32::INFINITY, usize::MAX); c];
        inside[0] = true;
        for (b, slot) in best.iter_mut().enumerate().skip(1) {
            *slot = (dist(0, b), 0);
        }
        for _ in 1..c {
            let next = (0..c)
                .filter(|&b| !inside[b])
                .min_by(|&x, &y| best[x].0.total_cmp(&best[y].0))
                .expect("a cluster left");
            inside[next] = true;
            let from = best[next].1;
            if crossing[from][next] > 0 {
                adjacent[from].push(next);
                adjacent[next].push(from);
            }
            for b in 0..c {
                if !inside[b] {
                    let v = dist(next, b);
                    if v < best[b].0 {
                        best[b] = (v, next);
                    }
                }
            }
        }
        // The trees: connected components.
        let mut tree = vec![usize::MAX; c];
        let mut trees: Vec<Vec<usize>> = Vec::new();
        for s in 0..c {
            if tree[s] != usize::MAX {
                continue;
            }
            let t = trees.len();
            let mut members = vec![s];
            tree[s] = t;
            let mut i = 0;
            while i < members.len() {
                for &b in &adjacent[members[i]] {
                    if tree[b] == usize::MAX {
                        tree[b] = t;
                        members.push(b);
                    }
                }
                i += 1;
            }
            members.sort_unstable();
            trees.push(members);
        }
        Self {
            cluster,
            clusters: c,
            adjacent,
            tree,
            trees,
            centroid,
        }
    }

    /// Tree `t`'s leaves (a lone cluster is its own leaf), ascending.
    pub fn leaves(&self, t: usize) -> Vec<usize> {
        let members = &self.trees[t];
        if members.len() == 1 {
            return members.clone();
        }
        members
            .iter()
            .copied()
            .filter(|&a| self.adjacent[a].len() == 1)
            .collect()
    }

    /// Every cluster's parent with tree `t` rooted at `root` (`None` at the
    /// root), and the clusters in breadth-first order.
    pub fn rooted(&self, root: usize) -> (Vec<Option<usize>>, Vec<usize>) {
        let mut parent = vec![None; self.clusters];
        let mut seen = vec![false; self.clusters];
        let mut order = vec![root];
        seen[root] = true;
        let mut i = 0;
        while i < order.len() {
            let a = order[i];
            let mut next: Vec<usize> = self.adjacent[a]
                .iter()
                .copied()
                .filter(|&b| !seen[b])
                .collect();
            next.sort_unstable();
            for b in next {
                seen[b] = true;
                parent[b] = Some(a);
                order.push(b);
            }
            i += 1;
        }
        (parent, order)
    }

    /// A provisional root for tree `t`: one end of its longest path between
    /// leaves, by centroid distance (deterministic).
    pub fn far_end(&self, t: usize) -> usize {
        let leaves = self.leaves(t);
        let far = |from: usize| -> usize {
            let (parent, order) = self.rooted(from);
            let mut depth = vec![0f32; self.clusters];
            for &a in &order[1..] {
                let up = parent[a].expect("a parent below the root");
                depth[a] = depth[up] + self.distance(up, a);
            }
            *leaves
                .iter()
                .max_by(|&&x, &&y| depth[x].total_cmp(&depth[y]).then(y.cmp(&x)))
                .expect("a leaf")
        };
        far(far(self.trees[t][0]))
    }

    fn distance(&self, a: usize, b: usize) -> f32 {
        (0..self.centroid.ncols())
            .map(|j| (self.centroid[(a, j)] - self.centroid[(b, j)]).powi(2))
            .sum::<f32>()
            .sqrt()
    }

    /// The lineages with every tree rooted at `roots[t]`: a path from the
    /// root to every other leaf (a lone cluster, its own path), trees in
    /// order, leaves ascending.
    pub fn paths(&self, roots: &[usize]) -> Vec<Path> {
        let mut out = Vec::new();
        for (t, &root) in roots.iter().enumerate() {
            let (parent, _) = self.rooted(root);
            let leaves: Vec<usize> = self.leaves(t).into_iter().filter(|&a| a != root).collect();
            if leaves.is_empty() {
                out.push(Path {
                    tree: t,
                    clusters: vec![root],
                });
                continue;
            }
            for leaf in leaves {
                let mut up = vec![leaf];
                while let Some(a) = parent[*up.last().expect("a cluster")] {
                    up.push(a);
                }
                up.reverse();
                out.push(Path {
                    tree: t,
                    clusters: up,
                });
            }
        }
        out
    }

    /// The edges of tree `t` rooted at `root`, each `(parent, child)`.
    pub fn edges(&self, root: usize) -> Vec<(usize, usize)> {
        let (parent, order) = self.rooted(root);
        order[1..]
            .iter()
            .map(|&b| (parent[b].expect("a parent below the root"), b))
            .collect()
    }
}

/// Every candidate root's log score for one tree: its edges `(parent, child)`
/// under the provisional root, every edge's genes' `log BF` for the
/// provisional orientation (`[E][G]`), and every candidate; an edge runs the
/// other way under a candidate inside the child's subtree. The edges, not the
/// genes, are the units: every edge's summed `log BF`, signed for the
/// candidate's orientation, is one measurement of the tree's direction
/// ([`edge_evidence`]), so a few loud edges cannot outvote the rest.
pub fn root_scores(
    forest: &Forest,
    provisional: usize,
    edge_lbf: &[Vec<f64>],
    candidates: &[usize],
) -> Vec<f64> {
    let edges = forest.edges(provisional);
    let (parent, _) = forest.rooted(provisional);
    // Whether `c` lies in the subtree of `b` (under the provisional root).
    let below = |b: usize, c: usize| -> bool {
        let mut x = Some(c);
        while let Some(a) = x {
            if a == b {
                return true;
            }
            x = parent[a];
        }
        false
    };
    let sums: Vec<f64> = edge_lbf.iter().map(|v| v.iter().sum()).collect();
    let noise: Vec<f64> = edge_lbf
        .iter()
        .map(|v| v.iter().map(|x| x * x).sum::<f64>().sqrt())
        .collect();
    candidates
        .iter()
        .map(|&r| {
            let y: Vec<f64> = edges
                .iter()
                .zip(&sums)
                .map(|(&(_, b), &v)| if below(b, r) { -v } else { v })
                .collect();
            edge_evidence(&y, &noise)
        })
        .collect()
}

/// `log ∫∫ Π_e N(y_e | μ, σ² + n_e²) dμ dσ`: the edges' signed evidence `y`
/// as draws about a common agreement `μ ≥ 0`, each with its own noise `n_e`
/// (its genes' `√Σ log BF²`, what the sum could be by chance) and a spread
/// `σ` between edges, `μ` and `log σ` uniform on grids. An edge's evidence
/// counts by its size against its noise, and edges that disagree widen `σ`,
/// so the call stays open.
pub fn edge_evidence(y: &[f64], noise: &[f64]) -> f64 {
    let m = y
        .iter()
        .chain(noise)
        .fold(0f64, |m, v| m.max(v.abs()))
        .max(1e-9);
    let mus: Vec<f64> = (0..64).map(|i| 2.0 * m * i as f64 / 63.0).collect();
    let sigmas: Vec<f64> = (0..48)
        .map(|i| (m * 1e-3).ln() + ((2.0 * m).ln() - (m * 1e-3).ln()) * i as f64 / 47.0)
        .map(f64::exp)
        .collect();
    let mut cells = Vec::with_capacity(mus.len() * sigmas.len());
    for &mu in &mus {
        for &sd in &sigmas {
            let ll: f64 = y
                .iter()
                .zip(noise)
                .map(|(&v, &n)| {
                    let var = sd * sd + n * n;
                    -0.5 * (v - mu).powi(2) / var - 0.5 * var.ln() - 0.918_938_533_204_672_8
                })
                .sum();
            cells.push(ll);
        }
    }
    super::start::log_sum_exp(&cells) - (cells.len() as f64).ln()
}

#[cfg(test)]
#[path = "tests/tree.rs"]
mod tests;
