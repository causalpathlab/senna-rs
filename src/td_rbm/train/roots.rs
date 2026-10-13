//! Step 2: every tree's root, which is its lineages' one direction, from the
//! splicing along its edges; then the lineages rebuilt from the roots.
//!
//! Every edge of a tree (clusters `a → b` from the provisional root) is read
//! by every lineage through it, on that lineage's finest pseudobulks in the
//! two clusters: every gene's `log BF` for running `a → b` ([`bayes`]), the
//! lineages' summed (they hold disjoint pseudobulks). A candidate root (a
//! leaf) runs every edge away from itself; the edges' summed evidence,
//! signed for it, are its measurements ([`root_scores`]), and the
//! candidates' posterior is their scores' softmax (a uniform prior). A tree
//! of one cluster has one lineage, read as it runs or mirrored.

use super::direction::{bayes, P_CALL};
use super::exact::Phase;
use crate::td_rbm::start::{row_states, start_along, STATE_DIMS};
use crate::td_rbm::tree::{edge_evidence, root_scores, Forest, Path};
use rand::rngs::StdRng;

/// A tree's root.
#[derive(Clone, Debug)]
pub struct RootCall {
    pub tree: usize,
    /// The root the lineages were measured from, and the candidates' (the
    /// tree's leaves) posterior probabilities.
    pub provisional: usize,
    pub candidates: Vec<usize>,
    pub p: Vec<f64>,
    pub chosen: usize,
    /// For a one-cluster tree: whether its lineage was mirrored.
    pub mirrored: bool,
    /// The call reaches [`P_CALL`], or the root was given.
    pub determined: bool,
    pub given: bool,
    /// Every edge `(parent, child)` under the provisional root, its genes'
    /// summed `log BF` for running parent → child, and the genes for it and
    /// against it.
    pub edges: Vec<(usize, usize, f64, usize, usize)>,
}

fn softmax(v: &[f64]) -> Vec<f64> {
    let top = v.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let w: Vec<f64> = v.iter().map(|x| (x - top).exp()).collect();
    let z: f64 = w.iter().sum();
    w.into_iter().map(|x| x / z).collect()
}

impl Phase<'_> {
    /// The finest pseudobulks of lineage `l` in clusters `within`, as `[R × F]`
    /// counts.
    fn rows_of(&self, forest: &Forest, l: usize, within: &[usize]) -> Vec<f32> {
        let finest = self.levels.len() - 1;
        let (x, f) = (&self.levels[finest].x, self.f);
        (0..self.levels[finest].cells.len())
            .filter(|&r| self.assigned[finest][r] == l && within.contains(&forest.cluster[r]))
            .flat_map(|r| x[r * f..(r + 1) * f].to_vec())
            .collect()
    }

    /// Every tree's root (module docs), `roots` the provisional ones the
    /// lineages `paths` run from and `given` the trees whose root is fixed.
    /// A one-cluster tree's lineage is mirrored here when the splicing says
    /// so.
    pub(super) fn choose_roots(
        &self,
        forest: &Forest,
        paths: &[Path],
        roots: &[usize],
        given: &[bool],
        rng: &mut StdRng,
    ) -> anyhow::Result<Vec<RootCall>> {
        let (f, cfg, dev) = (self.f, self.cfg, self.dev);
        let steady = self.energy.steady_tables(self.spline)?;
        let mut calls = Vec::with_capacity(forest.trees.len());
        for t in 0..forest.trees.len() {
            let provisional = roots[t];
            let lineages: Vec<usize> = (0..paths.len()).filter(|&l| paths[l].tree == t).collect();
            if given[t] {
                calls.push(RootCall {
                    tree: t,
                    provisional,
                    candidates: vec![provisional],
                    p: vec![1.0],
                    chosen: provisional,
                    mirrored: false,
                    determined: true,
                    given: true,
                    edges: Vec::new(),
                });
                continue;
            }
            if forest.trees[t].len() == 1 {
                // One cluster, one lineage: as it runs, or mirrored.
                let l = lineages[0];
                let x = self.rows_of(forest, l, &forest.trees[t]);
                let genes = if x.len() >= 2 * f {
                    bayes(self.energy, self.spline, &steady, &x, f, l, cfg, dev, rng)?.genes
                } else {
                    Vec::new()
                };
                let total: f64 = genes.iter().sum();
                let noise = [genes.iter().map(|v| v * v).sum::<f64>().sqrt()];
                let p = softmax(&[
                    edge_evidence(&[total], &noise),
                    edge_evidence(&[-total], &noise),
                ]);
                let mirrored = p[1] > p[0];
                if mirrored {
                    self.energy.mirror(l)?;
                }
                calls.push(RootCall {
                    tree: t,
                    provisional,
                    candidates: vec![provisional],
                    determined: p[0].max(p[1]) >= P_CALL,
                    p: vec![p[0].max(p[1])],
                    chosen: provisional,
                    mirrored,
                    given: false,
                    edges: Vec::new(),
                });
                continue;
            }
            let edges = forest.edges(provisional);
            let mut edge_lbf = Vec::with_capacity(edges.len());
            for &(a, b) in &edges {
                let mut sum = vec![0f64; f / 2];
                for &l in &lineages {
                    let on = paths[l].clusters.windows(2).any(|w| w[0] == a && w[1] == b);
                    if !on {
                        continue;
                    }
                    let x = self.rows_of(forest, l, &[a, b]);
                    if x.len() < 2 * f {
                        continue;
                    }
                    let genes =
                        bayes(self.energy, self.spline, &steady, &x, f, l, cfg, dev, rng)?.genes;
                    sum.iter_mut().zip(&genes).for_each(|(s, v)| *s += v);
                }
                edge_lbf.push(sum);
            }
            let candidates = forest.leaves(t);
            let p = softmax(&root_scores(forest, provisional, &edge_lbf, &candidates));
            // The likeliest; on a tie the provisional root stays.
            let best = (0..candidates.len())
                .max_by(|&i, &j| {
                    p[i].total_cmp(&p[j])
                        .then((candidates[i] == provisional).cmp(&(candidates[j] == provisional)))
                })
                .expect("a candidate");
            calls.push(RootCall {
                tree: t,
                provisional,
                chosen: candidates[best],
                determined: p[best] >= P_CALL,
                edges: edges
                    .iter()
                    .zip(&edge_lbf)
                    .map(|(&(a, b), v)| {
                        (
                            a,
                            b,
                            v.iter().sum(),
                            v.iter().filter(|&&x| x > 0.0).count(),
                            v.iter().filter(|&&x| x < 0.0).count(),
                        )
                    })
                    .collect(),
                candidates,
                p,
                mirrored: false,
                given: false,
            });
        }
        Ok(calls)
    }

    /// The lineages of the trees `moved` started again along `forest`
    /// rooted at `roots`: their paths' curves and their finest pseudobulks'
    /// lineages (every other tree's lineages, already fitted or mirrored,
    /// are left as they are; a tree's path count does not change with its
    /// root, so their indices hold); the paths.
    pub(super) fn rebuild(
        &mut self,
        forest: &Forest,
        roots: &[usize],
        moved: &[bool],
    ) -> anyhow::Result<Vec<Path>> {
        let finest = self.levels.len() - 1;
        let (f, cfg) = (self.f, self.cfg);
        let rho = self.energy.rho().flatten_all()?.to_vec1::<f32>()?;
        let h = self.energy.rho().dim(1)?;
        let values = self.spline.values.flatten_all()?.to_vec1::<f32>()?;
        let x = &self.levels[finest].x;
        let start = start_along(
            x,
            f,
            &row_states(x, f, STATE_DIMS),
            forest,
            roots,
            &rho,
            &values,
            (f / 2, h, cfg.grid, cfg.knots),
            cfg.knn,
        );
        for (l, c) in start.curve.chunks(cfg.knots * h).enumerate() {
            if moved[start.paths[l].tree] {
                self.energy.set_lineage_curve(l, c)?;
            }
        }
        for (r, &l) in start.lineage.iter().enumerate() {
            if moved[forest.tree[forest.cluster[r]]] {
                self.assigned[finest][r] = l;
            }
        }
        self.counts.iter_mut().for_each(|c| *c = 0.0);
        for (&a, &c) in self.assigned[finest].iter().zip(&self.levels[finest].cells) {
            self.counts[a] += f64::from(c);
        }
        Ok(start.paths)
    }
}
