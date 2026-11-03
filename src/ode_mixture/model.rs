//! The parameters and the levels they set on the grid.
//!
//! Every lineage `ℓ` has `n` grid points `t_k = k T_ℓ/(n − 1)`; at each the
//! transcription `α_gℓk = exp(ρ_gᵀ z_ℓ(t_k) + b_g)` and the trajectory's
//! levels `ū_gℓk`, `s̄_gℓk` ([`super::kinetics`]), shared rates `β_g`, `γ_g`.

use super::kinetics::trajectory;
use nalgebra::DMatrix;
use rayon::prelude::*;

/// Everything the levels depend on.
#[derive(Clone, Debug)]
pub struct Params {
    pub beta: Vec<f64>,
    pub gamma: Vec<f64>,
    /// `[G][H]` the genes' embedding and `[G]` baselines.
    pub rho: Vec<Vec<f64>>,
    pub base: Vec<f64>,
    /// `[L]` every lineage's duration.
    pub t_len: Vec<f64>,
    /// `[L][n][H]` every lineage's path on its grid.
    pub path: Vec<Vec<Vec<f64>>>,
}

/// Every `(lineage, grid point)`'s levels, rows `ℓ n + k`, a column per gene.
pub struct Levels {
    pub lineages: usize,
    pub points: usize,
    pub genes: usize,
    pub u: DMatrix<f64>,
    pub s: DMatrix<f64>,
}

impl Levels {
    /// The row of lineage `l`'s grid point `k`.
    pub fn row(&self, l: usize, k: usize) -> usize {
        l * self.points + k
    }
}

impl Params {
    /// Gene `g`'s transcription on lineage `l`'s grid.
    pub fn alpha(&self, g: usize, l: usize) -> Vec<f64> {
        self.path[l]
            .iter()
            .map(|z| {
                let dot: f64 = z.iter().zip(&self.rho[g]).map(|(a, b)| a * b).sum();
                (dot + self.base[g]).exp()
            })
            .collect()
    }

    /// Gene `g`'s levels on every lineage, `([L·n], [L·n])`.
    pub fn gene_levels(&self, g: usize) -> (Vec<f64>, Vec<f64>) {
        let (mut u, mut s) = (Vec::new(), Vec::new());
        for l in 0..self.path.len() {
            let n = self.path[l].len();
            let h = self.t_len[l] / (n - 1) as f64;
            let (ul, sl) = trajectory(&self.alpha(g, l), self.beta[g], self.gamma[g], h);
            u.extend(ul);
            s.extend(sl);
        }
        (u, s)
    }

    /// Every gene's levels, in parallel over genes.
    pub fn levels(&self) -> Levels {
        let (lineages, genes) = (self.path.len(), self.beta.len());
        let points = self.path.first().map_or(0, Vec::len);
        let cols: Vec<(Vec<f64>, Vec<f64>)> = (0..genes)
            .into_par_iter()
            .map(|g| self.gene_levels(g))
            .collect();
        let rows = lineages * points;
        Levels {
            lineages,
            points,
            genes,
            u: DMatrix::from_fn(rows, genes, |r, g| cols[g].0[r]),
            s: DMatrix::from_fn(rows, genes, |r, g| cols[g].1[r]),
        }
    }
}
