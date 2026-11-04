//! The parameters and the levels they set on the grid.
//!
//! Every lineage `ℓ` has `n` grid points `t_k = k T_ℓ/(n − 1)`; at each the
//! transcription `α_gℓk = exp(ρ_gᵀ z_ℓ(t_k) + b_g)` and the trajectory's
//! levels `ū_gℓk`, `s̄_gℓk` ([`super::kinetics`]), shared rates `β_g`, `γ_g`.

use super::kinetics::fine_trajectory;
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
    /// Positions per grid step a pseudobulk's time can take (`1`: the nodes
    /// alone). The levels are exact between nodes, so time is as fine as
    /// this, the path no finer.
    pub sub: usize,
    /// `[L − 1]` every other lineage's branch time off lineage 0 on one
    /// clock (every duration the unit): `z_ℓ = z_0 + √2 r(t) w_ℓ`
    /// ([`branch_ramp`]); `None`: the lineages apart.
    pub branch: Option<Vec<f64>>,
}

/// `r(t_k) = √2 (1 − e^{−((t_k − t_b)₊/κ)²})` on `n` grid points over
/// `[0, 1]`: a lineage's departure from the trunk is `r ⊙ w`, `w` a path
/// from the prior, so it parts smoothly at `t_b` and far past it is as far
/// from the trunk as two independent paths are from each other.
pub fn branch_ramp(n: usize, tb: f64, kappa: f64) -> Vec<f64> {
    (0..n)
        .map(|k| {
            let x = (k as f64 / (n - 1) as f64 - tb).max(0.0) / kappa;
            std::f64::consts::SQRT_2 * (1.0 - (-x * x).exp())
        })
        .collect()
}

/// Every `(lineage, position)`'s levels, rows `ℓ points + k` (positions
/// [`Params::positions`]: the nodes and `sub − 1` between each two), a
/// column per gene.
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
    /// Positions per lineage, `(n − 1) sub + 1`.
    pub fn positions(&self) -> usize {
        (self.path[0].len() - 1) * self.sub.max(1) + 1
    }

    /// The time of lineage `l`'s position `k`.
    pub fn time_at(&self, l: usize, k: usize) -> f64 {
        k as f64 * self.t_len[l] / (self.positions() - 1) as f64
    }

    /// The paths' mean over every lineage's grid points, `[H]`.
    pub fn path_mean(&self) -> Vec<f64> {
        let h = self.rho.first().map_or(0, Vec::len);
        let points: Vec<&Vec<f64>> = self.path.iter().flatten().collect();
        (0..h)
            .map(|d| points.iter().map(|z| z[d]).sum::<f64>() / points.len().max(1) as f64)
            .collect()
    }

    /// Gene `g`'s levels on every lineage, `([L·n], [L·n])`.
    pub fn gene_levels(&self, g: usize) -> (Vec<f64>, Vec<f64>) {
        self.levels_of(self.beta[g], self.gamma[g], &self.rho[g], self.base[g])
    }

    /// The levels on every lineage of a gene with these rates, embedding and
    /// baseline, on the current paths.
    pub fn levels_of(&self, beta: f64, gamma: f64, rho: &[f64], base: f64) -> (Vec<f64>, Vec<f64>) {
        let (mut u, mut s) = (Vec::new(), Vec::new());
        for (path, &t_len) in self.path.iter().zip(&self.t_len) {
            let alpha: Vec<f64> = path
                .iter()
                .map(|z| (z.iter().zip(rho).map(|(a, b)| a * b).sum::<f64>() + base).exp())
                .collect();
            let h = t_len / (path.len() - 1) as f64;
            let (ul, sl) = fine_trajectory(&alpha, beta, gamma, h, self.sub);
            u.extend(ul);
            s.extend(sl);
        }
        (u, s)
    }

    /// Every gene's levels, in parallel over genes.
    pub fn levels(&self) -> Levels {
        let (lineages, genes) = (self.path.len(), self.beta.len());
        let points = if self.path.is_empty() {
            0
        } else {
            self.positions()
        };
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
