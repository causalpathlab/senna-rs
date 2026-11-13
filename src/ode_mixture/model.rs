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
    /// The global log capture `c` of the unspliced reads against the
    /// spliced: every unspliced level is the kinetics' `ū` times `e^c`.
    pub log_capture: f64,
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
    #[cfg(test)]
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

    /// Every row's totals over the genes, `(Z_u, Z_s)`.
    pub fn totals(&self) -> Vec<(f64, f64)> {
        (0..self.u.nrows())
            .map(|r| (self.u.row(r).sum(), self.s.row(r).sum()))
            .collect()
    }
}

/// Lineage `l`'s name in the outputs, `L1…`. Lineage 0 is the trunk's;
/// [`Params::branch`]`[i]` is lineage `i + 1`'s.
pub fn lineage_label(l: usize) -> Box<str> {
    format!("L{}", l + 1).into()
}

/// Log rates, durations and jitters are clamped to `±LOG_CLAMP` before
/// `exp`: a large deviation saturates instead of overflowing.
pub const LOG_CLAMP: f64 = 30.0;

/// The log capture's prior, `c ~ N(0, 2²)`: weak, unspliced reads captured
/// within a factor of `e^{±4}` of the spliced.
pub const CAPTURE_PRIOR: (f64, f64) = (0.0, 2.0);

/// `exp(x)`, `x` clamped to `±`[`LOG_CLAMP`] (its slope zero outside).
pub fn clamped_exp(x: f64) -> f64 {
    x.clamp(-LOG_CLAMP, LOG_CLAMP).exp()
}

/// The transcription at every node of a path, `α_k = exp(ρᵀ z_k + b)`
/// ([`clamped_exp`]).
pub fn alpha_on(path: &[Vec<f64>], rho: &[f64], base: f64) -> Vec<f64> {
    path.iter()
        .map(|z| clamped_exp(z.iter().zip(rho).map(|(a, b)| a * b).sum::<f64>() + base))
        .collect()
}

impl Params {
    /// Positions per lineage, `(n − 1) sub + 1`.
    pub fn positions(&self) -> usize {
        (self.path[0].len() - 1) * self.sub.max(1) + 1
    }

    /// The time of lineage `l`'s position `k`.
    #[cfg(test)]
    pub fn time_at(&self, l: usize, k: usize) -> f64 {
        self.phase_at(k) * self.t_len[l]
    }

    /// Position `k`'s share of a lineage's grid, `k/(positions − 1)`: the
    /// clock every lineage shares and the branch times are on.
    pub fn phase_at(&self, k: usize) -> f64 {
        k as f64 / (self.positions() - 1).max(1) as f64
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
        let capture = self.capture();
        for (path, &t_len) in self.path.iter().zip(&self.t_len) {
            let alpha = alpha_on(path, rho, base);
            let h = t_len / (path.len() - 1) as f64;
            let (ul, sl) = fine_trajectory(&alpha, beta, gamma, h, self.sub);
            u.extend(ul.into_iter().map(|x| x * capture));
            s.extend(sl);
        }
        (u, s)
    }

    /// `e^c`, the unspliced reads' capture against the spliced
    /// ([`Self::log_capture`], [`clamped_exp`]).
    pub fn capture(&self) -> f64 {
        clamped_exp(self.log_capture)
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
            #[cfg(test)]
            lineages,
            points,
            genes,
            u: DMatrix::from_fn(rows, genes, |r, g| cols[g].0[r]),
            s: DMatrix::from_fn(rows, genes, |r, g| cols[g].1[r]),
        }
    }

    /// Every `(lineage, position)`'s velocities, rows as [`Levels`]:
    /// `dū/dt = α − β ū`, `ds̄/dt = β ū − γ s̄`, `α` the step's the
    /// position is in (the last step's at the last node).
    #[cfg(test)]
    pub fn velocities(&self) -> (DMatrix<f64>, DMatrix<f64>) {
        self.velocities_of(&self.levels())
    }

    /// [`Params::velocities`] from these parameters' levels `lv` (the
    /// kinetics' `ū`, the capture taken off).
    pub fn velocities_of(&self, lv: &Levels) -> (DMatrix<f64>, DMatrix<f64>) {
        let sub = self.sub.max(1);
        let capture = self.capture();
        let cols: Vec<(Vec<f64>, Vec<f64>)> = (0..lv.genes)
            .into_par_iter()
            .map(|g| {
                let (b, y) = (self.beta[g], self.gamma[g]);
                let (mut du, mut ds) = (Vec::new(), Vec::new());
                for (l, path) in self.path.iter().enumerate() {
                    let alpha = alpha_on(path, &self.rho[g], self.base[g]);
                    for k in 0..lv.points {
                        let a = alpha[(k / sub).min(path.len() - 2)];
                        let r = lv.row(l, k);
                        let (u, s) = (lv.u[(r, g)] / capture, lv.s[(r, g)]);
                        du.push(a - b * u);
                        ds.push(b * u - y * s);
                    }
                }
                (du, ds)
            })
            .collect();
        // Every gene's column, in column-major order.
        let rows = lv.u.nrows();
        let (du, ds): (Vec<Vec<f64>>, Vec<Vec<f64>>) = cols.into_iter().unzip();
        (
            DMatrix::from_vec(rows, lv.genes, du.concat()),
            DMatrix::from_vec(rows, lv.genes, ds.concat()),
        )
    }
}
