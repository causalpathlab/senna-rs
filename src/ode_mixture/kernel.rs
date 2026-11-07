//! A Gaussian process prior over a lineage path on the time grid.
//!
//! Every dimension of a path `z_ℓ` is a Matérn-5/2 process on the lineage's
//! own time, variance 1 (the scale lives in the genes' embedding):
//!
//! ```text
//! k(t, t') = (1 + √5 r/κ + 5r²/(3κ²)) e^{−√5 r/κ},   r = |t − t'|
//! ```
//!
//! on the grid `t_k = k T/(n − 1)`; it depends on `κ/T` alone. Paths are
//! carried whitened, `z = C η`, `η ~ N(0, I)`, `C` the Cholesky factor.

use super::band::Band;
use nalgebra::{DMatrix, DVector};
use rand::Rng;
use rand_distr::{Distribution, StandardNormal};

/// Added to the diagonal: a smooth kernel on a fine grid is near singular.
const JITTER: f64 = 1e-8;

/// The Matérn-5/2 correlation at distance `r`, length scale `kappa`.
pub fn matern52(r: f64, kappa: f64) -> f64 {
    let x = 5f64.sqrt() * r.abs() / kappa;
    (1.0 + x + x * x / 3.0) * (-x).exp()
}

/// The `[n × n]` covariance on the grid of `n` points over `[0, t_len]`.
pub fn grid_cov(n: usize, t_len: f64, kappa: f64) -> DMatrix<f64> {
    let step = t_len / (n.max(2) - 1) as f64;
    DMatrix::from_fn(n, n, |i, j| {
        matern52((i as f64 - j as f64) * step, kappa) + if i == j { JITTER } else { 0.0 }
    })
}

/// Nodes a path's banded precisions reach either way: the prior's
/// ([`markov`]) and the reads' Fisher information in the path. Matérn-5/2
/// is near Markov on the grid: conditioned on this many nodes before, a
/// node's whitened precision is within some tens of percent of the exact.
pub const BAND: usize = 16;

/// The precision of `N(0, K)` banded to `b` nodes by conditioning every
/// node on the `b` before it (Vecchia): `Σ_k (e_k − a_k)(e_k − a_k)ᵀ/σ_k²`,
/// `a_k` its regression on them and `σ_k²` what is left. Positive definite
/// whatever `b`; exact as `b` reaches `n`.
pub fn markov(k: &DMatrix<f64>, b: usize) -> Band {
    let n = k.nrows();
    let mut q = Band::zeros(n, b);
    for i in 0..n {
        let lo = i.saturating_sub(b);
        let m = i - lo;
        let sub = k.view((lo, lo), (m, m)).into_owned();
        let cross = DVector::from_fn(m, |r, _| k[(lo + r, i)]);
        let a = sub
            .cholesky()
            .map_or(DVector::zeros(m), |c| c.solve(&cross));
        let var = (k[(i, i)] - cross.dot(&a)).max(JITTER);
        let e: Vec<(usize, f64)> = (0..m)
            .map(|r| (lo + r, -a[r]))
            .chain(std::iter::once((i, 1.0)))
            .collect();
        for &(x, u) in &e {
            for &(y, v) in &e {
                if x >= y {
                    q.add(x, y, u * v / var);
                }
            }
        }
    }
    q
}

/// The prior on one grid: its Cholesky factor, to colour and whiten.
pub struct GridPrior {
    chol: DMatrix<f64>,
    /// The prior's precision banded to [`BAND`] nodes ([`markov`]).
    markov: Band,
    /// The length scale over the duration.
    kappa: f64,
}

impl GridPrior {
    pub fn new(n: usize, t_len: f64, kappa: f64) -> anyhow::Result<Self> {
        let cov = grid_cov(n, t_len, kappa);
        let chol = cov
            .clone()
            .cholesky()
            .ok_or_else(|| anyhow::anyhow!("grid covariance not positive definite"))?
            .l();
        Ok(Self {
            markov: markov(&cov, BAND),
            chol,
            kappa: kappa / t_len,
        })
    }

    /// The length scale as a share of the duration.
    pub fn kappa(&self) -> f64 {
        self.kappa
    }

    /// The Cholesky factor `C`, `[n × n]` lower triangular.
    pub fn chol(&self) -> &DMatrix<f64> {
        &self.chol
    }

    /// The prior's precision banded to [`BAND`] nodes, positive definite
    /// ([`markov`]).
    pub fn markov(&self) -> &Band {
        &self.markov
    }

    /// `z = C η`.
    pub fn colour(&self, eta: &DVector<f64>) -> DVector<f64> {
        &self.chol * eta
    }

    /// `η = C⁻¹ z`.
    pub fn whiten(&self, z: &DVector<f64>) -> DVector<f64> {
        self.chol
            .solve_lower_triangular(z)
            .expect("a Cholesky factor has a nonzero diagonal")
    }

    /// The most probable `η` under `N(0, I)` given `z ≈ diag(scale) C η`
    /// to within `eps`: `argmin |diag(scale) Cη − z|²/ε² + |η|²`. Smooth
    /// where `scale` vanishes (those points say nothing of `η`), as
    /// dividing by it is not.
    pub fn whiten_scaled(&self, scale: &[f64], z: &DVector<f64>, eps: f64) -> DVector<f64> {
        let n = self.chol.nrows();
        let a = DMatrix::from_fn(n, n, |i, j| scale[i] * self.chol[(i, j)]);
        let normal = (a.transpose() * &a) / (eps * eps) + DMatrix::identity(n, n);
        normal
            .cholesky()
            .expect("I plus a Gram matrix is positive definite")
            .solve(&(a.transpose() * z / (eps * eps)))
    }

    /// One draw of the path.
    pub fn draw(&self, rng: &mut impl Rng) -> DVector<f64> {
        let eta = DVector::from_fn(self.chol.nrows(), |_, _| StandardNormal.sample(rng));
        self.colour(&eta)
    }
}
