//! Every lineage's curve `μ_ℓ` in the embedding, on the ode run's grid.
//!
//! The grid is the ode run's positions, `n = (points − 1) sub + 1`, at phase
//! `k/(n − 1)`. Every dimension of a curve is a Matérn-5/2 path on phase
//! ([`GridPrior`], variance 1), carried whitened, `z = B η`, `η ~ N(0, I)`,
//! and scaled by one amplitude `a`, the embedding's scale not the prior's:
//!
//! ```text
//! no trunk:  μ_ℓ = a z_ℓ
//! trunk:     μ_0 = a z_0,   μ_ℓ = μ_0 + a r_ℓ ⊙ z_ℓ,   r_ℓ = branch_ramp(t_b,ℓ)
//! ```
//!
//! with the branch times `t_b` fixed (the ode run's), so `r` is a constant.
//! `B` is the prior's Cholesky factor on the positions; should it fail, the
//! factor on the `points` nodes alone, `B = A C`, `A` the straight line
//! between two nodes at every position between them.

use crate::ode_mixture::kernel::GridPrior;
use crate::ode_mixture::model::{branch_ramp, off_the_edges, phase, positions};
use legume_numeric::candle::candle_core::{Device, Result, Tensor};
use nalgebra::DMatrix;

/// What a set of curves is built from: the ode run's grid and trunk.
#[derive(Clone, Debug)]
pub struct CurveSpec {
    pub lineages: usize,
    /// The ode run's grid nodes and positions per step.
    pub points: usize,
    pub sub: usize,
    /// The length scale as a share of the duration.
    pub kappa: f64,
    /// `[L − 1]` lineage `i + 1`'s branch time off lineage 0, on phase;
    /// `None`: no trunk.
    pub branch: Option<Vec<f64>>,
}

impl CurveSpec {
    /// Positions per lineage, `(points − 1) sub + 1`.
    pub fn positions(&self) -> usize {
        positions(self.points, self.sub)
    }
}

/// The constant parts of every curve: the basis and the ramps.
pub struct Curves {
    lineages: usize,
    positions: usize,
    /// `[m × n]` `Bᵀ`, so a row-vector product `η Bᵀ` is `B η`.
    basis_t: Tensor,
    basis_size: usize,
    /// `[L − 1 × n]` every branch's ramp.
    ramp: Option<Tensor>,
    branch: Option<Vec<f64>>,
    on_nodes: bool,
}

impl Curves {
    /// On every position, or on the nodes alone if the prior on the
    /// positions has no Cholesky factor.
    pub fn new(spec: &CurveSpec, dev: &Device) -> anyhow::Result<Self> {
        let n = spec.positions();
        match GridPrior::new(n, 1.0, spec.kappa) {
            Ok(prior) => Self::build(spec, prior.chol().clone(), prior.kappa(), false, dev),
            Err(_) => Self::on_the_nodes(spec, dev),
        }
    }

    /// On the `points` nodes alone, straight lines between them.
    pub fn on_the_nodes(spec: &CurveSpec, dev: &Device) -> anyhow::Result<Self> {
        let (m, n, sub) = (spec.points.max(2), spec.positions(), spec.sub.max(1));
        let prior = GridPrior::new(m, 1.0, spec.kappa)?;
        let lerp = DMatrix::from_fn(n, m, |k, j| {
            let (node, f) = (k / sub, (k % sub) as f64 / sub as f64);
            if j == node {
                1.0 - f
            } else if j == node + 1 {
                f
            } else {
                0.0
            }
        });
        Self::build(spec, lerp * prior.chol(), prior.kappa(), true, dev)
    }

    /// From `basis [n × m]`, `z = basis η`.
    fn build(
        spec: &CurveSpec,
        basis: DMatrix<f64>,
        kappa: f64,
        on_nodes: bool,
        dev: &Device,
    ) -> anyhow::Result<Self> {
        let (n, m) = basis.shape();
        // nalgebra is column-major: its `iter` over `[n × m]` reads `Bᵀ` row
        // by row.
        let basis_t = Tensor::from_vec(
            basis.iter().map(|&v| v as f32).collect::<Vec<_>>(),
            (m, n),
            dev,
        )?;
        let branch: Option<Vec<f64>> = spec
            .branch
            .as_ref()
            .filter(|_| spec.lineages > 1)
            .map(|b| b.iter().map(|&t| off_the_edges(t)).collect());
        let ramp = match &branch {
            Some(tb) => {
                anyhow::ensure!(
                    tb.len() == spec.lineages - 1,
                    "{} branch times for {} lineages",
                    tb.len(),
                    spec.lineages
                );
                let r: Vec<f32> = tb
                    .iter()
                    .flat_map(|&t| branch_ramp(n, t, kappa))
                    .map(|v| v as f32)
                    .collect();
                Some(Tensor::from_vec(r, (tb.len(), n), dev)?)
            }
            None => None,
        };
        Ok(Self {
            lineages: spec.lineages,
            positions: n,
            basis_t,
            basis_size: m,
            ramp,
            branch,
            on_nodes,
        })
    }

    pub fn lineages(&self) -> usize {
        self.lineages
    }

    /// Points per curve, the ode run's positions.
    pub fn positions(&self) -> usize {
        self.positions
    }

    /// `η`'s length per lineage and dimension: the positions, or the nodes.
    pub fn basis_size(&self) -> usize {
        self.basis_size
    }

    /// Whether the curves live on the nodes alone.
    pub fn on_nodes(&self) -> bool {
        self.on_nodes
    }

    /// The branch times as used, clamped off the ends.
    pub fn branch_times(&self) -> Option<&[f64]> {
        self.branch.as_deref()
    }

    /// Position `k`'s phase, `k/(n − 1)`.
    pub fn phase(&self, k: usize) -> f64 {
        phase(k, self.positions)
    }

    /// `[L × n × H]` the curves from `eta [L × m × H]` and `log_a [1]`.
    pub fn mu(&self, eta: &Tensor, log_a: &Tensor) -> Result<Tensor> {
        let (l, m, h) = eta.dims3()?;
        let n = self.positions;
        let z = eta
            .transpose(1, 2)?
            .reshape((l * h, m))?
            .matmul(&self.basis_t)?
            .reshape((l, h, n))?
            .transpose(1, 2)?
            .broadcast_mul(&log_a.exp()?)?;
        let Some(ramp) = &self.ramp else {
            return Ok(z);
        };
        let trunk = z.narrow(0, 0, 1)?;
        let parted =
            trunk.broadcast_add(&z.narrow(0, 1, l - 1)?.broadcast_mul(&ramp.unsqueeze(2)?)?)?;
        Tensor::cat(&[&trunk, &parted], 0)
    }
}

/// `[L × n × H]` `dμ/dphase` from `mu [L × n × H]` (row-major) on a grid
/// of step `1/(n − 1)`: the central difference inside, the second-order
/// one-sided one at the ends.
pub fn grid_velocity(mu: &[f32], lineages: usize, n: usize, h: usize) -> Vec<f32> {
    let mut v = vec![0f32; lineages * n * h];
    if n < 2 {
        return v;
    }
    let scale = (n - 1) as f32;
    for l in 0..lineages {
        let at = |k: usize, d: usize| mu[(l * n + k) * h + d];
        for k in 0..n {
            for d in 0..h {
                let slope = if n == 2 {
                    at(1, d) - at(0, d)
                } else if k == 0 {
                    (-3.0 * at(0, d) + 4.0 * at(1, d) - at(2, d)) / 2.0
                } else if k == n - 1 {
                    (3.0 * at(k, d) - 4.0 * at(k - 1, d) + at(k - 2, d)) / 2.0
                } else {
                    (at(k + 1, d) - at(k - 1, d)) / 2.0
                };
                v[(l * n + k) * h + d] = slope * scale;
            }
        }
    }
    v
}
