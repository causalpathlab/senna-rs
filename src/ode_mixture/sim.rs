//! Pseudobulks drawn from the mixture of ODEs, the truth kept.
//!
//! Genes: `β_g`, `γ_g`, an embedding `ρ_g ~ N(0, σ²I)` and a baseline `b_g`.
//! Every lineage: a duration `T_ℓ` and a path `z_ℓ` from the Matérn prior
//! ([`super::kernel`]) on a fine grid of its own (the model fits on a
//! coarser one), transcription `α = exp(ρ_gᵀ z_ℓ + b_g)` and the exact
//! trajectory ([`super::kinetics`]). A pseudobulk is a lineage, a time
//! uniform on the fine grid and a depth `N_p`; its reads are Poisson at
//! `λ_p ū`, `λ_p s̄`, `λ_p = N_p / Σ_g(ū + s̄)`.
//!
//! Branching ([`SimConfig::branch_at`]): every lineage `ℓ > 0` is lineage
//! 0's path plus `√2 r(t) w_ℓ(t)`, `w_ℓ` a draw from the prior and
//! `r(t) = 1 − e^{−((t − t_b)/κ)²}` past the branch `t_b` (0 before), so
//! value and slope are continuous there; far past it the lineages are as
//! far apart as two independent draws. Every lineage starts at steady state
//! on the same path, so the trunk's levels are shared too.

use super::kernel::GridPrior;
use super::kinetics::trajectory;
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};
use rand_distr::{Distribution, Gamma, Normal, Poisson, Uniform};

#[derive(Clone, Debug)]
pub struct SimConfig {
    pub n_genes: usize,
    pub n_pb: usize,
    /// The paths' dimension `H`.
    pub dim: usize,
    pub lineages: usize,
    /// Mean reads per pseudobulk.
    pub depth: f64,
    /// Points on the simulator's own grid.
    pub fine: usize,
    /// Every lineage's duration (lineage 1's is the unit) and the length
    /// scale as a share of it.
    pub t_len: f64,
    pub kappa: f64,
    /// Sd of the embedding's entries: how far transcription swings.
    pub rho_sd: f64,
    /// Every gene at steady state at every time (no dynamics to read).
    pub steady: bool,
    /// Every gene's unspliced reads scaled by `e^{d (t/T − ½)}`: a drift of
    /// the unspliced share shared by every gene, no kinetics in it.
    pub unspliced_drift: f64,
    /// With `Some(f)` and two lineages or more, every lineage on lineage
    /// 0's path until `f T`, apart after.
    pub branch_at: Option<f64>,
    pub seed: u64,
}

impl Default for SimConfig {
    fn default() -> Self {
        Self {
            n_genes: 100,
            n_pb: 300,
            dim: 3,
            lineages: 1,
            depth: 5000.0,
            fine: 400,
            t_len: 1.0,
            kappa: 0.25,
            rho_sd: 0.7,
            steady: false,
            unspliced_drift: 0.0,
            branch_at: None,
            seed: 1,
        }
    }
}

/// What the reads were drawn from.
pub struct Truth {
    pub beta: Vec<f64>,
    pub gamma: Vec<f64>,
    /// `[G][H]`.
    pub rho: Vec<Vec<f64>>,
    pub base: Vec<f64>,
    pub t_len: Vec<f64>,
    /// Every lineage's path, `[fine][H]`.
    pub path: Vec<Vec<Vec<f64>>>,
    /// Every pseudobulk's lineage and time.
    pub lineage: Vec<usize>,
    pub time: Vec<f64>,
    /// Every pseudobulk on the shared trunk (at or before the branch).
    pub trunk: Vec<bool>,
    /// `[P × G]` the reads' means.
    pub u_mean: Vec<f64>,
    pub s_mean: Vec<f64>,
}

pub struct Sim {
    /// `[P × G]` row-major reads.
    pub u: Vec<f64>,
    pub s: Vec<f64>,
    pub n_pb: usize,
    pub n_genes: usize,
    /// Every pseudobulk's total reads `N_p`.
    pub depth: Vec<f64>,
    pub truth: Truth,
}

pub fn simulate(cfg: &SimConfig) -> anyhow::Result<Sim> {
    let mut rng = StdRng::seed_from_u64(cfg.seed);
    let (g, h, l, n) = (cfg.n_genes, cfg.dim, cfg.lineages, cfg.fine);
    let unit = Normal::new(0.0, 1.0)?;
    let span = Uniform::new(0.0f64, 1.0)?;
    // Decay over one lineage from 1 to 10 per unit time; splicing faster.
    let gamma: Vec<f64> = (0..g)
        .map(|_| (10f64.ln() * span.sample(&mut rng)).exp() / cfg.t_len)
        .collect();
    let beta: Vec<f64> = gamma
        .iter()
        .map(|&y| y * (4f64.ln() * span.sample(&mut rng)).exp())
        .collect();
    let rho: Vec<Vec<f64>> = (0..g)
        .map(|_| (0..h).map(|_| cfg.rho_sd * unit.sample(&mut rng)).collect())
        .collect();
    let base: Vec<f64> = (0..g).map(|_| 0.5 * unit.sample(&mut rng)).collect();

    let t_len = vec![cfg.t_len; l];
    let prior = GridPrior::new(n, cfg.t_len, cfg.kappa * cfg.t_len)?;
    let mut path: Vec<Vec<Vec<f64>>> = (0..l)
        .map(|_| {
            let dims: Vec<_> = (0..h).map(|_| prior.draw(&mut rng)).collect();
            (0..n)
                .map(|k| dims.iter().map(|d| d[k]).collect())
                .collect()
        })
        .collect();
    let step = cfg.t_len / (n - 1) as f64;
    let branch = cfg.branch_at.filter(|_| l > 1).map(|f| f * cfg.t_len);
    if let Some(tb) = branch {
        let width = cfg.kappa * cfg.t_len;
        let (trunk, others) = path.split_at_mut(1);
        for other in others {
            for (k, (z, z0)) in other.iter_mut().zip(&trunk[0]).enumerate() {
                let after = (k as f64 * step - tb).max(0.0) / width;
                let ramp = std::f64::consts::SQRT_2 * (1.0 - (-after * after).exp());
                for (zd, z0d) in z.iter_mut().zip(z0) {
                    *zd = z0d + ramp * *zd;
                }
            }
        }
    }
    // Every (lineage, gene)'s trajectory on the fine grid, `[L][G][n]`.
    let traj: Vec<Vec<(Vec<f64>, Vec<f64>)>> = path
        .iter()
        .map(|z| {
            (0..g)
                .map(|j| {
                    let alpha: Vec<f64> = z
                        .iter()
                        .map(|zk| {
                            let dot: f64 = zk.iter().zip(&rho[j]).map(|(a, b)| a * b).sum();
                            (dot + base[j]).exp()
                        })
                        .collect();
                    if cfg.steady {
                        (
                            alpha.iter().map(|a| a / beta[j]).collect(),
                            alpha.iter().map(|a| a / gamma[j]).collect(),
                        )
                    } else {
                        trajectory(&alpha, beta[j], gamma[j], step)
                    }
                })
                .collect()
        })
        .collect();

    let depth_draw = Gamma::new(4.0, cfg.depth / 4.0)?;
    let p_n = cfg.n_pb;
    let (mut u, mut s) = (vec![0f64; p_n * g], vec![0f64; p_n * g]);
    let (mut u_mean, mut s_mean) = (vec![0f64; p_n * g], vec![0f64; p_n * g]);
    let (mut lineage, mut time, mut depth) = (vec![0; p_n], vec![0f64; p_n], vec![0f64; p_n]);
    for p in 0..p_n {
        let lin = p % l;
        let k = rng.random_range(0..n);
        let t = k as f64 * step;
        let drift = (cfg.unspliced_drift * (t / t_len[lin] - 0.5)).exp();
        let total: f64 = (0..g)
            .map(|j| traj[lin][j].0[k] * drift + traj[lin][j].1[k])
            .sum();
        let n_p = depth_draw.sample(&mut rng);
        let lambda = n_p / total;
        for j in 0..g {
            let (mu, ms) = (
                lambda * traj[lin][j].0[k] * drift,
                lambda * traj[lin][j].1[k],
            );
            u_mean[p * g + j] = mu;
            s_mean[p * g + j] = ms;
            u[p * g + j] = Poisson::new(mu.max(1e-12))?.sample(&mut rng);
            s[p * g + j] = Poisson::new(ms.max(1e-12))?.sample(&mut rng);
        }
        (lineage[p], time[p], depth[p]) = (lin, t, n_p);
    }
    let trunk = time
        .iter()
        .map(|&t| branch.is_some_and(|tb| t <= tb))
        .collect();
    Ok(Sim {
        u,
        s,
        n_pb: p_n,
        n_genes: g,
        depth,
        truth: Truth {
            beta,
            gamma,
            rho,
            base,
            t_len,
            path,
            lineage,
            time,
            trunk,
            u_mean,
            s_mean,
        },
    })
}
