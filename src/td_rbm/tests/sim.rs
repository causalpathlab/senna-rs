//! Cells drawn from the model, with the truth kept.
//!
//! Genes are embedded at `ρ_g ~ N(0, I)` in `H₀` dimensions; a cubic curve
//! `z(τ)` with random coefficients runs through that space. A cell on the
//! trajectory at time `τ_c ~ U(0, 1)` has spliced shares
//! `π = softmax(⟨z(τ), ρ_g⟩ + b_g)` and unspliced from the splicing half of
//! the ODE, `u_g/s_g = m⁺_g/β_g`, `m_g = γ_g + ⟨ρ_g − ρ̄, ż(τ)⟩`; a bystander
//! sits still at its own state `z_b` (no velocity). Reads are Poisson at the
//! cell's depth.

use crate::td_rbm::curve::Spline;
use legume_numeric::candle::candle_core::Device;
use rand::rngs::StdRng;
use rand::SeedableRng;
use rand_distr::{Distribution, Gamma, Normal, Poisson, Uniform};

#[derive(Clone, Debug)]
pub struct SimConfig {
    pub n_cells: usize,
    pub n_genes: usize,
    pub dim: usize,
    pub knots: usize,
    /// Scale of the curve's coefficients: how far the trajectory travels.
    pub travel: f64,
    pub mean_depth: f64,
    /// Share of cells that are bystanders.
    pub bystanders: f64,
    /// Independent trajectories (lineages), cells split evenly.
    pub lineages: usize,
    pub seed: u64,
}

impl Default for SimConfig {
    fn default() -> Self {
        Self {
            n_cells: 2000,
            n_genes: 120,
            dim: 3,
            knots: 5,
            travel: 1.0,
            mean_depth: 2000.0,
            bystanders: 0.0,
            lineages: 1,
            seed: 1,
        }
    }
}

#[derive(Clone)]
pub struct Sim {
    /// `[C × G]` row-major reads.
    pub unspliced: Vec<f32>,
    pub spliced: Vec<f32>,
    pub n_cells: usize,
    pub n_genes: usize,
    /// Per cell: true time (bystanders: NaN-free 0.5 placeholder) and whether
    /// it is a bystander.
    pub tau: Vec<f32>,
    pub bystander: Vec<bool>,
    /// Every cell's lineage.
    pub lineage: Vec<usize>,
    /// `[C × G]` every cell's true `d log s_g/dτ` (0 for bystanders).
    pub log_slope: Vec<Vec<f32>>,
    /// The truth: `[G × H₀]` embedding, `[M × H₀]` curve, `[G]` baselines and rates.
    pub rho: Vec<Vec<f64>>,
    pub coef: Vec<Vec<f64>>,
    pub base: Vec<f64>,
    pub beta: Vec<f64>,
    pub gamma: Vec<f64>,
}

fn softmax(x: &[f64]) -> Vec<f64> {
    let m = x.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let e: Vec<f64> = x.iter().map(|v| (v - m).exp()).collect();
    let s: f64 = e.iter().sum();
    e.into_iter().map(|v| v / s).collect()
}

pub fn simulate(cfg: &SimConfig) -> anyhow::Result<Sim> {
    let mut rng = StdRng::seed_from_u64(cfg.seed);
    let unit = Normal::new(0.0, 1.0)?;
    let (c, g, h) = (cfg.n_cells, cfg.n_genes, cfg.dim);
    let rho: Vec<Vec<f64>> = (0..g)
        .map(|_| (0..h).map(|_| unit.sample(&mut rng)).collect())
        .collect();
    let base: Vec<f64> = (0..g).map(|_| 0.5 * unit.sample(&mut rng)).collect();
    let log_rate = Uniform::new(0.0f64, 1.0)?;
    let beta: Vec<f64> = (0..g)
        .map(|_| (2f64.ln() + 1.5 * log_rate.sample(&mut rng)).exp())
        .collect();
    let gamma: Vec<f64> = (0..g)
        .map(|_| (0.5f64.ln() + 1.5 * log_rate.sample(&mut rng)).exp())
        .collect();
    // Per lineage a random walk of coefficients from its own start: a
    // smooth, travelling curve.
    let mut coefs = Vec::with_capacity(cfg.lineages);
    for _ in 0..cfg.lineages {
        let mut coef = vec![vec![0f64; h]; cfg.knots];
        for c in coef[0].iter_mut() {
            *c = if cfg.lineages > 1 {
                2.0 * unit.sample(&mut rng)
            } else {
                0.0
            };
        }
        for m in 1..cfg.knots {
            let (done, rest) = coef.split_at_mut(m);
            for (c, &prev) in rest[0].iter_mut().zip(&done[m - 1]) {
                *c = prev + cfg.travel * unit.sample(&mut rng);
            }
        }
        coefs.push(coef);
    }
    let coef = coefs[0].clone();
    let still: Vec<f64> = (0..h).map(|_| 1.5 * unit.sample(&mut rng)).collect();

    let uniform = Uniform::new(0.0f32, 1.0)?;
    let tau: Vec<f32> = (0..c).map(|_| uniform.sample(&mut rng)).collect();
    let bystander: Vec<bool> = (0..c)
        .map(|_| f64::from(uniform.sample(&mut rng)) < cfg.bystanders)
        .collect();
    let lineage: Vec<usize> = (0..c).map(|i| i % cfg.lineages).collect();
    let spline = Spline::new(cfg.knots, &tau, &Device::Cpu)?;
    let (bv, bs) = (
        spline.values.to_vec2::<f32>()?,
        spline.slopes.to_vec2::<f32>()?,
    );
    let depth_draw = Gamma::new(4.0, cfg.mean_depth / 4.0)?;

    let (mut unspliced, mut spliced) = (vec![0f32; c * g], vec![0f32; c * g]);
    let mut log_slope = Vec::with_capacity(c);
    for i in 0..c {
        let (z, v): (Vec<f64>, Vec<f64>) = if bystander[i] {
            (still.clone(), vec![0.0; h])
        } else {
            let coef = &coefs[lineage[i]];
            let at = |b: &[f32], d: usize| {
                (0..cfg.knots)
                    .map(|m| f64::from(b[m]) * coef[m][d])
                    .sum::<f64>()
            };
            (
                (0..h).map(|d| at(&bv[i], d)).collect(),
                (0..h).map(|d| at(&bs[i], d)).collect(),
            )
        };
        let dot = |a: &[f64], b: &[f64]| a.iter().zip(b).map(|(x, y)| x * y).sum::<f64>();
        let pi = softmax(
            &(0..g)
                .map(|j| dot(&z, &rho[j]) + base[j])
                .collect::<Vec<_>>(),
        );
        let rho_bar: Vec<f64> = (0..h)
            .map(|d| (0..g).map(|j| pi[j] * rho[j][d]).sum())
            .collect();
        let slope: Vec<f64> = (0..g)
            .map(|j| dot(&rho[j], &v) - dot(&rho_bar, &v))
            .collect();
        // The same smooth floor as the model.
        let ratio: Vec<f64> = (0..g)
            .map(|j| {
                let (m, soft) = (gamma[j] + slope[j], 0.05 * gamma[j]);
                let sp = (m / soft).max(0.0) + (-(m / soft).abs()).exp().ln_1p();
                soft * sp / beta[j]
            })
            .collect();
        let weight: Vec<f64> = (0..g).map(|j| pi[j] * (1.0 + ratio[j])).collect();
        let total: f64 = weight.iter().sum();
        let depth = depth_draw.sample(&mut rng);
        for j in 0..g {
            let lam_s = depth * pi[j] / total;
            let lam_u = lam_s * ratio[j];
            spliced[i * g + j] = Poisson::new(lam_s.max(1e-9))?.sample(&mut rng) as f32;
            unspliced[i * g + j] = Poisson::new(lam_u.max(1e-9))?.sample(&mut rng) as f32;
        }
        log_slope.push(slope.iter().map(|&x| x as f32).collect());
    }
    Ok(Sim {
        unspliced,
        spliced,
        n_cells: c,
        n_genes: g,
        tau,
        bystander,
        lineage,
        log_slope,
        rho,
        coef,
        base,
        beta,
        gamma,
    })
}

/// Pearson correlation of two equal-length series; centring removes the
/// common shift `ρ̄`.
pub fn pearson(a: &[f32], b: &[f32]) -> f64 {
    let n = a.len() as f64;
    let ma = a.iter().map(|&x| f64::from(x)).sum::<f64>() / n;
    let mb = b.iter().map(|&x| f64::from(x)).sum::<f64>() / n;
    let (mut c, mut va, mut vb) = (0.0, 0.0, 0.0);
    for (&x, &y) in a.iter().zip(b) {
        let (x, y) = (f64::from(x) - ma, f64::from(y) - mb);
        c += x * y;
        va += x * x;
        vb += y * y;
    }
    c / (va * vb).sqrt().max(1e-300)
}
