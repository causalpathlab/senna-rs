//! Pseudobulks drawn from the stage-1 model, with the truth kept: for judging
//! how well the fit recovers each pseudobulk's time and each module's kinetics.
//!
//! Each pseudobulk `p` gets a time `τ_p ~ U(0, 1)` and a capture offset
//! `κ_p ~ N(0, σ_κ²)`; each gene `g` a module, a steady offset
//! `b_g ~ N(b̄, σ_b²)` and a read depth `μ_g = μ̄ e^{N(0, σ_μ²)}`. A pseudobulk
//! reads `n_pg ~ Poisson(μ_g ℓ_m(τ_p))` of the gene, `ℓ_m` its module's level
//! `u_m + s_m` over its mean across the pseudobulks, of which
//! `x^u_pg ~ Binom(n_pg, σ(κ_p + b_g + log u_m(τ_p) − log s_m(τ_p)))` are
//! unspliced.

use super::kinetics::{curves, Kinetics, ModuleKinetics};
use legume_numeric::candle::candle_core::{DType, Device, Tensor};
use rand::rngs::StdRng;
use rand::SeedableRng;
use rand_distr::{Binomial, Distribution, Normal, Poisson, Uniform};

#[derive(Clone, Debug)]
pub struct SimConfig {
    pub n_pb: usize,
    pub genes_per_module: usize,
    /// `μ̄`: a gene's typical reads per pseudobulk, both tracks together.
    pub mean_reads: f64,
    /// `σ_μ`: spread of the genes' log depth.
    pub depth_sd: f64,
    /// `b̄`, `σ_b`: the genes' steady offsets.
    pub offset_mean: f64,
    pub offset_sd: f64,
    /// `σ_κ`: spread of the pseudobulks' capture offsets.
    pub kappa_sd: f64,
    pub seed: u64,
}

impl Default for SimConfig {
    fn default() -> Self {
        Self {
            n_pb: 800,
            genes_per_module: 40,
            mean_reads: 20.0,
            depth_sd: 1.0,
            offset_mean: 0.0,
            offset_sd: 0.5,
            kappa_sd: 0.3,
            seed: 1,
        }
    }
}

/// Simulated counts and the truth behind them.
pub struct Simulated {
    /// `[P × G]` row-major unspliced and spliced reads.
    pub unspliced: Vec<f32>,
    pub spliced: Vec<f32>,
    pub n_pb: usize,
    pub n_genes: usize,
    pub module_of_gene: Vec<u32>,
    pub tau: Vec<f32>,
    pub kappa: Vec<f32>,
    pub offset: Vec<f32>,
    pub modules: Vec<ModuleKinetics>,
}

pub fn simulate(modules: &[ModuleKinetics], cfg: &SimConfig) -> anyhow::Result<Simulated> {
    let mut rng = StdRng::seed_from_u64(cfg.seed);
    let unit = Normal::new(0.0, 1.0)?;
    let (p, m) = (cfg.n_pb, modules.len());
    let g = m * cfg.genes_per_module;
    let module_of_gene: Vec<u32> = (0..g).map(|j| (j / cfg.genes_per_module) as u32).collect();
    let uniform = Uniform::new(0.0f32, 1.0)?;
    let tau: Vec<f32> = (0..p).map(|_| uniform.sample(&mut rng)).collect();
    let kappa: Vec<f32> = (0..p)
        .map(|_| (cfg.kappa_sd * unit.sample(&mut rng)) as f32)
        .collect();
    let offset: Vec<f32> = (0..g)
        .map(|_| (cfg.offset_mean + cfg.offset_sd * unit.sample(&mut rng)) as f32)
        .collect();
    let depth: Vec<f64> = (0..g)
        .map(|_| cfg.mean_reads * (cfg.depth_sd * unit.sample(&mut rng)).exp())
        .collect();

    let dev = Device::Cpu;
    let k = Kinetics::from_modules(modules, DType::F64, &dev)?;
    let t = Tensor::from_vec(tau.clone(), p, &dev)?.to_dtype(DType::F64)?;
    let (u, s) = curves(&k, &t)?;
    let ratio = (u.log()? - s.log()?)?.to_vec2::<f64>()?;
    let level = (&u + &s)?;
    let level = level
        .broadcast_div(&level.mean_keepdim(0)?)?
        .to_vec2::<f64>()?;

    let mut unspliced = vec![0f32; p * g];
    let mut spliced = vec![0f32; p * g];
    for i in 0..p {
        for j in 0..g {
            let mo = module_of_gene[j] as usize;
            let logit = f64::from(kappa[i] + offset[j]) + ratio[i][mo];
            let n = Poisson::new(depth[j] * level[i][mo])?.sample(&mut rng) as u64;
            let share = 1.0 / (1.0 + (-logit).exp());
            let xu = Binomial::new(n, share)?.sample(&mut rng);
            unspliced[i * g + j] = xu as f32;
            spliced[i * g + j] = (n - xu) as f32;
        }
    }
    Ok(Simulated {
        unspliced,
        spliced,
        n_pb: p,
        n_genes: g,
        module_of_gene,
        tau,
        kappa,
        offset,
        modules: modules.to_vec(),
    })
}

/// A world for the joint fit: units on one curve `θ^k(τ) = W φ^k(τ)`, genes
/// that follow their own module (some of them the other way) and a little of
/// the rest.
#[derive(Clone, Debug)]
pub struct JointSimConfig {
    pub n_pb: usize,
    pub genes_per_module: usize,
    /// Width of the embedding.
    pub h: usize,
    /// Reads per unit over every (gene, track) row.
    pub reads_per_pb: f64,
    pub seed: u64,
}

pub struct JointSimulated {
    /// `[P × G]` row-major.
    pub unspliced: Vec<f32>,
    pub spliced: Vec<f32>,
    pub n_pb: usize,
    pub n_genes: usize,
    /// Each gene's own module.
    pub module_of_gene: Vec<u32>,
    pub tau: Vec<f32>,
    /// `[M × H]`, `[G × H]`, `[G]`.
    pub directions: nalgebra::DMatrix<f32>,
    pub rho: nalgebra::DMatrix<f32>,
    pub bias: Vec<f32>,
    /// `[P × H]` every unit's state `θ^s(τ_p)` and velocity `dθ^s/dτ`.
    pub theta: nalgebra::DMatrix<f32>,
    pub velocity: nalgebra::DMatrix<f32>,
    pub modules: Vec<ModuleKinetics>,
}

/// Reads from `score(g, k | p) = ⟨W φ^k(τ_p), ρ_g⟩ + b_g + [k=u](b^u_g + κ_p)`
/// (see [`super::joint`]), Poisson at `reads_per_pb` times the softmax over
/// every row.
pub fn simulate_joint(
    modules: &[ModuleKinetics],
    cfg: &JointSimConfig,
) -> anyhow::Result<JointSimulated> {
    use super::kinetics::spliced_log_rate;
    use nalgebra::DMatrix;
    let mut rng = StdRng::seed_from_u64(cfg.seed);
    let unit = Normal::new(0.0f32, 1.0)?;
    let uniform = Uniform::new(0.0f32, 1.0)?;
    let (p, m, h) = (cfg.n_pb, modules.len(), cfg.h);
    let g = m * cfg.genes_per_module;
    let module_of_gene: Vec<u32> = (0..g).map(|j| (j / cfg.genes_per_module) as u32).collect();
    // Orthonormal module directions.
    let raw = DMatrix::<f32>::from_fn(h, m, |_, _| unit.sample(&mut rng));
    let directions = raw.qr().q().transpose(); // [M × H]
    let rho = DMatrix::<f32>::from_fn(g, h, |j, k| {
        let own = module_of_gene[j] as usize;
        let sign = if j % 5 == 0 { -1.0 } else { 1.0 };
        sign * directions[(own, k)] + 0.2 * unit.sample(&mut rng)
    });
    let bias: Vec<f32> = (0..g).map(|_| unit.sample(&mut rng)).collect();
    let bias_u: Vec<f32> = (0..g).map(|_| -1.5 + 0.3 * unit.sample(&mut rng)).collect();
    let kappa: Vec<f32> = (0..p).map(|_| 0.2 * unit.sample(&mut rng)).collect();
    let tau: Vec<f32> = (0..p).map(|_| uniform.sample(&mut rng)).collect();

    let dev = Device::Cpu;
    let k = Kinetics::from_modules(modules, DType::F32, &dev)?;
    let t = Tensor::from_vec(tau.clone(), p, &dev)?;
    let (u, s) = curves(&k, &t)?;
    let to = |x: &Tensor| -> anyhow::Result<DMatrix<f32>> {
        Ok(DMatrix::from_row_iterator(
            p,
            m,
            x.flatten_all()?.to_vec1::<f32>()?,
        ))
    };
    let (ls, lu) = (to(&s.log()?)?, to(&u.log()?)?);
    let theta = &ls * &directions;
    let velocity = to(&spliced_log_rate(&k, &t)?)? * &directions;
    let score_s = &theta * rho.transpose();
    let score_u = (&lu * &directions) * rho.transpose();
    let mut unspliced = vec![0f32; p * g];
    let mut spliced = vec![0f32; p * g];
    for i in 0..p {
        let ss: Vec<f64> = (0..g)
            .map(|j| f64::from(score_s[(i, j)] + bias[j]))
            .collect();
        let su: Vec<f64> = (0..g)
            .map(|j| f64::from(score_u[(i, j)] + bias[j] + bias_u[j] + kappa[i]))
            .collect();
        let mx = ss.iter().chain(&su).copied().fold(f64::MIN, f64::max);
        let z: f64 = ss.iter().chain(&su).map(|x| (x - mx).exp()).sum();
        let mut draw = |x: f64| -> anyhow::Result<f32> {
            let rate = cfg.reads_per_pb * (x - mx).exp() / z;
            Ok(if rate > 0.0 {
                Poisson::new(rate)?.sample(&mut rng) as f32
            } else {
                0.0
            })
        };
        for j in 0..g {
            spliced[i * g + j] = draw(ss[j])?;
            unspliced[i * g + j] = draw(su[j])?;
        }
    }
    Ok(JointSimulated {
        unspliced,
        spliced,
        n_pb: p,
        n_genes: g,
        module_of_gene,
        tau,
        directions,
        rho,
        bias,
        theta,
        velocity,
        modules: modules.to_vec(),
    })
}
