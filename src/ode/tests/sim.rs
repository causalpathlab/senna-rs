//! Pseudobulks drawn from the stage-1 model, with the truth kept: for judging
//! how well the fit recovers each pseudobulk's time and each module's kinetics.
//!
//! Each pseudobulk `p` gets a time `τ_p ~ U(0, 1)`; each gene `g` a module,
//! its own rates around its module's, a steady offset
//! `b_g ~ N(b̄, σ_b²)` and a read depth `μ_g = μ̄ e^{N(0, σ_μ²)}`. A pseudobulk
//! reads `n_pg ~ Poisson(μ_g ℓ_g(τ_p))` of the gene, `ℓ_g` its level
//! `u_g + s_g` over its mean across the pseudobulks, of which
//! `x^u_pg ~ Binom(n_pg, σ(b_g + log u_g(τ_p) − log s_g(τ_p)))` are
//! unspliced.

use super::fit::GeneRates;
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
    /// Spread of a gene's log rates around its module's.
    pub rate_sd: f64,
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
            rate_sd: 0.3,
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
    pub offset: Vec<f32>,
    pub modules: Vec<ModuleKinetics>,
    /// Every gene's own rates: its module's, scattered log-normally.
    pub rates: Vec<GeneRates>,
}

pub fn simulate(modules: &[ModuleKinetics], cfg: &SimConfig) -> anyhow::Result<Simulated> {
    let mut rng = StdRng::seed_from_u64(cfg.seed);
    let unit = Normal::new(0.0, 1.0)?;
    let (p, m) = (cfg.n_pb, modules.len());
    let g = m * cfg.genes_per_module;
    let module_of_gene: Vec<u32> = (0..g).map(|j| (j / cfg.genes_per_module) as u32).collect();
    let uniform = Uniform::new(0.0f32, 1.0)?;
    let tau: Vec<f32> = (0..p).map(|_| uniform.sample(&mut rng)).collect();
    let offset: Vec<f32> = (0..g)
        .map(|_| (cfg.offset_mean + cfg.offset_sd * unit.sample(&mut rng)) as f32)
        .collect();
    let depth: Vec<f64> = (0..g)
        .map(|_| cfg.mean_reads * (cfg.depth_sd * unit.sample(&mut rng)).exp())
        .collect();

    let rates: Vec<GeneRates> = module_of_gene
        .iter()
        .map(|&mo| {
            let m = &modules[mo as usize];
            let jitter = |x: f32, rng: &mut StdRng| {
                (f64::from(x) * (cfg.rate_sd * unit.sample(rng)).exp()) as f32
            };
            GeneRates {
                beta: jitter(m.beta, &mut rng),
                gamma: jitter(m.gamma, &mut rng),
            }
        })
        .collect();
    // Every gene's curve: its module's transcription through its own rates.
    let per_gene: Vec<ModuleKinetics> = module_of_gene
        .iter()
        .zip(&rates)
        .map(|(&mo, r)| ModuleKinetics {
            beta: r.beta,
            gamma: r.gamma,
            ..modules[mo as usize]
        })
        .collect();
    let dev = Device::Cpu;
    let k = Kinetics::from_modules(&per_gene, DType::F64, &dev)?;
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
            let logit = f64::from(offset[j]) + ratio[i][j];
            let n = Poisson::new(depth[j] * level[i][j])?.sample(&mut rng) as u64;
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
        offset,
        modules: modules.to_vec(),
        rates,
    })
}
