//! Stage 1: a time per pseudobulk, each module's transcription and each
//! gene's splicing and degradation, from which gene and which track each read
//! is from.
//!
//! Transcription is a module's: genes of a module share its switch-on and
//! switch-off times, its relaxation rate `λ` and its basal fraction. Splicing
//! `β_g` and degradation `γ_g` are each gene's own, so gene `g` follows
//! `u_g(τ), s_g(τ)`: its module's transcription through its own rates
//! ([`super::kinetics`]).
//!
//! A pseudobulk's reads factor exactly as `P(g, k | p) = P(g | p) · P(k | p, g)`.
//! Which track, for the reads of gene `g` in pseudobulk `p`:
//!
//! ```text
//! x^u_pg ~ Binom(x^u_pg + x^s_pg, σ(ℓ_pg)),   ℓ_pg = b_g + log u_g(τ_p) − log s_g(τ_p)
//! ```
//!
//! Which gene, for the `n_pg` reads of gene `g` (both tracks):
//!
//! ```text
//! n_p· ~ Mult(n_p, softmax_g(c_g + log(u_g(τ_p) + s_g(τ_p))))
//! ```
//!
//! with `b_g` the gene's capture of unspliced reads (its intron structure;
//! at steady state it cannot be told from `log γ_g/β_g`, which only the
//! transients pin) and `c_g` its log loading. There is no capture offset per
//! pseudobulk: within one library capture does not vary between them, and a
//! free one would absorb the unspliced shares' common rise and fall that
//! modules switching together make, the arrow of time itself. The ratio alone barely tells time's direction; the genes' levels
//! rising after their unspliced share is what does. The loss is the negative
//! log-likelihood per read.
//!
//! The fit:
//! 1. `b_g` of the flat model (no curve);
//! 2. each module's switch times by a grid over candidates at the starting
//!    times;
//! 3. rounds of a per-module grid over switch times, a per-pseudobulk grid
//!    over `τ` (both non-local moves gradients cannot make) and Adam steps on
//!    everything together.
//!
//! The starting times are an ordering only; their orientation is decided by
//! fitting both to the end and keeping the lower loss. A slower start can
//! trail early and still win, so the call is never made part way.

use super::kinetics::{curves, log_ratio, Kinetics, ModuleKinetics};
use legume_numeric::candle::candle_core::{DType, Device, Result as CResult, Tensor, Var};
use legume_numeric::candle::candle_nn::ops::{log_softmax, sigmoid};
use legume_numeric::candle::candle_nn::{AdamW, Optimizer, ParamsAdamW};
use legume_numeric::candle::grad_clip::clipped_backward_step;
use legume_numeric::candle::loss::log_sigmoid;

/// Switch-on times live in `(T_ON_LO, T_ON_LO + T_ON_SPAN)`: a module may be
/// on before the first pseudobulk (pure repression) or switch on late.
const T_ON_LO: f64 = -0.5;
const T_ON_SPAN: f64 = 1.5;
const MIN_BASAL: f64 = 1e-3;
/// Log range of the rates. A rate maps smoothly onto it,
/// `log rate = lo + (hi − lo) σ(z)`, so a rate near a bound keeps a gradient
/// and comes back when the reads pull it (a hard clamp would trap it).
const LOG_RATE: (f64, f64) = (-3.0, 7.0);
/// Largest global gradient norm per Adam step.
const GRAD_CLIP: f64 = 5.0;
/// Switch-off times live in `(t_on, T_OFF_HI)`. The ODE is causal: a
/// switch-off after the last pseudobulk changes nothing the reads see, so any
/// such time is one and the same, and its gradient is zero. Moving a module
/// between "off at the end" and "off inside" is the timing step's job
/// ([`Run::timing_step`]), not the gradient's.
const T_OFF_HI: f64 = 1.0;
/// Spacing of the timing step's grid over switch-on and switch-off times.
const TIMING_STEP: f32 = 0.1;
/// Adam's learning rate at the last round, as a share of the first.
const MIN_LR_SCALE: f64 = 0.02;
/// Largest Newton step on an offset.
const MAX_NEWTON: f64 = 2.0;
/// Newton steps per offset profile.
const NEWTON_STEPS: usize = 4;
/// Elements per chunk of a `[chunk × K × G]` grid tensor.
const GRID_ELEMS: usize = 1 << 24;

/// Pseudobulk reads on the genes of the fit.
pub struct PbCounts {
    /// `[P × G]` f32.
    pub unspliced: Tensor,
    pub spliced: Tensor,
    /// The module of every gene, `< n_modules`.
    pub module_of_gene: Vec<u32>,
    pub n_modules: usize,
}

#[derive(Clone, Debug)]
pub struct FitConfig {
    /// Rounds of grid search and Adam steps, on each orientation.
    pub rounds: usize,
    pub adam_steps: usize,
    pub learning_rate: f64,
    /// Grid points over `τ ∈ (0, 1)`.
    pub grid: usize,
}

impl Default for FitConfig {
    fn default() -> Self {
        Self {
            rounds: 30,
            adam_steps: 50,
            learning_rate: 0.01,
            grid: 64,
        }
    }
}

/// A graph's edges, every edge `(p, q, w_pq)` once.
pub type Edges = Vec<(usize, usize, f32)>;

/// A prior on the pseudobulks' times, in nats beside the reads' likelihood.
#[derive(Clone, Debug, Default)]
pub struct TauPrior {
    /// `(λ_a, a)`: `λ_a/2 (τ_p − a_p)²`, e.g. toward a coarser level's times.
    pub anchor: Option<(f64, Vec<f32>)>,
    /// `(λ_g, edges)`: `λ_g/2 w_pq (τ_p − τ_q)²` over a graph of the
    /// pseudobulks, so neighbours share their evidence.
    pub graph: Option<(f64, Edges)>,
}

/// [`TauPrior`] ready for both the gradient steps and the grid step.
struct PreparedPrior {
    anchor: Option<(f64, Vec<f32>, Tensor)>,
    graph: Option<GraphPrior>,
}

/// The graph part of a [`TauPrior`].
struct GraphPrior {
    lambda: f64,
    /// Each pseudobulk's neighbours with their weights.
    neighbours: Vec<Vec<(usize, f32)>>,
    /// The edges as `[E]` tensors `(p, q, w)`.
    edges: [Tensor; 3],
}

impl PreparedPrior {
    fn new(prior: &TauPrior, p: usize, dev: &Device) -> anyhow::Result<Self> {
        let anchor = match &prior.anchor {
            Some((lambda, a)) => {
                anyhow::ensure!(a.len() == p, "{} anchor times for {p} pseudobulks", a.len());
                Some((*lambda, a.clone(), Tensor::from_slice(a, p, dev)?))
            }
            None => None,
        };
        let graph = match &prior.graph {
            Some((lambda, edges)) => {
                let mut nb = vec![Vec::new(); p];
                for &(a, b, w) in edges {
                    anyhow::ensure!(a < p && b < p, "edge ({a}, {b}) outside {p} pseudobulks");
                    nb[a].push((b, w));
                    nb[b].push((a, w));
                }
                let e = edges.len();
                let col = |f: fn(&(usize, usize, f32)) -> u32| {
                    Tensor::from_vec(edges.iter().map(f).collect::<Vec<u32>>(), e, dev)
                };
                let w = Tensor::from_vec(edges.iter().map(|x| x.2).collect::<Vec<f32>>(), e, dev)?;
                Some(GraphPrior {
                    lambda: *lambda,
                    neighbours: nb,
                    edges: [col(|x| x.0 as u32)?, col(|x| x.1 as u32)?, w],
                })
            }
            None => None,
        };
        Ok(Self { anchor, graph })
    }

    /// The prior at times `tau` `[P]`, a scalar tensor in nats.
    fn nats(&self, tau: &Tensor) -> CResult<Tensor> {
        let mut total = tau.zeros_like()?.sum_all()?;
        if let Some((lambda, _, a)) = &self.anchor {
            total = (total + ((tau - a)?.sqr()?.sum_all()? * (lambda / 2.0))?)?;
        }
        if let Some(graph) = &self.graph {
            let [i, j, w] = &graph.edges;
            let d = (tau.index_select(i, 0)? - tau.index_select(j, 0)?)?;
            total = (total + ((d.sqr()? * w)?.sum_all()? * (graph.lambda / 2.0))?)?;
        }
        Ok(total)
    }

    /// The prior's change for pseudobulk `p` at time `t`, its neighbours at
    /// `tau`.
    fn at(&self, p: usize, t: f32, tau: &[f32]) -> f32 {
        let mut v = 0.0;
        if let Some((lambda, a, _)) = &self.anchor {
            v += (*lambda as f32) / 2.0 * (t - a[p]).powi(2);
        }
        if let Some(graph) = &self.graph {
            let s: f32 = graph.neighbours[p]
                .iter()
                .map(|&(q, w)| w * (t - tau[q]).powi(2))
                .sum();
            v += (graph.lambda as f32) / 2.0 * s;
        }
        v
    }
}

/// Splicing and degradation rates of one gene.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GeneRates {
    pub beta: f32,
    pub gamma: f32,
}

pub struct FitResult {
    pub tau: Vec<f32>,
    /// Per gene: its steady offset `b_g`, log loading `c_g` and rates.
    pub offset: Vec<f32>,
    pub loading: Vec<f32>,
    pub rates: Vec<GeneRates>,
    /// Per module: its transcription, with its genes' geometric-mean rates.
    pub modules: Vec<ModuleKinetics>,
    /// Loss per read after every round.
    pub loss: Vec<f64>,
    /// Final loss per read of the starting order and of its reverse.
    pub orientation_loss: [f64; 2],
    /// Whether the reverse order was kept.
    pub reversed: bool,
}

/// The kinetics on the free scale: each module's transcription (`[M]`:
/// switch-on, on-duration, relaxation, basal fraction) and each gene's
/// splicing and degradation (`[G]`).
struct FreeKinetics {
    on: Var,
    duration: Var,
    lambda: Var,
    basal: Var,
    beta: Var,
    gamma: Var,
    /// `[G]` u32: every gene's module.
    module_idx: Tensor,
}

fn logit(p: f64) -> f64 {
    let p = p.clamp(1e-4, 1.0 - 1e-4);
    (p / (1.0 - p)).ln()
}

impl FreeKinetics {
    /// Modules' transcription from `modules`; each gene's rates from `rates`,
    /// or from its module's when not given.
    fn new(
        modules: &[ModuleKinetics],
        rates: Option<&[GeneRates]>,
        module_of_gene: &[u32],
        dev: &Device,
    ) -> CResult<Self> {
        let var = |f: &dyn Fn(&ModuleKinetics) -> f64| {
            let v: Vec<f32> = modules.iter().map(|m| f(m) as f32).collect();
            Var::from_tensor(&Tensor::from_vec(v, modules.len(), dev)?)
        };
        let gene = |f: fn(&GeneRates) -> f32, g: fn(&ModuleKinetics) -> f32| {
            let v: Vec<f32> = module_of_gene
                .iter()
                .enumerate()
                .map(|(j, &m)| {
                    free_rate(match rates {
                        Some(r) => f(&r[j]),
                        None => g(&modules[m as usize]),
                    }) as f32
                })
                .collect();
            Var::from_tensor(&Tensor::from_vec(v, module_of_gene.len(), dev)?)
        };
        Ok(Self {
            on: var(&|m| logit((f64::from(m.t_on) - T_ON_LO) / T_ON_SPAN))?,
            duration: var(&|m| {
                // The switch-on as the free scale can hold it, so some room
                // is always left before T_OFF_HI.
                let t_on = f64::from(m.t_on).clamp(T_ON_LO + 1e-4, T_ON_LO + T_ON_SPAN - 1e-4);
                logit(f64::from(m.duration) / (T_OFF_HI - t_on).max(1e-4))
            })?,
            lambda: var(&|m| free_rate(m.lambda))?,
            basal: var(&|m| logit((f64::from(m.basal) - MIN_BASAL) / (1.0 - MIN_BASAL)))?,
            beta: gene(|r| r.beta, |m| m.beta)?,
            gamma: gene(|r| r.gamma, |m| m.gamma)?,
            module_idx: Tensor::from_slice(module_of_gene, module_of_gene.len(), dev)?,
        })
    }

    /// The modules' transcription on the constrained scale, `[M]` each; their
    /// `beta`, `gamma` are placeholders (the genes carry the rates).
    fn transcription(&self) -> CResult<Kinetics> {
        let t_on = ((sigmoid(self.on.as_tensor())? * T_ON_SPAN)? + T_ON_LO)?;
        // t_off = t_on + (T_OFF_HI − t_on) σ(·).
        let duration = (sigmoid(self.duration.as_tensor())? * (T_OFF_HI - &t_on)?)?;
        let ones = t_on.ones_like()?;
        Ok(Kinetics {
            t_on,
            duration,
            lambda: rate(self.lambda.as_tensor())?,
            basal: ((sigmoid(self.basal.as_tensor())? * (1.0 - MIN_BASAL))? + MIN_BASAL)?,
            beta: ones.clone(),
            gamma: ones,
        })
    }

    /// Every gene's kinetics, `[G]` each: its module's transcription through
    /// its own rates.
    fn genes(&self) -> CResult<Kinetics> {
        let t = self.transcription()?;
        let pick = |x: &Tensor| x.index_select(&self.module_idx, 0);
        Ok(Kinetics {
            t_on: pick(&t.t_on)?,
            duration: pick(&t.duration)?,
            lambda: pick(&t.lambda)?,
            basal: pick(&t.basal)?,
            beta: rate(self.beta.as_tensor())?,
            gamma: rate(self.gamma.as_tensor())?,
        })
    }

    fn rates(&self) -> CResult<Vec<GeneRates>> {
        let beta = rate(self.beta.as_tensor())?.to_vec1::<f32>()?;
        let gamma = rate(self.gamma.as_tensor())?.to_vec1::<f32>()?;
        Ok(beta
            .into_iter()
            .zip(gamma)
            .map(|(beta, gamma)| GeneRates { beta, gamma })
            .collect())
    }

    /// The modules as plain numbers, each with its genes' geometric-mean
    /// rates.
    fn modules(&self) -> CResult<Vec<ModuleKinetics>> {
        let mut out = self.transcription()?.to_modules()?;
        let rates = self.rates()?;
        let module_of = self.module_idx.to_vec1::<u32>()?;
        let mut sums = vec![(0f64, 0f64, 0usize); out.len()];
        for (r, &m) in rates.iter().zip(&module_of) {
            let s = &mut sums[m as usize];
            s.0 += f64::from(r.beta).ln();
            s.1 += f64::from(r.gamma).ln();
            s.2 += 1;
        }
        for (k, (lb, lg, n)) in out.iter_mut().zip(sums) {
            let n = n.max(1) as f64;
            k.beta = (lb / n).exp() as f32;
            k.gamma = (lg / n).exp() as f32;
        }
        Ok(out)
    }

    fn vars(&self) -> Vec<Var> {
        vec![
            self.on.clone(),
            self.duration.clone(),
            self.lambda.clone(),
            self.basal.clone(),
            self.beta.clone(),
            self.gamma.clone(),
        ]
    }

    /// Move the modules' transcription to `modules`, the genes' rates kept.
    fn set_transcription(&self, modules: &[ModuleKinetics], dev: &Device) -> CResult<()> {
        let module_of = self.module_idx.to_vec1::<u32>()?;
        let fresh = Self::new(modules, Some(&self.rates()?), &module_of, dev)?;
        for (old, new) in self.vars().iter().zip(fresh.vars()).take(4) {
            old.set(new.as_tensor())?;
        }
        Ok(())
    }
}

/// A rate from its free scale.
fn rate(x: &Tensor) -> CResult<Tensor> {
    ((sigmoid(x)? * (LOG_RATE.1 - LOG_RATE.0))? + LOG_RATE.0)?.exp()
}

/// A rate's free scale, the inverse of [`rate`].
fn free_rate(x: f32) -> f64 {
    logit((f64::from(x).ln() - LOG_RATE.0) / (LOG_RATE.1 - LOG_RATE.0))
}

/// Elementwise binomial negative log-likelihood of `xu` unspliced and `xs`
/// spliced reads at logit `l` (broadcasting).
fn binomial_nll(l: &Tensor, xu: &Tensor, xs: &Tensor) -> CResult<Tensor> {
    let pos = log_sigmoid(l)?.broadcast_mul(xu)?;
    let neg = log_sigmoid(&l.neg()?)?.broadcast_mul(xs)?;
    (pos + neg)?.neg()
}

/// Newton steps on an additive offset `off` (size 1 along `dim`) maximising
/// the binomial likelihood of logits `a + off`; `xu` and `n` broadcast
/// against `a`.
fn profile_offset(
    a: &Tensor,
    xu: &Tensor,
    n: &Tensor,
    off: &Tensor,
    dim: usize,
) -> CResult<Tensor> {
    let mut off = off.clone();
    for _ in 0..NEWTON_STEPS {
        let p = sigmoid(&a.broadcast_add(&off)?)?;
        let np = p.broadcast_mul(n)?;
        let grad = np.broadcast_sub(xu)?.sum_keepdim(dim)?;
        let curv = (&np * (1.0 - &p)?)?.sum_keepdim(dim)?;
        let step = (grad / (curv + 1e-6)?)?.clamp(-MAX_NEWTON, MAX_NEWTON)?;
        off = (off - step)?;
    }
    Ok(off)
}

/// Candidate settings each module's timing starts from, at the starting
/// rates every gene begins with.
fn timing_candidates() -> Vec<ModuleKinetics> {
    let mut out = Vec::new();
    for t_on in [-0.3f32, -0.1, 0.1, 0.3, 0.5, 0.7] {
        for duration in [0.2f32, 0.4, 0.8, 1.2] {
            out.push(ModuleKinetics {
                t_on,
                duration,
                lambda: 10.0,
                basal: 0.05,
                beta: 6.0,
                gamma: 3.0,
            });
        }
    }
    out
}

/// The (switch-on, switch-off) pairs of the timing step's grid.
fn timing_pairs() -> Vec<(f32, f32)> {
    let mut pairs = Vec::new();
    let steps = (T_ON_SPAN as f32 / TIMING_STEP).round() as usize;
    for i in 0..steps {
        let on = T_ON_LO as f32 + TIMING_STEP * (i + 1) as f32;
        if on >= T_OFF_HI as f32 - 1e-3 {
            break;
        }
        let mut off = on + TIMING_STEP;
        while off < T_OFF_HI as f32 - 1e-3 {
            pairs.push((on, off));
            off += TIMING_STEP;
        }
        pairs.push((on, T_OFF_HI as f32));
    }
    pairs
}

/// One run of the fit from one starting order.
struct Run<'a> {
    counts: &'a PbCounts,
    cfg: &'a FitConfig,
    xu: Tensor,
    xs: Tensor,
    /// `[P × G]` reads of every gene, both tracks.
    n: Tensor,
    /// `[G × M]` every gene's module, one-hot.
    onehot: Tensor,
    reads: f64,
    free: FreeKinetics,
    /// `[1 × G]` the genes' steady offsets `b_g` and log loadings `c_g`.
    offset: Var,
    load: Var,
    /// `[P]`, `τ = σ(z)`.
    z: Var,
    prior: PreparedPrior,
    adam: AdamW,
    loss: Vec<f64>,
}

impl<'a> Run<'a> {
    /// A run from times `tau0`; kinetics from `start` when given, else the
    /// timing grid at `tau0`.
    fn new(
        counts: &'a PbCounts,
        tau0: &[f32],
        start: Option<(&[ModuleKinetics], &[GeneRates])>,
        prior: &TauPrior,
        cfg: &'a FitConfig,
    ) -> anyhow::Result<Self> {
        let xu = counts.unspliced.to_dtype(DType::F32)?;
        let xs = counts.spliced.to_dtype(DType::F32)?;
        let dev = xu.device().clone();
        let (p, g) = xu.dims2()?;
        let m = counts.n_modules;
        anyhow::ensure!(
            xs.dims2()? == (p, g),
            "unspliced and spliced differ in shape"
        );
        anyhow::ensure!(
            tau0.len() == p,
            "{} starting times for {p} pseudobulks",
            tau0.len()
        );
        anyhow::ensure!(
            counts.module_of_gene.len() == g
                && counts.module_of_gene.iter().all(|&mo| (mo as usize) < m),
            "every gene needs a module below {m}"
        );
        let n = (&xu + &xs)?;
        let reads = n.sum_all()?.to_scalar::<f32>()?.max(1.0) as f64;
        let mut onehot = vec![0f32; g * m];
        for (j, &mo) in counts.module_of_gene.iter().enumerate() {
            onehot[j * m + mo as usize] = 1.0;
        }
        let onehot = Tensor::from_vec(onehot, (g, m), &dev)?;
        let z: Vec<f32> = tau0.iter().map(|&t| logit(f64::from(t)) as f32).collect();
        let z = Var::from_tensor(&Tensor::from_vec(z, p, &dev)?)?;
        // The flat model's offsets: every gene's pooled log ratio.
        let offset = ((xu.sum_keepdim(0)? + 0.5)?.log()? - (xs.sum_keepdim(0)? + 0.5)?.log()?)?;
        let free = match start {
            Some((modules, rates)) => {
                anyhow::ensure!(
                    modules.len() == m && rates.len() == g,
                    "start: {} modules, {} genes' rates, for {m} and {g}",
                    modules.len(),
                    rates.len()
                );
                FreeKinetics::new(modules, Some(rates), &counts.module_of_gene, &dev)?
            }
            None => FreeKinetics::new(
                &vec![timing_candidates()[0]; m],
                None,
                &counts.module_of_gene,
                &dev,
            )?,
        };
        let prior = PreparedPrior::new(prior, p, &dev)?;
        let load = Var::zeros((1, g), DType::F32, &dev)?;
        let offset = Var::from_tensor(&offset)?;
        let mut vars = free.vars();
        vars.extend([offset.clone(), z.clone(), load.clone()]);
        let adam = AdamW::new(
            vars,
            ParamsAdamW {
                lr: cfg.learning_rate,
                weight_decay: 0.0,
                ..Default::default()
            },
        )?;
        let mut run = Self {
            counts,
            cfg,
            xu,
            xs,
            n,
            onehot,
            reads,
            free,
            offset,
            load,
            z,
            prior,
            adam,
            loss: Vec::new(),
        };
        if start.is_none() {
            run.init_timing()?;
        }
        run.init_loadings()?;
        Ok(run)
    }

    fn tau(&self) -> CResult<Tensor> {
        sigmoid(self.z.as_tensor())
    }

    /// At times `tau` `[T]`: the track logits and the magnitude
    /// logits `c_g + log(u_g + s_g)`, each `[T × G]`.
    fn logits_at(&self, tau: &Tensor) -> CResult<(Tensor, Tensor)> {
        let (u, s) = curves(&self.free.genes()?, tau)?;
        let track = (u.log()? - s.log()?)?.broadcast_add(self.offset.as_tensor())?;
        let magnitude = (u + s)?.log()?.broadcast_add(self.load.as_tensor())?;
        Ok((track, magnitude))
    }

    /// [`Self::logits_at`] the current times.
    fn logits(&self) -> CResult<(Tensor, Tensor)> {
        self.logits_at(&self.tau()?)
    }

    /// Per pseudobulk, `−Σ_g n_pg log softmax_g(magnitude)`, `[P]`.
    fn gene_nll(&self, magnitude: &Tensor) -> CResult<Tensor> {
        (&self.n * log_softmax(magnitude, 1)?)?.sum(1)?.neg()
    }

    /// Each gene's log loading at the current kinetics and times: its reads
    /// over its curve's summed level.
    fn init_loadings(&mut self) -> CResult<()> {
        let (u, s) = curves(&self.free.genes()?, &self.tau()?.detach())?;
        let level = (u + s)?.sum_keepdim(0)?;
        let reads = (self.n.sum_keepdim(0)? + 0.5)?;
        self.load.set(&(reads.log()? - level.log()?)?)
    }

    /// The reads' negative log-likelihood, which track plus which gene, at
    /// logits `(track, magnitude)`.
    fn reads_nll(&self, track: &Tensor, magnitude: &Tensor) -> CResult<Tensor> {
        let nll = binomial_nll(track, &self.xu, &self.xs)?;
        nll.sum_all()? + self.gene_nll(magnitude)?.sum_all()?
    }

    /// Each module's switch times: the candidate whose curve, at the current
    /// times with each gene's offset profiled, fits its genes' unspliced
    /// shares best. Every gene still has the starting rates here, so a
    /// candidate is one curve per module.
    fn init_timing(&mut self) -> CResult<()> {
        let cands = timing_candidates();
        let c = cands.len();
        let (p, g) = self.xu.dims2()?;
        let dev = self.xu.device().clone();
        let k = Kinetics::from_modules(&cands, DType::F32, &dev)?;
        // `[C × P × 1]` candidate curves.
        let curves = log_ratio(&k, &self.tau()?.detach())?
            .t()?
            .unsqueeze(2)?
            .contiguous()?;
        let chunk = (GRID_ELEMS / (c * p)).max(1);
        let mut per_gene = Vec::with_capacity(g);
        for start in (0..g).step_by(chunk) {
            let w = chunk.min(g - start);
            let xu = self.xu.narrow(1, start, w)?.unsqueeze(0)?;
            let xs = self.xs.narrow(1, start, w)?.unsqueeze(0)?;
            let n = self.n.narrow(1, start, w)?.unsqueeze(0)?;
            let a = curves.broadcast_as((c, p, w))?.contiguous()?;
            let off0 = self
                .offset
                .as_tensor()
                .detach()
                .narrow(1, start, w)?
                .unsqueeze(0)?
                .broadcast_as((c, 1, w))?
                .contiguous()?;
            let off = profile_offset(&a, &xu, &n, &off0, 1)?;
            per_gene.push(binomial_nll(&a.broadcast_add(&off)?, &xu, &xs)?.sum(1)?);
        }
        let per_module = Tensor::cat(&per_gene, 1)?.matmul(&self.onehot)?; // [C × M]
        let best = per_module.argmin(0)?.to_vec1::<u32>()?;
        let chosen: Vec<ModuleKinetics> = best.iter().map(|&i| cands[i as usize]).collect();
        self.free.set_transcription(&chosen, &dev)
    }

    /// Move every module to the (switch-on, switch-off) pair of a grid that
    /// fits best, the times, the genes' rates and the other modules where
    /// they are, when that beats its own: the non-local move gradients cannot
    /// make (a switch-off past the last pseudobulk has no gradient). A pair is
    /// scored on its genes, each through its own rates: their track term,
    /// offsets profiled, and the which-gene term with their curves swapped in,
    /// each gene's level kept.
    fn timing_step(&mut self) -> CResult<()> {
        let current = self.free.transcription()?.to_modules()?;
        let m = current.len();
        let pairs = timing_pairs();
        // Candidate 0 is each module's own timing.
        let c = pairs.len() + 1;
        let timing = |ci: usize, mo: usize| -> (f32, f32) {
            if ci == 0 {
                (current[mo].t_on, current[mo].duration)
            } else {
                let (on, off) = pairs[ci - 1];
                (on, off - on)
            }
        };
        let (p, g) = self.xu.dims2()?;
        let dev = self.xu.device().clone();
        // Nothing here needs gradients: every input is detached, so no graph
        // holds the candidates' tensors alive across the chunks.
        let tau = self.tau()?.detach();
        let offset = self.offset.as_tensor().detach();
        let rates = self.free.rates()?;
        let module_of = &self.counts.module_of_gene;
        let (_, mag) = self.logits()?;
        let mag = mag.detach(); // [P × G]
                                // A common shift per pseudobulk keeps every exponential in range.
        let top = mag.max_keepdim(1)?; // [P × 1]
        let n_p = self.n.sum_keepdim(1)?; // [P × 1]
        let total_now = mag.broadcast_sub(&top)?.exp()?.sum_keepdim(1)?; // [P × 1]
                                                                         // Each module's current share of that sum.
        let module_now = mag.broadcast_sub(&top)?.exp()?.matmul(&self.onehot)?; // [P × M]

        let chunk = (GRID_ELEMS / (c * p)).max(1);
        let mut track_parts = Vec::new();
        let mut swapped = Tensor::zeros((p, c * m), DType::F32, &dev)?; // Σ_{g∈m} exp(cand − top)
        let mut fitted = Tensor::zeros((c, m), DType::F32, &dev)?; // Σ_p Σ_{g∈m} n_pg cand
        for start in (0..g).step_by(chunk) {
            let w = chunk.min(g - start);
            // `[C·w]` candidate kinetics, candidate-major.
            let mut cands = Vec::with_capacity(c * w);
            for ci in 0..c {
                for j in start..start + w {
                    let mo = module_of[j] as usize;
                    let (t_on, duration) = timing(ci, mo);
                    cands.push(ModuleKinetics {
                        t_on,
                        duration,
                        beta: rates[j].beta,
                        gamma: rates[j].gamma,
                        ..current[mo]
                    });
                }
            }
            let (u, s) = curves(&Kinetics::from_modules(&cands, DType::F32, &dev)?, &tau)?;
            let ratio = (u.log()? - s.log()?)?.reshape((p, c, w))?;
            let level = (u + s)?.log()?.reshape((p, c, w))?;

            // Track term per (pair, gene), offsets profiled.
            let a = ratio.transpose(0, 1)?.contiguous()?; // [C × P × w]
            let xu = self.xu.narrow(1, start, w)?.unsqueeze(0)?;
            let xs = self.xs.narrow(1, start, w)?.unsqueeze(0)?;
            let n = self.n.narrow(1, start, w)?;
            let off0 = offset
                .narrow(1, start, w)?
                .unsqueeze(0)?
                .broadcast_as((c, 1, w))?
                .contiguous()?;
            let off = profile_offset(&a, &xu, &n.unsqueeze(0)?, &off0, 1)?;
            track_parts.push(binomial_nll(&a.broadcast_add(&off)?, &xu, &xs)?.sum(1)?); // [C × w]

            // Which-gene term: each candidate level shifted to the gene's
            // read-weighted mean level now.
            let wt = n.broadcast_div(&(n.sum_keepdim(0)? + 1e-6)?)?; // [P × w]
            let mag_w = mag.narrow(1, start, w)?;
            let mean_now = (&mag_w * &wt)?.sum_keepdim(0)?; // [1 × w]
            let mean_cand = level.broadcast_mul(&wt.unsqueeze(1)?)?.sum_keepdim(0)?; // [1 × C × w]
            let cand = level
                .broadcast_sub(&mean_cand)?
                .broadcast_add(&mean_now.unsqueeze(1)?)?; // [P × C × w]
            let onehot_w = self.onehot.narrow(0, start, w)?; // [w × M]
            let e = cand
                .broadcast_sub(&top.unsqueeze(2)?)?
                .clamp(-80.0, 80.0)?
                .exp()?
                .reshape((p * c, w))?
                .matmul(&onehot_w)?
                .reshape((p, c * m))?;
            swapped = (swapped + e)?;
            let nc = cand
                .broadcast_mul(&n.unsqueeze(1)?)?
                .sum(0)?
                .matmul(&onehot_w)?; // [C × M]
            fitted = (fitted + nc)?;
        }
        let track = Tensor::cat(&track_parts, 1)?.matmul(&self.onehot)?; // [C × M]
                                                                         // log Σ over every gene with module m's genes swapped in.
        let others = total_now.broadcast_sub(&module_now)?; // [P × M]
        let lse = swapped
            .reshape((p, c, m))?
            .broadcast_add(&others.unsqueeze(1)?)?
            .maximum(1e-30)?
            .log()?
            .broadcast_add(&top.unsqueeze(2)?)?; // [P × C × M]
        let magnitude = (lse.broadcast_mul(&n_p.unsqueeze(2)?)?.sum(0)? - fitted)?; // [C × M]

        let score = (track + magnitude)?.to_vec2::<f32>()?;
        let mut moved = current.clone();
        for mo in 0..m {
            let best = (0..c)
                .min_by(|&x, &y| score[x][mo].total_cmp(&score[y][mo]))
                .expect("pairs");
            if best != 0 && score[best][mo] < score[0][mo] - 1e-3 {
                let (t_on, duration) = timing(best, mo);
                moved[mo] = ModuleKinetics {
                    t_on,
                    duration,
                    ..moved[mo]
                };
            }
        }
        self.free.set_transcription(&moved, &dev)
    }

    /// Loss per read at the current parameters, the prior included.
    fn evaluate(&mut self) -> CResult<f64> {
        let (a, mag) = self.logits()?;
        let (a, mag) = (a.detach(), mag.detach());
        let total = (self.reads_nll(&a, &mag)? + self.prior.nats(&self.tau()?)?)?;
        Ok(f64::from(total.to_scalar::<f32>()?) / self.reads)
    }

    fn adam_steps(&mut self) -> CResult<()> {
        for _ in 0..self.cfg.adam_steps {
            let (a, mag) = self.logits()?;
            let loss =
                ((self.reads_nll(&a, &mag)? + self.prior.nats(&self.tau()?)?)? / self.reads)?;
            clipped_backward_step(&mut self.adam, &loss, GRAD_CLIP)?;
        }
        Ok(())
    }

    /// Move every pseudobulk to the grid time that fits its reads and the
    /// prior best (its neighbours where they are), when that beats its current
    /// time.
    fn grid_step(&mut self) -> CResult<()> {
        let kg = self.cfg.grid;
        let (p, g) = self.xu.dims2()?;
        let dev = self.xu.device().clone();
        let grid: Vec<f32> = (0..kg).map(|i| (i as f32 + 0.5) / kg as f32).collect();
        let (a_grid, mag_grid) = self.logits_at(&Tensor::from_vec(grid.clone(), kg, &dev)?)?;
        let a_grid = a_grid.detach().unsqueeze(0)?; // [1 × K × G]
                                                    // Every pseudobulk's which-gene term at every grid time `[P × K]`, and
                                                    // at its own `[P]`.
        let lsm = log_softmax(&mag_grid.detach(), 1)?;
        let mag_grid = self.n.matmul(&lsm.t()?)?.neg()?.to_vec2::<f32>()?;
        let (a_now, mag_now) = self.logits()?;
        let (a_now, mag_now) = (a_now.detach(), mag_now.detach());
        let mag_now = self.gene_nll(&mag_now)?.to_vec1::<f32>()?;
        let now = binomial_nll(&a_now, &self.xu, &self.xs)?
            .sum(1)?
            .to_vec1::<f32>()?;
        let mut tau = self.tau()?.to_vec1::<f32>()?;
        let current = tau.clone();
        let chunk = (GRID_ELEMS / (kg * g)).max(1);
        for start in (0..p).step_by(chunk) {
            let w = chunk.min(p - start);
            let xu = self.xu.narrow(0, start, w)?.unsqueeze(1)?;
            let xs = self.xs.narrow(0, start, w)?.unsqueeze(1)?;
            let nll = binomial_nll(&a_grid, &xu, &xs)?.sum(2)?.to_vec2::<f32>()?; // [w × K]
            for (i, row) in nll.iter().enumerate() {
                let pb = start + i;
                let score =
                    |k: usize| row[k] + mag_grid[pb][k] + self.prior.at(pb, grid[k], &current);
                let best = (0..kg)
                    .min_by(|&x, &y| score(x).total_cmp(&score(y)))
                    .expect("grid");
                let stay = now[pb] + mag_now[pb] + self.prior.at(pb, current[pb], &current);
                if score(best) < stay - 1e-4 {
                    tau[pb] = grid[best];
                }
            }
        }
        let z: Vec<f32> = tau.iter().map(|&t| logit(f64::from(t)) as f32).collect();
        self.z.set(&Tensor::from_vec(z, p, &dev)?)
    }

    fn round(&mut self) -> CResult<()> {
        self.timing_step()?;
        self.grid_step()?;
        // Cosine decay of Adam's step over the rounds: Adam moves a parameter
        // by about its learning rate whatever its gradient's size, so late
        // steps must shrink for weakly pinned parameters to settle.
        let r = self.loss.len() as f64 / self.cfg.rounds.max(1) as f64;
        let scale =
            MIN_LR_SCALE + (1.0 - MIN_LR_SCALE) * 0.5 * (1.0 + (std::f64::consts::PI * r).cos());
        self.adam.set_learning_rate(self.cfg.learning_rate * scale);
        self.adam_steps()?;
        let l = self.evaluate()?;
        self.loss.push(l);
        Ok(())
    }

    fn finish(mut self, orientation_loss: [f64; 2], reversed: bool) -> anyhow::Result<FitResult> {
        self.evaluate()?;
        let row =
            |v: &Var| -> anyhow::Result<Vec<f32>> { Ok(v.as_tensor().flatten_all()?.to_vec1()?) };
        Ok(FitResult {
            tau: self.tau()?.to_vec1()?,
            offset: row(&self.offset)?,
            loading: row(&self.load)?,
            rates: self.free.rates()?,
            modules: self.free.modules()?,
            loss: self.loss,
            orientation_loss,
            reversed,
        })
    }
}

/// Fit stage 1 from the starting times `tau0` (an ordering in `(0, 1)`; its
/// orientation is decided here).
pub fn fit(
    counts: &PbCounts,
    tau0: &[f32],
    prior: &TauPrior,
    cfg: &FitConfig,
) -> anyhow::Result<FitResult> {
    let (reverse, reverse_prior) = reversed(tau0, prior);
    let forward = fit_oriented(counts, tau0, prior, cfg)?;
    let backward = fit_oriented(counts, &reverse, &reverse_prior, cfg)?;
    let last = |r: &FitResult| r.loss.last().copied().unwrap_or(f64::INFINITY);
    let orientation_loss = [last(&forward), last(&backward)];
    let reversed = orientation_loss[1] < orientation_loss[0];
    let mut out = if reversed { backward } else { forward };
    out.orientation_loss = orientation_loss;
    out.reversed = reversed;
    Ok(out)
}

/// The reverse of starting times `tau0` and of a prior on them: an anchor is
/// on the starting order's scale, so it reverses too.
fn reversed(tau0: &[f32], prior: &TauPrior) -> (Vec<f32>, TauPrior) {
    let flip = |t: &[f32]| t.iter().map(|&x| 1.0 - x).collect::<Vec<f32>>();
    (
        flip(tau0),
        TauPrior {
            anchor: prior.anchor.as_ref().map(|(l, a)| (*l, flip(a))),
            graph: prior.graph.clone(),
        },
    )
}

/// One orientation of the fit from `tau0`: each module's timing from the
/// candidate grid at `tau0`, then the rounds.
fn fit_oriented(
    counts: &PbCounts,
    tau0: &[f32],
    prior: &TauPrior,
    cfg: &FitConfig,
) -> anyhow::Result<FitResult> {
    let mut run = Run::new(counts, tau0, None, prior, cfg)?;
    for _ in 0..cfg.rounds {
        run.round()?;
    }
    let last = run.loss.last().copied().unwrap_or(f64::NAN);
    run.finish([last, f64::NAN], false)
}

/// Fit from the times `tau0`, the modules' transcription and the genes'
/// rates of a coarser fit, its orientation kept: one step of a coarse-to-fine
/// sequence.
pub fn refine(
    counts: &PbCounts,
    tau0: &[f32],
    modules: &[ModuleKinetics],
    rates: &[GeneRates],
    prior: &TauPrior,
    cfg: &FitConfig,
) -> anyhow::Result<FitResult> {
    let mut run = Run::new(counts, tau0, Some((modules, rates)), prior, cfg)?;
    for _ in 0..cfg.rounds {
        run.round()?;
    }
    let last = run.loss.last().copied().unwrap_or(f64::NAN);
    run.finish([last, f64::NAN], false)
}

#[cfg(test)]
#[path = "tests/fit.rs"]
mod tests;
