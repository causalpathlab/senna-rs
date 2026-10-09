//! Stage 1: module kinetics and a time per pseudobulk, from which module and
//! which track each read is from.
//!
//! A pseudobulk's reads factor exactly as
//! `P(g, k | p) = P(m(g) | p) · P(g | m, p) · P(k | p, g)`. Which track, for
//! the reads of gene `g` in pseudobulk `p`:
//!
//! ```text
//! x^u_pg ~ Binom(x^u_pg + x^s_pg, σ(ℓ_pg)),
//! ℓ_pg = κ_p + b_g + log u_m(g)(τ_p) − log s_m(g)(τ_p)
//! ```
//!
//! Which module, for the `n_pm` reads of module `m` (both tracks):
//!
//! ```text
//! n_p· ~ Mult(n_p, softmax_m(c_m + log(u_m(τ_p) + s_m(τ_p))))
//! ```
//!
//! with `u_m, s_m` the module's closed-form splicing curves
//! ([`super::kinetics`]), `κ_p` the pseudobulk's and `b_g` the gene's capture
//! of unspliced reads (`κ_p` profiled by Newton steps, never a free
//! parameter; `b_g` the gene's intron structure) and `c_m` the module's log
//! loading. Splicing `β` and
//! degradation `γ` are each module's. Without `κ_p`, a rise of every gene's
//! unspliced share together over time (seen on real data) can only be the
//! relaxation after a common switch-on, which puts the high-unspliced end
//! first and turns time around; `κ_p` takes up that common trend, so the
//! genes' own lags set the direction. Which gene within a module carries no time
//! and is left out. The ratio alone barely tells time's direction; the
//! modules' levels rising after their unspliced share is what does. The loss
//! is the negative log-likelihood per read.
//!
//! The fit:
//! 1. `b_g` and `κ_p` of the flat model (no curve);
//! 2. each module's switch times by a grid over candidate (on, duration,
//!    rate) settings at the starting times;
//! 3. rounds of a per-pseudobulk grid search over `τ` (so a pseudobulk can
//!    jump between arcs) and Adam steps on kinetics, offsets and times
//!    together.
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
/// Log range of the rates.
const LOG_RATE: (f64, f64) = (-3.0, 7.0);
/// Switch-off times live in `(t_on, T_OFF_HI)`. The ODE is causal: a
/// switch-off after the last pseudobulk changes nothing the reads see, so any
/// such time is one and the same, and its gradient is zero. Moving a module
/// between "off at the end" and "off inside" is the timing step's job
/// ([`Run::timing_step`]), not the gradient's.
const T_OFF_HI: f64 = 1.0;
/// Spacing of the timing step's grid over switch-on and switch-off times.
const TIMING_STEP: f32 = 0.1;
/// Largest global gradient norm per Adam step.
const GRAD_CLIP: f64 = 5.0;
/// Adam's learning rate at the last round, as a share of the first.
const MIN_LR_SCALE: f64 = 0.02;
/// Largest Newton step on an offset.
const MAX_NEWTON: f64 = 2.0;
/// Newton steps per offset profile.
const NEWTON_STEPS: usize = 4;
/// Flat-model alternations of `κ` and `b` at the start.
const FLAT_ROUNDS: usize = 10;
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
    /// `(λ_g, edges)`, every edge `(p, q, w_pq)` once: `λ_g/2 w_pq (τ_p − τ_q)²`
    /// over a graph of the pseudobulks, so neighbours share their evidence.
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

pub struct FitResult {
    pub tau: Vec<f32>,
    /// Per pseudobulk, its capture offset `κ_p`.
    pub kappa: Vec<f32>,
    pub offset: Vec<f32>,
    /// Per module, its log loading `c_m` in the magnitude term.
    pub loading: Vec<f32>,
    pub modules: Vec<ModuleKinetics>,
    /// Loss per read after every round.
    pub loss: Vec<f64>,
    /// Final loss per read of the starting order and of its reverse.
    pub orientation_loss: [f64; 2],
    /// Whether the reverse order was kept.
    pub reversed: bool,
}

/// The kinetics on the free scale; every Var is `[M]`.
struct FreeKinetics {
    on: Var,
    duration: Var,
    lambda: Var,
    basal: Var,
    beta: Var,
    gamma: Var,
}

fn sigmoid_f64(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

fn logit(p: f64) -> f64 {
    let p = p.clamp(1e-4, 1.0 - 1e-4);
    (p / (1.0 - p)).ln()
}

impl FreeKinetics {
    fn new(modules: &[ModuleKinetics], dev: &Device) -> CResult<Self> {
        let var = |f: &dyn Fn(&ModuleKinetics) -> f64| {
            let v: Vec<f32> = modules.iter().map(|m| f(m) as f32).collect();
            Var::from_tensor(&Tensor::from_vec(v, modules.len(), dev)?)
        };
        Ok(Self {
            on: var(&|m| logit((f64::from(m.t_on) - T_ON_LO) / T_ON_SPAN))?,
            duration: var(&|m| {
                let t_on = T_ON_LO
                    + T_ON_SPAN * sigmoid_f64(logit((f64::from(m.t_on) - T_ON_LO) / T_ON_SPAN));
                logit(f64::from(m.duration) / (T_OFF_HI - t_on))
            })?,
            lambda: var(&|m| f64::from(m.lambda).ln())?,
            basal: var(&|m| logit((f64::from(m.basal) - MIN_BASAL) / (1.0 - MIN_BASAL)))?,
            beta: var(&|m| f64::from(m.beta).ln())?,
            gamma: var(&|m| f64::from(m.gamma).ln())?,
        })
    }

    fn kinetics(&self) -> CResult<Kinetics> {
        let rate = |v: &Var| v.as_tensor().clamp(LOG_RATE.0, LOG_RATE.1)?.exp();
        let t_on = ((sigmoid(self.on.as_tensor())? * T_ON_SPAN)? + T_ON_LO)?;
        // t_off = t_on + (T_OFF_HI − t_on) σ(·).
        let duration = (sigmoid(self.duration.as_tensor())? * (T_OFF_HI - &t_on)?)?;
        Ok(Kinetics {
            t_on,
            duration,
            lambda: rate(&self.lambda)?,
            basal: ((sigmoid(self.basal.as_tensor())? * (1.0 - MIN_BASAL))? + MIN_BASAL)?,
            beta: rate(&self.beta)?,
            gamma: rate(&self.gamma)?,
        })
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

/// Candidate settings each module's timing starts from.
fn timing_candidates() -> Vec<ModuleKinetics> {
    let mut out = Vec::new();
    for t_on in [-0.3f32, -0.1, 0.1, 0.3, 0.5, 0.7] {
        for duration in [0.2f32, 0.4, 0.8, 1.2] {
            for (beta, gamma) in [(6.0f32, 3.0f32), (3.0, 6.0)] {
                out.push(ModuleKinetics {
                    t_on,
                    duration,
                    lambda: 10.0,
                    basal: 0.05,
                    beta,
                    gamma,
                });
            }
        }
    }
    out
}

/// One run of the fit from one starting order.
struct Run<'a> {
    counts: &'a PbCounts,
    cfg: &'a FitConfig,
    xu: Tensor,
    xs: Tensor,
    n: Tensor,
    /// `[G]` u32.
    module_idx: Tensor,
    reads: f64,
    free: FreeKinetics,
    /// `[1 × G]`.
    offset: Var,
    /// `[P]`, `τ = σ(z)`.
    z: Var,
    /// `[P × 1]`, profiled.
    kappa: Var,
    /// `[P × M]` reads of every module, both tracks.
    module_totals: Tensor,
    /// `[1 × M]` the modules' log loadings `c_m`.
    load: Var,
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
        start: Option<&[ModuleKinetics]>,
        prior: &TauPrior,
        cfg: &'a FitConfig,
    ) -> anyhow::Result<Self> {
        let xu = counts.unspliced.to_dtype(DType::F32)?;
        let xs = counts.spliced.to_dtype(DType::F32)?;
        let dev = xu.device().clone();
        let (p, g) = xu.dims2()?;
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
                && counts
                    .module_of_gene
                    .iter()
                    .all(|&m| (m as usize) < counts.n_modules),
            "every gene needs a module below {}",
            counts.n_modules
        );
        let n = (&xu + &xs)?;
        let reads = n.sum_all()?.to_scalar::<f32>()?.max(1.0) as f64;
        let module_idx = Tensor::from_vec(counts.module_of_gene.clone(), g, &dev)?;
        let z: Vec<f32> = tau0.iter().map(|&t| logit(f64::from(t)) as f32).collect();
        let z = Var::from_tensor(&Tensor::from_vec(z, p, &dev)?)?;
        // The flat model's offsets: pooled log ratios, then κ and b in turn.
        let mut offset =
            ((xu.sum_keepdim(0)? + 0.5)?.log()? - (xs.sum_keepdim(0)? + 0.5)?.log()?)?;
        let mut kappa = Tensor::zeros((p, 1), DType::F32, &dev)?;
        let zero = Tensor::zeros((p, g), DType::F32, &dev)?;
        for _ in 0..FLAT_ROUNDS {
            kappa = profile_offset(&zero.broadcast_add(&offset)?, &xu, &n, &kappa, 1)?;
            offset = profile_offset(&zero.broadcast_add(&kappa)?, &xu, &n, &offset, 0)?;
        }
        let free = match start {
            Some(m) => {
                anyhow::ensure!(
                    m.len() == counts.n_modules,
                    "{} start modules for {}",
                    m.len(),
                    counts.n_modules
                );
                FreeKinetics::new(m, &dev)?
            }
            None => FreeKinetics::new(&vec![timing_candidates()[0]; counts.n_modules], &dev)?,
        };
        let prior = PreparedPrior::new(prior, p, &dev)?;
        let m = counts.n_modules;
        let mut onehot = vec![0f32; g * m];
        for (j, &mo) in counts.module_of_gene.iter().enumerate() {
            onehot[j * m + mo as usize] = 1.0;
        }
        let module_totals = n.matmul(&Tensor::from_vec(onehot, (g, m), &dev)?)?;
        let load = Var::zeros((1, m), DType::F32, &dev)?;
        let offset = Var::from_tensor(&offset)?;
        let kappa = Var::from_tensor(&kappa)?;
        let mut vars = free.vars();
        vars.push(offset.clone());
        vars.push(z.clone());
        vars.push(load.clone());
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
            module_idx,
            reads,
            free,
            offset,
            z,
            kappa,
            module_totals,
            load,
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

    /// At times `tau` `[T]`: the track logits `[T × G]`, and the
    /// magnitude logits `c_m + log(u_m + s_m)` `[T × M]`.
    fn logits_at(&self, tau: &Tensor) -> CResult<(Tensor, Tensor)> {
        let k = self.free.kinetics()?;
        let (u, s) = curves(&k, tau)?;
        let track = (u.log()? - s.log()?)?
            .index_select(&self.module_idx, 1)?
            .broadcast_add(self.offset.as_tensor())?;
        let magnitude = (u + s)?.log()?.broadcast_add(self.load.as_tensor())?;
        Ok((track, magnitude))
    }

    /// [`Self::logits_at`] the current times.
    fn logits(&self) -> CResult<(Tensor, Tensor)> {
        self.logits_at(&self.tau()?)
    }

    /// Per pseudobulk, `−Σ_m n_pm log softmax_m(magnitude)`, `[P]`.
    fn module_nll(&self, magnitude: &Tensor) -> CResult<Tensor> {
        (&self.module_totals * log_softmax(magnitude, 1)?)?
            .sum(1)?
            .neg()
    }

    /// Each module's log loading at the current kinetics and times: its share
    /// of the reads over its curve's mean level.
    fn init_loadings(&mut self) -> CResult<()> {
        let k = self.free.kinetics()?;
        let (u, s) = curves(&k, &self.tau()?)?;
        let level = (u + s)?.sum_keepdim(0)?;
        let reads = (self.module_totals.sum_keepdim(0)? + 0.5)?;
        self.load.set(&(reads.log()? - level.log()?)?)
    }

    /// The reads' negative log-likelihood, which track plus which module, at
    /// logits `(track, magnitude)` with `κ` as it is.
    fn reads_nll(&self, track: &Tensor, magnitude: &Tensor) -> CResult<Tensor> {
        let nll = binomial_nll(
            &track.broadcast_add(self.kappa.as_tensor())?,
            &self.xu,
            &self.xs,
        )?;
        nll.sum_all()? + self.module_nll(magnitude)?.sum_all()?
    }

    /// Each module's switch times and rate order: the candidate whose curve,
    /// at the current times with each gene's offset profiled, fits its genes
    /// best.
    fn init_timing(&mut self) -> CResult<()> {
        let cands = timing_candidates();
        let c = cands.len();
        let (p, g) = self.xu.dims2()?;
        let dev = self.xu.device().clone();
        let k = Kinetics::from_modules(&cands, DType::F32, &dev)?;
        // `[C × P × 1]` candidate curves plus κ.
        let curves = log_ratio(&k, &self.tau()?.detach())?
            .broadcast_add(&self.kappa.as_tensor().detach())?
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
            let nll = binomial_nll(&a.broadcast_add(&off)?, &xu, &xs)?.sum(1)?; // [C × w]
            per_gene.push(nll);
        }
        let nll = Tensor::cat(&per_gene, 1)?; // [C × G]
        let m = self.counts.n_modules;
        let mut onehot = vec![0f32; g * m];
        for (j, &mo) in self.counts.module_of_gene.iter().enumerate() {
            onehot[j * m + mo as usize] = 1.0;
        }
        let per_module = nll.matmul(&Tensor::from_vec(onehot, (g, m), &dev)?)?; // [C × M]
        let best = per_module.argmin(0)?.to_vec1::<u32>()?;
        let chosen: Vec<ModuleKinetics> = best.iter().map(|&i| cands[i as usize]).collect();
        let fresh = FreeKinetics::new(&chosen, &dev)?;
        for (old, new) in self.free.vars().iter().zip(fresh.vars()) {
            old.set(new.as_tensor())?;
        }
        Ok(())
    }

    /// Move every module to the (switch-on, switch-off) pair of a grid that
    /// fits best, its rates, the times and the other modules where they are,
    /// when that beats its own: the non-local move gradients cannot make (a
    /// switch-off past the last pseudobulk has no gradient). A pair is scored
    /// on its genes' track term, their offsets profiled, and on the
    /// which-module term with the module's column swapped in, its level kept.
    fn timing_step(&mut self) -> CResult<()> {
        let current = self.free.kinetics()?.to_modules()?;
        let m = current.len();
        let mut pairs: Vec<(f32, f32)> = vec![(f32::NAN, f32::NAN)]; // 0: the module's own
        let steps = (1.5 / TIMING_STEP).round() as usize;
        for i in 0..steps {
            let on = T_ON_LO as f32 + 0.1 + TIMING_STEP * i as f32;
            let mut off = on + TIMING_STEP;
            while off < T_OFF_HI as f32 - 1e-3 {
                pairs.push((on, off));
                off += TIMING_STEP;
            }
            pairs.push((on, T_OFF_HI as f32));
        }
        let c = pairs.len();
        let mut cands = Vec::with_capacity(c * m);
        for &(on, off) in &pairs {
            for k in &current {
                cands.push(if on.is_nan() {
                    *k
                } else {
                    ModuleKinetics {
                        t_on: on,
                        duration: off - on,
                        ..*k
                    }
                });
            }
        }
        let (p, g) = self.xu.dims2()?;
        let dev = self.xu.device().clone();
        let (u, s) = curves(
            &Kinetics::from_modules(&cands, DType::F32, &dev)?,
            &self.tau()?.detach(),
        )?;
        // [P × C × M].
        let ratio = (u.log()? - s.log()?)?.reshape((p, c, m))?;
        let level = (u + s)?.log()?.reshape((p, c, m))?;

        // The track term per (pair, module): every gene against its own
        // module's pairs, offsets profiled, summed within modules.
        let chunk = (GRID_ELEMS / (c * p)).max(1);
        let mut per_gene = Vec::with_capacity(g);
        for start in (0..g).step_by(chunk) {
            let w = chunk.min(g - start);
            let idx = self.module_idx.narrow(0, start, w)?;
            let a = ratio
                .index_select(&idx, 2)?
                .transpose(0, 1)?
                .broadcast_add(&self.kappa.as_tensor().detach().unsqueeze(0)?)?
                .contiguous()?; // [C × P × w]
            let xu = self.xu.narrow(1, start, w)?.unsqueeze(0)?;
            let xs = self.xs.narrow(1, start, w)?.unsqueeze(0)?;
            let n = self.n.narrow(1, start, w)?.unsqueeze(0)?;
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
        let mut onehot = vec![0f32; g * m];
        for (j, &mo) in self.counts.module_of_gene.iter().enumerate() {
            onehot[j * m + mo as usize] = 1.0;
        }
        let track = Tensor::cat(&per_gene, 1)?.matmul(&Tensor::from_vec(onehot, (g, m), &dev)?)?; // [C × M]

        // The which-module term per (pair, module): the candidate's log level,
        // shifted to the module's read-weighted mean level now, replaces its
        // column of the softmax.
        let (_, mag) = self.logits()?;
        let mag = mag.detach(); // [P × M]
        let w = self
            .module_totals
            .broadcast_div(&(self.module_totals.sum_keepdim(0)? + 1e-6)?)?; // [P × M]
        let mean_now = (&mag * &w)?.sum_keepdim(0)?; // [1 × M]
        let mean_cand = level.broadcast_mul(&w.unsqueeze(1)?)?.sum_keepdim(0)?; // [1 × C × M]
        let cand = level
            .broadcast_sub(&mean_cand)?
            .broadcast_add(&mean_now.unsqueeze(1)?)?; // [P × C × M]
        let top = mag
            .max_keepdim(1)?
            .unsqueeze(2)?
            .broadcast_maximum(&cand.max_keepdim(2)?.max_keepdim(1)?)?; // [P × 1 × 1]
        let total_now = mag
            .unsqueeze(1)?
            .broadcast_sub(&top)?
            .exp()?
            .sum_keepdim(2)?; // [P × 1 × 1]
        let others = total_now.broadcast_sub(&mag.unsqueeze(1)?.broadcast_sub(&top)?.exp()?)?; // [P × 1 × M]
        let swapped = others
            .broadcast_add(&cand.broadcast_sub(&top)?.exp()?)?
            .maximum(1e-30)?;
        let lse = swapped.log()?.broadcast_add(&top)?; // [P × C × M]
        let n_p = self.module_totals.sum_keepdim(1)?.unsqueeze(2)?; // [P × 1 × 1]
        let magnitude = (lse.broadcast_mul(&n_p)?
            - cand.broadcast_mul(&self.module_totals.unsqueeze(1)?)?)?
        .sum(0)?; // [C × M]

        let score = (track + magnitude)?.to_vec2::<f32>()?;
        let mut moved = current.clone();
        for mo in 0..m {
            let best = (0..c)
                .min_by(|&x, &y| score[x][mo].total_cmp(&score[y][mo]))
                .expect("pairs");
            if best != 0 && score[best][mo] < score[0][mo] - 1e-3 {
                moved[mo] = cands[best * m + mo];
            }
        }
        let fresh = FreeKinetics::new(&moved, &dev)?;
        for (old, new) in self.free.vars().iter().zip(fresh.vars()) {
            old.set(new.as_tensor())?;
        }
        Ok(())
    }

    /// `κ` at its optimum for the logits `a` (without `κ`), from its current
    /// value.
    fn profiled_kappa(&self, a: &Tensor) -> CResult<Tensor> {
        profile_offset(a, &self.xu, &self.n, self.kappa.as_tensor(), 1)
    }

    /// Move `κ` to its optimum for `a`.
    fn refresh_kappa(&mut self, a: &Tensor) -> CResult<()> {
        self.kappa.set(&self.profiled_kappa(a)?)
    }

    /// Loss per read at the current parameters, the prior included.
    fn evaluate(&mut self) -> CResult<f64> {
        let (a, mag) = self.logits()?;
        let (a, mag) = (a.detach(), mag.detach());
        self.refresh_kappa(&a)?;
        let total = (self.reads_nll(&a, &mag)? + self.prior.nats(&self.tau()?)?)?;
        Ok(f64::from(total.to_scalar::<f32>()?) / self.reads)
    }

    fn adam_steps(&mut self) -> CResult<()> {
        for _ in 0..self.cfg.adam_steps {
            let (a, mag) = self.logits()?;
            self.refresh_kappa(&a.detach())?;
            let loss =
                ((self.reads_nll(&a, &mag)? + self.prior.nats(&self.tau()?)?)? / self.reads)?;
            clipped_backward_step(&mut self.adam, &loss, GRAD_CLIP)?;
        }
        Ok(())
    }

    /// Move every pseudobulk to the grid time that fits its reads and the
    /// prior best (its neighbours where they are), when that beats its current
    /// time; `κ` is profiled at every candidate and at the current time, and a
    /// pseudobulk that moves takes its candidate's.
    fn grid_step(&mut self) -> CResult<()> {
        let kg = self.cfg.grid;
        let (p, g) = self.xu.dims2()?;
        let dev = self.xu.device().clone();
        let grid: Vec<f32> = (0..kg).map(|i| (i as f32 + 0.5) / kg as f32).collect();
        let (a_grid, mag_grid) = self.logits_at(&Tensor::from_vec(grid.clone(), kg, &dev)?)?;
        let a_grid = a_grid.detach().unsqueeze(0)?; // [1 × K × G]
                                                    // `[P × K]` and `[P]`: every pseudobulk's which-module term at every
                                                    // grid time and at its own.
        let (a_now, mag_now) = self.logits()?;
        let (a_now, mag_now) = (a_now.detach(), mag_now.detach());
        let lsm = log_softmax(&mag_grid.detach(), 1)?;
        let mag_grid = self
            .module_totals
            .matmul(&lsm.t()?)?
            .neg()?
            .to_vec2::<f32>()?;
        let mag_now = self.module_nll(&mag_now)?.to_vec1::<f32>()?;
        let kappa_now = self.profiled_kappa(&a_now)?;
        let now = binomial_nll(&a_now.broadcast_add(&kappa_now)?, &self.xu, &self.xs)?
            .sum(1)?
            .to_vec1::<f32>()?;
        let mut kappa = kappa_now.flatten_all()?.to_vec1::<f32>()?;
        let mut tau = self.tau()?.to_vec1::<f32>()?;
        let current = tau.clone();
        let chunk = (GRID_ELEMS / (kg * g)).max(1);
        for start in (0..p).step_by(chunk) {
            let w = chunk.min(p - start);
            let xu = self.xu.narrow(0, start, w)?.unsqueeze(1)?;
            let xs = self.xs.narrow(0, start, w)?.unsqueeze(1)?;
            let n = self.n.narrow(0, start, w)?.unsqueeze(1)?;
            let a = a_grid.broadcast_as((w, kg, g))?.contiguous()?;
            let k0 = kappa_now
                .narrow(0, start, w)?
                .unsqueeze(2)?
                .broadcast_as((w, kg, 1))?
                .contiguous()?;
            let off = profile_offset(&a, &xu, &n, &k0, 2)?;
            let nll = binomial_nll(&a.broadcast_add(&off)?, &xu, &xs)?
                .sum(2)?
                .to_vec2::<f32>()?; // [w × K]
            let off = off.squeeze(2)?.to_vec2::<f32>()?;
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
                    kappa[pb] = off[i][best];
                }
            }
        }
        let z: Vec<f32> = tau.iter().map(|&t| logit(f64::from(t)) as f32).collect();
        self.z.set(&Tensor::from_vec(z, p, &dev)?)?;
        self.kappa.set(&Tensor::from_vec(kappa, (p, 1), &dev)?)
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
        Ok(FitResult {
            tau: self.tau()?.to_vec1()?,
            kappa: self.kappa.as_tensor().flatten_all()?.to_vec1()?,
            offset: self.offset.as_tensor().flatten_all()?.to_vec1()?,
            loading: self.load.as_tensor().flatten_all()?.to_vec1()?,
            modules: self.free.kinetics()?.to_modules()?,
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
pub fn fit_oriented(
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

/// Fit from the times `tau0` and kinetics `start` of a coarser fit, its
/// orientation kept: one step of a coarse-to-fine sequence.
pub fn refine(
    counts: &PbCounts,
    tau0: &[f32],
    start: &[ModuleKinetics],
    prior: &TauPrior,
    cfg: &FitConfig,
) -> anyhow::Result<FitResult> {
    let mut run = Run::new(counts, tau0, Some(start), prior, cfg)?;
    for _ in 0..cfg.rounds {
        run.round()?;
    }
    let last = run.loss.last().copied().unwrap_or(f64::NAN);
    run.finish([last, f64::NAN], false)
}

#[cfg(test)]
#[path = "tests/fit.rs"]
mod tests;
