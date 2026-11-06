//! The pseudobulk fit: every pseudobulk a point `x_p`, every gene a vector
//! `w_g` and a baseline `b_g`, the curves `μ_ℓ` ([`super::curves`]), all
//! candle variables, fit by AdamW on
//!
//! ```text
//! L = − Σ_p Σ_g y_pg log softmax_g(x_pᵀ w_g + b_g)
//!     + Σ_p [ (1/2σ²) Σ_{ℓ,k} q_p(ℓ,k) |x_p − μ_ℓ(k)|² + H log σ ]
//!     + ½|w|² + ½|η|² + ½ (log a)² + ½ G b̄²
//! ```
//!
//! every term at weight one. The softmax is over every gene, the scores
//! clamped to `±30`. `q_p` is the ode run's posterior of pseudobulk `p` on
//! the grid, renormalised; the bracket sums over the pseudobulks that have
//! one, each one point under one Gaussian whatever its depth (no weight by
//! its reads), so `σ` is learned with its normaliser. `½|w|²` also pins the
//! shift every `w_g` can share (`x_pᵀ v` added to every gene's score leaves
//! the softmax as it is), `½ G b̄²` the shift every `b_g` can.

use super::curves::Curves;
use crate::rbm_core::adam;
use legume_numeric::candle::candle_core::{DType, Device, Result, Tensor, Var};
use legume_numeric::candle::candle_nn::ops::log_softmax;
use legume_numeric::candle::candle_nn::Optimizer;
use legume_numeric::matrix::traits::RandomizedAlgs;
use nalgebra::DMatrix;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::SeedableRng;
use rand_distr::{Distribution, Normal};

/// Scores are clamped to this before the softmax.
pub(super) const SCORE_CLAMP: f64 = 30.0;

/// The sd of the genes' vectors at the start.
const W_INIT_SD: f64 = 0.01;

/// One pseudobulk's posterior mass at one lineage and position.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct QMass {
    pub pb: usize,
    pub lineage: usize,
    pub position: usize,
    pub q: f32,
}

impl QMass {
    /// `(lineage, position, mass)`.
    pub fn on_grid(&self) -> (usize, usize, f32) {
        (self.lineage, self.position, self.q)
    }
}

/// The reads to fit and the ode run's posterior over the grid.
pub struct Pseudobulks {
    /// `[P × G]` row-major reads.
    pub y: Vec<f32>,
    pub n_pb: usize,
    pub n_genes: usize,
    /// Sorted by pseudobulk, every pseudobulk's summing to one.
    q: Vec<QMass>,
    /// `[P + 1]` every pseudobulk's run in `q`.
    start: Vec<usize>,
}

impl Pseudobulks {
    /// `q` renormalised within every pseudobulk, mass that is not positive
    /// and finite dropped; a pseudobulk with none left has no prior.
    pub fn new(y: Vec<f32>, n_pb: usize, n_genes: usize, q: Vec<QMass>) -> anyhow::Result<Self> {
        anyhow::ensure!(
            y.len() == n_pb * n_genes,
            "{} reads for {n_pb} pseudobulks × {n_genes} genes",
            y.len()
        );
        if let Some(m) = q.iter().find(|m| m.pb >= n_pb) {
            anyhow::bail!("posterior mass on pseudobulk {} of {n_pb}", m.pb);
        }
        let mut q: Vec<QMass> = q
            .into_iter()
            .filter(|m| m.q.is_finite() && m.q > 0.0)
            .collect();
        q.sort_by_key(|m| (m.pb, m.lineage, m.position));
        let mut total = vec![0f32; n_pb];
        for m in &q {
            total[m.pb] += m.q;
        }
        let mut start = vec![0usize; n_pb + 1];
        for m in &mut q {
            m.q /= total[m.pb];
            start[m.pb + 1] += 1;
        }
        for p in 0..n_pb {
            start[p + 1] += start[p];
        }
        Ok(Self {
            y,
            n_pb,
            n_genes,
            q,
            start,
        })
    }

    pub fn q(&self) -> &[QMass] {
        &self.q
    }

    /// Pseudobulk `p`'s posterior; empty when it has none.
    pub fn of(&self, p: usize) -> &[QMass] {
        &self.q[self.start[p]..self.start[p + 1]]
    }

    /// Whether every pseudobulk has a prior.
    pub fn has_prior(&self) -> Vec<bool> {
        (0..self.n_pb).map(|p| !self.of(p).is_empty()).collect()
    }
}

/// Every parameter's value, row-major: `x [P × H]`, `w [G × H]`, `b [G]`,
/// `eta [L × m × H]`.
#[derive(Clone, Debug)]
pub struct Parts {
    pub x: Vec<f32>,
    pub w: Vec<f32>,
    pub b: Vec<f32>,
    pub eta: Vec<f32>,
    pub log_a: f32,
    pub log_sigma: f32,
}

/// The parameters, the curves and the reads on the device.
pub struct Model {
    curves: Curves,
    x: Var,
    w: Var,
    b: Var,
    eta: Var,
    log_a: Var,
    log_sigma: Var,
    /// `[P × G]`.
    y: Tensor,
    /// `[P + 1]` every pseudobulk's run in `cell` and `mass`, its
    /// posterior's grid rows `ℓ n + k` and masses.
    start: Vec<usize>,
    cell: Vec<u32>,
    mass: Vec<f32>,
    /// The prior's gather over every pseudobulk; `None` if none has one.
    all: Option<Gather>,
}

/// A set of pseudobulks' prior on the device: every mass's row in the set
/// (`local`), its grid row and its mass; how many have one.
struct Gather {
    local: Tensor,
    cell: Tensor,
    mass: Tensor,
    with_prior: usize,
}

fn var(v: Vec<f32>, shape: &[usize], dev: &Device) -> Result<Var> {
    Var::from_tensor(&Tensor::from_vec(v, shape, dev)?)
}

/// `[P × h]` the leading `h` components of every pseudobulk's log1p CPM,
/// scaled to entries of mean square one; components past the reads' rank
/// small noise.
fn pca_init(data: &Pseudobulks, h: usize, rng: &mut StdRng) -> anyhow::Result<Vec<f32>> {
    let (p_n, g) = (data.n_pb, data.n_genes);
    let mut m = DMatrix::<f64>::from_fn(p_n, g, |p, j| {
        let total: f64 = data.y[p * g..(p + 1) * g]
            .iter()
            .map(|&v| f64::from(v))
            .sum();
        if total > 0.0 {
            (f64::from(data.y[p * g + j]) / total * 1e6).ln_1p()
        } else {
            0.0
        }
    });
    for j in 0..g {
        let mean = m.column(j).mean();
        m.column_mut(j).add_scalar_mut(-mean);
    }
    let k = h.min(p_n.saturating_sub(1)).min(g);
    let mut x = vec![0f64; p_n * h];
    if k > 0 {
        let (u, d, _) = m.rsvd(k)?;
        for p in 0..p_n {
            for c in 0..k.min(d.len()) {
                x[p * h + c] = u[(p, c)] * d[c];
            }
        }
    }
    let rms = (x.iter().map(|v| v * v).sum::<f64>() / x.len().max(1) as f64).sqrt();
    let noise = Normal::new(0.0, W_INIT_SD)?;
    Ok(x.iter()
        .enumerate()
        .map(|(i, &v)| {
            if i % h >= k || rms == 0.0 {
                noise.sample(rng) as f32
            } else {
                (v / rms) as f32
            }
        })
        .collect())
}

impl Model {
    /// `x` from a PCA of log1p CPM, `w` small and random, `b` every gene's
    /// log share (centred), `η = 0` (every curve at the origin), `a = σ = 1`.
    pub fn init(
        data: &Pseudobulks,
        curves: Curves,
        dim: usize,
        seed: u64,
        dev: &Device,
    ) -> anyhow::Result<Self> {
        let mut rng = StdRng::seed_from_u64(seed);
        let (p_n, g) = (data.n_pb, data.n_genes);
        let x = pca_init(data, dim, &mut rng)?;
        let noise = Normal::new(0.0, W_INIT_SD)?;
        let w = (0..g * dim)
            .map(|_| noise.sample(&mut rng) as f32)
            .collect();
        let mut b: Vec<f64> = (0..g)
            .map(|j| (0..p_n).map(|p| f64::from(data.y[p * g + j])).sum::<f64>() + 0.5)
            .map(f64::ln)
            .collect();
        let mean = b.iter().sum::<f64>() / g.max(1) as f64;
        b.iter_mut().for_each(|v| *v -= mean);
        let parts = Parts {
            x,
            w,
            b: b.iter().map(|&v| v as f32).collect(),
            eta: vec![0.0; curves.lineages() * curves.basis_size() * dim],
            log_a: 0.0,
            log_sigma: 0.0,
        };
        Self::from_parts(data, curves, &parts, dev)
    }

    pub fn from_parts(
        data: &Pseudobulks,
        curves: Curves,
        parts: &Parts,
        dev: &Device,
    ) -> anyhow::Result<Self> {
        let (p_n, g) = (data.n_pb, data.n_genes);
        anyhow::ensure!(p_n > 0 && g > 0, "no pseudobulk or no gene to fit");
        let h = parts.x.len() / p_n;
        let (l, m, n) = (curves.lineages(), curves.basis_size(), curves.positions());
        anyhow::ensure!(
            h > 0
                && parts.x.len() == p_n * h
                && parts.w.len() == g * h
                && parts.b.len() == g
                && parts.eta.len() == l * m * h,
            "parameter sizes do not agree"
        );
        if let Some(q) = data.q().iter().find(|q| q.lineage >= l || q.position >= n) {
            anyhow::bail!(
                "posterior mass at lineage {} position {} off a grid of {l} × {n}",
                q.lineage,
                q.position
            );
        }
        let mut model = Self {
            x: var(parts.x.clone(), &[p_n, h], dev)?,
            w: var(parts.w.clone(), &[g, h], dev)?,
            b: var(parts.b.clone(), &[g], dev)?,
            eta: var(parts.eta.clone(), &[l, m, h], dev)?,
            log_a: var(vec![parts.log_a], &[1], dev)?,
            log_sigma: var(vec![parts.log_sigma], &[1], dev)?,
            y: Tensor::from_vec(data.y.clone(), (p_n, g), dev)?,
            start: data.start.clone(),
            cell: data
                .q()
                .iter()
                .map(|q| (q.lineage * n + q.position) as u32)
                .collect(),
            mass: data.q().iter().map(|q| q.q).collect(),
            all: None,
            curves,
        };
        model.all = model.gather(&(0..p_n).collect::<Vec<_>>())?;
        Ok(model)
    }

    /// The prior of the pseudobulks `rows` on the device.
    fn gather(&self, rows: &[usize]) -> Result<Option<Gather>> {
        let (mut local, mut cell, mut mass, mut with_prior) = (vec![], vec![], vec![], 0usize);
        for (i, &p) in rows.iter().enumerate() {
            let run = self.start[p]..self.start[p + 1];
            with_prior += usize::from(!run.is_empty());
            local.extend(std::iter::repeat_n(i as u32, run.len()));
            cell.extend_from_slice(&self.cell[run.clone()]);
            mass.extend_from_slice(&self.mass[run]);
        }
        if mass.is_empty() {
            return Ok(None);
        }
        let (t, dev) = (mass.len(), self.y.device());
        Ok(Some(Gather {
            local: Tensor::from_vec(local, (t,), dev)?,
            cell: Tensor::from_vec(cell, (t,), dev)?,
            mass: Tensor::from_vec(mass, (t,), dev)?,
            with_prior,
        }))
    }

    fn vars(&self) -> Vec<Var> {
        vec![
            self.x.clone(),
            self.w.clone(),
            self.b.clone(),
            self.eta.clone(),
            self.log_a.clone(),
            self.log_sigma.clone(),
        ]
    }

    pub fn curves(&self) -> &Curves {
        &self.curves
    }

    /// Pseudobulks.
    pub fn n_pb(&self) -> usize {
        self.y.dims()[0]
    }

    /// The embedding's dimension `H`.
    pub fn dim(&self) -> usize {
        self.x.as_tensor().dims()[1]
    }

    /// The loss over the pseudobulks `rows`, their terms scaled by
    /// `P/|rows|` so that its mean over a partition is the full loss.
    pub fn loss(&self, rows: &[usize]) -> Result<Tensor> {
        let (p_n, g, h) = (self.n_pb(), self.y.dims()[1], self.dim());
        let whole = rows.len() == p_n && rows.iter().enumerate().all(|(i, &p)| i == p);
        let part;
        let (x, y, prior) = if whole {
            (
                self.x.as_tensor().clone(),
                self.y.clone(),
                self.all.as_ref(),
            )
        } else {
            let idx = Tensor::from_vec(
                rows.iter().map(|&p| p as u32).collect::<Vec<_>>(),
                (rows.len(),),
                self.y.device(),
            )?;
            part = self.gather(rows)?;
            (
                self.x.as_tensor().index_select(&idx, 0)?,
                self.y.index_select(&idx, 0)?,
                part.as_ref(),
            )
        };
        let scale = p_n as f64 / rows.len().max(1) as f64;

        let scores = x
            .matmul(&self.w.as_tensor().t()?)?
            .broadcast_add(&self.b.as_tensor().unsqueeze(0)?)?
            .clamp(-SCORE_CLAMP, SCORE_CLAMP)?;
        let mut fit = (log_softmax(&scores, 1)? * &y)?.sum_all()?.neg()?;

        if let Some(q) = prior {
            let n = self.curves.positions();
            let mu = self
                .curves
                .mu(self.eta.as_tensor(), self.log_a.as_tensor())?
                .reshape((self.curves.lineages() * n, h))?;
            let xm = x.index_select(&q.local, 0)?;
            let mm = mu.index_select(&q.cell, 0)?;
            let d2 = (xm - mm)?.sqr()?.sum(1)?;
            let weighted = (d2 * &q.mass)?.sum_all()?;
            let log_sigma = self.log_sigma.as_tensor().squeeze(0)?;
            let inv_two_var = ((log_sigma.clone() * -2.0)?.exp()? * 0.5)?;
            let prior = ((weighted * inv_two_var)? + (log_sigma * (q.with_prior * h) as f64)?)?;
            fit = (fit + prior)?;
        }
        let w2 = (self.w.as_tensor().sqr()?.sum_all()? * 0.5)?;
        let eta2 = (self.eta.as_tensor().sqr()?.sum_all()? * 0.5)?;
        let a2 = (self.log_a.as_tensor().sqr()?.sum_all()? * 0.5)?;
        let b_mean = self.b.as_tensor().mean_all()?;
        let gauge = (b_mean.sqr()? * (0.5 * g as f64))?;
        (fit * scale)? + w2 + eta2 + a2 + gauge
    }

    /// `epochs` passes of AdamW at rate `lr`, over every pseudobulk at once
    /// or, with `pb_per_step`, over a shuffled partition of them (by
    /// `seed`) a part a step. Every epoch's loss (its steps' mean).
    pub fn fit(
        &self,
        epochs: usize,
        lr: f64,
        pb_per_step: Option<usize>,
        seed: u64,
    ) -> Result<Vec<f32>> {
        let p_n = self.n_pb();
        let step = pb_per_step.filter(|&s| s < p_n).unwrap_or(p_n).max(1);
        let mut opt = adam(self.vars(), lr)?;
        let mut rng = StdRng::seed_from_u64(seed ^ 0xb47c);
        let mut order: Vec<usize> = (0..p_n).collect();
        let mut trace = Vec::with_capacity(epochs);
        for _ in 0..epochs {
            if step < p_n {
                order.shuffle(&mut rng);
            }
            let (mut total, mut steps) = (None::<Tensor>, 0usize);
            for chunk in order.chunks(step) {
                let loss = self.loss(chunk)?;
                opt.backward_step(&loss)?;
                let loss = loss.detach();
                total = Some(match total {
                    Some(t) => (t + loss)?,
                    None => loss,
                });
                steps += 1;
            }
            if let Some(t) = total {
                trace.push(t.to_dtype(DType::F32)?.to_vec0::<f32>()? / steps as f32);
            }
        }
        Ok(trace)
    }

    /// `[P × H]` row-major.
    pub fn x(&self) -> Result<Vec<f32>> {
        self.x.as_tensor().flatten_all()?.to_vec1()
    }

    /// `[G × H]` row-major.
    pub fn w(&self) -> Result<Vec<f32>> {
        self.w.as_tensor().flatten_all()?.to_vec1()
    }

    /// `[G]`.
    pub fn b(&self) -> Result<Vec<f32>> {
        self.b.as_tensor().to_vec1()
    }

    /// `[L × n × H]` row-major, the curves.
    pub fn mu(&self) -> Result<Vec<f32>> {
        self.curves
            .mu(self.eta.as_tensor(), self.log_a.as_tensor())?
            .flatten_all()?
            .to_vec1()
    }

    pub fn sigma(&self) -> Result<f32> {
        Ok(self.log_sigma.as_tensor().to_vec1::<f32>()?[0].exp())
    }

    /// The curves' amplitude `a`.
    pub fn amplitude(&self) -> Result<f32> {
        Ok(self.log_a.as_tensor().to_vec1::<f32>()?[0].exp())
    }
}
