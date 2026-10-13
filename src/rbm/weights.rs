//! The weights `W` `[F × K]`, always factored through a feature (symbol)
//! embedding `ρ` `[R × H]` and every block's hidden-unit embedding `α_m`
//! `[K × H]` as `W_f = α_{m(f)} ρ_{r(f)}`: the machine reads a row only
//! through where its reads sit in the feature embedding. A feature's row `r(f)` is its own,
//! or, with tied symbols, the row of its name shared across blocks.
//!
//! An earlier run's table can start `ρ` (`Init`), hold the matched rows fixed
//! (`Freeze`), or anchor them with a trained low-rank residual
//! `ρ₀ + U V` (`Lora`).
//!
//! With feature modules (`senna bge`'s layer) every row is a sparse mixture of
//! `M` shared module vectors plus a small residual of its own,
//! `ρ_r = Σ_m sparsemax(logits_r)_m μ_m + r_r`: a feature with few reads still
//! gets a trained row through its module's siblings.

use legume_numeric::candle::candle_core::{DType, Device, Result, Tensor, Var};
use legume_numeric::candle::nn::sparsemax;
use rand::rngs::StdRng;
use rand::SeedableRng;
use rand_distr::{Distribution, Normal};
use std::cell::Cell;
use std::ops::Range;

/// Scale of the initial weights.
const INIT_SD: f64 = 0.01;

pub struct Weights {
    /// `[R × H]`
    pub rho: Var,
    /// Per block, `[K × H]`.
    pub alpha: Vec<Var>,
    /// `[F]` u32: every feature's row of `ρ`.
    pub row: Tensor,
    /// A given table: `ρ₀` `[R × H]` and its rows' mask `[R × 1]`.
    pub given: Option<(Tensor, Tensor)>,
    /// `Freeze`/`Lora` hold the given rows; `Init` only started from them.
    pub pinned: bool,
    /// The low-rank residual `U` `[R × r]`, `V` `[r × H]` on the given rows.
    pub lora: Option<(Var, Var)>,
    /// The module layer, which then composes `ρ` in place of the free table.
    pub modules: Option<Modules>,
}

/// `ρ = sparsemax(logits) μ + r`, as `senna bge` composes it.
pub struct Modules {
    /// `[R × M]` membership logits.
    pub logits: Var,
    /// `[M × H]` module vectors.
    pub mu: Var,
    /// `[R × H]` every row's own residual.
    pub residual: Var,
    /// Every row's starting module.
    pub labels: Vec<u32>,
    /// While set, the membership is held at its start.
    frozen: Cell<bool>,
}

impl Modules {
    /// The membership `π` `[R × M]`: rows on the simplex with exact zeros.
    pub fn membership(&self) -> Result<Tensor> {
        if self.frozen.get() {
            sparsemax(&self.logits.as_tensor().detach())
        } else {
            sparsemax(self.logits.as_tensor())
        }
    }
}

/// What an earlier table gives: the rows `ids` of `ρ` it matched, row-major
/// `[ids × H]`.
pub struct Given<'a> {
    pub ids: &'a [u32],
    pub rows: &'a [f32],
    pub mode: graph_embedding_util::PresetMode,
}

fn normal(n: usize, sd: f64, rng: &mut StdRng) -> Vec<f32> {
    let d = Normal::new(0.0, sd).expect("a valid sd");
    (0..n).map(|_| d.sample(rng) as f32).collect()
}

impl Weights {
    /// `ρ` `[R × H]` with `row[f]` every feature's row, one `α` per block.
    pub fn new(
        row: &[u32],
        k: usize,
        h: usize,
        blocks: usize,
        given: Option<Given<'_>>,
        seed: u64,
        dev: &Device,
    ) -> anyhow::Result<Self> {
        let mut rng = StdRng::seed_from_u64(seed);
        let r = row.iter().copied().max().map_or(0, |m| m as usize + 1);
        let mut rho = normal(r * h, 0.1, &mut rng);
        let mut mask = vec![0f32; r];
        let mut lora_rank = None;
        let mut pinned = false;
        if let Some(g) = given.as_ref() {
            anyhow::ensure!(
                g.rows.len() == g.ids.len() * h,
                "the given feature embedding is {} wide, not H = {h}",
                g.rows.len() / g.ids.len().max(1)
            );
            for (i, &id) in g.ids.iter().enumerate() {
                let dst = id as usize;
                rho[dst * h..(dst + 1) * h].copy_from_slice(&g.rows[i * h..(i + 1) * h]);
                mask[dst] = 1.0;
            }
            pinned = g.mode.pins();
            lora_rank = g.mode.lora().map(|s| s.rank);
        }
        // `α` small against `ρ`'s scale, so `W` starts at `INIT_SD`.
        let rms = (rho.iter().map(|x| f64::from(x * x)).sum::<f64>() / rho.len().max(1) as f64)
            .sqrt()
            .max(1e-3);
        let alpha_sd = INIT_SD / (rms * (h as f64).sqrt());
        let alpha = (0..blocks)
            .map(|_| {
                Var::from_tensor(&Tensor::from_vec(
                    normal(k * h, alpha_sd, &mut rng),
                    (k, h),
                    dev,
                )?)
            })
            .collect::<Result<Vec<_>>>()?;
        let rho_t = Tensor::from_vec(rho, (r, h), dev)?;
        let given_t = match given {
            Some(_) => Some((rho_t.clone(), Tensor::from_vec(mask, (r, 1), dev)?)),
            None => None,
        };
        let lora = match lora_rank {
            Some(rank) => Some((
                Var::zeros((r, rank), DType::F32, dev)?,
                Var::from_tensor(&Tensor::from_vec(
                    normal(rank * h, 0.1, &mut rng),
                    (rank, h),
                    dev,
                )?)?,
            )),
            None => None,
        };
        Ok(Self {
            rho: Var::from_tensor(&rho_t)?,
            alpha,
            row: Tensor::from_slice(row, row.len(), dev)?,
            given: given_t,
            pinned,
            lora,
            modules: None,
        })
    }

    /// The module layer over the rows of `ρ`, every row starting with
    /// `own_mass` of its membership on its module in `labels`.
    pub fn with_modules(
        mut self,
        labels: Vec<u32>,
        n_modules: usize,
        own_mass: f32,
        seed: u64,
    ) -> anyhow::Result<Self> {
        anyhow::ensure!(
            self.given.is_none(),
            "feature modules compose every row from shared vectors, so there is no per-feature \
             table to start from a given one: drop --feature-modules or the given embedding"
        );
        let dev = self.rho.device().clone();
        let (r, h) = self.rho.dims2()?;
        anyhow::ensure!(
            labels.len() == r,
            "{} module labels for {r} rows",
            labels.len()
        );
        let kappa = graph_embedding_util::model::module_logit_for_own_mass(own_mass, n_modules);
        let mut logits = vec![0f32; r * n_modules];
        for (i, &l) in labels.iter().enumerate() {
            logits[i * n_modules + l as usize] = kappa;
        }
        let mut rng = StdRng::seed_from_u64(seed);
        self.modules = Some(Modules {
            logits: Var::from_tensor(&Tensor::from_vec(logits, (r, n_modules), &dev)?)?,
            mu: Var::from_tensor(&Tensor::from_vec(
                normal(n_modules * h, 0.1, &mut rng),
                (n_modules, h),
                &dev,
            )?)?,
            residual: Var::zeros((r, h), DType::F32, &dev)?,
            labels,
            frozen: Cell::new(false),
        });
        Ok(self)
    }

    /// Hold (`true`) or release the module membership.
    pub fn set_frozen(&self, frozen: bool) {
        if let Some(m) = self.modules.as_ref() {
            m.frozen.set(frozen);
        }
    }

    /// Penalties beside the likelihood: L2 `l2` on the rows of the free table
    /// (or the module vectors) and of `α`; with modules, `bge`'s load-balance
    /// prior `KL(π̄ ‖ uniform)` at `balance` and the residual's ridge at `ridge`.
    pub fn penalty(&self, l2: f64, balance: f64, ridge: f64) -> Result<Tensor> {
        let row_l2 = |t: &Tensor| t.sqr()?.sum(1)?.mean_all();
        let mut total = match self.modules.as_ref() {
            Some(m) => row_l2(m.mu.as_tensor())?,
            None => row_l2(self.rho.as_tensor())?,
        };
        for a in &self.alpha {
            total = (total + row_l2(a.as_tensor())?)?;
        }
        total = (total * l2)?;
        if let Some(m) = self.modules.as_ref() {
            let pi_bar = m.membership()?.mean(0)?;
            let n = pi_bar.dim(0)? as f64;
            let kl = (&pi_bar * pi_bar.affine(n, 1e-12)?.log()?)?.sum_all()?;
            total = (total + (kl * balance)? + (row_l2(m.residual.as_tensor())? * ridge)?)?;
        }
        Ok(total)
    }

    /// Start from the data: the free rows of `ρ` on the leading singular
    /// directions of the centred reads `x` `[P × F]` (a row of `ρ` shared by
    /// several features takes their mean), every hidden unit on its own
    /// orthogonal row of `α`, scaled together so the units' inputs vary by
    /// one on average over the rows of `x`. Factors started small and equal grow along one direction
    /// together, and every unit ends on it; this keeps them apart.
    pub fn start_from(
        &self,
        x: &nalgebra::DMatrix<f32>,
        ranges: &[Range<usize>],
        seed: u64,
    ) -> anyhow::Result<()> {
        use legume_numeric::matrix::traits::*;
        let dev = self.rho.device().clone();
        let (r, h) = self.rho.dims2()?;
        let row: Vec<u32> = self.row.to_vec1()?;
        let mask: Vec<f32> = match self.given.as_ref() {
            Some((_, m)) => m.flatten_all()?.to_vec1()?,
            None => vec![0.0; r],
        };
        if mask.contains(&0.0) {
            let (u, _, _) = x.transpose().rsvd(h)?;
            let mut rho = self.rho.as_tensor().to_vec2::<f32>()?;
            let mut fresh = vec![vec![0f32; h]; r];
            let mut count = vec![0f32; r];
            for (f, &ri) in row.iter().enumerate() {
                let ri = ri as usize;
                for d in 0..u.ncols().min(h) {
                    fresh[ri][d] += u[(f, d)];
                }
                count[ri] += 1.0;
            }
            // Rows a given table leaves out start at its rows' typical length,
            // so the table does not set α's scale with the new rows near zero.
            let norm = |v: &[f32]| v.iter().map(|x| x * x).sum::<f32>().sqrt();
            let mean_norm = |rows: &mut dyn Iterator<Item = f32>| {
                let (s, n) = rows.fold((0f32, 0f32), |(s, n), x| (s + x, n + 1.0));
                (n > 0.0).then(|| s / n)
            };
            for ri in 0..r {
                if count[ri] > 0.0 {
                    fresh[ri].iter_mut().for_each(|v| *v /= count[ri]);
                }
            }
            let given_len =
                mean_norm(&mut (0..r).filter(|&ri| mask[ri] > 0.0).map(|ri| norm(&rho[ri])));
            let fresh_len = mean_norm(
                &mut (0..r)
                    .filter(|&ri| mask[ri] == 0.0 && count[ri] > 0.0)
                    .map(|ri| norm(&fresh[ri])),
            );
            let lift = match (given_len, fresh_len) {
                (Some(g), Some(f)) if f > 0.0 => g / f,
                _ => 1.0,
            };
            for ri in 0..r {
                if mask[ri] == 0.0 && count[ri] > 0.0 {
                    rho[ri] = fresh[ri].iter().map(|v| v * lift).collect();
                }
            }
            match self.modules.as_ref() {
                // Every module vector at the mean of its members' rows.
                Some(m) => {
                    let n = m.mu.dim(0)?;
                    let mut mu = vec![vec![0f32; h]; n];
                    let mut size = vec![0f32; n];
                    for (ri, &l) in m.labels.iter().enumerate() {
                        for d in 0..h {
                            mu[l as usize][d] += rho[ri][d];
                        }
                        size[l as usize] += 1.0;
                    }
                    for (row, &c) in mu.iter_mut().zip(&size) {
                        row.iter_mut().for_each(|v| *v /= c.max(1.0));
                    }
                    m.mu.set(&Tensor::from_vec(mu.concat(), (n, h), &dev)?)?;
                }
                None => self
                    .rho
                    .set(&Tensor::from_vec(rho.concat(), (r, h), &dev)?)?,
            }
        }
        let mut rng = StdRng::seed_from_u64(seed);
        for a in &self.alpha {
            let k = a.dim(0)?;
            let g = nalgebra::DMatrix::<f32>::from_vec(h, k, normal(h * k, 1.0, &mut rng));
            let rows = if k <= h {
                g.qr().q().columns(0, k).transpose()
            } else {
                let mut t = g.transpose();
                for mut row in t.row_iter_mut() {
                    let n = row.norm().max(1e-12);
                    row /= n;
                }
                t
            };
            a.set(&Tensor::from_iter(rows.transpose().iter().copied(), &dev)?.reshape((k, h))?)?;
        }
        // Every unit's input over the rows of `x`, to one standard deviation.
        let xs = Tensor::from_iter(x.transpose().iter().copied(), &dev)?.reshape(x.shape())?;
        let input = xs.matmul(&self.w(ranges)?)?;
        let sd: Vec<f32> = input
            .broadcast_sub(&input.mean_keepdim(0)?)?
            .sqr()?
            .mean(0)?
            .sqrt()?
            .to_vec1()?;
        // One scale for every unit, so their mean input variance is one: a
        // unit aimed at a weak direction starts small rather than amplified
        // onto noise.
        let rms = (sd.iter().map(|&v| f64::from(v * v)).sum::<f64>() / sd.len().max(1) as f64)
            .sqrt()
            .max(1e-6);
        for a in &self.alpha {
            a.set(&(a.as_tensor() / rms)?)?;
        }
        // The same units read back: a hidden state moves a feature's logit by
        // `W_f (h − λ)`, so cap the features' mean `‖W_f‖²` at one. Unit inputs
        // set to one alone can make it large when many units, or rows of `ρ`
        // shared through a module, add up on the way back.
        let w2: f32 = self.w(ranges)?.sqr()?.sum(1)?.mean_all()?.to_scalar()?;
        if w2 > 1.0 {
            let shrink = f64::from(w2.sqrt()).recip();
            for a in &self.alpha {
                a.set(&(a.as_tensor() * shrink)?)?;
            }
        }
        Ok(())
    }

    /// The variables the main optimiser trains.
    pub fn vars(&self) -> Vec<Var> {
        let mut v = self.alpha.clone();
        if self.modules.is_none() {
            v.push(self.rho.clone());
        }
        if let Some((u, _)) = self.lora.as_ref() {
            v.push(u.clone());
        }
        v
    }

    /// The module layer's variables, trained at their own, smaller rate: one
    /// module vector moves every member's row at once, and the background
    /// module can hold most of the features.
    pub fn module_vars(&self) -> Vec<Var> {
        self.modules.as_ref().map_or_else(Vec::new, |m| {
            vec![m.logits.clone(), m.mu.clone(), m.residual.clone()]
        })
    }

    /// The LoRA residual's shared factor, trained at its own rate (LoRA+).
    pub fn lora_shared(&self) -> Option<Var> {
        self.lora.as_ref().map(|(_, v)| v.clone())
    }

    /// The effective `ρ` `[R × H]`.
    pub fn rho(&self) -> Result<Tensor> {
        if let Some(m) = self.modules.as_ref() {
            return m.membership()?.matmul(m.mu.as_tensor())? + m.residual.as_tensor();
        }
        let rho = self.rho.as_tensor();
        Ok(match (self.given.as_ref(), self.pinned) {
            (Some((rho0, mask)), true) => {
                let mut held = rho0.clone();
                if let Some((u, v)) = self.lora.as_ref() {
                    held = (held + u.as_tensor().matmul(v.as_tensor())?)?;
                }
                (rho.broadcast_mul(&mask.affine(-1.0, 1.0)?)? + held.broadcast_mul(mask)?)?
            }
            _ => rho.clone(),
        })
    }

    /// The LoRA residual's ridge `Σ_r ‖U_r V‖²` on the given rows.
    pub fn lora_penalty(&self) -> Result<Option<Tensor>> {
        match (self.lora.as_ref(), self.given.as_ref()) {
            (Some((u, v)), Some((_, mask))) => Ok(Some(
                u.as_tensor()
                    .matmul(v.as_tensor())?
                    .broadcast_mul(mask)?
                    .sqr()?
                    .sum_all()?,
            )),
            _ => Ok(None),
        }
    }

    /// `W` `[F × K]`, blocks contiguous in `ranges`.
    pub fn w(&self, ranges: &[Range<usize>]) -> Result<Tensor> {
        let rho_f = self.rho()?.index_select(&self.row, 0)?;
        let parts = ranges
            .iter()
            .zip(&self.alpha)
            .map(|(r, a)| {
                rho_f
                    .narrow(0, r.start, r.len())?
                    .matmul(&a.as_tensor().t()?)
            })
            .collect::<Result<Vec<_>>>()?;
        Tensor::cat(&parts, 0)
    }
}

#[cfg(test)]
#[path = "tests/weights.rs"]
mod tests;
