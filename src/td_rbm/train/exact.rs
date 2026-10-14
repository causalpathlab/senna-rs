//! Step 1: every lineage and its time by the exact likelihood, the
//! lineages drawn by collapsed Gibbs under a truncated Dirichlet process.

use super::{adam, pass_reads, Level, TrainConfig};
use crate::logging::new_progress_bar;
use crate::td_rbm::curve::Spline;
use crate::td_rbm::model::Energy;
use crate::td_rbm::rbm::{lineage_log_likelihood, Reads};
use crate::td_rbm::start::resample;
use legume_numeric::candle::candle_core::{Device, Result, Tensor};
use legume_numeric::candle::candle_nn::Optimizer;
use log::info;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::RngExt;

/// An index drawn with log weights `w`.
fn draw(w: &[f64], rng: &mut StdRng) -> usize {
    let top = w.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let p: Vec<f64> = w.iter().map(|x| (x - top).exp()).collect();
    let mut u = rng.random::<f64>() * p.iter().sum::<f64>();
    for (i, &pi) in p.iter().enumerate() {
        if u < pi {
            return i;
        }
        u -= pi;
    }
    p.len() - 1
}

/// The rows' summed log-likelihood under their assigned lineages, `[]`,
/// with the graph: `ll` `[N × L]`, `assigned` `[N]`.
fn assigned_log_likelihood(ll: &Tensor, assigned: &[usize]) -> Result<Tensor> {
    let ids: Vec<u32> = assigned.iter().map(|&a| a as u32).collect();
    let ids = Tensor::from_vec(ids, (assigned.len(), 1), ll.device())?;
    ll.gather(&ids, 1)?.sum_all()
}

/// The lineage-and-time loop shared by the steady and the moving phases.
pub(super) struct Phase<'a> {
    pub(super) energy: &'a Energy,
    pub(super) spline: &'a Spline,
    pub(super) levels: &'a [Level],
    pub(super) f: usize,
    pub(super) cfg: &'a TrainConfig,
    pub(super) dev: &'a Device,
    /// Every level's rows' lineages, and the cells every lineage holds (the
    /// finest level's).
    pub(super) assigned: &'a mut Vec<Vec<usize>>,
    pub(super) counts: &'a mut Vec<f64>,
}

impl Phase<'_> {
    /// `epochs` epochs: every level drawn again, every row's lineage chosen by
    /// collapsed Gibbs under the truncated Dirichlet process (the finest
    /// level's rows move the counts; rich get richer), every lineage's curve
    /// fitted on its rows by the exact likelihood, time summed out.
    pub(super) fn epochs(
        &mut self,
        epochs: usize,
        label: &str,
        rng: &mut StdRng,
    ) -> anyhow::Result<()> {
        let (cfg, f, dev) = (self.cfg, self.f, self.dev);
        let l = cfg.lineages;
        let finest = self.levels.len() - 1;
        let prior = cfg.concentration / l as f64;
        let temper = cfg.assign_depth / cfg.depth as f64;
        let zero = vec![0f32; l];
        let pass_reads = pass_reads(self.levels, cfg.depth);
        let mut opt = adam(self.energy.vars(), cfg.lr)?;
        let bar = new_progress_bar(epochs as u64).with_message(format!("td-rbm {label}"));
        for epoch in 0..epochs {
            bar.inc(1);
            let (mut nll, mut reads) = (0f64, 0f64);
            let drawn: Vec<Reads> = self
                .levels
                .iter()
                .map(|lv| {
                    let rows = lv.cells.len();
                    Reads::from_features(Tensor::from_vec(
                        resample(&lv.x, f, cfg.depth, rng),
                        (rows, f),
                        dev,
                    )?)
                })
                .collect::<Result<_>>()?;
            let t = self.energy.tables(self.spline, false, &zero)?.detach();
            for (i, r) in drawn.iter().enumerate() {
                let ll = lineage_log_likelihood(r, &t)?.to_vec2::<f32>()?;
                let mut order: Vec<usize> = (0..ll.len()).collect();
                order.shuffle(rng);
                for row in order {
                    let cells = f64::from(self.levels[i].cells[row]);
                    if i == finest {
                        self.counts[self.assigned[i][row]] -= cells;
                    }
                    let w: Vec<f64> = (0..l)
                        .map(|j| {
                            (self.counts[j].max(0.0) + prior).ln() + f64::from(ll[row][j]) * temper
                        })
                        .collect();
                    let a = draw(&w, rng);
                    self.assigned[i][row] = a;
                    if i == finest {
                        self.counts[a] += cells;
                    }
                }
            }
            for (i, r) in drawn.iter().enumerate() {
                let mut order: Vec<u32> = (0..r.x.dim(0)? as u32).collect();
                order.shuffle(rng);
                for chunk in order.chunks(cfg.batch) {
                    let batch = r.select(&Tensor::from_slice(chunk, chunk.len(), dev)?)?;
                    let ids: Vec<usize> = chunk
                        .iter()
                        .map(|&c| self.assigned[i][c as usize])
                        .collect();
                    let t = self.energy.tables(self.spline, false, &zero)?;
                    let ll = assigned_log_likelihood(&lineage_log_likelihood(&batch, &t)?, &ids)?;
                    let loss = (ll.neg()? / (chunk.len() as f64 * cfg.depth as f64))?;
                    let prior = (self.energy.roughness()? * (cfg.smooth / pass_reads))?;
                    opt.backward_step(&(&loss + prior)?)?;
                    nll += f64::from(loss.to_scalar::<f32>()?) * chunk.len() as f64;
                    reads += chunk.len() as f64;
                }
            }
            if epoch % 10 == 0 || epoch + 1 == epochs {
                let used: Vec<String> = self
                    .counts
                    .iter()
                    .enumerate()
                    .filter(|(_, &n)| n > 0.0)
                    .map(|(j, n)| format!("L{}:{:.0}", j + 1, n))
                    .collect();
                info!(
                    "td-rbm {label} epoch {epoch}: negative log-likelihood per read {:.5}; lineages (cells) {}",
                    nll / reads.max(1.0),
                    used.join(" ")
                );
            }
        }
        bar.finish_and_clear();
        Ok(())
    }
}
