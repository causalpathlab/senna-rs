//! Step 4: the states, by persistent contrastive divergence. The positive
//! phase is the rows' free energy, the negative phase that of persistent
//! Gibbs chains ([`Chains`]); every parameter is fitted with the states.

use super::{adam, pass_reads, Level, TrainConfig};
use crate::td_rbm::curve::Spline;
use crate::td_rbm::model::Energy;
use crate::td_rbm::rbm::{free_energy, Chains, Reads};
use crate::td_rbm::start::resample;
use legume_numeric::candle::candle_core::{Device, Tensor};
use legume_numeric::candle::candle_nn::Optimizer;
use log::info;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;

/// `cfg.epochs − cfg.time_epochs` epochs of PCD on every level's resampled
/// pseudobulks, the lineages weighted by `log_weight`.
#[allow(clippy::too_many_arguments)]
pub(super) fn fit_states(
    energy: &Energy,
    spline: &Spline,
    levels: &[Level],
    f: usize,
    cfg: &TrainConfig,
    dev: &Device,
    log_weight: &[f32],
    rng: &mut StdRng,
) -> anyhow::Result<()> {
    let finest = levels.len() - 1;
    // Everything, the states with it: with the states taking up what lies
    // off every trajectory, the curves and rates refit cleaner (training
    // the states alone leaves the velocity markedly worse in simulation).
    let mut vars = energy.vars();
    vars.push(energy.load_var());
    let mut opt = adam(vars, cfg.lr)?;
    let pass_reads = pass_reads(levels, cfg.depth);
    let p = levels[finest].cells.len();
    let drawn = resample(&levels[finest].x, f, cfg.depth, rng);
    let x0: Vec<f32> = (0..cfg.chains)
        .flat_map(|j| drawn[(j % p) * f..(j % p + 1) * f].to_vec())
        .collect();
    let mut chains = Chains::new(x0, f, cfg.seed.wrapping_add(1));
    for epoch in cfg.time_epochs..cfg.epochs {
        let (mut gap, mut steps) = (0f64, 0usize);
        for lv in levels {
            let rows = lv.cells.len();
            let all = Reads::from_features(Tensor::from_vec(
                resample(&lv.x, f, cfg.depth, rng),
                (rows, f),
                dev,
            )?)?;
            let mut order: Vec<u32> = (0..rows as u32).collect();
            order.shuffle(rng);
            for chunk in order.chunks(cfg.batch) {
                let batch = all.select(&Tensor::from_slice(chunk, chunk.len(), dev)?)?;
                let t = energy.tables(spline, true, log_weight)?;
                let fantasy = chains.reads(dev)?;
                let loss = ((free_energy(&batch, &t)?.mean_all()?
                    - free_energy(&fantasy, &t)?.mean_all()?)?
                    / cfg.depth as f64)?;
                let prior = (energy.roughness()? * (cfg.smooth / pass_reads))?;
                opt.backward_step(&(&loss + prior)?)?;
                gap += f64::from(loss.to_scalar::<f32>()?);
                steps += 1;
                let after = energy.tables(spline, true, log_weight)?.detach();
                for _ in 0..cfg.sweeps {
                    chains.sweep(&after)?;
                }
            }
        }
        if epoch % 10 == 0 || epoch + 1 == cfg.epochs {
            info!(
                "td-rbm epoch {epoch} (states): free-energy gap per read {:.5}",
                gap / steps.max(1) as f64
            );
        }
    }
    Ok(())
}
