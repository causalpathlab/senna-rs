//! A lineage's direction: both orientations climbed and compared.
//!
//! The model starts every lineage at steady state at `t = 0`, so the start
//! is part of the fit and the two orientations of a lineage are two models.
//! Each is climbed by Adam from the start ([`super::init::initial_params`])
//! with that lineage turned round or not, the others as the start has
//! them; the evidence for the direction is the better objective minus the
//! worse. With no dynamics in the reads the two fit alike and the evidence
//! is small: the direction is undetermined.

use super::fit::Fit;
use super::init::{initial_params, Climb};
use super::kernel::GridPrior;
use super::likelihood::Reads;
use legume_numeric::candle::candle_core::{Device, Error, Result};

/// One orientation's fit and its objective.
pub struct Oriented {
    pub fit: Fit,
    pub objective: f32,
}

/// A lineage climbed as the start orders it and turned round.
pub struct Orientations {
    pub forward: Oriented,
    pub reversed: Oriented,
}

impl Orientations {
    /// Whether the turned-round orientation fits better.
    pub fn reversed_wins(&self) -> bool {
        self.reversed.objective > self.forward.objective
    }

    /// The better orientation's fit.
    pub fn better(&self) -> &Fit {
        if self.reversed_wins() {
            &self.reversed.fit
        } else {
            &self.forward.fit
        }
    }

    /// The better objective minus the worse.
    pub fn evidence(&self) -> f64 {
        (f64::from(self.forward.objective) - f64::from(self.reversed.objective)).abs()
    }
}

/// Lineage `lineage` climbed both ways from the start, every other lineage
/// as the start orders it.
pub fn orientations(
    reads: &Reads,
    opts: &Climb,
    lineage: usize,
    dev: &Device,
) -> Result<Orientations> {
    let prior =
        GridPrior::new(opts.points, 1.0, opts.kappa).map_err(|e| Error::Msg(e.to_string()))?;
    let run = |turned: bool| -> Result<Oriented> {
        let mut reversed = vec![false; opts.lineages];
        reversed[lineage] = turned;
        let fit = Fit::from_params(&initial_params(reads, opts, &reversed), &prior, dev)?;
        let objective = fit.adam(reads, opts.steps, opts.lr)?;
        Ok(Oriented { fit, objective })
    };
    Ok(Orientations {
        forward: run(false)?,
        reversed: run(true)?,
    })
}
