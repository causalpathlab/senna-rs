//! A lineage's direction: its order held, the two ways it can run compared.
//!
//! The model starts every lineage at steady state at `t = 0`, so a fit's
//! direction is part of the fit. Comparing separate climbs started each way
//! does not measure it: a climb started turned round can turn back or fold,
//! and climbs to one explanation land in local modes far apart, as far as
//! the direction moves the objective. So the order comes first and is held:
//!
//! 1. [`super::init::climb`] from scratch; every pseudobulk on its lineage
//!    and nearest grid point of its posterior mean time.
//! 2. Those points held, every parameter climbed again from the fit
//!    (`forward`); and with the lineage's points mirrored, `k → n − 1 − k`,
//!    from the fit with its path mirrored (`reversed`). Each climb finished
//!    at a tenth of the rate.
//! 3. The evidence is the forward objective minus the reversed one. Held
//!    points leave no latent mixture to fold and no order to change, so
//!    the two climbs are the same smooth problem but for the direction:
//!    with no dynamics in the reads the evidence is near zero.

use super::fit::Fit;
use super::init::{climb, Climb};
use super::kernel::GridPrior;
use super::likelihood::Reads;
use super::model::Params;
use legume_numeric::candle::candle_core::{Device, Error, Result};

/// Steps of the low-rate finish after a held climb, at a tenth of its rate:
/// the evidence compares objectives, so they should be near their modes.
pub const FINISH_STEPS: usize = 1000;

/// A lineage's direction: the climbed fit and the two held refits.
pub struct Direction {
    /// [`super::init::climb`]'s fit and objective.
    pub fit: Fit,
    pub objective: f32,
    /// Every pseudobulk's held row `ℓ n + k` the way the fit runs.
    pub rows: Vec<u32>,
    /// The held refits the way the fit runs and turned round, and their
    /// objectives.
    pub forward: Fit,
    pub forward_objective: f32,
    pub reversed: Fit,
    pub reversed_objective: f32,
}

/// Evidence per pseudobulk below which a direction is undetermined.
///
/// The evidence is a sum over pseudobulks, so it is scaled by their number.
/// On simulated lineages (100 genes, 300 pseudobulks of ~5000 reads) the
/// held refits leave 0.6–2.4 nats per pseudobulk between the two ways when
/// the reads are at steady state (the climbs' own resolution: nothing in
/// the reads tells the ways apart), and 35–64 with kinetics. 5 sits above
/// the first by twice its largest and below the second by seven times its
/// smallest.
pub const UNDETERMINED_PER_PB: f64 = 5.0;

/// What the evidence says of a lineage's direction.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Call {
    /// The climb's own direction fits better.
    Forward,
    /// The lineage turned round fits better.
    Reversed,
    /// The reads do not tell the two ways apart.
    Undetermined,
    /// A held refit was not finite: no evidence either way.
    Diverged,
}

impl Call {
    pub fn as_str(self) -> &'static str {
        match self {
            Call::Forward => "forward",
            Call::Reversed => "reversed",
            Call::Undetermined => "undetermined",
            Call::Diverged => "diverged",
        }
    }
}

/// The call from `evidence` (forward minus reversed) over `n` pseudobulks
/// ([`UNDETERMINED_PER_PB`] a pseudobulk the threshold).
pub fn call_of(evidence: f64, n: usize) -> Call {
    if !evidence.is_finite() {
        Call::Diverged
    } else if evidence.abs() <= UNDETERMINED_PER_PB * n as f64 {
        Call::Undetermined
    } else if evidence > 0.0 {
        Call::Forward
    } else {
        Call::Reversed
    }
}

impl Direction {
    /// The forward held objective minus the reversed: positive when the
    /// climb's own direction fits better.
    pub fn evidence(&self) -> f64 {
        f64::from(self.forward_objective) - f64::from(self.reversed_objective)
    }

    /// What the evidence calls ([`call_of`]).
    pub fn call(&self) -> Call {
        call_of(self.evidence(), self.rows.len())
    }

    /// Whether the evidence calls a direction.
    pub fn determined(&self) -> bool {
        matches!(self.call(), Call::Forward | Call::Reversed)
    }
}

/// `params` with lineage `lineage`'s path turned round in time.
pub fn mirrored(params: &Params, lineage: usize) -> Params {
    let mut out = params.clone();
    out.path[lineage].reverse();
    out
}

/// Every pseudobulk's row at its most likely lineage and the grid point
/// nearest its posterior mean time there.
fn held_rows(fit: &Fit, reads: &Reads, points: usize) -> Result<Vec<u32>> {
    let (weights, time) = fit.posterior(&fit.reads(reads)?)?;
    let t_len = fit.params()?.t_len;
    Ok(weights
        .iter()
        .zip(&time)
        .map(|(w, &t)| {
            let l = (0..w.len())
                .max_by(|&a, &b| w[a].total_cmp(&w[b]))
                .expect("lineages");
            let k = (t / t_len[l] * (points - 1) as f64)
                .round()
                .clamp(0.0, (points - 1) as f64) as usize;
            (l * points + k) as u32
        })
        .collect())
}

/// Lineage `lineage`'s direction: climbed from scratch, then refitted with
/// its order held both ways round ([`direction_of`]).
pub fn direction(reads: &Reads, opts: &Climb, lineage: usize, dev: &Device) -> Result<Direction> {
    let (fit, objective) = climb(reads, opts, dev)?;
    direction_of(fit, objective, reads, opts, lineage, dev)
}

/// Lineage `lineage`'s direction from a climbed `fit` (its objective
/// `objective`): refitted with its order held both ways round.
pub fn direction_of(
    fit: Fit,
    objective: f32,
    reads: &Reads,
    opts: &Climb,
    lineage: usize,
    dev: &Device,
) -> Result<Direction> {
    let prior =
        GridPrior::new(opts.points, 1.0, opts.kappa).map_err(|e| Error::Msg(e.to_string()))?;
    let n = opts.points;
    let rows = held_rows(&fit, reads, n)?;
    let turned: Vec<u32> = rows
        .iter()
        .map(|&r| {
            let (l, k) = (r as usize / n, r as usize % n);
            if l == lineage {
                (l * n + n - 1 - k) as u32
            } else {
                r
            }
        })
        .collect();
    let params = fit.params()?;
    let share = fit.share_sd()?;
    let held = |start: &Params, rows: &[u32]| -> Result<(Fit, f32)> {
        let refit = Fit::from_params(start, &prior, dev)?.with_share(share)?;
        refit.adam_held(reads, rows, opts.steps, opts.lr)?;
        let objective = refit.adam_held(reads, rows, FINISH_STEPS, opts.lr / 10.0)?;
        Ok((refit, objective))
    };
    let (forward, forward_objective) = held(&params, &rows)?;
    let (reversed, reversed_objective) = held(&mirrored(&params, lineage), &turned)?;
    Ok(Direction {
        fit,
        objective,
        rows,
        forward,
        forward_objective,
        reversed,
        reversed_objective,
    })
}
