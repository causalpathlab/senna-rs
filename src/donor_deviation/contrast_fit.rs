//! The contrast, fitted one gene at a time.
//!
//! For a gene, every anchor unit opens a set: the anchor and its matched
//! other-donor units. The positive is drawn among the set's members in
//! proportion to the gene's activity on them (pooled counts divided by the
//! same-section factor). Every member is scored by ITS OWN donor's deviation
//! of the gene at its state coordinates:
//!
//! ```text
//! score(m)  = deviation[gene, donor(m)] · state_coordinates(m)
//! loss      = Σ_sets Σ_{m in set} activity(m) · (logsumexp_{set} score − score(m))
//!             / Σ_sets Σ_{m in set} activity(m)
//!           + group_lasso_penalty · Σ_donors ‖deviation[gene, donor]‖
//! ```
//!
//! Where every donor's units in a state carry the gene alike, the members of
//! a set are drawn equally and the deviations stay at 0. Where one donor's
//! units carry more of it than their matches, that donor's deviation grows
//! in the direction of those units' state. Scoring every member by its own
//! donor, all donors fitted together, lets the penalty keep the majority at
//! 0, so one deviating donor does not push the others'.
//!
//! The problem is convex and separate for each gene; it is solved exactly
//! (every member of every set, no sampled negatives) by FISTA with the
//! group-lasso proximal step and a gradient-based momentum restart.

use super::matched_other_donor_units::MatchedOtherDonorUnits;

/// The members of every set with any activity of one gene.
pub struct ContrastSetsOfGene {
    /// `[sets × members_per_set]`: each set's anchor unit, then its matched
    /// other-donor units.
    pub member_unit: Vec<u32>,
    /// The gene's activity on each member, aligned with `member_unit`.
    pub member_activity: Vec<f32>,
    pub members_per_set: usize,
}

impl ContrastSetsOfGene {
    pub fn number_of_sets(&self) -> usize {
        self.member_unit.len() / self.members_per_set.max(1)
    }
}

/// The sets of `anchor_units` in which the gene has any activity
/// (`activity_of_unit`, dense over all units); a set with none adds nothing
/// to the loss and is left out.
pub fn contrast_sets_of_gene(
    activity_of_unit: &[f32],
    anchor_units: &[u32],
    matched: &MatchedOtherDonorUnits,
) -> ContrastSetsOfGene {
    let members_per_set = matched.members_per_set();
    let mut member_unit = Vec::new();
    let mut member_activity = Vec::new();
    for &anchor in anchor_units {
        let set_total: f32 = matched
            .members_of_set(anchor)
            .map(|m| activity_of_unit[m as usize])
            .sum();
        if set_total <= 0.0 {
            continue;
        }
        for m in matched.members_of_set(anchor) {
            member_unit.push(m);
            member_activity.push(activity_of_unit[m as usize]);
        }
    }
    ContrastSetsOfGene {
        member_unit,
        member_activity,
        members_per_set,
    }
}

/// What the deviations are scored on: every unit's state coordinates
/// (row-major, `coordinates_per_unit` wide, intercept first) and its donor.
pub struct UnitStates<'a> {
    pub state_coordinates: &'a [f32],
    pub coordinates_per_unit: usize,
    pub donor_index_of_unit: &'a [usize],
    pub number_of_donors: usize,
}

impl UnitStates<'_> {
    fn coordinates_of(&self, unit: u32) -> &[f32] {
        let start = unit as usize * self.coordinates_per_unit;
        &self.state_coordinates[start..start + self.coordinates_per_unit]
    }
}

/// One gene's deviation: `[donors × coordinates_per_unit]`, row-major
/// (donor `d`'s coefficients at `d * coordinates_per_unit ..`).
pub type DeviationOfGene = Vec<f32>;

/// A section whose units the loss leaves out (for the section jackknife):
/// a set anchored in it is dropped, and its units drop out of every other
/// set (neither drawn as the positive nor in the softmax).
pub struct LeftOutSection<'a> {
    pub section_index_of_unit: &'a [usize],
    pub left_out_section: usize,
}

impl LeftOutSection<'_> {
    fn leaves_out(&self, unit: u32) -> bool {
        self.section_index_of_unit[unit as usize] == self.left_out_section
    }
}

/// The smooth part of the loss at `deviation`, optionally without one
/// section's units, normalized by the activity it keeps; its gradient goes
/// into `gradient` (same layout).
pub fn contrast_loss_and_gradient(
    deviation: &[f32],
    sets: &ContrastSetsOfGene,
    states: &UnitStates,
    left_out: Option<&LeftOutSection>,
    gradient: &mut [f32],
) -> f32 {
    let width = states.coordinates_per_unit;
    gradient.iter_mut().for_each(|g| *g = 0.0);
    let kept = |unit: u32| left_out.is_none_or(|l| !l.leaves_out(unit));
    let mut total_activity = 0f64;
    let mut loss = 0f64;
    let mut member_scores = vec![0f32; sets.members_per_set];
    for set in 0..sets.number_of_sets() {
        let range = set * sets.members_per_set..(set + 1) * sets.members_per_set;
        let units = &sets.member_unit[range.clone()];
        let activity = &sets.member_activity[range];
        // the set belongs to its anchor, the first member
        if !kept(units[0]) {
            continue;
        }
        for (score, &unit) in member_scores.iter_mut().zip(units) {
            let donor = states.donor_index_of_unit[unit as usize];
            *score = if kept(unit) {
                dot(
                    &deviation[donor * width..(donor + 1) * width],
                    states.coordinates_of(unit),
                )
            } else {
                f32::NEG_INFINITY
            };
        }
        let set_activity: f32 = units
            .iter()
            .zip(activity)
            .filter(|(&p, _)| kept(p))
            .map(|(_, &a)| a)
            .sum();
        if set_activity <= 0.0 {
            continue;
        }
        total_activity += set_activity as f64;
        let max_score = member_scores
            .iter()
            .copied()
            .fold(f32::NEG_INFINITY, f32::max);
        let partition: f32 = member_scores.iter().map(|s| (s - max_score).exp()).sum();
        let log_partition = max_score + partition.ln();
        for ((&score, &unit), &a) in member_scores.iter().zip(units).zip(activity) {
            if !kept(unit) {
                continue;
            }
            loss += (a * (log_partition - score)) as f64;
            // d loss / d score(m) = set activity · P(m) − activity(m)
            let weight = set_activity * (score - log_partition).exp() - a;
            let donor = states.donor_index_of_unit[unit as usize];
            for (g, x) in gradient[donor * width..(donor + 1) * width]
                .iter_mut()
                .zip(states.coordinates_of(unit))
            {
                *g += weight * x;
            }
        }
    }
    if total_activity <= 0.0 {
        return 0.0;
    }
    gradient
        .iter_mut()
        .for_each(|g| *g = (*g as f64 / total_activity) as f32);
    (loss / total_activity) as f32
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// The group-lasso proximal step: each donor's coefficient row shrunk in
/// norm by `threshold`, or set to 0 when its norm is below it.
pub fn shrink_each_donor_row_by_group_lasso(deviation: &mut [f32], width: usize, threshold: f32) {
    for row in deviation.chunks_mut(width) {
        let norm = row.iter().map(|x| x * x).sum::<f32>().sqrt();
        let keep = if norm > threshold {
            1.0 - threshold / norm
        } else {
            0.0
        };
        row.iter_mut().for_each(|x| *x *= keep);
    }
}

/// One fit's result: the deviation, and whether it reached `max_iterations`
/// before its gradient mapping fell below the tolerance (the run warns when
/// any fit did).
pub struct FitOfGene {
    pub deviation: DeviationOfGene,
    pub reached_iteration_cap: bool,
}

/// How the per-gene fit runs.
#[derive(Clone, Copy, Debug, serde::Serialize)]
pub struct ContrastFitSettings {
    pub group_lasso_penalty: f32,
    pub max_iterations: usize,
    /// Stop when the gradient mapping, (y − prox(y − step ∇f(y))) / step,
    /// is below this in norm: the proximal gradient is then near 0, so the
    /// fit is at its optimum whatever it started from (a step-size rule
    /// would stop a warm-started refit almost at once).
    pub gradient_mapping_tolerance: f32,
}

/// Fit one gene's deviation by FISTA, from 0 or from `start`, optionally
/// without one section's units (a section jackknife replicate, started from
/// the full fit: `section_jackknife`). The step is 1 / L with L the largest
/// squared norm of a member's state coordinates, a bound on the smooth
/// loss's curvature (a softmax's covariance of the members' coordinates).
pub fn fit_deviation_of_gene(
    sets: &ContrastSetsOfGene,
    states: &UnitStates,
    start: Option<&[f32]>,
    left_out: Option<&LeftOutSection>,
    settings: &ContrastFitSettings,
) -> FitOfGene {
    let size = states.number_of_donors * states.coordinates_per_unit;
    let mut deviation = match start {
        Some(start) => start.to_vec(),
        None => vec![0f32; size],
    };
    if sets.number_of_sets() == 0 {
        return FitOfGene {
            deviation,
            reached_iteration_cap: false,
        };
    }
    let curvature_bound = sets
        .member_unit
        .iter()
        .map(|&p| states.coordinates_of(p).iter().map(|x| x * x).sum::<f32>())
        .fold(0f32, f32::max)
        .max(1e-6);
    let step = 1.0 / curvature_bound;
    let threshold = step * settings.group_lasso_penalty;

    let mut extrapolated = deviation.clone();
    let mut gradient = vec![0f32; size];
    let mut momentum = 1f32;
    for _ in 0..settings.max_iterations {
        contrast_loss_and_gradient(&extrapolated, sets, states, left_out, &mut gradient);
        let mut next: Vec<f32> = extrapolated
            .iter()
            .zip(&gradient)
            .map(|(y, g)| y - step * g)
            .collect();
        shrink_each_donor_row_by_group_lasso(&mut next, states.coordinates_per_unit, threshold);
        // restart the momentum when the step goes against the gradient
        let against_gradient: f32 = extrapolated
            .iter()
            .zip(&next)
            .zip(&deviation)
            .map(|((y, n), d)| (y - n) * (n - d))
            .sum();
        let blend = if against_gradient > 0.0 {
            momentum = 1.0;
            0.0
        } else {
            let next_momentum = (1.0 + (1.0 + 4.0 * momentum * momentum).sqrt()) / 2.0;
            let blend = (momentum - 1.0) / next_momentum;
            momentum = next_momentum;
            blend
        };
        // the gradient mapping (y − next) / step: 0 only at an optimum
        let gradient_mapping_squared: f32 = extrapolated
            .iter()
            .zip(&next)
            .map(|(y, n)| (y - n) * (y - n))
            .sum::<f32>()
            / (step * step);
        for ((y, n), d) in extrapolated.iter_mut().zip(&next).zip(&deviation) {
            *y = n + blend * (n - d);
        }
        deviation = next;
        if gradient_mapping_squared.sqrt() <= settings.gradient_mapping_tolerance {
            return FitOfGene {
                deviation,
                reached_iteration_cap: false,
            };
        }
    }
    FitOfGene {
        deviation,
        reached_iteration_cap: true,
    }
}
