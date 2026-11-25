//! Stage-2 gains read out where the viewer needs them: at each cell, and
//! at the mean state of a donor's cells in a group, for ranking genes.
//!
//! A deviation is linear in the state coordinates, so a donor's mean gain
//! over a set of its cells is its gain at their mean coordinates; the
//! jackknife z of that mean is computed the same way.

use super::tables::{Gain, Stage2};
use crate::view::data::NONE;
use rayon::prelude::*;

/// A donor needs this many of its cells in a group for the group's gain to
/// be ranked: fewer is a state the donor barely has.
pub(crate) const MIN_CELLS_IN_GROUP: usize = 10;

/// `|z|` at which a gain is drawn in full; below it, it fades toward muted.
pub(crate) const Z_FULL: f32 = 2.0;

/// How much of a gain's colour to draw at jackknife `z`: all of it from
/// `|z| = Z_FULL` up, fading quadratically to nothing at 0. Without a
/// jackknife (`z` NaN) nothing fades.
#[must_use]
pub(crate) fn weight(z: f32) -> f32 {
    if z.is_nan() {
        1.0
    } else {
        (z.abs() / Z_FULL).min(1.0).powi(2)
    }
}

/// Each cell's gain for `gene` (`rows`: each cell's row in stage 2, `None`
/// when it is not there): the deviation of `donor`, or of the cell's own
/// donor when `None`, at the cell's state.
pub(crate) fn cell_gains(
    s: &Stage2,
    gene: usize,
    donor: Option<usize>,
    rows: &[Option<u32>],
) -> Vec<Option<Gain>> {
    rows.par_iter()
        .map(|r| {
            r.map(|c| {
                let d = donor.unwrap_or_else(|| s.donor_of(c));
                s.gain(gene, d, s.coordinates_of(c))
            })
        })
        .collect()
}

/// A donor's cells in one group: their mean state coordinates.
pub(crate) struct DonorState {
    pub donor: usize,
    pub group: u32,
    /// The donor's cells in the group.
    pub cells: usize,
    pub mean: Vec<f32>,
}

/// The mean state coordinates of each donor's cells in each group, where a
/// donor has at least [`MIN_CELLS_IN_GROUP`] there. `groups`: each cell's
/// group in `0..n_groups` ([`NONE`]: in none), beside its stage-2 row in
/// `rows`; `None` puts every cell in one group, 0.
pub(crate) fn donor_group_states(
    s: &Stage2,
    rows: &[Option<u32>],
    groups: Option<&[u32]>,
    n_groups: usize,
) -> Vec<DonorState> {
    let (width, n_groups) = (s.width(), n_groups.max(1));
    // Dense (donor, group) sums: a donor and group count is small.
    let slots = s.donors.len() * n_groups;
    let mut counts = vec![0usize; slots];
    let mut sums = vec![0f64; slots * width];
    for (i, r) in rows.iter().enumerate() {
        let g = groups.map_or(0, |g| g[i]);
        let (Some(c), true) = (r, g != NONE) else {
            continue;
        };
        let slot = s.donor_of(*c) * n_groups + g as usize;
        counts[slot] += 1;
        for (acc, &x) in sums[slot * width..(slot + 1) * width]
            .iter_mut()
            .zip(s.coordinates_of(*c))
        {
            *acc += f64::from(x);
        }
    }
    (0..slots)
        .filter(|&slot| counts[slot] >= MIN_CELLS_IN_GROUP)
        .map(|slot| DonorState {
            donor: slot / n_groups,
            group: (slot % n_groups) as u32,
            cells: counts[slot],
            mean: sums[slot * width..(slot + 1) * width]
                .iter()
                .map(|x| (x / counts[slot] as f64) as f32)
                .collect(),
        })
        .collect()
}

/// A ranked gene and where it departs most: its score, and its gain at
/// that donor-group state.
pub(crate) struct Ranked {
    pub gene: usize,
    /// `|z|`, or `|log gain|` without a jackknife.
    pub score: f32,
    pub donor: usize,
    pub group: u32,
    pub gain: Gain,
}

/// Genes by their largest `|z|` over the donor-group states (`donor`'s
/// only, when given), the largest first, the first `top`; by their largest
/// `|log gain|` when there is no jackknife (one batch). Genes with no
/// departure anywhere (every gain 0) are left out.
pub(crate) fn rank_genes(
    s: &Stage2,
    states: &[DonorState],
    donor: Option<usize>,
    top: usize,
) -> Vec<Ranked> {
    let states: Vec<&DonorState> = states
        .iter()
        .filter(|st| donor.is_none_or(|want| st.donor == want))
        .collect();
    let jackknife = s.has_jackknife();
    let score = |gain: &Gain| {
        let v = if jackknife {
            gain.z.abs()
        } else {
            gain.log_gain.abs()
        };
        // A z without a spread (all refits alike) is no evidence either way.
        if v.is_finite() {
            v
        } else {
            0.0
        }
    };
    let mut ranked: Vec<Ranked> = (0..s.genes.len())
        .into_par_iter()
        .filter_map(|g| {
            let (st, gain, best) = states
                .iter()
                .map(|st| {
                    let gain = s.gain(g, st.donor, &st.mean);
                    (st, gain, score(&gain))
                })
                .max_by(|a, b| a.2.total_cmp(&b.2))?;
            (best > 0.0).then_some(Ranked {
                gene: g,
                score: best,
                donor: st.donor,
                group: st.group,
                gain,
            })
        })
        .collect();
    ranked.sort_by(|a, b| b.score.total_cmp(&a.score).then(a.gene.cmp(&b.gene)));
    ranked.truncate(top);
    ranked
}
