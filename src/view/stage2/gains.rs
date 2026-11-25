//! Stage-2 gains read out where the viewer needs them: at each cell, and
//! at the mean state of a donor's cells in a group, for ranking genes.
//!
//! A deviation is linear in the state coordinates, so a donor's mean gain
//! over a set of its cells is its gain at their mean coordinates; the
//! jackknife z of that mean is computed the same way.

use super::tables::{Gain, Stage2};
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
    rows: &[Option<usize>],
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

/// The mean state coordinates of each donor's cells in each group, where a
/// donor has at least [`MIN_CELLS_IN_GROUP`] there: `(donor, group, mean)`.
/// `group_of`: each cell's group (`None`: in none), beside its stage-2 row.
pub(crate) fn donor_group_states(
    s: &Stage2,
    rows: &[Option<usize>],
    group_of: &[Option<u32>],
) -> Vec<(usize, u32, Vec<f32>)> {
    let width = s.width();
    let mut sums: std::collections::BTreeMap<(usize, u32), (usize, Vec<f64>)> =
        std::collections::BTreeMap::new();
    for (r, g) in rows.iter().zip(group_of) {
        let (Some(c), Some(g)) = (r, g) else {
            continue;
        };
        let entry = sums
            .entry((s.donor_of(*c), *g))
            .or_insert_with(|| (0, vec![0.0; width]));
        entry.0 += 1;
        for (acc, &x) in entry.1.iter_mut().zip(s.coordinates_of(*c)) {
            *acc += f64::from(x);
        }
    }
    sums.into_iter()
        .filter(|(_, (n, _))| *n >= MIN_CELLS_IN_GROUP)
        .map(|((d, g), (n, sum))| (d, g, sum.iter().map(|x| (x / n as f64) as f32).collect()))
        .collect()
}

/// Genes by their largest `|z|` over the donor-group states (`donor`'s
/// only, when given), the largest first, the first `top`. Genes with no
/// supported departure anywhere (every z 0) are left out.
pub(crate) fn rank_genes(
    s: &Stage2,
    states: &[(usize, u32, Vec<f32>)],
    donor: Option<usize>,
    top: usize,
) -> Vec<(Box<str>, f32)> {
    let states: Vec<&(usize, u32, Vec<f32>)> = states
        .iter()
        .filter(|(d, ..)| donor.is_none_or(|want| *d == want))
        .collect();
    let mut ranked: Vec<(usize, f32)> = (0..s.genes.len())
        .into_par_iter()
        .filter_map(|g| {
            let best = states
                .iter()
                .map(|(d, _, x)| s.gain(g, *d, x).z.abs())
                .filter(|z| !z.is_nan())
                .fold(0f32, f32::max);
            (best > 0.0).then_some((g, best))
        })
        .collect();
    ranked.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
    ranked.truncate(top);
    ranked
        .into_iter()
        .map(|(g, z)| (s.genes[g].clone(), z))
        .collect()
}
