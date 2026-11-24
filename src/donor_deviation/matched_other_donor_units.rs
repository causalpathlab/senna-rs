//! The contrast's negatives and the coordinates its deviations live in.
//!
//! * Matched other-donor units: for every unit, the units of OTHER donors
//!   nearest to it in the stage-1 embedding. Stage 1 places units in the same
//!   shared state together, so these are "the same state, elsewhere": what
//!   the shared state explains cancels between a unit and its matches.
//! * State coordinates: each unit's position in the stage-1 embedding,
//!   reduced to its leading principal axes and scaled to unit variance, with
//!   an intercept first. A donor's deviation of a gene is a linear function
//!   of these, so it can differ between states.

use legume_numeric::matrix::knn_match::{ColumnDict, SearchScratch};
use rayon::prelude::*;
use senna::embed_common::*;

/// Every unit's matched other-donor units, nearest first, and their
/// Euclidean distances in the stage-1 embedding; both flattened row-major,
/// `matched_per_unit` entries per unit.
pub struct MatchedOtherDonorUnits {
    pub matched_unit_of_unit: Vec<u32>,
    pub distance_to_matched_unit: Vec<f32>,
    pub matched_per_unit: usize,
}

impl MatchedOtherDonorUnits {
    /// `unit`'s matched other-donor units, nearest first.
    pub fn matched_of(&self, unit: usize) -> &[u32] {
        let k = self.matched_per_unit;
        &self.matched_unit_of_unit[unit * k..(unit + 1) * k]
    }

    /// Members of the contrast set anchored at `anchor`: the anchor, then its
    /// matched units (the layout both fit paths use).
    pub fn members_of_set(&self, anchor: u32) -> impl Iterator<Item = u32> + '_ {
        std::iter::once(anchor).chain(self.matched_of(anchor as usize).iter().copied())
    }

    pub fn members_per_set(&self) -> usize {
        self.matched_per_unit + 1
    }

    /// The units whose matched set spans at least `min_other_donors` donors
    /// (other than their own): the anchors the contrast may use. A set drawn
    /// from one other donor compares two donors, not a donor with the shared
    /// state, so its anchor is left out (it is still read out, and still a
    /// match of others).
    pub fn anchors_spanning_other_donors(
        &self,
        donor_index_of_unit: &[usize],
        min_other_donors: usize,
    ) -> Vec<u32> {
        let mut donors: Vec<usize> = Vec::with_capacity(self.matched_per_unit);
        (0..donor_index_of_unit.len())
            .filter(|&unit| {
                donors.clear();
                donors.extend(
                    self.matched_of(unit)
                        .iter()
                        .map(|&m| donor_index_of_unit[m as usize]),
                );
                donors.sort_unstable();
                donors.dedup();
                donors.len() >= min_other_donors
            })
            .map(|unit| unit as u32)
            .collect()
    }

    /// For each of the `number_of_units` units, the units that list it among
    /// their matches (the reverse lookup).
    pub fn units_matching_each_unit(&self, number_of_units: usize) -> Vec<Vec<u32>> {
        let mut matching = vec![Vec::new(); number_of_units];
        for (slot, &matched) in self.matched_unit_of_unit.iter().enumerate() {
            matching[matched as usize].push((slot / self.matched_per_unit) as u32);
        }
        matching
    }
}

/// For every unit, its `matched_per_unit` nearest units of other
/// donors in the stage-1 embedding (`[units × dims]`), nearest first.
pub fn find_matched_other_donor_units(
    stage1_unit_embedding: &Mat,
    donor_index_of_unit: &[usize],
    number_of_donors: usize,
    matched_per_unit: usize,
) -> anyhow::Result<MatchedOtherDonorUnits> {
    let embedding_dims = stage1_unit_embedding.ncols();
    // one search index per donor, over that donor's units only
    let units_in_donor: Vec<usize> = (0..number_of_donors)
        .map(|donor| donor_index_of_unit.iter().filter(|&&d| d == donor).count())
        .collect();
    let index_of_donor: Vec<ColumnDict<u32>> = (0..number_of_donors)
        .map(|donor| {
            let units_of_donor: Vec<usize> = (0..donor_index_of_unit.len())
                .filter(|&unit| donor_index_of_unit[unit] == donor)
                .collect();
            ColumnDict::<u32>::from_dmatrix(
                Mat::from_fn(embedding_dims, units_of_donor.len(), |dim, j| {
                    stage1_unit_embedding[(units_of_donor[j], dim)]
                }),
                units_of_donor.iter().map(|&p| p as u32).collect(),
            )
        })
        .collect();
    // per unit, (distance, matched unit), nearest first
    let matched_of_unit: Vec<Vec<(f32, u32)>> = (0..donor_index_of_unit.len())
        .into_par_iter()
        .map_init(SearchScratch::default, |scratch, unit| {
            let query: Vec<f32> = stage1_unit_embedding.row(unit).iter().copied().collect();
            // the nearest of each other donor, then the nearest of them all
            let mut candidates: Vec<(f32, u32)> = Vec::new();
            for (donor, index) in index_of_donor.iter().enumerate() {
                if donor == donor_index_of_unit[unit] || units_in_donor[donor] == 0 {
                    continue;
                }
                let (ids, distances) = index
                    .search_by_query_data_reuse(&query, matched_per_unit, scratch)
                    .map_err(|e| anyhow::anyhow!("matched units of unit {unit}: {e}"))?;
                candidates.extend(distances.into_iter().zip(ids));
            }
            candidates.sort_by(|a, b| a.0.total_cmp(&b.0));
            anyhow::ensure!(
                candidates.len() >= matched_per_unit,
                "unit {unit}: only {} units in the other donors, fewer than \
                 --matched-samples {matched_per_unit}",
                candidates.len()
            );
            candidates.truncate(matched_per_unit);
            Ok(candidates)
        })
        .collect::<anyhow::Result<_>>()?;
    let (distance_to_matched_unit, matched_unit_of_unit) =
        matched_of_unit.into_iter().flatten().unzip();
    Ok(MatchedOtherDonorUnits {
        matched_unit_of_unit,
        distance_to_matched_unit,
        matched_per_unit,
    })
}

/// `[units × (1 + state_axes)]`, row-major in a flat vector, and its width
/// (`1 + state_axes`, the axes capped at the embedding's dims): an intercept
/// column of ones, then the unit's scores on the stage-1 embedding's
/// `state_axes` leading principal axes, each scaled to unit variance.
pub fn state_coordinates_of_units(
    stage1_unit_embedding: &Mat,
    state_axes: usize,
) -> (Vec<f32>, usize) {
    let number_of_units = stage1_unit_embedding.nrows();
    let embedding_dims = stage1_unit_embedding.ncols();
    let state_axes = state_axes.min(embedding_dims);
    let width = 1 + state_axes;
    let mut centred = stage1_unit_embedding.clone();
    centred.centre_columns_inplace();
    let degrees_of_freedom = (number_of_units.max(2) - 1) as f32;
    let covariance = centred.tr_mul(&centred) / degrees_of_freedom;
    let eigen = covariance.symmetric_eigen();
    let mut axes_by_variance: Vec<usize> = (0..embedding_dims).collect();
    axes_by_variance.sort_by(|&a, &b| eigen.eigenvalues[b].total_cmp(&eigen.eigenvalues[a]));
    // every leading axis's scores in one product, [units × state_axes]
    let leading_axes = eigen
        .eigenvectors
        .select_columns(axes_by_variance.iter().take(state_axes));
    let scores = &centred * leading_axes;
    let mut coordinates = vec![0f32; number_of_units * width];
    for unit in 0..number_of_units {
        coordinates[unit * width] = 1.0;
    }
    for (axis_rank, axis_scores) in scores.column_iter().enumerate() {
        let sd = (axis_scores.norm_squared() / degrees_of_freedom)
            .sqrt()
            .max(1e-12);
        for (unit, score) in axis_scores.iter().enumerate() {
            coordinates[unit * width + 1 + axis_rank] = score / sd;
        }
    }
    (coordinates, width)
}
