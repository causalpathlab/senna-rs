//! The per-gene contrast fit (`contrast_fit`), for a block of genes at once
//! on a candle device (CPU, CUDA or Metal).
//!
//! Genes share the units, the contrast sets and the state coordinates; they
//! differ only in their activities. So everything shared is built once:
//!
//! * `member_block_coordinates` (Z̃): `[sets × members, donors × width]`,
//!   each member's state coordinates placed in its own donor's block, so a
//!   member's score under a deviation Δ (`[donors × width]`, flattened) is
//!   `Δ · z̃`, its own donor's row at its coordinates;
//! * `member_kept_of_fit`: `[fits × sets × members]`, 1 where the member
//!   counts and 0 where it does not. Fit 0 is the full fit; fit `s + 1`
//!   leaves section `s` out (its members drop out of every set, and a set
//!   anchored in it is dropped whole), the section jackknife's refits.
//!
//! A batch item is one (gene, fit). Every anchor set is kept for every
//! gene, with zero activity where the gene has none, so shapes stay fixed;
//! such a set adds nothing to the loss or its gradient. One FISTA iteration
//! for all items:
//!
//! ```text
//! scores    = Δ Z̃ᵀ, with members left out at −1e30   [items × sets × members]
//! P         = softmax of the scores within each set
//! weights   = set activity · P − activity
//! gradient  = weights Z̃ / total activity              [items × donors·width]
//! ```
//!
//! then the group-lasso proximal step on each donor row, and momentum with
//! a gradient-based restart, per item, as in `fit_deviation_of_gene`. An item
//! stops at its own tolerance and is frozen until the block finishes.
//!
//! A block runs in two phases: the genes' full fits, then their refits
//! without each section, each started from its gene's full fit (the
//! jackknife refits are exact; the start only shortens them,
//! `section_jackknife`).
//!
//! Sums on a GPU are not ordered as on a CPU, so results agree with
//! `contrast_fit` to the fit's tolerance, not bit for bit.

use super::contrast_fit::{ContrastFitSettings, UnitStates};
use super::fit_genes::FittedGene;
use super::matched_other_donor_units::MatchedOtherDonorUnits;
use legume_numeric::candle::candle_core::{DType, Device, Tensor, D};

/// The score given to a member that is left out: far below any real score,
/// so its softmax weight is 0, but finite, so a set with every member left
/// out gives no NaN (its activity is 0, so it adds nothing).
const LEFT_OUT_SCORE: f64 = -1e30;

/// FISTA iterations between checks of whether every item has stopped (each
/// check copies a flag per item back from the device).
const ITERATIONS_PER_STOPPING_CHECK: usize = 10;

/// What every gene's fit shares, on the device.
pub struct SharedContrastTensors {
    /// Each set's members: its anchor, then its matched units
    /// (`[sets × members]`, row-major).
    pub member_unit: Vec<u32>,
    pub members_per_set: usize,
    /// Z̃: `[sets × members, donors × width]`.
    member_block_coordinates: Tensor,
    /// Z̃ᵀ, contiguous: `[donors × width, sets × members]`.
    member_block_coordinates_transposed: Tensor,
    /// The squared norm of each member's state coordinates,
    /// `[sets, members]`: the per-gene step size's curvature bound.
    member_squared_norm: Tensor,
    /// `[fits, sets, members]`, 1 for a member that counts in that fit.
    member_kept_of_fit: Tensor,
    pub number_of_fits: usize,
    coordinates_per_unit: usize,
    number_of_donors: usize,
    device: Device,
}

impl SharedContrastTensors {
    /// Build the shared tensors for the sets of `anchor_units` (each with
    /// its matched units) on `device`; fit 0 is the full
    /// fit, fit `s + 1` the one without section `s`.
    pub fn new(
        anchor_units: &[u32],
        matched: &MatchedOtherDonorUnits,
        states: &UnitStates,
        section_index_of_unit: &[usize],
        number_of_sections: usize,
        device: &Device,
    ) -> anyhow::Result<Self> {
        let width = states.coordinates_per_unit;
        let block_width = states.number_of_donors * width;
        let members_per_set = matched.members_per_set();
        let number_of_sets = anchor_units.len();
        let member_unit: Vec<u32> = anchor_units
            .iter()
            .flat_map(|&anchor| matched.members_of_set(anchor))
            .collect();
        let mut block_coordinates = vec![0f32; member_unit.len() * block_width];
        let mut squared_norm = vec![0f32; member_unit.len()];
        for (slot, &unit) in member_unit.iter().enumerate() {
            let p = unit as usize;
            let coordinates = &states.state_coordinates[p * width..(p + 1) * width];
            let donor = states.donor_index_of_unit[p];
            let row = slot * block_width + donor * width;
            block_coordinates[row..row + width].copy_from_slice(coordinates);
            squared_norm[slot] = coordinates.iter().map(|x| x * x).sum();
        }
        let number_of_fits = 1 + number_of_sections;
        let mut kept = vec![1f32; number_of_fits * member_unit.len()];
        for section in 0..number_of_sections {
            let fit =
                &mut kept[(1 + section) * member_unit.len()..(2 + section) * member_unit.len()];
            for (set_slots, set_units) in fit
                .chunks_mut(members_per_set)
                .zip(member_unit.chunks(members_per_set))
            {
                if section_index_of_unit[set_units[0] as usize] == section {
                    set_slots.iter_mut().for_each(|k| *k = 0.0);
                    continue;
                }
                for (k, &unit) in set_slots.iter_mut().zip(set_units) {
                    if section_index_of_unit[unit as usize] == section {
                        *k = 0.0;
                    }
                }
            }
        }
        let member_block_coordinates =
            Tensor::from_vec(block_coordinates, (member_unit.len(), block_width), device)?;
        Ok(Self {
            member_block_coordinates_transposed: member_block_coordinates.t()?.contiguous()?,
            member_block_coordinates,
            member_squared_norm: Tensor::from_vec(
                squared_norm,
                (number_of_sets, members_per_set),
                device,
            )?,
            member_kept_of_fit: Tensor::from_vec(
                kept,
                (number_of_fits, number_of_sets, members_per_set),
                device,
            )?,
            member_unit,
            members_per_set,
            number_of_fits,
            coordinates_per_unit: width,
            number_of_donors: states.number_of_donors,
            device: device.clone(),
        })
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn number_of_sets(&self) -> usize {
        self.member_unit.len() / self.members_per_set
    }

    /// Device bytes one batch item takes while it is fitted, as a count of
    /// `[sets × members]` f32 tensors: its activity and score offset, kept
    /// for the whole fit, and an iteration's scores, offset scores,
    /// exponents, probabilities, weights and their products, alive at once
    /// (in pinto, 6 per item ran a 3080 out of memory).
    pub fn bytes_per_item(&self) -> usize {
        const PER_MEMBER_TENSORS: usize = 12;
        PER_MEMBER_TENSORS * self.member_unit.len() * std::mem::size_of::<f32>()
    }
}

/// Fit a block of genes: `member_activity_of_gene[g]` is gene `g`'s activity
/// on every member of every set (`[sets × members]`, row-major, as
/// `shared.member_unit`). Returns, per gene, its full fit and its refits
/// without each section.
pub fn fit_block_of_genes(
    shared: &SharedContrastTensors,
    member_activity_of_gene: &[Vec<f32>],
    settings: &ContrastFitSettings,
) -> anyhow::Result<Vec<FittedGene>> {
    let genes = member_activity_of_gene.len();
    if genes == 0 {
        return Ok(Vec::new());
    }
    let sets = shared.number_of_sets();
    let members = shared.members_per_set;
    let device = &shared.device;
    let activity_of_gene = Tensor::from_vec(
        member_activity_of_gene.concat(),
        (genes, sets, members),
        device,
    )?;
    // per gene, the step's curvature bound: the largest squared norm of a
    // member of a set the gene has any activity in
    let set_has_activity = activity_of_gene
        .sum_keepdim(D::Minus1)?
        .gt(0.0)?
        .to_dtype(DType::F32)?;
    let curvature_bound_of_gene = set_has_activity
        .broadcast_mul(
            &shared
                .member_squared_norm
                .max_keepdim(D::Minus1)?
                .unsqueeze(0)?,
        )?
        .flatten_from(1)?
        .max_keepdim(1)?
        .maximum(1e-6)?;

    // phase 1: each gene's full fit (fit 0), from 0
    let gene_index: Vec<u32> = (0..genes as u32).collect();
    let full_fits = fista_on_items(
        shared,
        &activity_of_gene,
        &curvature_bound_of_gene,
        &gene_index,
        &vec![0u32; genes],
        None,
        settings,
    )?;
    // phase 2: each gene's refit without each section, from its full fit
    let sections = shared.number_of_fits - 1;
    let mut fits = full_fits
        .deviation
        .to_vec2::<f32>()?
        .into_iter()
        .zip(&full_fits.reached_iteration_cap)
        .map(|(deviation, &reached_cap)| FittedGene {
            deviation,
            deviation_without_section: Vec::with_capacity(sections),
            fits_at_iteration_cap: usize::from(reached_cap),
        })
        .collect::<Vec<_>>();
    if sections > 0 {
        let item_gene: Vec<u32> = (0..genes as u32)
            .flat_map(|g| std::iter::repeat_n(g, sections))
            .collect();
        let item_fit: Vec<u32> = (0..genes).flat_map(|_| 1..=sections as u32).collect();
        let starts = gather_items(
            &full_fits.deviation,
            &Tensor::new(item_gene.as_slice(), device)?,
        )?;
        let refits = fista_on_items(
            shared,
            &activity_of_gene,
            &curvature_bound_of_gene,
            &item_gene,
            &item_fit,
            Some(starts),
            settings,
        )?;
        for ((refit, &reached_cap), &g) in refits
            .deviation
            .to_vec2::<f32>()?
            .into_iter()
            .zip(&refits.reached_iteration_cap)
            .zip(&item_gene)
        {
            let fit = &mut fits[g as usize];
            fit.deviation_without_section.push(refit);
            fit.fits_at_iteration_cap += usize::from(reached_cap);
        }
    }
    Ok(fits)
}

/// A batch's fitted items: `[items, donors × width]`, and per item whether
/// it reached `max_iterations` before converging.
struct ItemFits {
    deviation: Tensor,
    reached_iteration_cap: Vec<bool>,
}

/// FISTA on a batch of items, item `i` being gene `item_gene[i]` (a row of
/// `activity_of_gene`) under fit `item_fit[i]` (a row of the keep masks),
/// from `start` (`[items, donors × width]`) or 0.
fn fista_on_items(
    shared: &SharedContrastTensors,
    activity_of_gene: &Tensor,
    curvature_bound_of_gene: &Tensor,
    item_gene: &[u32],
    item_fit: &[u32],
    start: Option<Tensor>,
    settings: &ContrastFitSettings,
) -> anyhow::Result<ItemFits> {
    let device = &shared.device;
    let items = item_gene.len();
    let width = shared.coordinates_per_unit;
    let donors = shared.number_of_donors;
    let block_width = donors * width;
    let sets = shared.number_of_sets();
    let members = shared.members_per_set;
    let item_gene = Tensor::new(item_gene, device)?;
    let item_fit = Tensor::new(item_fit, device)?;

    let kept = gather_items(&shared.member_kept_of_fit, &item_fit)?;
    let activity = gather_items(activity_of_gene, &item_gene)?.mul(&kept)?;
    // 0 for a member that counts, LEFT_OUT_SCORE for one that does not
    let score_offset = kept.affine(-LEFT_OUT_SCORE, LEFT_OUT_SCORE)?;
    let set_activity = activity.sum_keepdim(D::Minus1)?;
    let total_activity = set_activity
        .flatten_from(1)?
        .sum_keepdim(1)?
        .maximum(f32::MIN_POSITIVE as f64)?;
    drop(kept);
    // per item: step 1 / L and the proximal threshold step · λ
    let step = gather_items(curvature_bound_of_gene, &item_gene)?.recip()?;
    let threshold = step.affine(settings.group_lasso_penalty as f64, 0.0)?;

    let mut deviation = match start {
        Some(start) => start,
        None => Tensor::zeros((items, block_width), DType::F32, device)?,
    };
    let mut extrapolated = deviation.clone();
    let mut momentum = Tensor::ones((items, 1), DType::F32, device)?;
    // 1 while the item is still being fitted (an item left without activity
    // has a zero gradient and shrinks to 0, as in `fit_deviation_of_gene`)
    let mut active = Tensor::ones((items, 1), DType::F32, device)?;
    let z = &shared.member_block_coordinates;
    let z_transposed = &shared.member_block_coordinates_transposed;
    for iteration in 0..settings.max_iterations {
        let scores = extrapolated
            .matmul(z_transposed)?
            .reshape((items, sets, members))?
            .add(&score_offset)?;
        let max_score = scores.max_keepdim(D::Minus1)?;
        let exponent = scores.broadcast_sub(&max_score)?.exp()?;
        let probability = exponent.broadcast_div(&exponent.sum_keepdim(D::Minus1)?)?;
        // d loss / d score(m) = set activity · P(m) − activity(m)
        let weights = probability.broadcast_mul(&set_activity)?.sub(&activity)?;
        let gradient = weights
            .reshape((items, sets * members))?
            .matmul(z)?
            .broadcast_div(&total_activity)?;
        let stepped = extrapolated.sub(&gradient.broadcast_mul(&step)?)?;
        let next = shrink_each_donor_row(&stepped, &threshold, donors, width)?;
        // restart the momentum when the step goes against the gradient
        let against_gradient = extrapolated
            .sub(&next)?
            .mul(&next.sub(&deviation)?)?
            .sum_keepdim(1)?;
        let restart = against_gradient.gt(0.0)?;
        let next_momentum = momentum.sqr()?.affine(4.0, 1.0)?.sqrt()?.affine(0.5, 0.5)?;
        let blend = momentum.affine(1.0, -1.0)?.div(&next_momentum)?;
        let zeros = blend.zeros_like()?;
        let blend = restart.where_cond(&zeros, &blend)?;
        momentum = restart.where_cond(&momentum.ones_like()?, &next_momentum)?;
        // stop at the gradient mapping (y − next) / step, 0 only at an optimum:
        // ‖y − next‖² ≤ (tolerance · step)²
        let tolerance_squared = (settings.gradient_mapping_tolerance as f64).powi(2);
        let stopped = extrapolated
            .sub(&next)?
            .sqr()?
            .sum_keepdim(1)?
            .le(&step.sqr()?.affine(tolerance_squared, 0.0)?)?
            .to_dtype(DType::F32)?;
        // frozen items keep their coefficients; active ones take the step
        let next = blend_where_active(&next, &deviation, &active)?;
        // a frozen item's next is its deviation, so its extrapolation is too
        extrapolated = next.add(&next.sub(&deviation)?.broadcast_mul(&blend)?)?;
        deviation = next;
        active = active.mul(&stopped.affine(-1.0, 1.0)?)?;
        if (iteration + 1) % ITERATIONS_PER_STOPPING_CHECK == 0
            && active.sum_all()?.to_scalar::<f32>()? == 0.0
        {
            break;
        }
    }
    // an item still active ran out of iterations before it converged
    let reached_iteration_cap = active
        .flatten_all()?
        .to_vec1::<f32>()?
        .into_iter()
        .map(|a| a > 0.0)
        .collect();
    Ok(ItemFits {
        deviation,
        reached_iteration_cap,
    })
}

/// `value` where `active` (`[items, 1]`) is 1, `otherwise` where it is 0.
fn blend_where_active(
    value: &Tensor,
    otherwise: &Tensor,
    active: &Tensor,
) -> anyhow::Result<Tensor> {
    Ok(otherwise.add(&value.sub(otherwise)?.broadcast_mul(active)?)?)
}

/// The group-lasso proximal step on `[items, donors × width]`: each donor
/// row shrunk in norm by the item's `threshold` (`[items, 1]`), or set to 0
/// when its norm is below it.
fn shrink_each_donor_row(
    deviation: &Tensor,
    threshold: &Tensor,
    donors: usize,
    width: usize,
) -> anyhow::Result<Tensor> {
    let items = deviation.dim(0)?;
    let rows = deviation.reshape((items, donors, width))?;
    let norm = rows.sqr()?.sum_keepdim(D::Minus1)?.sqrt()?.maximum(1e-30)?;
    let keep = norm
        .broadcast_div(&threshold.unsqueeze(2)?)?
        .recip()?
        .affine(-1.0, 1.0)?
        .relu()?;
    Ok(rows
        .broadcast_mul(&keep)?
        .reshape((items, donors * width))?)
}

/// `table[ids]` along the first dimension, for a table of any rank:
/// `fast_index::gather_rows` on the table flattened to `[rows, rest]`.
fn gather_items(table: &Tensor, ids: &Tensor) -> anyhow::Result<Tensor> {
    let mut shape = table.dims().to_vec();
    let flat = table.flatten_from(1)?;
    let gathered = legume_numeric::candle::fast_index::gather_rows(&flat, ids)?;
    shape[0] = ids.dim(0)?;
    Ok(gathered.reshape(shape)?)
}
