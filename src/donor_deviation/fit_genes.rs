//! Fitting every gene, with its section jackknife, in one of two ways:
//!
//! * `fit_genes_on_cpu_threads` (`--device cpu`): one gene per rayon task,
//!   `contrast_fit` and `section_jackknife` on the CPU;
//! * `fit_genes_in_blocks` (`--device cuda` or `metal`): blocks of genes on
//!   a candle device, `batched_contrast_fit`, the block size taken from the
//!   device's free memory.
//!
//! Both read a gene's activity the same way (`write_activity_of_gene`): its
//! pooled counts on each unit, divided by its same-section factor.

use super::batched_contrast_fit::{fit_block_of_genes, SharedContrastTensors};
use super::contrast_fit::{
    contrast_sets_of_gene, fit_deviation_of_gene, ContrastFitSettings, UnitStates,
};
use super::matched_other_donor_units::MatchedOtherDonorUnits;
use super::{same_section_gene_factor, section_jackknife, IndexedSections};
use indicatif::ParallelProgressIterator;
use rayon::prelude::*;

/// One gene's fit: its deviation, and its deviation without each section
/// (in section index order).
pub(crate) struct FittedGene {
    pub deviation: super::contrast_fit::DeviationOfGene,
    pub deviation_without_section: Vec<super::contrast_fit::DeviationOfGene>,
    /// Of its full fit and refits, how many reached `--max-iterations`.
    pub fits_at_iteration_cap: usize,
}

/// What a gene's activity is read with.
pub(super) struct ActivityInputs<'a> {
    pub matched: &'a MatchedOtherDonorUnits,
    /// For every unit, the units that list it among their matches.
    pub units_matching_unit: &'a [Vec<u32>],
    pub sections: &'a IndexedSections,
}

/// One thread's scratch over all units, all 0 / false between genes.
struct GeneScratch {
    activity_of_unit: Vec<f32>,
    unit_has_signal: Vec<bool>,
}

impl ActivityInputs<'_> {
    fn number_of_units(&self) -> usize {
        self.sections.section_index_of_unit.len()
    }

    fn new_scratch(&self) -> GeneScratch {
        GeneScratch {
            activity_of_unit: vec![0f32; self.number_of_units()],
            unit_has_signal: vec![false; self.number_of_units()],
        }
    }

    /// Write a gene's activity into the scratch at its `nonzero_units`: the
    /// pooled count divided by the exp of its section's log factor.
    /// `clear_activity_of_gene` puts the scratch back to 0.
    fn write_activity_of_gene(&self, nonzero_units: &[(u32, f32)], scratch: &mut GeneScratch) {
        for &(unit, count) in nonzero_units {
            scratch.activity_of_unit[unit as usize] = count;
        }
        let units_with_signal = self.units_with_own_or_matched_counts(nonzero_units, scratch);
        let log_factor_of_section = same_section_gene_factor::log_factor_of_section_for_gene(
            &scratch.activity_of_unit,
            &units_with_signal,
            self.matched,
            &self.sections.section_index_of_unit,
            &self.sections.number_of_units_in_section,
        );
        for &(unit, _) in nonzero_units {
            let unit = unit as usize;
            scratch.activity_of_unit[unit] /=
                log_factor_of_section[self.sections.section_index_of_unit[unit]].exp();
        }
    }

    /// The units whose log ratio to their matches is not exactly 0 for a
    /// gene: those with a count of their own, and those with one among their
    /// matches; each once (marked in the scratch, then unmarked).
    fn units_with_own_or_matched_counts(
        &self,
        nonzero_units: &[(u32, f32)],
        scratch: &mut GeneScratch,
    ) -> Vec<u32> {
        let mut units = Vec::with_capacity(nonzero_units.len());
        for &(unit, _) in nonzero_units {
            let own_and_matching = std::iter::once(unit)
                .chain(self.units_matching_unit[unit as usize].iter().copied());
            for p in own_and_matching {
                if !scratch.unit_has_signal[p as usize] {
                    scratch.unit_has_signal[p as usize] = true;
                    units.push(p);
                }
            }
        }
        for &p in &units {
            scratch.unit_has_signal[p as usize] = false;
        }
        units
    }
}

fn clear_activity_of_gene(nonzero_units: &[(u32, f32)], scratch: &mut GeneScratch) {
    for &(unit, _) in nonzero_units {
        scratch.activity_of_unit[unit as usize] = 0.0;
    }
}

/// `--device cpu`: each gene a rayon task, its full fit then its refits
/// without each section, started from it.
pub(super) fn fit_genes_on_cpu_threads(
    nonzero_units_of_gene: &[Vec<(u32, f32)>],
    inputs: &ActivityInputs,
    anchor_units: &[u32],
    states: &UnitStates,
    settings: &ContrastFitSettings,
) -> Vec<FittedGene> {
    nonzero_units_of_gene
        .par_iter()
        .progress_count(nonzero_units_of_gene.len() as u64)
        .map_init(
            || inputs.new_scratch(),
            |scratch, nonzero_units| {
                inputs.write_activity_of_gene(nonzero_units, scratch);
                let sets =
                    contrast_sets_of_gene(&scratch.activity_of_unit, anchor_units, inputs.matched);
                clear_activity_of_gene(nonzero_units, scratch);
                let full_fit = fit_deviation_of_gene(&sets, states, None, None, settings);
                let refits = section_jackknife::deviation_without_each_section(
                    &full_fit.deviation,
                    &sets,
                    states,
                    &inputs.sections.section_index_of_unit,
                    inputs.sections.section_names.len(),
                    settings,
                );
                let fits_at_iteration_cap = std::iter::once(&full_fit)
                    .chain(&refits)
                    .filter(|fit| fit.reached_iteration_cap)
                    .count();
                FittedGene {
                    deviation: full_fit.deviation,
                    deviation_without_section: refits
                        .into_iter()
                        .map(|fit| fit.deviation)
                        .collect(),
                    fits_at_iteration_cap,
                }
            },
        )
        .collect()
}

/// Genes per block on a device without a memory query (CPU, Metal).
const GENES_PER_BLOCK_WITHOUT_MEMORY_QUERY: usize = 16;

/// How many genes a block fits in: on CUDA, as many as fit `memory_fraction`
/// of the device's free memory (each gene its activity and its refits'
/// working tensors, the refit phase being the larger), else a fixed count.
pub(super) fn genes_per_block(shared: &SharedContrastTensors, memory_fraction: f32) -> usize {
    let refits_per_gene = (shared.number_of_fits - 1).max(1);
    let bytes_per_gene = shared.bytes_per_item() * refits_per_gene
        + shared.member_unit.len() * std::mem::size_of::<f32>();
    match legume_numeric::candle::device::gpu_mem_info(shared.device()) {
        Some((free, _total)) => {
            let room = f64::from(memory_fraction.clamp(0.05, 0.95)) * free as f64;
            ((room / bytes_per_gene as f64) as usize).max(1)
        }
        None => GENES_PER_BLOCK_WITHOUT_MEMORY_QUERY,
    }
}

/// `--device cuda` or `metal`: blocks of `genes_per_block` genes, each
/// block's activities read on CPU threads and fitted on the device.
pub(super) fn fit_genes_in_blocks(
    nonzero_units_of_gene: &[Vec<(u32, f32)>],
    inputs: &ActivityInputs,
    shared: &SharedContrastTensors,
    settings: &ContrastFitSettings,
    genes_per_block: usize,
) -> anyhow::Result<Vec<FittedGene>> {
    let progress = indicatif::ProgressBar::new(nonzero_units_of_gene.len() as u64);
    let mut fitted = Vec::with_capacity(nonzero_units_of_gene.len());
    for block in nonzero_units_of_gene.chunks(genes_per_block.max(1)) {
        // each gene's activity on every member of every set
        let member_activity_of_gene: Vec<Vec<f32>> = block
            .par_iter()
            .map_init(
                || inputs.new_scratch(),
                |scratch, nonzero_units| {
                    inputs.write_activity_of_gene(nonzero_units, scratch);
                    let member_activity = shared
                        .member_unit
                        .iter()
                        .map(|&unit| scratch.activity_of_unit[unit as usize])
                        .collect();
                    clear_activity_of_gene(nonzero_units, scratch);
                    member_activity
                },
            )
            .collect();
        fitted.extend(fit_block_of_genes(
            shared,
            &member_activity_of_gene,
            settings,
        )?);
        progress.inc(block.len() as u64);
    }
    progress.finish_and_clear();
    Ok(fitted)
}
