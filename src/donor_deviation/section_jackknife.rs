//! The section jackknife: each gene's deviation refitted without each
//! section in turn.
//!
//! Sections are the replicates, so the spread of these refits is the
//! deviation's jackknife uncertainty. Downstream evaluates any quantity (a
//! state's mean gain, say) on each and takes
//! `(sections − 1) / sections · Σ_s (value_without_s − mean)²` as its
//! variance.
//!
//! A refit is exact but cheap: the same FISTA, on the loss with the
//! section's units left out (`contrast_fit::LeftOutSection`), started from
//! the full fit, which differs from the refit's optimum by one section's
//! worth of data. The problem is convex, so the start changes the
//! iterations, not the answer. This is how pinto's `lra` region jackknife refits
//! its nuisance models (`lr_activity::ligand_effect::jackknife`), after
//! Giordano, Stephenson, Liu, Jordan & Broderick (2019, "A Swiss army
//! infinitesimal jackknife", AISTATS, PMLR 89:1139). Each refit stops at
//! its own optimum (`ContrastFitSettings::gradient_mapping_tolerance`).
//!
//! One approximation: each gene's same-section factor is estimated once,
//! from every section, and kept for the refits. The left-out section's
//! units still enter the other sections' factors as matches, so the
//! replicates are not wholly free of it and the variance can come out a
//! little small.

use super::contrast_fit::{
    fit_deviation_of_gene, ContrastFitSettings, ContrastSetsOfGene, FitOfGene, LeftOutSection,
    UnitStates,
};

/// The gene's deviation without each section in turn (section index
/// order), each refitted from its full fit `deviation`.
pub fn deviation_without_each_section(
    deviation: &[f32],
    sets: &ContrastSetsOfGene,
    states: &UnitStates,
    section_index_of_unit: &[usize],
    number_of_sections: usize,
    settings: &ContrastFitSettings,
) -> Vec<FitOfGene> {
    (0..number_of_sections)
        .map(|left_out_section| {
            fit_deviation_of_gene(
                sets,
                states,
                Some(deviation),
                Some(&LeftOutSection {
                    section_index_of_unit,
                    left_out_section,
                }),
                settings,
            )
        })
        .collect()
}
