//! What stage 2 writes: only what cannot be recomputed cheaply from the rest.
//! Every gain, state summary or ranking is derived downstream from the
//! tables, with no cutoff decided here.
//!
//! A cell's LOG GAIN for a gene is its donor's deviation at the cell's state
//! coordinates, the fitted log ratio of the gene's activity on the cell's
//! state to that on matched samples of other donors:
//!
//! ```text
//! log_gain(gene, cell) = intercept[gene, donor(cell)]
//!                      + Σ_k state_axis_k[gene, donor(cell)] · state_axis_k[cell]
//! ```
//!
//! It is linear in the coordinates, so its mean over any set of a donor's
//! cells (a cluster, say) is the deviation at their mean coordinates.
//!
//! * `{out}.donor_gene_deviation.parquet`: a row per (gene, donor), the
//!   deviation's coefficients (`intercept`, `state_axis_1..`).
//! * `{out}.donor_gene_deviation_without_batch.parquet`: a row per (gene,
//!   donor, left-out batch), the coefficients refitted without that batch's
//!   samples: the jackknife replicates; a quantity's variance is
//!   `(B − 1) / B · Σ_b (q_b − mean)²`.
//! * `{out}.donor_sample_states.parquet`: a row per pseudobulk sample, its
//!   batch, donor, cells, whether it anchored a contrast set, how many other
//!   donors its matches came from, its distances to its matched samples in
//!   the stage-1 embedding (nearest and mean: large where no other donor has
//!   the sample's state), and its state coordinates.
//! * `{out}.donor_cell_states.parquet`: a row per cell, its sample, batch and
//!   donor and its own state coordinates (the read-out).
//! * `{out}.donor_deviation.json`: the settings and donors of the run.

use super::contrast_fit::{ContrastFitSettings, DeviationOfGene};
use super::fit_genes::FittedGene;
use super::inputs::Stage1Cells;
use super::matched_other_donor_units::MatchedOtherDonorUnits;
use super::samples::PbSamples;
use super::DonorDeviationArgs;
use legume_numeric::matrix::parquet::{write_named_table, Column};
use senna::embed_common::*;

/// `state_axis_1`, `state_axis_2`, ...: the names of the state axes, shared
/// by the gene tables' coefficients and the state tables' coordinates.
fn state_axis_names(number_of_state_axes: usize) -> Vec<Box<str>> {
    (1..=number_of_state_axes)
        .map(|a| format!("state_axis_{a}").into())
        .collect()
}

/// Everything the tables are written from.
pub(crate) struct DonorDeviationResults<'a> {
    pub gene_names: &'a [Box<str>],
    pub section_names: &'a [Box<str>],
    pub cells: &'a Stage1Cells,
    /// `[cells × width]`, row-major, intercept first.
    pub state_coordinates_of_cell: &'a [f32],
    pub width: usize,
    pub samples: &'a PbSamples,
    pub matched: &'a MatchedOtherDonorUnits,
    pub anchor_samples: &'a [u32],
    pub fitted_genes: &'a [FittedGene],
}

pub(crate) fn write_donor_deviation_outputs(
    out: &str,
    results: &DonorDeviationResults,
) -> anyhow::Result<()> {
    mkdir_parent(out)?;
    write_gene_table(
        &format!("{out}.donor_gene_deviation.parquet"),
        results,
        WhichFits::FullFits,
    )?;
    write_gene_table(
        &format!("{out}.donor_gene_deviation_without_batch.parquet"),
        results,
        WhichFits::RefitsWithoutEachBatch,
    )?;
    write_sample_state_table(&format!("{out}.donor_sample_states.parquet"), results)?;
    write_cell_state_table(&format!("{out}.donor_cell_states.parquet"), results)?;
    Ok(())
}

/// Which of each gene's fits a gene table holds.
#[derive(Clone, Copy, PartialEq, Eq)]
enum WhichFits {
    /// The full fit: a row per (gene, donor).
    FullFits,
    /// The jackknife refits: a row per (gene, donor, left-out batch).
    RefitsWithoutEachBatch,
}

/// A gene table: per row the gene, the donor, the left-out batch (refits
/// only) and the deviation's coefficients (`intercept`, `state_axis_1..`).
fn write_gene_table(
    path: &str,
    results: &DonorDeviationResults,
    which: WhichFits,
) -> anyhow::Result<()> {
    let width = results.width;
    let donor_names = &results.cells.donor_names;
    let mut gene = Vec::new();
    let mut donor = Vec::new();
    let mut left_out_batch = Vec::new();
    let mut coefficient_of_column: Vec<Vec<f32>> = vec![Vec::new(); width];
    for (gene_name, fit) in results.gene_names.iter().zip(results.fitted_genes) {
        let deviations: Vec<(Option<&Box<str>>, &DeviationOfGene)> = match which {
            WhichFits::FullFits => vec![(None, &fit.deviation)],
            WhichFits::RefitsWithoutEachBatch => results
                .section_names
                .iter()
                .map(Some)
                .zip(&fit.deviation_without_section)
                .collect(),
        };
        for (batch_name, deviation) in deviations {
            for (donor_name, row) in donor_names.iter().zip(deviation.chunks(width)) {
                gene.push(gene_name.clone());
                donor.push(donor_name.clone());
                if let Some(batch_name) = batch_name {
                    left_out_batch.push(batch_name.clone());
                }
                for (column, &c) in coefficient_of_column.iter_mut().zip(row) {
                    column.push(c);
                }
            }
        }
    }
    let mut columns: Vec<(Box<str>, Column)> = vec![("donor".into(), Column::Str(&donor))];
    if which == WhichFits::RefitsWithoutEachBatch {
        columns.push(("left_out_batch".into(), Column::Str(&left_out_batch)));
    }
    let coefficient_names = std::iter::once("intercept".into()).chain(state_axis_names(width - 1));
    for (name, column) in coefficient_names.zip(&coefficient_of_column) {
        columns.push((name, Column::F32(column)));
    }
    write_named_table(path, "gene", &gene, &columns)?;
    info!("Wrote {path}");
    Ok(())
}

/// The coordinates past the intercept of `[rows × width]` (row-major), one
/// column per state axis.
fn coordinate_columns(coordinates: &[f32], width: usize) -> Vec<Vec<f32>> {
    (1..width)
        .map(|axis| coordinates.chunks(width).map(|row| row[axis]).collect())
        .collect()
}

fn write_sample_state_table(path: &str, results: &DonorDeviationResults) -> anyhow::Result<()> {
    let DonorDeviationResults {
        cells,
        samples,
        matched,
        anchor_samples,
        width,
        ..
    } = *results;
    let donor: Vec<Box<str>> = samples
        .donor_index_of_sample
        .iter()
        .map(|&d| cells.donor_names[d].clone())
        .collect();
    let number_of_cells: Vec<i32> = samples.cells_in_sample.iter().map(|&n| n as i32).collect();
    let mut is_anchor = vec![0i32; samples.number_of_samples()];
    for &s in anchor_samples {
        is_anchor[s as usize] = 1;
    }
    let other_donors: Vec<i32> = (0..samples.number_of_samples())
        .map(|s| {
            let mut donors: Vec<usize> = matched
                .matched_of(s)
                .iter()
                .map(|&m| samples.donor_index_of_sample[m as usize])
                .collect();
            donors.sort_unstable();
            donors.dedup();
            donors.len() as i32
        })
        .collect();
    let distances: Vec<&[f32]> = matched
        .distance_to_matched_unit
        .chunks(matched.matched_per_unit)
        .collect();
    let nearest_matched_distance: Vec<f32> = distances.iter().map(|d| d[0]).collect();
    let mean_matched_distance: Vec<f32> = distances
        .iter()
        .map(|d| d.iter().sum::<f32>() / d.len() as f32)
        .collect();
    let axes = coordinate_columns(&samples.state_coordinates_of_sample, width);
    let axis_names = state_axis_names(width - 1);
    let mut columns: Vec<(Box<str>, Column)> = vec![
        ("batch".into(), Column::Str(&samples.batch_of_sample)),
        ("donor".into(), Column::Str(&donor)),
        ("cells".into(), Column::I32(&number_of_cells)),
        ("anchor".into(), Column::I32(&is_anchor)),
        ("matched_other_donors".into(), Column::I32(&other_donors)),
        (
            "nearest_matched_distance".into(),
            Column::F32(&nearest_matched_distance),
        ),
        (
            "mean_matched_distance".into(),
            Column::F32(&mean_matched_distance),
        ),
    ];
    for (name, column) in axis_names.iter().zip(&axes) {
        columns.push((name.clone(), Column::F32(column)));
    }
    write_named_table(path, "sample", &samples.sample_names, &columns)?;
    info!("Wrote {path}");
    Ok(())
}

fn write_cell_state_table(path: &str, results: &DonorDeviationResults) -> anyhow::Result<()> {
    let DonorDeviationResults {
        cells,
        samples,
        state_coordinates_of_cell,
        width,
        ..
    } = *results;
    let sample: Vec<Box<str>> = samples
        .sample_of_cell
        .iter()
        .map(|&s| samples.sample_names[s].clone())
        .collect();
    let donor: Vec<Box<str>> = cells
        .donor_index_of_cell
        .iter()
        .map(|&d| cells.donor_names[d].clone())
        .collect();
    let axes = coordinate_columns(state_coordinates_of_cell, width);
    let axis_names = state_axis_names(width - 1);
    let mut columns: Vec<(Box<str>, Column)> = vec![
        ("sample".into(), Column::Str(&sample)),
        ("batch".into(), Column::Str(&cells.batch_of_cell)),
        ("donor".into(), Column::Str(&donor)),
    ];
    for (name, column) in axis_names.iter().zip(&axes) {
        columns.push((name.clone(), Column::F32(column)));
    }
    write_named_table(path, "cell", &cells.cell_names, &columns)?;
    info!("Wrote {path}");
    Ok(())
}

pub(crate) fn write_settings_record(
    args: &DonorDeviationArgs,
    donor_names: &[Box<str>],
    settings: &ContrastFitSettings,
    number_of_samples: usize,
    number_of_anchor_samples: usize,
) -> anyhow::Result<()> {
    let record = serde_json::json!({
        "command": "donor-deviation",
        "version": env!("CARGO_PKG_VERSION"),
        "stage1_run": args.from,
        "donor_of_batch": args.donor_of_batch,
        "donors": donor_names,
        "units": "pseudobulk samples (finest pseudobulk × batch), counts averaged per cell",
        "samples": number_of_samples,
        "anchor_samples": number_of_anchor_samples,
        "matched_samples": args.matched_samples,
        "min_other_donors": args.min_other_donors,
        "state_axes": args.state_axes,
        "fit": settings,
        "seed": args.seed,
        "batch_jackknife": "each batch left out in turn, refitted from the full fit",
    });
    let path = format!("{}.donor_deviation.json", args.out);
    std::fs::write(&path, serde_json::to_string_pretty(&record)?)?;
    info!("Wrote {path}");
    Ok(())
}
