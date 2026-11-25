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
//!   donor), and per left-out batch `b` the coefficients refitted without
//!   its samples (`intercept.without_{b}`, `state_axis_1.without_{b}`, ..):
//!   the jackknife replicates; a quantity's variance is
//!   `(B − 1) / B · Σ_b (q_b − mean)²`.
//! * `{out}.donor_sample_states.parquet`: a row per pseudobulk sample, its
//!   batch, donor, cells, whether it anchored a contrast set, how many other
//!   donors its matches came from, its distances to its matched samples in
//!   the stage-1 embedding (nearest and mean: large where no other donor has
//!   the sample's state), and its state coordinates.
//! * `{out}.donor_cell_states.parquet`: a row per cell, its sample, batch and
//!   donor and its own state coordinates (the read-out).
//! * `{out}.donor_deviation.json`: the settings and donors of the run.

use super::contrast_fit::ContrastFitSettings;
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

/// The tables' names, in the order they are written: each is
/// `{out}.{name}.parquet`.
pub(crate) const TABLES: [&str; 4] = [
    "donor_gene_deviation",
    "donor_gene_deviation_without_batch",
    "donor_sample_states",
    "donor_cell_states",
];

/// `{out}.{table}.parquet`.
pub(crate) fn table_path(out: &str, table: &str) -> String {
    format!("{out}.{table}.parquet")
}

pub(crate) fn write_donor_deviation_outputs(
    out: &str,
    results: &DonorDeviationResults,
) -> anyhow::Result<()> {
    mkdir_parent(out)?;
    let [genes, without_batch, samples, cells] = TABLES.map(|t| table_path(out, t));
    write_gene_table(&genes, results, WhichFits::FullFits)?;
    write_gene_table(&without_batch, results, WhichFits::RefitsWithoutEachBatch)?;
    write_sample_state_table(&samples, results)?;
    write_cell_state_table(&cells, results)?;
    Ok(())
}

/// Record the tables in the stage-1 run's manifest (`from`), so a viewer of
/// the run finds them; a later stage 2 on the run takes the slot over.
pub(crate) fn record_in_stage1_manifest(from: &str, out: &str) -> anyhow::Result<()> {
    use senna::run_manifest::{manifest_path_for, RunManifest, RunStage2};
    let path = manifest_path_for(from);
    RunManifest::update(std::path::Path::new(&path), |m, rel| {
        let [genes, without_batch, samples, cells] = TABLES.map(|t| Some(rel(&table_path(out, t))));
        m.stage2 = RunStage2 {
            donor_gene_deviation: genes,
            donor_gene_deviation_without_batch: without_batch,
            donor_sample_states: samples,
            donor_cell_states: cells,
            settings: Some(rel(&settings_path(out))),
            written: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .ok()
                .and_then(|d| u64::try_from(d.as_nanos()).ok()),
            unknown: std::mem::take(&mut m.stage2.unknown),
        };
    })?;
    info!("Recorded stage 2 in {path}");
    Ok(())
}

/// `{out}.donor_deviation.json`.
fn settings_path(out: &str) -> String {
    format!("{out}.donor_deviation.json")
}

/// Which of each gene's fits a gene table holds.
#[derive(Clone, Copy, PartialEq, Eq)]
enum WhichFits {
    /// The full fit: per coefficient, one column.
    FullFits,
    /// The jackknife refits: per left-out batch and coefficient, one column
    /// (`{coefficient}.without_{batch}`), so the table keeps a row per (gene,
    /// donor) however many batches there are.
    RefitsWithoutEachBatch,
}

/// A gene table: a row per (gene, donor), the deviation's coefficients
/// (`intercept`, `state_axis_1..`) of the full fit, or of each refit.
fn write_gene_table(
    path: &str,
    results: &DonorDeviationResults,
    which: WhichFits,
) -> anyhow::Result<()> {
    let width = results.width;
    let donor_names = &results.cells.donor_names;
    let coefficient_names: Vec<Box<str>> = std::iter::once("intercept".into())
        .chain(state_axis_names(width - 1))
        .collect();
    // (column name, which fit: None = full, Some(b) = without batch b, coefficient)
    let column_specs: Vec<(Box<str>, Option<usize>, usize)> = match which {
        WhichFits::FullFits => coefficient_names
            .iter()
            .enumerate()
            .map(|(k, name)| (name.clone(), None, k))
            .collect(),
        WhichFits::RefitsWithoutEachBatch => results
            .section_names
            .iter()
            .enumerate()
            .flat_map(|(b, batch)| {
                coefficient_names
                    .iter()
                    .enumerate()
                    .map(move |(k, name)| (format!("{name}.without_{batch}").into(), Some(b), k))
            })
            .collect(),
    };
    let number_of_rows = results.gene_names.len() * donor_names.len();
    let mut gene = Vec::with_capacity(number_of_rows);
    let mut donor = Vec::with_capacity(number_of_rows);
    let mut values: Vec<Vec<f32>> = vec![Vec::with_capacity(number_of_rows); column_specs.len()];
    for (gene_name, fit) in results.gene_names.iter().zip(results.fitted_genes) {
        for (d, donor_name) in donor_names.iter().enumerate() {
            gene.push(gene_name.clone());
            donor.push(donor_name.clone());
            for (column, &(_, refit, k)) in values.iter_mut().zip(&column_specs) {
                let deviation = match refit {
                    None => &fit.deviation,
                    Some(b) => &fit.deviation_without_section[b],
                };
                column.push(deviation[d * width + k]);
            }
        }
    }
    let mut columns: Vec<(Box<str>, Column)> = vec![("donor".into(), Column::Str(&donor))];
    columns.extend(
        column_specs
            .iter()
            .zip(&values)
            .map(|((name, _, _), column)| (name.clone(), Column::F32(column))),
    );
    write_named_table(path, "gene", &gene, &columns)?;
    info!("Wrote {path}");
    Ok(())
}

/// The coordinates past the intercept of `[rows × width]` (row-major), one
/// named column per state axis.
fn state_axis_columns(coordinates: &[f32], width: usize) -> Vec<(Box<str>, Vec<f32>)> {
    state_axis_names(width - 1)
        .into_iter()
        .enumerate()
        .map(|(a, name)| {
            let column = coordinates.chunks(width).map(|row| row[1 + a]).collect();
            (name, column)
        })
        .collect()
}

/// `names[i]` for each index.
fn names_at(names: &[Box<str>], indices: &[usize]) -> Vec<Box<str>> {
    indices.iter().map(|&i| names[i].clone()).collect()
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
    let donor = names_at(&cells.donor_names, &samples.donor_index_of_sample);
    let number_of_cells: Vec<i32> = samples.cells_in_sample.iter().map(|&n| n as i32).collect();
    let mut is_anchor = vec![0i32; samples.number_of_samples()];
    for &s in anchor_samples {
        is_anchor[s as usize] = 1;
    }
    let other_donors: Vec<i32> = (0..samples.number_of_samples())
        .map(|s| matched.other_donors_of(s, &samples.donor_index_of_sample) as i32)
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
    let axes = state_axis_columns(&samples.state_coordinates_of_sample, width);
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
    columns.extend(axes.iter().map(|(name, c)| (name.clone(), Column::F32(c))));
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
    let sample = names_at(&samples.sample_names, &samples.sample_of_cell);
    let donor = names_at(&cells.donor_names, &cells.donor_index_of_cell);
    let axes = state_axis_columns(state_coordinates_of_cell, width);
    let mut columns: Vec<(Box<str>, Column)> = vec![
        ("sample".into(), Column::Str(&sample)),
        ("batch".into(), Column::Str(&cells.batch_of_cell)),
        ("donor".into(), Column::Str(&donor)),
    ];
    columns.extend(axes.iter().map(|(name, c)| (name.clone(), Column::F32(c))));
    write_named_table(path, "cell", &cells.cell_names, &columns)?;
    info!("Wrote {path}");
    Ok(())
}

pub(crate) fn write_settings_record(
    out: &str,
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
    let path = settings_path(out);
    std::fs::write(&path, serde_json::to_string_pretty(&record)?)?;
    info!("Wrote {path}");
    Ok(())
}
