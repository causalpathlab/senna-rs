//! Entry point for `senna tde`: track divergence embedding.
//!
//! The spliced counts alone place the cells and the genes, through `senna
//! bge`'s driver ([`crate::bge::driver::fit_embed_family`]): the bilinear
//! score `ρ_g·θ_c + b_g`, phase-1 multilevel-pseudobulk training and the
//! phase-2 cell encoder, and the same output set `senna bge` writes. Inside
//! the same fit, each gene's unspliced share of its reads in a cell is read
//! as a phase portrait: the gene's ratio plus its direction `η_g` in the
//! spliced space, against the gene's steady state, which gives each cell a
//! velocity in that space (`graph_embedding_util::DivergenceConfig`).
//! The unspliced counts never move `θ` or `ρ`.
//!
//! [`crate::tde::load::sample_ids`] names each input file's sample,
//! [`crate::tde::load::load_tde_data`] loads them and reads the
//! [`crate::tde::tracks::TrackPlan`] off the row grammar, and
//! [`crate::tde::hvg::hvg_row_weights`] ranks genes on their spliced rows
//! for HVG weighting. `graph_embedding_util::split_divergence` then cuts the axis to
//! the spliced rows, named by their genes, so the fit, the preset and every
//! output read like a `bge` run's.

use crate::bge::driver::{fit_embed_family, EmbedPlan};
use crate::tde::args::TdeArgs;
use crate::tde::hvg::hvg_row_weights;
use crate::tde::load::{load_tde_data, sample_ids};
use crate::tde::tracks::TrackPlan;
use data_beans::aux::feature_rows::{COUNT, UNSPLICED};
use graph_embedding_util as ge;
use legume_numeric::matrix::common_io::mkdir_parent;
use legume_numeric::matrix::parquet::{write_named_table, Column};
use senna::embed_common::*;
use senna::run_manifest::RunKind;

pub fn run_tde(args: &TdeArgs) -> anyhow::Result<()> {
    mkdir_parent(&args.out)?;
    args.collapse.reject_pb_reference(RunKind::Tde)?;

    let batch_files = senna::senna_input::effective_batch_files(
        args.collapse.ignore_batch,
        args.batch_files.as_deref(),
    );

    let ids = sample_ids(&args.genes, &args.genes_sample_strip)?;
    let (mut unified, plan) = load_tde_data(&args.genes, &ids, batch_files, args.preload_data)?;
    let hvg_weights = hvg_row_weights(&unified, &plan, &args.hvg, args.block_size)?;

    let axis = split_unspliced(&mut unified, &plan)?;

    let (mut preset_features, mut carried) = crate::feature_preset::resolve_preset(
        args.feature_embedding.resolve()?,
        &unified.feature_names,
        &ge::FeatureNameKind::Gene { delim: '_' },
    )?;
    let embedding_dim =
        crate::feature_preset::resolve_dim(args.embedding_dim, &mut preset_features, &mut carried)?;
    anyhow::ensure!(embedding_dim > 0, "--embedding-dim must be > 0");

    fit_embed_family(EmbedPlan {
        kind: RunKind::Tde,
        knobs: args.knobs(embedding_dim),
        unified,
        data_files: args.genes.clone(),
        multiome: None,
        hvg_weights,
        preset_features,
        carried,
        pb_reference: None,
        init_from: None,
        train_args: senna::run_manifest::record_train_args(args)?,
        divergence: Some(args.divergence(axis)),
    })
}

/// Cut the axis to its spliced rows, named by their genes, and say where each
/// gene's unspliced counts are.
fn split_unspliced(
    unified: &mut ge::UnifiedData,
    plan: &TrackPlan,
) -> anyhow::Result<ge::DivergenceAxis> {
    let pairs = plan.pair_rows()?;
    anyhow::ensure!(
        pairs.unspliced.iter().any(Option::is_some),
        "tde needs `{{gene}}/count/unspliced` rows beside the spliced ones; the input has none"
    );
    let track = format!("{COUNT}/{UNSPLICED}");
    let axis = ge::split_divergence(unified, &pairs.spliced, &pairs.unspliced, &track)?;
    unified.feature_names = pairs.genes;
    log::info!(
        "tde: {} genes on the spliced axis, {} with unspliced counts",
        unified.n_features(),
        axis.support().len()
    );
    Ok(axis)
}

/// The unspliced track's tables, beside the base fit's (see
/// [`senna::run_manifest::DivergenceSlots`]): per QC-kept cell its velocity
/// `θ̇_c` (`h0..`) and `κ_c`; per gene its ratio `δ_g`, steady state `ā_g`
/// and `log γ_g`; per gene its direction `η_g`.
pub(crate) fn write_divergence(
    out: &str,
    div: &ge::DivergenceOutput,
    unified: &ge::UnifiedData,
    keep: Option<&[usize]>,
) -> anyhow::Result<senna::run_manifest::DivergenceSlots> {
    let slots = senna::run_manifest::DivergenceSlots {
        track: div.track_name.to_string(),
        cell: "cell_velocity.parquet".into(),
        feature: "feature_divergence.parquet".into(),
        loading: "divergence_loading.parquet".into(),
    };
    let h = div.velocity_cell.ncols();

    let cells: Vec<usize> = keep.map_or_else(
        || (0..div.velocity_cell.nrows()).collect(),
        <[usize]>::to_vec,
    );
    let barcodes: Vec<Box<str>> = cells.iter().map(|&c| unified.barcodes[c].clone()).collect();
    let mut cell_table = div.velocity_cell.select_rows(&cells).insert_column(h, 0.0);
    for (i, &c) in cells.iter().enumerate() {
        cell_table[(i, h)] = div.kappa_cell[c];
    }
    let mut cell_cols = axis_id_names("h", h);
    cell_cols.push("kappa".into());
    cell_table.to_parquet_with_names(
        &format!("{out}.{}", slots.cell),
        (Some(&barcodes), Some("cell")),
        Some(&cell_cols),
    )?;

    let genes: Vec<Box<str>> = div
        .genes
        .iter()
        .map(|&g| unified.feature_names[g as usize].clone())
        .collect();
    write_named_table(
        &format!("{out}.{}", slots.feature),
        "feature",
        &genes,
        &[
            (Box::from("ratio"), Column::F32(&div.ratio)),
            (Box::from("steady_anchor"), Column::F32(&div.steady_anchor)),
            (Box::from("log_gamma"), Column::F32(&div.log_gamma)),
        ],
    )?;

    div.loading.to_parquet_with_names(
        &format!("{out}.{}", slots.loading),
        (Some(&genes), Some("feature")),
        Some(&axis_id_names("h", h)),
    )?;
    log::info!(
        "Wrote the `{}` divergence tables to {out}.{{{},{},{}}}",
        div.track_name,
        slots.cell,
        slots.feature,
        slots.loading
    );
    Ok(slots)
}

#[cfg(test)]
#[path = "run/tests.rs"]
mod tests;
