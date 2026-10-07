//! Entry point for `senna tde`: track divergence embedding.
//!
//! The spliced counts alone place the cells and the genes, through `senna
//! bge`'s driver ([`crate::bge::driver::fit_embed_family`]): the bilinear
//! score `ρ_g·θ_c + b_g`, phase-1 multilevel-pseudobulk training and the
//! phase-2 cell encoder, and the same output set `senna bge` writes. Inside
//! the same fit, each gene's unspliced share of its reads in a unit is read
//! as a phase portrait: the gene's average ratio plus a displacement `d` of
//! the unit within that space, against the gene's steady state
//! (`graph_embedding_util::DisplacedTrackConfig`). The unspliced counts never
//! move `θ` or `ρ`.
//!
//! [`crate::tde::load::resolve_inputs`] sample-id-matches the input files,
//! [`crate::tde::load::load_gem_data`] loads them and reads the
//! [`crate::tde::tracks::TrackPlan`] off the row grammar, and
//! [`crate::tde::hvg::gem_hvg_row_weights`] pools its rows per gene for HVG
//! weighting. `graph_embedding_util::split_displaced` then cuts the axis to
//! the spliced rows, named by their genes, so the fit, the preset and every
//! output read like a `bge` run's.

use crate::bge::driver::{fit_embed_family, EmbedPlan};
use crate::tde::args::TdeArgs;
use crate::tde::hvg::gem_hvg_row_weights;
use crate::tde::load::{load_gem_data, resolve_inputs};
use crate::tde::tracks::TrackPlan;
use data_beans::aux::feature_rows::{parse_feature_row, COUNT, SPLICED, UNSPLICED};
use graph_embedding_util as ge;
use legume_numeric::matrix::common_io::mkdir_parent;
use senna::run_manifest::RunKind;

pub fn run_tde(args: &TdeArgs) -> anyhow::Result<()> {
    mkdir_parent(&args.out)?;
    args.collapse.reject_pb_reference(RunKind::Tde)?;

    let batch_files = senna::senna_input::effective_batch_files(
        args.collapse.ignore_batch,
        args.batch_files.as_deref(),
    );

    let inputs = resolve_inputs(&args.genes, &[], &args.genes_sample_strip)?;
    let (mut unified, plan) = load_gem_data(&inputs, batch_files, args.preload_data)?;
    let hvg_weights = gem_hvg_row_weights(&unified, &plan, &args.hvg, args.block_size)?;

    let axis = split_unspliced(&mut unified, &plan)?;

    let (mut preset_features, mut carried) = match args.feature_embedding.resolve()? {
        Some((prefix, mode)) => {
            let (rows, carried) = crate::feature_preset::load_preset_rows(
                prefix,
                mode,
                &unified.feature_names,
                &ge::FeatureNameKind::Gene { delim: '_' },
                Some(&gene_of_spliced_row),
            )?;
            (Some(rows), carried)
        }
        None => (None, None),
    };
    let embedding_dim =
        crate::feature_preset::resolve_dim(args.embedding_dim, &mut preset_features, &mut carried)?;
    anyhow::ensure!(embedding_dim > 0, "--embedding-dim must be > 0");

    let data_files = inputs.files.clone();
    fit_embed_family(EmbedPlan {
        kind: RunKind::Tde,
        knobs: args.knobs(embedding_dim),
        unified,
        data_files,
        multiome: None,
        hvg_weights,
        preset_features,
        carried,
        pb_reference: None,
        init_from: None,
        train_args: senna::run_manifest::record_train_args(args)?,
        displaced: Some(args.displaced(axis)),
    })
}

/// Cut the axis to its spliced rows, named by their genes, and say where each
/// gene's unspliced counts are.
fn split_unspliced(
    unified: &mut ge::UnifiedData,
    plan: &TrackPlan,
) -> anyhow::Result<ge::DisplacedAxis> {
    let other: Vec<String> = plan
        .tracks
        .iter()
        .filter(|t| t.modality.as_ref() != COUNT)
        .map(|t| format!("{}/{}", t.modality, t.channel))
        .collect();
    anyhow::ensure!(
        other.is_empty(),
        "tde fits `count/spliced` and `count/unspliced` rows only; the input also has {other:?}"
    );
    anyhow::ensure!(
        plan.tracks.iter().any(|t| t.channel.as_ref() == UNSPLICED),
        "tde needs `{{gene}}/count/unspliced` rows beside the spliced ones; the input has none"
    );
    let (base_rows, unspliced_rows) = plan.pair_rows(1)?;
    let track = format!("{COUNT}/{UNSPLICED}");
    let axis = ge::split_displaced(unified, &base_rows, &unspliced_rows, &track)?;
    for name in &mut unified.feature_names {
        let gene = parse_feature_row(name)
            .map(|row| Box::<str>::from(row.gene))
            .ok_or_else(|| anyhow::anyhow!("spliced row {name:?} has no gene field"))?;
        *name = gene;
    }
    log::info!(
        "tde: {} genes on the spliced axis, {} with unspliced counts",
        unified.n_features(),
        axis.support().len()
    );
    Ok(axis)
}

/// A preset table's row onto the gene axis: a `{gene}/count/spliced` row (an
/// earlier joint run's table) is its gene; any other name is itself, so a
/// bare gene table matches as it does for bge.
fn gene_of_spliced_row(name: &str) -> Box<str> {
    match parse_feature_row(name) {
        Some(row) if row.modality == COUNT && row.channel == SPLICED && row.subunit.is_none() => {
            row.gene.into()
        }
        _ => name.into(),
    }
}

#[cfg(test)]
#[path = "run/tests.rs"]
mod tests;
