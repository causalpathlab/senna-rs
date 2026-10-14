//! `--{freeze,init,lora}-feature-embedding <prefix>` for every model with a
//! gene table (`senna bge`, `senna simba`, `senna fne`): the gene and
//! genomic-window rows of an earlier run's feature table, matched onto the
//! caller's feature axis, to pin, to start from, or to anchor a low-rank
//! residual to. The result is a [`ge::PresetRows`] with ids into the given
//! axis; the PBG commands lift it to their node ids with `map_ids`.
//!
//! The source is any run whose prefix resolves through
//! [`senna::run_manifest::resolve_feature_embedding`] — typically `senna fne`,
//! whose table also holds terms, words and cell types. Its
//! `feature_types.parquet`, when it lists the table's rows, keeps those rows
//! from matching; without one every row is a gene. Features of this axis
//! with no source row stay free.
//!
//! Under a pinning mode the rows the match left unused — genes the data lacks
//! and every non-gene row — come back out as [`CarriedRows`]: appended
//! unchanged to the run's own ρ table once it is written, with
//! `feature_types.parquet` naming every row's type, so a run on a narrow
//! feature axis (a panel) still hands on the full table it was given.

use data_beans::aux::feature_types::FeatureType;
use data_beans::aux::frozen_features::{load_frozen_feature_host_matching, FrozenLoadArgs};
use graph_embedding_util as ge;
use graph_embedding_util::PresetMode;
use log::info;
use senna::carried_rows::matchable_rows;
use std::cell::Cell;

pub(crate) use senna::carried_rows::CarriedRows;

/// The preset of a command's `--{freeze,init,lora}-feature-embedding`, when
/// one was given: the rows to pin or start from, and the rows to carry.
pub(crate) fn resolve_preset(
    resolved: Option<(&str, PresetMode)>,
    feature_names: &[Box<str>],
    kind: &ge::FeatureNameKind,
) -> anyhow::Result<(Option<ge::PresetRows>, Option<CarriedRows>)> {
    let Some((prefix, mode)) = resolved else {
        return Ok((None, None));
    };
    let (rows, carried) = load_preset_genes(prefix, mode, feature_names, kind)?;
    Ok((Some(rows), carried))
}

pub(crate) fn load_preset_genes(
    prefix: &str,
    mode: PresetMode,
    feature_names: &[Box<str>],
    kind: &ge::FeatureNameKind,
) -> anyhow::Result<(ge::PresetRows, Option<CarriedRows>)> {
    let flag = crate::feature_embedding_args::flag_name(mode);
    let (dictionary_path, _bias) = senna::run_manifest::resolve_feature_embedding(prefix)
        .map_err(|e| anyhow::anyhow!("{flag} {prefix}: {e}"))?;

    // The source's row types, when it wrote them for this table: only its
    // gene and region rows may match (a term, word or cell type may share a
    // gene's name).
    let ctx = |e: anyhow::Error| anyhow::anyhow!("{flag} {prefix}: {e}");
    let written = senna::run_manifest::feature_types_beside(&dictionary_path).map_err(ctx)?;
    let checked: Cell<Option<&[FeatureType]>> = Cell::new(None);
    let host = load_frozen_feature_host_matching(
        FrozenLoadArgs {
            dictionary_path: &dictionary_path,
            bias_path: None,
            target_feature_names: feature_names,
            name_kind: kind.clone(),
            source_name_map: None,
        },
        |names| {
            let (marks, types) = matchable_rows(written.as_deref(), names, &dictionary_path);
            checked.set(types);
            Ok(marks)
        },
    )
    .map_err(ctx)?;
    let src_types: &[FeatureType] = checked.get().unwrap_or(&[]);

    let h = host.e_feat.ncols();
    let ids: Vec<u32> = host.keep_target_indices.iter().map(|&t| t as u32).collect();
    // Row-major: the transpose's column-major storage.
    let rows: Vec<f32> = host.e_feat.transpose().data.into();
    info!(
        "Feature side from {dictionary_path} (H={h}): {} of {} features {}",
        ids.len(),
        feature_names.len(),
        mode.describe()
    );
    let carried = CarriedRows::from_unmatched(
        mode.pins(),
        flag,
        &host,
        feature_names,
        kind,
        src_types,
        &dictionary_path,
    )?;
    Ok((ge::PresetRows { ids, rows, mode }, carried))
}

/// The width H of a run given `preset`: the larger of `--embedding-dim`
/// (`auto` is the table's own width) and the table's width, so a table
/// narrower than the command's default never refuses the run. A table
/// narrower than H has its rows — given and carried — widened with zero
/// columns: under `--init-` and `--lora-` those columns
/// train like the rest; under `--freeze-` a pinned gene keeps zeros there,
/// and the extra dimensions are the cells' and the free genes'.
pub(crate) fn resolve_dim(
    cli_embedding_dim: ge::EmbeddingDim,
    preset: &mut Option<ge::PresetRows>,
    carried: &mut Option<CarriedRows>,
) -> anyhow::Result<usize> {
    let Some(h) = preset.as_ref().map(ge::PresetRows::width) else {
        return cli_embedding_dim.resolve(None)?.ok_or_else(|| {
            anyhow::anyhow!(
                "--embedding-dim auto takes H from a given feature embedding; none was given"
            )
        });
    };
    anyhow::ensure!(h > 0, "the given feature embedding has no columns");
    let width = match cli_embedding_dim {
        ge::EmbeddingDim::Auto => h,
        ge::EmbeddingDim::Fixed(d) => d.max(h),
    };
    // A LoRA rank is low against the width the run trains at, not the table's.
    if let Some(p) = preset.as_ref() {
        p.mode.validate(width)?;
    }
    if width == h {
        info!("H = {h}, the given feature embedding's width");
        return Ok(h);
    }
    let pad = width - h;
    if preset
        .as_ref()
        .is_some_and(|p| matches!(p.mode, PresetMode::Freeze))
    {
        // Pinned rows stay zero there: only the cells and the free genes
        // can use the extra dimensions.
        log::warn!(
            "H = {width}: the given feature embedding is {h} wide; its pinned rows keep zeros \
             in the {pad} extra column(s), which only the free genes can fill. \
             --embedding-dim {h} (or auto) keeps the table's width"
        );
    } else {
        info!("H = {width}: the given feature embedding is {h} wide; its rows get {pad} zero column(s)");
    }
    if let Some(p) = preset.as_mut() {
        p.rows = zero_pad_rows(&p.rows, h, width);
    }
    if let Some(c) = carried.as_mut() {
        let rows = std::mem::replace(&mut c.rows, nalgebra::DMatrix::zeros(0, 0));
        c.rows = rows.resize_horizontally(width, 0.0);
    }
    Ok(width)
}

/// Row-major `[n × h]` rows as `[n × width]`, the new columns zero.
fn zero_pad_rows(rows: &[f32], h: usize, width: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(rows.len() / h.max(1) * width);
    for row in rows.chunks_exact(h) {
        out.extend_from_slice(row);
        out.resize(out.len() + width - h, 0.0);
    }
    out
}

#[cfg(test)]
mod tests;

/// For the engines' end-to-end tests: a source run wider than the data.
#[cfg(test)]
pub(crate) mod test_support {
    use legume_numeric::matrix::traits::IoOps;
    use nalgebra::DMatrix;

    /// The extra rows [`widen`] adds: a gene the data lacks and a term.
    pub(crate) const EXTRA: [(&str, &str); 2] = [("EXTRA1", "gene"), ("GO:9999999", "term")];

    /// Write `{out}.feature_embedding.parquet` = the ρ table at `src_rho_path`
    /// plus [`EXTRA`], with a types table over every row; returns the extra
    /// rows' values for the caller to look for in a run's output.
    pub(crate) fn widen(src_rho_path: &str, out: &str) -> DMatrix<f32> {
        let t = DMatrix::<f32>::from_parquet(src_rho_path).unwrap();
        let (n, h) = (t.mat.nrows(), t.mat.ncols());
        let extra =
            DMatrix::<f32>::from_fn(EXTRA.len(), h, |i, k| (i + 1) as f32 * 0.25 + k as f32);
        let mat =
            legume_numeric::matrix::dmatrix_util::concatenate_vertical(&[t.mat, extra.clone()])
                .unwrap();
        let mut names = t.rows.clone();
        let mut types: Vec<Box<str>> = vec!["gene".into(); n];
        for (name, ty) in EXTRA {
            names.push(name.into());
            types.push(ty.into());
        }
        mat.to_parquet_with_names(
            &format!("{out}.feature_embedding.parquet"),
            (Some(&names), Some("gene")),
            Some(&t.cols),
        )
        .unwrap();
        data_beans::aux::feature_types::write_feature_types(out, &names, &types).unwrap();
        extra
    }

    /// Assert the run at `out` wrote [`EXTRA`] after its own rows in `rho_path`,
    /// row for row equal to `extra`, and typed them in its types table.
    pub(crate) fn assert_carried(out: &str, rho_path: &str, own_rows: usize, extra: &DMatrix<f32>) {
        let t = DMatrix::<f32>::from_parquet(rho_path).unwrap();
        assert_eq!(t.mat.nrows(), own_rows + EXTRA.len(), "{rho_path}");
        for (i, (name, _)) in EXTRA.iter().enumerate() {
            assert_eq!(t.rows[own_rows + i].as_ref(), *name);
            assert_eq!(
                t.mat.row(own_rows + i),
                extra.row(i),
                "{name} is carried unchanged"
            );
        }
        let types = data_beans::aux::feature_types::read_feature_types(out)
            .unwrap()
            .expect("a types table over every row");
        assert_eq!(types.len(), own_rows + EXTRA.len());
        for (i, (name, ty)) in EXTRA.iter().enumerate() {
            assert_eq!(types[own_rows + i], ((*name).into(), (*ty).into()));
        }
    }
}
