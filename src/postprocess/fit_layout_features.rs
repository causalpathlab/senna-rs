//! Feature (gene) layouts for `senna layout`.
//!
//! Two views, both keyed by layout method in `manifest.layout.methods`:
//!
//! - **features on the cell map**: after a cell layout, each feature in
//!   `feature_coembedding` (genes averaged onto the cell manifold, same H-space
//!   as the cell embedding) is placed by the same Nyström kernel that places
//!   cells, against a subsample of the just-laid-out cells. Runs automatically
//!   for embedding runs that wrote a co-embed; no flag.
//! - **features on their own** (`--target features`): the feature embedding ρ
//!   is laid out directly, gene-gene structure only. Needs no count data.

use super::fit_layout_common::LayoutCommonArgs;
use crate::geometry::cell_layout::project_cells_nystrom;
use rand::rngs::SmallRng;
use rand::SeedableRng;
use senna::embed_common::*;
use senna::run_manifest::{
    self, manifest_dir, rel_to_manifest, CellSpace, LayoutEntry, RunManifest,
};
use std::path::{Path, PathBuf};

/// Which axis `senna layout` lays out.
#[derive(ValueEnum, Clone, Copy, Debug, PartialEq, Eq, Default)]
#[clap(rename_all = "kebab-case")]
pub enum LayoutTarget {
    /// Cells (the default). Embedding runs with a co-embed also get their
    /// features placed on the resulting cell map.
    #[default]
    Cells,
    /// The feature embedding ρ on its own; reads only the manifest.
    Features,
}

/// Cells used as Nyström landmarks when placing features on a cell map.
/// Each feature is compared against every landmark, so this bounds the cost
/// at D × this × H.
const FEATURE_ON_CELL_LANDMARKS: usize = 5000;

fn write_xy(path: &str, names: &[Box<str>], coords: &Mat, row_label: &str) -> anyhow::Result<()> {
    let cols: Vec<Box<str>> = vec!["x".into(), "y".into()];
    coords.to_parquet_with_names(path, (Some(names), Some(row_label)), Some(&cols))?;
    Ok(())
}

/// Place `feature_coembedding` on a finished cell layout.
///
/// `cell_feat_kn` is the (H × N) cell table the layout was computed from and
/// `cell_coords` its (N × 2) result. Returns the written path, or `None` when
/// the run has no co-embed in the same space (topic/SVD runs, interrupted
/// embedding runs, or a layout that ran on a projection instead of Z).
pub(crate) fn place_features_on_cells(
    args: &LayoutCommonArgs,
    manifest: Option<&RunManifest>,
    manifest_path: Option<&PathBuf>,
    out: &str,
    method: &str,
    cell_feat_kn: &Mat,
    cell_coords: &Mat,
) -> anyhow::Result<Option<String>> {
    let (Some(m), Some(mp)) = (manifest, manifest_path) else {
        return Ok(None);
    };
    if m.kind.cell_space() != CellSpace::Embedding {
        return Ok(None);
    }
    let Some(rel) = m.outputs.feature_coembedding.as_deref() else {
        return Ok(None);
    };
    let path = run_manifest::resolve(manifest_dir(mp), rel)
        .to_string_lossy()
        .into_owned();

    let MatWithNames {
        rows: feature_names,
        mat: coembed_dh,
        ..
    } = Mat::from_parquet_with_row_names(&path, Some(0))?;
    if coembed_dh.ncols() != cell_feat_kn.nrows() {
        log::warn!(
            "feature co-embed has {} dims but the cell layout ran on {}; \
             skipping feature placement",
            coembed_dh.ncols(),
            cell_feat_kn.nrows()
        );
        return Ok(None);
    }

    // Landmarks: a seeded subsample of cells with finite coordinates.
    let finite: Vec<usize> = (0..cell_coords.nrows())
        .filter(|&i| cell_coords[(i, 0)].is_finite() && cell_coords[(i, 1)].is_finite())
        .collect();
    if finite.is_empty() {
        return Ok(None);
    }
    let mut rng = SmallRng::seed_from_u64(args.seed);
    let cells: Vec<usize> = rand::seq::index::sample(
        &mut rng,
        finite.len(),
        FEATURE_ON_CELL_LANDMARKS.min(finite.len()),
    )
    .into_iter()
    .map(|i| finite[i])
    .collect();
    let landmark_kp = cell_feat_kn.select_columns(&cells);
    let landmark_xy = cell_coords.select_rows(&cells);

    info!(
        "Placing {} features on the {method} cell map ({} landmark cells)",
        feature_names.len(),
        cells.len()
    );
    let coords = project_cells_nystrom(
        &coembed_dh.transpose(),
        &landmark_kp,
        &landmark_xy,
        args.knn,
        args.kernel_alpha,
    );

    let out_path = format!("{out}.{method}.feature_on_cell_coords.parquet");
    write_xy(&out_path, &feature_names, &coords, "feature")?;
    info!("Saved {out_path}");
    Ok(Some(out_path))
}

/// Input for `--target features`: the feature embedding, one column per
/// feature, L2-normalized so the kNN graph is cosine.
pub(crate) struct FeatureLayoutInput {
    pub manifest: RunManifest,
    pub manifest_path: PathBuf,
    pub manifest_dir: PathBuf,
    pub out: String,
    pub names: Vec<Box<str>>,
    pub feat_kn: Mat,
}

/// A run's feature embedding ρ, one L2-normalized row per feature, so dot
/// products are cosines: what feature layouts, feature clusters and the
/// viewer's nearest features all compare.
pub(crate) fn read_feature_rows(
    manifest: &RunManifest,
    dir: &Path,
) -> anyhow::Result<(Vec<Box<str>>, Mat)> {
    let (path, _bias) = run_manifest::resolve_feature_embedding_for(manifest, dir)?;
    let MatWithNames { rows, mut mat, .. } = Mat::from_parquet_with_row_names(&path, Some(0))?;
    l2_normalize_rows_inplace(&mut mat);
    info!(
        "Feature embedding: {} features × {} dims from {path} (cosine)",
        mat.nrows(),
        mat.ncols()
    );
    Ok((rows, mat))
}

pub(crate) fn load_feature_layout_input(
    args: &LayoutCommonArgs,
) -> anyhow::Result<FeatureLayoutInput> {
    let from = args.from.as_deref().ok_or_else(|| {
        anyhow::anyhow!(
            "--target features needs --from <run.senna.json> to find the feature embedding"
        )
    })?;
    let manifest_path = PathBuf::from(from);
    let (manifest, dir) = RunManifest::load(&manifest_path)?;
    let (names, rows) = read_feature_rows(&manifest, &dir)?;

    let out: String = args
        .out
        .as_deref()
        .map_or_else(|| manifest.prefix.clone(), String::from);
    mkdir_parent(&out)?;

    let feat_kn = rows.transpose();

    Ok(FeatureLayoutInput {
        manifest,
        manifest_path,
        manifest_dir: dir,
        out,
        names,
        feat_kn,
    })
}

/// Write a `--target features` result and record it under
/// `manifest.layout.methods[method].feature_coords`. Cell slots are untouched.
pub(crate) fn write_feature_layout(
    input: &mut FeatureLayoutInput,
    method: &str,
    coords: &Mat,
) -> anyhow::Result<()> {
    let out_path = format!("{}.{method}.feature_coords.parquet", input.out);
    write_xy(&out_path, &input.names, coords, "feature")?;
    info!("Saved {out_path}");

    let rel = rel_to_manifest(&input.manifest_dir, &out_path);
    input
        .manifest
        .layout
        .methods
        .entry(method.to_string())
        .or_default()
        .feature_coords = Some(rel);
    input.manifest.save(&input.manifest_path)
}

/// Record a cell layout under `manifest.layout.methods[method]` and point the
/// top-level slots at it. `written` holds the paths just written (as written;
/// they are made manifest-relative here). A feature layout already recorded
/// for the method is kept.
///
/// `pb_gene_mean` is advertised only from the gene-space recompute path: the
/// fast path writes a projection-space file that `lupin annotate`
/// (enrichment) would misread.
pub(crate) fn record_cell_layout(
    manifest: &mut RunManifest,
    manifest_path: &Path,
    method: &str,
    written: &LayoutEntry,
    pb_gene_mean: Option<&str>,
) -> anyhow::Result<()> {
    let dir = manifest_dir(manifest_path);
    let rel = |p: &Option<String>| p.as_deref().map(|p| rel_to_manifest(dir, p));

    let layout = &mut manifest.layout;
    layout.cell_coords = rel(&written.cell_coords);
    // DirectCells mode emits no pb_coords, so the slot is cleared; callers that
    // need PB-level coords must branch on `kind`.
    layout.pb_coords = rel(&written.pb_coords);
    layout.pb_gene_mean = pb_gene_mean.map(|p| rel_to_manifest(dir, p));
    layout.current = Some(method.to_string());

    let prev = layout.methods.remove(method).unwrap_or_default();
    layout.methods.insert(
        method.to_string(),
        LayoutEntry {
            cell_coords: rel(&written.cell_coords),
            pb_coords: rel(&written.pb_coords),
            feature_on_cell_coords: rel(&written.feature_on_cell_coords),
            feature_coords: prev.feature_coords,
        },
    );
    manifest.save(manifest_path)
}

#[cfg(test)]
mod tests {
    use super::*;
    use senna::run_manifest::RunKind;

    #[test]
    fn rerunning_a_cell_layout_keeps_that_methods_feature_layout() {
        let dir = tempfile::tempdir().unwrap();
        let mp = dir.path().join("r.senna.json");
        // The files must exist: paths are made manifest-relative through
        // `canonicalize`, as they are after a real layout writes them.
        let p = |s: &str| {
            let path = dir.path().join(s);
            std::fs::write(&path, b"").unwrap();
            path.to_string_lossy().into_owned()
        };
        let mut m = RunManifest::new(RunKind::Bge, "r");
        m.layout.methods.insert(
            "umap".into(),
            LayoutEntry {
                feature_coords: Some("r.umap.feature_coords.parquet".into()),
                ..Default::default()
            },
        );

        let written = LayoutEntry {
            cell_coords: Some(p("r.phate.cell_coords.parquet")),
            ..Default::default()
        };
        record_cell_layout(&mut m, &mp, "phate", &written, None).unwrap();
        let written = LayoutEntry {
            cell_coords: Some(p("r.umap.cell_coords.parquet")),
            ..Default::default()
        };
        record_cell_layout(&mut m, &mp, "umap", &written, None).unwrap();

        let umap = &m.layout.methods["umap"];
        assert_eq!(
            umap.cell_coords.as_deref(),
            Some("r.umap.cell_coords.parquet")
        );
        assert_eq!(
            umap.feature_coords.as_deref(),
            Some("r.umap.feature_coords.parquet")
        );
        assert!(m.layout.methods.contains_key("phate"));
        assert_eq!(m.layout.current.as_deref(), Some("umap"));
        assert_eq!(
            m.layout.cell_coords.as_deref(),
            Some("r.umap.cell_coords.parquet")
        );
    }
}
