//! Inheriting a parent run's data and cell → pseudobulk partition through
//! its manifest.

use super::{resolve, Mat, RunKind, RunManifest};
use legume_numeric::matrix::traits::IoOps;
use std::path::Path;

/// `(cell_to_pb_per_level [finest-last], cell_names)` — the raw payload
/// loaded from a `cell_to_pb.parquet`, ready to be aligned to a
/// caller's data axis.
pub type InheritedPartition = (Vec<Vec<usize>>, Vec<Box<str>>);

/// Beside a run's `{prefix}.cell_to_pb.parquet`, the same partition over every
/// cell it trained on: written only when QC held some cells back from the
/// per-cell outputs, which a fit collapsing on the partition still loads.
#[must_use]
pub fn cell_to_pb_all_path(cell_to_pb: &str) -> String {
    match cell_to_pb.strip_suffix(".cell_to_pb.parquet") {
        Some(prefix) => format!("{prefix}.cell_to_pb_all.parquet"),
        None => format!("{cell_to_pb}.all"),
    }
}

/// Read `{prefix}.cell_to_pb.parquet` (N × `num_levels` f32, cell-name
/// rows, `level_0..level_{L-1}` columns) into the same
/// [`InheritedPartition`] shape that
/// [`InheritedFromManifest::load_cell_to_pb`] returns. Exposed so
/// callers that don't hold a full `InheritedFromManifest` (e.g.
/// `senna layout`) can reuse the loader.
pub fn load_cell_to_pb_raw(path: &str) -> anyhow::Result<InheritedPartition> {
    let mat_with_names = Mat::from_parquet_with_row_names(path, Some(0))?;
    let cell_names_src = mat_with_names.rows;
    let mat = mat_with_names.mat;
    let n_src = mat.nrows();
    let num_levels = mat.ncols();
    anyhow::ensure!(
        num_levels >= 1,
        "{path}: cell_to_pb parquet has 0 data columns"
    );
    anyhow::ensure!(
        cell_names_src.len() == n_src,
        "{}: row-name count {} != matrix rows {}",
        path,
        cell_names_src.len(),
        n_src
    );
    // Parquet columns are level_0..level_{L-1} in the writer's order,
    // finest-last as `PreparedData.collapsed_levels` (`write_cell_to_pb`).
    let mut cell_to_pb_per_level: Vec<Vec<usize>> = Vec::with_capacity(num_levels);
    for lvl in 0..num_levels {
        let mut col: Vec<usize> = Vec::with_capacity(n_src);
        for i in 0..n_src {
            col.push(mat[(i, lvl)] as usize);
        }
        cell_to_pb_per_level.push(col);
    }
    log::info!(
        "loaded the inherited cell → pb partition {path} (num_levels={num_levels}, N_src={n_src})",
    );
    Ok((cell_to_pb_per_level, cell_names_src))
}

/// Resolved chain-from inputs that a downstream training subcommand
/// (`senna {masked-topic, cell-embedded-topic}`) inherits via `--from`.
/// All paths are already resolved against the manifest's directory so
/// the caller can hand them straight to data loaders.
pub struct InheritedFromManifest {
    pub data_files: Vec<Box<str>>,
    pub batch_files: Vec<Box<str>>,
    /// `--{freeze,init,lora}-feature-embedding` prefix.
    /// Resolved to the manifest's `prefix` after directory resolution;
    /// the spec resolver downstream will probe `{prefix}.dictionary.parquet`
    /// (bge/fne layout) or `{prefix}.feature_embedding.parquet`
    /// (topic-family layout).
    pub feature_embedding_prefix: Box<str>,
    /// Path to the source run's `{prefix}.cell_to_pb.parquet`
    /// (`outputs.cell_to_pb`), resolved against the manifest dir.
    /// When present, downstream trainers can use the post-refinement
    /// cell→pb membership and skip the BBKNN + Poisson DC-SBM step.
    /// `None` when the source manifest has no `cell_to_pb` output.
    pub cell_to_pb_path: Option<Box<str>>,
    /// Source manifest kind — useful for logging.
    pub source_kind: RunKind,
    /// The multiome layout `data_files` was trained under. Carried beside the
    /// files, not derived separately, so a consumer that inherits the inputs
    /// cannot silently forget how they were namespaced and load the modalities
    /// back as extra cells.
    pub reload: crate::multiome_layout::ReloadLayout,
}

impl InheritedFromManifest {
    /// Refuse to inherit a multiome source into a loader that cannot replay
    /// its namespacing. Without this the modalities come back as extra cells
    /// and every feature name misses the dictionary — silently.
    pub fn ensure_replayable(inherited: Option<&Self>, command: &str) -> anyhow::Result<()> {
        if let Some(inh) = inherited {
            anyhow::ensure!(
                !inh.reload.is_multiome(),
                "--from names a multiome run, and `senna {command}` cannot reproduce its \
                 load: cells there are glued by barcode and features namespaced per \
                 modality. Re-run it on the same files directly, or chain from a \
                 single-modality run."
            );
        }
        Ok(())
    }

    /// Pick the effective input file list: explicit CLI wins; otherwise
    /// the manifest's inherited list. Bails if neither is non-empty so
    /// the data loader gets a clear "no inputs" error instead of an
    /// empty `Vec`.
    pub fn resolve_data(
        inherited: Option<&Self>,
        cli: &[Box<str>],
    ) -> anyhow::Result<Vec<Box<str>>> {
        let out = if cli.is_empty() {
            inherited.map(|i| i.data_files.clone()).unwrap_or_default()
        } else {
            cli.to_vec()
        };
        anyhow::ensure!(
            !out.is_empty(),
            "no input data files — pass at least one positional .zarr/.h5 path or use --from"
        );
        Ok(out)
    }

    /// Pick the effective batch file list: explicit CLI wins; otherwise
    /// the manifest's inherited list when non-empty.
    pub fn resolve_batch(
        inherited: Option<&Self>,
        cli: Option<&[Box<str>]>,
    ) -> Option<Vec<Box<str>>> {
        match (cli, inherited) {
            (Some(b), _) => Some(b.to_vec()),
            (None, Some(i)) if !i.batch_files.is_empty() => Some(i.batch_files.clone()),
            _ => None,
        }
    }

    /// Read the inherited `cell_to_pb.parquet` into raw
    /// `(cell_to_pb_per_level [finest-last][N_src], cell_names_src)`.
    /// `None` when the source manifest had no `cell_to_pb` output.
    /// Caller is expected to call [`Self::align_cell_to_pb_to_cells`]
    /// against its own `data_vec.column_names()` before feeding the
    /// partition into the collapse.
    pub fn load_cell_to_pb(&self) -> anyhow::Result<Option<InheritedPartition>> {
        let Some(path) = self.cell_to_pb_path.as_deref() else {
            return Ok(None);
        };
        Ok(Some(load_cell_to_pb_raw(path)?))
    }

    /// Align a loaded partition to `data_cell_names`: short-circuit
    /// when orders already match; otherwise reorder by cell name and
    /// bail (with a preview of misses) if any data cell is absent
    /// from the source. Each level's inner `Vec` ends up with
    /// `data_cell_names.len()` entries.
    pub fn align_cell_to_pb_to_cells(
        cell_to_pb_per_level_src: Vec<Vec<usize>>,
        cell_names_src: &[Box<str>],
        data_cell_names: &[Box<str>],
    ) -> anyhow::Result<Vec<Vec<usize>>> {
        if cell_names_src == data_cell_names {
            log::info!("inherited partition: cell-name order matches data axis (no reorder)");
            return Ok(cell_to_pb_per_level_src);
        }
        let src_index: rustc_hash::FxHashMap<&str, usize> = cell_names_src
            .iter()
            .enumerate()
            .map(|(i, n)| (n.as_ref(), i))
            .collect();
        let mut missing: Vec<&str> = Vec::new();
        let mut perm: Vec<usize> = Vec::with_capacity(data_cell_names.len());
        for name in data_cell_names {
            match src_index.get(name.as_ref()) {
                Some(&i) => perm.push(i),
                None => missing.push(name.as_ref()),
            }
        }
        if !missing.is_empty() {
            let preview: Vec<&str> = missing.iter().copied().take(5).collect();
            anyhow::bail!(
                "{} of {} data cells are absent from the inherited cell → pb partition \
                 (e.g. {:?}); the source run (--from / --pb-from) was trained on a \
                 different cell set",
                missing.len(),
                data_cell_names.len(),
                preview,
            );
        }
        let n_data = data_cell_names.len();
        let mut out: Vec<Vec<usize>> = Vec::with_capacity(cell_to_pb_per_level_src.len());
        for lvl in cell_to_pb_per_level_src {
            let mut col: Vec<usize> = Vec::with_capacity(n_data);
            for &src_i in &perm {
                col.push(lvl[src_i]);
            }
            out.push(col);
        }
        log::info!(
            "inherited partition: reordered by cell name ({}→{} cells aligned)",
            cell_names_src.len(),
            n_data,
        );
        Ok(out)
    }
}

/// The recorded settings in which `here` differs from `there` (two
/// serialized argument structs, or parts of them), as `path: there →
/// here`. A setting `there` does not record (an older senna's) is passed;
/// numbers are compared as the f32 most settings are, since a JSON round
/// trip need not give back the same f64.
#[must_use]
pub fn settings_differences(there: &serde_json::Value, here: &serde_json::Value) -> Vec<String> {
    fn walk(there: &serde_json::Value, here: &serde_json::Value, path: &str) -> Vec<String> {
        use serde_json::Value;
        match (there, here) {
            (Value::Object(a), Value::Object(b)) => b
                .iter()
                .filter_map(|(k, v)| a.get(k).map(|w| (k, w, v)))
                .flat_map(|(k, w, v)| {
                    let p = if path.is_empty() {
                        k.clone()
                    } else {
                        format!("{path}.{k}")
                    };
                    walk(w, v, &p)
                })
                .collect(),
            (Value::Number(a), Value::Number(b))
                if a.as_f64().map(|x| x as f32) == b.as_f64().map(|x| x as f32) =>
            {
                Vec::new()
            }
            _ if there == here => Vec::new(),
            _ => vec![format!("{path}: {there} there, {here} here")],
        }
    }
    walk(there, here, "")
}

/// Load a `senna.json` manifest and extract the fields a downstream
/// trainer would inherit. Bails for source kinds that don't write a
/// feature embedding (SVD / joint-SVD); accepts bge, fne, and the
/// topic-family kinds.
pub fn inherit_from(manifest_path: &str) -> anyhow::Result<InheritedFromManifest> {
    let (m, dir) = RunManifest::load(Path::new(manifest_path))?;
    match m.kind {
        RunKind::Bge
        | RunKind::Fne
        | RunKind::ResolveEmbeddingSpace
        | RunKind::Topic
        | RunKind::Itopic
        | RunKind::MaskedVae
        | RunKind::JointTopic
        | RunKind::Vae
        | RunKind::Simba
        | RunKind::Rbm => {}
        RunKind::Svd | RunKind::JointSvd | RunKind::Ode => anyhow::bail!(
            "--from manifest kind '{}' has no feature embedding to inherit; \
             use a bge / fne / topic-family run as the source",
            m.kind
        ),
    }
    let dir = dir.as_path();
    let to_box = |s: &str| -> Box<str> { resolve(dir, s).to_string_lossy().into_owned().into() };
    let data_files = m.data_inputs(dir);
    let batch_files = m.data_batches(dir);
    let feature_embedding_prefix: Box<str> = to_box(&m.prefix);
    let cell_to_pb_path: Option<Box<str>> = m.cell_to_pb_path(dir).map(Into::into);
    let reload =
        crate::multiome_layout::recorded_layout(m.data.multiome.as_ref(), m.data.input.len())?;
    Ok(InheritedFromManifest {
        data_files,
        batch_files,
        feature_embedding_prefix,
        cell_to_pb_path,
        source_kind: m.kind,
        reload,
    })
}
