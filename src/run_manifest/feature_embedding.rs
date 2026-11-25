//! Locating a run's per-gene embedding ρ (and its bias and feature types)
//! from a prefix.

use super::{default_path, resolve, RunManifest};
use std::path::{Path, PathBuf};

/// The suffix of a run's ρ table under its prefix.
pub const RHO_TABLE_SUFFIX: &str = ".feature_embedding.parquet";

/// Locate a run's per-gene embedding `ρ` from its `--out` prefix (or its ρ
/// table or manifest; see below).
///
/// **The single place that knows where ρ can live.** Consumers used to probe for
/// it independently — `FrozenFeatureSpec` by filename, `deconvolve` by manifest
/// slot, `annotate` by a fixed field — and every layout change broke each of
/// them differently. Three separate bugs came out of that.
///
/// The run's manifest, when there is one, decides first: the ρ it records
/// stands while that file exists. Otherwise `{prefix}.feature_embedding.parquet`
/// is looked for under the prefix as given (else the one the manifest records,
/// for a renamed manifest).
///
/// `prefix` may also name the table itself or the run's manifest, so a path
/// pasted from a listing works as well as the stem. A named table must be the
/// one its run resolves to, when the run has a manifest.
///
/// Returns `(rho_path, bias_path)`; the bias is `None` when absent (callers
/// default it to zero).
pub fn resolve_feature_embedding(prefix: &str) -> anyhow::Result<(String, Option<String>)> {
    if let Some(stem) = strip_table_suffix(prefix) {
        anyhow::ensure!(
            Path::new(prefix).exists(),
            "no per-gene embedding ρ at `{prefix}`: the file does not exist"
        );
        let manifest = default_path(stem);
        if let Some((m, dir)) = readable_manifest(&manifest) {
            let (rho, bias) = rho_of_run(&m, &dir, Some(stem))?;
            anyhow::ensure!(
                same_file(Path::new(&rho), Path::new(prefix)),
                "`{prefix}` is not the run's ρ: by {manifest} it is `{rho}`"
            );
            // The caller's spelling of the same file.
            return Ok((prefix.to_string(), bias));
        }
        return Ok((prefix.to_string(), bias_beside(stem)));
    }
    let stem = prefix.strip_suffix(".senna.json").unwrap_or(prefix);
    match readable_manifest(&default_path(stem)) {
        Some((m, dir)) => rho_of_run(&m, &dir, Some(stem)),
        None => probe_feature_embedding(stem),
    }
}

/// The manifest at `path`, when there is one this binary reads; one it
/// cannot is warned about and left out, as if absent.
fn readable_manifest(path: &str) -> Option<(RunManifest, PathBuf)> {
    if !Path::new(path).is_file() {
        return None;
    }
    RunManifest::load(Path::new(path))
        .map_err(|e| log::warn!("{e}; going by the tables beside it"))
        .ok()
}

/// ρ of the run `m` describes: the table it records while that exists,
/// else probed under `given` (the prefix the caller named), then under the
/// manifest's own prefix. See [`resolve_feature_embedding`].
fn rho_of_run(
    m: &RunManifest,
    manifest_dir: &Path,
    given: Option<&str>,
) -> anyhow::Result<(String, Option<String>)> {
    if let Some(rel) = m.outputs.feature_embedding.as_deref() {
        let rho = resolve(manifest_dir, rel).to_string_lossy().into_owned();
        if Path::new(&rho).exists() {
            let bias = strip_table_suffix(&rho).and_then(bias_beside);
            return Ok((rho, bias));
        }
    }
    let own = resolve(manifest_dir, &m.prefix)
        .to_string_lossy()
        .into_owned();
    match given {
        Some(g) if g != own => {
            probe_feature_embedding(g).or_else(|e| probe_feature_embedding(&own).map_err(|_| e))
        }
        _ => probe_feature_embedding(&own),
    }
}

/// Whether `a` and `b` are one existing file, however each is spelled.
#[must_use]
pub fn same_file(a: &Path, b: &Path) -> bool {
    if a == b {
        return a.exists();
    }
    match (a.canonicalize(), b.canonicalize()) {
        (Ok(x), Ok(y)) => x == y,
        _ => false,
    }
}

/// `{prefix}.feature_embedding.parquet`, when it exists (see
/// [`resolve_feature_embedding`]).
fn probe_feature_embedding(prefix: &str) -> anyhow::Result<(String, Option<String>)> {
    let rho = format!("{prefix}{RHO_TABLE_SUFFIX}");
    anyhow::ensure!(
        Path::new(&rho).exists(),
        "no per-gene embedding ρ found for prefix `{prefix}` — looked for {rho}"
    );
    Ok((rho, bias_beside(prefix)))
}

/// The types table written beside the run whose ρ is at `rho` (as resolved
/// by [`resolve_feature_embedding`]), when it wrote one. Like the bias it
/// sits under the table's stem, which a flag naming the table or the
/// manifest is not. Whether it describes `rho`'s rows is checked against
/// them once read ([`crate::carried_rows::matchable_rows`]), by every reader.
pub fn feature_types_beside(
    rho: &str,
) -> anyhow::Result<Option<Vec<data_beans::aux::feature_types::FeatureType>>> {
    let stem = strip_table_suffix(rho).unwrap_or(rho);
    data_beans::aux::feature_types::read_feature_types(stem)
}

/// `path`'s stem, when it ends with [`RHO_TABLE_SUFFIX`].
fn strip_table_suffix(path: &str) -> Option<&str> {
    path.strip_suffix(RHO_TABLE_SUFFIX)
}

/// `{stem}.feature_bias.parquet`, when it exists.
fn bias_beside(stem: &str) -> Option<String> {
    let bias = format!("{stem}.feature_bias.parquet");
    Path::new(&bias).exists().then_some(bias)
}

/// Resolve `ρ` (and its bias) for a run that has a manifest in hand.
///
/// The manifest's `outputs.feature_embedding` is authoritative when present —
/// a recorded path beats probing. A manifest with no ρ slot, or one whose recorded table is gone, falls through to probing the
/// run prefix, as [`resolve_feature_embedding`] describes.
///
/// This is the entry point for manifest-holding consumers;
/// [`resolve_feature_embedding`] is the one for callers handed a path (a
/// prefix, ρ table or manifest), such as `--freeze-feature-embedding`.
pub fn resolve_feature_embedding_for(
    m: &RunManifest,
    manifest_dir: &Path,
) -> anyhow::Result<(String, Option<String>)> {
    rho_of_run(m, manifest_dir, None)
}
