//! Writing a fresh manifest from a training run's [`RunDescription`].

use super::{default_path, rel_to_manifest, RunKind, RunManifest, TrainArgsRecord};
use std::path::Path;

/// Per-run description assembled by each training subcommand, handed to
/// `write_run_manifest` which owns the `RunManifest` / `save` plumbing.
pub struct RunDescription<'a> {
    pub kind: RunKind,
    /// The `--out` prefix; used both for the manifest filename and as
    /// the `prefix` field inside.
    pub prefix: &'a str,
    /// The fit's own arguments, from [`record_train_args`](super::record_train_args). `Some` for the
    /// families `senna update` can continue (topic, masked-*, svd, vae);
    /// `None` for runs with no re-runnable fit of their own.
    pub train_args: Option<TrainArgsRecord>,
    pub data_input: &'a [String],
    pub data_batch: &'a [String],
    pub data_input_null: &'a [String],
    /// The multiome layout `data_input` was loaded under, when there was one.
    pub data_multiome: Option<crate::multiome_layout::RunMultiome>,
    /// Suffix after `{basename}.` for the dictionary parquet, e.g.
    /// `"dictionary.parquet"` or (joint-topic) `"base_dictionary.parquet"`.
    /// `None` to omit — SVD runs still produce one, topic runs always do.
    pub dictionary_suffix: Option<&'a str>,
    /// True if the run emits `{basename}.safetensors` +
    /// `{basename}.model.json` (topic + masked-topic; not joint-topic, not
    /// SVD).
    pub has_model: bool,
    /// True if the run emits `{basename}.cell_proj.parquet` — the
    /// cached per-cell random projection layout reuses. All training
    /// subcommands that produce PBs (topic, masked-topic, joint-topic, svd,
    /// joint-svd) should set this.
    pub has_cell_proj: bool,
    /// Suffix after `{basename}.` for the PB-level gene aggregates parquet,
    /// e.g. `"pb_gene.parquet"`. `None` to omit.
    pub pb_gene_suffix: Option<&'a str>,
    /// Suffix after `{basename}.` for the PB-level latent parquet, e.g.
    /// `"pb_latent.parquet"`. `None` to omit.
    pub pb_latent_suffix: Option<&'a str>,
    /// Suffix after `{basename}.` for the carried pseudobulk backend, e.g.
    /// `"pb_reference.zarr.zip"`. `None` unless the run emitted them.
    pub pb_reference_suffix: Option<&'a str>,
    /// Suffix after `{basename}.` for the empirical NB-Fisher-weighted
    /// dictionary parquet, e.g. `"dictionary_empirical.parquet"`. `None`
    /// to omit.
    pub dictionary_empirical_suffix: Option<&'a str>,
    /// Suffix after `{basename}.` for the RAW model-axis per-gene embedding
    /// ρ parquet, e.g. `"feature_embedding.parquet"`. `None` to omit.
    pub feature_embedding_suffix: Option<&'a str>,
    /// Suffix after `{basename}.` for the SIMBA co-embed, e.g.
    /// `"feature_coembedding.parquet"`. `None` when the run did not co-embed.
    /// See [`RunOutputs::feature_coembedding`](super::RunOutputs::feature_coembedding) for why this is separate from
    /// `feature_embedding_suffix`.
    pub feature_coembedding_suffix: Option<&'a str>,
    /// The given feature table's rows that matched no feature of this run,
    /// to append to the ρ file (`feature_embedding_suffix`) before the
    /// manifest is written — so every engine's carry-through is this one
    /// field.
    pub carried: Option<&'a crate::carried_rows::CarriedRows>,
    /// e.g. `"module_membership.parquet"` for a gene-module run; `None` to omit.
    pub module_membership_suffix: Option<&'a str>,
    /// e.g. `"module_dictionary.parquet"`; paired with the membership.
    pub module_dictionary_suffix: Option<&'a str>,
    /// Suffix after `{basename}.` for the log-simplex topic dictionary β, e.g.
    /// `"dictionary.parquet"`. Set by kinds whose dictionary IS a
    /// `log_softmax`-over-genes simplex; `None` for signed loadings, which stay
    /// in `dictionary_suffix`. The two may name the SAME file — the point is
    /// that the manifest slot states the scale.
    pub softmax_dictionary_suffix: Option<&'a str>,
    /// Suffix after `{basename}.` for the H-space per-cell embedding Z
    /// parquet, e.g. `"cell_embedding.parquet"`. Set by every embedding
    /// command (`bge`, `fne`, `resolve-embedding-space`) — Z always lands
    /// here, never in `latent`. `None` to omit.
    pub cell_embedding_suffix: Option<&'a str>,
    /// Suffix after `{basename}.` for the cell encoder, e.g.
    /// `"cell_encoder.safetensors"`. `None` to omit.
    pub cell_encoder_suffix: Option<&'a str>,
    /// Default `--colour-by` for downstream plot / layout.
    pub default_colour_by: &'a str,
    /// True if the run emits `{basename}.latent.parquet`. Topic-family fits
    /// set this for log θ; SVD kinds for component scores. Embedding commands
    /// set it ONLY when they also resolved topics (`bge` without `--skip-etm`)
    /// — their Z goes to `cell_embedding_suffix` instead.
    pub has_latent: bool,
    /// True if the run emits `{basename}.cell_to_pb.parquet` — the
    /// post-refinement cell→pseudobulk membership per coarsening level.
    /// Set by topic-family fits that ran `collapse_columns_multilevel_*`
    /// so a downstream `--from` chain can skip the refinement step.
    pub has_cell_to_pb: bool,
    /// True if the run emits `{basename}.pb_tree.json` — see
    /// [`RunOutputs::pb_tree`](super::RunOutputs::pb_tree).
    pub has_pb_tree: bool,
}

/// Write `{prefix}.senna.json` describing the run that just finished.
///
/// All artifact paths inside the manifest are stored as *basenames*
/// (e.g. `"run1.latent.parquet"`) so they resolve correctly relative to
/// the manifest's own directory — even when the run directory is moved
/// after writing.
pub fn write_run_manifest(desc: &RunDescription<'_>) -> anyhow::Result<()> {
    write_run_manifest_with(desc, |_, _| {})
}

/// [`write_run_manifest`], with `edit` applied before the one write: for a
/// run that records more than [`RunDescription`] says, so its manifest is
/// never there without it. `edit` is given the outputs' base name.
pub fn write_run_manifest_with(
    desc: &RunDescription<'_>,
    edit: impl FnOnce(&mut RunManifest, &str),
) -> anyhow::Result<()> {
    match (desc.carried, desc.feature_embedding_suffix) {
        (Some(c), Some(suffix)) => c.append_to(desc.prefix, suffix)?,
        (Some(_), None) => anyhow::bail!("{}: no ρ file to carry rows into", desc.prefix),
        (None, Some(suffix)) => crate::carried_rows::clear_stale_types(
            desc.prefix,
            &format!("{}.{suffix}", desc.prefix),
        )?,
        (None, None) => {}
    }
    let basename = Path::new(desc.prefix).file_name().map_or_else(
        || desc.prefix.to_string(),
        |s| s.to_string_lossy().into_owned(),
    );

    // Paths on the command line are relative to the working directory, but a
    // reader of `data.*` resolves them against the manifest's directory, so
    // they are stored manifest-relative. `prefix` is stored absolute: some
    // readers resolve it and others use it as given, as a default `--out`.
    let path = default_path(desc.prefix);
    let prefix = std::path::absolute(desc.prefix).map_or_else(
        |_| desc.prefix.to_string(),
        |p| p.to_string_lossy().into_owned(),
    );
    let manifest_dir = Path::new(&path).parent().unwrap_or(Path::new(""));
    let recorded = |paths: &[String]| -> Vec<String> {
        paths
            .iter()
            .map(|p| rel_to_manifest(manifest_dir, p))
            .collect()
    };

    let mut m = RunManifest::new(desc.kind, &prefix);
    m.data.input = recorded(desc.data_input);
    m.data.input_null = recorded(desc.data_input_null);
    m.data.batch = recorded(desc.data_batch);
    m.data.multiome = desc.data_multiome.clone();
    m.train_args = desc.train_args.clone();

    if desc.has_latent {
        m.outputs.latent = Some(format!("{basename}.latent.parquet"));
    }
    if let Some(suf) = desc.dictionary_suffix {
        m.outputs.dictionary = Some(format!("{basename}.{suf}"));
    }
    if desc.has_model {
        m.outputs.model = Some(format!("{basename}.safetensors"));
        m.outputs.metadata = Some(format!("{basename}.model.json"));
    }
    if desc.has_cell_proj {
        m.outputs.cell_proj = Some(format!("{basename}.cell_proj.parquet"));
    }
    if let Some(suf) = desc.pb_gene_suffix {
        m.outputs.pb_gene = Some(format!("{basename}.{suf}"));
    }
    if let Some(suf) = desc.pb_latent_suffix {
        m.outputs.pb_latent = Some(format!("{basename}.{suf}"));
    }
    if let Some(suf) = desc.pb_reference_suffix {
        m.outputs.pb_reference = Some(format!("{basename}.{suf}"));
    }
    if let Some(suf) = desc.dictionary_empirical_suffix {
        m.outputs.dictionary_empirical = Some(format!("{basename}.{suf}"));
    }
    if let Some(suf) = desc.feature_embedding_suffix {
        m.outputs.feature_embedding = Some(format!("{basename}.{suf}"));
    }
    if let Some(suf) = desc.feature_coembedding_suffix {
        m.outputs.feature_coembedding = Some(format!("{basename}.{suf}"));
    }
    if let Some(suf) = desc.module_membership_suffix {
        m.outputs.module_membership = Some(format!("{basename}.{suf}"));
    }
    if let Some(suf) = desc.module_dictionary_suffix {
        m.outputs.module_dictionary = Some(format!("{basename}.{suf}"));
    }
    if let Some(suf) = desc.softmax_dictionary_suffix {
        m.outputs.softmax_dictionary = Some(format!("{basename}.{suf}"));
    }
    if let Some(suf) = desc.cell_encoder_suffix {
        m.outputs.cell_encoder = Some(format!("{basename}.{suf}"));
    }
    if let Some(suf) = desc.cell_embedding_suffix {
        m.outputs.cell_embedding = Some(format!("{basename}.{suf}"));
    }
    if desc.has_cell_to_pb {
        m.outputs.cell_to_pb = Some(format!("{basename}.cell_to_pb.parquet"));
    }
    if desc.has_pb_tree {
        m.outputs.pb_tree = Some(format!("{basename}.pb_tree.json"));
    }
    m.defaults.colour_by = Some(desc.default_colour_by.into());
    edit(&mut m, &basename);

    m.save(Path::new(&path))
}
