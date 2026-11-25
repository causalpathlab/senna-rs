//! Run manifest — the single JSON artifact that ties a senna run
//! together across subcommands.
//!
//! Shape: `senna topic` / `masked-topic` / `joint-topic` write a fresh
//! manifest at the end of training. `senna layout` reads it, produces 2D
//! coords, and updates the `layout{}` section in place. `lupin plot` (and
//! future postprocess commands) read the fully-enriched manifest and
//! work with zero further flags. CLI flags on those commands stay
//! available and win over manifest values when both are supplied.
//!
//! The schema is deliberately narrow — data paths + output artifact
//! paths + a couple of UI defaults. It is a run descriptor, not a config
//! language, and nothing here should grow into one.
//!
//! The one exception is [`RunManifest::train_args`], added for `senna update`,
//! which continues a run's training and therefore needs the parent's exact fit
//! configuration to reproduce its architecture. It is stored as an **opaque**
//! [`serde_json::Value`] precisely to preserve the rule above: no command other
//! than `update` parses it, so the manifest stays readable by every version
//! regardless of which flags a fit command has gained. Shell history and
//! Makefiles remain the answer for re-running by hand.
//!
//! All path values are resolved relative to the manifest file's own
//! directory, so a run directory can be moved or copied without
//! breaking downstream reads.
//!
//! # Output conventions: axis, and SCALE
//!
//! A slot name fixes the axis and shape. It does **not** fix the numeric scale,
//! and scale is where the bugs live — signed vs log-simplex vs probability-simplex
//! all look alike in a `D × K` float table, so a misread produces `NaN` or a
//! degenerate model with no shape mismatch to catch it. Classify with
//! [`ArtifactScale::detect`], or assert with [`ArtifactScale::ensure`].
//!
//! | slot | shape | scale | holds |
//! |---|---|---|---|
//! | `latent` | N × K | log-simplex ROWS (topic) / signed (SVD, masked-vae) | per-cell `log θ` or component scores — gate on [`RunKind::latent_is_log_simplex`] |
//! | `cell_embedding` | N × H | signed | per-cell embedding `Z` |
//! | `feature_embedding` | D × H | signed | per-gene embedding `ρ` — MULTIPLIES `Z` in the log-rate `ρ_g·z_n + a_g + b_n` |
//! | `feature_coembedding` | D × H | signed | where each gene SITS, re-projected onto the cell manifold (SIMBA co-embed) |
//! | `softmax_dictionary` | D × K | **log**-simplex COLUMNS | topic dictionary `β` — `Σ_g exp(β[g,k]) = 1` |
//! | `dictionary` | D × K | signed | SVD component loadings; also where pre-split manifests land |
//! | `dictionary_empirical` | D × K | probability-simplex COLUMNS | empirical `β`, full gene resolution |
//! | `topic_embedding` | K × H | signed | per-topic embedding `α` |
//! | `feature_bias` / `cell_bias` | D / N | signed (log-scale offsets) | `a_g` / `b_n` |
//!
//! Read a dictionary through [`RunOutputs::gene_dictionary`] when either form
//! will do; branch on `kind` (or detect) when the scale matters.
//!
//! `feature_embedding` and `feature_coembedding` are NOT interchangeable: the
//! co-embed is a lossy derived view of ρ (a convex combination of cell
//! embeddings), so ρ → co-embed is one-way.

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

mod feature_embedding;
mod inherit;
mod paths;
mod run_kind;
mod sections;
mod write;

pub use feature_embedding::{
    feature_types_beside, resolve_feature_embedding, resolve_feature_embedding_for, same_file,
    RHO_TABLE_SUFFIX,
};
pub use inherit::{
    cell_to_pb_all_path, inherit_from, load_cell_to_pb_raw, settings_differences,
    InheritedFromManifest, InheritedPartition,
};
#[cfg(test)]
pub use paths::write_kind_only;
pub use paths::{
    canonical_so_far, default_path, derive_out_prefix, from_cwd, load_for, manifest_dir,
    manifest_path_for, normalize, out_prefix, rebase, rel_to_manifest, relative_to, resolve,
};
pub use run_kind::{CellSpace, ExpressionModel, RunKind};
pub use sections::{
    record_train_args, ArtifactScale, Decision, DecisionOption, LayoutEntry, RunAnnotate,
    RunCluster, RunData, RunDefaults, RunHistory, RunLayout, RunOutputs, TrainArgsRecord,
};
pub use write::{write_run_manifest, write_run_manifest_with, RunDescription};

type Mat = nalgebra::DMatrix<f32>;
/// Manifest keys outside this version's schema, carried through verbatim.
pub type Unknown = BTreeMap<String, serde_json::Value>;

/// Schema version. Bump only on breaking renames or semantic changes.
/// Readers accept any version and log a warning for newer-than-known.
///
/// - v2: `feature_embedding` is ρ for every command and the SIMBA co-embed
///   has its own slot, `feature_coembedding`.
pub const MANIFEST_VERSION: u32 = 2;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunManifest {
    pub version: u32,
    pub kind: RunKind,
    /// The `--out` prefix the training command was run with.
    pub prefix: String,
    #[serde(default)]
    pub data: RunData,
    #[serde(default)]
    pub outputs: RunOutputs,
    #[serde(default)]
    pub layout: RunLayout,
    #[serde(default)]
    pub cluster: RunCluster,
    #[serde(default)]
    pub annotate: RunAnnotate,
    #[serde(default)]
    pub defaults: RunDefaults,
    /// The fit configuration this run was trained with — see
    /// [`TrainArgsRecord`]. Absent for runs written before it existed, and for
    /// commands that produce no re-runnable fit.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub train_args: Option<TrainArgsRecord>,
    /// What a round of `senna update` continued from and trained against
    /// (a model's chain of versions, one per round). Absent on a fit that is
    /// no round.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub history: Option<RunHistory>,
    /// What the fit could not settle from its data and leaves to its user
    /// (or to an agent, to ask its user), each with the reruns that settle
    /// it; empty when the data settled everything.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub decisions: Vec<Decision>,
    /// The fit's objective at its mode (its log posterior, up to a
    /// constant): comparable between fits of the same data, genes and
    /// settings, as reruns with another answer to a decision are.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub objective: Option<f64>,
    /// Fields this version does not know, written by another tool or a
    /// newer senna; kept so a load/save round trip never drops them.
    #[serde(flatten, default)]
    pub unknown: Unknown,
}

impl RunManifest {
    pub fn new(kind: RunKind, prefix: &str) -> Self {
        Self {
            version: MANIFEST_VERSION,
            kind,
            prefix: prefix.into(),
            data: RunData::default(),
            outputs: RunOutputs::default(),
            layout: RunLayout::default(),
            cluster: RunCluster::default(),
            annotate: RunAnnotate::default(),
            defaults: RunDefaults::default(),
            train_args: None,
            history: None,
            decisions: Vec::new(),
            objective: None,
            unknown: Unknown::default(),
        }
    }

    /// The run's input files (`data.input`), found from `manifest_dir` (see
    /// [`Self::data_file`]).
    #[must_use]
    pub fn data_inputs(&self, manifest_dir: &Path) -> Vec<Box<str>> {
        self.data_files(manifest_dir, &self.data.input)
    }

    /// The run's batch files (`data.batch`), found as [`Self::data_inputs`].
    #[must_use]
    pub fn data_batches(&self, manifest_dir: &Path) -> Vec<Box<str>> {
        self.data_files(manifest_dir, &self.data.batch)
    }

    fn data_files(&self, manifest_dir: &Path, recorded: &[String]) -> Vec<Box<str>> {
        recorded
            .iter()
            .map(|p| {
                self.data_file(manifest_dir, p)
                    .to_string_lossy()
                    .into_owned()
                    .into_boxed_str()
            })
            .collect()
    }

    /// A data file the run recorded, found here even when the run was
    /// trained on another machine or the tree has moved: as recorded
    /// ([`resolve`]); else at the same place relative to the run's
    /// directory as it was relative to the training `prefix`'s; else under
    /// the manifest's directory or an ancestor of it, by the longest tail of
    /// the recorded path that exists there. Unfound, as recorded. The same
    /// rule as lupin's, so both find a moved run's data alike.
    #[must_use]
    pub fn data_file(&self, manifest_dir: &Path, recorded: &str) -> PathBuf {
        let given = resolve(manifest_dir, recorded);
        if given.exists() {
            return given;
        }
        // Lexically absolute, not canonical: the recorded paths are as
        // written, so resolving symlinks here alone would skew the rebase.
        let here = normalize(&from_cwd(manifest_dir));
        let recorded_abs = normalize(Path::new(recorded));
        let trained_in = Path::new(&self.prefix)
            .is_absolute()
            .then(|| normalize(Path::new(&self.prefix)))
            .and_then(|p| p.parent().map(Path::to_path_buf));
        let rebased = trained_in
            .filter(|_| recorded_abs.is_absolute())
            .map(|old| rebase(&recorded_abs, &old, &here))
            .filter(|p| p.exists());
        let found = rebased.or_else(|| {
            let parts: Vec<_> = recorded_abs
                .components()
                .filter(|c| matches!(c, std::path::Component::Normal(_)))
                .collect();
            // At least the file and its folder must match, so an unrelated
            // file of the same name elsewhere is not taken for the data.
            let min = parts.len().min(2);
            (0..=parts.len().checked_sub(min.max(1))?).find_map(|skip| {
                let tail: PathBuf = parts[skip..].iter().collect();
                here.ancestors().map(|a| a.join(&tail)).find(|p| p.exists())
            })
        });
        match found {
            Some(p) => {
                log::warn!("{recorded} is not here; using {}", p.display());
                p
            }
            None => given,
        }
    }

    /// Point data file `index` at `chosen`, and move each other data, null
    /// or batch file that is not here the way that one moved: to the same
    /// place relative to `chosen` as it was recorded relative to the file it
    /// replaces, if a file is there. Paths are recorded relative to
    /// `manifest_dir`. In memory only; returns how many others moved.
    pub fn relocate_data(
        &mut self,
        manifest_dir: &Path,
        index: usize,
        chosen: &Path,
    ) -> anyhow::Result<usize> {
        let Some(old) = self.data.input.get(index) else {
            anyhow::bail!("the manifest lists no data file {}", index + 1);
        };
        let old_dir = resolve(manifest_dir, old)
            .parent()
            .map(Path::to_path_buf)
            .unwrap_or_default();
        let new_dir = chosen.parent().unwrap_or(Path::new("."));
        let record = |p: &Path| rel_to_manifest(manifest_dir, &p.to_string_lossy());
        self.data.input[index] = record(chosen);
        let before = self.clone();
        let mut also = 0;
        let data = &mut self.data;
        for list in [&mut data.input, &mut data.input_null, &mut data.batch] {
            for rec in list.iter_mut() {
                if before.data_file(manifest_dir, rec).exists() {
                    continue;
                }
                let there = rebase(&resolve(manifest_dir, rec), &old_dir, new_dir);
                if there.exists() {
                    *rec = record(&there);
                    also += 1;
                }
            }
        }
        Ok(also)
    }

    /// Read the manifest and return it together with its parent
    /// directory (used to resolve the relative paths inside).
    pub fn load(path: &Path) -> anyhow::Result<(Self, PathBuf)> {
        let raw = fs::read_to_string(path)
            .map_err(|e| anyhow::anyhow!("read {}: {e}", path.display()))?;
        let m: Self = serde_json::from_str(&raw).map_err(|e| {
            // A kind this senna does not know: a retired command's, or a
            // newer senna's.
            let unknown = serde_json::from_str::<serde_json::Value>(&raw)
                .ok()
                .and_then(|v| v.get("kind").cloned())
                .filter(|k| serde_json::from_value::<RunKind>(k.clone()).is_err());
            match unknown.as_ref().and_then(serde_json::Value::as_str) {
                Some(kind) => anyhow::anyhow!(
                    "{} is a `senna {kind}` run, a kind this senna does not open \
                     (retired, or from a newer senna)",
                    path.display()
                ),
                None => anyhow::anyhow!("parse {}: {e}", path.display()),
            }
        })?;
        if m.version > MANIFEST_VERSION {
            log::warn!(
                "manifest {} is v{} but this binary supports up to v{MANIFEST_VERSION}; \
                 proceeding (unknown fields will be ignored)",
                path.display(),
                m.version
            );
        }
        let dir = manifest_dir(path).to_path_buf();
        Ok((m, dir))
    }

    /// The run's cell → pb partition, resolved against `manifest_dir`: of
    /// every cell it trained on, when it wrote that beside its per-cell
    /// table (QC held cells back from the outputs).
    #[must_use]
    pub fn cell_to_pb_path(&self, manifest_dir: &Path) -> Option<String> {
        let full = resolve(manifest_dir, self.outputs.cell_to_pb.as_deref()?)
            .to_string_lossy()
            .into_owned();
        let all = cell_to_pb_all_path(&full);
        Some(if Path::new(&all).is_file() { all } else { full })
    }

    /// Write the manifest to `path`, through a link, and on to the disk
    /// before returning: the file keeps its owner, mode and links, and a
    /// read-only one is refused. Each run writes its manifest last, after
    /// every table it names; a run killed while writing it can leave it
    /// cut short, which a reader then fails to parse.
    pub fn save(&self, path: &Path) -> anyhow::Result<()> {
        use std::io::Write;
        let s = serde_json::to_string_pretty(self)?;
        let written = fs::File::create(path).and_then(|mut f| {
            f.write_all(s.as_bytes())?;
            f.sync_all()
        });
        written.map_err(|e| anyhow::anyhow!("write {}: {e}", path.display()))?;
        log::info!("wrote {}", path.display());
        Ok(())
    }

    /// Rebuild the fit's argument struct from [`RunManifest::train_args`].
    ///
    /// `T` must be the arg struct matching [`RunManifest::kind`]; the caller
    /// establishes that by dispatching on the kind first. `prefix` only names
    /// the run in error messages.
    ///
    /// A field absent from the record takes clap's declared default, so a
    /// senna that has gained a flag can still replay an older fit. What does
    /// fail is a record that cannot be read at all — a renamed or retyped
    /// field. This is the *only* reader of the blob, so such a failure never
    /// blocks opening the run for anything else.
    pub fn train_args_as<T: serde::de::DeserializeOwned>(&self, prefix: &str) -> anyhow::Result<T> {
        let rec = self.train_args.as_ref().ok_or_else(|| {
            anyhow::anyhow!(
                "{prefix}: this run's manifest records no fit configuration. It predates \
                 `train_args`, so its settings are not recoverable — re-train it, or drive the \
                 family command directly with `--init-from {prefix}`."
            )
        })?;
        // Borrow the blob rather than cloning it: serde_json implements
        // `Deserializer` for `&Value`.
        T::deserialize(&rec.args).map_err(|e| {
            anyhow::anyhow!(
                "{prefix}: cannot replay the recorded fit configuration ({e}).\n\
                 It was written by senna {} and this is senna {}. A setting that merely went \
                 missing would take its declared default, so this is a setting that changed \
                 shape — renamed, or retyped. Re-train the parent, or drive the family command \
                 directly with `--init-from {prefix}`.",
                rec.senna_version,
                env!("CARGO_PKG_VERSION"),
            )
        })
    }
}

#[cfg(test)]
#[path = "run_manifest_tests.rs"]
mod tests;

#[cfg(test)]
#[path = "run_manifest_save_tests.rs"]
mod save_tests;

#[cfg(test)]
#[path = "run_manifest_path_tests.rs"]
mod path_tests;
