//! The manifest's sections: history, data, outputs (and their scale),
//! layout, cluster, annotate, stage 2, defaults.

use super::{Mat, Unknown};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// A choice a fit left open ([`RunManifest::decisions`](super::RunManifest::decisions)): the question, and
/// every answer as the arguments to add to the fit's command to rerun it
/// with that answer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Decision {
    /// A short name for what is to be decided, e.g. `root`.
    pub topic: String,
    /// The question, put to a person.
    pub question: String,
    /// How to judge between the options, for whoever puts the question
    /// (an agent, say); empty when there is nothing to add.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub hint: String,
    pub options: Vec<DecisionOption>,
}

/// One answer to a [`Decision`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecisionOption {
    /// The answer, put to a person.
    pub label: String,
    /// The arguments that rerun the fit with this answer.
    pub rerun_args: Vec<String>,
    /// The objective the fit reached with this answer
    /// ([`RunManifest::objective`](super::RunManifest::objective)), when it tried it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub objective: Option<f64>,
    /// The cells this answer stands for, a list of names a line, when it
    /// stands for cells: what an annotation can say they are.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cells: Option<String>,
}

/// One round's place in a model's history: the version it continued from, the
/// critique labels it trained against, and the partition it collapsed on. By
/// path for now, relative to the manifest; the label file's hash and the
/// round's seed (plan §1) arrive with stage 2, which reads that file.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunHistory {
    /// The parent version's prefix.
    pub parent: String,
    /// `{out}.critique.labels.{model}.parquet` from `senna critique`, if any.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub peer_labels: Option<String>,
    /// The run whose partition this round collapsed on, if any.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pb_from: Option<String>,
    #[serde(flatten, default)]
    pub unknown: Unknown,
}

/// The training subcommand's own argument struct, recorded so `senna update`
/// can continue the run without the user retyping its configuration.
///
/// **`args` is deliberately untyped.** Every downstream command that touches a
/// manifest loads it, mutates a section and saves it back, so all of them
/// round-trip this field. Storing it as a typed struct would mean a senna
/// version that added a flag could no longer *open* a run — `lupin plot` would
/// fail, not just `senna update`. As a [`serde_json::Value`] it passes through
/// untouched and only the one command that interprets it can be affected.

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainArgsRecord {
    /// senna version that wrote the record, named in the error when a later
    /// version cannot replay it.
    pub senna_version: Box<str>,
    /// The serialized argument struct for [`RunManifest::kind`](super::RunManifest::kind).
    pub args: serde_json::Value,
}

/// Serialize a fit's argument struct for [`RunManifest::train_args`](super::RunManifest::train_args).
pub fn record_train_args<T: Serialize>(args: &T) -> anyhow::Result<TrainArgsRecord> {
    Ok(TrainArgsRecord {
        senna_version: env!("CARGO_PKG_VERSION").into(),
        args: serde_json::to_value(args)?,
    })
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RunData {
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub input: Vec<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub input_null: Vec<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub batch: Vec<String>,
    /// The multiome layout `input` was loaded under, when there was one.
    /// Positional against `input`, which the trainer writes in group order.
    /// Absent for a single-modality run — and for multiome runs written
    /// before this was recorded, which is why every consumer treats `None`
    /// as the plain load rather than failing.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub multiome: Option<crate::multiome_layout::RunMultiome>,
    /// Fields this version does not know, written by another tool or a
    /// newer senna; kept so a load/save round trip never drops them.
    #[serde(flatten, default)]
    pub unknown: Unknown,
}

/// The numeric SCALE a gene × component artifact is stored in.
///
/// Every feature-side parquet in a senna run is one of these, and confusing them
/// is the single most productive source of bugs in the chain: reading a
/// `LogSimplexColumns` table as if it were `ProbabilitySimplexColumns` gives
/// `ln(negative)` — silent `NaN`s and a degenerate downstream model, with no
/// shape mismatch to catch it. Slot names say which axis; this says which scale.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ArtifactScale {
    /// Unconstrained real values: embeddings (ρ, co-embed) and SVD loadings.
    Signed,
    /// Each COLUMN is a log-simplex over rows: `Σ_g exp(x[g,k]) = 1`. The
    /// `log_softmax`-over-genes topic dictionary β.
    LogSimplexColumns,
    /// Each COLUMN is a probability simplex over rows: `Σ_g x[g,k] = 1`, all
    /// non-negative. The empirical dictionary.
    ProbabilitySimplexColumns,
}

impl ArtifactScale {
    /// Classify a loaded matrix by its own contents. Independent of filename and
    /// of manifest bookkeeping, both of which have moved between versions.
    #[must_use]
    pub fn detect(m: &Mat) -> Self {
        if m.ncols() == 0 || m.nrows() == 0 {
            return Self::Signed;
        }
        // Both checks short-circuit on the first column that disagrees, so the
        // common (signed) case costs one column, not the whole matrix.
        let sums_to_one = |f: fn(f32) -> f64| {
            (0..m.ncols()).all(|k| {
                let s: f64 = m.column(k).iter().map(|&v| f(v)).sum();
                (s - 1.0).abs() < 1e-2
            })
        };
        if sums_to_one(|v| f64::from(v).exp()) {
            Self::LogSimplexColumns
        } else if m.iter().all(|&v| v >= 0.0) && sums_to_one(f64::from) {
            Self::ProbabilitySimplexColumns
        } else {
            Self::Signed
        }
    }

    /// Fail loudly when a file is not in the scale the caller assumes.
    pub fn ensure(m: &Mat, want: Self, what: &str) -> anyhow::Result<()> {
        let got = Self::detect(m);
        anyhow::ensure!(
            got == want,
            "{what}: expected {want:?} but the values are {got:?} — reading one as the other \
             silently produces NaN or a degenerate model"
        );
        Ok(())
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RunOutputs {
    /// `{out}.latent.parquet`: cell × K matrix. For topic runs this is
    /// log-softmax topic proportions; for SVD runs it's component
    /// scores. Consumers that argmax (e.g. `lupin plot --colour-by
    /// topic`) should check `kind` before assuming topic semantics.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub latent: Option<String>,
    /// `{out}.dictionary.parquet` — gene × K. **Overloaded slot; check `kind`.**
    ///
    /// - topic family: the **softmax dictionary** β, i.e. `log_softmax` over
    ///   GENES, so each column exponentiates to 1 (`Σ_g exp(β[g,k]) = 1`). It is
    ///   in LOG space — a frequent source of bugs when read as probabilities.
    /// - SVD family: signed component loadings (no simplex).
    ///
    /// Anything that needs ρ should read `feature_embedding` and never this slot.
    ///
    /// Retained for the SVD family (signed loadings) and as the parse target for
    /// pre-split manifests. Topic runs now write
    /// [`RunOutputs::softmax_dictionary`]; read either through
    /// [`RunOutputs::gene_dictionary`].
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dictionary: Option<String>,
    /// `{out}.softmax_dictionary.parquet` — gene × K topic dictionary β, in
    /// **log space** and column-wise a simplex over genes
    /// ([`ArtifactScale::LogSimplexColumns`]).
    ///
    /// Split out of `dictionary` so the name states both the axis and the scale:
    /// `dictionary` had meant β, SVD signed loadings, or the embedding ρ,
    /// and the resulting shape/scale ambiguity
    /// produced real bugs — reading this table as probabilities yields
    /// `ln(negative)`, i.e. silent NaN.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub softmax_dictionary: Option<String>,
    /// Optional `group_id<TAB>display_name` TSV for `lupin plot` labels.
    /// User-populated; no senna subcommand writes this field.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub anchor_labels: Option<String>,
    /// `{out}.cell_proj.parquet` — cell × `proj_dim` random projection
    /// computed during training. Cached so `senna layout` can re-derive
    /// PB structure (via RSVD + multi-level collapse on the projection)
    /// without touching raw data.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cell_proj: Option<String>,
    /// `{out}.safetensors` — trained VAE weights (topic / masked-topic /
    /// joint-topic only).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    /// `{out}.model.json` — topic-model metadata for `senna
    /// predict` (topic / masked-topic / joint-topic only).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub metadata: Option<String>,
    /// `{out}.pb_gene.parquet` — G × P pseudobulk gene aggregates at the
    /// finest collapse level. Consumed by `lupin annotate` (enrichment) to build a
    /// permutation null without touching the raw zarr.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pb_gene: Option<String>,
    /// `{out}.pb_latent.parquet` — P × K PB-level mean topic proportions
    /// (topic kinds) or mean SVD component scores (svd kinds). For topic
    /// kinds this is derived from the encoder forward on the finest
    /// collapse; for SVD it's `proj_kn.transpose()` at the finest level.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pb_latent: Option<String>,
    /// `{out}.pb_reference.zarr.zip` — this run's pseudobulks, carried forward so a
    /// later `senna update` can absorb a sample without re-reading every cell
    /// already seen. `None` under `--no-emit-pb-reference`. The
    /// sidecar `{out}.pb_reference.json` sits beside it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pb_reference: Option<String>,
    /// `{out}.dictionary_empirical.parquet` — G × K empirical β at full
    /// gene resolution: row-scaled by NB Fisher-info weights and column-
    /// normalized to the topic simplex. Avoids the lossy expand-from-coarse
    /// approximation in `dictionary` (which ships at the feature-coarsened
    /// resolution and is interpolated back). `lupin annotate` (enrichment) prefers this
    /// when present; falls back to `dictionary` otherwise.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dictionary_empirical: Option<String>,
    /// `{out}.feature_embedding.parquet` — D × H RAW per-gene embedding ρ on
    /// the model's own axis, i.e. the loading that pairs with `cell_embedding`
    /// in the Poisson rate `exp(ρ_g · z_n + a_g + b_n)`. For the masked topic
    /// family it is the ρ shared by encoder and decoder under
    /// β = `log_softmax_d(α · ρᵀ)`; for `fne` it spans every node type.
    ///
    /// Distinct from `feature_coembedding`, the SIMBA co-embed (genes
    /// re-projected ONTO the cell manifold). Both are D × H and neither
    /// substitutes for the other: the co-embed is what nearest-centroid
    /// annotation needs, ρ is what a rate reconstruction needs.
    ///
    /// Written by every embedding command on EVERY path, interrupted or not.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub feature_embedding: Option<String>,
    /// `{out}.feature_coembedding.parquet` — D × H SIMBA co-embed: each gene as
    /// a softmax-over-cells weighted average of the cell embeddings, so genes
    /// sit on the cell manifold and a Euclidean nearest-centroid call
    /// (`lupin annotate` (projection), `lineage`) is well posed. A lossy, one-way
    /// derived view of `feature_embedding`. Present only when the co-embed
    /// ran: `bge` / `simba` / `resolve-embedding-space`, not on an
    /// interrupted run.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub feature_coembedding: Option<String>,
    /// `{out}.module_membership.parquet`: gene × M learned-module membership
    /// (rows on the simplex with exact zeros). Present only for a run trained with
    /// gene modules; `feature_embedding` still holds the composed row, so a reader
    /// that ignores this slot loses nothing.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub module_membership: Option<String>,
    /// `{out}.module_dictionary.parquet`: M × H module vectors, the `μ` in
    /// `ρ_g = Σ_m π_gm μ_m + r_g`. Paired with `module_membership`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub module_dictionary: Option<String>,
    /// `{out}.cell_embedding.parquet` — N × H per-cell embedding Z in the
    /// SAME H-space as `feature_embedding` ρ.
    ///
    /// Written by EVERY embedding command (`bge`, `fne`,
    /// `resolve-embedding-space`), regardless of whether that run also
    /// resolved topics: Z lives here and only here, and `latent` is reserved
    /// for log θ. Consumers doing geometry (kNN, layout, clustering,
    /// trajectory) should go through [`RunOutputs::geometry_latent`] rather
    /// than reading `latent` directly, so they pick Z up automatically.
    ///
    /// Manifests written before this contract put Z in `latent` and left this
    /// `None`; `geometry_latent` falls back for them.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cell_embedding: Option<String>,
    /// `{out}.cell_encoder.safetensors` — the distilled pooled-gene encoder a
    /// `bge` run placed its cells with (trunk weights plus the per-gene mean it
    /// divides by). `senna predict` places query cells through it, so a query
    /// and the run's own cells share one estimator. Absent on runs whose
    /// phase 2 was the block SGD.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cell_encoder: Option<String>,
    /// `{out}.cell_to_pb.parquet` — N × `num_levels` u32 matrix of the
    /// post-refinement cell→pseudobulk membership per coarsening level
    /// (finest-last to match `collapsed_levels`). Cached so a downstream
    /// `senna {topic, masked-topic} --from` chain can skip the
    /// expensive HNSW + binary-sort + DC-SBM refinement step and feed
    /// the precomputed partition straight into the per-PB Gamma fit.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cell_to_pb: Option<String>,
    /// `{out}.pb_tree.json` — the tree behind the finest
    /// pseudobulk partition: per node, the recursive splits with their
    /// contrast genes, noise-edge verdicts and likelihood ratios. Written
    /// by the topic-family fits whose collapse rewrote the high bits.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pb_tree: Option<String>,
    /// `{out}.cell_time.parquet`, read by `senna view` when present: every
    /// cell's lineage, time within it and its spread, and lineage
    /// posteriors (string and number columns, keyed by `cell`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cell_time: Option<String>,
    /// `{out}.cell_velocity.parquet`, read by `senna view` when present:
    /// every cell's velocity in the coordinates of `cell_embedding`, row
    /// for row.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cell_velocity: Option<String>,
    /// `{out}.pb_time.parquet`: every finest pseudobulk's posterior over
    /// lineage and time on the grid, a row per `(pb, lineage, position)`
    /// with its `phase` on the grid (its lineage's own time) and mass `q`,
    /// keyed by the `cell_to_pb` finest level.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pb_time: Option<String>,
    /// `{out}.branch_time.tsv`: every other lineage's branch time off the
    /// trunk at the mode, on `pb_time`'s phase scale; only with `--trunk`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub branch_time: Option<String>,
    /// The cells at the root an ode fit grew from, one name a line: the
    /// `--root-cells` list it was given, or `{out}.root.txt`, the cells of
    /// the root it chose; absent when no tree start won (or on older runs).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub root_cells: Option<String>,
    /// Fields this version does not know, written by another tool or a
    /// newer senna; kept so a load/save round trip never drops them.
    #[serde(flatten, default)]
    pub unknown: Unknown,
}

impl RunOutputs {
    /// The gene × component dictionary, whichever slot holds it: the topic
    /// family's [`Self::softmax_dictionary`] (log-simplex β) or the SVD family's
    /// [`Self::dictionary`] (signed loadings), the latter also being where
    /// pre-split manifests land.
    ///
    /// Callers that care about the SCALE must still branch on `kind`, or classify
    /// the loaded matrix with [`ArtifactScale::detect`] — the two differ by more
    /// than a name, and mixing them silently produces NaN.
    #[must_use]
    pub fn gene_dictionary(&self) -> Option<&str> {
        self.softmax_dictionary
            .as_deref()
            .or(self.dictionary.as_deref())
    }

    /// The cell × H table to use for GEOMETRY — kNN graphs, layout,
    /// clustering, trajectory: prefer `cell_embedding`, fall back to `latent`.
    ///
    /// The embedding commands (`bge`, `fne`, `resolve-embedding-space`) write
    /// their H-space Z to `cell_embedding` and reserve `latent` for log θ, so
    /// reading `latent` directly on a topic-resolving embedding run would hand
    /// a metric consumer log-simplex coordinates and silently apply the wrong
    /// distance. Going through this accessor also keeps pre-contract manifests
    /// (Z stored in `latent`, no `cell_embedding`) working unchanged.
    ///
    /// Topic-semantics consumers — `lupin plot --colour-by topic`, `lupin plot-topic` —
    /// must NOT use this; they need θ first and go through
    /// [`RunOutputs::structure_latent`], whose priority is the reverse.
    #[must_use]
    pub fn geometry_latent(&self) -> Option<&str> {
        self.cell_embedding.as_deref().or(self.latent.as_deref())
    }

    /// The cell × K table to use for a COMPOSITION view — `lupin plot-topic`'s
    /// structure bars, `lupin plot --colour-by topic`: prefer `latent` (log θ),
    /// fall back to `cell_embedding`.
    ///
    /// Those consumers exponentiate and row-normalise whatever they read, so
    /// a signed embedding falls through the same arithmetic as a per-cell
    /// softmax over its axes: an honest "which direction dominates this
    /// cell" picture for a run with no topics (`simba`, `bge --skip-etm`).
    /// The priority is the opposite of [`RunOutputs::geometry_latent`] on
    /// purpose: a topic-resolving embedding run has both tables, and θ is the
    /// composition while Z is the geometry.
    #[must_use]
    pub fn structure_latent(&self) -> Option<&str> {
        self.latent.as_deref().or(self.cell_embedding.as_deref())
    }

    /// The gene × K table paired with [`RunOutputs::structure_latent`]:
    /// `dictionary_empirical` (full resolution), else the topic / SVD
    /// dictionary, else the raw gene table of an embedding run. The last
    /// pairs with `cell_embedding` — both are `H` wide — which is what keeps
    /// the dictionary panel's K check honest when the latent fell through.
    #[must_use]
    pub fn structure_dictionary(&self) -> Option<&str> {
        self.dictionary_empirical
            .as_deref()
            .or_else(|| self.gene_dictionary())
            .or(self.feature_embedding.as_deref())
    }
}

/// Paths written by `senna layout`.
///
/// The top-level `cell_coords` / `pb_coords` always point at the most recent
/// cell layout, so single-layout consumers (`plot`, `lineage`) keep reading one
/// slot. `methods` keeps every layout that has been run, keyed by method name
/// (`umap`, `phate`, `tsne`), so a viewer can switch between them.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RunLayout {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cell_coords: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pb_coords: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pb_gene_mean: Option<String>,
    /// Method name of the layout the top-level slots point at.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub current: Option<String>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub methods: BTreeMap<String, LayoutEntry>,
    /// Fields this version does not know, written by another tool or a
    /// newer senna; kept so a load/save round trip never drops them.
    #[serde(flatten, default)]
    pub unknown: Unknown,
}

/// One layout method's outputs. Cells and features are laid out by separate
/// commands, so any subset of the slots may be present.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct LayoutEntry {
    /// `{out}.{method}.cell_coords.parquet`: N × 2+ cell positions.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cell_coords: Option<String>,
    /// `{out}.{method}.pb_coords.parquet`: pseudobulk positions (PB path only).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pb_coords: Option<String>,
    /// `{out}.{method}.feature_on_cell_coords.parquet`: D × 2 feature
    /// positions on this method's cell map, placed from `feature_coembedding`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub feature_on_cell_coords: Option<String>,
    /// `{out}.{method}.feature_coords.parquet`: D × 2 layout of the feature
    /// embedding ρ on its own (`senna layout {method} --target features`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub feature_coords: Option<String>,
}

/// Paths to artifacts produced by `senna cluster`. Populated by `senna
/// cluster` when invoked with `--from <manifest>`; otherwise empty.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RunCluster {
    /// `{cluster_out}.clusters.parquet` — cells × 1 cluster id (NaN for unassigned).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub clusters: Option<String>,
    /// `{cluster_out}.feature_clusters.parquet` — features × 1 cluster id of
    /// the feature embedding (`senna clustering --target features`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub feature_clusters: Option<String>,
    /// Fields this version does not know, written by another tool or a
    /// newer senna; kept so a load/save round trip never drops them.
    #[serde(flatten, default)]
    pub unknown: Unknown,
}

/// Paths to artifacts produced by `lupin annotate` (enrichment) — the cluster-based
/// marker enrichment annotation pass. Populated by annotate, not by training.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RunAnnotate {
    /// `{annotate_out}.annotation.parquet` — N × C cell posterior.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub annotation: Option<String>,
    /// `{annotate_out}.argmax.tsv` or `.argmax.parquet` — per-cell label + max probability.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub argmax: Option<String>,
    /// `{annotate_out}.cluster_celltype_q.parquet` — nClusters × C FDR-sparse
    /// softmax-normalized Q matrix.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cluster_celltype_q: Option<String>,
    /// `{annotate_out}.cluster_celltype_es.parquet` — nClusters × C raw ES.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cluster_celltype_es: Option<String>,
    /// `{annotate_out}.cluster_expression.parquet` — G × nClusters NB-Fisher-
    /// adjusted per-cluster mean expression.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cluster_expression: Option<String>,
    /// Input marker-gene TSV path (provenance).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub markers: Option<String>,
    /// `{annotate_out}.ontology_assignment.tsv` — per-cluster TreeBH cell-type
    /// call on the Cell Ontology (assigned node, abstention, disagreement).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ontology_assignment: Option<String>,
    /// `{annotate_out}.ontology_node_mass.parquet` — nClusters × induced-CL-node
    /// soft mass (Σ descendant-leaf Q) for visualization.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ontology_node_mass: Option<String>,
    /// `{annotate_out}.ontology_signature.tsv` — per-cluster GO/GMT signature
    /// (top terms by cross-cluster-contrasted module-score effect). Written by
    /// the `--gaf/--gmt` gene-set mode (distinct from the CL TreeBH assignment).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ontology_signature: Option<String>,
    /// `{annotate_out}.ontology_term_effect.parquet` — nClusters × nTerms
    /// contrasted module-score effect backing the GO/GMT signature.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ontology_term_effect: Option<String>,
    /// Fields this version does not know, written by another tool or a
    /// newer senna; kept so a load/save round trip never drops them.
    #[serde(flatten, default)]
    pub unknown: Unknown,
}

/// Paths to the tables of `senna donor-deviation` (stage 2) on this run,
/// recorded by it so a viewer of the run finds them. Empty until it runs.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct RunStage2 {
    /// `{out}.donor_gene_deviation.parquet`: per (gene, donor) the
    /// deviation's `intercept` and `state_axis_*` coefficients.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub donor_gene_deviation: Option<String>,
    /// `{out}.donor_gene_deviation_without_batch.parquet`: the batch
    /// jackknife refits of the same coefficients.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub donor_gene_deviation_without_batch: Option<String>,
    /// `{out}.donor_sample_states.parquet`: per pseudobulk sample its donor,
    /// matched distances and state coordinates.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub donor_sample_states: Option<String>,
    /// `{out}.donor_cell_states.parquet`: per cell its sample, donor and
    /// state coordinates.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub donor_cell_states: Option<String>,
    /// `{out}.donor_deviation.json`: the run's settings and donors.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub settings: Option<String>,
    /// When the tables were written (nanoseconds since the Unix epoch), so
    /// a stage 2 run again to the same paths still reads as new.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub written: Option<u64>,
    /// Fields this version does not know, written by another tool or a
    /// newer senna; kept so a load/save round trip never drops them.
    #[serde(flatten, default)]
    pub unknown: Unknown,
}

impl RunStage2 {
    /// Whether nothing is recorded (left out of the manifest).
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.donor_gene_deviation.is_none()
            && self.donor_gene_deviation_without_batch.is_none()
            && self.donor_sample_states.is_none()
            && self.donor_cell_states.is_none()
            && self.settings.is_none()
            && self.written.is_none()
            && self.unknown.is_empty()
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RunDefaults {
    /// Default `--colour-by` for `lupin plot`: `"topic" | "cluster" | "pb-id"`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub colour_by: Option<String>,
    /// Default `--palette` for `lupin plot`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub palette: Option<String>,
    /// Fields this version does not know, written by another tool or a
    /// newer senna; kept so a load/save round trip never drops them.
    #[serde(flatten, default)]
    pub unknown: Unknown,
}
