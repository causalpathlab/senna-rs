//! [`RunKind`], the subcommand that produced a run, and what its tables mean.

use super::Mat;
use serde::{Deserialize, Serialize};

/// Subcommand that produced the run. Serde-encoded as kebab-case strings
/// (`"topic"`, `"itopic"`, `"joint-topic"`, `"svd"`, `"joint-svd"`,
/// `"bge"`, `"fne"`) so the JSON wire format is stable across renames.
///
/// `Itopic` is the wire string for the `masked-topic` command: the command
/// was renamed (`itopic` → `masked-topic`) but the manifest kind is kept as
/// the legacy `"itopic"` so existing manifests keep parsing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum RunKind {
    Topic,
    Itopic,
    /// `senna masked-vae` — same masked-imputation pipeline and same
    /// simplex β as [`RunKind::Itopic`], but its `{out}.latent.parquet` holds
    /// an **unconstrained Gaussian `z`**, not `log θ`. Distinct from `Itopic`
    /// precisely so consumers stop applying the log-θ contract to it; see
    /// [`RunKind::latent_is_log_simplex`].
    MaskedVae,
    JointTopic,
    Vae,
    Svd,
    JointSvd,
    Bge,
    Fne,
    ResolveEmbeddingSpace,
    /// `senna simba` — SIMBA's cell × gene node embeddings from the binned
    /// bipartite expression graph. Euclidean `Z` in `cell_embedding`, the raw
    /// gene table in `feature_embedding`, SIMBA's fixed-T co-embedded genes in
    /// `feature_coembedding`. No decoder. Every downstream command reads it as
    /// a `bge` run: `predict`, `probe` and `impute` project a query onto the
    /// frozen gene table with a zero gene bias (SIMBA's score is a pure dot
    /// product), `deconvolve` takes the gene axis and Z, and `update` re-fits
    /// on the union.
    Simba,
    /// `senna rbm`: a feature-embedded restricted Boltzmann machine.
    Rbm,
    /// `senna ode` — every cell's lineage, time and velocity from two-track
    /// counts, by a mixture of splicing ODEs. `cell_embedding` is the map
    /// space (every cell's expression on its leading principal components),
    /// `cell_velocity` every cell's velocity there, `cell_time` its lineage
    /// and time. No gene table or decoder a viewer can read.
    Ode,
}

/// How a run's tables give each cell's expected level of each feature, up
/// to a constant per cell (its depth) and, where no bias was kept, per
/// feature. See [`RunKind::expression_model`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExpressionModel {
    /// `log rate = ρ_g · z_n + a_g`: `cell_embedding` (z), `feature_embedding`
    /// (ρ) and, when kept, `feature_bias` (a).
    Embedding,
    /// `π_n = softmax_g(z_n · W + b)`: `latent` (z) and `dictionary` (W, gene
    /// × factor). The bias b lives only in the checkpoint, per coarse
    /// feature group, so it is left out.
    LatentLoadings,
    /// `rate = Σ_k θ_nk β_kg`: `latent` is log θ, the dictionary log β.
    Topic,
    /// As [`Self::Topic`], with `latent` the raw z the decoder reads through
    /// `log θ = log_softmax(z)`.
    TopicOfLogits,
    /// No model of expression (a decomposition, or no cells): only the
    /// observed counts.
    Unmodelled,
}

/// What the table [`RunOutputs::geometry_latent`](super::RunOutputs::geometry_latent) returns actually IS.
///
/// Every downstream geometry decision — exponentiate or not, cosine or z-scored
/// Euclidean kNN, lay out on cells or on pseudobulk landmarks — is a question
/// about this, not about which command ran. They used to be asked as
/// `matches!(kind, Bge | Fne)` and `_ =>` at five separate sites, so a new
/// `RunKind` compiled fine and silently took the fallback at every one of them.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum CellSpace {
    /// Euclidean coordinates where MAGNITUDE carries signal. Angular distance
    /// is the right metric (plain Euclidean is dominated by the depth axis on a
    /// raw embedding), and the layout can run on cells directly.
    Embedding,
    /// Rows are `log θ`: `exp()` gives a probability vector summing to 1.
    LogSimplex,
    /// Signed scores — component loadings, or a Gaussian `z`. No simplex, and
    /// no magnitude semantics to preserve.
    Signed,
}

impl CellSpace {
    /// Put a table of rows (cells or pseudobulks) in place into the space
    /// where Euclidean distance compares rows as this space calls for, and
    /// say whether its rows then have unit length (so a mean of rows is
    /// renormalized to stay in it):
    ///
    /// * [`CellSpace::LogSimplex`]: rows of θ (probabilities, not log θ)
    ///   become √θ, whose Euclidean distance is the Hellinger distance;
    /// * [`CellSpace::Embedding`]: rows scaled to unit length, the angular
    ///   distance (an embedding's magnitude follows depth);
    /// * [`CellSpace::Signed`]: columns z-scored (population sd), so no one
    ///   component's scale decides the neighbours.
    ///
    /// The one metric policy for every reader that compares cells of a run
    /// (`senna critique`, `senna donor-deviation`).
    pub fn to_euclidean(self, x: &mut Mat) -> bool {
        match self {
            CellSpace::LogSimplex => {
                x.apply(|v| *v = v.max(0.0).sqrt());
                true
            }
            CellSpace::Embedding => {
                legume_numeric::matrix::dense_mat_io::l2_normalize_rows_inplace(x);
                true
            }
            CellSpace::Signed => {
                crate::embed_common::zscore_columns_inplace(x);
                false
            }
        }
    }
}

impl RunKind {
    /// The space of this kind's per-cell geometry table.
    ///
    /// Deliberately a full `match` with no `_` arm: a new variant must say what
    /// its cells are before it compiles. That is the whole point — the previous
    /// shape let a kind be added to the enum and silently inherit a fallback.
    #[must_use]
    pub fn cell_space(self) -> CellSpace {
        match self {
            // These write `cell_embedding` (or, for fne, a latent that IS the
            // graph embedding), so `geometry_latent` hands back Euclidean Z.
            RunKind::Bge | RunKind::Fne | RunKind::ResolveEmbeddingSpace | RunKind::Simba => {
                CellSpace::Embedding
            }
            // `latent` is log θ and `cell_embedding` is absent, so the geometry
            // table is the simplex itself.
            RunKind::Topic | RunKind::Itopic | RunKind::JointTopic => CellSpace::LogSimplex,
            // masked-vae is a Gaussian z despite its simplex β; svd is signed
            // loadings; vae is an unconstrained latent.
            RunKind::MaskedVae
            | RunKind::Vae
            | RunKind::Svd
            | RunKind::JointSvd
            | RunKind::Rbm
            | RunKind::Ode => CellSpace::Signed,
        }
    }

    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            RunKind::Topic => "topic",
            RunKind::Itopic => "itopic",
            RunKind::MaskedVae => "masked-vae",
            RunKind::JointTopic => "joint-topic",
            RunKind::Vae => "vae",
            RunKind::Svd => "svd",
            RunKind::JointSvd => "joint-svd",
            RunKind::Bge => "bge",
            RunKind::Fne => "fne",
            RunKind::ResolveEmbeddingSpace => "resolve-embedding-space",
            RunKind::Simba => "simba",
            RunKind::Rbm => "rbm",
            RunKind::Ode => "ode",
        }
    }

    /// Topic-family kinds (topic / masked-topic / masked-vae / joint-topic) —
    /// produce a probability-simplex β. SVD-family kinds produce signed
    /// loadings.
    ///
    /// Note this says nothing about the *latent*: `masked-vae` has a simplex β
    /// but a Gaussian latent. Use [`Self::latent_is_log_simplex`] for that.
    #[must_use]
    pub fn is_topic_family(self) -> bool {
        match self {
            // masked-vae belongs here despite its Gaussian latent: the family is
            // defined by a simplex β, not by what the latent itself looks like.
            RunKind::Topic | RunKind::Itopic | RunKind::MaskedVae | RunKind::JointTopic => true,
            RunKind::Vae
            | RunKind::Rbm
            | RunKind::Ode
            | RunKind::Svd
            | RunKind::JointSvd
            | RunKind::Bge
            | RunKind::Fne
            | RunKind::ResolveEmbeddingSpace
            | RunKind::Simba => false,
        }
    }

    /// Masked-imputation kinds — `masked-topic`, `masked-sbp` (both
    /// [`RunKind::Itopic`]) and `masked-vae`. They share one artifact layout
    /// (`MaskedModel`), one scorer, and one warm-start shape, so consumers that
    /// care about *how the model is built* rather than what its latent means
    /// gate on this. The three heads are told apart only by `model_type`.
    #[must_use]
    pub fn is_masked_family(self) -> bool {
        matches!(self, RunKind::Itopic | RunKind::MaskedVae)
    }

    /// Kinds whose whole gene-side model is a frozen `gene × H` table in
    /// `feature_embedding` — no checkpoint, no encoder — so a query is placed
    /// by projecting each cell onto it (`predict`, `probe`, `impute`,
    /// `deconvolve`, `predict --bulk` all go through `BgeEmbedding`).
    ///
    /// A full match on purpose: a new kind must say whether it reads this
    /// way before this compiles, rather than being missed at six call sites.
    #[must_use]
    pub fn has_frozen_gene_table(self) -> bool {
        match self {
            RunKind::Bge | RunKind::Simba => true,
            RunKind::Topic
            | RunKind::Itopic
            | RunKind::MaskedVae
            | RunKind::JointTopic
            | RunKind::Vae
            | RunKind::Rbm
            | RunKind::Ode
            | RunKind::Svd
            | RunKind::JointSvd
            | RunKind::Fne
            | RunKind::ResolveEmbeddingSpace => false,
        }
    }

    /// Whether the frozen gene table comes with a per-gene bias
    /// (`feature_bias.parquet`). bge fits one and cannot be scored without
    /// it; SIMBA's score is a pure dot product, so its bias is zero by
    /// construction and the file never exists. Only meaningful for kinds
    /// with [`Self::has_frozen_gene_table`].
    #[must_use]
    pub fn has_gene_bias(self) -> bool {
        match self {
            RunKind::Bge => true,
            RunKind::Simba => false,
            // Not a frozen-gene-table kind; the question does not arise.
            RunKind::Topic
            | RunKind::Itopic
            | RunKind::MaskedVae
            | RunKind::JointTopic
            | RunKind::Vae
            | RunKind::Rbm
            | RunKind::Ode
            | RunKind::Svd
            | RunKind::JointSvd
            | RunKind::Fne
            | RunKind::ResolveEmbeddingSpace => false,
        }
    }

    /// Whether the run has cells at all: every kind embeds or decomposes a
    /// cell × feature table except `fne`, which embeds a graph of features
    /// alone. A full `match`, so a new kind must answer.
    #[must_use]
    pub fn has_cells(self) -> bool {
        match self {
            RunKind::Fne => false,
            RunKind::Topic
            | RunKind::Itopic
            | RunKind::MaskedVae
            | RunKind::JointTopic
            | RunKind::Vae
            | RunKind::Rbm
            | RunKind::Ode
            | RunKind::Svd
            | RunKind::JointSvd
            | RunKind::Bge
            | RunKind::ResolveEmbeddingSpace
            | RunKind::Simba => true,
        }
    }

    /// True when `{out}.latent.parquet` holds `log θ` on the probability
    /// simplex — i.e. `exp()` of it gives per-cell topic proportions that sum
    /// to 1.
    ///
    /// False for `masked-vae` (raw Gaussian `z`), the SVD family (signed
    /// loadings) and the embedding kinds. Anything about to `exp()` a latent,
    /// renormalize its rows, or read it as proportions must gate on this
    /// rather than on [`Self::is_topic_family`].
    #[must_use]
    pub fn latent_is_log_simplex(self) -> bool {
        self.expression_model() == ExpressionModel::Topic
    }

    /// How this kind's tables give a cell's expected expression, which the
    /// viewer colours and ranks features by. A full `match`, so a new kind
    /// must say how its decoder reads its tables (or that it has none).
    #[must_use]
    pub fn expression_model(self) -> ExpressionModel {
        match self {
            RunKind::Topic | RunKind::Itopic | RunKind::JointTopic => ExpressionModel::Topic,
            RunKind::MaskedVae => ExpressionModel::TopicOfLogits,
            // vae's decoder, rbm's visible units given the hidden.
            RunKind::Vae | RunKind::Rbm => ExpressionModel::LatentLoadings,
            RunKind::Bge | RunKind::ResolveEmbeddingSpace | RunKind::Simba => {
                ExpressionModel::Embedding
            }
            RunKind::Svd | RunKind::JointSvd | RunKind::Fne | RunKind::Ode => {
                ExpressionModel::Unmodelled
            }
        }
    }
}

impl std::fmt::Display for RunKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}
