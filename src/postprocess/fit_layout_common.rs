//! Shared plumbing for `senna layout {tsne, phate, umap}`:
//! - CLI args shared by the layouts.
//! - The cell table to lay out, from the run's own outputs: its latent (or
//!   cell embedding), else its cached cell projection. The count data is
//!   never read, so a run can be laid out where its data is not.
//! - PBs from landmarks or projection bits, PB-PB similarity, tail pruning.
//! - Cheap Nyström cell placement.
//! - Shared output writer.
//!
//! Every layout subcommand delegates here for everything except the
//! actual 2D layout algorithm.

use super::fit_layout_features::{
    place_features_on_cells, record_cell_layout, write_features_on_cells, FeatureSpace,
    LayoutTarget,
};
use super::viz_prep::{aggregate_features_by_group, select_pb_coverage};
use crate::geometry::cell_layout::project_cells_nystrom;
use crate::geometry::similarity::{
    compute_cosine_similarity, local_scale_similarity, regularize_similarity, threshold_similarity,
};
use data_beans::alg::random_projection::binary_sort_columns;
use rand::{rngs::SmallRng, SeedableRng};
use rayon::prelude::*;
use senna::embed_common::*;
use senna::run_manifest::{self, LayoutEntry, RunManifest};
use std::path::PathBuf;

/// How to pick landmarks for the latent-driven layout path.
#[derive(ValueEnum, Clone, Copy, Debug, PartialEq)]
#[clap(rename_all = "kebab-case")]
pub enum LandmarkStrategy {
    /// Uniform random subsample of cells. Largest topics dominate the
    /// landmark set in proportion to their cell count.
    Random,
    /// Argmax-stratified: equal landmark budget per active topic. Use
    /// when one topic dominates and you want minor topics resolved
    /// without intra-major-topic substructure overwhelming the layout.
    /// Falls back to Random for non-topic kinds (SVD).
    PerTopic,
}

/// Default winsorization of layout features, in MADs (`--trim-cell-mads`).
pub(crate) const DEFAULT_TRIM_MADS: f32 = 5.0;

/// Self-loop amount added during similarity regularization to prevent
/// isolated nodes from collapsing the layout.
const SELF_LOOP_REG: f32 = 0.01;

/// Args shared by `senna layout phate`, `tsne` and `umap`.
///
/// These drive the PB construction / batch-correction / similarity /
/// cell-placement path. Layout-specific args live on the per-command
/// structs and are not duplicated here.
///
/// Every command that flattens this struct should declare its own
/// `long_about`. clap otherwise renders THIS doc comment as that
/// command's description, which is how `tsne` and `umap` came to
/// introduce themselves as "args shared by ...".
#[derive(Args, Debug, Clone)]
pub struct LayoutCommonArgs {
    #[arg(
        long,
        short = 'f',
        help = "Run manifest JSON from a `senna` topic/svd command",
        long_help = "The run to lay out. Its latent (or cell embedding), else its cached\n\
                     cell projection, is what is laid out; the count data is not read.\n\
                     Without --out, files go beside it, named after it.\n\
                     \n\
                     The manifest is then updated in place: `layout.methods` and\n\
                     `layout.cell_coords` / `layout.pb_coords` point at the files just written."
    )]
    pub from: Box<str>,

    #[arg(
        long,
        value_enum,
        default_value = "cells",
        help = "What to lay out: cells, or the feature embedding on its own",
        long_help = "cells    (default): lay out cells. On an embedding run with a co-embed,\n\
                     \x20 features are also placed on the new cell map.\n\
                     features: lay out the run's features by themselves (cosine kNN):\n\
                     \x20 the co-embedding when the run has one, else ρ (see --feature-space).\n\
                     \x20 Needs --from; reads no count data.\n\
                     \n\
                     Each method keeps its own files, `{out}.{method}.*.parquet`,\n\
                     recorded under `manifest.layout.methods`."
    )]
    pub target: LayoutTarget,

    #[arg(
        long,
        value_enum,
        default_value = "auto",
        help = "With --target features: the co-embedding (auto, when the run has one) or ρ",
        long_help = "Which table of the run's features --target features lays out.\n\
                     \n\
                     - auto (default): the co-embedding when the run wrote one\n\
                     \x20 (bge, simba, tde, resolve-embedding-space), else ρ.\n\
                     - coembedding: each feature where the cells it is active in are,\n\
                     \x20 so neighbouring features are active in the same cells.\n\
                     - rho: the feature embedding ρ as trained."
    )]
    pub feature_space: FeatureSpace,

    #[arg(
        long,
        short,
        help = "Output prefix (defaults to the --from path without `.senna.json`)",
        long_help = "Output header for results.\n\
                     \n\
                     {out}.{method}.pb_coords.parquet: Pseudobulk sample coordinates (n_pb × 2)\n\
                     \n\
                     {out}.{method}.cell_coords.parquet:\n\
                     Cell coordinates (n_cells × 2 or 3) with optional pb_id / cluster columns\n\
                     \n\
                     {out}.pb_proj_mean.parquet:\n\
                     Each PB's mean cell features (n_pb × dims), a diagnostic."
    )]
    pub out: Option<Box<str>>,

    #[arg(
        long,
        short = 'd',
        default_value_t = 10,
        help = "Top {d} projection bits for PB partitioning (≈ 2^d PBs)"
    )]
    pub sort_dim: usize,

    #[arg(
        long,
        help = "Cells per rayon job (omit for auto-scaling by feature count)",
        hide = true
    )]
    pub block_size: Option<usize>,

    #[arg(
        long,
        default_value_t = 0.0,
        help = "Similarity threshold for graph edges",
        long_help = "Edges with similarity below this are zeroed. Default 0 = no threshold."
    )]
    pub similarity_threshold: f32,

    #[arg(
        long = "trim-cell-mads",
        default_value_t = DEFAULT_TRIM_MADS,
        help = "Winsorize cell features to ±N MADs before the layout; 0 = off",
        long_help = "Winsorize cell features before the layout. This is ON by default, at N=5.\n\
                     \n\
                     Each feature dimension is clipped to `median ± N · MAD · 1.4826` across cells.\n\
                     A few extreme-outlier cells then cannot stretch the UMAP/t-SNE/PHATE layout.\n\
                     Nor can they dominate the PB-PB similarity.\n\
                     \n\
                     The rule is MAD-based, so it assumes no Gaussian shape.\n\
                     Only the most extreme tails are touched. Set it to 0 to disable."
    )]
    pub trim_cell_mads: f32,

    #[arg(
        long,
        help = "Local scaling using k-th neighbor distance",
        long_help = "Zelnik-Manor & Perona local scaling.\n\
                     σ_i = distance to k-th nearest neighbour;\n\
                     S_scaled(i,j) = S(i,j) / sqrt(σ_i σ_j)."
    )]
    pub local_scale_k: Option<usize>,

    #[arg(
        long,
        short = 'k',
        default_value_t = 15,
        help = "Number of nearest PBs for cell projection"
    )]
    pub knn: usize,

    #[arg(
        long,
        default_value_t = 10.0,
        help = "Alpha-decay kernel exponent for Nyström cell projection"
    )]
    pub kernel_alpha: f32,

    #[arg(
        long,
        default_value_t = 0.95,
        help = "Keep smallest PB set covering this fraction of cells",
        long_help = "Drop PB samples in the long tail,\n\
                     until the remaining set holds at least this fraction of all cells.\n\
                     - 1.0: keep every PB (no filtering)\n\
                     - 0.95 (default): remove minor subpopulations"
    )]
    pub pb_coverage: f32,

    #[arg(
        long,
        help = "Cluster assignments file (from `senna clustering`)",
        long_help = "Optional cluster assignments parquet file (cell × 1 matrix of IDs).\n\
                     If provided, cluster labels are added to cell_coords."
    )]
    pub clusters: Option<Box<str>>,

    #[arg(
        long,
        default_value_t = 42,
        help = "RNG seed for random PB-coordinate initialisation (tsne / mst)"
    )]
    pub seed: u64,

    #[arg(
        long,
        default_value_t = 1000,
        help = "Landmark count for latent-driven layout",
        long_help = "Used only when a trained latent is available.\n\
                     That means a topic or svd manifest.\n\
                     \n\
                     A random subsample of cells becomes the PB landmarks.\n\
                     Each cell is assigned to its nearest landmark. Lower values give fewer,\n\
                     denser PBs, so crisper clusters. Higher values give finer resolution,\n\
                     but blobbier output.\n\
                     \n\
                     A run laid out from its cell projection ignores this."
    )]
    pub n_landmarks: usize,

    #[arg(
        long,
        default_value_t = 1.0,
        help = "Temperature τ applied to topic θ before Hellinger transform",
        long_help = "Re-softmax the trained log-θ at temperature τ.\n\
                     This happens before the latent-path layout:\n\
                     feat ∝ sqrt(softmax(log_θ / τ)).\n\
                     \n\
                     The default τ=1.0 leaves θ unchanged. τ<1 sharpens it.\n\
                     Cells with mixed topics are pulled toward their dominant topic,\n\
                     which tightens clusters. The cost is losing intermediate positions.\n\
                     τ>1 softens instead.\n\
                     \n\
                     This affects the latent path only. Projection-space layouts ignore it."
    )]
    pub theta_temperature: f32,

    #[arg(
        long,
        value_enum,
        default_value = "per-topic",
        help = "Landmark sampling strategy for latent-driven layout",
        long_help = "per-topic — the default.\n\
                     \x20 Equal landmark budget per active argmax topic.\n\
                     \x20 Tightens cluster-level structure when one topic dominates.\n\
                     \x20 Topic-only; it falls back to random for SVD.\n\
                     random    — uniform random subsample.\n\
                     \x20 Largest topics get the most landmarks."
    )]
    pub landmark_strategy: LandmarkStrategy,
}

/// PHATE diffusion-embedding args. Shared by `layout phate` (main
/// layout) and `layout tsne` (PHATE-based t-SNE initialization).
#[derive(Args, Debug, Clone)]
pub struct PhateCliArgs {
    #[arg(long, default_value_t = 20, help = "PHATE diffusion time t")]
    pub phate_t: usize,

    #[arg(long, default_value_t = 5, help = "PHATE adaptive-bandwidth kNN")]
    pub phate_knn: usize,

    #[arg(
        long,
        default_value_t = 40.0,
        help = "PHATE alpha-decay kernel exponent"
    )]
    pub phate_alpha: f32,

    #[arg(long, default_value_t = 300, help = "Max SMACOF iterations")]
    pub phate_mds_iter: usize,

    #[arg(
        long,
        default_value_t = 1e-4,
        help = "SMACOF relative-stress tolerance"
    )]
    pub phate_mds_tol: f32,
}

impl From<&PhateCliArgs> for crate::geometry::phate::PhateArgs {
    fn from(a: &PhateCliArgs) -> Self {
        Self {
            t: a.phate_t,
            knn: a.phate_knn,
            alpha: a.phate_alpha,
            mds_iter: a.phate_mds_iter,
            mds_tol: a.phate_mds_tol,
        }
    }
}

/// The run a layout is for, read once: its manifest, where it is, and the
/// output prefix. The layout's files are recorded back into the manifest.
pub(crate) struct ResolvedViz {
    pub out: String,
    pub manifest_path: PathBuf,
    pub manifest: RunManifest,
    /// The manifest's directory, which its paths are relative to.
    pub dir: PathBuf,
}

impl ResolvedViz {
    /// A path the manifest records, resolved against its directory.
    pub(crate) fn resolve(&self, rel: &str) -> String {
        run_manifest::resolve(&self.dir, rel)
            .to_string_lossy()
            .into_owned()
    }

    /// Record the files just written under `manifest.layout` (relative to
    /// the manifest's directory, so the run directory can move) and save it.
    fn record(&mut self, method: &str, written: &LayoutEntry) -> anyhow::Result<()> {
        record_cell_layout(&mut self.manifest, &self.manifest_path, method, written)
    }
}

pub(crate) fn resolve_inputs(args: &LayoutCommonArgs) -> anyhow::Result<ResolvedViz> {
    let from = args.from.as_ref();
    let manifest_path = PathBuf::from(from);
    let (manifest, dir) = RunManifest::load(&manifest_path)?;
    info!("Loaded run manifest {from} (kind: {})", manifest.kind);
    let out = run_manifest::out_prefix(args.out.as_deref(), from);
    mkdir_parent(&out)?;
    Ok(ResolvedViz {
        out,
        manifest_path,
        manifest,
        dir,
    })
}

/// The landmark (PB) layout PHATE and t-SNE run on: PBs of the run's cell
/// table, their similarity, and the cells to place by Nyström.
pub(crate) struct PbLayoutPrep {
    pub cells: CellTable,
    pub pb_size: Vec<usize>,
    pub pb_membership_kept: Vec<usize>,
    /// `(n_pb × n_pb)` PB-PB similarity, post threshold / local scaling
    /// / diagonal regularization.
    pub pb_similarity: Mat,
    /// `(dims × n_pb)` each PB's mean cell features.
    pub pb_kp: Mat,
}

/// The run's cell table: what every layout lays out, UMAP cell by cell.
pub(crate) struct CellTable {
    /// Cell per column of `cells_kn`.
    pub names: Vec<Box<str>>,
    /// The cells' layout features (see [`latent_layout_features`]),
    /// column-per-cell.
    pub cells_kn: Mat,
}

/// The run's cell table for a cell-level layout: its latent (or cell
/// embedding) as [`latent_layout_features`] makes it, else its cached cell
/// projection, z-scored. Never the count data.
///
/// `geometry_latent` (not `outputs.latent`): on an embedding run the cell
/// table is the H-space Z in `cell_embedding`, while `latent` holds log θ,
/// which [`latent_layout_features`] would misread as raw Euclidean.
pub(crate) fn cell_table(
    args: &LayoutCommonArgs,
    resolved: &ResolvedViz,
) -> anyhow::Result<CellTable> {
    let m = &resolved.manifest;
    let (names, cells_kn) = match m.outputs.geometry_latent() {
        Some(p) => read_latent_features(args, &resolved.resolve(p), m.kind)?,
        None => read_cell_proj(args, &cell_proj_path(resolved)?)?,
    };
    Ok(CellTable { names, cells_kn })
}

/// The run's cached cell projection, which a run without a latent lays out.
fn cell_proj_path(resolved: &ResolvedViz) -> anyhow::Result<String> {
    resolved
        .manifest
        .outputs
        .cell_proj
        .as_deref()
        .map(|p| resolved.resolve(p))
        .ok_or_else(|| {
            anyhow::anyhow!(
                "{} records no latent, cell embedding or cell projection to lay out; \
                 train the run again with this senna",
                resolved.manifest_path.display()
            )
        })
}

/// The latent at `latent_path` as layout features (`dims × cells`) and its
/// cell names. The latent names its cells, so the data backend is not
/// opened (it may not even be here).
fn read_latent_features(
    args: &LayoutCommonArgs,
    latent_path: &str,
    kind: senna::run_manifest::RunKind,
) -> anyhow::Result<(Vec<Box<str>>, Mat)> {
    let MatWithNames {
        rows: cell_names,
        mat: latent_nk,
        ..
    } = Mat::from_parquet_with_row_names(latent_path, Some(0))?;
    let (feat_kn, latent_desc) = latent_layout_features(
        kind,
        &latent_nk,
        args.theta_temperature,
        args.trim_cell_mads,
    );
    info!(
        "Loaded latent: {} cells × {} dims ({latent_desc})",
        feat_kn.ncols(),
        feat_kn.nrows(),
    );
    Ok((cell_names, feat_kn))
}

/// The cached cell projection at `path` (written as cells × proj_dim) as
/// `proj_dim × cells`, each dimension z-scored across cells so none
/// dominates, then winsorized; and its cell names.
fn read_cell_proj(args: &LayoutCommonArgs, path: &str) -> anyhow::Result<(Vec<Box<str>>, Mat)> {
    let MatWithNames {
        rows: cell_names,
        mat: proj_nk,
        ..
    } = Mat::from_parquet_with_row_names(path, Some(0))?;
    let mut proj_kn: Mat = proj_nk.transpose();
    info!(
        "Loaded cell_proj: {} cells × {} proj-dims",
        proj_kn.ncols(),
        proj_kn.nrows()
    );
    proj_kn.scale_rows_inplace();
    winsorize_rows_inplace(&mut proj_kn, args.trim_cell_mads);
    Ok((cell_names, proj_kn))
}

/// The landmark layout for PHATE and t-SNE, whose O(n³) MDS / O(n²)
/// repulsion cannot take every cell: landmarks of the run's cell table (its
/// latent, else its cell projection), laid out on their own, cells then
/// placed by Nyström.
pub(crate) fn preprocess_layout_data(
    args: &LayoutCommonArgs,
    resolved: &ResolvedViz,
) -> anyhow::Result<PbLayoutPrep> {
    let cells = cell_table(args, resolved)?;
    let m = &resolved.manifest;
    if m.outputs.geometry_latent().is_some() {
        info!("PBs: landmarks of the latent (kind={})", m.kind);
        preprocess_layout_data_from_latent(args, cells, m.kind)
    } else {
        info!("PBs: projection bits of the cell projection");
        preprocess_layout_data_from_cache(args, cells)
    }
}

/// The per-cell table a layout runs on, from a run's geometry latent
/// (`cells × dims`, as read). Returns `dims × cells` (one column per cell)
/// and a short description of the transform for logging.
///
/// - topic family: `softmax(log θ / τ)` then a Hellinger square root;
/// - embedding kinds (bge, fne, …): the raw embedding, since its magnitude
///   carries signal and a DistL2 kNN should see it (the caller must have
///   resolved the table through `geometry_latent`, so this is Z, never log θ);
/// - everything else: per-dimension z-scores.
///
/// Each dimension is then winsorized to ±`trim_mads` MADs.
pub(crate) fn latent_layout_features(
    kind: senna::run_manifest::RunKind,
    latent_nk: &Mat,
    theta_temperature: f32,
    trim_mads: f32,
) -> (Mat, String) {
    let tau = theta_temperature.max(1e-6);
    let mut feat_kn: Mat = latent_nk.transpose();
    let latent_desc: String = if kind.is_topic_family() {
        // softmax(log_θ / τ) per cell → Hellinger sqrt. τ=1 reduces to
        // sqrt(exp(log_θ)); τ<1 sharpens, τ>1 softens.
        if (tau - 1.0).abs() > 1e-6 {
            feat_kn.apply(|v| *v /= tau);
        }
        feat_kn.normalize_exp_logits_columns_inplace();
        feat_kn.apply(|v| *v = v.sqrt());
        if (tau - 1.0).abs() < 1e-6 {
            "Hellinger-θ".into()
        } else {
            format!("Hellinger-θ, τ={tau:.3}")
        }
    } else if kind.cell_space() == senna::run_manifest::CellSpace::Embedding {
        // Unit-sphere/cosine normalization collapsed magnitude and sheared
        // populations apart; a raw t-UMAP on the same embedding does not.
        "raw Euclidean".into()
    } else {
        feat_kn.scale_rows_inplace();
        "z-scored scores".into()
    };
    winsorize_rows_inplace(&mut feat_kn, trim_mads);
    (feat_kn, latent_desc)
}

/// Latent-driven layout: topic θ is log-softmax on disk, so we apply
/// Hellinger (`exp().sqrt()`) to make cosine ≡ Bhattacharyya; SVD
/// scores are z-scored instead. PBs come from a random landmark
/// subsample (nearest-landmark assignment via HNSW) and PB-PB
/// similarity uses a UMAP-style fuzzy kNN graph — dense cosine on
/// Hellinger-θ saturates and collapses the layout to rank-1.
fn preprocess_layout_data_from_latent(
    args: &LayoutCommonArgs,
    cells: CellTable,
    kind: senna::run_manifest::RunKind,
) -> anyhow::Result<PbLayoutPrep> {
    let feat_kn = &cells.cells_kn;
    let n_cells = feat_kn.ncols();

    let n_pb_target = args.n_landmarks.min(n_cells).max(1);
    let use_per_topic =
        args.landmark_strategy == LandmarkStrategy::PerTopic && kind.is_topic_family();

    let landmark_cells: Vec<usize> = if use_per_topic {
        // Bucket cells by argmax topic, then pick ~equal budget from each.
        // Small topics contribute all their cells; budget is the floor.
        // feat_kn is [n_topics × n_cells] column-major, so feat_kn.column(c)
        // is contiguous — cache-friendly vs strided latent_nk row access.
        let n_topics = feat_kn.nrows();
        let argmax_per_cell: Vec<usize> = (0..n_cells)
            .into_par_iter()
            .map(|c| feat_kn.column(c).argmax().0)
            .collect();
        let mut by_topic: Vec<Vec<usize>> = vec![Vec::new(); n_topics];
        for (c, &t) in argmax_per_cell.iter().enumerate() {
            by_topic[t].push(c);
        }
        let active: Vec<usize> = (0..n_topics).filter(|&t| !by_topic[t].is_empty()).collect();
        let n_active = active.len().max(1);
        let per_topic_budget = (n_pb_target / n_active).max(1);

        let mut rng = SmallRng::seed_from_u64(args.seed);
        let mut picked: Vec<usize> = Vec::with_capacity(n_pb_target);
        for &t in &active {
            let cells = &by_topic[t];
            let take = per_topic_budget.min(cells.len());
            let sampled: Vec<usize> = rand::seq::index::sample(&mut rng, cells.len(), take)
                .into_iter()
                .map(|i| cells[i])
                .collect();
            picked.extend(sampled);
        }
        picked.sort_unstable();
        info!(
            "Per-topic landmarks: {} active topic(s) × ~{} budget = {} landmarks (seed={})",
            n_active,
            per_topic_budget,
            picked.len(),
            args.seed
        );
        picked
    } else {
        let mut rng = SmallRng::seed_from_u64(args.seed);
        let mut picked: Vec<usize> = rand::seq::index::sample(&mut rng, n_cells, n_pb_target)
            .into_iter()
            .collect();
        picked.sort_unstable();
        info!(
            "Random landmarks: picked {} cells as PBs (seed={})",
            picked.len(),
            args.seed
        );
        picked
    };
    let n_pb_full = landmark_cells.len();

    let landmark_cols: Vec<nalgebra::DVectorView<f32>> =
        landmark_cells.iter().map(|&c| feat_kn.column(c)).collect();
    let landmark_dict = legume_numeric::matrix::knn_match::ColumnDict::from_dvector_views(
        landmark_cols,
        (0..n_pb_full).collect(),
    );

    let membership_full: Vec<usize> = (0..n_cells)
        .into_par_iter()
        .map_init(
            || Vec::<f32>::with_capacity(feat_kn.nrows()),
            |query, c| {
                query.clear();
                query.extend(feat_kn.column(c).iter().copied());
                match landmark_dict.search_by_query_data(query, 1) {
                    Ok((ids, _)) => ids.first().copied().unwrap_or(usize::MAX),
                    Err(_) => usize::MAX,
                }
            },
        )
        .collect();

    let pb_size_full: Vec<usize> = {
        let mut counts = vec![0usize; n_pb_full];
        for &g in &membership_full {
            if g < n_pb_full {
                counts[g] += 1;
            }
        }
        counts
    };
    let kept_indices = select_pb_coverage(&pb_size_full, args.pb_coverage);
    let covered: usize = kept_indices.iter().map(|&i| pb_size_full[i]).sum();
    let total_cells: usize = pb_size_full.iter().sum();
    info!(
        "Coverage filter: kept {} / {} landmarks covering {} / {} cells ({:.1}%)",
        kept_indices.len(),
        n_pb_full,
        covered,
        total_cells,
        100.0 * covered as f32 / total_cells.max(1) as f32
    );

    let mut old_to_new = vec![usize::MAX; n_pb_full];
    for (new_i, &old_i) in kept_indices.iter().enumerate() {
        old_to_new[old_i] = new_i;
    }
    let pb_membership_kept: Vec<usize> = membership_full
        .iter()
        .map(|&g| {
            if g < n_pb_full {
                old_to_new[g]
            } else {
                usize::MAX
            }
        })
        .collect();
    let pb_size: Vec<usize> = kept_indices.iter().map(|&i| pb_size_full[i]).collect();

    let pb_centroids_kp =
        aggregate_features_by_group(feat_kn, &pb_membership_kept, kept_indices.len());

    let n_pb = pb_centroids_kp.ncols();
    let knn = args.knn.clamp(1, n_pb.saturating_sub(1).max(1));
    info!("Building fuzzy kNN graph on PB centroids: n_pb={n_pb}, knn={knn}");
    let (graph, fuzzy) = legume_numeric::matrix::knn_graph::KnnGraph::from_columns_fuzzy(
        &pb_centroids_kp,
        legume_numeric::matrix::knn_graph::KnnGraphArgs {
            knn,
            block_size: args.block_size.unwrap_or(1000),
            reciprocal: false,
        },
    )?;

    // √(sᵢ·sⱼ)/mean_size normalises so weighted mean ≈ 1, keeping fuzzy
    // magnitudes comparable to the unweighted case.
    let mean_size = (total_cells as f32) / (n_pb as f32).max(1.0);
    let sqrt_size: Vec<f32> = pb_size.iter().map(|&s| (s as f32).sqrt()).collect();

    let mut sim = Mat::zeros(n_pb, n_pb);
    for (&(i, j), &w) in graph.edges.iter().zip(fuzzy.iter()) {
        let scale = (sqrt_size[i] * sqrt_size[j]) / mean_size.max(1e-6);
        let ws = w * scale;
        sim[(i, j)] = ws;
        sim[(j, i)] = ws;
    }
    let pb_similarity = regularize_similarity(&sim, SELF_LOOP_REG);

    Ok(PbLayoutPrep {
        cells,
        pb_size,
        pb_membership_kept,
        pb_similarity,
        pb_kp: pb_centroids_kp,
    })
}

/// PBs of a run's cached cell projection (a run without a latent): cells
/// partitioned by hashing the projection, PB features and similarity in
/// projection space.
fn preprocess_layout_data_from_cache(
    args: &LayoutCommonArgs,
    cells: CellTable,
) -> anyhow::Result<PbLayoutPrep> {
    let proj_kn = &cells.cells_kn;

    // 3. Partition: binary_sort_columns runs RSVD + sign-hashing on the
    //    projection and returns a per-cell bucket code. Canonicalize
    //    those codes to contiguous PB ids in [0, n_pb). No data_vec
    //    groups needed — the partition is a pure function of proj_kn.
    let n_cells = proj_kn.ncols();
    let kk = args.sort_dim.min(proj_kn.nrows()).min(n_cells);
    let codes = binary_sort_columns(proj_kn, kk)?;
    let (membership_full, n_pb_full) = canonicalize_codes(&codes);
    info!("Partitioned {n_cells} cells into {n_pb_full} PBs");

    // 4. Coverage-prune PBs (same policy as the latent path).
    let pb_size_full: Vec<usize> = {
        let mut counts = vec![0usize; n_pb_full];
        for &g in &membership_full {
            if g < n_pb_full {
                counts[g] += 1;
            }
        }
        counts
    };
    let kept_indices = select_pb_coverage(&pb_size_full, args.pb_coverage);
    let covered: usize = kept_indices.iter().map(|&i| pb_size_full[i]).sum();
    let total_cells: usize = pb_size_full.iter().sum();
    info!(
        "Coverage filter: kept {} / {} PBs covering {} / {} cells ({:.1}%)",
        kept_indices.len(),
        n_pb_full,
        covered,
        total_cells,
        100.0 * covered as f32 / total_cells.max(1) as f32
    );

    let mut old_to_new = vec![usize::MAX; n_pb_full];
    for (new_i, &old_i) in kept_indices.iter().enumerate() {
        old_to_new[old_i] = new_i;
    }
    let pb_membership_kept: Vec<usize> = membership_full
        .iter()
        .map(|&g| {
            if g < n_pb_full {
                old_to_new[g]
            } else {
                usize::MAX
            }
        })
        .collect();
    let pb_size: Vec<usize> = kept_indices.iter().map(|&i| pb_size_full[i]).collect();

    // 5. PB centroids in proj space (kp = proj_dim × n_pb_kept).
    let pb_centroids_kp =
        aggregate_features_by_group(proj_kn, &pb_membership_kept, kept_indices.len());

    // 6. PB-PB cosine similarity directly on the proj-space centroids —
    //    columns of `pb_centroids_kp` are the PB vectors, which is the
    //    convention `compute_cosine_similarity` expects.
    let sim = compute_cosine_similarity(&pb_centroids_kp);
    let sim = if args.similarity_threshold > 0.0 {
        threshold_similarity(&sim, args.similarity_threshold)
    } else {
        sim
    };
    let sim = if let Some(k) = args.local_scale_k {
        let scaled = local_scale_similarity(&sim, k);
        info!("Applied local scaling with k={k}");
        scaled
    } else {
        sim
    };
    let pb_similarity = regularize_similarity(&sim, SELF_LOOP_REG);

    Ok(PbLayoutPrep {
        cells,
        pb_size,
        pb_membership_kept,
        pb_similarity,
        pb_kp: pb_centroids_kp,
    })
}

/// Robustly winsorize each feature dimension (row) of a column-per-cell matrix
/// to `median ± n_mads · MAD · 1.4826`, so a few extreme-outlier cells can't
/// stretch the layout or dominate the PB-PB similarity. `n_mads <= 0` (or fewer
/// than 8 cells) is a no-op. MAD-based ⇒ no Gaussian assumption; only the
/// extreme tails are clipped.
fn winsorize_rows_inplace(feat: &mut Mat, n_mads: f32) {
    let (d, n) = (feat.nrows(), feat.ncols());
    if n_mads <= 0.0 || n < 8 {
        return;
    }
    let mut clipped = 0usize;
    let mut buf: Vec<f32> = Vec::with_capacity(n);
    for r in 0..d {
        buf.clear();
        buf.extend((0..n).map(|c| feat[(r, c)]));
        let med = median_of(&mut buf);
        buf.iter_mut().for_each(|v| *v = (*v - med).abs());
        let mad = median_of(&mut buf) * 1.4826;
        if mad <= 1e-12 {
            continue;
        }
        let (lo, hi) = (med - n_mads * mad, med + n_mads * mad);
        for c in 0..n {
            let v = feat[(r, c)];
            if v < lo {
                feat[(r, c)] = lo;
                clipped += 1;
            } else if v > hi {
                feat[(r, c)] = hi;
                clipped += 1;
            }
        }
    }
    if clipped > 0 {
        info!("Winsorized {clipped} outlier feature value(s) to ±{n_mads} MADs ({d} dims × {n} cells)");
    }
}

/// Median via quickselect on a scratch buffer (mutates its order). 0 for empty.
fn median_of(buf: &mut [f32]) -> f32 {
    if buf.is_empty() {
        return 0.0;
    }
    let mid = buf.len() / 2;
    buf.select_nth_unstable_by(mid, |a, b| {
        a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
    });
    buf[mid]
}

/// Canonicalize raw bucket codes from `binary_sort_columns` into
/// contiguous PB ids `[0, n_pb)`. Returns `(membership, n_pb)`.
fn canonicalize_codes(codes: &[usize]) -> (Vec<usize>, usize) {
    use std::collections::HashMap;
    let mut mapping: HashMap<usize, usize> = HashMap::new();
    let mut membership = Vec::with_capacity(codes.len());
    for &c in codes {
        let next_id = mapping.len();
        let id = *mapping.entry(c).or_insert(next_id);
        membership.push(id);
    }
    let n = mapping.len();
    (membership, n)
}

/// Load cluster assignments (one-column parquet of cell IDs) and validate length.
pub(crate) fn load_cluster_assignments(
    path: &str,
    n_cells_expected: usize,
) -> anyhow::Result<Vec<usize>> {
    info!("Reading cluster assignments from {path}...");
    let MatWithNames {
        rows: _cluster_cell_names,
        cols: _,
        mat: cluster_mat,
    } = Mat::from_parquet(path)?;
    if cluster_mat.nrows() != n_cells_expected {
        anyhow::bail!(
            "Cluster file has {} cells but data has {} cells",
            cluster_mat.nrows(),
            n_cells_expected
        );
    }
    let clusters: Vec<usize> = (0..cluster_mat.nrows())
        .map(|i| cluster_mat[(i, 0)] as usize)
        .collect();
    info!("Loaded {} cluster assignments", clusters.len());
    Ok(clusters)
}

/// Uniform-random 2D coordinate init for PB-level layouts, in `[-1, 1]²`.
/// Shared by `layout tsne` and `layout mst` so both are reproducible
/// via the common `--seed` flag.
pub(crate) fn random_init_2d(n: usize, seed: u64) -> Mat {
    use rand::rngs::SmallRng;
    use rand::{RngExt, SeedableRng};
    let mut rng = SmallRng::seed_from_u64(seed);
    let mut out = Mat::zeros(n, 2);
    for i in 0..n {
        out[(i, 0)] = rng.random_range(-1.0..1.0);
        out[(i, 1)] = rng.random_range(-1.0..1.0);
    }
    out
}

/// Cheap Nyström cell placement: each cell from its nearest PBs in the cell
/// table's space.
fn nystrom_cell_coords(args: &LayoutCommonArgs, prep: &PbLayoutPrep, pb_coords: &Mat) -> Mat {
    let coords = project_cells_nystrom(
        &prep.cells.cells_kn,
        &prep.pb_kp,
        pb_coords,
        args.knn,
        args.kernel_alpha,
    );
    info!("Placed cells by Nyström");
    coords
}

/// Finish a landmark layout: place cells by Nyström, write the PB and cell
/// coordinates, and record them in the run's manifest.
pub(crate) fn finalize_viz(
    args: &LayoutCommonArgs,
    resolved: &mut ResolvedViz,
    prep: &PbLayoutPrep,
    pb_coords: &Mat,
    method: &str,
) -> anyhow::Result<()> {
    let cell_coords = nystrom_cell_coords(args, prep, pb_coords);
    write_viz_outputs_pb(args, resolved, prep, pb_coords, &cell_coords, method)
}

fn write_viz_outputs_pb(
    args: &LayoutCommonArgs,
    resolved: &mut ResolvedViz,
    prep: &PbLayoutPrep,
    pb_coords: &Mat,
    cell_coords: &Mat,
    method: &str,
) -> anyhow::Result<()> {
    let pb_names: Vec<Box<str>> = (0..pb_coords.nrows())
        .map(|i| format!("PB_{i}").into_boxed_str())
        .collect();
    let cell_names = &prep.cells.names;

    let out = &resolved.out;
    let pb_coords_path = format!("{out}.{method}.pb_coords.parquet");
    let cell_coords_path = format!("{out}.{method}.cell_coords.parquet");

    let coord_cols: Vec<Box<str>> = vec!["x".into(), "y".into()];
    pb_coords.to_parquet_with_names(
        &pb_coords_path,
        (Some(&pb_names), Some("pb")),
        Some(&coord_cols),
    )?;

    // Diagnostic: not recorded in the manifest.
    let pb_feat_path = format!("{out}.pb_proj_mean.parquet");
    let pb_features = prep.pb_kp.transpose();
    let feat_col_names: Vec<Box<str>> = (0..pb_features.ncols())
        .map(|i| format!("p{i}").into_boxed_str())
        .collect();
    pb_features.to_parquet_with_names(
        &pb_feat_path,
        (Some(&pb_names), Some("pb")),
        Some(&feat_col_names),
    )?;

    let cluster_ids = match &args.clusters {
        Some(path) => Some(load_cluster_assignments(
            path,
            prep.pb_membership_kept.len(),
        )?),
        None => None,
    };

    let n_cells = cell_coords.nrows();
    let n_extra = 1 + cluster_ids.as_ref().map_or(0, |_| 1);
    let mut cell_out = Mat::zeros(n_cells, 2 + n_extra);
    for i in 0..n_cells {
        cell_out[(i, 0)] = cell_coords[(i, 0)];
        cell_out[(i, 1)] = cell_coords[(i, 1)];
        let pb_id = prep.pb_membership_kept[i];
        // NaN flags orphan cells (pb dropped by coverage filter); downstream
        // readers can mask with `is_nan` without collision against real IDs.
        cell_out[(i, 2)] = if pb_id == usize::MAX {
            f32::NAN
        } else {
            pb_id as f32
        };
    }
    let mut col_names: Vec<Box<str>> = vec!["x".into(), "y".into(), "pb_id".into()];
    if let Some(ref clusters) = cluster_ids {
        for (i, &c) in clusters.iter().enumerate() {
            cell_out[(i, 3)] = c as f32;
        }
        col_names.push("cluster".into());
    }
    cell_out.to_parquet_with_names(
        &cell_coords_path,
        (Some(cell_names), Some("cell")),
        Some(&col_names),
    )?;

    info!("Saved {pb_coords_path}, {pb_feat_path}, {cell_coords_path}");

    let feature_path = place_features_on_cells(
        args,
        &resolved.manifest,
        &resolved.manifest_path,
        out,
        method,
        &prep.cells.cells_kn,
        cell_coords,
    )?;

    resolved.record(
        method,
        &LayoutEntry {
            cell_coords: Some(cell_coords_path),
            pb_coords: Some(pb_coords_path),
            feature_on_cell_coords: feature_path,
            feature_coords: None,
        },
    )?;

    Ok(())
}

pub(crate) fn write_cell_layout(
    args: &LayoutCommonArgs,
    resolved: &mut ResolvedViz,
    prep: &CellTable,
    cell_coords: &Mat,
    method: &str,
    placed_features: Option<(&[Box<str>], &Mat)>,
) -> anyhow::Result<()> {
    let cell_names = &prep.names;
    let n_cells = cell_coords.nrows();
    anyhow::ensure!(
        cell_names.len() == n_cells,
        "cell_coords rows ({n_cells}) ≠ cells ({})",
        cell_names.len()
    );

    let out = &resolved.out;
    let cell_coords_path = format!("{out}.{method}.cell_coords.parquet");

    let cluster_ids = match &args.clusters {
        Some(path) => Some(load_cluster_assignments(path, n_cells)?),
        None => None,
    };

    let n_extra = cluster_ids.as_ref().map_or(0, |_| 1);
    let mut cell_out = Mat::zeros(n_cells, 2 + n_extra);
    cell_out.column_mut(0).copy_from(&cell_coords.column(0));
    cell_out.column_mut(1).copy_from(&cell_coords.column(1));

    let mut col_names: Vec<Box<str>> = vec!["x".into(), "y".into()];
    if let Some(ref clusters) = cluster_ids {
        for (i, &c) in clusters.iter().enumerate() {
            cell_out[(i, 2)] = c as f32;
        }
        col_names.push("cluster".into());
    }

    cell_out.to_parquet_with_names(
        &cell_coords_path,
        (Some(cell_names), Some("cell")),
        Some(&col_names),
    )?;

    info!("Saved {cell_coords_path}");

    // Features laid out with the cells (a joint layout) keep their own
    // coordinates; otherwise they are placed on the finished cell map.
    let feature_path = match placed_features {
        Some((names, coords)) => Some(write_features_on_cells(out, method, names, coords)?),
        None => place_features_on_cells(
            args,
            &resolved.manifest,
            &resolved.manifest_path,
            out,
            method,
            &prep.cells_kn,
            cell_coords,
        )?,
    };

    resolved.record(
        method,
        &LayoutEntry {
            cell_coords: Some(cell_coords_path),
            feature_on_cell_coords: feature_path,
            ..Default::default()
        },
    )?;

    Ok(())
}
