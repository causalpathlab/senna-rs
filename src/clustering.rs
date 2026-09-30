//! Clustering command for single-cell data
//!
//! Cluster cells based on latent representations (topic proportions, SVD
//! embeddings), or, with `--target features`, the run's feature embedding.

use crate::cluster_bhc::{run_cluster_bhc, ClusterBhcConfig};
use crate::postprocess::fit_layout_features::{read_feature_rows, FeatureSpace};
use senna::cluster::{
    hsblock_clustering, kmeans_clustering, leiden_clustering_with_metric, ClusterMethod,
    ClusterResult, LatentMetric,
};
use senna::embed_common::*;
use senna::run_manifest::{RunCluster, RunManifest};
use senna::senna_input::{read_data_on_shared_columns, ReadSharedColumnsArgs};
use std::path::Path;

/// Clustering method CLI enum
#[derive(ValueEnum, Clone, Debug, Default, PartialEq)]
#[clap(rename_all = "lowercase")]
pub enum ClusterMethodCli {
    /// K-means clustering
    #[default]
    Kmeans,
    /// Leiden clustering (graph-based)
    Leiden,
    /// Hierarchical Stochastic Block Model (graph-based)
    Hsblock,
}

impl From<ClusterMethodCli> for ClusterMethod {
    fn from(cli: ClusterMethodCli) -> Self {
        match cli {
            ClusterMethodCli::Kmeans => ClusterMethod::KMeans,
            ClusterMethodCli::Leiden => ClusterMethod::Leiden,
            ClusterMethodCli::Hsblock => ClusterMethod::Hsblock,
        }
    }
}

/// What `senna clustering` groups.
#[derive(ValueEnum, Clone, Copy, Debug, Default, PartialEq, Eq)]
#[clap(rename_all = "kebab-case")]
pub enum ClusterTarget {
    /// Cells, on --latent (the default).
    #[default]
    Cells,
    /// The run's features, by cosine (see --feature-space); needs --from.
    Features,
}

impl ClusterTarget {
    /// Output file suffix and the name of its row column.
    fn output(self) -> (&'static str, &'static str) {
        match self {
            Self::Cells => ("clusters", "cell"),
            Self::Features => ("feature_clusters", "feature"),
        }
    }

    /// The kNN metric for leiden: cells as ever; features by angle, as
    /// their layout compares them.
    fn metric(self) -> LatentMetric {
        match self {
            Self::Cells => LatentMetric::ZScoreEuclidean,
            Self::Features => LatentMetric::Cosine,
        }
    }

    /// The manifest slot the output is recorded in.
    fn slot(self, cluster: &mut RunCluster) -> &mut Option<String> {
        match self {
            Self::Cells => &mut cluster.clusters,
            Self::Features => &mut cluster.feature_clusters,
        }
    }
}

#[derive(Args, Debug)]
pub struct ClusteringArgs {
    #[arg(
        long,
        value_enum,
        default_value = "cells",
        help = "What to cluster: cells (on --latent) or features (the run's feature embedding)",
        long_help = "What to cluster.\n\
                     \n\
                     - cells: rows of --latent (the default).\n\
                     - features: the features of the --from run (`senna bge`,\n\
                     `senna fne`, ...): the co-embedding when the run has one,\n\
                     else ρ (see --feature-space); rows L2-normalized so\n\
                     distances are cosine; leiden is the usual choice.\n\
                     Writes {out}.feature_clusters.parquet and records it\n\
                     under `cluster.feature_clusters`, where `senna view`\n\
                     colours feature maps by it."
    )]
    target: ClusterTarget,

    #[arg(
        long,
        value_enum,
        default_value = "auto",
        help = "With --target features: the co-embedding (auto, when the run has one) or ρ",
        long_help = "Which table of the run's features --target features clusters, as\n\
                     `senna layout --target features` lays them out: auto (default) is the\n\
                     co-embedding when the run wrote one, else ρ; or coembedding, or rho."
    )]
    feature_space: FeatureSpace,

    #[arg(
        long,
        short = 'l',
        help = "Latent representation file (cells × K); not with --target features",
        long_help = "Latent topic proportions or SVD projection (cells × K matrix).\n\
                     Used as feature space for clustering.\n\
                     \n\
                     Expected formats:\n\
                     - From `senna topic`: .latent.parquet (cells × topics)\n\
                     - From `senna svd`: .projection.parquet (cells × components)\n\
                     - First column: cell names"
    )]
    latent: Option<Box<str>>,

    #[arg(
        long,
        short = 'k',
        help = "Number of clusters",
        long_help = "Number of clusters for k-means. If not specified,\n\
                     defaults to the number of topics/components in latent.\n\
                     \n\
                     Tuning:\n\
                     - Start with number of expected cell types\n\
                     - Use silhouette score or elbow method to optimize\n\
                     - For hierarchical data, start with broader clusters"
    )]
    num_clusters: Option<usize>,

    #[arg(
        long,
        short = 'm',
        default_value = "kmeans",
        help = "Clustering method",
        long_help = "Clustering algorithm:\n\
                     \n\
                     - kmeans: K-means clustering (default)\n\
                     Fast, and works well for spherical clusters.\n\
                     Requires specifying k.\n\
                     \n\
                     - leiden: Leiden algorithm (graph-based)\n\
                     Finds communities in the cell similarity graph,\n\
                     determining the number of clusters automatically.\n\
                     Use --knn and --resolution to tune.\n\
                     \n\
                     - hsblock: Hierarchical Stochastic Block Model (graph-based)\n\
                     Collapsed Gibbs sampling plus greedy refinement.\n\
                     Number of clusters = 2^(tree_depth-1).\n\
                     Use --knn, --tree-depth, and --edge-scale to tune"
    )]
    method: ClusterMethodCli,

    #[arg(long, default_value_t = 100, help = "Maximum iterations for k-means")]
    max_iter: usize,

    #[arg(
        long,
        default_value_t = 15,
        help = "Number of nearest neighbors for graph-based clustering (Leiden/Hsblock)"
    )]
    knn: usize,

    #[arg(
        long,
        default_value_t = 1.0,
        help = "Resolution parameter for Leiden modularity",
        long_help = "Resolution parameter for Leiden modularity.\n\
                     Higher gives more clusters. The default is 1.0."
    )]
    resolution: f64,

    #[arg(
        long,
        default_value_t = 3,
        help = "Tree depth for HSBM (clusters = 2^(depth-1), default 3 → 4 clusters)"
    )]
    tree_depth: usize,

    #[arg(
        long,
        default_value_t = true,
        help = "Use degree-corrected HSBM (default true)"
    )]
    degree_corrected: bool,

    #[arg(
        long,
        default_value_t = 100.0,
        help = "Edge weight scale for HSBM (default 100.0)",
        long_help = "Edge weight scale for HSBM (default 100.0).\n\
                     It scales fuzzy KNN weights to count-like values.\n\
                     Those are what the graph-tool backend expects."
    )]
    edge_scale: f64,

    #[arg(long, help = "Random seed for graph-based clustering")]
    seed: Option<u64>,

    #[arg(
        long,
        default_value_t = 2,
        help = "Minimum cluster size to report (default 2)",
        long_help = "Minimum cluster size to report;\n\
                     smaller clusters become unassigned (default 2)."
    )]
    min_cluster_size: usize,

    #[arg(
        long,
        short = 'o',
        required = true,
        help = "Output file prefix",
        long_help = "Output file prefix.\n\
                     \n\
                     Generates:\n\
                     - {out}.clusters.parquet: Cluster assignments (cell × cluster)\n\
                     - {out}.bhc.merges.parquet: BHC merge tree (when --data is given)\n\
                     - {out}.bhc.cut.parquet:    BHC consensus cut (when --data is given)"
    )]
    out: Box<str>,

    #[arg(
        long,
        value_delimiter = ',',
        help = "Raw count data files (.zarr or .h5) — enables BHC postprocess",
        long_help = "When provided,\n\
                     a Bayesian hierarchical clustering pass runs over the fitted clusters,\n\
                     using per-gene sufficient stats T_{k,g}=Σ_{n∈k} y_{n,g},\n\
                     and an empirical-Bayes Dirichlet prior centred on the pooled gene marginal bg.\n\
                     Same recipe pinto uses for link communities.\n\
                     Must match the cell order of --latent. Omit to skip BHC."
    )]
    data_files: Option<Vec<Box<str>>>,

    #[arg(
        long,
        default_value_t = 1.0,
        help = "Per-gene prior strength for the BHC empirical-Bayes Dirichlet prior",
        long_help = "Total Dirichlet concentration γ = bhc_gamma_per_gene × G,\n\
                     where G is the feature dimension.\n\
                     Default 1.0 = Bayes-Laplace (one prior count per gene).\n\
                     Larger values pull every cluster more strongly toward the pooled background,\n\
                     making BHC more eager to merge."
    )]
    bhc_gamma_per_gene: f64,

    #[arg(
        long,
        default_value_t = 0.0,
        help = "log-BF cutoff for the BHC consensus cut (0 = natural Bayes break)"
    )]
    bhc_cut: f64,

    #[arg(
        long,
        default_value_t = 1024,
        help = "Cells per CSC read block when computing BHC sufficient stats"
    )]
    bhc_block_size: usize,

    #[arg(
        long = "from",
        required_if_eq("target", "features"),
        help = "Run manifest from `senna topic|masked-topic|joint-topic|svd|joint-svd`",
        long_help = "When given, the manifest is updated in place with the cluster output path,\n\
                     under `cluster.clusters`,\n\
                     so `lupin annotate` picks the cluster parquet up automatically."
    )]
    from: Option<Box<str>>,
}

pub fn run_clustering(args: &ClusteringArgs) -> anyhow::Result<()> {
    mkdir_parent(&args.out)?;
    let (cell_names, latent) = match (args.target, args.latent.as_deref(), args.from.as_deref()) {
        (ClusterTarget::Cells, Some(latent), _) => {
            let MatWithNames { rows, mat, .. } = read_mat(latent)?;
            info!(
                "Loaded latent representation: {} cells × {} features",
                mat.nrows(),
                mat.ncols()
            );
            (rows, mat)
        }
        (ClusterTarget::Features, _, Some(from)) => {
            anyhow::ensure!(
                args.data_files.is_none(),
                "--data (BHC over cell counts) does not apply to --target features"
            );
            let (manifest, dir) = RunManifest::load(Path::new(from))?;
            read_feature_rows(&manifest, &dir, args.feature_space)?
        }
        // clap requires --from for an explicit --target features; the
        // default target is not seen by its conditions.
        _ => anyhow::bail!("--latent is required to cluster cells"),
    };

    // Determine number of clusters
    let k = args.num_clusters.unwrap_or_else(|| {
        let default_k = latent.ncols();
        info!("Number of clusters not specified, using {default_k} (number of features)");
        default_k
    });

    // Run clustering
    let mut result = match args.method {
        ClusterMethodCli::Kmeans => {
            info!(
                "Running k-means clustering with k={}, max_iter={}",
                k, args.max_iter
            );
            kmeans_clustering(&latent, k, args.max_iter)?
        }
        ClusterMethodCli::Leiden => {
            info!(
                "Running Leiden clustering with knn={}, resolution={:.2}, target_k={:?}",
                args.knn, args.resolution, args.num_clusters
            );
            leiden_clustering_with_metric(
                &latent,
                args.knn,
                args.resolution,
                args.num_clusters,
                args.seed,
                args.target.metric(),
            )?
        }
        ClusterMethodCli::Hsblock => {
            info!(
                "Running HSBM clustering with knn={}, tree_depth={}, degree_corrected={}",
                args.knn, args.tree_depth, args.degree_corrected
            );
            hsblock_clustering(
                &latent,
                args.knn,
                args.tree_depth,
                args.degree_corrected,
                args.edge_scale,
                args.seed,
            )?
        }
    };

    // Remove small clusters
    if args.min_cluster_size > 1 {
        result.remove_small_clusters(args.min_cluster_size);
    }

    let n_unassigned = result.labels.iter().filter(|&&l| l == usize::MAX).count();
    info!(
        "Clustering complete: {} cells assigned to {} clusters ({} unassigned)",
        result.labels.len(),
        result.n_clusters,
        n_unassigned,
    );

    // Display cluster statistics when verbose logging is enabled (top 100 biggest)
    if log::log_enabled!(log::Level::Info) {
        eprintln!();
        eprintln!("{}", result.histogram_ascii(50, 100));
        eprintln!();
    }

    // Output cluster assignments as parquet
    let (suffix, row_label) = args.target.output();
    let output_file = format!("{}.{suffix}.parquet", args.out);
    write_cluster_assignments(&result, &cell_names, &output_file, row_label)?;

    info!("Wrote cluster assignments to {output_file}");

    if let Some(from) = args.from.as_deref() {
        update_manifest_cluster_path(from, &output_file, args.target)?;
    }

    if let Some(data_files) = args.data_files.as_ref() {
        run_bhc_postprocess(data_files, &result, &cell_names, args)?;
    }

    Ok(())
}

/// Update the run manifest in place with the cluster parquet path. Path is
/// stored relative to the manifest directory so the run dir stays portable.
fn update_manifest_cluster_path(
    manifest_path: &str,
    cluster_path: &str,
    target: ClusterTarget,
) -> anyhow::Result<()> {
    let path = Path::new(manifest_path);
    let (mut manifest, manifest_dir) = RunManifest::load(path)?;
    let rel = Path::new(cluster_path)
        .strip_prefix(&manifest_dir)
        .map_or_else(
            |_| cluster_path.to_string(),
            |p| p.to_string_lossy().into_owned(),
        );
    *target.slot(&mut manifest.cluster) = Some(rel);
    manifest.save(path)?;
    info!("Updated manifest {manifest_path} with cluster path");
    Ok(())
}

fn run_bhc_postprocess(
    data_files: &[Box<str>],
    result: &ClusterResult,
    cell_names: &[Box<str>],
    args: &ClusteringArgs,
) -> anyhow::Result<()> {
    info!(
        "BHC: loading raw count data from {} file(s)",
        data_files.len()
    );
    let stack = read_data_on_shared_columns(ReadSharedColumnsArgs {
        data_files: data_files.to_vec(),
        batch_files: None,
        num_types: 1,
        preload: true,
        // Clustering reads an existing cell set — never drop cells.
        qc: None,
        qc_block_size: None,
        qc_report_out: None,
    })?;
    anyhow::ensure!(
        stack.data_stack.stack.len() == 1,
        "BHC: expected a single data stack, got {}",
        stack.data_stack.stack.len()
    );
    let data_vec = &stack.data_stack.stack[0];
    anyhow::ensure!(
        data_vec.num_columns() == cell_names.len(),
        "BHC: data has {} cells but latent has {}",
        data_vec.num_columns(),
        cell_names.len()
    );

    run_cluster_bhc(
        data_vec,
        &result.labels,
        &result.cluster_sizes(),
        &args.out,
        &ClusterBhcConfig {
            gamma_per_gene: args.bhc_gamma_per_gene,
            cutoff: args.bhc_cut,
            block_size: args.bhc_block_size,
        },
    )
}

/// Write cluster assignments to parquet
fn write_cluster_assignments(
    result: &ClusterResult,
    cell_names: &[Box<str>],
    output_path: &str,
    row_label: &str,
) -> anyhow::Result<()> {
    // Create a simple matrix: rows × 1 column (cluster id, NaN if unassigned)
    let mut data = Mat::zeros(cell_names.len(), 1);
    for (i, &cluster_id) in result.labels.iter().enumerate() {
        data[(i, 0)] = if cluster_id == usize::MAX {
            f32::NAN
        } else {
            cluster_id as f32
        };
    }

    let col_names = vec!["cluster".into()];
    data.to_parquet_with_names(
        output_path,
        (Some(cell_names), Some(row_label)),
        Some(&col_names),
    )?;

    Ok(())
}
