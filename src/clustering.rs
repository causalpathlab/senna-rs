//! Clustering command for single-cell data
//!
//! Cluster cells based on latent representations (topic proportions, SVD
//! embeddings), or, with `--target features`, the run's feature embedding.

use crate::cluster_bhc::{run_cluster_bhc, ClusterBhcConfig};
use crate::postprocess::fit_layout_features::{read_feature_rows, FeatureSpace};
use anyhow::Context;
use senna::cluster::{
    hsblock_clustering, kmeans_clustering, leiden_clustering, ClusterResult, LatentMetric,
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
        help = "Latent representation file (cells × K) [default: the --from run's cell table]; not with --target features",
        long_help = "Latent topic proportions or SVD projection (cells × K matrix).\n\
                     Used as feature space for clustering.\n\
                     \n\
                     Expected formats:\n\
                     - From `senna topic`: .latent.parquet (cells × topics)\n\
                     - From `senna svd`: .projection.parquet (cells × components)\n\
                     - First column: cell names\n\
                     \n\
                     With --from and no --latent: the run's cell table, as its layouts\n\
                     read it (its cell embedding, else its latent)."
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

    #[arg(long, help = "Random seed for k-means and graph-based clustering")]
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
        required_unless_present = "from",
        help = "Output file prefix [default: the --from run's own]",
        long_help = "Output file prefix; with --from and no --out, the run's own\n\
                     (the manifest's path without `.senna.json`).\n\
                     \n\
                     Generates:\n\
                     - {out}.clusters.parquet: Cluster assignments (cell × cluster)\n\
                     - {out}.bhc.merges.parquet: BHC merge tree (when --data is given)\n\
                     - {out}.bhc.cut.parquet:    BHC consensus cut (when --data is given)"
    )]
    out: Option<Box<str>>,

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
        long_help = "When given and the output goes beside the run (its directory), the\n\
                     manifest is updated in place with the cluster output path, under\n\
                     `cluster.clusters`, so `lupin annotate` picks the cluster parquet up\n\
                     automatically. With --out in another directory it is left as it was."
    )]
    from: Option<Box<str>>,
}

pub fn run_clustering(args: &ClusteringArgs) -> anyhow::Result<()> {
    let out = args.out().context("--out is required without --from")?;
    mkdir_parent(&out)?;
    let (cell_names, latent) = match (args.target, args.latent.as_deref(), args.from.as_deref()) {
        (ClusterTarget::Cells, latent, from) => {
            let latent = match (latent, from) {
                (Some(latent), _) => latent.to_string(),
                (None, Some(from)) => run_cell_table(from)?,
                (None, None) => {
                    anyhow::bail!("--latent (or --from, a run) is required to cluster cells")
                }
            };
            let MatWithNames { rows, mat, .. } = read_mat(&latent)?;
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
        (ClusterTarget::Features, _, None) => {
            anyhow::bail!("--target features clusters the features of a run: pass --from")
        }
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
            kmeans_clustering(&latent, k, args.max_iter, args.seed)?
        }
        ClusterMethodCli::Leiden => {
            info!(
                "Running Leiden clustering with knn={}, resolution={:.2}, target_k={:?}",
                args.knn, args.resolution, args.num_clusters
            );
            leiden_clustering(
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
    let output_file = format!("{out}.{suffix}.parquet");
    write_cluster_assignments(&result, &cell_names, &output_file, row_label)?;

    info!("Wrote cluster assignments to {output_file}");

    if let Some(from) = args
        .from
        .as_deref()
        .filter(|from| outputs_beside_run(from, &out))
    {
        update_manifest_cluster_path(from, &output_file, args.target)?;
    }

    if let Some(data_files) = args.data_files.as_ref() {
        run_bhc_postprocess(data_files, &result, &cell_names, &out, args)?;
    }

    Ok(())
}

impl ClusteringArgs {
    /// The output prefix: `--out`, else the `--from` run's own.
    fn out(&self) -> Option<String> {
        self.out.as_deref().map(String::from).or_else(|| {
            self.from
                .as_deref()
                .map(senna::run_manifest::derive_out_prefix)
        })
    }
}

/// The cell table of the run whose manifest is `from`, as its layouts read
/// it: its geometry latent (the cell embedding, else the latent).
fn run_cell_table(from: &str) -> anyhow::Result<String> {
    let (manifest, dir) = RunManifest::load(Path::new(from))?;
    let rel = manifest
        .outputs
        .geometry_latent()
        .with_context(|| format!("{from}: the run records no cell table to cluster"))?;
    Ok(senna::run_manifest::resolve(&dir, rel)
        .to_string_lossy()
        .into_owned())
}

/// Whether `out` sits beside the run `from` names, so its manifest records
/// the clusters; with `--out` elsewhere the run is left as it was.
fn outputs_beside_run(from: &str, out: &str) -> bool {
    let manifest = senna::run_manifest::manifest_path_for(from);
    let beside = senna::run_manifest::outputs_beside(Path::new(&manifest), out);
    if !beside {
        info!("--out {out} is outside the run's directory: {manifest} is left as it was");
    }
    beside
}

/// Update the run manifest in place with the cluster parquet path. Path is
/// stored relative to the manifest directory so the run dir stays portable.
fn update_manifest_cluster_path(
    manifest_path: &str,
    cluster_path: &str,
    target: ClusterTarget,
) -> anyhow::Result<()> {
    RunManifest::update(Path::new(manifest_path), |m, rel| {
        *target.slot(&mut m.cluster) = Some(rel(cluster_path));
    })?;
    info!("Updated manifest {manifest_path} with cluster path");
    Ok(())
}

fn run_bhc_postprocess(
    data_files: &[Box<str>],
    result: &ClusterResult,
    cell_names: &[Box<str>],
    out: &str,
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
        out,
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

#[cfg(test)]
#[path = "tests/clustering.rs"]
mod tests;
