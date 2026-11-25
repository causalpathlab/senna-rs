//! Shared plumbing for `senna layout {tsne, phate, umap, ...}`:
//! - CLI args shared by the layouts.
//! - The cell table to lay out, from the run's own outputs: its latent (or
//!   cell embedding), else its cached cell projection. The count data is
//!   never read, so a run can be laid out where its data is not.
//! - The run's PBs as legume's [`Coarsened`]: its cell kNN graph coarsened
//!   (or, without a latent, projection bits with the tail pruned by legume's
//!   [`groups_covering`]), and the graph between them.
//! - Shared output writer.
//!
//! Every layout subcommand delegates here for everything except the
//! actual 2D layout algorithm.

use super::fit_layout_features::{
    place_features_on_cells, record_cell_layout, write_features_on_cells, FeatureSpace,
    LayoutTarget,
};
use data_beans::alg::random_projection::binary_sort_columns;
use legume_numeric::matrix::common_io::read_lines;
use legume_numeric::matrix::dmatrix_util::winsorize_rows_inplace;
use legume_numeric::matrix::knn::all_pairs::knn_rows_l2;
use legume_numeric::matrix::layout::{
    coarsened, fuzzy_graph_of_columns, groups_covering, CoarsenArgs, Coarsened, PhateArgs,
};
use senna::embed_common::*;
use senna::run_manifest::{self, LayoutEntry, RunManifest};
use std::path::PathBuf;

/// Default winsorization of layout features, in MADs (`--trim-cell-mads`).
pub(crate) const DEFAULT_TRIM_MADS: f32 = 5.0;
pub(crate) const DEFAULT_SORT_DIM: usize = 10;
pub(crate) const DEFAULT_PB_COVERAGE: f32 = 0.95;
pub(crate) const DEFAULT_THETA_TEMPERATURE: f32 = 1.0;

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
                     \x20 (bge, simba, resolve-embedding-space), else ρ.\n\
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
        default_value_t = DEFAULT_SORT_DIM,
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
        short = 'k',
        alias = "umap-knn",
        default_value_t = 15,
        help = "Neighbours per cell, counting itself, in the cell kNN graph",
        long_help = "Neighbours per cell in the cell kNN graph every layout reads (counting\n\
                     the cell itself, as uwot's n_neighbors): the PB layouts coarsen and\n\
                     collapse it. Also the neighbours of a --target features or --joint UMAP."
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
        default_value_t = 3,
        help = "Nearest PBs that place each cell after a PB layout",
        long_help = "How many nearest PBs (in the feature space) place each cell after a\n\
                     PB layout, by an alpha-decay kernel over their layout coordinates.\n\
                     Keep it small: a cell averaged over many PBs from two groups lands\n\
                     between them, as a streak across the layout."
    )]
    pub place_knn: usize,

    #[arg(
        long,
        default_value_t = DEFAULT_PB_COVERAGE,
        help = "Keep smallest PB set covering this fraction of cells (projection-bit PBs)",
        long_help = "Drop PB samples in the long tail,\n\
                     until the remaining set holds at least this fraction of all cells.\n\
                     - 1.0: keep every PB (no filtering)\n\
                     - 0.95 (default): remove minor subpopulations\n\
                     Only the projection-bit PBs of a run without a latent have such a\n\
                     tail: PBs coarsened from the cell graph are even, and all kept."
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
        help = "RNG seed for the random starts of tsne, stress and forceatlas2"
    )]
    pub seed: u64,

    #[arg(
        long,
        default_value_t = 1000,
        help = "PB count for the PB layouts (UMAP, PHATE, t-SNE, ...)",
        long_help = "How many PBs the PB layouts (UMAP, PHATE, t-SNE, ...) lay out.\n\
                     The cell kNN graph is coarsened by rounds of heavy-edge matching\n\
                     until this many groups remain: each round merges cells, then groups,\n\
                     with the neighbour they are most strongly linked to, so a PB never\n\
                     spans a gap in the graph. Fewer gives larger, smoother PBs; more\n\
                     gives finer ones. Aim for a few dozen cells per PB. A run laid out\n\
                     from its cell projection ignores this.\n\
                     \n\
                     With --target features: how many PBs the feature graph is coarsened to."
    )]
    pub n_landmarks: usize,

    #[arg(
        long,
        default_value_t = OutlierCut::Auto,
        help = "Drop outlier PBs before the layout: auto, off, or a cut in MADs",
        long_help = "Leave out the cells of outlier PBs, so a few stray cells cannot pull\n\
                     the layout. A PB's outlier score is its isolation: the log distance\n\
                     to its k-th nearest PB (k = --knn), in the features laid out.\n\
                     \n\
                     auto: split the scores in two by k-means; the upper group is dropped\n\
                       when it is a minority and empty space parts it from the rest (a\n\
                       gap of at least two of the rest's MADs). One mode, however skewed,\n\
                       has no such gap, so nothing is dropped and clean data stays whole.\n\
                     off:  keep every PB.\n\
                     N:    drop PBs whose score is more than N MADs above the median.\n\
                     \n\
                     auto needs at least 20 PBs to tell a tail from the rest; with fewer\n\
                     it keeps every PB. PB layouts only: `umap --joint` lays out every cell.\n\
                     Dropped cells get no coordinates and are listed in\n\
                     {out}.layout_outliers.tsv; `senna view` leaves them out."
    )]
    pub drop_outliers: OutlierCut,

    #[arg(
        long,
        help = "File of cell names to leave out of the layout, one per line",
        long_help = "Cells to leave out of the layout, one name per line (a first line\n\
                     `cell`, as a table's header, is skipped): a population to look past,\n\
                     say. They get no coordinates; `senna view` leaves them out."
    )]
    pub drop_cells: Option<Box<str>>,

    #[arg(
        long,
        default_value_t = DEFAULT_THETA_TEMPERATURE,
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
}

/// PHATE diffusion-embedding args of `layout phate`.
#[derive(Args, Debug, Clone)]
pub struct PhateCliArgs {
    #[arg(long, default_value_t = 20, help = "PHATE diffusion time t")]
    pub phate_t: usize,

    #[arg(long, default_value_t = 300, help = "Max SMACOF iterations")]
    pub phate_mds_iter: usize,

    #[arg(
        long,
        default_value_t = 1e-4,
        help = "SMACOF relative-stress tolerance"
    )]
    pub phate_mds_tol: f32,
}

impl From<&PhateCliArgs> for PhateArgs {
    fn from(a: &PhateCliArgs) -> Self {
        Self {
            t: a.phate_t,
            mds_iter: a.phate_mds_iter,
            mds_tol: a.phate_mds_tol,
            ..Self::default()
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

/// What the PB layouts lay out: the run's PBs, the graph between them, and
/// the cells to place by Nyström.
pub(crate) struct PbLayoutPrep {
    pub cells: CellTable,
    /// The PBs (`usize::MAX` for a cell in a pruned one), the cell graph
    /// collapsed onto them, and their mean cell features.
    pub pbs: Coarsened,
}

/// The run's cell table: what every layout lays out.
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
/// which [`latent_layout_features`] would misread as an embedding.
pub(crate) fn cell_table(
    args: &LayoutCommonArgs,
    resolved: &ResolvedViz,
) -> anyhow::Result<CellTable> {
    let m = &resolved.manifest;
    let (names, cells_kn) = match m.outputs.geometry_latent() {
        Some(p) => read_latent_features(args, &resolved.resolve(p), m.kind)?,
        None => read_cell_proj(args, &cell_proj_path(resolved)?)?,
    };
    let table = CellTable { names, cells_kn };
    let Some(path) = args.drop_cells.as_deref() else {
        return Ok(table);
    };
    let drop: std::collections::HashSet<Box<str>> = read_lines(path)
        .map_err(|e| anyhow::anyhow!("--drop-cells {path}: {e}"))?
        .into_iter()
        .enumerate()
        .map(|(i, l)| (i, Box::<str>::from(l.trim())))
        .filter(|(i, l)| !l.is_empty() && (*i != 0 || &**l != "cell"))
        .map(|(_, l)| l)
        .collect();
    let kept = table.keep(|name| !drop.contains(name));
    anyhow::ensure!(
        !kept.names.is_empty(),
        "--drop-cells {path} leaves no cell to lay out"
    );
    info!(
        "--drop-cells: {} of {} cells left out",
        table.names.len() - kept.names.len(),
        table.names.len()
    );
    Ok(kept)
}

impl CellTable {
    /// The cells `keep` says yes to by name, in order.
    fn keep(&self, keep: impl Fn(&str) -> bool) -> Self {
        self.keep_cols(|c| keep(&self.names[c]))
    }

    /// The cells `keep` says yes to by column, in order.
    fn keep_cols(&self, keep: impl Fn(usize) -> bool) -> Self {
        let cols: Vec<usize> = (0..self.names.len()).filter(|&c| keep(c)).collect();
        Self {
            names: cols.iter().map(|&c| self.names[c].clone()).collect(),
            cells_kn: self.cells_kn.select_columns(&cols),
        }
    }
}

/// How `senna layout` drops outlier PBs (`--drop-outliers`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum OutlierCut {
    /// The upper group of a two-means split, when it stands clear.
    Auto,
    Off,
    /// More than this many MADs above the median score.
    Mads(f32),
}

impl std::fmt::Display for OutlierCut {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Auto => f.write_str("auto"),
            Self::Off => f.write_str("off"),
            Self::Mads(n) => write!(f, "{n}"),
        }
    }
}

impl std::str::FromStr for OutlierCut {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, String> {
        match s {
            "auto" => Ok(Self::Auto),
            "off" => Ok(Self::Off),
            n => match n.parse::<f32>() {
                Ok(v) if v > 0.0 && v.is_finite() => Ok(Self::Mads(v)),
                _ => Err(format!("`{n}`: auto, off, or a positive number of MADs")),
            },
        }
    }
}

/// `auto` drops the upper group only when it is at most this share of PBs…
const OUTLIER_MAX_SHARE: f64 = 0.25;
/// …and empty space parts it from the rest: its lowest score this many of
/// the rest's MADs above their highest. Two-means splits any distribution,
/// one long skewed mode too; only a gap tells a tail apart.
const OUTLIER_MIN_GAP_MADS: f64 = 2.0;
/// `auto` keeps every PB of a run with fewer.
const OUTLIER_MIN_PBS: usize = 20;

fn median_mad(sorted: &[f64]) -> (f64, f64) {
    let med = sorted[sorted.len() / 2];
    let mut dev: Vec<f64> = sorted.iter().map(|x| (x - med).abs()).collect();
    dev.sort_by(f64::total_cmp);
    (med, dev[dev.len() / 2])
}

/// Which PBs `cut` calls outliers by their isolation `score` (one per PB).
pub(crate) fn outlier_pbs(score: &[f64], cut: OutlierCut) -> Vec<bool> {
    let n = score.len();
    match cut {
        OutlierCut::Off => vec![false; n],
        OutlierCut::Mads(k) => {
            if n < 3 {
                return vec![false; n];
            }
            let mut sorted = score.to_vec();
            sorted.sort_by(f64::total_cmp);
            let (med, mad) = median_mad(&sorted);
            // Most scores tied: no spread to measure a tail against.
            if mad <= f64::EPSILON * med.abs().max(1.0) {
                return vec![false; n];
            }
            score
                .iter()
                .map(|&s| (s - med) / mad > f64::from(k))
                .collect()
        }
        // Too few to tell a tail from the rest.
        OutlierCut::Auto if n < OUTLIER_MIN_PBS => vec![false; n],
        OutlierCut::Auto => {
            let col =
                nalgebra::DMatrix::<f32>::from_iterator(n, 1, score.iter().map(|&s| s as f32));
            let (_, label) = legume_numeric::matrix::kmeans::kmeans_centroids(&col, 2, 100);
            let group = |k: usize| -> Vec<f64> {
                score
                    .iter()
                    .zip(&label)
                    .filter(|(_, &l)| l == k)
                    .map(|(&x, _)| x)
                    .collect()
            };
            let mean = |g: &[f64]| g.iter().sum::<f64>() / g.len().max(1) as f64;
            let (g0, g1) = (group(0), group(1));
            let upper = usize::from(mean(&g1) > mean(&g0));
            let (mut rest, tail) = if upper == 1 { (g0, g1) } else { (g1, g0) };
            if tail.is_empty()
                || rest.is_empty()
                || tail.len() as f64 > OUTLIER_MAX_SHARE * n as f64
            {
                return vec![false; n];
            }
            rest.sort_by(f64::total_cmp);
            let (_, mad) = median_mad(&rest);
            let top_rest = rest[rest.len() - 1];
            let low_tail = tail.iter().copied().fold(f64::INFINITY, f64::min);
            if low_tail - top_rest < OUTLIER_MIN_GAP_MADS * mad {
                return vec![false; n];
            }
            label.iter().map(|&l| l == upper).collect()
        }
    }
}

/// Every PB's isolation: the log distance to its `k`-th nearest PB.
fn isolation(pbs: &Coarsened, k: usize) -> Vec<f64> {
    let n = pbs.n_pb();
    if n < 2 {
        return vec![0.0; n];
    }
    let (_, dist) = knn_rows_l2(&pbs.pb_features.transpose(), k.max(1));
    dist.iter()
        .map(|d| f64::from(d.last().copied().unwrap_or(0.0).max(1e-12)).ln())
        .collect()
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
    winsorize_cells(&mut proj_kn, args.trim_cell_mads);
    Ok((cell_names, proj_kn))
}

/// The PB layout for UMAP, PHATE, t-SNE and the graph layouts: the run's
/// cell table (its latent, else its cell projection), the cells grouped
/// into PBs on its kNN graph, and the graph between the PBs.
///
/// With a latent the groups come from coarsening the cell graph (rounds of
/// heavy-edge matching down to `--n-landmarks`), so a PB never spans a gap
/// in it; from a cached projection they are its projection bits.
pub(crate) fn preprocess_layout_data(
    args: &LayoutCommonArgs,
    resolved: &ResolvedViz,
) -> anyhow::Result<PbLayoutPrep> {
    let cells = cell_table(args, resolved)?;
    let pbs = cell_pbs(&cells, args, resolved)?;
    let listed = format!(
        "{}.layout_outliers.tsv",
        run_manifest::out_prefix(args.out.as_deref(), &args.from)
    );
    let out = match args.drop_outliers {
        OutlierCut::Off => vec![false; pbs.n_pb()],
        cut => outlier_pbs(&isolation(&pbs, args.knn), cut),
    };
    if !out.iter().any(|&o| o) {
        // No list of an earlier layout's left over to mislead.
        match std::fs::remove_file(&listed) {
            Err(e) if e.kind() != std::io::ErrorKind::NotFound => {
                anyhow::bail!("{listed}: {e}")
            }
            _ => {}
        }
        return Ok(PbLayoutPrep { cells, pbs });
    }
    // Their cells out, and the PBs formed again from the rest (a cell of no
    // PB, `usize::MAX`, stays).
    let is_out: Vec<bool> = (0..cells.names.len())
        .map(|c| {
            pbs.membership
                .get(c)
                .is_some_and(|&g| out.get(g) == Some(&true))
        })
        .collect();
    let kept = cells.keep_cols(|c| !is_out[c]);
    let dropped: Vec<&str> = (0..cells.names.len())
        .filter(|&c| is_out[c])
        .map(|c| &*cells.names[c])
        .collect();
    info!(
        "--drop-outliers {}: {} of {} PBs isolated, {} cells left out",
        args.drop_outliers,
        out.iter().filter(|&&o| o).count(),
        out.len(),
        dropped.len()
    );
    std::fs::write(&listed, format!("cell\n{}\n", dropped.join("\n")))
        .map_err(|e| anyhow::anyhow!("{listed}: {e}"))?;
    let pbs = cell_pbs(&kept, args, resolved)?;
    Ok(PbLayoutPrep { cells: kept, pbs })
}

/// The cells grouped into PBs and the graph between them (see
/// [`preprocess_layout_data`]).
fn cell_pbs(
    cells: &CellTable,
    args: &LayoutCommonArgs,
    resolved: &ResolvedViz,
) -> anyhow::Result<Coarsened> {
    let n_cells = cells.cells_kn.ncols();
    let block_size = args.block_size.unwrap_or(1000);
    let pbs = if resolved.manifest.outputs.geometry_latent().is_some() {
        let n_landmarks = args.n_landmarks.min(n_cells).max(1);
        info!(
            "PBs: the cell graph coarsened to {n_landmarks} (kind={})",
            resolved.manifest.kind
        );
        // Even groups, so no long tail to prune: the smallest are the ones
        // that could not merge, small populations of their own.
        coarsen_columns(&cells.cells_kn, args)?
    } else {
        info!("PBs: projection bits of the cell projection");
        let edges = fuzzy_graph_of_columns(&cells.cells_kn, args.knn, block_size)?;
        let kk = args.sort_dim.min(cells.cells_kn.nrows()).min(n_cells);
        let membership =
            groups_covering(&binary_sort_columns(&cells.cells_kn, kk)?, args.pb_coverage);
        log_coverage(&membership);
        Coarsened::from_membership(&cells.cells_kn, &edges, membership)
    };
    info!("PB graph: {} PBs, {} edges", pbs.n_pb(), pbs.pb_edges.len());
    Ok(pbs)
}

/// The columns of `table_kn` (cells or features) as `--n-landmarks` PBs: their
/// fuzzy kNN graph coarsened by heavy-edge matching, and the graph between
/// the groups.
pub(crate) fn coarsen_columns(
    table_kn: &Mat,
    args: &LayoutCommonArgs,
) -> anyhow::Result<Coarsened> {
    coarsened(
        table_kn,
        &CoarsenArgs {
            knn: args.knn,
            n_landmarks: args.n_landmarks.min(table_kn.ncols()).max(1),
            block_size: args.block_size.unwrap_or(1000),
            seed: args.seed,
        },
    )
}

/// Log how many cells the PBs kept by the coverage filter hold.
fn log_coverage(membership: &[usize]) {
    let covered = membership.iter().filter(|&&g| g != usize::MAX).count();
    let total = membership.len();
    info!(
        "Coverage filter: kept PBs covering {covered} / {total} cells ({:.1}%)",
        100.0 * covered as f32 / total.max(1) as f32
    );
}

/// The per-cell table a layout runs on, from a run's geometry latent
/// (`cells × dims`, as read). Returns `dims × cells` (one column per cell)
/// and a short description of the transform for logging.
///
/// - topic family: `softmax(log θ / τ)` then a Hellinger square root;
/// - embedding kinds (bge, fne, …): each cell's embedding scaled to unit
///   length (cosine distance, matching lupin's trajectory and senna's
///   critique); the caller must have resolved the table through
///   `geometry_latent`, so this is Z, never log θ;
/// - everything else: per-dimension z-scores.
///
/// Each dimension of a topic or z-scored table is then winsorized to
/// ±`trim_mads` MADs.
pub(crate) fn latent_layout_features(
    kind: senna::run_manifest::RunKind,
    latent_nk: &Mat,
    theta_temperature: f32,
    trim_mads: f32,
) -> (Mat, String) {
    let tau = theta_temperature.max(1e-6);
    let mut feat_kn: Mat = latent_nk.transpose();
    // The topic check comes first: masked-vae is a topic kind whose cell
    // space is signed.
    let (latent_desc, clip): (String, bool) = if kind.is_topic_family() {
        // softmax(log_θ / τ) per cell → Hellinger sqrt. τ=1 reduces to
        // sqrt(exp(log_θ)); τ<1 sharpens, τ>1 softens.
        if (tau - 1.0).abs() > 1e-6 {
            feat_kn.apply(|v| *v /= tau);
        }
        feat_kn.normalize_exp_logits_columns_inplace();
        feat_kn.apply(|v| *v = v.sqrt());
        let desc = if (tau - 1.0).abs() < 1e-6 {
            "Hellinger-θ".into()
        } else {
            format!("Hellinger-θ, τ={tau:.3}")
        };
        (desc, true)
    } else if kind.cell_space() == senna::run_manifest::CellSpace::Embedding {
        // Not clipped: clipping a dimension would shorten the unit vectors,
        // and the co-embedded features they share a graph with are not.
        feat_kn.normalize_columns_inplace();
        ("unit-length (cosine)".into(), false)
    } else {
        feat_kn.scale_rows_inplace();
        ("z-scored scores".into(), true)
    };
    if clip {
        winsorize_cells(&mut feat_kn, trim_mads);
    }
    (feat_kn, latent_desc)
}

/// [`winsorize_rows_inplace`] of a column-per-cell table, logged.
fn winsorize_cells(feat: &mut Mat, n_mads: f32) {
    let clipped = winsorize_rows_inplace(feat, n_mads);
    if clipped > 0 {
        info!(
            "Winsorized {clipped} outlier feature value(s) to ±{n_mads} MADs ({} dims × {} cells)",
            feat.nrows(),
            feat.ncols()
        );
    }
}

/// Load cluster assignments (one-column parquet of cell IDs) and validate length.
pub(crate) fn load_cluster_assignments(
    path: &str,
    cells: &[Box<str>],
) -> anyhow::Result<Vec<usize>> {
    info!("Reading cluster assignments from {path}...");
    let MatWithNames {
        rows: names,
        cols: _,
        mat: cluster_mat,
    } = Mat::from_parquet(path)?;
    let at: std::collections::HashMap<&str, usize> = names
        .iter()
        .enumerate()
        .map(|(i, n)| (n.as_ref(), i))
        .collect();
    // By name, so cells left out of the layout drop out of the file too;
    // by position only for a file of exactly these cells under other names.
    let rows: Vec<usize> = match cells
        .iter()
        .map(|c| at.get(c.as_ref()).copied())
        .collect::<Option<Vec<_>>>()
    {
        Some(rows) => rows,
        None if cluster_mat.nrows() == cells.len() => (0..cells.len()).collect(),
        None => anyhow::bail!(
            "Cluster file has {} cells, not named as the layout's {} cells",
            cluster_mat.nrows(),
            cells.len()
        ),
    };
    let clusters: Vec<usize> = rows.iter().map(|&i| cluster_mat[(i, 0)] as usize).collect();
    info!("Loaded {} cluster assignments", clusters.len());
    Ok(clusters)
}

/// Finish a PB layout: place cells by Nyström, write the PB and cell
/// coordinates, and record them in the run's manifest.
pub(crate) fn finalize_viz(
    args: &LayoutCommonArgs,
    resolved: &mut ResolvedViz,
    prep: &PbLayoutPrep,
    pb_coords: &Mat,
    method: &str,
) -> anyhow::Result<()> {
    let cell_coords = prep.pbs.place_cells(
        &prep.cells.cells_kn,
        pb_coords,
        args.place_knn,
        args.kernel_alpha,
    );
    info!("Placed cells by Nyström");
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
    let pb_features = prep.pbs.pb_features.transpose();
    let feat_col_names: Vec<Box<str>> = (0..pb_features.ncols())
        .map(|i| format!("p{i}").into_boxed_str())
        .collect();
    pb_features.to_parquet_with_names(
        &pb_feat_path,
        (Some(&pb_names), Some("pb")),
        Some(&feat_col_names),
    )?;

    let cluster_ids = match &args.clusters {
        Some(path) => Some(load_cluster_assignments(path, &prep.cells.names)?),
        None => None,
    };

    let n_cells = cell_coords.nrows();
    let n_extra = 1 + cluster_ids.as_ref().map_or(0, |_| 1);
    let mut cell_out = Mat::zeros(n_cells, 2 + n_extra);
    for i in 0..n_cells {
        cell_out[(i, 0)] = cell_coords[(i, 0)];
        cell_out[(i, 1)] = cell_coords[(i, 1)];
        let pb_id = prep.pbs.membership[i];
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
        Some(path) => Some(load_cluster_assignments(path, cell_names)?),
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

#[cfg(test)]
#[path = "fit_layout_common_tests.rs"]
mod tests;
