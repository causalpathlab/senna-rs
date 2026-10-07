//! `senna layout umap`: t-UMAP of every cell, as `uwot::tumap` does it. The
//! run's cell table (latent, cell embedding or cell projection; see
//! [`cell_table`]) gives a cell-cell fuzzy kNN graph, and SGD lays it out
//! from a PCA(2) start. No landmarks, no Nyström: those are for PHATE and
//! t-SNE, which cannot take every cell.

use super::fit_layout_common::{
    cell_table, resolve_inputs, write_cell_layout, CellTable, LayoutCommonArgs, ResolvedViz,
};
use super::fit_layout_features::{
    load_feature_layout_input, read_coembedding, write_feature_layout, LayoutTarget,
};
use legume_numeric::matrix::knn_graph::umap_memberships;
use legume_numeric::matrix::pca::{init_2d_from_scores, pc_scores, random_init_2d};
use legume_numeric::matrix::umap::Umap;
use rayon::prelude::*;
use senna::embed_common::*;

#[derive(Args, Debug)]
pub struct LayoutUmapArgs {
    #[clap(flatten)]
    common: LayoutCommonArgs,

    #[arg(
        long,
        help = "Number of SGD epochs (default: 500 up to 10,000 points, else 200, as uwot)"
    )]
    umap_epochs: Option<usize>,

    #[arg(
        long,
        default_value_t = NEGATIVE_RATE,
        help = "Negative samples per attractive step (uwot: 5)"
    )]
    umap_negative_rate: usize,

    #[arg(long, default_value_t = LEARNING_RATE, help = "Initial learning rate")]
    umap_lr: f32,

    #[arg(
        long,
        alias = "umap-finetune-knn",
        default_value_t = N_NEIGHBORS,
        help = "Neighbours per point, counting the point itself (uwot's n_neighbors)"
    )]
    umap_knn: usize,

    #[arg(
        long,
        help = "Lay out cells and features together, as the `joint` method",
        long_help = "Lay out cells and features in one map, recorded as the `joint` method.\n\
                     For runs whose cells and features share one embedding (bge, simba,\n\
                     tde, resolve-embedding-space) and that wrote a feature co-embedding.\n\
                     One t-UMAP runs over cells and features together: a kNN over all of\n\
                     them, plus each feature's nearest cells, so features sit among the\n\
                     cells they belong to and pull on them, rather than being placed on a\n\
                     finished cell map. `senna view` shows it as another layout method."
    )]
    joint: bool,
}

/// PCA(2) initialization for the cell-level UMAP: project cells onto the top-2
/// principal components of the (Euclidean) embedding. Uniform-random init is
/// uwot's *fallback*; its default is spectral/PCA, which seeds the global
/// structure so SGD only has to refine it locally — random init leaves the
/// macro-layout seed-dependent and scrambled. PCA(2) of the already-low-dim
/// embedding is a cheap, faithful stand-in.
///
/// Centering happens here rather than through `pc_scores`'s drop-the-leading-
/// component route: this embedding is a signed projection, not the nonnegative
/// latent that route assumes, so its first component is real structure — and
/// usually the most of it.
///
/// `rows` is `[n_points × dims]`, consumed (centred in place); returns
/// `[n_points × 2]`.
fn pca_init_2d(mut rows: Mat, seed: u64) -> Mat {
    let (n, d) = rows.shape();
    let as_coords = |flat: Vec<f32>| Mat::from_row_iterator(n, 2, flat);
    if n < 3 || d < 2 {
        return as_coords(random_init_2d(n, seed));
    }
    // PCA needs each dimension mean-centred across the points.
    rows.centre_columns_inplace();
    match pc_scores(&rows, 2, 0) {
        Ok(scores) => as_coords(init_2d_from_scores(&scores, seed)),
        Err(_) => as_coords(random_init_2d(n, seed)),
    }
}

pub fn fit_layout_umap(args: &LayoutUmapArgs) -> anyhow::Result<()> {
    if args.common.target == LayoutTarget::Features {
        return fit_feature_layout_umap(args);
    }
    let mut resolved = resolve_inputs(&args.common)?;
    if args.joint {
        // Before any layout work: only an embedding run lays out directly.
        let kind = resolved.manifest.kind;
        anyhow::ensure!(
            kind.cell_space() == senna::run_manifest::CellSpace::Embedding,
            "--joint needs a run whose cells and features share one embedding \
             (bge, simba, tde, resolve-embedding-space), not a `{kind}` run"
        );
    }
    let prep = cell_table(&args.common, &resolved)?;
    if args.joint {
        fit_layout_joint(args, &mut resolved, &prep)
    } else {
        fit_cell_layout_umap(args, &mut resolved, &prep)
    }
}

/// `--joint`: cells and the feature co-embedding in one t-UMAP, written as
/// the `joint` method (cells, and features on that same map).
fn fit_layout_joint(
    args: &LayoutUmapArgs,
    resolved: &mut ResolvedViz,
    prep: &CellTable,
) -> anyhow::Result<()> {
    let (names, coembed_dh) = read_coembedding(
        &resolved.manifest,
        &resolved.manifest_path,
        prep.cells_kn.nrows(),
    )?
    .ok_or_else(|| {
        anyhow::anyhow!(
            "--joint needs the run's feature co-embedding (outputs.feature_coembedding), \
             in the cells' space; this run has none"
        )
    })?;
    info!(
        "UMAP joint mode: {} cells + {} features",
        prep.cells_kn.ncols(),
        names.len()
    );
    let (cell_coords, feature_coords) = joint_tumap(
        &prep.cells_kn,
        &coembed_dh.transpose(),
        &args.tumap_params(),
    )?;
    write_cell_layout(
        &args.common,
        resolved,
        prep,
        &cell_coords,
        "joint",
        Some((&names, &feature_coords)),
    )
}

/// Cell-level t-UMAP of the run's cell table. Output: only
/// `cell_coords.parquet`, no `pb_coords`.
fn fit_cell_layout_umap(
    args: &LayoutUmapArgs,
    resolved: &mut ResolvedViz,
    prep: &CellTable,
) -> anyhow::Result<()> {
    let n = prep.cells_kn.ncols();
    info!("UMAP: {n} cells");

    // PCA(2) init (uwot's `init = "pca"`; its default is spectral) so the
    // global structure is seeded from the embedding rather than noise.
    let cell_coords = tumap_on_columns(&prep.cells_kn, &args.tumap_params())?;

    write_cell_layout(&args.common, resolved, prep, &cell_coords, "umap", None)
}

/// `--target features`: t-UMAP of the feature embedding on its own, over a
/// cosine fuzzy kNN graph with a PCA(2) init. Same SGD as the direct cell path.
fn fit_feature_layout_umap(args: &LayoutUmapArgs) -> anyhow::Result<()> {
    let mut input = load_feature_layout_input(&args.common)?;
    let coords = tumap_on_columns(&input.feat_kn, &args.tumap_params())?;
    write_feature_layout(&mut input, "umap", &coords)
}

/// Settings for a t-UMAP run straight on per-point features.
pub(crate) struct TumapParams {
    /// Neighbours per point, counting the point itself (uwot's
    /// `n_neighbors`).
    pub knn: usize,
    /// `None`: uwot's rule, 500 epochs up to 10,000 points, else 200.
    pub epochs: Option<usize>,
    pub negative_rate: usize,
    pub lr: f32,
    pub block_size: usize,
    pub seed: u64,
}

/// uwot's defaults, shared by the CLI and [`TumapParams::default`].
const N_NEIGHBORS: usize = 15;
const NEGATIVE_RATE: usize = 5;
const LEARNING_RATE: f32 = 1.0;

impl Default for TumapParams {
    fn default() -> Self {
        Self {
            knn: N_NEIGHBORS,
            epochs: None,
            negative_rate: NEGATIVE_RATE,
            lr: LEARNING_RATE,
            block_size: 1000,
            seed: 42,
        }
    }
}

impl LayoutUmapArgs {
    fn tumap_params(&self) -> TumapParams {
        TumapParams {
            knn: self.umap_knn,
            epochs: self.umap_epochs,
            negative_rate: self.umap_negative_rate,
            lr: self.umap_lr,
            block_size: self.common.block_size.unwrap_or(1000),
            seed: self.common.seed,
        }
    }
}

/// t-UMAP of the points given as columns of `feat_kn` (`dims × n`): a fuzzy
/// kNN graph on the features, a PCA(2) start, then SGD. Returns `n × 2`.
pub(crate) fn tumap_on_columns(feat_kn: &Mat, p: &TumapParams) -> anyhow::Result<Mat> {
    let rows = feat_kn.transpose();
    let edges = build_cell_cell_fuzzy_edges(&rows, p.knn, p.block_size)?;
    let mut coords = pca_init_2d(rows, p.seed);
    run_cell_level_umap_in_place(&mut coords, &edges, p.epochs, p.negative_rate, p.lr, p.seed);
    Ok(coords)
}

/// t-UMAP of cells and features that share one space: `cells_kn`
/// (`dims × cells`, the cell embedding) and `features_kn` (`dims × features`,
/// e.g. the co-embedding) become the points of one graph. It is a fuzzy kNN
/// over all of them, plus each feature's `knn` nearest cells, so no feature
/// is left tied only to other features. One SGD from a joint PCA(2) start.
/// Returns (`cells × 2`, `features × 2`).
pub(crate) fn joint_tumap(
    cells_kn: &Mat,
    features_kn: &Mat,
    p: &TumapParams,
) -> anyhow::Result<(Mat, Mat)> {
    let (h, n, d) = (cells_kn.nrows(), cells_kn.ncols(), features_kn.ncols());
    anyhow::ensure!(
        features_kn.nrows() == h,
        "features have {} dims but cells {h}: not one shared space",
        features_kn.nrows()
    );
    // Every point as a row: cells, then features.
    let mut all = Mat::zeros(n + d, h);
    all.rows_mut(0, n).tr_copy_from(cells_kn);
    all.rows_mut(n, d).tr_copy_from(features_kn);

    let mut edges = build_cell_cell_fuzzy_edges(&all, p.knn, p.block_size)?;
    edges.extend(feature_to_cell_edges(cells_kn, features_kn, p.knn)?);
    let edges = merge_fuzzy_edges(edges);
    info!(
        "Joint graph: {n} cells + {d} features, {} edges",
        edges.len()
    );

    let mut coords = pca_init_2d(all, p.seed);
    run_cell_level_umap_in_place(&mut coords, &edges, p.epochs, p.negative_rate, p.lr, p.seed);
    Ok((
        coords.rows(0, n).into_owned(),
        coords.rows(n, d).into_owned(),
    ))
}

/// Each feature to its nearest cells in the shared space (`knn` counting
/// the feature itself, as uwot counts `n_neighbors`), weighted by UMAP's
/// kernel for that feature: `exp(-(dist - ρ) / σ)`, where ρ is the nearest
/// distance and σ makes the weights sum to `log2(knn)`. Edges are
/// `(cell, n + feature, w)`.
fn feature_to_cell_edges(
    cells_kn: &Mat,
    features_kn: &Mat,
    knn: usize,
) -> anyhow::Result<Vec<(usize, usize, f32)>> {
    use legume_numeric::matrix::knn::{ColumnDict, SearchScratch};
    let n = cells_kn.ncols();
    // Views onto the cells, not a copy of the table.
    let index = ColumnDict::from_dvector_views(cells_kn.column_iter().collect(), (0..n).collect());
    let per_feature: Vec<Vec<(usize, usize, f32)>> = (0..features_kn.ncols())
        .into_par_iter()
        .map_init(SearchScratch::default, |scratch, f| {
            let query: Vec<f32> = features_kn.column(f).iter().copied().collect();
            let (cells, dists) = index.search_by_query_data_reuse(&query, others(knn), scratch)?;
            Ok(cells
                .into_iter()
                .zip(umap_memberships(&dists))
                .map(|(c, w)| (c, n + f, w))
                .collect())
        })
        .collect::<anyhow::Result<_>>()?;
    Ok(per_feature.into_iter().flatten().collect())
}

/// One edge per pair (`i < j`), weights of repeated pairs combined as a
/// fuzzy union, `a + b - a·b`.
fn merge_fuzzy_edges(edges: Vec<(usize, usize, f32)>) -> Vec<(usize, usize, f32)> {
    let mut edges: Vec<(usize, usize, f32)> = edges
        .into_iter()
        .map(|(i, j, w)| (i.min(j), i.max(j), w))
        .collect();
    edges.par_sort_unstable_by_key(|&(i, j, _)| (i, j));
    let mut out: Vec<(usize, usize, f32)> = Vec::with_capacity(edges.len());
    for (i, j, w) in edges {
        match out.last_mut() {
            Some(last) if (last.0, last.1) == (i, j) => last.2 = last.2 + w - last.2 * w,
            _ => out.push((i, j, w)),
        }
    }
    out
}

/// The neighbours a point has besides itself, when `knn` counts the point
/// as uwot's `n_neighbors` does.
fn others(knn: usize) -> usize {
    knn.saturating_sub(1).max(1)
}

/// The undirected fuzzy kNN edge list of the rows of `rows` (points ×
/// dims), as uwot builds it (`knn` counting each point itself): one edge
/// per pair (`i < j`) with its UMAP membership.
fn build_cell_cell_fuzzy_edges(
    rows: &Mat,
    knn: usize,
    block_size: usize,
) -> anyhow::Result<Vec<(usize, usize, f32)>> {
    let n = rows.nrows();
    info!("Cell-cell fuzzy kNN (n={n}, n_neighbors={knn}) ...");
    let (graph, fuzzy) = legume_numeric::matrix::knn_graph::KnnGraph::from_rows_fuzzy(
        rows,
        legume_numeric::matrix::knn_graph::KnnGraphArgs {
            knn: others(knn),
            block_size,
            reciprocal: false,
        },
    )?;

    let edges: Vec<(usize, usize, f32)> = graph
        .edges
        .par_iter()
        .zip(fuzzy.par_iter())
        .filter_map(|(&(i, j), &w)| (w > 0.0).then_some((i, j, w)))
        .collect();
    info!(
        "Cell-cell fuzzy kNN: {} cells, {} edges (mean deg = {:.1})",
        n,
        edges.len(),
        2.0 * edges.len() as f32 / n.max(1) as f32
    );
    Ok(edges)
}

/// Run cell-level t-UMAP SGD from `coords`, mutating them in place.
/// `epochs` unset: uwot's rule, 500 up to 10,000 points, else 200.
fn run_cell_level_umap_in_place(
    coords: &mut Mat,
    edges: &[(usize, usize, f32)],
    epochs: Option<usize>,
    negative_rate: usize,
    learning_rate: f32,
    seed: u64,
) {
    let n = coords.nrows();
    let epochs = epochs.unwrap_or(if n <= 10_000 { 500 } else { 200 });
    let init_flat: Vec<f32> = (0..n)
        .flat_map(|i| [coords[(i, 0)], coords[(i, 1)]])
        .collect();

    let umap = Umap {
        n_epochs: epochs,
        negative_sample_rate: negative_rate,
        learning_rate,
        seed,
        ..Umap::tumap()
    };
    info!("Cell-level UMAP SGD (epochs={epochs}) ...");
    let refined = umap.fit(edges, n, &init_flat);

    for i in 0..n {
        coords[(i, 0)] = refined[i * 2];
        coords[(i, 1)] = refined[i * 2 + 1];
    }
    info!("Cell-level UMAP done");
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::SmallRng;
    use rand::{RngExt, SeedableRng};

    /// Two groups of cells in a shared 6-D space, and genes co-embedded at
    /// each group's centre. Genes of a group are nearly identical, so a kNN
    /// over the union alone would tie them mostly to each other.
    fn two_groups(per: usize, genes: usize) -> (Mat, Mat) {
        let h = 6;
        let mut rng = SmallRng::seed_from_u64(7);
        let centre = |g: usize, d: usize| if d % 2 == g { 4.0 } else { 0.0 };
        let mut cells = Mat::zeros(h, 2 * per);
        for c in 0..2 * per {
            for d in 0..h {
                cells[(d, c)] = centre(c / per, d) + rng.random_range(-1.0..1.0);
            }
        }
        let mut gene_kn = Mat::zeros(h, genes);
        for f in 0..genes {
            for d in 0..h {
                gene_kn[(d, f)] = centre(f % 2, d) + rng.random_range(-0.05..0.05);
            }
        }
        (cells, gene_kn)
    }

    fn params() -> TumapParams {
        TumapParams {
            knn: 15,
            epochs: Some(200),
            negative_rate: 5,
            lr: 1.0,
            block_size: 128,
            seed: 1,
        }
    }

    #[test]
    fn a_joint_layout_puts_each_gene_among_its_own_cells() {
        let per = 200;
        let (cells, genes) = two_groups(per, 40);
        let (cell_xy, gene_xy) = joint_tumap(&cells, &genes, &params()).unwrap();
        assert_eq!((cell_xy.nrows(), cell_xy.ncols()), (2 * per, 2));
        assert_eq!((gene_xy.nrows(), gene_xy.ncols()), (40, 2));

        let centroid = |g: usize| {
            let rows = (g * per..(g + 1) * per).map(|c| [cell_xy[(c, 0)], cell_xy[(c, 1)]]);
            let (sx, sy) = rows.fold((0.0, 0.0), |(x, y), [a, b]| (x + a, y + b));
            [sx / per as f32, sy / per as f32]
        };
        let dist =
            |p: [f32; 2], q: [f32; 2]| ((p[0] - q[0]).powi(2) + (p[1] - q[1]).powi(2)).sqrt();
        for f in 0..40 {
            let at = [gene_xy[(f, 0)], gene_xy[(f, 1)]];
            let (own, other) = (centroid(f % 2), centroid(1 - f % 2));
            assert!(
                dist(at, own) < dist(at, other),
                "gene {f} sits nearer the other group's cells"
            );
        }
    }

    #[test]
    fn genes_sit_inside_their_cells_not_on_an_island_beside_them() {
        // A small population with more genes than cells, as a rare cell type
        // with many markers: a kNN over the union alone ties each gene to
        // other genes, and they drift off as a clump beside their cells.
        let (per, n_genes) = (200, 800);
        let (cells, genes) = two_groups(per, n_genes);
        let (cell_xy, gene_xy) = joint_tumap(&cells, &genes, &params()).unwrap();
        let xy = |m: &Mat, i: usize| [m[(i, 0)], m[(i, 1)]];
        let dist =
            |p: [f32; 2], q: [f32; 2]| ((p[0] - q[0]).powi(2) + (p[1] - q[1]).powi(2)).sqrt();
        // A group's radius: the median distance of its cells to their centroid.
        let radius = |g: usize| {
            let rows: Vec<usize> = (g * per..(g + 1) * per).collect();
            let c = rows.iter().fold([0.0, 0.0], |[x, y], &r| {
                [
                    x + cell_xy[(r, 0)] / per as f32,
                    y + cell_xy[(r, 1)] / per as f32,
                ]
            });
            let mut d: Vec<f32> = rows.iter().map(|&r| dist(xy(&cell_xy, r), c)).collect();
            d.sort_by(f32::total_cmp);
            d[per / 2]
        };
        for f in 0..n_genes {
            let at = xy(&gene_xy, f);
            let (nearest, d) = (0..2 * per)
                .map(|c| (c, dist(at, xy(&cell_xy, c))))
                .min_by(|a, b| a.1.total_cmp(&b.1))
                .unwrap();
            assert_eq!(
                nearest / per,
                f % 2,
                "gene {f}'s nearest cell is of the other group"
            );
            // Genes take room on the map too, so allow a margin past the
            // cells' median radius; a clump beside the cells lies well beyond.
            assert!(
                d < 1.5 * radius(f % 2),
                "gene {f} is {d:.2} from the nearest cell; its group's radius is {:.2}",
                radius(f % 2)
            );
        }
    }
}
