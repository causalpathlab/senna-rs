//! `senna layout umap` — UMAP-style SGD on the PB-PB fuzzy kNN graph.
//! Same prep/finalize pipeline as `layout tsne`; differs only in the
//! 2D layout algorithm. Optionally runs a cell-level SGD pass after
//! Nyström to resolve intra-PB structure (`--umap-finetune-epochs > 0`).
//!
//! For `RunKind::Bge` / `RunKind::Fne` manifests, `preprocess_layout_data`
//! returns `LayoutMode::DirectCells` and we skip the PB pass entirely —
//! build a cell-cell fuzzy kNN graph straight from the L2-normalized
//! latent and run UMAP SGD on cells directly. No Nyström, no PB output.

use super::fit_layout_common::{
    nystrom_cell_coords, preprocess_layout_data, resolve_inputs, write_viz_outputs_direct,
    write_viz_outputs_pb, DirectLayoutPrep, LayoutCommonArgs, LayoutPrep, PbLayoutPrep,
    ResolvedViz,
};
use super::fit_layout_features::{
    load_feature_layout_input, read_coembedding, write_feature_layout, LayoutTarget,
};
use legume_numeric::matrix::pca::{init_2d_from_scores, pc_scores, random_init_2d};
use legume_numeric::matrix::umap::Umap;
use rayon::prelude::*;
use senna::embed_common::*;

#[derive(Args, Debug)]
pub struct LayoutUmapArgs {
    #[clap(flatten)]
    common: LayoutCommonArgs,

    #[arg(long, default_value_t = 500, help = "Number of SGD epochs")]
    umap_epochs: usize,

    #[arg(
        long,
        default_value_t = 20,
        help = "Negative samples per attractive step"
    )]
    umap_negative_rate: usize,

    #[arg(long, default_value_t = 1.0, help = "Initial learning rate")]
    umap_lr: f32,

    #[arg(
        long,
        default_value_t = 100,
        help = "Cell-level UMAP fine-tune epochs after Nyström (0 = disabled)",
        long_help = "Run an extra UMAP SGD after the PB-level pass.\n\
                     The PB-level UMAP and Nyström placement come first.\n\
                     The extra pass uses a cell-cell fuzzy kNN graph, in latent space,\n\
                     warm-started from Nyström.\n\
                     \n\
                     It resolves intra-PB structure that Nyström collapses.\n\
                     Cost scales about linearly in cell count.\n\
                     50 to 100 epochs usually suffices, since the init is close."
    )]
    umap_finetune_epochs: usize,

    #[arg(
        long,
        default_value_t = 15,
        help = "kNN for cell-cell graph during fine-tune"
    )]
    umap_finetune_knn: usize,

    #[arg(
        long,
        help = "Lay out cells and features together, as the `joint` method",
        long_help = "Lay out cells and features in one map, recorded as the `joint` method.\n\
                     For runs whose cells and features share one embedding (bge, simba,\n\
                     gem, resolve-embedding-space) and that wrote a feature co-embedding.\n\
                     One t-UMAP runs over cells and features together: a kNN over all of\n\
                     them, plus each feature's nearest cells, so features sit among the\n\
                     cells they belong to and pull on them, rather than being placed on a\n\
                     finished cell map. `senna view` shows it as another layout method."
    )]
    joint: bool,
}

impl Default for LayoutUmapArgs {
    fn default() -> Self {
        Self {
            common: LayoutCommonArgs::default(),
            umap_epochs: 500,
            umap_negative_rate: 20,
            umap_lr: 1.0,
            umap_finetune_epochs: 100,
            umap_finetune_knn: 15,
            joint: false,
        }
    }
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
/// `feat_kn` is `[dims × n_cells]` (columns = cells); returns `[n_cells × 2]`.
fn pca_init_2d(feat_kn: &Mat, seed: u64) -> Mat {
    let d = feat_kn.nrows();
    let n = feat_kn.ncols();
    let as_coords = |flat: Vec<f32>| Mat::from_row_iterator(n, 2, flat);
    if n < 3 || d < 2 {
        return as_coords(random_init_2d(n, seed));
    }
    // Center each dimension across cells (PCA needs mean-centered data):
    // `column_mean` is the per-dimension mean over the n columns; subtract it
    // from every column (contiguous column-major access, no strided row sums).
    let mut centered = feat_kn.clone();
    let mean = feat_kn.column_mean();
    centered.column_iter_mut().for_each(|mut col| col -= &mean);
    // `pc_scores` takes rows as points, so hand it the [n_cells × dims] view.
    match pc_scores(&centered.transpose(), 2, 0) {
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
        let kind = resolved.manifest.as_ref().map(|m| m.kind);
        anyhow::ensure!(
            kind.is_some_and(|k| k.cell_space() == senna::run_manifest::CellSpace::Embedding),
            "--joint needs a run whose cells and features share one embedding \
             (bge, simba, gem, resolve-embedding-space), not {}",
            kind.map_or("one given without --from".into(), |k| format!(
                "a `{k}` run"
            ))
        );
    }
    let prep = preprocess_layout_data(&args.common, &resolved, /*allow_direct_cells=*/ true)?;

    match &prep {
        LayoutPrep::DirectCells(p) if args.joint => fit_layout_joint(args, &mut resolved, p),
        LayoutPrep::PbThenNystrom(p) => fit_layout_umap_pb(args, &mut resolved, p),
        LayoutPrep::DirectCells(p) => fit_layout_umap_direct(args, &mut resolved, p),
    }
}

/// `--joint`: cells and the feature co-embedding in one t-UMAP, written as
/// the `joint` method (cells, and features on that same map).
fn fit_layout_joint(
    args: &LayoutUmapArgs,
    resolved: &mut ResolvedViz,
    prep: &DirectLayoutPrep,
) -> anyhow::Result<()> {
    let (names, coembed_dh) = read_coembedding(
        resolved.manifest.as_ref(),
        resolved.manifest_path.as_ref(),
        prep.cell_proj_kn.nrows(),
    )?
    .ok_or_else(|| {
        anyhow::anyhow!(
            "--joint needs the run's feature co-embedding (outputs.feature_coembedding), \
             in the cells' space; this run has none"
        )
    })?;
    info!(
        "UMAP joint mode: {} cells + {} features",
        prep.cell_proj_kn.ncols(),
        names.len()
    );
    let (cell_coords, feature_coords) = joint_tumap(
        &prep.cell_proj_kn,
        &coembed_dh.transpose(),
        &args.direct_params(),
    )?;
    write_viz_outputs_direct(
        &args.common,
        resolved,
        prep,
        &cell_coords,
        "joint",
        Some((&names, &feature_coords)),
    )
}

fn fit_layout_umap_pb(
    args: &LayoutUmapArgs,
    resolved: &mut ResolvedViz,
    prep: &PbLayoutPrep,
) -> anyhow::Result<()> {
    let n = prep.pb_similarity.nrows();
    let edges = extract_upper_edges(&prep.pb_similarity);
    info!(
        "UMAP: {} PBs, {} edges (mean deg = {:.1})",
        n,
        edges.len(),
        2.0 * edges.len() as f32 / n.max(1) as f32
    );

    // PB layout keeps the random init: the PB-PB similarity arrives as a dense
    // matrix, not the feature vectors a PC init would need.
    let init_flat = random_init_2d(n, args.common.seed);

    let umap = Umap {
        n_epochs: args.umap_epochs,
        negative_sample_rate: args.umap_negative_rate,
        learning_rate: args.umap_lr,
        seed: args.common.seed,
        ..Umap::tumap()
    };
    info!(
        "Running UMAP SGD (epochs={}, neg={}) ...",
        args.umap_epochs, args.umap_negative_rate
    );
    let result = umap.fit(&edges, n, &init_flat);

    let mut pb_coords = Mat::zeros(n, 2);
    for i in 0..n {
        pb_coords[(i, 0)] = result[i * 2];
        pb_coords[(i, 1)] = result[i * 2 + 1];
    }
    info!("UMAP done");

    let mut cell_coords = nystrom_cell_coords(&args.common, prep, &pb_coords);
    if args.umap_finetune_epochs > 0 {
        let edges = build_cell_cell_fuzzy_edges(
            &prep.cell_proj_kn,
            args.umap_finetune_knn,
            args.common.block_size.unwrap_or(1000),
        )?;
        run_cell_level_umap_in_place(
            &mut cell_coords,
            &edges,
            args.umap_finetune_epochs,
            args.umap_negative_rate,
            args.umap_lr,
            args.common.seed.wrapping_add(1),
        );
    }

    write_viz_outputs_pb(
        &args.common,
        resolved,
        prep,
        &pb_coords,
        &cell_coords,
        "umap",
    )
}

/// Direct cell-level UMAP for `RunKind::Bge` / `RunKind::Fne` (and any
/// future kind that returns `LayoutPrep::DirectCells`). Skips landmark
/// sampling, fuzzy kNN on PB centroids, and Nyström — the graph-trained
/// latent is already manifold-aware, so we put cells straight on the
/// cell-cell fuzzy kNN graph and run UMAP SGD there. Output: only
/// `cell_coords.parquet`, no `pb_coords`.
fn fit_layout_umap_direct(
    args: &LayoutUmapArgs,
    resolved: &mut ResolvedViz,
    prep: &DirectLayoutPrep,
) -> anyhow::Result<()> {
    let n = prep.cell_proj_kn.ncols();
    info!("UMAP direct cell-level mode: {n} cells");

    // PCA(2) init (uwot's default is spectral/PCA, not random) so the global
    // structure is seeded from the embedding rather than noise.
    let cell_coords = tumap_on_columns(&prep.cell_proj_kn, &args.direct_params())?;

    write_viz_outputs_direct(&args.common, resolved, prep, &cell_coords, "umap", None)
}

/// `--target features`: t-UMAP of the feature embedding on its own, over a
/// cosine fuzzy kNN graph with a PCA(2) init. Same SGD as the direct cell path.
fn fit_feature_layout_umap(args: &LayoutUmapArgs) -> anyhow::Result<()> {
    let mut input = load_feature_layout_input(&args.common)?;
    let coords = tumap_on_columns(&input.feat_kn, &args.direct_params())?;
    write_feature_layout(&mut input, "umap", &coords)
}

/// Settings for a t-UMAP run straight on per-point features.
pub(crate) struct DirectUmap {
    pub knn: usize,
    pub epochs: usize,
    pub negative_rate: usize,
    pub lr: f32,
    pub block_size: usize,
    pub seed: u64,
}

impl Default for DirectUmap {
    fn default() -> Self {
        let a = LayoutUmapArgs::default();
        a.direct_params()
    }
}

impl LayoutUmapArgs {
    fn direct_params(&self) -> DirectUmap {
        DirectUmap {
            knn: self.umap_finetune_knn,
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
pub(crate) fn tumap_on_columns(feat_kn: &Mat, p: &DirectUmap) -> anyhow::Result<Mat> {
    let edges = build_cell_cell_fuzzy_edges(feat_kn, p.knn, p.block_size)?;
    let mut coords = pca_init_2d(feat_kn, p.seed);
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
    p: &DirectUmap,
) -> anyhow::Result<(Mat, Mat)> {
    let (h, n, d) = (cells_kn.nrows(), cells_kn.ncols(), features_kn.ncols());
    anyhow::ensure!(
        features_kn.nrows() == h,
        "features have {} dims but cells {h}: not one shared space",
        features_kn.nrows()
    );
    let mut all = Mat::zeros(h, n + d);
    all.columns_mut(0, n).copy_from(cells_kn);
    all.columns_mut(n, d).copy_from(features_kn);

    let mut edges = build_cell_cell_fuzzy_edges(&all, p.knn, p.block_size)?;
    edges.extend(feature_to_cell_edges(cells_kn, features_kn, p.knn)?);
    let edges = merge_fuzzy_edges(edges);
    info!(
        "Joint graph: {n} cells + {d} features, {} edges",
        edges.len()
    );

    let mut coords = pca_init_2d(&all, p.seed);
    run_cell_level_umap_in_place(&mut coords, &edges, p.epochs, p.negative_rate, p.lr, p.seed);
    Ok((
        coords.rows(0, n).into_owned(),
        coords.rows(n, d).into_owned(),
    ))
}

/// Each feature to its `knn` nearest cells in the shared space, weighted by
/// UMAP's kernel for that feature: `exp(-(dist - ρ) / σ)`, where ρ is the
/// nearest distance and σ makes the weights sum to `log2(knn)`. Edges are
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
            let (cells, dists) = index.search_by_query_data_reuse(&query, knn, scratch)?;
            Ok(cells
                .into_iter()
                .zip(smooth_knn_weights(&dists))
                .map(|(c, w)| (c, n + f, w))
                .collect())
        })
        .collect::<anyhow::Result<_>>()?;
    Ok(per_feature.into_iter().flatten().collect())
}

/// UMAP's membership weights for one point's neighbour distances (nearest
/// first): `exp(-(d - ρ) / σ)` with ρ the nearest distance and σ found by
/// bisection so the weights sum to `log2(k)`.
fn smooth_knn_weights(dists: &[f32]) -> Vec<f32> {
    let Some(&rho) = dists.first() else {
        return Vec::new();
    };
    let target = (dists.len() as f32).log2().max(1.0);
    let weights = |sigma: f32| {
        dists
            .iter()
            .map(move |&d| (-(d - rho).max(0.0) / sigma).exp())
    };
    let (mut lo, mut hi) = (1e-6_f32, 1e6_f32);
    for _ in 0..64 {
        let mid = (lo * hi).sqrt();
        if weights(mid).sum::<f32>() > target {
            hi = mid;
        } else {
            lo = mid;
        }
    }
    weights((lo * hi).sqrt()).collect()
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

/// Build the undirected cell-cell fuzzy kNN edge list used by both the
/// `PbThenNystrom` fine-tune pass and the `DirectCells` cell-level UMAP.
/// `KnnGraph::edges` is already canonical (`i < j`, sorted, deduped)
/// and `fuzzy_kernel_weights` returns one weight per edge.
fn build_cell_cell_fuzzy_edges(
    feat_kn: &Mat,
    knn: usize,
    block_size: usize,
) -> anyhow::Result<Vec<(usize, usize, f32)>> {
    let n = feat_kn.ncols();
    info!("Cell-cell fuzzy kNN (n={n}, knn={knn}) ...");
    let graph = legume_numeric::matrix::knn_graph::KnnGraph::from_columns(
        feat_kn,
        legume_numeric::matrix::knn_graph::KnnGraphArgs {
            knn,
            block_size,
            reciprocal: false,
        },
    )?;
    let fuzzy = graph.fuzzy_kernel_weights();

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

/// Run cell-level UMAP SGD, mutating `coords` in place. Common back-end
/// for both the PB-mode fine-tune pass and (via the shared edge
/// builder) the `DirectCells` path.
fn run_cell_level_umap_in_place(
    coords: &mut Mat,
    edges: &[(usize, usize, f32)],
    epochs: usize,
    negative_rate: usize,
    learning_rate: f32,
    seed: u64,
) {
    let n = coords.nrows();
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

/// Collect upper-triangle non-zero entries as `(i, j, w)` with `i < j`.
/// Self-loop reg added by `regularize_similarity` is ignored (i == j).
fn extract_upper_edges(sim: &Mat) -> Vec<(usize, usize, f32)> {
    let n = sim.nrows();
    let mut edges = Vec::with_capacity(n * 15);
    for i in 0..n {
        for j in (i + 1)..n {
            let w = sim[(i, j)];
            if w > 0.0 {
                edges.push((i, j, w));
            }
        }
    }
    edges
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

    fn params() -> DirectUmap {
        DirectUmap {
            knn: 15,
            epochs: 200,
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
