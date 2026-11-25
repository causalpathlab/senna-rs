//! `senna layout umap`: t-UMAP of the run's PBs, as t-SNE and PHATE lay
//! them out, or UMAP's kernel when `--umap-min-dist` or `--umap-spread` is
//! given. The graph between the PBs (the cell kNN graph, coarsened and
//! collapsed) is laid out by SGD from a random start, and every cell is
//! then placed by Nyström from its nearest PBs.
//!
//! `--target features` takes the same route over the feature embedding
//! (`feature_pb_layout`). `--joint` still lays out every point: the cells'
//! and features' fuzzy kNN graph by SGD from a PCA(2) start.

use super::fit_layout_common::{
    cell_table, finalize_viz, preprocess_layout_data, resolve_inputs, write_cell_layout, CellTable,
    LayoutCommonArgs, OutlierCut, ResolvedViz,
};
use super::fit_layout_features::{feature_pb_layout, read_coembedding, LayoutTarget};
use legume_numeric::matrix::layout::{
    joint_umap_layout_2d_of_columns, umap_layout_2d_on_graph, UmapLayoutArgs,
};
use legume_numeric::matrix::umap::{Umap, MIN_DIST, SPREAD};
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
        default_value_t = Umap::default().negative_sample_rate,
        help = "Negative samples per attractive step (uwot: 5)"
    )]
    umap_negative_rate: usize,

    #[arg(long, default_value_t = Umap::default().learning_rate, help = "Initial learning rate")]
    umap_lr: f32,

    #[arg(
        long,
        value_name = "DIST",
        help = "UMAP's min_dist: how tightly points pack [default: t-UMAP's kernel]",
        long_help = "How close points may sit on the map, as umap-learn's min_dist.\n\
                     Larger values loosen groups into continua; smaller ones pack them\n\
                     into tight clumps. Must lie in [0, spread].\n\
                     \n\
                     With neither this nor --umap-spread the layout is t-UMAP\n\
                     (uwot::tumap, the kernel 1/(1 + d^2)). With either, it is UMAP's\n\
                     kernel 1/(1 + a d^(2b)), its (a, b) fitted to min_dist and spread\n\
                     as umap-learn fits them; the other defaults to umap-learn's\n\
                     (min_dist 0.1, spread 1)."
    )]
    umap_min_dist: Option<f32>,

    #[arg(
        long,
        value_name = "SCALE",
        help = "UMAP's spread: the scale of the map's distances (see --umap-min-dist)"
    )]
    umap_spread: Option<f32>,

    #[arg(
        long,
        help = "Lay out cells and features together, as the `joint` method",
        long_help = "Lay out cells and features in one map, recorded as the `joint` method.\n\
                     For runs whose cells and features share one embedding (bge, simba,\n\
                     resolve-embedding-space) and that wrote a feature co-embedding.\n\
                     One t-UMAP runs over cells and features together: a kNN over all of\n\
                     them, plus each feature's nearest cells, so features sit among the\n\
                     cells they belong to and pull on them, rather than being placed on a\n\
                     finished cell map. `senna view` shows it as another layout method."
    )]
    joint: bool,
}

pub fn fit_layout_umap(args: &LayoutUmapArgs) -> anyhow::Result<()> {
    // First, so a bad kernel is refused before the run is read.
    let params = args.umap_layout_args()?;
    if args.common.target == LayoutTarget::Features {
        anyhow::ensure!(
            !args.joint,
            "--joint lays out cells and features together; drop --target features for it"
        );
        return feature_pb_layout(&args.common, "umap", |pbs| {
            let n = pbs.n_pb();
            Ok(umap_layout_2d_on_graph(n, &pbs.pb_edges, &params.umap(n)))
        });
    }
    let mut resolved = resolve_inputs(&args.common)?;
    if args.joint {
        // Before any layout work: only an embedding run lays out directly.
        let kind = resolved.manifest.kind;
        anyhow::ensure!(
            kind.cell_space() == senna::run_manifest::CellSpace::Embedding,
            "--joint needs a run whose cells and features share one embedding \
             (bge, simba, resolve-embedding-space), not a `{kind}` run"
        );
        anyhow::ensure!(
            !matches!(args.common.drop_outliers, OutlierCut::Mads(_)),
            "--drop-outliers is for PB layouts; --joint lays out every cell"
        );
        let prep = cell_table(&args.common, &resolved)?;
        fit_layout_joint(args, &params, &mut resolved, &prep)
    } else {
        fit_cell_layout_umap(args, &params, &mut resolved)
    }
}

/// `--joint`: cells and the feature co-embedding in one t-UMAP, written as
/// the `joint` method (cells, and features on that same map).
fn fit_layout_joint(
    args: &LayoutUmapArgs,
    params: &UmapLayoutArgs,
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
    let (cell_coords, feature_coords) =
        joint_umap_layout_2d_of_columns(&prep.cells_kn, &coembed_dh.transpose(), params)?;
    write_cell_layout(
        &args.common,
        resolved,
        prep,
        &cell_coords,
        "joint",
        Some((&names, &feature_coords)),
    )
}

/// t-UMAP of the run's PBs on the graph between them, then every cell
/// placed by Nyström; written as the PB layouts write theirs.
fn fit_cell_layout_umap(
    args: &LayoutUmapArgs,
    params: &UmapLayoutArgs,
    resolved: &mut ResolvedViz,
) -> anyhow::Result<()> {
    let prep = preprocess_layout_data(&args.common, resolved)?;
    let n = prep.pbs.n_pb();
    let umap = params.umap(n);
    info!(
        "UMAP SGD on the PB graph: {n} PBs (epochs={}) ...",
        umap.n_epochs
    );
    let pb_coords = umap_layout_2d_on_graph(n, &prep.pbs.pb_edges, &umap);
    info!("UMAP done");
    finalize_viz(&args.common, resolved, &prep, &pb_coords, "umap")
}

impl LayoutUmapArgs {
    /// The SGD's settings, the kernel's range checked.
    pub(crate) fn umap_layout_args(&self) -> anyhow::Result<UmapLayoutArgs> {
        Ok(UmapLayoutArgs {
            knn: self.common.knn,
            block_size: self.common.block_size.unwrap_or(1000),
            epochs: self.umap_epochs,
            sgd: Umap {
                negative_sample_rate: self.umap_negative_rate,
                learning_rate: self.umap_lr,
                seed: self.common.seed,
                ..kernel(self.umap_min_dist, self.umap_spread)?
            },
        })
    }
}

/// The low-d kernel for `min_dist` and `spread`: t-UMAP's when neither is
/// given, else UMAP's fitted to them, whichever is not given taken as
/// umap-learn's.
fn kernel(min_dist: Option<f32>, spread: Option<f32>) -> anyhow::Result<Umap> {
    if min_dist.is_none() && spread.is_none() {
        return Ok(Umap::tumap());
    }
    let (min_dist, spread) = (min_dist.unwrap_or(MIN_DIST), spread.unwrap_or(SPREAD));
    let umap = Umap::tumap()
        .with_min_dist(min_dist, spread)
        .map_err(|e| anyhow::anyhow!("{e} (--umap-min-dist, --umap-spread)"))?;
    info!(
        "UMAP kernel for min_dist {min_dist}, spread {spread}: a = {:.4}, b = {:.4}",
        umap.a, umap.b
    );
    Ok(umap)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ab(u: &Umap) -> (f32, f32) {
        (u.a, u.b)
    }

    fn fitted(min_dist: f32, spread: f32) -> (f32, f32) {
        ab(&Umap::tumap().with_min_dist(min_dist, spread).unwrap())
    }

    #[test]
    fn without_min_dist_or_spread_the_kernel_stays_t_umap() {
        assert_eq!(ab(&kernel(None, None).unwrap()), (1.0, 1.0));
        // Either alone takes umap-learn's default for the other.
        assert_eq!(ab(&kernel(None, Some(1.0)).unwrap()), fitted(0.1, 1.0));
        assert_eq!(ab(&kernel(Some(0.3), None).unwrap()), fitted(0.3, 1.0));
        assert!(kernel(Some(1.5), Some(1.0)).is_err());
    }

    #[test]
    fn the_umap_flags_reach_the_sgd() {
        #[derive(clap::Parser)]
        struct Cli {
            #[command(flatten)]
            args: LayoutUmapArgs,
        }
        let params = |flags: &[&str]| {
            let argv = ["layout", "-f", "run.senna.json"].iter().chain(flags);
            <Cli as clap::Parser>::try_parse_from(argv)
                .unwrap()
                .args
                .umap_layout_args()
                .unwrap()
        };
        let p = params(&[]);
        assert_eq!(ab(&p.sgd), (1.0, 1.0));
        assert_eq!(p.umap(10).n_epochs, 500);
        let p = params(&[
            "--umap-min-dist",
            "0.5",
            "--umap-spread",
            "2",
            "--umap-epochs",
            "7",
            "--umap-lr",
            "0.5",
            "--seed",
            "3",
        ]);
        assert_eq!(ab(&p.sgd), fitted(0.5, 2.0));
        let u = p.umap(10);
        assert_eq!((u.n_epochs, u.learning_rate, u.seed), (7, 0.5, 3));
    }
}
