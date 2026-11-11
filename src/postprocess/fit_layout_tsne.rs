//! `senna layout tsne`: the run's PBs laid out by an Rtsne-style t-SNE whose
//! affinities are the graph between them (the cell kNN graph, coarsened and
//! collapsed), taken as they are rather than calibrated to a perplexity.
//! Cells placed by cheap Nyström. Initialised from random 2D
//! coordinates (seeded via the shared `--seed` flag).

use super::fit_layout_common::{
    finalize_viz, preprocess_layout_data, resolve_inputs, LayoutCommonArgs,
};
use super::fit_layout_features::LayoutTarget;
use crate::geometry::tsne::TSne;
use legume_numeric::matrix::pca::random_init_2d;
use senna::embed_common::*;

#[derive(Args, Debug)]
pub struct LayoutTsneArgs {
    #[clap(flatten)]
    common: LayoutCommonArgs,

    #[arg(long, default_value_t = 1000, help = "Number of iterations for t-SNE")]
    tsne_iter: usize,

    #[arg(
        long,
        default_value_t = 0.0,
        help = "densMAP density-preservation strength (λ)"
    )]
    tsne_density_lambda: f32,
}

pub fn fit_layout_tsne(args: &LayoutTsneArgs) -> anyhow::Result<()> {
    anyhow::ensure!(
        args.common.target == LayoutTarget::Cells,
        "`layout tsne` lays out cells only; use `layout umap` or `layout phate` \
         with --target features"
    );
    let mut resolved = resolve_inputs(&args.common)?;
    // t-SNE is PB-level (the PB graph → 2D); cells placed by Nyström.
    let prep = preprocess_layout_data(&args.common, &resolved)?;
    let n = prep.pb_kp.ncols();
    info!(
        "Running t-SNE on the PB graph (iter={}, seed={}) on {n} PBs ...",
        args.tsne_iter, args.common.seed
    );
    let init = random_init_2d(n, args.common.seed);
    let result = TSne::default()
        .n_iter(args.tsne_iter)
        .weights(&prep.pb_size)
        .density_lambda(args.tsne_density_lambda)
        .fit_graph(n, &prep.pb_edges, &init)
        .map_err(|e| anyhow::anyhow!("t-SNE failed: {e}"))?;

    let mut pb_coords = Mat::zeros(n, 2);
    for i in 0..n {
        pb_coords[(i, 0)] = result[i * 2];
        pb_coords[(i, 1)] = result[i * 2 + 1];
    }
    info!("t-SNE done");

    finalize_viz(&args.common, &mut resolved, &prep, &pb_coords, "tsne")
}
