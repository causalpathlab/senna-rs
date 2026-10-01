//! `senna layout tsne`: PB landmarks of the run's cell table laid out
//! with an Rtsne-style t-SNE over the PB-PB similarity. Cells
//! placed by cheap Nyström. Initialised from random 2D
//! coordinates (seeded via the shared `--seed` flag).

use super::fit_layout_common::{
    finalize_viz, preprocess_layout_data, random_init_2d, resolve_inputs, LayoutCommonArgs,
};
use super::fit_layout_features::LayoutTarget;
use crate::geometry::tsne::{similarity_to_distance, TSne};
use senna::embed_common::*;

#[derive(Args, Debug)]
pub struct LayoutTsneArgs {
    #[clap(flatten)]
    common: LayoutCommonArgs,

    #[arg(long, default_value_t = 30.0, help = "Perplexity for t-SNE")]
    perplexity: f32,

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
    // t-SNE is PB-level (PB-PB similarity → 2D): landmarks of the run's cell
    // table, cells placed by Nyström.
    let prep = preprocess_layout_data(&args.common, &resolved)?;

    let n = prep.pb_similarity.nrows();

    info!(
        "Initialising t-SNE from random 2D coordinates (seed={})",
        args.common.seed
    );
    let init = random_init_2d(n, args.common.seed);
    // TSne::fit expects row-major flat.
    let mut init_flat = Vec::with_capacity(n * 2);
    for i in 0..n {
        init_flat.push(init[(i, 0)]);
        init_flat.push(init[(i, 1)]);
    }

    let mut sim_flat = Vec::with_capacity(n * n);
    for i in 0..n {
        for j in 0..n {
            sim_flat.push(prep.pb_similarity[(i, j)]);
        }
    }
    let distances = similarity_to_distance(&sim_flat, n);

    let tsne = TSne::default()
        .perplexity(args.perplexity)
        .n_iter(args.tsne_iter)
        .weights(&prep.pb_size)
        .density_lambda(args.tsne_density_lambda);
    info!(
        "Running t-SNE (perplexity={}, iter={}) on {} PBs ...",
        args.perplexity, args.tsne_iter, n
    );
    let result = tsne
        .fit(&distances, n, Some(&init_flat))
        .map_err(|e| anyhow::anyhow!("t-SNE failed: {e}"))?;

    let mut pb_coords = Mat::zeros(n, 2);
    for i in 0..n {
        pb_coords[(i, 0)] = result[i * 2];
        pb_coords[(i, 1)] = result[i * 2 + 1];
    }
    info!("t-SNE done");

    finalize_viz(&args.common, &mut resolved, &prep, &pb_coords, "tsne")
}
