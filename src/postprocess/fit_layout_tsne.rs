//! `senna layout tsne`: the run's PBs laid out by an Rtsne-style t-SNE whose
//! affinities are the graph between them (the cell kNN graph, coarsened and
//! collapsed), taken as they are rather than calibrated to a perplexity.
//! Cells placed by cheap Nyström. Initialised from random 2D
//! coordinates (seeded via the shared `--seed` flag). `--target features`
//! takes the same route over the feature embedding (`feature_pb_layout`).

use super::fit_layout_common::{
    finalize_viz, preprocess_layout_data, resolve_inputs, LayoutCommonArgs,
};
use super::fit_layout_features::{feature_pb_layout, LayoutTarget};
use legume_numeric::matrix::layout::{tsne_layout_2d_on_graph, Coarsened, TsneArgs};
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
    if args.common.target == LayoutTarget::Features {
        return feature_pb_layout(&args.common, "tsne", |pbs| tsne_of_pbs(args, pbs));
    }
    let mut resolved = resolve_inputs(&args.common)?;
    // t-SNE is PB-level (the PB graph → 2D); cells placed by Nyström.
    let prep = preprocess_layout_data(&args.common, &resolved)?;
    let pb_coords = tsne_of_pbs(args, &prep.pbs)?;
    finalize_viz(&args.common, &mut resolved, &prep, &pb_coords, "tsne")
}

/// t-SNE of the PBs on the graph between them, each weighted by its size.
fn tsne_of_pbs(args: &LayoutTsneArgs, pbs: &Coarsened) -> anyhow::Result<Mat> {
    let n = pbs.n_pb();
    info!(
        "Running t-SNE on the PB graph (iter={}, seed={}) on {n} PBs ...",
        args.tsne_iter, args.common.seed
    );
    let pb_coords = tsne_layout_2d_on_graph(
        n,
        &pbs.pb_edges,
        &TsneArgs {
            n_iter: args.tsne_iter,
            density_lambda: args.tsne_density_lambda,
            weights: Some(pbs.pb_size.clone()),
            seed: args.common.seed,
        },
    )
    .map_err(|e| anyhow::anyhow!("t-SNE failed: {e}"))?;
    info!("t-SNE done");
    Ok(pb_coords)
}
