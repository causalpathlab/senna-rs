//! `senna layout phate`: PHATE on the graph between the run's PBs (its cell
//! kNN graph, coarsened and collapsed). Cells placed by cheap Nyström.

use super::fit_layout_common::{
    finalize_viz, preprocess_layout_data, resolve_inputs, LayoutCommonArgs, PhateCliArgs,
};
use super::fit_layout_features::{feature_pb_layout, LayoutTarget};
use legume_numeric::matrix::layout::phate_layout_2d_on_graph;
use senna::embed_common::*;

#[derive(Args, Debug)]
pub struct LayoutPhateArgs {
    #[clap(flatten)]
    common: LayoutCommonArgs,

    #[clap(flatten)]
    phate: PhateCliArgs,
}

pub fn fit_layout_phate(args: &LayoutPhateArgs) -> anyhow::Result<()> {
    if args.common.target == LayoutTarget::Features {
        let phate = (&args.phate).into();
        return feature_pb_layout(&args.common, "phate", |pbs| {
            Ok(phate_layout_2d_on_graph(pbs.n_pb(), &pbs.pb_edges, &phate))
        });
    }
    let mut resolved = resolve_inputs(&args.common)?;
    // PHATE is PB-level (diffusion-MDS, O(n³) MDS): it diffuses on the
    // graph between the PBs, the run's cell graph collapsed onto them;
    // cells are then placed by Nyström.
    let prep = preprocess_layout_data(&args.common, &resolved)?;
    let pb_coords =
        phate_layout_2d_on_graph(prep.pbs.n_pb(), &prep.pbs.pb_edges, &(&args.phate).into());

    finalize_viz(&args.common, &mut resolved, &prep, &pb_coords, "phate")
}
