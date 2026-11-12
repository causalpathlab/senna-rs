//! `senna layout lle`: modified LLE of the run's PBs on the graph between
//! them (its cell kNN graph, coarsened and collapsed), each PB rebuilt from
//! its graph neighbours' mean features. Cells placed by cheap Nyström.

use super::fit_layout_common::{
    finalize_viz, preprocess_layout_data, resolve_inputs, LayoutCommonArgs,
};
use super::fit_layout_features::LayoutTarget;
use legume_numeric::matrix::layout::lle_layout_2d_on_graph;
use senna::embed_common::*;

#[derive(Args, Debug)]
pub struct LayoutLleArgs {
    #[clap(flatten)]
    common: LayoutCommonArgs,
}

pub fn fit_layout_lle(args: &LayoutLleArgs) -> anyhow::Result<()> {
    anyhow::ensure!(
        args.common.target == LayoutTarget::Cells,
        "`layout lle` lays out cells only; use `layout umap` or `layout phate` \
         with --target features"
    );
    let mut resolved = resolve_inputs(&args.common)?;
    // PB-level, as PHATE: the PBs' mean features, their neighbours in the
    // graph between them; cells then placed by Nyström.
    let prep = preprocess_layout_data(&args.common, &resolved)?;
    let pb_coords = lle_layout_2d_on_graph(&prep.pb_kp.transpose(), &prep.pb_edges);
    finalize_viz(&args.common, &mut resolved, &prep, &pb_coords, "lle")
}
