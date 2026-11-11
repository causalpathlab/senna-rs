//! `senna layout sugiyama`: a layered drawing of the run's PBs on the graph
//! between them (its cell kNN graph, coarsened and collapsed), top to
//! bottom from a root, layers its breadth-first levels and branches side
//! by side. Cells placed by cheap Nyström.
//!
//! The root is one end of the graph's longest path for now; a run that
//! knows its root (a lineage's start) can give it later.

use super::fit_layout_common::{
    finalize_viz, preprocess_layout_data, resolve_inputs, LayoutCommonArgs,
};
use super::fit_layout_features::LayoutTarget;
use legume_numeric::matrix::layout::sugiyama_layout_2d_on_graph;
use senna::embed_common::*;

#[derive(Args, Debug)]
pub struct LayoutSugiyamaArgs {
    #[clap(flatten)]
    common: LayoutCommonArgs,
}

pub fn fit_layout_sugiyama(args: &LayoutSugiyamaArgs) -> anyhow::Result<()> {
    anyhow::ensure!(
        args.common.target == LayoutTarget::Cells,
        "`layout sugiyama` lays out cells only; use `layout umap` or `layout phate` \
         with --target features"
    );
    let mut resolved = resolve_inputs(&args.common)?;
    let prep = preprocess_layout_data(&args.common, &resolved)?;
    let pb_coords = sugiyama_layout_2d_on_graph(prep.pb_kp.ncols(), &prep.pb_edges, None);
    finalize_viz(&args.common, &mut resolved, &prep, &pb_coords, "sugiyama")
}
