//! `senna layout {stress|forceatlas2|diffusion|radial}`: graph layouts of
//! the run's PBs on the graph between them (its cell kNN graph, coarsened
//! and collapsed). Each reads only that graph; cells are then placed by
//! cheap Nyström.

use super::fit_layout_common::{
    finalize_viz, preprocess_layout_data, resolve_inputs, LayoutCommonArgs,
};
use super::fit_layout_features::LayoutTarget;
use legume_numeric::matrix::layout::{
    diffusion_map_2d_on_graph, forceatlas2_layout_2d_on_graph, radial_tree_layout_2d_on_graph,
    stress_layout_2d_on_graph,
};
use senna::embed_common::*;

type Edges = [(usize, usize, f32)];

/// Lays the run's PBs out by `layout` on the graph between them (`n` PBs
/// and their edges), places the cells and writes it all under `method`.
fn fit_on_pb_graph(
    common: &LayoutCommonArgs,
    method: &str,
    layout: impl FnOnce(usize, &Edges) -> Mat,
) -> anyhow::Result<()> {
    anyhow::ensure!(
        common.target == LayoutTarget::Cells,
        "`layout {method}` lays out cells only; use `layout umap` or `layout phate` \
         with --target features"
    );
    let mut resolved = resolve_inputs(common)?;
    let prep = preprocess_layout_data(common, &resolved)?;
    let pb_coords = layout(prep.pb_kp.ncols(), &prep.pb_edges);
    finalize_viz(common, &mut resolved, &prep, &pb_coords, method)
}

#[derive(Args, Debug)]
pub struct LayoutStressArgs {
    #[clap(flatten)]
    common: LayoutCommonArgs,
}

pub fn fit_layout_stress(args: &LayoutStressArgs) -> anyhow::Result<()> {
    let seed = args.common.seed;
    fit_on_pb_graph(&args.common, "stress", |n, e| {
        stress_layout_2d_on_graph(n, e, seed)
    })
}

#[derive(Args, Debug)]
pub struct LayoutForceAtlas2Args {
    #[clap(flatten)]
    common: LayoutCommonArgs,

    #[arg(long, default_value_t = 500, help = "ForceAtlas2 iterations")]
    iterations: usize,
}

pub fn fit_layout_forceatlas2(args: &LayoutForceAtlas2Args) -> anyhow::Result<()> {
    let (iterations, seed) = (args.iterations, args.common.seed);
    fit_on_pb_graph(&args.common, "forceatlas2", |n, e| {
        forceatlas2_layout_2d_on_graph(n, e, iterations, seed)
    })
}

#[derive(Args, Debug)]
pub struct LayoutDiffusionArgs {
    #[clap(flatten)]
    common: LayoutCommonArgs,

    #[arg(
        long,
        default_value_t = 1,
        help = "Diffusion time t: each component is scaled by its eigenvalue to the t"
    )]
    diffusion_t: usize,
}

pub fn fit_layout_diffusion(args: &LayoutDiffusionArgs) -> anyhow::Result<()> {
    let t = args.diffusion_t;
    fit_on_pb_graph(&args.common, "diffusion", |n, e| {
        diffusion_map_2d_on_graph(n, e, t)
    })
}

#[derive(Args, Debug)]
pub struct LayoutRadialArgs {
    #[clap(flatten)]
    common: LayoutCommonArgs,
}

pub fn fit_layout_radial(args: &LayoutRadialArgs) -> anyhow::Result<()> {
    fit_on_pb_graph(&args.common, "radial", |n, e| {
        radial_tree_layout_2d_on_graph(n, e, None)
    })
}
