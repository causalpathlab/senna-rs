//! `senna layout phate`: PB landmarks of the run's cell table laid out
//! with PHATE diffusion embedding over their mean features.
//! Cells placed by cheap Nyström.

use super::fit_layout_common::{
    finalize_viz, preprocess_layout_data, resolve_inputs, LayoutCommonArgs, PhateCliArgs,
};
use super::fit_layout_features::{load_feature_layout_input, write_feature_layout, LayoutTarget};
use super::viz_prep::apply_svd_preprocessing;
use crate::geometry::cell_layout::project_cells_nystrom;
use crate::geometry::phate::phate_layout_2d;
use senna::embed_common::*;

#[derive(Args, Debug)]
pub struct LayoutPhateArgs {
    #[clap(flatten)]
    common: LayoutCommonArgs,

    #[clap(flatten)]
    phate: PhateCliArgs,

    #[arg(
        long,
        default_value_t = 0,
        help = "SVD preprocessing: keep top N components (0 = skip, use raw features)"
    )]
    svd_dims: usize,
}

pub fn fit_layout_phate(args: &LayoutPhateArgs) -> anyhow::Result<()> {
    if args.common.target == LayoutTarget::Features {
        return fit_feature_layout_phate(args);
    }
    let mut resolved = resolve_inputs(&args.common)?;
    // PHATE is PB-level (diffusion-MDS over landmark features → 2D, O(n³)
    // MDS): landmarks of the run's cell table, cells placed by Nyström.
    let prep = preprocess_layout_data(&args.common, &resolved)?;

    // Each PB's mean cell features, a row per PB.
    let features = prep.pb_kp.transpose();
    let features = if args.svd_dims > 0 {
        apply_svd_preprocessing(&features, args.svd_dims)?
    } else {
        features
    };

    let pb_coords = phate_layout_2d(&features, &(&args.phate).into());

    finalize_viz(&args.common, &mut resolved, &prep, &pb_coords, "phate")
}

/// `--target features`: PHATE of the feature embedding on its own. PHATE's
/// MDS is cubic in the point count, so it runs on a seeded subsample of
/// `--n-landmarks` features and the rest are placed by Nyström.
fn fit_feature_layout_phate(args: &LayoutPhateArgs) -> anyhow::Result<()> {
    use rand::SeedableRng;

    let mut input = load_feature_layout_input(&args.common)?;
    let n = input.feat_kn.ncols();
    let mut rng = rand::rngs::SmallRng::seed_from_u64(args.common.seed);
    let mut landmarks: Vec<usize> = rand::seq::index::sample(
        &mut rng,
        n,
        args.common.n_landmarks.clamp(3, n.max(3)).min(n),
    )
    .into_vec();
    landmarks.sort_unstable();
    let landmark_kp = input.feat_kn.select_columns(&landmarks);
    info!(
        "Feature PHATE on {} landmark features of {n}",
        landmarks.len()
    );
    let landmark_xy = phate_layout_2d(&landmark_kp.transpose(), &(&args.phate).into());
    let coords = project_cells_nystrom(
        &input.feat_kn,
        &landmark_kp,
        &landmark_xy,
        args.common.knn,
        args.common.kernel_alpha,
    );
    write_feature_layout(&mut input, "phate", &coords)
}
