//! `senna layout phate`: PB landmarks of the run's cell table laid out
//! with PHATE diffusion embedding over their mean features.
//! Cells placed by cheap Nyström.

use super::fit_layout_common::{
    finalize_viz, preprocess_layout_data, resolve_inputs, LayoutCommonArgs, PbLayoutPrep,
    PhateCliArgs, ResolvedViz,
};
use super::fit_layout_features::{load_feature_layout_input, write_feature_layout, LayoutTarget};
use super::viz_prep::apply_svd_preprocessing;
use crate::geometry::cell_layout::project_cells_nystrom;
use crate::geometry::orient::rotate_root_to_bottom;
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

    #[arg(
        long,
        default_value_t = false,
        help = "Rotate so the pseudotime root sits at the bottom of the layout",
        long_help = "Rigid 2D rotation of the PHATE layout.\n\
                     It puts the pseudotime root anchor below the tip anchor, along the y-axis.\n\
                     PHATE coords are defined only up to rotation and reflection,\n\
                     so this is principled. All distances and cluster structure are preserved.\n\
                     \n\
                     `manifest.pseudotime` must already be populated by `lupin pseudotime`.\n\
                     Otherwise orientation is refused (senna no longer fits pseudotime)."
    )]
    orient_by_root: bool,

    #[arg(long, help = "Ignored; root selection belongs to `lupin pseudotime`")]
    root_cell: Option<Box<str>>,

    #[arg(long, help = "Ignored; root selection belongs to `lupin pseudotime`")]
    root_node: Option<usize>,

    #[arg(
        long,
        default_value_t = 0.10,
        help = "Anchor-bin quantile width for orientation (bottom/top fraction)"
    )]
    orient_tip_quantile: f32,
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

    let pb_coords = if args.orient_by_root {
        let pseudotime = obtain_pseudotime(args, &resolved, &prep)?;
        let pb_pt = aggregate_cell_pt_to_pb(&pseudotime, &prep);
        rotate_root_to_bottom(&pb_coords, &pb_pt, args.orient_tip_quantile)?
    } else {
        pb_coords
    };

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

/// Resolve per-cell pseudotime for orientation from a cached
/// `manifest.pseudotime.pseudotime` parquet written by `lupin pseudotime`.
fn obtain_pseudotime(
    _args: &LayoutPhateArgs,
    resolved: &ResolvedViz,
    prep: &PbLayoutPrep,
) -> anyhow::Result<Vec<f32>> {
    let cell_names = &prep.cells.names;
    let pt_rel = resolved
        .manifest
        .pseudotime
        .pseudotime
        .as_deref()
        .ok_or_else(|| {
            anyhow::anyhow!(
                "--orient-by-root needs `manifest.pseudotime.pseudotime`; \
             run `lupin pseudotime --from ...` first"
            )
        })?;
    let pt_path = resolved.resolve(pt_rel);
    info!("Reading cached pseudotime from {pt_path}");
    let MatWithNames {
        rows: pt_cells,
        mat: pt_mat,
        ..
    } = read_mat(&pt_path)?;
    anyhow::ensure!(
        pt_mat.nrows() == cell_names.len(),
        "pseudotime has {} rows but data has {} cells",
        pt_mat.nrows(),
        cell_names.len()
    );
    if &pt_cells != cell_names {
        log::warn!("pseudotime row names differ from data cell names — using positional alignment");
    }
    Ok((0..pt_mat.nrows()).map(|i| pt_mat[(i, 0)]).collect())
}

/// Mean per-PB pseudotime; NaN for PBs with no contributing cells (dropped
/// by coverage filter or all-NaN inputs). Aligns to `pb_coords` row order
/// via `prep.pb_membership_kept`.
fn aggregate_cell_pt_to_pb(pseudotime: &[f32], prep: &PbLayoutPrep) -> Vec<f32> {
    let n_pb = prep.pb_kp.ncols();
    let mut sum = vec![0.0_f32; n_pb];
    let mut count = vec![0_usize; n_pb];
    for (cell_i, &pb_id) in prep.pb_membership_kept.iter().enumerate() {
        if pb_id == usize::MAX {
            continue;
        }
        let pt = pseudotime[cell_i];
        if !pt.is_finite() {
            continue;
        }
        sum[pb_id] += pt;
        count[pb_id] += 1;
    }
    (0..n_pb)
        .map(|i| {
            if count[i] > 0 {
                sum[i] / count[i] as f32
            } else {
                f32::NAN
            }
        })
        .collect()
}
