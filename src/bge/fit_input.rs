//! The one bridge between data-beans' pseudobulks and the embedding's fit:
//! lge trains on plain matrices and a count stream, data-beans builds them.

use data_beans::unified::{EmbeddingPseudobulks, UnifiedData};
use graph_embedding_util as ge;
use nalgebra::DMatrix;
use std::borrow::Cow;

/// Fit the embedding on `unified`'s cells and the pseudobulks built from
/// them (`data_beans::unified::build_embedding_pseudobulks`).
pub(crate) fn fit_on_pseudobulks(
    unified: &UnifiedData,
    pseudobulks: &EmbeddingPseudobulks,
    config: ge::FitConfig,
) -> anyhow::Result<ge::FitOutput> {
    let rates: Vec<Cow<'_, DMatrix<f32>>> = (0..pseudobulks.collapsed_levels.len())
        .map(|l| pseudobulks.rates(l))
        .collect();
    let levels = rates
        .iter()
        .zip(&pseudobulks.cell_to_pb_per_level)
        .map(|(rates, cell_to_pb)| ge::PbLevel { rates, cell_to_pb })
        .collect();
    let delta = pseudobulks.finest_delta();
    let finest_counts = || pseudobulks.finest_counts();
    ge::fit(
        ge::FitInput {
            cells: unified,
            batch_membership: &unified.batch_membership,
            batch_names: &unified.batch_names,
            feature_modality: unified.feature_modality.clone(),
            levels,
            finest_counts: &finest_counts,
            batch_delta: delta.as_deref().map(|delta| ge::BatchDelta {
                delta,
                collapse_batch_names: &pseudobulks.collapse_batch_names,
            }),
        },
        config,
    )
}
