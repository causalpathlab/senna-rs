//! Gene-level HVG projection weights for `senna tde`.
//!
//! A tde gene carries a spliced (base) row and usually an unspliced row. The
//! spliced counts alone define the cell state, so genes are ranked on their
//! spliced rows only: a gene whose variance shows up only in its unspliced
//! counts earns no projection weight. [`hvg_row_weights`] hands back a per-ROW
//! weight vector, 1.0 only on the base row (`count/spliced`) of a selected
//! gene and 0 on every other row.

use data_beans::alg::hvg::{
    load_must_train, select_hvg_by_stats, union_indices, HvgCliArgs, MustTrainFeatures,
};
use data_beans::alg::sparse_streaming::streaming_sparse_running_stats;
use data_beans::utilities::name_matching::GeneIndex;
use graph_embedding_util as ge;
use legume_numeric::matrix::traits::RunningStatOps;
use log::info;
use rustc_hash::FxHashSet;

use crate::tde::tracks::TrackPlan;

/// Per-row HVG projection weights over a tde feature axis. `None` when
/// selection is off: the projection then reads every live (spliced) row
/// unweighted, as `senna bge` does.
///
/// - `--feature-list-file` REPLACES the ranking with exactly the named
///   genes, resolved against `plan.gene_names` (lenient matching, see
///   [`MustTrainFeatures::resolve_with`]).
/// - Otherwise, when `hvg.n_hvg > 0`, genes are ranked by NB dispersion-trend
///   excess over the `(mean, variance)` of their base (spliced) row; a gene's
///   other rows do not enter its rank.
/// - `--must-train-features` UNIONS a curated panel into the selection,
///   resolved the same way, regardless of which of the two rules produced
///   the base selection.
/// - Weights: `w[r] = 1.0` iff `plan.base_rows[r]` and `row_gene[r]` was
///   selected, else `0.0` — computed over ALL selected genes, `Some` even
///   when nothing was selected (an all-zero vector), so the caller never has
///   to re-derive "selection ran but kept nothing" from a `None`.
pub(crate) fn hvg_row_weights(
    unified: &ge::UnifiedData,
    plan: &TrackPlan,
    hvg: &HvgCliArgs,
    block_size: Option<usize>,
) -> anyhow::Result<Option<Vec<f32>>> {
    let selection_on = hvg.n_hvg > 0 || hvg.feature_list_file.is_some();
    if !selection_on {
        return Ok(None);
    }

    let n_genes = plan.gene_names.len();
    let gene_index = GeneIndex::build(&plan.gene_names);

    let mut selected: Vec<usize> = if let Some(path) = hvg.feature_list_file.as_deref() {
        MustTrainFeatures::load(path)?.resolve_with(&gene_index)
    } else {
        let stat = streaming_sparse_running_stats(unified.count_backend(), block_size, "tde HVG")?;
        let (means, vars) = (stat.mean(), stat.variance());
        let mut gmean = vec![0f32; n_genes];
        let mut gvar = vec![0f32; n_genes];
        for (r, (&m, &v)) in means.iter().zip(vars.iter()).enumerate() {
            if plan.base_rows[r] {
                gmean[plan.row_gene[r] as usize] = m;
                gvar[plan.row_gene[r] as usize] = v;
            }
        }
        select_hvg_by_stats(&gmean, &gvar, hvg.n_hvg)
    };

    if let Some(must_train) = load_must_train(hvg.must_train_features.as_deref(), selection_on)? {
        let forced = must_train.resolve_with(&gene_index);
        let added = union_indices(&mut selected, &forced);
        info!(
            "tde HVG: {added} gene(s) force-added on top of the selection ({} of the {} \
             matched were already selected)",
            forced.len() - added,
            forced.len()
        );
    }

    let keep: FxHashSet<usize> = selected.into_iter().collect();
    let mut w = vec![0.0f32; unified.n_features()];
    for (r, slot) in w.iter_mut().enumerate() {
        if plan.base_rows[r] && keep.contains(&(plan.row_gene[r] as usize)) {
            *slot = 1.0;
        }
    }
    let n_weighted = w.iter().filter(|&&x| x > 0.0).count();
    info!(
        "tde HVG (--n-hvg {}): {} of {n_genes} genes selected, ranked on their spliced rows; \
         {n_weighted} of {} feature row(s) carry the projection weight",
        hvg.n_hvg,
        keep.len(),
        unified.n_features()
    );
    Ok(Some(w))
}

#[cfg(test)]
#[path = "hvg/tests.rs"]
mod tests;
