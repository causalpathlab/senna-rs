//! The gene modules the kinetics are shared within: the fitted genes grouped
//! by their pseudobulk totals with `data-beans`' feature coarsening (the
//! grouping the base fit's partition uses), at a count of our own. Genes whose
//! totals carry no grouping evidence (its background group) are left out of
//! the kinetics.

use data_beans::alg::feature_coarsening::{coarsen_features, informative_features};
use nalgebra::DMatrix;

/// Per gene of `totals` `[G × P]` (pseudobulk sizes `sizes`), its module below
/// the returned count, or `None` for a gene with no grouping evidence.
pub fn gene_modules(
    totals: &DMatrix<f32>,
    sizes: &[f32],
    n_modules: usize,
    seed: u64,
) -> anyhow::Result<(Vec<Option<u32>>, usize)> {
    let informative = informative_features(totals, sizes);
    let has_background = informative.iter().any(|&i| !i);
    // The background takes one of the coarsening's groups.
    let target = n_modules + usize::from(has_background);
    let level = coarsen_features(totals, sizes, &[target], seed)?
        .pop()
        .expect("one level requested");
    let background = has_background.then(|| level.num_coarse - 1);
    // Renumber the remaining groups densely.
    let mut id = vec![None; level.num_coarse];
    let mut next = 0u32;
    for (c, slot) in id.iter_mut().enumerate() {
        if Some(c) != background && !level.coarse_to_fine[c].is_empty() {
            *slot = Some(next);
            next += 1;
        }
    }
    let modules = level.fine_to_coarse.iter().map(|&c| id[c]).collect();
    Ok((modules, next as usize))
}

#[cfg(test)]
#[path = "modules/tests.rs"]
mod tests;
