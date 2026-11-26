//! Coarse groups of similar cells from a fit's random projection: the
//! cells' top SVD components, with chosen per-cell covariates regressed out
//! of each, standardised and clustered by legume-numeric's seeded k-means.
//!
//! A rule that asks "is this group of like cells all of one kind?" (the
//! cell-cycle covariate's one-phase guard) needs groups that the kind
//! itself did not make: regressing its covariates out first keeps a group
//! from being split by them alone.

use legume_numeric::matrix::kmeans::{
    kmeans_rows_seeded, KmeansMetric, KmeansRowsFit, KmeansRowsOpts,
};
use legume_numeric::matrix::traits::SymmetricTopEigen;
use nalgebra::DMatrix;

/// Each cell's group, `0..n_groups`: k-means of its top `n_dims` SVD
/// components of `proj_kn` (dims × cells, centred across cells), each
/// with the `remove` covariates (one value per cell each) regressed out by
/// least squares on them and an intercept, then standardised.
///
/// # Errors
/// A covariate whose length is not the number of cells, or an eigen
/// solve that fails.
pub fn coarse_groups(
    proj_kn: &DMatrix<f32>,
    remove: &[&[f32]],
    n_groups: usize,
    n_dims: usize,
    seed: u64,
) -> anyhow::Result<Vec<usize>> {
    let n = proj_kn.ncols();
    anyhow::ensure!(
        remove.iter().all(|c| c.len() == n),
        "a covariate to remove has a value per cell"
    );
    if n == 0 {
        return Ok(Vec::new());
    }
    // Centred across cells: uncentred, the first component would be the
    // cells' shared offset, not a direction they vary along.
    let mut x = proj_kn.clone();
    for mut row in x.row_iter_mut() {
        let m = row.mean();
        row.add_scalar_mut(-m);
    }
    // Top right singular vectors through the small dims × dims Gram matrix;
    // the components (cells × r) are then one product, each column mean 0.
    let r = n_dims.min(x.nrows()).max(1);
    let gram = (&x * x.transpose()).map(f64::from);
    let u = gram.top_eigen(r)?.vectors.map(|v| v as f32);
    let mut comp = x.transpose() * u;
    if !remove.is_empty() {
        // The covariates centred, which takes the intercept; components
        // are already centred.
        let mut c = DMatrix::from_fn(n, remove.len(), |i, j| remove[j][i]);
        for mut col in c.column_iter_mut() {
            let m = col.mean();
            col.add_scalar_mut(-m);
        }
        if let Some(inv) = (c.transpose() * &c).map(f64::from).try_inverse() {
            let beta = inv.map(|v| v as f32) * (c.transpose() * &comp);
            comp -= &c * beta;
        }
    }
    crate::embed_common::zscore_columns_inplace(&mut comp);
    Ok(seeded_kmeans(&comp, n_groups.clamp(1, n), seed).labels)
}

/// Seeded Euclidean k-means of the rows of `x`, run to convergence (at
/// most 100 iterations), as the cell-cycle covariate's bins and groups use.
#[must_use]
pub fn seeded_kmeans(x: &DMatrix<f32>, k: usize, seed: u64) -> KmeansRowsFit {
    kmeans_rows_seeded(
        x,
        &KmeansRowsOpts {
            k,
            max_iter: 100,
            seed,
            metric: KmeansMetric::Euclidean,
            min_changed_frac: 0.0,
            init_sample: 0,
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn groups_follow_the_structure_left_once_a_covariate_is_removed() {
        // 200 cells in two kinds along dimension 0; dimension 1 is the
        // covariate, larger than the kinds' gap.
        let n = 200;
        let kind: Vec<f32> = (0..n).map(|i| if i < 100 { -1.0 } else { 1.0 }).collect();
        let cov: Vec<f32> = (0..n)
            .map(|i| if i % 2 == 0 { -5.0 } else { 5.0 })
            .collect();
        let proj = DMatrix::from_fn(2, n, |d, i| if d == 0 { kind[i] } else { cov[i] });
        let g = coarse_groups(&proj, &[&cov], 2, 2, 7).unwrap();
        // With the covariate out, the groups are the kinds.
        let first = g[0];
        assert!(g[..100].iter().all(|&x| x == first));
        assert!(g[100..].iter().all(|&x| x != first));
        assert!(coarse_groups(&proj, &[&cov[..10]], 2, 2, 7).is_err());
    }
}
