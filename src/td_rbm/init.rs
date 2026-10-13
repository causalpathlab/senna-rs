//! Starting times: an ordering of the pseudobulks along their main trajectory
//! in the base space, which the fit then orients and refines.
//!
//! The pseudobulks' states `θ_p` are joined to their `k` nearest neighbours
//! by cosine distance `d`, weighted `exp(−d² / (σ_p σ_q))` with `σ_p` the
//! distance to the `k`-th neighbour, and symmetrised. The ordering is the
//! leading non-trivial diffusion component of that graph, `D^{-1/2} v₂` of
//! `D^{-1/2} W D^{-1/2}`, taken as ranks.

use super::counts::PbTracks;
use nalgebra::{DMatrix, SymmetricEigen};

/// Lowest and highest starting time.
const SPAN: (f32, f32) = (0.05, 0.95);

/// The pseudobulks' neighbour graph: every edge `(p, q, w_pq)` once, `p < q`,
/// for states `theta` `[P × H]`.
#[must_use]
pub fn knn_edges(theta: &DMatrix<f32>, k: usize) -> Vec<(usize, usize, f32)> {
    let p = theta.nrows();
    if p < 2 {
        return Vec::new();
    }
    let k = k.clamp(1, p - 1);
    let mut unit = theta.map(f64::from);
    for mut row in unit.row_iter_mut() {
        let n = row.norm();
        if n > 0.0 {
            row /= n;
        }
    }
    let dist = (unit.clone() * unit.transpose()).map(|c| (1.0 - c).max(0.0));
    let mut neighbours: Vec<Vec<usize>> = Vec::with_capacity(p);
    let mut sigma = vec![0f64; p];
    for i in 0..p {
        let mut order: Vec<usize> = (0..p).filter(|&j| j != i).collect();
        order.sort_by(|&a, &b| dist[(i, a)].total_cmp(&dist[(i, b)]));
        order.truncate(k);
        sigma[i] = dist[(i, order[k - 1])].max(1e-12);
        neighbours.push(order);
    }
    let mut weight = std::collections::BTreeMap::new();
    for i in 0..p {
        for &j in &neighbours[i] {
            let v = (-dist[(i, j)].powi(2) / (sigma[i] * sigma[j])).exp();
            let e = weight.entry((i.min(j), i.max(j))).or_insert(0f64);
            *e = e.max(v);
        }
    }
    weight
        .into_iter()
        .map(|((a, b), w)| (a, b, w as f32))
        .collect()
}

/// The pseudobulks' expression states `[P × H]`, `H = min(dims, P)`: the
/// leading principal components of every pseudobulk's log reads per 10⁴ on
/// `genes`, both tracks together. From the `[P × P]` Gram matrix, as there
/// are far fewer pseudobulks than genes.
#[must_use]
pub fn expression_states(tracks: &PbTracks, genes: &[usize], dims: usize) -> DMatrix<f32> {
    let p = tracks.cells.len();
    let total = &tracks.unspliced + &tracks.spliced;
    let depth: Vec<f64> = total
        .row_iter()
        .map(|r| f64::from(r.sum()).max(1.0))
        .collect();
    let mut x = DMatrix::<f64>::from_fn(p, genes.len(), |i, c| {
        (f64::from(total[(i, genes[c])]) / depth[i] * 1e4).ln_1p()
    });
    for mut col in x.column_iter_mut() {
        let mean = col.mean();
        col.add_scalar_mut(-mean);
    }
    let eig = SymmetricEigen::new(&x * x.transpose());
    let mut by_value: Vec<usize> = (0..p).collect();
    by_value.sort_by(|&a, &b| eig.eigenvalues[b].total_cmp(&eig.eigenvalues[a]));
    let h = dims.min(p);
    DMatrix::<f32>::from_fn(p, h, |i, k| {
        let j = by_value[k];
        (eig.eigenvectors[(i, j)] * eig.eigenvalues[j].max(0.0).sqrt()) as f32
    })
}

/// Starting times in `(0, 1)` for the pseudobulks `theta` `[P × H]`.
#[must_use]
pub fn diffusion_order(theta: &DMatrix<f32>, k: usize) -> Vec<f32> {
    let p = theta.nrows();
    if p < 3 {
        return (0..p)
            .map(|i| SPAN.0 + (SPAN.1 - SPAN.0) * i as f32 / p.max(2) as f32)
            .collect();
    }
    let mut w = DMatrix::<f64>::zeros(p, p);
    for (a, b, v) in knn_edges(theta, k) {
        w[(a, b)] = f64::from(v);
        w[(b, a)] = f64::from(v);
    }
    let inv_sqrt: Vec<f64> = w
        .row_iter()
        .map(|r| 1.0 / r.sum().max(1e-12).sqrt())
        .collect();
    let m = DMatrix::from_fn(p, p, |i, j| w[(i, j)] * inv_sqrt[i] * inv_sqrt[j]);
    let eig = SymmetricEigen::new(m);
    let mut by_value: Vec<usize> = (0..p).collect();
    by_value.sort_by(|&a, &b| eig.eigenvalues[b].total_cmp(&eig.eigenvalues[a]));
    let v2 = eig.eigenvectors.column(by_value[1]);
    let phi: Vec<f64> = (0..p).map(|i| v2[i] * inv_sqrt[i]).collect();
    let mut order: Vec<usize> = (0..p).collect();
    order.sort_by(|&a, &b| phi[a].total_cmp(&phi[b]));
    let mut tau = vec![0f32; p];
    for (rank, &i) in order.iter().enumerate() {
        tau[i] = SPAN.0 + (SPAN.1 - SPAN.0) * (rank as f32 + 0.5) / p as f32;
    }
    tau
}

#[cfg(test)]
#[path = "tests/init.rs"]
mod tests;
