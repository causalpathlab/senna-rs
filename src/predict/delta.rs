//! The held-out per-batch rate multiplier δ that scoring needs, from the
//! encoded cells:
//!
//! ```text
//! pred[d,j]   = lib[j] · Σ_k θ̂[j,k] · exp(β[d,k])
//! w[d,j]      = φ[d] / (pred[d,j] + φ[d])     (NB; w = 1 if no φ)
//! δ[d,b]      = Σ_{j∈b} w[d,j]·x[d,j] / Σ_{j∈b} w[d,j]·pred[d,j]
//! ```
//!
//! The Poisson / NB-Fisher-weighted MLE of δ given θ̂. The encoder's own
//! input null is built apart, the way training built it (`predict::null`),
//! so θ̂ does not depend on δ and one pass is all there is.

use crate::logging::new_progress_bar;
use crate::topic::eval::GeneRemap;
use crate::topic::model_metadata::TopicModelMetadata;
use data_beans::sparse_io_vector::SparseIoVec;
use indicatif::ParallelProgressIterator;
use log::info;
use nalgebra_sparse::CscMatrix;
use rayon::prelude::*;
use senna::embed_common::*;

/// Lower / upper bounds on per-(gene, batch) δ. Stops one noisy batch from
/// blowing up a gene's held-out rate when the batch barely covers a gene the
/// model predicts high.
pub const DELTA_CLAMP_MIN: f32 = 0.01;
pub const DELTA_CLAMP_MAX: f32 = 100.0;
/// Floor on the predicted denominator in the obs/pred ratio, avoiding
/// division by zero for genes that no topic produces.
pub const DELTA_PRED_EPS: f32 = 1e-10;

/// Per-batch observed and predicted sums `[D_train, B]`, NB-Fisher-weighted
/// when φ is given; otherwise unweighted (Poisson / multinomial).
pub struct DeltaSums {
    pub pb_obs: Mat,
    pub pb_pred: Mat,
}

/// Closed-form δ from per-batch obs and predicted sums:
/// `δ[d,b] = pb_obs[d,b] / max(pb_pred[d,b], eps)`, clamped to
/// `[DELTA_CLAMP_MIN, DELTA_CLAMP_MAX]`. When every batch shares one law, δ
/// stays near 1.
#[must_use]
pub fn solve_delta_from_sums(sums: &DeltaSums) -> Mat {
    sums.pb_obs.zip_map(&sums.pb_pred, |num, den| {
        (num / den.max(DELTA_PRED_EPS)).clamp(DELTA_CLAMP_MIN, DELTA_CLAMP_MAX)
    })
}

/// NB Fisher information per cell-gene at the working δ = 1 hypothesis,
/// `w = φ[d] / (μ + φ[d])`; `1.0` when `phi` is `None`.
#[must_use]
pub fn nb_fisher_weight(phi: Option<&[f32]>, mu: f32, d: usize) -> f32 {
    match phi {
        None => 1.0,
        Some(phi) => {
            let pd = phi[d].max(1e-6);
            pd / (mu + pd)
        }
    }
}

/// The saved per-gene dispersion φ when the model's decoder is negative
/// binomial (`nb`, `nbmixture`, and the masked `nb_masked*` decoders), so
/// the dense and masked families weight δ by one rule; `None` otherwise.
pub(crate) fn nb_dispersion(
    model: &str,
    metadata: &TopicModelMetadata,
) -> anyhow::Result<Option<Vec<f32>>> {
    let nb = metadata
        .decoder_types
        .first()
        .is_some_and(|t| t.starts_with("nb"));
    if nb {
        crate::topic::model_metadata::load_dispersion(model)
    } else {
        Ok(None)
    }
}

/// What every block reads.
struct DeltaInputs<'a> {
    theta_nk: &'a Mat,
    gene_remap: Option<&'a GeneRemap>,
    /// `exp(β)`, `[D_train, K]`.
    exp_beta_dk: &'a Mat,
    phi: Option<&'a [f32]>,
}

/// Running sums over blocks. `obs` is `[D_train, B]`. `pred` is the
/// weighted `[D_train, B]` predicted sum under NB; with no φ every weight
/// is 1 and it is `Σ_{j∈b} lib_j θ_j`, `[K, B]`, which `exp(β)` turns into
/// the predicted sum once at the end.
struct Acc {
    obs: Mat,
    pred: Mat,
}

impl Acc {
    fn zeros(inp: &DeltaInputs<'_>, n_batches: usize) -> Self {
        let pred_rows = if inp.phi.is_some() {
            inp.exp_beta_dk.nrows()
        } else {
            inp.exp_beta_dk.ncols()
        };
        Self {
            obs: Mat::zeros(inp.exp_beta_dk.nrows(), n_batches),
            pred: Mat::zeros(pred_rows, n_batches),
        }
    }

    fn merge(mut self, other: Self) -> Self {
        self.obs += &other.obs;
        self.pred += &other.pred;
        self
    }

    fn finish(self, inp: &DeltaInputs<'_>) -> DeltaSums {
        let pb_pred = if inp.phi.is_some() {
            self.pred
        } else {
            inp.exp_beta_dk * &self.pred
        };
        DeltaSums {
            pb_obs: self.obs,
            pb_pred,
        }
    }
}

/// Add the cells of one block, `csc` (query rows × cells) with `theta`
/// their rows of `θ̂` and `batch_ids` their batches, to `acc`. The observed
/// sums run over the nonzeros only.
fn accumulate(
    acc: &mut Acc,
    inp: &DeltaInputs<'_>,
    csc: &CscMatrix<f32>,
    theta: nalgebra::DMatrixView<'_, f32>,
    batch_ids: &[usize],
) {
    let d_train = inp.exp_beta_dk.nrows();
    let to_train = |row: usize| match inp.gene_remap {
        Some(r) => r.new_to_train[row],
        None => Some(row),
    };
    let pred_dn = inp.phi.map(|_| inp.exp_beta_dk * theta.transpose());
    for (j, (col, &b)) in csc.col_iter().zip(batch_ids).enumerate() {
        let lib_j: f32 = col.values().iter().sum();
        match pred_dn.as_ref() {
            None => {
                let mut dst = acc.pred.column_mut(b);
                dst.axpy(lib_j, &theta.row(j).transpose(), 1.0);
                for (&row, &x) in col.row_indices().iter().zip(col.values()) {
                    if let Some(d) = to_train(row) {
                        acc.obs[(d, b)] += x;
                    }
                }
            }
            Some(pred_dn) => {
                let mu = |d: usize| lib_j * pred_dn[(d, j)];
                for d in 0..d_train {
                    let m = mu(d);
                    acc.pred[(d, b)] += nb_fisher_weight(inp.phi, m, d) * m;
                }
                for (&row, &x) in col.row_indices().iter().zip(col.values()) {
                    if let Some(d) = to_train(row) {
                        acc.obs[(d, b)] += nb_fisher_weight(inp.phi, mu(d), d) * x;
                    }
                }
            }
        }
    }
}

/// Held-out δ `[D_train, B]` from the cells' proportions `theta_nk`
/// (`[N, K]`, rows summing to 1), or `None` with fewer than two batches: one
/// batch has nothing to contrast. `block_size` is the cells per block
/// (`None` = the default for the row count); this pass keeps no block
/// statistics, so it need not match the encoder's.
pub fn delta_from_theta(
    data_vec: &SparseIoVec,
    theta_nk: &Mat,
    gene_remap: Option<&GeneRemap>,
    beta_dk: &Mat,
    phi: Option<&[f32]>,
    block_size: Option<usize>,
) -> anyhow::Result<Option<Mat>> {
    let n_batches = data_vec.num_batches();
    if n_batches <= 1 {
        return Ok(None);
    }
    let exp_beta_dk = beta_dk.map(f32::exp);
    let inp = DeltaInputs {
        theta_nk,
        gene_remap,
        exp_beta_dk: &exp_beta_dk,
        phi,
    };
    let block_size = block_size
        .unwrap_or_else(|| legume_numeric::matrix::utils::default_block_size(data_vec.num_rows()));
    let jobs = create_jobs(data_vec.num_columns(), 0, Some(block_size));
    let njobs = jobs.len() as u64;
    let total = jobs
        .par_iter()
        .progress_with(new_progress_bar(njobs))
        .try_fold(
            || Acc::zeros(&inp, n_batches),
            |mut acc, &(lb, ub)| -> anyhow::Result<Acc> {
                let csc = data_vec.read_columns_csc(lb..ub)?;
                let batch_ids = data_vec.get_batch_membership(lb..ub);
                let theta = inp.theta_nk.rows(lb, ub - lb);
                accumulate(&mut acc, &inp, &csc, theta, &batch_ids);
                Ok(acc)
            },
        )
        .try_reduce(|| Acc::zeros(&inp, n_batches), |a, b| Ok(a.merge(b)))?;
    let delta = solve_delta_from_sums(&total.finish(&inp));
    info!(
        "Held-out δ: {} genes × {n_batches} batches, mean {:.3}",
        delta.nrows(),
        delta.iter().sum::<f32>() / delta.len() as f32,
    );
    Ok(Some(delta))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::topic::eval::{build_gene_remap_with, QueryNameOpts};

    /// The formula as it stood before this module: scatter the block dense
    /// on the training axis, then sum `w·x` and `w·μ` over every gene.
    fn reference(
        csc: &CscMatrix<f32>,
        theta_nk: &Mat,
        remap: Option<&GeneRemap>,
        beta_dk: &Mat,
        phi: Option<&[f32]>,
        batch_ids: &[usize],
        n_batches: usize,
    ) -> Mat {
        let d_train = beta_dk.nrows();
        let mut x = Mat::zeros(d_train, csc.ncols());
        for (j, col) in csc.col_iter().enumerate() {
            for (&row, &v) in col.row_indices().iter().zip(col.values()) {
                match remap {
                    Some(r) => {
                        if let Some(t) = r.new_to_train[row] {
                            x[(t, j)] += v;
                        }
                    }
                    None => x[(row, j)] = v,
                }
            }
        }
        let pred_dn = beta_dk.map(f32::exp) * theta_nk.transpose();
        let mut sums = DeltaSums {
            pb_obs: Mat::zeros(d_train, n_batches),
            pb_pred: Mat::zeros(d_train, n_batches),
        };
        for j in 0..csc.ncols() {
            let b = batch_ids[j];
            let lib: f32 = csc.col(j).values().iter().sum();
            for d in 0..d_train {
                let mu = lib * pred_dn[(d, j)];
                let w = nb_fisher_weight(phi, mu, d);
                sums.pb_obs[(d, b)] += w * x[(d, j)];
                sums.pb_pred[(d, b)] += w * mu;
            }
        }
        solve_delta_from_sums(&sums)
    }

    /// δ by this module, the cells split into two blocks.
    fn by_blocks(
        csc: &CscMatrix<f32>,
        theta_nk: &Mat,
        remap: Option<&GeneRemap>,
        beta_dk: &Mat,
        phi: Option<&[f32]>,
        batch_ids: &[usize],
        n_batches: usize,
    ) -> Mat {
        let exp_beta_dk = beta_dk.map(f32::exp);
        let inp = DeltaInputs {
            theta_nk,
            gene_remap: remap,
            exp_beta_dk: &exp_beta_dk,
            phi,
        };
        let n = csc.ncols();
        let mut total = Acc::zeros(&inp, n_batches);
        for (lb, ub) in [(0, n / 2), (n / 2, n)] {
            let mut acc = Acc::zeros(&inp, n_batches);
            let block = column_block(csc, lb, ub);
            accumulate(
                &mut acc,
                &inp,
                &block,
                theta_nk.rows(lb, ub - lb),
                &batch_ids[lb..ub],
            );
            total = total.merge(acc);
        }
        solve_delta_from_sums(&total.finish(&inp))
    }

    fn column_block(csc: &CscMatrix<f32>, lb: usize, ub: usize) -> CscMatrix<f32> {
        let mut coo = nalgebra_sparse::CooMatrix::new(csc.nrows(), ub - lb);
        for (j, col) in csc.col_iter().enumerate().take(ub).skip(lb) {
            for (&r, &v) in col.row_indices().iter().zip(col.values()) {
                coo.push(r, j - lb, v);
            }
        }
        CscMatrix::from(&coo)
    }

    struct Example {
        csc: CscMatrix<f32>,
        theta: Mat,
        remap: GeneRemap,
        beta: Mat,
        batches: Vec<usize>,
    }

    /// Five query rows (one unmatched, two on one training gene) × six
    /// cells in two batches, three topics over four training genes.
    fn example() -> Example {
        let (rows, cells) = (5, 6);
        let mut coo = nalgebra_sparse::CooMatrix::new(rows, cells);
        for j in 0..cells {
            for r in 0..rows {
                let v = ((3 * j + 5 * r) % 7) as f32;
                if v > 1.0 {
                    coo.push(r, j, v - 1.0);
                }
            }
        }
        let theta = Mat::from_fn(cells, 3, |j, k| 1.0 + ((j + 2 * k) % 4) as f32);
        let theta = Mat::from_rows(
            &theta
                .row_iter()
                .map(|r| r.unscale(r.sum()))
                .collect::<Vec<_>>(),
        );
        let beta = Mat::from_fn(4, 3, |d, k| -1.0 - 0.3 * ((d * 3 + k) % 5) as f32);
        let query: Vec<Box<str>> = ["GENE1", "GENE9", "GENE2", "GENE3", "GENE3"]
            .map(Box::from)
            .to_vec();
        let train: Vec<Box<str>> = ["GENE0", "GENE1", "GENE2", "GENE3"].map(Box::from).to_vec();
        let remap = build_gene_remap_with(&train, &query, &QueryNameOpts::default());
        let batches = (0..cells).map(|j| j % 2).collect();
        Example {
            csc: CscMatrix::from(&coo),
            theta,
            remap,
            beta,
            batches,
        }
    }

    #[test]
    fn delta_matches_the_dense_formula() {
        let Example {
            csc,
            theta,
            remap,
            beta,
            batches,
        } = example();
        let phi = [0.5_f32, 2.0, 7.0, 0.1];
        assert_eq!(remap.new_to_train[4], Some(3));
        for phi in [None, Some(&phi[..])] {
            let want = reference(&csc, &theta, Some(&remap), &beta, phi, &batches, 2);
            let got = by_blocks(&csc, &theta, Some(&remap), &beta, phi, &batches, 2);
            let gap = (&want - &got).abs().max();
            assert!(gap < 1e-4, "phi {phi:?}: gap {gap}\n{want}\n{got}");
            assert!(want.iter().any(|&v| (v - 1.0).abs() > 0.05));
        }
        // No remap: the query rows are the training rows.
        let csc4 = {
            let mut coo = nalgebra_sparse::CooMatrix::new(4, csc.ncols());
            for (j, col) in csc.col_iter().enumerate() {
                for (&r, &v) in col.row_indices().iter().zip(col.values()) {
                    coo.push(r % 4, j, v);
                }
            }
            CscMatrix::from(&coo)
        };
        for phi in [None, Some(&phi[..])] {
            let want = reference(&csc4, &theta, None, &beta, phi, &batches, 2);
            let got = by_blocks(&csc4, &theta, None, &beta, phi, &batches, 2);
            assert!((&want - &got).abs().max() < 1e-4, "phi {phi:?}");
        }
    }
}
