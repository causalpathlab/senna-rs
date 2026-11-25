//! The per-batch rate multiplier δ a held-out score is taken under.
//!
//! The model's rate for cell `j` of batch `b` is `δ[d,b] · Σ_k θ̂[j,k] ·
//! exp(β[d,k])`, renormalised over genes. δ comes from one of two places
//! ([`ScoringDeltaMode`]):
//!
//! - **trained**: `{model}.delta.parquet`, for a held-out batch the model
//!   was trained on. Nothing is fitted on the held-out cells.
//! - **fitted** from the encoded held-out cells, for a batch it never saw:
//!
//! ```text
//! pred[d,j]   = lib[j] · Σ_k θ̂[j,k] · exp(β[d,k])
//! w[d,j]      = φ[d] / (pred[d,j] + φ[d])     (NB; w = 1 if no φ)
//! δ[d,b]      = Σ_{j∈F} w[d,j]·x[d,j] / Σ_{j∈F} w[d,j]·pred[d,j]
//! ```
//!
//! the Poisson / NB-Fisher-weighted MLE of δ given θ̂ over the fitting cells
//! `F`. With `F` the batch itself the score is in-sample: `D × B` parameters
//! fitted on the counts they are scored on. Cross-fitted, the batch's cells
//! are split into folds and a fold is scored under the δ of the other folds,
//! so no count is scored under a δ it helped fit.
//!
//! The encoder's own input null is built apart, the way training built it
//! (`predict::null`), so θ̂ does not depend on δ and one pass is all there is.

use crate::logging::new_progress_bar;
use crate::topic::eval::GeneRemap;
use crate::topic::model_metadata::TopicModelMetadata;
use data_beans::sparse_io_vector::SparseIoVec;
use indicatif::ParallelProgressIterator;
use log::info;
use nalgebra_sparse::CscMatrix;
use rayon::prelude::*;
use senna::embed_common::*;
use trained::load_trained_delta;

mod trained;

pub(crate) use trained::{
    fold_by_labels, has_trained_delta, projection_fold, trained_delta_by_batch,
    trained_delta_of_batches,
};

/// Lower / upper bounds on per-(gene, batch) δ. Stops one noisy batch from
/// blowing up a gene's held-out rate when the batch barely covers a gene the
/// model predicts high.
pub const DELTA_CLAMP_MIN: f32 = 0.01;
pub const DELTA_CLAMP_MAX: f32 = 100.0;
/// Floor on a trained δ as it is read back: the one training divided by, so
/// what adjusts held-out cells is what adjusted the training cells.
const TRAINED_DELTA_FLOOR: f32 = 1e-6;
/// Floor on the predicted denominator in the obs/pred ratio, avoiding
/// division by zero for genes that no topic produces.
pub const DELTA_PRED_EPS: f32 = 1e-10;

/// Per-batch observed and predicted sums `[D_train, B]`, NB-Fisher-weighted
/// when φ is given; otherwise unweighted (Poisson / multinomial).
pub struct DeltaSums {
    pub pb_obs: Mat,
    pub pb_pred: Mat,
}

/// Closed-form δ from an observed and a predicted sum,
/// `(obs + prior·scale) / max(pred + prior, eps)`, clamped to
/// `[DELTA_CLAMP_MIN, DELTA_CLAMP_MAX]`: `prior` pseudo-counts of prediction
/// at the ratio `scale`. With no prior it is `obs / pred`. When every batch
/// shares one law, δ stays near 1.
#[must_use]
pub fn delta_ratio(obs: f32, pred: f32, prior: f32, scale: f32) -> f32 {
    ((obs + prior * scale) / (pred + prior).max(DELTA_PRED_EPS))
        .clamp(DELTA_CLAMP_MIN, DELTA_CLAMP_MAX)
}

/// [`delta_ratio`] over per-batch sums, with no prior.
#[cfg(test)]
fn solve_delta_from_sums(sums: &DeltaSums) -> Mat {
    sums.pb_obs
        .zip_map(&sums.pb_pred, |obs, pred| delta_ratio(obs, pred, 0.0, 1.0))
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

/// How a family turns a cell's latent into its rate over the model's genes.
#[derive(Clone, Copy)]
pub(crate) enum RateModel<'a> {
    /// `exp(β)·θ`: `exp_beta_dk` is `[D_train, K]` with columns summing to 1
    /// and `θ` a row of proportions, so the rate is normalised as it is.
    Topic { exp_beta_dk: &'a Mat },
    /// `exp(b + ρ·θ)` normalised over genes (vae, bge, simba).
    Embedding { rho_dh: &'a Mat, b_feat: &'a [f32] },
}

impl RateModel<'_> {
    fn n_genes(&self) -> usize {
        match self {
            Self::Topic { exp_beta_dk } => exp_beta_dk.nrows(),
            Self::Embedding { rho_dh, .. } => rho_dh.nrows(),
        }
    }

    /// `[D_train, n]`, each cell's rate summing to 1 over genes.
    fn rate(&self, theta: nalgebra::DMatrixView<'_, f32>) -> Mat {
        match self {
            Self::Topic { exp_beta_dk } => *exp_beta_dk * theta.transpose(),
            Self::Embedding { rho_dh, b_feat } => {
                let mut rate =
                    crate::topic::predict_eval::embedding_rate(rho_dh, b_feat, &theta.into_owned());
                for mut col in rate.column_iter_mut() {
                    let z = col.sum().max(1e-12);
                    col /= z;
                }
                rate
            }
        }
    }
}

/// What every block reads.
struct DeltaInputs<'a> {
    theta_nk: &'a Mat,
    gene_remap: Option<&'a GeneRemap>,
    rate: RateModel<'a>,
    phi: Option<&'a [f32]>,
}

impl<'a> DeltaInputs<'a> {
    /// `exp(β)` when the predicted sum can be kept as `Σ lib·θ` (`[K, G]`)
    /// and turned into genes once at the end: a topic rate with no φ. Any
    /// other case sums the predicted rate gene by gene.
    fn linear(&self) -> Option<&'a Mat> {
        match (self.rate, self.phi) {
            (RateModel::Topic { exp_beta_dk }, None) => Some(exp_beta_dk),
            _ => None,
        }
    }
}

/// Running sums over blocks, one column per group of cells. `obs` is
/// `[D_train, G]`. `pred` is the weighted `[D_train, G]` predicted sum under
/// NB; with no φ every weight is 1 and it is `Σ_{j∈g} lib_j θ_j`, `[K, G]`,
/// which `exp(β)` turns into the predicted sum once at the end. `raw` is the
/// unweighted observed sum, kept apart only under NB (with no φ it is `obs`).
struct Acc {
    obs: Mat,
    pred: Mat,
    raw: Option<Mat>,
}

/// The sums δ is solved from, and the raw observed counts beside them.
struct Fitted {
    sums: DeltaSums,
    /// `None` with no φ: the observed sums are unweighted already.
    raw: Option<Mat>,
}

impl Fitted {
    fn raw(&self) -> &Mat {
        self.raw.as_ref().unwrap_or(&self.sums.pb_obs)
    }
}

impl Acc {
    fn zeros(inp: &DeltaInputs<'_>, n_groups: usize) -> Self {
        let d_train = inp.rate.n_genes();
        let pred_rows = inp.linear().map_or(d_train, Mat::ncols);
        Self {
            obs: Mat::zeros(d_train, n_groups),
            pred: Mat::zeros(pred_rows, n_groups),
            raw: inp.phi.map(|_| Mat::zeros(d_train, n_groups)),
        }
    }

    fn merge(mut self, other: Self) -> Self {
        self.obs += &other.obs;
        self.pred += &other.pred;
        if let (Some(a), Some(b)) = (self.raw.as_mut(), other.raw.as_ref()) {
            *a += b;
        }
        self
    }

    fn finish(self, inp: &DeltaInputs<'_>) -> Fitted {
        let pb_pred = match inp.linear() {
            Some(exp_beta_dk) => exp_beta_dk * &self.pred,
            None => self.pred,
        };
        Fitted {
            sums: DeltaSums {
                pb_obs: self.obs,
                pb_pred,
            },
            raw: self.raw,
        }
    }
}

/// Add the cells of one block, `csc` (query rows × cells) with `theta`
/// their rows of `θ̂` and `group_ids` their groups, to `acc`; a cell with
/// no group is skipped. The observed sums run over the nonzeros only.
fn accumulate(
    acc: &mut Acc,
    inp: &DeltaInputs<'_>,
    csc: &CscMatrix<f32>,
    theta: nalgebra::DMatrixView<'_, f32>,
    group_ids: &[Option<usize>],
) {
    let d_train = inp.rate.n_genes();
    let to_train = |row: usize| match inp.gene_remap {
        Some(r) => r.new_to_train[row],
        None => Some(row),
    };
    let pred_dn = inp.linear().is_none().then(|| inp.rate.rate(theta));
    for (j, (col, &b)) in csc.col_iter().zip(group_ids).enumerate() {
        let Some(b) = b else { continue };
        // The cell's counts on the model's genes: the rate sums to 1 over
        // those, so this is the total it spreads. The whole library would
        // scale each cell's prediction by its share of unmapped counts, and
        // where that share differs by cell state δ would pick it up gene by
        // gene.
        let lib_j: f32 = col
            .row_indices()
            .iter()
            .zip(col.values())
            .filter(|(&row, _)| to_train(row).is_some())
            .map(|(_, &x)| x)
            .sum();
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
                        if let Some(raw) = acc.raw.as_mut() {
                            raw[(d, b)] += x;
                        }
                    }
                }
            }
        }
    }
}

/// Where the δ a held-out score is taken under comes from.
#[derive(clap::ValueEnum, Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ScoringDeltaMode {
    /// The trained δ for a batch the model was trained on, cross-fitted δ
    /// for any other; none at all for a model trained without batches.
    #[default]
    Auto,
    /// The trained δ only: every held-out batch must be a training batch.
    Train,
    /// δ for each fold of cells, fitted on the batch's other folds.
    Crossfit,
    /// No δ: the batch-free rate.
    None,
    /// δ fitted on the very cells it scores. In-sample, so optimistic.
    Heldout,
}

/// How to get the scoring δ.
#[derive(Clone, Copy, Debug)]
pub(crate) struct ScoringSpec {
    pub mode: ScoringDeltaMode,
    /// Folds per batch for the cross-fit (at least 2).
    pub folds: usize,
}

/// Default folds per batch of the cross-fitted δ.
pub(crate) const DEFAULT_SCORING_FOLDS: usize = 5;

/// Pseudo-counts of the cross-fitted δ, `(obs + a·r) / (pred + a)` with `r`
/// the fitting cells' overall observed-to-predicted ratio: δ is shrunk to
/// the batch's own scale, so a gene the fitting folds never saw in a batch
/// is not priced at the clamp on the fold that did see it. `r` is 1 up to
/// the NB weights, since both sums run over the model's genes.
const CROSSFIT_PRIOR: f32 = 1.0;

/// How one held-out batch gets its δ.
#[derive(Clone, Copy, PartialEq, Eq)]
enum BatchPlan {
    /// Column of the trained δ.
    Trained(usize),
    /// Fitted from the held-out cells (other folds, or all of them).
    Fitted,
    /// δ = 1.
    Unit,
}

/// Which columns each batch's groups take: `width` of them from `offset`.
/// A cross-fitted batch has one per fold, every other batch one.
struct Layout {
    offset: Vec<usize>,
    width: Vec<usize>,
}

impl Layout {
    fn new(plans: &[BatchPlan], folds: usize) -> Self {
        let width: Vec<usize> = plans
            .iter()
            .map(|p| if *p == BatchPlan::Fitted { folds } else { 1 })
            .collect();
        let offset = width
            .iter()
            .scan(0, |at, w| {
                let here = *at;
                *at += w;
                Some(here)
            })
            .collect();
        Self { offset, width }
    }

    fn n_groups(&self) -> usize {
        self.width.iter().sum()
    }

    fn columns(&self, batch: usize) -> std::ops::Range<usize> {
        self.offset[batch]..self.offset[batch] + self.width[batch]
    }
}

/// The δ a held-out score is taken under: one column per group of cells,
/// a group being a batch, or a (batch, fold) pair when δ is cross-fitted.
pub(crate) struct ScoringDelta {
    /// `[D_train, G]`.
    delta: Mat,
    /// Each cell's group.
    group: Vec<u32>,
    /// `[D_train, G]` raw observed counts of the cells each group's δ was
    /// fitted on; a zero column where δ was not fitted from held-out cells.
    fit_obs: Mat,
    /// Whether each group's δ was fitted from held-out cells.
    fitted: Vec<bool>,
    /// Each cell's batch.
    batch: Vec<u32>,
    /// `[D_train, B]`, one δ per batch whatever the folds: the trained one,
    /// or the whole batch's fit. Not for scoring.
    batch_delta: Mat,
    label: &'static str,
}

impl ScoringDelta {
    /// The group of `cell`.
    #[must_use]
    pub fn group(&self, cell: usize) -> usize {
        self.group[cell] as usize
    }

    /// Scale each cell's rate in `rate_dn` (`[D_train, n]`, the cells from
    /// `lb`) by its δ and renormalise it over genes.
    pub fn apply(&self, rate_dn: &mut Mat, lb: usize) {
        for (j, mut col) in rate_dn.column_iter_mut().enumerate() {
            col.component_mul_assign(&self.delta.column(self.group(lb + j)));
            let z = col.sum().max(1e-12);
            col /= z;
        }
    }

    /// The δ of `cell`'s batch at `gene`, to regress out of its counts: one
    /// δ per batch, so cells of a batch are not divided by different folds'
    /// estimates. For a fitted batch it is fitted on the batch's own cells.
    #[must_use]
    pub fn of_batch(&self, gene: usize, cell: usize) -> f32 {
        self.batch_delta[(gene, self.batch[cell] as usize)]
    }

    /// How the score's δ was got, for the record of a score.
    #[must_use]
    pub fn label(&self) -> &'static str {
        self.label
    }

    #[must_use]
    pub fn mean(&self) -> f32 {
        self.delta.mean()
    }

    /// The null a group is scored against, given the same batch term the
    /// model got. `null_comp` is the pooled null over the training genes,
    /// `scored` the genes it is normalised over.
    ///
    /// Where δ is the trained one the null is `δ ∘ null_comp`, what a
    /// predictor with no cell-specific information says in that batch, when
    /// `null_comp` is `batch_free` (the training marginal). A null read off
    /// the held-out cells already carries their batches and is left as it is.
    /// Where δ was fitted from held-out cells, the null is the gene
    /// composition of those same fitting cells, shrunk to `null_comp` by one
    /// pseudo-count per scored gene: `D` free parameters per group from the
    /// same cells, as the model's δ has. Unnormalised.
    #[must_use]
    pub fn null_composition(
        &self,
        group: usize,
        null_comp: &[f32],
        scored: &[usize],
        batch_free: bool,
    ) -> Vec<f32> {
        if !self.fitted[group] {
            if !batch_free {
                return null_comp.to_vec();
            }
            let delta = self.delta.column(group);
            return null_comp
                .iter()
                .zip(delta.iter())
                .map(|(n, d)| n * d)
                .collect();
        }
        let kappa = scored.len() as f32;
        let mut out = vec![0f32; null_comp.len()];
        for &g in scored {
            out[g] = self.fit_obs[(g, group)] + kappa * null_comp[g];
        }
        out
    }
}

/// What [`scoring_delta`] reads.
pub(crate) struct ScoringDeltaArgs<'a> {
    pub spec: ScoringSpec,
    pub model: &'a str,
    /// The model's gene axis, which the rate and δ live on.
    pub training_genes: &'a [Box<str>],
    pub data_vec: &'a SparseIoVec,
    /// The cells' latent `[N, K]`.
    pub theta_nk: &'a Mat,
    /// Query rows onto the training axis, WITHOUT any ablation: δ is a rate
    /// multiplier on the scored counts, so it reads the genes the encoder
    /// was not shown as well.
    pub gene_remap: Option<&'a GeneRemap>,
    /// The latent's rate over genes; `theta_nk` is whatever latent it reads
    /// (proportions for a topic rate, the embedding for the others).
    pub rate: RateModel<'a>,
    pub phi: Option<&'a [f32]>,
    /// Cells per block (`None` = the default for the row count); this pass
    /// keeps no block statistics, so it need not match the encoder's.
    pub block_size: Option<usize>,
}

/// One plan per held-out batch, or `None` when the score takes no δ.
fn plan_batches(
    a: &ScoringDeltaArgs<'_>,
    trained: Option<&(Vec<Box<str>>, Mat)>,
) -> anyhow::Result<Option<Vec<BatchPlan>>> {
    let n_batches = a.data_vec.num_batches();
    if n_batches == 0 {
        return Ok(None);
    }
    let names = a.data_vec.batch_names().unwrap_or_default();
    let trained_col = |b: usize| -> Option<usize> {
        let name = names.get(b)?;
        trained?.0.iter().position(|t| t == name)
    };
    let plans: Vec<BatchPlan> = match a.spec.mode {
        ScoringDeltaMode::None => return Ok(None),
        ScoringDeltaMode::Train => {
            anyhow::ensure!(
                trained.is_some(),
                "--scoring-delta train: {} was trained without batches, so it has no δ",
                a.model
            );
            let missing: Vec<&str> = (0..n_batches)
                .filter(|&b| trained_col(b).is_none())
                .map(|b| names.get(b).map_or("?", |n| &**n))
                .collect();
            anyhow::ensure!(
                missing.is_empty(),
                "--scoring-delta train: held-out batches the model was not trained on: {}. \
                 Use `auto` or `crossfit` for those",
                missing.join(", ")
            );
            (0..n_batches)
                .map(|b| BatchPlan::Trained(trained_col(b).expect("checked")))
                .collect()
        }
        ScoringDeltaMode::Auto => {
            // A model trained without batches has no batch term to score
            // under, however many batches the query brings. And one batch
            // alone has nothing to contrast: a δ fitted on it would be the
            // model's gene-level misfit, not a batch effect.
            let fit_rest = trained.is_some() && n_batches > 1;
            let plans: Vec<BatchPlan> = (0..n_batches)
                .map(|b| match trained_col(b) {
                    Some(c) => BatchPlan::Trained(c),
                    None if fit_rest => BatchPlan::Fitted,
                    None => BatchPlan::Unit,
                })
                .collect();
            if plans.iter().all(|p| *p == BatchPlan::Unit) {
                info!(
                    "No scoring δ: {}",
                    if trained.is_none() {
                        "the model was trained without batches"
                    } else {
                        "one held-out batch the model was not trained on has nothing to contrast"
                    }
                );
            }
            plans
        }
        ScoringDeltaMode::Crossfit | ScoringDeltaMode::Heldout => {
            if n_batches <= 1 {
                info!("One held-out batch: nothing to contrast, so no scoring δ");
                return Ok(None);
            }
            vec![BatchPlan::Fitted; n_batches]
        }
    };
    Ok(plans.iter().any(|p| *p != BatchPlan::Unit).then_some(plans))
}

/// The batches whose δ is fitted from held-out cells.
fn fitted_batches(plans: &[BatchPlan]) -> impl Iterator<Item = usize> + '_ {
    plans
        .iter()
        .enumerate()
        .filter(|(_, p)| **p == BatchPlan::Fitted)
        .map(|(b, _)| b)
}

/// Each cell's group from its batch: a cross-fitted batch deals its cells
/// to its folds in rounds, so every fold of a batch of two or more cells has
/// other folds to be fitted on, whatever the block size or thread count.
/// Each round starts at its own fold, so cells stored in a repeating
/// pattern do not land one kind to a fold.
fn assign_groups(batches: &[usize], layout: &Layout) -> Vec<u32> {
    let mut dealt = vec![0usize; layout.width.len()];
    batches
        .iter()
        .map(|&b| {
            let width = layout.width[b];
            let round = legume_numeric::matrix::rand_util::mix_seed(0, (dealt[b] / width) as u64);
            let fold = (dealt[b] + round as usize % width) % width;
            dealt[b] += 1;
            (layout.offset[b] + fold) as u32
        })
        .collect()
}

/// Where [`fill_fitted`] writes.
struct FittedOut<'a> {
    /// `[D_train, G]`, a group's δ to score under.
    delta: &'a mut Mat,
    /// `[D_train, G]`, the raw counts of the cells each δ was fitted on.
    fit_obs: &'a mut Mat,
    /// `[D_train, B]`, the whole batch's δ.
    batch_delta: &'a mut Mat,
}

/// Solve δ for the groups of every fitted batch from the per-group sums in
/// `total`. Cross-fitted, a group reads its batch's other folds; otherwise
/// the whole batch.
fn fill_fitted(
    total: &Fitted,
    plans: &[BatchPlan],
    layout: &Layout,
    crossfit: bool,
    out: &mut FittedOut<'_>,
) {
    let prior = if crossfit { CROSSFIT_PRIOR } else { 0.0 };
    let d_train = out.delta.nrows();
    for b in fitted_batches(plans) {
        let cols = layout.columns(b);
        let sum = |m: &Mat| -> DVec { m.columns_range(cols.clone()).column_sum() };
        let (obs_b, pred_b, raw_b) = (
            sum(&total.sums.pb_obs),
            sum(&total.sums.pb_pred),
            sum(total.raw()),
        );
        for d in 0..d_train {
            out.batch_delta[(d, b)] = delta_ratio(obs_b[d], pred_b[d], 0.0, 1.0);
        }
        for g in cols {
            let leave = |all: &DVec, m: &Mat| -> DVec {
                if crossfit {
                    all - m.column(g)
                } else {
                    all.clone()
                }
            };
            let obs = leave(&obs_b, &total.sums.pb_obs);
            let pred = leave(&pred_b, &total.sums.pb_pred);
            let (obs_all, pred_all) = (obs.sum(), pred.sum());
            let scale = if pred_all > 0.0 {
                obs_all / pred_all
            } else {
                1.0
            };
            for d in 0..d_train {
                out.delta[(d, g)] = delta_ratio(obs[d], pred[d], prior, scale);
            }
            out.fit_obs.set_column(g, &leave(&raw_b, total.raw()));
        }
    }
}

/// The δ the held-out cells are scored under, or `None` when they take none.
///
/// A fitted δ is the Poisson / NB-Fisher-weighted MLE given `θ̂`,
/// `Σ w·x / Σ w·pred` over a batch's cells. Fitted on the cells it then
/// scores (`heldout`) it has `D × B` parameters fitted in-sample, and the
/// score is optimistic. Cross-fitted, each fold's δ comes from the batch's
/// other folds, so no count is scored under a δ it helped fit.
///
/// A fitted δ is the batch's whole observed-to-predicted ratio per gene: it
/// is not pinned against the other batches as the trained δ is, so it also
/// takes up whatever the dictionary misses in that batch.
pub(crate) fn scoring_delta(a: &ScoringDeltaArgs<'_>) -> anyhow::Result<Option<ScoringDelta>> {
    let wants_trained = matches!(
        a.spec.mode,
        ScoringDeltaMode::Auto | ScoringDeltaMode::Train
    );
    let trained = if wants_trained {
        load_trained_delta(a.model, a.training_genes)?
    } else {
        None
    };
    let Some(plans) = plan_batches(a, trained.as_ref())? else {
        return Ok(None);
    };
    let n_batches = plans.len();
    let n_cells = a.data_vec.num_columns();
    let n_fitted = fitted_batches(&plans).count();
    let n_trained = plans
        .iter()
        .filter(|p| matches!(p, BatchPlan::Trained(_)))
        .count();
    let (any_fitted, any_trained) = (n_fitted > 0, n_trained > 0);
    let crossfit = any_fitted && a.spec.mode != ScoringDeltaMode::Heldout;
    let folds = if crossfit { a.spec.folds } else { 1 };
    let layout = Layout::new(&plans, folds);
    let n_groups = layout.n_groups();
    let batches = a.data_vec.get_batch_membership(0..n_cells);
    let group = assign_groups(&batches, &layout);

    let d_train = a.rate.n_genes();
    let mut delta = Mat::from_element(d_train, n_groups, 1.0);
    let mut batch_delta = Mat::from_element(d_train, n_batches, 1.0);
    let mut fit_obs = Mat::zeros(d_train, n_groups);
    let mut fitted = vec![false; n_groups];

    if any_fitted {
        let inp = DeltaInputs {
            theta_nk: a.theta_nk,
            gene_remap: a.gene_remap,
            rate: a.rate,
            phi: a.phi,
        };
        let block_size = a.block_size.unwrap_or_else(|| {
            legume_numeric::matrix::utils::default_block_size(a.data_vec.num_rows())
        });
        let jobs = create_jobs(n_cells, 0, Some(block_size));
        let njobs = jobs.len() as u64;
        let total = jobs
            .par_iter()
            .progress_with(new_progress_bar(njobs))
            .try_fold(
                || Acc::zeros(&inp, n_groups),
                |mut acc, &(lb, ub)| -> anyhow::Result<Acc> {
                    // Only the cells of a fitted batch are summed.
                    let groups: Vec<Option<usize>> = (lb..ub)
                        .map(|j| {
                            (plans[batches[j]] == BatchPlan::Fitted).then_some(group[j] as usize)
                        })
                        .collect();
                    if groups.iter().all(Option::is_none) {
                        return Ok(acc);
                    }
                    let csc = a.data_vec.read_columns_csc(lb..ub)?;
                    let theta = inp.theta_nk.rows(lb, ub - lb);
                    accumulate(&mut acc, &inp, &csc, theta, &groups);
                    Ok(acc)
                },
            )
            .try_reduce(|| Acc::zeros(&inp, n_groups), |x, y| Ok(x.merge(y)))?
            .finish(&inp);
        fill_fitted(
            &total,
            &plans,
            &layout,
            crossfit,
            &mut FittedOut {
                delta: &mut delta,
                fit_obs: &mut fit_obs,
                batch_delta: &mut batch_delta,
            },
        );
        for b in fitted_batches(&plans) {
            fitted[layout.columns(b)].fill(true);
        }
        if crossfit {
            let alone = fitted_batches(&plans)
                .filter(|&b| a.data_vec.batch_to_columns(b).map_or(0, Vec::len) < 2)
                .count();
            if alone > 0 {
                log::warn!(
                    "{alone} held-out batches of one cell have no other cell to fit δ on: \
                     they are scored with δ = 1 (a δ from the cell itself would score it \
                     against its own counts). --residual-include-delta still divides by \
                     the batch's own δ"
                );
            }
        }
    }
    if let Some((_, trained_db)) = trained.as_ref() {
        for (b, plan) in plans.iter().enumerate() {
            if let BatchPlan::Trained(c) = *plan {
                // Clamped for the score, as a fitted δ is; what adjusts the
                // cells before they are placed reads the trained values.
                let col = trained_db
                    .column(c)
                    .map(|v| v.clamp(DELTA_CLAMP_MIN, DELTA_CLAMP_MAX));
                batch_delta.set_column(b, &col);
                for g in layout.columns(b) {
                    delta.set_column(g, &col);
                }
            }
        }
    }

    let label = match (any_trained, any_fitted, crossfit) {
        (_, false, _) => "trained",
        (false, true, true) => "cross-fitted",
        (true, true, true) => "trained and cross-fitted, by batch",
        (false, true, false) => "in-sample",
        (true, true, false) => "trained and in-sample, by batch",
    };
    let how = if !any_fitted {
        "none fitted on the held-out cells".to_string()
    } else if crossfit {
        format!("cross-fitted over {folds} folds")
    } else {
        "fitted IN-SAMPLE on the cells it scores: the score is optimistic".to_string()
    };
    let out = ScoringDelta {
        delta,
        group,
        fit_obs,
        fitted,
        batch: batches.iter().map(|&b| b as u32).collect(),
        batch_delta,
        label,
    };
    info!(
        "Scoring δ: {d_train} genes × {n_batches} batches ({n_trained} trained, {n_fitted} \
         fitted, {} unit; {how}), mean {:.3}",
        n_batches - n_trained - n_fitted,
        out.mean(),
    );
    Ok(Some(out))
}

#[cfg(test)]
#[path = "delta_tests.rs"]
mod tests;
