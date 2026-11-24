//! `senna probe` — read-only drift probe for a trained masked model.
//!
//! Scores query cells' per-cell predictive fit under a trained masked model
//! (`masked-topic` / `masked-vae` / `masked-sbp`), calibrates a null from an
//! in-distribution `--calibration` backend, flags query cells whose fit falls
//! below the null tail, and emits a batch-level **covered vs novel** verdict.
//!
//! The default verdict rests on a **goodness-of-fit / reconstruction residual**: the
//! predictive log-likelihood of each cell under the *frozen* model — the potential
//! outcome `Y(0)` ("does the current model explain this cell"). That is a novelty
//! **proxy**. Scoring reconstruction error against an in-distribution null to call
//! unseen cells is the CAMLU strategy (Li et al. 2022); the query-to-reference
//! setting is that of scArches (Lotfollahi et al. 2021) and the open-world,
//! uncertainty-aware scPoli (De Donno et al. 2023).
//!
//! **`probe` never writes a model and never trains one.**
//!
//! # References
//! - Li et al. (2022) *A machine learning-based method for automatically identifying
//!   novel cells in annotating single-cell RNA-seq data* (CAMLU). Bioinformatics 38:4885.
//! - Lotfollahi et al. (2021) *Mapping single-cell data to reference atlases by transfer
//!   learning* (scArches). Nat. Biotechnol. 39:1436.
//! - De Donno et al. (2023) *Population-level integration of single-cell datasets enables
//!   multi-scale analysis across samples* (scPoli). Nat. Methods 20:1683.

use crate::predict::MaskedScored;
use crate::topic::masked_artifact::MaskedModel;
use log::info;
use senna::embed_common::*;
use std::f64::consts::SQRT_2;

#[derive(Args, Debug)]
pub struct ProbeArgs {
    #[arg(
        required = true,
        value_delimiter = ',',
        help = "Query data files to probe (.zarr or .h5)"
    )]
    data_files: Vec<Box<str>>,

    #[arg(
        long,
        required = true,
        help = "Trained masked model prefix (output of `senna masked-topic/-vae/-sbp` -o)"
    )]
    model: Box<str>,

    #[arg(
        long,
        required = true,
        help = "In-distribution calibration backend that defines the null",
        long_help = "A backend of cells the model already explains.\n\
                     Held-out training-distribution cells are the usual choice.\n\
                     Its per-cell fit distribution sets the null;\n\
                     the query is flagged relative to its lower tail."
    )]
    calibration: Box<str>,

    #[arg(short, long, required = true, help = "Output file prefix")]
    out: Box<str>,

    #[arg(
        long,
        default_value_t = 0.05,
        help = "Null tail probability = per-cell false-positive rate"
    )]
    alpha: f64,

    #[arg(long, default_value_t = 500, help = "Evaluation minibatch size")]
    minibatch_size: usize,

    #[arg(
        long,
        help = "Load all columns into memory before scoring",
        hide = true
    )]
    preload_data: bool,

    #[arg(
        long,
        default_value_t = ComputeDevice::Cpu,
        value_enum,
        help = "Compute device",
        long_help = "Compute device. `cuda` / `metal` require the matching cargo feature.\n\
                     \n\
                     Matters most for a bge or simba model, which has no encoder: probe fits every\n\
                     cell by Poisson SGD, and it does so TWICE — once for the calibration\n\
                     arm and once for the query."
    )]
    device: ComputeDevice,

    #[arg(long, default_value_t = 0, help = "Device ordinal (for cuda/metal)")]
    device_no: usize,
}

impl ProbeArgs {
    /// The compute device this run asks for, resolved once.
    fn resolve_device(&self) -> anyhow::Result<legume_numeric::candle::candle_core::Device> {
        self.device.to_device(self.device_no)
    }
}

/// Standard normal upper tail `P(Z > z)` via the Abramowitz-Stegun erf.
fn norm_sf(z: f64) -> f64 {
    fn erf(x: f64) -> f64 {
        let sign = if x < 0.0 { -1.0 } else { 1.0 };
        let x = x.abs();
        let t = 1.0 / (1.0 + 0.327_591_1 * x);
        let y = 1.0
            - (((((1.061_405_429 * t - 1.453_152_027) * t) + 1.421_413_741) * t - 0.284_496_736)
                * t
                + 0.254_829_592)
                * t
                * (-x * x).exp();
        sign * y
    }
    0.5 * (1.0 - erf(z / SQRT_2))
}

/// Lower-tail quantile of a slice (nearest-rank).
fn quantile(xs: &[f32], q: f64) -> f32 {
    let mut v = xs.to_vec();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let idx = ((q * (v.len().max(1) - 1) as f64).round() as usize).min(v.len().saturating_sub(1));
    v.get(idx).copied().unwrap_or(f32::NEG_INFINITY)
}

/// Per-cell fit = predictive log-likelihood / count (depth-invariant).
fn per_cell_fit(llik: &[f32], total: &[f32]) -> Vec<f32> {
    llik.iter()
        .zip(total)
        .map(|(&l, &t)| if t > 0.0 { l / t } else { 0.0 })
        .collect()
}

/// Everything the verdict needs, once a family-specific scorer has produced it.
///
/// The verdict itself is model-agnostic — a quantile of the calibration fits, a flag rate,
/// and a one-sided binomial test — so it lives here once rather than in each branch. Only
/// the *scoring* differs by family.
struct Verdict<'a> {
    args: &'a ProbeArgs,
    model_type: &'a str,
    cal_fit: Vec<f32>,
    q_fit: Vec<f32>,
    q_names: Vec<Box<str>>,
}

pub fn run_probe(args: &ProbeArgs) -> anyhow::Result<()> {
    use crate::topic::model_metadata::resolve_run_kind;
    use senna::run_manifest::RunKind;

    mkdir_parent(&args.out)?;
    // `resolve_run_kind` says what the run *is*; the match below is probe's own
    // support policy. `update` asks the same question and answers it differently
    // (it supports svd, which has no scorer here), so the two must not share a
    // pre-filtered "family" enum — only the resolution.
    //
    // Head selection *within* the masked family stays with `MaskedModel::open`,
    // which reads `model_type` anyway: `masked-topic` and `masked-sbp` share
    // `RunKind::Itopic` and are told apart only there.
    let kind = resolve_run_kind(&args.model)?;
    match kind {
        k if k.is_masked_family() => probe_masked(args),
        k if k.has_frozen_gene_table() => probe_bge(args, kind),
        RunKind::Topic | RunKind::Vae => probe_fit_only(args, kind),
        other => anyhow::bail!(
            "probe does not support a '{}' run. Supported: masked-topic / masked-sbp / \
             masked-vae, topic, vae, bge, simba.",
            other.as_str()
        ),
    }
}

/// Fit score for a `senna bge` or `senna simba` run.
fn probe_bge(args: &ProbeArgs, kind: senna::run_manifest::RunKind) -> anyhow::Result<()> {
    use crate::bge::score::BgeEmbedding;

    let model = BgeEmbedding::open(&args.model)?;
    // No coverage floor: a thin panel is exactly what probe exists to score.
    let qopts = crate::topic::eval::QueryNameOpts::default();
    let dev = args.resolve_device()?;
    let cal = model.score(
        std::slice::from_ref(&args.calibration),
        args.preload_data,
        args.minibatch_size,
        &qopts,
        &dev,
    )?;
    let query = model.score(
        &args.data_files,
        args.preload_data,
        args.minibatch_size,
        &qopts,
        &dev,
    )?;

    write_verdict(Verdict {
        args,
        model_type: kind.as_str(),
        cal_fit: per_cell_fit(&cal.llik, &cal.total),
        q_fit: per_cell_fit(&query.llik, &query.total),
        q_names: query.data_vec.column_names()?,
    })
}

fn probe_masked(args: &ProbeArgs) -> anyhow::Result<()> {
    let model = MaskedModel::open(&args.model)?;
    let dev = args.resolve_device()?;
    let scored =
        |files: &[Box<str>]| model.score(files, args.preload_data, args.minibatch_size, true, &dev);

    let cal = scored(std::slice::from_ref(&args.calibration))?;
    let query = scored(&args.data_files)?;

    write_verdict(Verdict {
        args,
        model_type: &model.metadata.model_type,
        cal_fit: per_cell_fit(&cal.llik, &cal.total),
        q_fit: per_cell_fit(&query.llik, &query.total),
        q_names: query.data_vec.column_names()?,
    })
}

/// The three fields the verdict needs, from whichever family produced them.
///
/// `MaskedScored`, `DenseScored`, `VaeScored` and `BgeFit` all carry exactly these; without the
/// trait, "score the calibration set, score the query, reduce both" was written out four times
/// and the definition of *per-cell fit* lived in four places.
trait Scored {
    fn parts(&self) -> (&[f32], &[f32], &data_beans::sparse_io_vector::SparseIoVec);
}

macro_rules! impl_scored {
    ($($t:ty),+) => { $(impl Scored for $t {
        fn parts(&self) -> (&[f32], &[f32], &data_beans::sparse_io_vector::SparseIoVec) {
            (&self.llik, &self.total, &self.data_vec)
        }
    })+ };
}
impl_scored!(
    MaskedScored,
    crate::predict::DenseScored,
    crate::predict::VaeScored,
    crate::bge::score::BgeFit
);

/// `(calibration fits, query fits, query cell names)` — what the verdict consumes.
type FitPair = (Vec<f32>, Vec<f32>, Vec<Box<str>>);

/// Score calibration then query, and reduce both to per-cell fits.
///
/// The calibration score is scoped so its backend — a whole `SparseIoVec`, the full matrix
/// under `--preload-data` — drops *before* the query one is opened. Only its two `f32` vectors
/// are needed afterwards, and `update` already takes this care explicitly.
fn cal_and_query<S: Scored>(
    args: &ProbeArgs,
    score: impl Fn(&[Box<str>]) -> anyhow::Result<S>,
) -> anyhow::Result<FitPair> {
    let cal_fit = {
        let cal = score(std::slice::from_ref(&args.calibration))?;
        let (llik, total, _) = cal.parts();
        per_cell_fit(llik, total)
    };
    let query = score(&args.data_files)?;
    let (llik, total, data_vec) = query.parts();
    Ok((cal_fit, per_cell_fit(llik, total), data_vec.column_names()?))
}

fn probe_fit_only(args: &ProbeArgs, kind: senna::run_manifest::RunKind) -> anyhow::Result<()> {
    use crate::predict::null::HeldOut;
    use crate::predict::{score_dense_backend, score_vae_backend, DenseScoreArgs, VaeScoreArgs};
    use crate::topic::eval::QueryNameOpts;
    use crate::topic::model_metadata::TopicModelMetadata;
    use crate::topic::predict_common::LatentMode;
    use legume_numeric::candle::topic_refinement::TopicRefinementConfig;
    use senna::run_manifest::RunKind;

    let metadata = TopicModelMetadata::load(&args.model)?;

    let qopts = QueryNameOpts::default();
    let dev = args.resolve_device()?;
    let (cal_fit, q_fit, q_names) = match kind {
        RunKind::Vae => cal_and_query(args, |files| {
            score_vae_backend(VaeScoreArgs {
                model: &args.model,
                data_files: files,
                batch_files: None,
                preload: args.preload_data,
                minibatch_size: args.minibatch_size,
                query_name_opts: &qopts,
                metadata: &metadata,
                need_llik: true,
                held_out: HeldOut::Ignore,
                dev: &dev,
            })
        })?,
        // `LatentMode::Encoder` mirrors the masked path (encoder-only, no per-cell refinement),
        // so `refine_config` is never read.
        //
        // `HeldOut::Ignore`: no held-out null for the encoder and no δ. Either, fitted
        // from the query, would let a genuinely novel batch explain itself away as a
        // batch effect, the aliasing failure, on exactly the query this tool exists to
        // flag.
        RunKind::Topic => cal_and_query(args, |files| {
            score_dense_backend(DenseScoreArgs {
                model: &args.model,
                data_files: files,
                batch_files: None,
                preload: args.preload_data,
                minibatch_size: args.minibatch_size,
                block_size: None,
                held_out: HeldOut::Ignore,
                query_name_opts: &qopts,
                metadata: &metadata,
                mode: LatentMode::Encoder,
                refine_config: &TopicRefinementConfig::default(),
                dev: &dev,
            })
        })?,
        // `run_probe` routes every other kind elsewhere or rejects it.
        other => unreachable!("{other} has its own path"),
    };

    write_verdict(Verdict {
        args,
        model_type: &metadata.model_type,
        cal_fit,
        q_fit,
        q_names,
    })
}

fn write_verdict(v: Verdict<'_>) -> anyhow::Result<()> {
    let Verdict {
        args,
        model_type,
        cal_fit,
        q_fit,
        q_names,
    } = v;

    let thr = quantile(&cal_fit, args.alpha);
    let n = q_fit.len();
    anyhow::ensure!(n > 0, "probe: query has no cells");
    let n_flag = q_fit.iter().filter(|&&f| f < thr).count();
    let rate = n_flag as f64 / n as f64;

    // One-sided test: are more query cells flagged than the null FPR α?
    let se = (args.alpha * (1.0 - args.alpha) / n as f64)
        .sqrt()
        .max(1e-12);
    let pval = norm_sf((rate - args.alpha) / se);
    let novel = rate > args.alpha && pval < 0.05;
    let verdict = if novel {
        "NOVEL — the model does not explain this batch"
    } else {
        "COVERED — certify"
    };

    // Per-cell fit and flag: the only place *which* cells were flagged is recorded.
    let mut tsv = String::from("cell\tfit\tflag\n");
    for (nm, &f) in q_names.iter().zip(&q_fit) {
        tsv.push_str(&format!("{nm}\t{f:.6}\t{}\n", u8::from(f < thr)));
    }
    std::fs::write(format!("{}.probe.tsv", args.out), tsv)?;

    info!(
        "probe [{}]: null p{:.0} fit ≤ {:.4}; flagged {}/{} query cells ({:.1}%), p = {:.2e}",
        model_type,
        args.alpha * 100.0,
        thr,
        n_flag,
        n,
        rate * 100.0,
        pval
    );
    info!("probe verdict: {verdict}");
    info!("Wrote {}.probe.tsv", args.out);

    // The batch's numbers, for a reader that is not the log (`senna mcp`'s
    // run_probe).
    let summary = serde_json::json!({
        "model_type": model_type,
        "verdict": if novel { "novel" } else { "covered" },
        "alpha": args.alpha,
        "threshold": thr,
        "cells": n,
        "flagged": n_flag,
        "flag_rate": rate,
        "p_value": pval,
    });
    std::fs::write(
        format!("{}.probe.json", args.out),
        serde_json::to_string_pretty(&summary)?,
    )?;
    info!("Wrote {}.probe.json", args.out);
    Ok(())
}

#[cfg(test)]
#[path = "tests/probe_verdict.rs"]
mod tests;
