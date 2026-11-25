//! The shared tail of every topic-family predict: agreement score and the output files.

use crate::topic::eval::GeneRemap;
use crate::topic::predict_eval::{
    evaluate_predictions, resolve_eval_genes, EvalArgs, EvalOutcome, Reconstruction,
};
use senna::embed_common::*;

use super::{build_remap, delta, eval_gene_source, read_query, scoring_remap, PredictArgs};
use crate::logging::new_progress_bar;
use data_beans::sparse_io_vector::SparseIoVec;
use indicatif::ParallelProgressIterator;
use log::info;
use rayon::prelude::*;
use senna::sparse_output::{write_backend, Entries};

/// Shared tail of both predict paths: per-cell latent + predictive scores,
/// then (optionally) the residual-expression backend. Both the dense and
/// indexed drivers have the same artifacts available, so they funnel through
/// here rather than duplicating the two write calls.
pub(super) struct FinalizePredict<'a> {
    pub(super) args: &'a PredictArgs,
    pub(super) training_genes: &'a [Box<str>],
    pub(super) data_vec: &'a SparseIoVec,
    pub(super) z_nk: &'a Mat,
    pub(super) llik: &'a [f32],
    pub(super) total: &'a [f32],
    pub(super) beta_dk: &'a Mat,
    pub(super) delta_db: Option<&'a delta::ScoringDelta>,
    pub(super) gene_remap: Option<&'a GeneRemap>,
}

pub(super) fn finalize_predict(f: FinalizePredict<'_>) -> anyhow::Result<()> {
    let agreement = topic_agreement(&f)?;
    write_outputs(
        f.args,
        f.data_vec,
        f.z_nk,
        f.llik,
        f.total,
        agreement.as_ref(),
    )?;
    write_residual_backend(
        f.args,
        f.data_vec,
        f.z_nk,
        f.beta_dk,
        f.delta_db,
        f.gene_remap,
    )
}

/// Everything the agreement pass needs, independent of which family produced it.
pub(super) struct AgreementInputs<'a> {
    pub(super) args: &'a PredictArgs,
    /// The model's feature axis — what `recon` and the eval indices live on.
    pub(super) training_genes: &'a [Box<str>],
    pub(super) data_vec: &'a SparseIoVec,
    pub(super) recon: Reconstruction<'a>,
}

/// Score the reconstruction against the observed test counts.
///
/// A separate streaming pass rather than a hook inside each backend's likelihood
/// loop. It duplicates one matrix product per block — but it buys one formula in
/// one place for every family, and the test half is the small side of a split by
/// construction.
pub(super) fn evaluate_agreement(a: AgreementInputs<'_>) -> anyhow::Result<Option<EvalOutcome>> {
    // Training axis, not query axis: the pass densifies the observed counts onto
    // the model's features before scoring, so an index that came from the query's
    // row names would point at the wrong gene whenever the two axes differ.
    let (restrict, restrict_flag) = eval_gene_source(a.args);
    let eval_genes = resolve_eval_genes(restrict, a.training_genes, restrict_flag)?;
    if eval_genes.is_empty() {
        info!("No evaluation features matched; skipping agreement metrics");
        return Ok(None);
    }

    // Built here rather than taken from the caller, and always without the
    // ablation. The backend's remap has the hidden genes pointing at `None`,
    // which is right for the encoder and wrong here: densifying the *observed*
    // counts through it would read the scored genes as zero and grade every model
    // against a blank. Rebuilding costs one name-matching pass and removes the
    // chance of a caller handing over the encoder's view by mistake.
    let score_remap = scoring_remap(
        a.training_genes,
        &a.data_vec.row_names()?,
        &a.args.query_name_opts()?,
    )?;

    let delta_label = a.recon.scoring_delta().map_or("none", |d| d.label());
    let mut out = evaluate_predictions(EvalArgs {
        data_vec: a.data_vec,
        null_comp: training_marginal(a.args, a.training_genes, &eval_genes)?,
        recon: a.recon,
        gene_remap: score_remap.as_ref(),
        eval_genes,
        minibatch_size: a.args.minibatch_size,
        keep_per_gene: restrict.is_some(),
    })?;
    out.train_gene_names = a.training_genes.to_vec();
    info!(
        "Agreement (scoring δ: {delta_label}; the null takes the same batch term): {}",
        out.summary().line()
    );
    Ok(Some(out))
}

/// The training composition over the scored genes — the null every arm shares.
///
/// A count-weighted marginal over the training cells: `Σ_cells x_g`, normalised
/// over the scored genes. Two properties earn it the job. It does not depend on
/// the model, so every arm scored on the same training half and genes is measured
/// against one identical floor. And it is not fitted on the test half, so it is a
/// baseline a model could actually have matched.
///
/// Deliberately NOT `{model}.feature_mean.parquet`, which was the first thing
/// tried here. That file is a per-gene *rate the encoder divides by*, formed as a
/// Gamma posterior mean averaged over pseudobulk groups — so its prior floors
/// every gene unseen in training at the same positive value, and a third of the
/// axis ends up sharing it. As a composition it therefore spends real probability
/// mass on genes that carry almost none, which measured 0.49 nats/count weaker
/// than the count-weighted marginal on the same genes. A weak floor flatters every
/// model, which is the one direction a null must not err in.
fn training_marginal(
    args: &PredictArgs,
    training_genes: &[Box<str>],
    eval_genes: &[usize],
) -> anyhow::Result<Option<Vec<f32>>> {
    let Some(files) = args.null_from.as_deref() else {
        log::warn!(
            "no --null-from: the null falls back to the TEST half's own composition, which \
             is fitted on the data being scored and is an upper reference rather than a \
             floor. Pass the training half to make gains comparable across runs"
        );
        return Ok(None);
    };
    // Resolved once: building this reads `--ablate-features` from disk.
    // The training half is aligned onto the model's axis the same way the query
    // is, so a name-kind difference between the two files cannot silently drop
    // genes from the floor that the model is being scored on.
    let mut opts = args.query_name_opts()?;
    opts.hide = None;
    let train = read_query(files, None, args.preload_data, &opts, training_genes)?;
    let remap = build_remap(training_genes, &train.row_names()?, &opts)?;
    let comp = crate::topic::predict_eval::empirical_composition(
        &train,
        remap.as_ref(),
        eval_genes,
        training_genes.len(),
        args.minibatch_size,
    )?;
    anyhow::ensure!(
        comp.iter().any(|&v| v > 0.0),
        "--null-from carries no counts on the scored genes; the null would be empty"
    );
    info!(
        "Null: gene composition of {} training cells from {}",
        train.num_columns(),
        files.join(", ")
    );
    Ok(Some(comp))
}

/// The topic families' reconstruction inputs, in the shape the eval pass wants.
///
/// The latent is stored on the log scale and every consumer exponentiates it;
/// doing that here once keeps the two callers from disagreeing about which scale
/// they hold.
pub(super) fn topic_agreement(f: &FinalizePredict<'_>) -> anyhow::Result<Option<EvalOutcome>> {
    let theta_nk = f.z_nk.map(f32::exp);
    evaluate_agreement(AgreementInputs {
        args: f.args,
        training_genes: f.training_genes,
        data_vec: f.data_vec,
        recon: Reconstruction::Topic {
            exp_beta_dk: f.beta_dk.map(f32::exp),
            theta_nk: &theta_nk,
            delta_db: f.delta_db,
        },
    })
}

pub(super) fn write_outputs(
    args: &PredictArgs,
    data_vec: &SparseIoVec,
    z_nk: &Mat,
    llik: &[f32],
    total: &[f32],
    agreement: Option<&EvalOutcome>,
) -> anyhow::Result<()> {
    let cell_names = data_vec.column_names()?;

    z_nk.to_parquet_with_names(
        &(args.out.to_string() + ".latent.parquet"),
        (Some(&cell_names), Some("cell")),
        Some(&axis_id_names("T", z_nk.ncols())),
    )?;
    info!("Wrote {}.latent.parquet", args.out);

    // Per-cell predictive scores: [llik, total, llik_per_count]
    let n = llik.len();
    let mut pred = Mat::zeros(n, 3);
    for i in 0..n {
        pred[(i, 0)] = llik[i];
        pred[(i, 1)] = total[i];
        // NaN, not 0: a cell with no counts has no per-count score, and 0 nats
        // per count reads as a PERFECT prediction to anything that averages this
        // column. Ablation makes the case common — a cell can easily carry
        // nothing in the hidden gene set.
        pred[(i, 2)] = if total[i] > 0.0 {
            llik[i] / total[i]
        } else {
            f32::NAN
        };
    }
    let mut pred_cols: Vec<Box<str>> = vec!["llik".into(), "total".into(), "llik_per_count".into()];
    if let Some(a) = agreement {
        anyhow::ensure!(
            a.per_cell.len() == n && a.per_cell_llik.len() == n,
            "agreement pass scored {} cells, likelihood pass scored {n}",
            a.per_cell.len()
        );
        pred = pred.insert_columns(3, 5, 0.0);
        for (i, c) in a.per_cell.iter().enumerate() {
            pred[(i, 3)] = c.spearman;
            pred[(i, 4)] = c.pearson_log1p;
        }
        // The cross-family likelihood. Written per cell, not just summarised in
        // the log, because a benchmark reads this file: the `llik` column above
        // is the backend's own and is not comparable between decoders, so these
        // are the only columns two families may be ranked on.
        for (i, l) in a.per_cell_llik.iter().enumerate() {
            pred[(i, 5)] = l.count;
            let per_count = |v: f32| if l.count > 0.0 { v / l.count } else { f32::NAN };
            pred[(i, 6)] = per_count(l.model);
            pred[(i, 7)] = per_count(l.null);
        }
        pred_cols.push("spearman".into());
        pred_cols.push("pearson_log1p".into());
        pred_cols.push("eval_count".into());
        pred_cols.push("eval_llik_per_count".into());
        pred_cols.push("eval_null_llik_per_count".into());
    }
    pred.to_parquet_with_names(
        &(args.out.to_string() + ".predictive.parquet"),
        (Some(&cell_names), Some("cell")),
        Some(&pred_cols),
    )?;

    if let Some(a) = agreement.filter(|a| !a.per_gene.is_empty()) {
        // The other axis: each gene's agreement *across* cells. A model can score
        // well per cell — every cell's genes ranked about right — while getting a
        // gene's variation across cells backwards, which is the direction that
        // matters for anything downstream that compares cells.
        let mut gm = Mat::zeros(a.per_gene.len(), 3);
        let mut rows: Vec<Box<str>> = Vec::with_capacity(a.per_gene.len());
        for (i, &(g, sp, pe, mean_obs)) in a.per_gene.iter().enumerate() {
            gm[(i, 0)] = sp;
            gm[(i, 1)] = pe;
            gm[(i, 2)] = mean_obs;
            rows.push(
                a.train_gene_names
                    .get(g)
                    .cloned()
                    .unwrap_or_else(|| "?".into()),
            );
        }
        let cols: Vec<Box<str>> = vec![
            "spearman".into(),
            "pearson_log1p".into(),
            "mean_observed".into(),
        ];
        gm.to_parquet_with_names(
            &(args.out.to_string() + ".gene_agreement.parquet"),
            (Some(&rows), Some("gene")),
            Some(&cols),
        )?;
        info!("Wrote {}.gene_agreement.parquet", args.out);
    }
    info!("Wrote {}.predictive.parquet", args.out);
    Ok(())
}

/// Regress the reference reconstruction `μ` out of the held-out counts **by
/// division**, reusing legume_numeric::matrix's `adjust_by_division_inplace`, and write
/// the leftover ("residual expression") to a NEW sparse backend.
///
/// Blocks of cells run in parallel (rayon, like [`run_predict_blocks`]). Per
/// block we form the expected per-gene rate as one `nalgebra` matmul
/// `pred = exp(β) · θᵀ` (`[D_train, n_block]`) — never an `N × D` dense
/// matrix, peak intermediate is `D × minibatch`. `pred` is scattered onto the
/// held-out gene axis (via `gene_remap`) as the per-cell denominator `μ_dn`
/// (`[D_test, n_block]`), optionally weighted by the per-batch δ when
/// `--residual-include-delta` is set. Then
/// `csc.adjust_by_division_inplace(&μ_dn)` performs, per cell `j`,
///   `x_dj ← x_dj / (μ_dj · λ_j)`,  `λ_j = Σ_d x_dj / Σ_d μ_dj`
/// — the same self-normalizing division `senna svd` uses for batch
/// adjustment (`svd/fit.rs`). Absolute scale of `μ` cancels in `λ`, so `pred`
/// is used directly (no library rescale). Genes absent from the reference
/// model have `μ = 0` and are passed through unchanged. Surviving entries
/// above `--residual-threshold` (all ≥ 0) are written as triplets, mirroring
/// the `svd` backend-write idiom.
fn write_residual_backend(
    args: &PredictArgs,
    data_vec: &SparseIoVec,
    z_nk: &Mat,
    beta_dk: &Mat,
    delta_db: Option<&delta::ScoringDelta>,
    gene_remap: Option<&GeneRemap>,
) -> anyhow::Result<()> {
    let Some(path) = args.residual_out.as_deref() else {
        return Ok(());
    };

    let threshold = args.residual_threshold;
    // δ to fold into μ (None ⇒ topics-only denominator).
    let delta = args.residual_include_delta.then_some(delta_db).flatten();
    if args.residual_include_delta && delta.is_none() && data_vec.num_batches() > 1 {
        log::warn!(
            "--residual-include-delta: this run has no scoring δ (see --scoring-delta), so the \
             residual keeps its batch effects; `crossfit` or `heldout` fits one from the cells"
        );
    }

    let kk = beta_dk.ncols();
    anyhow::ensure!(
        z_nk.ncols() == kk,
        "residual: latent topics ({}) != dictionary topics ({kk})",
        z_nk.ncols(),
    );

    // exp(β) once: [D_train, K], shared read-only across blocks.
    let exp_beta_dk = beta_dk.map(f32::exp);

    let ntot = data_vec.num_columns();
    let d_test = data_vec.num_rows();

    info!(
        "Computing residual expression by division (include_delta={}, threshold={threshold}) \
         over {ntot} cells",
        delta.is_some(),
    );

    let jobs = create_jobs(ntot, 0, Some(args.minibatch_size));
    let njobs = jobs.len() as u64;
    let triplets: Vec<(u64, u64, f32)> = jobs
        .par_iter()
        .progress_with(new_progress_bar(njobs))
        .map(|&(lb, ub)| -> anyhow::Result<Vec<(u64, u64, f32)>> {
            let mut csc = data_vec.read_columns_csc(lb..ub)?;
            let n_block = csc.ncols();

            // θ for this block: exp of stored log θ → [K, n_block].
            let theta_kn = z_nk.rows(lb, n_block).map(f32::exp).transpose();
            // pred[d, j] = Σ_k exp(β_dk) θ_jk  → [D_train, n_block].
            let pred_dn = &exp_beta_dk * theta_kn;

            // Scatter pred onto the held-out gene axis as the per-cell
            // denominator μ_dn [D_test, n_block]; optionally weight by δ.
            let mut mu_dn = Mat::zeros(d_test, n_block);
            for jloc in 0..n_block {
                for &row_new in csc.col(jloc).row_indices() {
                    let Some(row_train) = (match gene_remap {
                        Some(rm) => rm.new_to_train[row_new],
                        None => Some(row_new),
                    }) else {
                        continue; // gene absent from the reference model → μ = 0
                    };
                    let mut mu = pred_dn[(row_train, jloc)];
                    if let Some(delta) = delta {
                        mu *= delta.of_batch(row_train, lb + jloc);
                    }
                    mu_dn[(row_new, jloc)] = mu;
                }
            }

            // Regress out by division (self-normalizing column scale λ = Σx/Σμ).
            csc.adjust_by_division_inplace(&mu_dn);

            Ok(csc
                .triplet_iter()
                .filter(|&(_, _, &val)| val > threshold)
                .map(|(i, j_local, &val)| (i as u64, (lb + j_local) as u64, val))
                .collect())
        })
        .collect::<anyhow::Result<Vec<_>>>()?
        .into_iter()
        .flatten()
        .collect();

    let nnz = triplets.len();
    let written = write_backend(
        path,
        Entries::Triplets(triplets),
        &data_vec.row_names()?,
        &data_vec.column_names()?,
    )?;

    info!("Wrote residual backend: {written} ({d_test} genes × {ntot} cells, {nnz} nonzeros)");
    Ok(())
}
