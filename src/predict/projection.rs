//! Projection-only families (`senna bge`, `senna simba`, `senna svd`): no
//! decoder, only a fit onto the frozen dictionary.

use crate::topic::model_metadata::load_dictionary;
use crate::topic::predict_eval::{mean_finite, resolve_eval_genes, Reconstruction};
use senna::embed_common::*;

use super::{
    build_remap, delta, embedding_scoring_delta, eval_gene_source, finalize::evaluate_agreement,
    finalize::write_outputs, finalize::AgreementInputs, read_query, EmbeddingScore, PredictArgs,
};
use log::info;
use rayon::prelude::*;

/// Score a held-out backend against a `senna bge` or `senna simba` run.
///
/// bge's whole gene-side model is `(ρ, b_feat)` — there is no checkpoint, no encoder and no
/// decoder — so this is the projection path and nothing else: fit each query cell against
/// the frozen ρ by Poisson MAP, then score it with the profile (multinomial) likelihood.
/// A simba run is the same object with `b_feat = 0` and its gene node table as ρ; the
/// multinomial over `softmax_g ⟨z, e_g⟩` is SIMBA's own gene-side conditional, so `llik`
/// here IS comparable between the two. The `--init-*` gene alignment runs on both: with
/// no gene modules (every simba run) an unseen query gene is placed at its neighbours'
/// row average, and its moment-matched bias sits at zero like every trained gene's.
/// `--decoder-only` and `--refine-steps` have nothing to act on here and are rejected
/// rather than silently ignored, so a command line that asks for refinement does not come
/// back looking like it got it. With batches, the counts of a batch the run was
/// trained on are divided by its trained per-batch gene fold before projection, and
/// the agreement score takes a δ under `--scoring-delta`.
///
/// ⚠️ Read `.predictive.parquet` from a bge model as a FLOOR on novelty, not a verdict —
/// see [`crate::bge::score::BgeEmbedding::score`] for the measurement. `--eval-mask-fraction`
/// on the training run, or `senna probe`, answer "has the model seen this biology" better
/// than a per-cell fit with H free parameters can.
pub(super) fn predict_bge(
    args: &PredictArgs,
    kind: senna::run_manifest::RunKind,
) -> anyhow::Result<()> {
    args.refuse_latent_refinement(&format!(
        "a `senna {kind}` run, whose only inference is the Poisson-MAP projection onto the frozen ρ"
    ))?;
    if args.residual_out.is_some() {
        log::warn!(
            "--residual-out is not available for a {kind} run (no count decoder to regress \
             out); ignoring it"
        );
    }

    let model = crate::bge::score::BgeEmbedding::open(&args.model)?;
    // `--scoring-delta none` is the run with no batch term anywhere: the
    // counts are projected as they are, too.
    let batches = (args.scoring_delta != delta::ScoringDeltaMode::None).then_some(
        crate::bge::score::QueryBatches {
            files: args.batch_files.as_deref(),
            model: &args.model,
        },
    );
    let qopts = args.query_name_opts()?;
    let fit = model.score_with_init(
        crate::bge::score::QueryRead {
            files: &args.data_files,
            batches,
            preload: args.preload_data,
            block: args.minibatch_size,
        },
        &qopts,
        crate::bge::score::InitOpts {
            align: (!args.no_init_genes).then_some(graph_embedding_util::transfer::AlignKnobs {
                k: args.init_neighbours,
                similarity_floor: args.init_similarity_floor,
            }),
            in_fit: args.init_genes_in_fit,
        },
        &args.resolve_device()?,
    )?;

    // ρ is held row-major `[D, H]` for the projection; the eval pass wants it as
    // a matrix, so it is reshaped once here rather than per block.
    let rho_dh = Mat::from_row_slice(model.gene_names.len(), model.h, &model.rho);
    let scoring = embedding_scoring_delta(EmbeddingScore {
        args,
        model: &args.model,
        training_genes: &model.gene_names,
        data_vec: &fit.data_vec,
        rho_dh: &rho_dh,
        b_feat: &model.b_feat,
        theta_nh: &fit.latent,
    })?;
    let agreement = evaluate_agreement(AgreementInputs {
        args,
        training_genes: &model.gene_names,
        data_vec: &fit.data_vec,
        recon: Reconstruction::Embedding {
            rho_dh,
            b_feat: &model.b_feat,
            theta_nh: &fit.latent,
            delta_db: scoring.as_ref(),
        },
    })?;

    write_outputs(
        args,
        &fit.data_vec,
        &fit.latent,
        &fit.llik,
        &fit.total,
        agreement.as_ref(),
    )?;
    crate::bge::transfer::write_init_outputs(&args.out, &model, &fit, args.emit_gene_rates)?;

    Ok(())
}

/// Score a held-out backend against a `senna svd` run.
///
/// svd writes no checkpoint, so there is no encoder to run; the query side is
/// the Nyström projection onto the frozen dictionary
/// ([`crate::svd::project::project_onto_dictionary`]), the same map `impute`
/// uses.
///
/// **What the columns mean here, because svd is not a count model.**
/// `spearman` / `pearson_log1p` come from the same
/// [`legume_numeric::matrix::agreement::agreement_from_rate`] every other family is
/// graded by, on predicted COUNTS, so they are directly comparable across
/// families — that is the axis to compare svd on. `llik` is a GAUSSIAN
/// log-likelihood in `log1p` space, which is the loss svd actually minimises,
/// at the per-cell ML σ̂; like every other family's `llik` it is the backend's
/// own and must NOT be compared against another family's. It is normalised as
/// `llik_per_gene`, not `llik_per_count`: it sums over the scored gene axis
/// and does not scale with library size, so it does not belong in a column
/// that means nats-per-count everywhere else. The multinomial
/// `eval_llik_*` columns are deliberately absent rather than filled with a
/// differently-scaled number: svd has no count likelihood to put there.
pub(super) fn predict_svd(args: &PredictArgs) -> anyhow::Result<()> {
    use legume_numeric::matrix::agreement::agreement_from_rate;

    args.refuse_latent_refinement(
        "a `senna svd` run, whose only inference is the projection onto the frozen dictionary",
    )?;
    if args.residual_out.is_some() {
        log::warn!("--residual-out is not available for an svd run (no count decoder); ignoring");
    }
    if args.null_from.is_some() {
        log::warn!(
            "--null-from has no effect on an svd run: the null it builds is a count \
             composition for the multinomial columns, which this path does not write \
             (svd is scored on the correlations and a Gaussian log1p llik)"
        );
    }

    let (training_genes, u_dk) = load_dictionary(&args.model)?;
    let column_sum_norm = crate::svd::project::column_sum_norm(&args.model);

    // Resolved once: building these reads `--ablate-features` from disk.
    let qopts = args.query_name_opts()?;
    let data_vec = read_query(
        &args.data_files,
        args.batch_files.as_deref(),
        args.preload_data,
        &qopts,
        &training_genes,
    )?;
    // Counts of a batch the run was trained on are divided by its trained δ
    // before they are projected; with `--scoring-delta none` by ones, which
    // keeps the division's rescaling that training applied.
    let fold = crate::svd::project::batch_fold(
        &args.model,
        &training_genes,
        &data_vec,
        args.scoring_delta != delta::ScoringDeltaMode::None,
        args.scoring_delta == delta::ScoringDeltaMode::Train,
    )?;
    // The projection sees the caller's query-axis rules, INCLUDING
    // `--ablate-features`: without that the latent is fitted on the genes it
    // is about to be scored on, which is a reconstruction, not a prediction.
    // Only the remap is built here — the projection itself happens inside the
    // scoring loop, off the same block read, so the data is streamed once.
    let proj_remap =
        crate::svd::project::projection_remap(&data_vec, &training_genes, &qopts, "query")?;

    // Restrict to the scored genes exactly as the other families do, so a
    // benchmark passing one --eval-features file grades every arm on the
    // same curriculum.
    let (restrict, restrict_flag) = eval_gene_source(args);
    let eval_genes = resolve_eval_genes(restrict, &training_genes, restrict_flag)?;
    anyhow::ensure!(
        !eval_genes.is_empty(),
        "no evaluation features matched the model's genes"
    );
    // Scoring reads the OBSERVED counts, hidden genes included — that is the
    // point of hiding them from the projection above.
    let mut opts = qopts;
    opts.hide = None;
    let remap = build_remap(&training_genes, &data_vec.row_names()?, &opts)?;

    let ntot = data_vec.num_columns();
    let d_train = training_genes.len();
    // The scored slice of the dictionary, transposed once: `[K, n_eval]`, so
    // a block's reconstruction is one matmul rather than a strided scalar dot
    // per (cell, gene). `Mat` is column-major, so `u_dk.row(g)` is a strided
    // view and the per-cell form was O(N x n_eval x K) of unvectorisable loads.
    let u_eval_t = Mat::from_fn(u_dk.ncols(), eval_genes.len(), |c, i| {
        u_dk[(eval_genes[i], c)]
    });

    let mut z_nk = Mat::zeros(ntot, u_dk.ncols());
    let mut per_cell: Vec<CellScore> = Vec::with_capacity(ntot);
    for (lb, ub) in create_jobs(ntot, 0, Some(args.minibatch_size)) {
        // ONE read per block: the latent and the observed counts both come off
        // it. Projecting up front and re-reading to score doubled the pass
        // over the backend.
        let csc = data_vec.read_columns_csc(lb..ub)?;
        let batches = fold.as_ref().map(|_| data_vec.get_batch_membership(lb..ub));
        let z_block = crate::svd::project::project_block(
            &csc,
            &proj_remap.new_to_train,
            &u_dk,
            column_sum_norm,
            fold.as_ref().zip(batches.as_deref()),
        )
        .transpose();
        z_nk.rows_range_mut(lb..ub).copy_from(&z_block);
        // `[n, n_eval]` for this block only; the block loop is sequential, so
        // one of these is live at a time.
        let recon_std_block = &z_block * &u_eval_t;
        let block: Vec<CellScore> = (0..csc.ncols())
            .into_par_iter()
            .map(|j| {
                // The cell's fold, when the projection divided by one.
                let fold_j = fold
                    .as_ref()
                    .zip(batches.as_ref())
                    .map(|(f, b)| f.column(b[j]));
                // Observed counts on the model's gene axis.
                let mut obs_full = vec![0f32; d_train];
                let col = csc.col(j);
                for (&row_new, &v) in col.row_indices().iter().zip(col.values()) {
                    let row_train = match remap.as_ref() {
                        Some(rm) => rm.new_to_train[row_new],
                        None => Some(row_new),
                    };
                    if let Some(r) = row_train {
                        obs_full[r] += v;
                    }
                }

                // The training chain's own per-cell view: L2-normalise to the
                // recorded scale, divide by the fold the projection divided
                // by (as `nystrom_preprocess_columns` does), then log1p. This
                // is the space svd fits in, and `u·z` reconstructs its
                // STANDARDISED form.
                let l2 = obs_full.iter().map(|v| v * v).sum::<f32>().sqrt().max(1e-8);
                let mut view: Vec<f32> =
                    obs_full.iter().map(|v| v / l2 * column_sum_norm).collect();
                if let Some(d) = fold_j.as_ref() {
                    let nz: Vec<usize> = (0..d_train).filter(|&g| view[g] > 0.0).collect();
                    let mut vals: Vec<f32> = nz.iter().map(|&g| view[g]).collect();
                    legume_numeric::matrix::dmatrix_util::adjust_by_poisson_ratio(&mut vals, |k| {
                        d[nz[k]]
                    });
                    for (&g, v) in nz.iter().zip(vals) {
                        view[g] = v;
                    }
                }
                let obs_log1p: Vec<f32> = view.iter().map(|v| v.ln_1p()).collect();
                // Mean/sd over the stored (non-zero) entries, matching how the
                // sparse standardisation was computed at training time.
                let nz: Vec<f32> = obs_log1p
                    .iter()
                    .zip(&obs_full)
                    .filter(|(_, &c)| c > 0.0)
                    .map(|(&l, _)| l)
                    .collect();
                let (mu, sd) = mean_sd(&nz);

                // Reconstruct, then undo the standardisation to land back in
                // log1p space; `expm1` gives a predicted count profile, which
                // `agreement_from_rate` renormalises to the observed depth.
                let mut pred_counts = vec![0f32; eval_genes.len()];
                let mut obs_eval = vec![0f32; eval_genes.len()];
                let mut sq_resid = 0f64;
                for (slot, &g) in eval_genes.iter().enumerate() {
                    let pred_log1p = recon_std_block[(j, slot)] * sd + mu;
                    // Clamped before `exp_m1`: one overflowing gene would make
                    // the profile's sum infinite, and `rate_to_counts` divides
                    // by that sum — zeroing the whole cell and turning both
                    // correlations into NaN. `ln(f32::MAX)` is ~88.
                    // Back on the observed counts' scale: the reconstruction is
                    // of the batch-adjusted profile, so the batch's δ is put
                    // back before it is compared with the raw counts.
                    let refold = fold_j.as_ref().map_or(1.0, |d| d[g]);
                    pred_counts[slot] = pred_log1p.min(80.0).exp_m1().max(0.0) * refold;
                    obs_eval[slot] = obs_full[g];
                    let r = f64::from(obs_log1p[g] - pred_log1p);
                    sq_resid += r * r;
                }
                let agree = agreement_from_rate(&obs_eval, &pred_counts);

                // Gaussian log-likelihood in log1p space at the ML σ̂:
                // `-0.5·Σ[r²/σ̂² + ln(2πσ̂²)]`, which at σ̂² = mean(r²) reduces
                // to `-0.5·D·(1 + ln(2πσ̂²))`.
                let d = eval_genes.len() as f64;
                let sigma2 = (sq_resid / d).max(1e-12);
                let llik = -0.5 * d * (1.0 + (std::f64::consts::TAU * sigma2).ln());
                CellScore {
                    llik: llik as f32,
                    total: obs_eval.iter().sum(),
                    spearman: agree.spearman,
                    pearson_log1p: agree.pearson_log1p,
                }
            })
            .collect();
        per_cell.extend(block);
    }

    let cell_names = data_vec.column_names()?;
    senna::output_helpers::save_latent(&args.out, &z_nk, &cell_names, None)?;

    let n_eval = eval_genes.len() as f32;
    let mut pred = Mat::zeros(per_cell.len(), 5);
    for (i, c) in per_cell.iter().enumerate() {
        pred[(i, 0)] = c.llik;
        pred[(i, 1)] = c.total;
        // Per GENE, not per count: this llik is a Gaussian over the scored
        // gene axis and does not scale with library size, so dividing it by
        // counts would put a different estimand under a column name that means
        // nats-per-count for every other family.
        pred[(i, 2)] = c.llik / n_eval;
        pred[(i, 3)] = c.spearman;
        pred[(i, 4)] = c.pearson_log1p;
    }
    let cols: Vec<Box<str>> = vec![
        "llik".into(),
        "total".into(),
        "llik_per_gene".into(),
        "spearman".into(),
        "pearson_log1p".into(),
    ];
    pred.to_parquet_with_names(
        &(args.out.to_string() + ".predictive.parquet"),
        (Some(&cell_names), Some("cell")),
        Some(&cols),
    )?;
    info!(
        "Agreement (svd, Gaussian log1p llik — compare families on the correlations): \
         per-cell spearman {:.3}, pearson(log1p) {:.3}; {} genes scored",
        mean_finite(per_cell.iter().map(|c| c.spearman)),
        mean_finite(per_cell.iter().map(|c| c.pearson_log1p)),
        eval_genes.len(),
    );
    info!("Wrote {}.predictive.parquet", args.out);
    Ok(())
}

/// One scored cell on the svd path. Named rather than a positional 4-tuple
/// because the aggregate log reads two of the fields by name.
struct CellScore {
    llik: f32,
    total: f32,
    spearman: f32,
    pearson_log1p: f32,
}

/// Mean and (population) standard deviation, with a floor so a constant slice
/// cannot divide by zero.
fn mean_sd(v: &[f32]) -> (f32, f32) {
    if v.is_empty() {
        return (0.0, 1.0);
    }
    let n = v.len() as f32;
    let mu = v.iter().sum::<f32>() / n;
    let var = v.iter().map(|x| (x - mu) * (x - mu)).sum::<f32>() / n;
    (mu, var.sqrt().max(1e-8))
}
