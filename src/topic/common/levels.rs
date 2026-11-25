//! The collapse levels of a fit: feature coarsenings per level, epochs per
//! level, posterior sampling of the pseudobulks, batch nulls, and the
//! per-level training data.

use senna::embed_common::*;

/// The finest collapse and the cell → pseudobulk membership that sizes its
/// pseudobulks: what the feature coarsening reads its counts from.
pub(crate) struct FinestPseudobulks<'a> {
    pub collapsed: &'a CollapsedOut,
    pub cell_to_pb: &'a [usize],
}

/// The per-level feature coarsenings of a fit: inherited verbatim from an
/// `--init-from` parent (its weights are keyed to the parent's grouping),
/// otherwise [`coarsen_features`] of the finest
/// pseudobulk counts when `max_coarse_features` caps the feature axis, with
/// log-spaced widths coarsest first; `None` per level when neither applies.
pub(crate) fn resolve_level_coarsenings(
    cap: Option<std::num::NonZeroUsize>,
    init_from: Option<&str>,
    finest: &FinestPseudobulks<'_>,
    num_levels: usize,
    n_features_full: usize,
    seed: u64,
    gene_axis: Option<&crate::topic::eval::GeneRemap>,
) -> anyhow::Result<Vec<Option<FeatureCoarsening>>> {
    if let Some(parent) = init_from {
        // Continuing keys the child to the SOURCE run's groups, read from its
        // `coarsening.json`, because every decoder is keyed to their count.
        // So the cap is inert here, and saying so beats letting someone
        // believe they changed the resolution of a continued fit.
        let levels = inherit_level_coarsenings(
            parent,
            num_levels,
            n_features_full,
            gene_axis,
            finest.collapsed.mu_observed.posterior_mean(),
        )?;
        let inherited = levels
            .last()
            .and_then(Option::as_ref)
            .map(|fc| fc.num_coarse);
        if inherited != cap.map(std::num::NonZeroUsize::get) {
            log::info!(
                "--init-from {parent}: keeping that run's {} coarse feature(s); the coarsening \
                 options do not apply to a continued fit, whose decoders are keyed to the \
                 groups they were trained on",
                inherited.map_or_else(|| "no".to_string(), |n| n.to_string()),
            );
        }
        return Ok(levels);
    }
    let Some(cap) = cap
        .map(std::num::NonZeroUsize::get)
        .filter(|c| n_features_full > *c)
    else {
        return Ok(vec![None; num_levels]);
    };
    let finest_target = cap;
    let min_target = (finest_target / num_levels).max(50);
    let level_targets: Vec<usize> = (0..num_levels)
        .map(|i| {
            let frac = if num_levels > 1 {
                i as f64 / (num_levels - 1) as f64
            } else {
                1.0
            };
            let log_min = (min_target as f64).ln();
            let log_max = (finest_target as f64).ln();
            let target = (log_min + frac * (log_max - log_min)).exp().round() as usize;
            target.clamp(min_target, finest_target)
        })
        .collect();
    let (counts, sizes) = finest.collapsed.observed_counts(finest.cell_to_pb)?;
    Ok(coarsen_features(&counts, &sizes, &level_targets, seed)?
        .into_iter()
        .map(Some)
        .collect())
}

/// Load the parent's per-level feature coarsenings for a `--init-from` run.
///
/// See the call site for why recomputing them is unsafe. Both the "parent had
/// none" and "parent had some" cases have to agree with this run, so a
/// mismatch is reported rather than silently reconciled.
///
/// With `gene_axis`, a level keyed to the source run's axis is GROWN onto this
/// run's by name instead of refused: known genes keep their coarse group,
/// unknown ones are placed by their pseudobulk profile (`profiles_dn`, this
/// run's finest posterior, read the way bge's alignment reads one:
/// depth-normalized, log, centred), and the coarse-feature count the decoders
/// are keyed to stays.
pub(super) fn inherit_level_coarsenings(
    parent: &str,
    num_levels: usize,
    n_features_full: usize,
    gene_axis: Option<&crate::topic::eval::GeneRemap>,
    profiles_dn: &Mat,
) -> anyhow::Result<Vec<Option<FeatureCoarsening>>> {
    use crate::topic::model_metadata::load_coarsening_levels;

    let Some(mut levels) = load_coarsening_levels(parent)? else {
        // No file: the parent trained at full resolution. Match it, so the
        // encoder's input width is `n_features_full` on both sides.
        log::info!(
            "--init-from {parent}: parent trained without feature coarsening; \
             training at full resolution ({n_features_full} features) to match"
        );
        return Ok(vec![None; num_levels]);
    };

    anyhow::ensure!(
        levels.len() == num_levels,
        "--init-from {parent}: parent has {} coarsening level(s) but this run has \
         --num-levels {num_levels}. The per-level decoders are keyed to their own \
         groupings, so the ladders must match — pass --num-levels {}.",
        levels.len(),
        levels.len(),
    );
    // One reading of every gene's profile, shared by all levels.
    let unit_profiles = gene_axis.map(|_| {
        anyhow::ensure!(
            profiles_dn.nrows() == n_features_full,
            "gene axis growth: {} pseudobulk profiles for {n_features_full} genes",
            profiles_dn.nrows(),
        );
        Ok(graph_embedding_util::transfer::unit_log_profile_rows(
            profiles_dn,
        ))
    });
    let unit_profiles = unit_profiles.transpose()?;
    for (i, lvl) in levels.iter_mut().enumerate() {
        let Some(fc) = lvl else { continue };
        // A remap means the axes differ by NAME, so the level is grown even
        // when the two axes happen to have the same length: positions no
        // longer mean the same gene.
        if let (Some(remap), Some(unit)) = (gene_axis, unit_profiles.as_ref()) {
            anyhow::ensure!(
                remap.d_train == fc.fine_to_coarse.len(),
                "--init-from {parent}: level {i}'s coarsening covers {} features but the \
                 source run's gene axis has {}",
                fc.fine_to_coarse.len(),
                remap.d_train,
            );
            *fc = fc.grow_by_profile(&remap.new_to_train, unit)?;
        } else {
            anyhow::ensure!(
                fc.fine_to_coarse.len() == n_features_full,
                "--init-from {parent}: level {i}'s coarsening covers {} features but this \
                 run has {n_features_full}. The two cohorts do not share a gene axis — either \
                 they spell some genes differently (see --feature-name-kind) or the new data \
                 measures genes the model has never seen, which cannot be added to a trained \
                 model.",
                fc.fine_to_coarse.len(),
            );
        }
    }
    let widths: Vec<String> = levels
        .iter()
        .map(|l| {
            l.as_ref()
                .map_or_else(|| "full".into(), |c| c.num_coarse.to_string())
        })
        .collect();
    log::info!(
        "--init-from {parent}: inheriting feature coarsening, level widths [{}] \
         (--max-coarse-features is ignored)",
        widths.join(", "),
    );
    Ok(levels)
}

/// Compute per-level epoch allocation for progressive training.
///
/// Coarser levels (lower index) get more epochs: `w[i] = num_levels - i`.
pub(crate) fn compute_level_epochs(total_epochs: usize, num_levels: usize) -> Vec<usize> {
    let total_weight: usize = (1..=num_levels).sum();
    (0..num_levels)
        .map(|i| {
            let w = num_levels - i;
            (total_epochs * w / total_weight).max(1)
        })
        .collect()
}

/// The seed of level `level`'s posterior draw in a run seeded `seed`.
pub(crate) fn posterior_seed(seed: u64, level: usize) -> u64 {
    legume_numeric::matrix::rand_util::stream_seed(seed, "posterior", 0, level)
}

/// Draw `(mixed_nd, batch_nd, target_nd)` from the collapsed posteriors
/// (one sample per Gamma matrix), reproducibly from `seed`
/// ([`posterior_seed`]).
pub(crate) fn sample_collapsed_data(
    collapsed: &CollapsedOut,
    seed: u64,
    batch_null: Option<&Mat>,
) -> anyhow::Result<(Mat, Option<Mat>, Mat)> {
    let (mixed_dn, batch_dn, target_dn) = sample_collapsed_data_dp(collapsed, seed, batch_null)?;
    Ok((
        mixed_dn.transpose(),
        batch_dn.map(|b| b.transpose()),
        target_dn.transpose(),
    ))
}

/// [`sample_collapsed_data`] in the posterior's own `[D, P]` orientation —
/// genes down, pseudobulk samples across.
///
/// The transposes above are a strided copy of the whole matrix each, and a
/// consumer that uploads the triple to a device does not need them: a
/// column-major `[D, P]` buffer already IS the row-major `[P, D]` one (see
/// `legume_numeric::candle::data::masked_dense`). Callers that want host `[P, D]` matrices
/// still get them from [`sample_collapsed_data`].
///
/// The encoder's null is `batch_null` (`[D, P]`, see [`batch_level_nulls`])
/// when one is given, else a draw of the pseudobulks' own fold `μ_residual`.
pub(crate) fn sample_collapsed_data_dp(
    collapsed: &CollapsedOut,
    seed: u64,
    batch_null: Option<&Mat>,
) -> anyhow::Result<(Mat, Option<Mat>, Mat)> {
    use legume_numeric::matrix::rand_util::name_seed;
    let mixed_dn = collapsed
        .mu_observed
        .posterior_sample_seeded(name_seed(seed, "observed"))?;

    let batch_dn = match batch_null {
        Some(null) => Some(null.clone()),
        None => collapsed
            .mu_residual
            .as_ref()
            .map(|x| x.posterior_sample_seeded(name_seed(seed, "residual")))
            .transpose()?,
    };

    let target_dn = if let Some(adj) = &collapsed.mu_adjusted {
        adj.posterior_sample_seeded(name_seed(seed, "adjusted"))?
    } else {
        mixed_dn.clone()
    };

    Ok((mixed_dn, batch_dn, target_dn))
}

/// The encoder's training null of each level under `--adj-method batch`,
/// `[D, P_level]`, or `None` under `residual` or when the collapse found one
/// batch.
///
/// At inference a cell reads its batch's `δ[g, b]`. A training pseudobulk
/// mixes cells of several batches, so it reads δ mixed by its batch shares,
/// `Σ_b δ[g,b] · n_bp / n_p`: what its counts carry of the batch effect under
/// the collapse's own model `E[obs_gp] = μ_gp · Σ_b δ_gb n_bp`. The collapse
/// fits that model at every level, each with its own δ, and each level's
/// null reads its own: the δ its `μ_adjusted` (the decoder's target) was
/// divided by. The finest level's δ is the one written to
/// `{out}.delta.parquet` and read back at inference and by `predict`.
///
/// Without this the encoder trained on the pseudobulk's own fold `μ_residual`
/// whatever the method, and met a per-batch δ only at inference.
///
/// Levels and memberships both run coarsest..finest. The levels may be
/// borrowed, e.g. one modality's of a stacked collapse (`senna joint-topic`).
pub(crate) fn batch_level_nulls<C: std::borrow::Borrow<CollapsedOut>>(
    adj_method: &AdjMethod,
    collapsed_levels: &[C],
    cell_to_pb_per_level: Option<&[Vec<usize>]>,
    data_vec: &SparseIoVec,
) -> anyhow::Result<Option<Vec<Mat>>> {
    if !matches!(adj_method, AdjMethod::Batch) {
        return Ok(None);
    }
    if collapsed_levels
        .last()
        .is_none_or(|c| c.borrow().delta.is_none())
    {
        return Ok(None);
    }
    let cell_to_pb_per_level = cell_to_pb_per_level.ok_or_else(|| {
        anyhow::anyhow!("--adj-method batch needs the collapse's cell → pseudobulk membership")
    })?;
    anyhow::ensure!(
        cell_to_pb_per_level.len() == collapsed_levels.len(),
        "{} levels of cell → pseudobulk membership for {} collapsed levels",
        cell_to_pb_per_level.len(),
        collapsed_levels.len()
    );
    let batches = data_vec.get_batch_membership(0..data_vec.num_columns());
    // A carried pseudobulk column stands for many cells (`--pb-reference`);
    // the collapse weighs it so, and so must its share.
    let weights = data_vec.column_multiplicities();
    collapsed_levels
        .iter()
        .zip(cell_to_pb_per_level)
        .enumerate()
        .map(|(level, (collapsed, cell_to_pb))| {
            let collapsed = collapsed.borrow();
            let delta_db = collapsed
                .delta
                .as_ref()
                .ok_or_else(|| {
                    anyhow::anyhow!("collapse level {level} fit no δ, the finest level did")
                })?
                .posterior_mean();
            let n_pb = collapsed.mu_observed.posterior_mean().ncols();
            let shares = batch_shares(&batches, cell_to_pb, weights, delta_db.ncols(), n_pb)?;
            Ok(delta_db * shares)
        })
        .collect::<anyhow::Result<Vec<Mat>>>()
        .map(Some)
}

/// `[B, P]`: the share of each pseudobulk's cells that each batch holds,
/// a column counting for `weights` cells (one each without). A pseudobulk
/// with no cell takes an even share of every batch.
pub(super) fn batch_shares(
    batches: &[usize],
    cell_to_pb: &[usize],
    weights: Option<&[f32]>,
    n_batches: usize,
    n_pb: usize,
) -> anyhow::Result<Mat> {
    anyhow::ensure!(
        batches.len() == cell_to_pb.len(),
        "{} cells with a batch, {} with a pseudobulk",
        batches.len(),
        cell_to_pb.len()
    );
    anyhow::ensure!(
        weights.is_none_or(|w| w.len() == batches.len()),
        "column weights do not cover the {} cells",
        batches.len()
    );
    let mut shares = Mat::zeros(n_batches, n_pb);
    for (cell, (&b, &p)) in batches.iter().zip(cell_to_pb).enumerate() {
        anyhow::ensure!(
            b < n_batches && p < n_pb,
            "cell in batch {b} of {n_batches}, pseudobulk {p} of {n_pb}"
        );
        shares[(b, p)] += weights.map_or(1.0, |w| w[cell]);
    }
    for mut col in shares.column_iter_mut() {
        let n = col.sum();
        if n > 0.0 {
            col /= n;
        } else {
            col.fill(1.0 / n_batches as f32);
        }
    }
    Ok(shares)
}

/// Per-gene mean rate `μ_d` from a `[D, n_pb]` pseudobulk posterior mean —
/// the row-mean across pseudobulks. This is the divisive gene-mean correction
/// fed to `anscombe_residual` (full-`D`; callers coarsen afterward if needed).
pub(crate) fn pseudobulk_feature_mean(mu_dp: &Mat) -> Vec<f32> {
    let n_pb = mu_dp.ncols().max(1) as f32;
    (0..mu_dp.nrows())
        .map(|d| mu_dp.row(d).iter().sum::<f32>() / n_pb)
        .collect()
}

/// Materialize `(encoder-input, batch, decoder-target)` `Mat` triples
/// once per training run, applying the encoder's and per-level decoder
/// coarsenings.
///
/// Shared by the families whose encoder and decoders both train at coarse
/// width. `enc_coarsening` is the finest level's, since that is what the
/// encoder reads; each decoder takes its own level's.
pub(crate) fn build_level_data(
    collapsed_levels: &[CollapsedOut],
    level_coarsenings: &[Option<FeatureCoarsening>],
    enc_coarsening: Option<&FeatureCoarsening>,
    seed: u64,
    batch_nulls: Option<&[Mat]>,
) -> anyhow::Result<Vec<(Mat, Option<Mat>, Mat)>> {
    collapsed_levels
        .iter()
        .zip(level_coarsenings.iter())
        .enumerate()
        .map(|(level, (collapsed, dec_fc))| {
            let (mixed_nd, batch_nd, target_nd) = sample_collapsed_data(
                collapsed,
                posterior_seed(seed, level),
                batch_nulls.map(|n| &n[level]),
            )?;

            let enc_nd = if let Some(fc) = enc_coarsening {
                fc.aggregate_columns_nd(&mixed_nd)
            } else {
                mixed_nd
            };

            let batch_nd = batch_nd.map(|b| {
                if let Some(fc) = enc_coarsening {
                    fc.aggregate_columns_nd(&b)
                } else {
                    b
                }
            });

            let dec_target = if let Some(fc) = dec_fc.as_ref() {
                fc.aggregate_columns_nd(&target_nd)
            } else {
                target_nd
            };

            Ok((enc_nd, batch_nd, dec_target))
        })
        .collect()
}
