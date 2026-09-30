//! `senna bge` ETM resolution (on by default; disable with `--skip-etm`):
//! resolve the ETM topic side from a finished bge run (no further training)
//! and write a topic-model-shaped output layout (`latent` = log θ,
//! `dictionary` = β). Called from the shared [`crate::bge::driver`].

use graph_embedding_util as ge;
use senna::embed_common::*;

//////////////////////////////////////////////////////////////////////
// ETM resolution from the bge cell embedding (default; --skip-etm) //
//////////////////////////////////////////////////////////////////////

/// Resolve the ETM topic side from a finished bge run, with no further
/// training, and write a topic-model-shaped output layout so that
/// `senna clustering` and `lupin {plot, plot-topic, annotate} --from` consume the
/// topics directly (matching the `senna topic` / `masked-topic` conventions:
/// `latent` = log θ, `dictionary` = β).
///
/// Archetypal analysis on the cell embedding `Z [N,H]` yields archetypes
/// `α [K,H]` (= topic embeddings) and per-cell simplex weights `θ [N,K]`
/// (= topic proportions); the dictionary is `β = log_softmax_d(ρ·αᵀ)`,
/// the same factorization the ETM decoder uses. Writes:
///   - `{out}.latent.parquet`           log θ [N,K]   (topic proportions)
///   - `{out}.dictionary.parquet`       β    [D,K]   (each topic column a gene simplex)
///   - `{out}.topic_embedding.parquet`  α    [K,H]   (for a later `masked-topic` finetune)
///   - `{out}.cell_embedding.parquet`   Z    [N,H]   (raw bge cell embedding)
///   - `{out}.feature_bias.parquet`     `b_feat` [D]
///   - `{out}.cell_bias.parquet`        `b_cell` [N]    (per-cell depth sink)
pub(super) fn resolve_etm_topics(
    model: &ge::JointEmbedModel,
    feature_names: &[Box<str>],
    barcodes: &[Box<str>],
    out: &str,
    cell_keep_idx: Option<&[usize]>,
    labels: &[usize],
) -> anyhow::Result<()> {
    let cpu = candle_core::Device::Cpu;
    let z_full = Mat::from_tensor(&model.e_cell.to_device(&cpu)?)?; // [N, H]
                                                                    // Drop QC-failed cells from archetype fitting + per-cell outputs. `z`
                                                                    // and `barcodes` are subset by the same `keep` so their rows stay
                                                                    // aligned; the dictionary β (from ρ + archetypes) is per-feature and
                                                                    // unaffected.
    let (z, barcodes): (Mat, Vec<Box<str>>) = match cell_keep_idx {
        Some(keep) => (
            z_full.select_rows(keep.iter()),
            keep.iter().map(|&i| barcodes[i].clone()).collect(),
        ),
        None => (z_full, barcodes.to_vec()),
    };
    let rho = Mat::from_tensor(&model.e_feat.to_device(&cpu)?)?; // [D, H]
    let h = z.ncols();

    // Robust topic recovery from CELL CLUSTERS, using the Leiden `labels` the
    // bge driver computed ONCE on this same kept-cell embedding (shared with the
    // co-embedding's temperature calibration). The previous approach (Arora/SPA
    // convex-hull anchors on ρ) was outlier-driven: a few extreme features (e.g.
    // immunoglobulin genes) became the hull vertices, SPA picked them, and the
    // min-cells guard nuked the rest → K collapsed (BM1 → 2).
    let (alpha, log_theta, beta_dk) = topics_from_clusters(&z, &rho, labels)?;
    let k = alpha.nrows();
    let topic_names = axis_id_names("T", k);
    let h_names = axis_id_names("h", h);

    // Topic-model layout — latent = log θ, dictionary = β.
    log_theta.to_parquet_with_names(
        &format!("{out}.latent.parquet"),
        (Some(&barcodes), Some("cell")),
        Some(&topic_names),
    )?;
    beta_dk.to_parquet_with_names(
        &format!("{out}.dictionary.parquet"),
        (Some(feature_names), Some("gene")),
        Some(&topic_names),
    )?;
    // Resolved topic embeddings α = cluster centroids (warm-start for a later
    // `masked-topic` finetune).
    alpha.to_parquet_with_names(
        &format!("{out}.topic_embedding.parquet"),
        (Some(&topic_names), Some("topic")),
        Some(&h_names),
    )?;
    // Raw bge cell embedding Z preserved (the reference manifold). The feature
    // side — {out}.feature_embedding.parquet (SIMBA co-embed) — is written by
    // the bge driver before this call, so it is not emitted here.
    z.to_parquet_with_names(
        &format!("{out}.cell_embedding.parquet"),
        (Some(&barcodes), Some("cell")),
        Some(&h_names),
    )?;
    ge::eval::save_bias(
        &format!("{out}.feature_bias.parquet"),
        &model.b_feat,
        feature_names,
        "feature",
    )?;
    // Per-cell bias `b_cell` (depth sink), subset by the same QC keep mask as
    // the cell rows above so barcodes/biases stay aligned.
    let b_cell = match cell_keep_idx {
        Some(keep) => {
            let idx: Vec<u32> = keep.iter().map(|&i| i as u32).collect();
            let idx_t = candle_core::Tensor::from_vec(idx, keep.len(), model.b_cell.device())?;
            model.b_cell.index_select(&idx_t, 0)?
        }
        None => model.b_cell.clone(),
    };
    ge::eval::save_bias(
        &format!("{out}.cell_bias.parquet"),
        &b_cell,
        &barcodes,
        "cell",
    )?;

    info!(
        "resolve-etm: wrote topic-model layout (latent=log θ, dictionary=β) + \
         cell_embedding.parquet to {out}.*"
    );
    Ok(())
}

/// Topics seeded from cell clusters on a cell embedding `z [N,H]`, one topic
/// per cluster (`labels`, one per row of `z`):
///   α [K,H] = L2-normalised cluster centroid (the topic *direction*),
///   θ [N,K] = softmax over clusters of ⟨z_i, α_k⟩ (soft assignment),
///   β [D,K] = `log_softmax_d(ρ·(α−ᾱ)ᵀ)`: a gene scores high in a topic when
///             its embedding `rho [D,H]` aligns with that cluster's cells.
/// Returns (α, log θ, β). `senna bge` resolves its topics so, and
/// `senna resolve-topics` does the same for a finished run.
pub(crate) fn topics_from_clusters(
    z: &Mat,
    rho: &Mat,
    labels: &[usize],
) -> anyhow::Result<(Mat, Mat, Mat)> {
    use legume_numeric::matrix::archetypal::topic_dictionary;
    anyhow::ensure!(
        rho.ncols() == z.ncols(),
        "cell embedding H={} != feature embedding H={}",
        z.ncols(),
        rho.ncols()
    );
    anyhow::ensure!(
        labels.len() == z.nrows(),
        "cluster labels ({}) != cells ({})",
        labels.len(),
        z.nrows()
    );
    let k = labels.iter().copied().max().map_or(0, |m| m + 1);
    anyhow::ensure!(k >= 2, "fewer than 2 clusters: nothing to make topics of");
    let alpha = cluster_centroids(z, labels, k); // [K, H]
    let theta = soft_theta(z, &alpha); // [N, K]
    let mut sizes = vec![0usize; k];
    for &l in labels {
        sizes[l] += 1;
    }
    info!("topics: cluster-seeded K={k}, cluster sizes={sizes:?}");
    let beta = topic_dictionary(rho, &alpha); // [D, K]
    let log_theta = theta.map(|x| (x + 1e-8).ln());
    Ok((alpha, log_theta, beta))
}

/// L2-normalized cluster centroids of `z` `[N,H]` → `α` `[K,H]` (topic
/// directions). Empty clusters (shouldn't occur) stay zero.
fn cluster_centroids(z: &Mat, labels: &[usize], k: usize) -> Mat {
    use rayon::prelude::*;
    let h = z.ncols();
    // Per-cluster sums and counts, a chunk of cells a thread, then combined.
    let (sums, counts) = (0..z.nrows())
        .into_par_iter()
        .fold(
            || (vec![0f32; k * h], vec![0usize; k]),
            |(mut sums, mut counts), i| {
                let l = labels[i];
                counts[l] += 1;
                for (s, &v) in sums[l * h..(l + 1) * h].iter_mut().zip(z.row(i).iter()) {
                    *s += v;
                }
                (sums, counts)
            },
        )
        .reduce(
            || (vec![0f32; k * h], vec![0usize; k]),
            |(mut a, mut ca), (b, cb)| {
                a.iter_mut().zip(&b).for_each(|(x, y)| *x += y);
                ca.iter_mut().zip(&cb).for_each(|(x, y)| *x += y);
                (a, ca)
            },
        );
    let mut alpha = Mat::zeros(k, h);
    for l in (0..k).filter(|&l| counts[l] > 0) {
        let mean: Vec<f32> = sums[l * h..(l + 1) * h]
            .iter()
            .map(|s| s / counts[l] as f32)
            .collect();
        let nrm = mean.iter().map(|v| v * v).sum::<f32>().sqrt().max(1e-8);
        for (j, v) in mean.iter().enumerate() {
            alpha[(l, j)] = v / nrm;
        }
    }
    alpha
}

/// Soft assignment θ [N,K]: each cell's softmax over ⟨z_i, α_k⟩, cells on
/// rayon's threads.
fn soft_theta(z: &Mat, alpha: &Mat) -> Mat {
    use rayon::prelude::*;
    let k = alpha.nrows();
    let rows: Vec<f32> = (0..z.nrows())
        .into_par_iter()
        .flat_map_iter(|i| {
            let zi = z.row(i);
            let s: Vec<f32> = (0..k).map(|c| zi.dot(&alpha.row(c))).collect();
            let mx = s.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let e: Vec<f32> = s.iter().map(|v| (v - mx).exp()).collect();
            let sum = e.iter().sum::<f32>().max(1e-8);
            e.into_iter().map(move |v| v / sum)
        })
        .collect();
    Mat::from_row_slice(z.nrows(), k, &rows)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn topics_follow_the_clusters_they_are_seeded_from() {
        // Two clusters of cells along h0 and h1; genes aligned with each.
        let z = Mat::from_row_slice(4, 2, &[3.0, 0.1, 2.5, 0.0, 0.0, 3.0, 0.2, 2.8]);
        let rho = Mat::from_row_slice(3, 2, &[2.0, 0.0, 0.0, 2.0, 0.1, 0.1]);
        let (alpha, log_theta, beta) = topics_from_clusters(&z, &rho, &[0, 0, 1, 1]).unwrap();
        for k in 0..2 {
            assert!((alpha.row(k).norm() - 1.0).abs() < 1e-5);
            // Each topic's dictionary is a simplex over genes, in log space.
            let s: f32 = beta.column(k).iter().map(|v| v.exp()).sum();
            assert!((s - 1.0).abs() < 1e-4);
        }
        for (i, want) in [0, 0, 1, 1].into_iter().enumerate() {
            let row = log_theta.row(i);
            assert!((row.iter().map(|v| v.exp()).sum::<f32>() - 1.0).abs() < 1e-4);
            assert_eq!(row.transpose().argmax().0, want);
        }
        // Gene 0 leads topic 0, gene 1 topic 1.
        assert_eq!(beta.column(0).argmax().0, 0);
        assert_eq!(beta.column(1).argmax().0, 1);
        assert!(topics_from_clusters(&z, &rho, &[0, 0, 0, 0]).is_err());
    }
}
