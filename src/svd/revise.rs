//! `senna revise` for an svd fit: the committee votes on genes, not on a
//! latent.
//!
//! SVD has no encoder to move and stays closed-form. What a merge says about
//! it is that the pair's difference lies along genes its components weigh too
//! little. So each labelled pair's two pseudobulks are profiled in the same
//! space the SVD solves in (normalised, log1p, standardised per cell), and
//! every gene is voted on by how far apart the merged pairs sit on it: up for
//! a gene that separates them more than the typical gene does, down for one
//! on which they look alike. The votes scale each gene's row before the solve,
//! so the components follow the separating genes.

use senna::embed_common::*;

use crate::peer_pairs::{read_level_pairs, PeerRevision};
use legume_numeric::matrix::parquet::{write_named_table, Column};
use rayon::prelude::*;
use rustc_hash::FxHashMap;

/// A vote moves a gene's weight by at most this factor either way.
const MAX_VOTE: f32 = 4.0;

/// Per-gene votes, in `data_vec`'s row order, from `peer`'s labels judged on
/// `partition_run`'s cell → pseudobulk partition: `(d_g / median d)`, clipped
/// to `[1/MAX_VOTE, MAX_VOTE]` and scaled to a geometric mean of 1, where
/// `d_g` is the label-weighted mean gap between a merged pair's profiles.
pub(crate) fn feature_votes(
    data_vec: &SparseIoVec,
    partition_run: &str,
    peer: &PeerRevision,
    column_sum_norm: f32,
) -> anyhow::Result<Vec<f32>> {
    let (m, dir) = senna::run_manifest::load_for(partition_run)?;
    let (per_level, src_names) =
        crate::refine_weighting::recorded_partition(partition_run, &m, &dir)?;
    let pairs = read_level_pairs(&peer.0.labels, per_level.len())?;
    anyhow::ensure!(
        pairs.iter().any(|l| !l.is_empty()),
        "{}: no labelled pairs to vote with",
        peer.0.labels
    );

    // Each labelled pseudobulk's data columns, from one pass per level.
    let col = crate::critique::tolerant_align(&src_names, &data_vec.column_names()?);
    let mut members: FxHashMap<(usize, u32), Vec<usize>> = pairs
        .iter()
        .enumerate()
        .flat_map(|(l, ps)| ps.iter().flat_map(move |&(a, b, _)| [(l, a), (l, b)]))
        .map(|k| (k, Vec::new()))
        .collect();
    for (l, level) in per_level.iter().enumerate() {
        for (&pb, &c) in level.iter().zip(&col) {
            if c != usize::MAX {
                if let Some(cells) = members.get_mut(&(l, pb as u32)) {
                    cells.push(c);
                }
            }
        }
    }

    // Each one profiled once, in the space the SVD solves in.
    let profiles: FxHashMap<(usize, u32), Vec<f32>> = members
        .into_par_iter()
        .map(|(key, cells)| {
            anyhow::ensure!(
                !cells.is_empty(),
                "labelled pseudobulk {} at level {} has no cells in the data",
                key.1,
                key.0
            );
            let n = cells.len() as f32;
            let mut x = data_vec.read_columns_csc(cells.into_iter())?;
            crate::svd::nystrom_preprocess_columns(&mut x, column_sum_norm, None);
            let mut mean = vec![0f32; x.nrows()];
            for (row, _, &v) in x.triplet_iter() {
                mean[row] += v / n;
            }
            Ok((key, mean))
        })
        .collect::<anyhow::Result<_>>()?;

    let d = data_vec.num_rows();
    let mut gap = vec![0f32; d];
    for (l, ps) in pairs.iter().enumerate() {
        for &(a, b, w) in ps {
            let (pa, pb) = (&profiles[&(l, a)], &profiles[&(l, b)]);
            for (g, (x, y)) in gap.iter_mut().zip(pa.iter().zip(pb)) {
                *g += w * (x - y).abs();
            }
        }
    }

    let mut scratch = gap.clone();
    let (_, &mut median, _) = scratch.select_nth_unstable_by(d / 2, f32::total_cmp);
    let median = median.max(f32::EPSILON);
    let mut votes: Vec<f32> = gap
        .iter()
        .map(|&g| (g.max(f32::EPSILON) / median).clamp(1.0 / MAX_VOTE, MAX_VOTE))
        .collect();
    let log_mean = votes.iter().map(|v| v.ln()).sum::<f32>() / d as f32;
    let scale = (-log_mean).exp();
    votes.iter_mut().for_each(|v| *v *= scale);
    let up = votes.iter().filter(|&&v| v > 1.0).count();
    info!(
        "revise [svd]: {} pair(s) voted on {d} genes: {up} up, {} down",
        pairs.iter().map(Vec::len).sum::<usize>(),
        d - up
    );
    Ok(votes)
}

/// What `senna revise` hands an svd fit: the labels, and the run whose
/// partition they were judged on (`None` when that run kept none).
#[derive(Clone, Debug)]
pub(crate) struct Vote {
    pub peer: PeerRevision,
    pub partition: Option<Box<str>>,
}

/// Per-gene weights from `--feature-weights`, in `gene_names` order; a gene
/// the file does not name keeps weight 1.
pub(crate) fn read_feature_weights(
    path: &str,
    gene_names: &[Box<str>],
) -> anyhow::Result<Vec<f32>> {
    let t = Mat::from_parquet_with_row_names(path, Some(0))?;
    let by_name: FxHashMap<&str, f32> = t
        .rows
        .iter()
        .enumerate()
        .map(|(i, n)| (n.as_ref(), t.mat[(i, 0)]))
        .collect();
    let w: Vec<f32> = gene_names
        .iter()
        .map(|g| by_name.get(g.as_ref()).copied().unwrap_or(1.0))
        .collect();
    anyhow::ensure!(
        w.iter().all(|&x| x.is_finite() && x > 0.0),
        "{path}: feature weights must be finite and positive"
    );
    Ok(w)
}

/// `{out}.feature_weights.parquet`: each gene's weight, and the vote this
/// revision cast on it.
pub(crate) fn write_feature_weights(
    path: &str,
    gene_names: &[Box<str>],
    weight: &[f32],
    vote: &[f32],
) -> anyhow::Result<()> {
    let cols: Vec<(Box<str>, Column)> = vec![
        ("weight".into(), Column::F32(weight)),
        ("vote".into(), Column::F32(vote)),
    ];
    write_named_table(path, "gene", gene_names, &cols)
}
