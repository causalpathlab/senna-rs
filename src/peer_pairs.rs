//! A round's peer labels as training pairs.
//!
//! `senna critique --questions` writes, per model, the pseudobulk pairs it
//! keeps near while its peers hold them far. A `senna update` round on that
//! model reads them here and pushes them apart through legume-numeric's pair
//! penalty (`train_mixed_with_pairs`).
//!
//! A label's level is its `cell_to_pb` column, which is also the training
//! level (coarsest first), and its pseudobulk ids are that level's data rows;
//! a round collapses on the partition the labels were judged on, so the two
//! agree. Each level's margin is the parent's own "far": the distance at the
//! critique's `--far-frac` among the level's pseudobulks, as the parent
//! encodes them. A merged pair is pushed until it is at least that far apart,
//! and no further.

use senna::embed_common::*;

use candle_core::Device;
use legume_numeric::candle::traits::EncoderModuleT;
use legume_numeric::candle::vae::pairs::{quantile_distance, LevelPairs, PairMetric, PairPenalty};
use legume_numeric::candle::vae::smooth_topics;
use legume_numeric::candle::vae::topic::LevelData;
use legume_numeric::matrix::parquet::read_table_columns;

/// Pairs each minibatch also encodes.
const PAIR_BATCH: usize = 64;

/// How a round trains on its peer labels. Set by `senna update`, never part of
/// a fit's recorded arguments: the manifest's history records it instead.
#[derive(Clone, Debug)]
pub(crate) struct PeerTraining {
    /// `{out}.critique.labels.{model}.parquet`.
    pub labels: Box<str>,
    /// λ, the penalty's weight against the per-sample ELBO.
    pub weight: f32,
    /// The quantile of the level's pair distances that counts as far.
    pub far_frac: f32,
}

impl PeerTraining {
    /// Each level's pairs, with the margin measured in `encoder`'s latent.
    pub(crate) fn levels<Enc: EncoderModuleT>(
        &self,
        encoder: &Enc,
        level_data: &[LevelData],
        metric: PairMetric,
        topic_smoothing: f64,
        dev: &Device,
    ) -> anyhow::Result<Vec<LevelPairs>> {
        let per_level = read_level_pairs(&self.labels, level_data.len())?;
        per_level
            .into_iter()
            .zip(level_data)
            .enumerate()
            .map(|(level, (pairs, &(input, null, _)))| {
                if pairs.is_empty() {
                    return Ok(LevelPairs::default());
                }
                let x = input.to_tensor(dev)?;
                let x0 = null.map(|b| b.to_tensor(dev)).transpose()?;
                let (z, _) = encoder.forward_t(&x, x0.as_ref(), false)?;
                let z = smooth_topics(z, topic_smoothing)?;
                let margin = quantile_distance(&z, metric, self.far_frac)?;
                log::info!(
                    "peer labels, level {level}: {} pairs pushed to at least {margin:.4} apart",
                    pairs.len()
                );
                Ok(LevelPairs { pairs, margin })
            })
            .collect()
    }

    pub(crate) fn penalty<'a>(
        &self,
        levels: &'a [LevelPairs],
        metric: PairMetric,
    ) -> PairPenalty<'a> {
        PairPenalty {
            per_level: levels,
            lambda: self.weight,
            metric,
            batch: PAIR_BATCH,
        }
    }
}

/// A critique label file as `(pb_a, pb_b, weight)` per level, for a fit of
/// `n_levels` levels. Only the pseudobulk channel is read; another channel,
/// a level the fit lacks or a negative id is an error, not a skipped row.
pub(crate) fn read_level_pairs(
    path: &str,
    n_levels: usize,
) -> anyhow::Result<Vec<Vec<(u32, u32, f32)>>> {
    let (strings, numbers) =
        read_table_columns(path, &["channel"], &["level", "pb_a", "pb_b", "weight"])?;
    let [level, a, b, w] = &numbers[..] else {
        unreachable!("four numeric columns requested")
    };
    let mut per_level = vec![Vec::new(); n_levels];
    for (i, channel) in strings[0].iter().enumerate() {
        anyhow::ensure!(
            channel.as_ref() == "A",
            "{path}: channel '{channel}' labels gene pairs, which no round trains on"
        );
        let id = |v: f64, what: &str| -> anyhow::Result<u32> {
            anyhow::ensure!(v >= 0.0, "{path}: row {i} has {what} {v}");
            Ok(v as u32)
        };
        let l = id(level[i], "level")? as usize;
        anyhow::ensure!(
            l < n_levels,
            "{path}: level {l} labelled, but the fit has {n_levels} levels; \
             were the labels judged on another partition?"
        );
        per_level[l].push((id(a[i], "pb_a")?, id(b[i], "pb_b")?, w[i] as f32));
    }
    Ok(per_level)
}

#[cfg(test)]
#[path = "peer_pairs_tests.rs"]
mod peer_pairs_tests;
