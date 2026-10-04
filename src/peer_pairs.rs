//! A critique's labels as pairs to revise a model on.
//!
//! `senna critique --questions` writes, per model, the pseudobulk pairs it
//! keeps near while its peers hold them far. `senna revise` reads them here
//! and moves that model's encoder until each pair is far, with nothing else
//! in the loss (legume-numeric's `revise_encoder`): the decoder is frozen and
//! the likelihood is a gate on the result, not a term.
//!
//! A label's level is its `cell_to_pb` column, which is also the training
//! level (coarsest first), and its pseudobulk ids are that level's data rows;
//! a revision collapses on the partition the labels were judged on, so the
//! two agree. Each level's margin is the parent's own "far": the distance at
//! `--far-frac` among the level's pseudobulks, as the parent encodes them. A
//! merged pair is pushed until it is at least that far apart, and no further.

use senna::embed_common::*;

use candle_core::Device;
use legume_numeric::candle::traits::{DecoderModuleT, EncoderModuleT};
use legume_numeric::candle::vae::pairs::{
    quantile_distance, revise_encoder, LevelPairs, PairMetric, ReviseConfig,
};
use legume_numeric::candle::vae::smooth_topics;
use legume_numeric::candle::vae::topic::{level_llik, LevelData};
use legume_numeric::matrix::parquet::read_table_columns;
use std::sync::atomic::AtomicBool;

/// How `senna revise` moves a fit. Set by the revise command, never part of a
/// fit's recorded arguments: the manifest's history records it instead.
#[derive(Clone, Debug)]
pub(crate) struct PeerRevision {
    /// `{out}.critique.labels.{model}.parquet`.
    pub labels: Box<str>,
    /// The quantile of the level's pair distances that counts as far.
    pub far_frac: f32,
    /// Passes over the labelled pairs at most; fewer once every pair is at
    /// its margin.
    pub epochs: usize,
    pub learning_rate: f32,
    /// Labelled pairs per step.
    pub batch: usize,
    /// The largest drop in any level's log-likelihood per sample the revision may
    /// cost before it is refused, as a fraction of the parent's: the
    /// likelihood's scale differs by family and data, a fraction does not.
    pub max_llik_drop: f32,
}

/// What a revision did, level by level, written next to the revised fit.
pub(crate) struct Revision {
    pub llik_before: Vec<f32>,
    pub llik_after: Vec<f32>,
    pub n_pairs: Vec<usize>,
    pub margin: Vec<f32>,
    /// Share of a level's labelled pairs at or beyond their margin.
    pub resolved_before: Vec<f32>,
    pub resolved_after: Vec<f32>,
    /// Optimizer steps taken; 0 when every pair was already far.
    pub steps: usize,
}

/// The encoder's variables in every family `senna revise` moves.
const ENCODER_PREFIX: &str = "nn.enc";

impl PeerRevision {
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
                info!(
                    "revise, level {level}: {} pairs to push to at least {margin:.4} apart",
                    pairs.len()
                );
                Ok(LevelPairs { pairs, margin })
            })
            .collect()
    }

    /// Move `encoder` on the labels alone and write what happened to
    /// `record` (`{out}.revise.parquet`). The result is refused, so the caller
    /// saves no model and the parent stays the latest version, when it was
    /// interrupted, when no pair needed moving, or when any level's likelihood
    /// fell by more than `max_llik_drop`.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn revise<Enc, Dec>(
        &self,
        record: &str,
        level_data: &[LevelData],
        parameters: &candle_nn::VarMap,
        encoder: &Enc,
        decoders: &[Dec],
        metric: PairMetric,
        topic_smoothing: f64,
        minibatch_size: usize,
        dev: &Device,
        stop: &AtomicBool,
    ) -> anyhow::Result<()>
    where
        Enc: EncoderModuleT,
        Dec: DecoderModuleT,
    {
        let levels = self.levels(encoder, level_data, metric, topic_smoothing, dev)?;
        let llik = |encoder: &Enc| {
            level_llik(
                level_data,
                encoder,
                decoders,
                dev,
                topic_smoothing,
                minibatch_size,
            )
        };
        let llik_before = llik(encoder)?;
        let trace = revise_encoder(
            level_data,
            encoder,
            parameters,
            ENCODER_PREFIX,
            &levels,
            &ReviseConfig {
                dev,
                metric,
                topic_smoothing,
                learning_rate: self.learning_rate,
                max_epochs: self.epochs,
                batch: self.batch,
                grad_clip: 0.0,
                stop,
                verbose: true,
            },
        )?;
        let llik_after = llik(encoder)?;
        let resolved = |row: Option<&Vec<f32>>| row.cloned().unwrap_or_default();
        let revision = Revision {
            llik_before,
            llik_after,
            n_pairs: levels.iter().map(|l| l.pairs.len()).collect(),
            margin: levels.iter().map(|l| l.margin).collect(),
            resolved_before: resolved(trace.satisfied.first()),
            resolved_after: resolved(trace.satisfied.last()),
            steps: trace.steps,
        };
        revision.log();
        revision.to_parquet(record)?;
        anyhow::ensure!(
            !stop.load(std::sync::atomic::Ordering::Relaxed),
            "revise interrupted after {} step(s); no model saved",
            revision.steps
        );
        anyhow::ensure!(
            revision.steps > 0,
            "revise: every labelled pair is already at its margin; no model saved"
        );
        revision.check(self.max_llik_drop)
    }
}

impl Revision {
    fn log(&self) {
        info!("revise: {} step(s)", self.steps);
        for l in 0..self.n_pairs.len() {
            info!(
                "  level {l}: {} pairs, resolved {:.3} -> {:.3}; llik {:.5} -> {:.5}",
                self.n_pairs[l],
                self.resolved_before[l],
                self.resolved_after[l],
                self.llik_before[l],
                self.llik_after[l],
            );
        }
    }

    /// Every level's likelihood finite, and down by at most `max_drop` of
    /// the parent's magnitude.
    fn check(&self, max_drop: f32) -> anyhow::Result<()> {
        for (l, (&b, &a)) in self.llik_before.iter().zip(&self.llik_after).enumerate() {
            anyhow::ensure!(
                a.is_finite(),
                "revise refused: level {l}'s log-likelihood is {a} after the revision"
            );
            anyhow::ensure!(
                b - a <= max_drop * b.abs(),
                "revise refused: level {l}'s log-likelihood fell {b:.5} -> {a:.5}, more than \
                 --max-llik-drop {max_drop} of it; no model saved"
            );
        }
        Ok(())
    }

    /// `{out}.revise.parquet`: one row per level.
    fn to_parquet(&self, path: &str) -> anyhow::Result<()> {
        use legume_numeric::matrix::parquet::{write_named_table, Column};
        let n = self.n_pairs.len();
        let key: Vec<Box<str>> = (0..n).map(|l| l.to_string().into()).collect();
        let n_pairs: Vec<i32> = self.n_pairs.iter().map(|&n| n as i32).collect();
        let cols: Vec<(Box<str>, Column)> = vec![
            ("n_pairs".into(), Column::I32(&n_pairs)),
            ("margin".into(), Column::F32(&self.margin)),
            ("resolved_before".into(), Column::F32(&self.resolved_before)),
            ("resolved_after".into(), Column::F32(&self.resolved_after)),
            ("llik_before".into(), Column::F32(&self.llik_before)),
            ("llik_after".into(), Column::F32(&self.llik_after)),
        ];
        write_named_table(path, "level", &key, &cols)
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
            "{path}: channel '{channel}' labels gene pairs, which no revision reads"
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
