//! `senna revise`: move a fit on critique labels alone.
//!
//! Continuing a fit's training reshuffles which pairs a model merges on its
//! own. A revision does not train: it sets the parent up as `senna update`
//! would, on its own cells (its recorded inputs, batches and architecture,
//! collapsed on the partition the labels were judged on, warm-started from its
//! weights), then moves only the encoder
//! until each labelled pair is as far apart as the level's `--far-frac`
//! quantile of pair distances. The decoder is frozen, and nothing but the
//! labels enters the loss (see [`crate::peer_pairs`]).
//!
//! The likelihood is a gate, not a term: each level's is measured before and
//! after, and a revision that costs more than `--max-llik-drop` of it saves no
//! model, so the parent stays the latest version. Neither does one that was
//! interrupted or had no pair to move. `{out}.revise.parquet` records every
//! attempt, refused or not.
//!
//! An svd fit has no encoder: the labels vote on its genes instead, and the
//! SVD is solved again on the reweighted genes (see [`crate::svd::revise`]).
//! It has no likelihood to gate on, and the encoder settings do not apply.

use crate::peer_pairs::PeerRevision;
use crate::update::{continue_fit, Round, UpdateArgs};
use clap::Args;
use senna::run_manifest::Revise;

#[derive(Args, Debug)]
pub struct ReviseArgs {
    #[arg(
        long,
        required = true,
        help = "Model prefix to revise (a topic, vae or svd run)"
    )]
    pub(crate) model: Box<str>,

    #[arg(
        long,
        required = true,
        value_name = "FILE",
        help = "Critique labels to revise on ({out}.critique.labels.{model}.parquet)",
        long_help = "Pairs of pseudobulks this model keeps near while its peers hold\n\
                     them far, written by `senna critique --questions` over the\n\
                     partition this revision collapses on."
    )]
    pub(crate) labels: Box<str>,

    #[arg(
        short,
        long,
        required = true,
        help = "Output prefix for the revised model (must differ from --model)"
    )]
    pub(crate) out: Box<str>,

    #[arg(
        long,
        value_name = "RUN",
        help = "Collapse on this run's partition (default: the model's own)",
        long_help = "The partition the labels were judged on. `senna critique` averages\n\
                     every model over one run's pseudobulks; pass that run here unless\n\
                     it is the model itself."
    )]
    pub(crate) pb_from: Option<Box<str>>,

    #[arg(
        long,
        default_value_t = 0.25,
        help = "Quantile of a level's pair distances a labelled pair is pushed out to",
        long_help = "How far a labelled pair is pushed: out to this quantile of the\n\
                     level's pair distances in the parent's latent, and no further.\n\
                     The default matches `senna critique --far-frac`'s, so a merged\n\
                     pair ends where the critique would call it far."
    )]
    pub(crate) far_frac: f32,

    #[arg(
        long,
        default_value_t = 200,
        help = "Passes over the labelled pairs at most (fewer once every pair is far; not svd)"
    )]
    pub(crate) epochs: usize,

    #[arg(long, default_value_t = 1e-3, help = "Encoder learning rate (not svd)")]
    pub(crate) learning_rate: f32,

    #[arg(long, default_value_t = 64, help = "Labelled pairs per step (not svd)")]
    pub(crate) pair_batch: usize,

    #[arg(
        long,
        default_value_t = 0.01,
        help = "Refuse a revision that lowers any level's log-likelihood by more than this fraction (not svd)"
    )]
    pub(crate) max_llik_drop: f32,
}

impl ReviseArgs {
    fn revision(&self) -> anyhow::Result<PeerRevision> {
        crate::critique::check_far_frac(f64::from(self.far_frac))?;
        anyhow::ensure!(
            self.epochs > 0 && self.pair_batch > 0,
            "--epochs and --pair-batch must be positive"
        );
        anyhow::ensure!(
            self.learning_rate > 0.0 && self.max_llik_drop >= 0.0,
            "--learning-rate must be positive and --max-llik-drop non-negative"
        );
        Ok(PeerRevision(Revise {
            labels: self.labels.to_string(),
            far_frac: self.far_frac,
            epochs: self.epochs,
            learning_rate: self.learning_rate,
            pair_batch: self.pair_batch,
            max_llik_drop: self.max_llik_drop,
        }))
    }
}

pub fn run_revise(args: &ReviseArgs) -> anyhow::Result<()> {
    let round = Round {
        pb_from: args.pb_from.clone(),
        peer: args.revision()?,
    };
    let fit = UpdateArgs::own_cells(args.model.clone(), args.out.clone());
    continue_fit(&fit, Some(&round))
}

#[cfg(test)]
#[path = "revise_tests.rs"]
mod revise_tests;
