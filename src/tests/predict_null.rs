//! `senna predict` gives the encoder the null it was trained with: on a fit's
//! own cells it reproduces the fit's latent, and it would not without the
//! null.

use crate::fit_cli::{latent, parse};
use crate::planted::planted_batched_zarr;
use crate::predict::{predict_model, PredictArgs};
use senna::embed_common::{softmax_rows_inplace, Mat};

/// Each row of a latent as proportions (softmax), whatever its scale.
fn proportions(prefix: &str) -> Mat {
    let mut p = latent(prefix);
    softmax_rows_inplace(&mut p);
    p
}

/// Mean total-variation distance between the rows of two proportion tables.
fn mean_tv(a: &Mat, b: &Mat) -> f32 {
    let rows = a.nrows() as f32;
    (a - b).abs().row_sum().sum() * 0.5 / rows
}

/// Fit `family` on the batched cells, predict them back with and without
/// their batches, and return the distance of each prediction from the fit.
fn in_sample<A: clap::Args>(fit: fn(&A) -> anyhow::Result<()>) -> (f32, f32) {
    let dir = tempfile::tempdir().unwrap();
    let (data, batches) = planted_batched_zarr(dir.path(), 240, 2);
    let at = |name: &str| dir.path().join(name).to_string_lossy().into_owned();
    let model = at("model");
    fit(&parse::<A>(&[
        &data,
        "-b",
        &batches,
        "-o",
        &model,
        "-t",
        "3",
        "-i",
        "20",
        "--minibatch-size",
        "60",
        "--seed",
        "7",
    ]))
    .unwrap();
    let trained = proportions(&model);

    let predict = |out: &str, with_batches: bool| -> Mat {
        let mut argv = vec![data.as_str(), "--model", &model, "-o", out];
        if with_batches {
            argv.extend_from_slice(&["-b", &batches]);
        }
        predict_model(&parse::<PredictArgs>(&argv)).unwrap();
        proportions(out)
    };
    let with_null = predict(&at("with"), true);
    let without = predict(&at("without"), false);
    (mean_tv(&with_null, &trained), mean_tv(&without, &trained))
}

// Bar: with its null, predict on the fit's own cells stays within 0.05 mean
// TV of the fit's latent; without the null (no batches given), it does not.
#[test]
fn a_topic_fit_is_reproduced_on_its_own_cells() {
    let (with_null, without) = in_sample(crate::fit_topic_model);
    eprintln!("topic: with the null {with_null:.4}, without {without:.4}");
    assert!(with_null < 0.05, "with the null: {with_null}");
    assert!(without > with_null, "without {without}, with {with_null}");
}

#[test]
fn a_vae_fit_is_reproduced_on_its_own_cells() {
    let (with_null, without) = in_sample(crate::vae::fit_vae_model);
    eprintln!("vae: with the null {with_null:.4}, without {without:.4}");
    assert!(with_null < 0.05, "with the null: {with_null}");
    assert!(without > with_null, "without {without}, with {with_null}");
}

#[test]
fn a_masked_topic_fit_is_reproduced_on_its_own_cells() {
    let (with_null, without) = in_sample(crate::masked_topic::fit_masked_topic_model);
    eprintln!("masked: with the null {with_null:.4}, without {without:.4}");
    assert!(with_null < 0.05, "with the null: {with_null}");
    assert!(without > with_null, "without {without}, with {with_null}");
}
