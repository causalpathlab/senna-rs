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

/// What a `--adj-method batch` fit on the planted batches gives: how far
/// predict lands from the fit's latent with the trained batches, with the
/// same cells under batch names it never saw, and how the latent splits by
/// batch (within a group) against by group.
struct BatchFit {
    known: f32,
    unknown: f32,
    by_batch: f32,
    by_group: f32,
    trained_on_delta: bool,
    /// The cells of batch B0, predicted with every cell labelled B0: one
    /// held-out batch, which only a trained δ can serve.
    one_batch: f32,
}

fn batch_fit<A: clap::Args>(fit: fn(&A) -> anyhow::Result<()>) -> BatchFit {
    let dir = tempfile::tempdir().unwrap();
    let n = 240;
    let (data, batches) = planted_batched_zarr(dir.path(), n, 2);
    let at = |name: &str| dir.path().join(name).to_string_lossy().into_owned();
    let model = at("model");
    let argv = [
        data.as_str(),
        "-b",
        &batches,
        "-o",
        &model,
        "-t",
        "3",
        "-i",
        "20",
    ];
    let more = [
        "--minibatch-size",
        "60",
        "--seed",
        "7",
        "--adj-method",
        "batch",
    ];
    fit(&parse::<A>(&[&argv[..], &more[..]].concat())).unwrap();
    let trained = proportions(&model);

    let renamed = at("renamed.batch.txt");
    let labels: String = (0..n).map(|c| format!("X{}\n", c % 2)).collect();
    std::fs::write(&renamed, labels).unwrap();
    let predict = |out: &str, b: &str| -> Mat {
        let out = at(out);
        let argv = [data.as_str(), "--model", &model, "-o", &out, "-b", b];
        predict_model(&parse::<PredictArgs>(&argv)).unwrap();
        proportions(&out)
    };
    let known = predict("known", &batches);
    let unknown = predict("unknown", &renamed);
    let all_b0 = at("b0.batch.txt");
    std::fs::write(&all_b0, "B0\n".repeat(n)).unwrap();
    let one = predict("one", &all_b0);
    let b0: Vec<usize> = (0..n).step_by(2).collect();
    let one_batch = mean_tv(&one.select_rows(&b0), &trained.select_rows(&b0));

    // Mean latent of the cells of group `g` in batch `b`.
    let mean_of = |g: usize, b: usize| -> Mat {
        let rows: Vec<usize> = (0..n)
            .filter(|c| usize::from(*c >= n / 2) == g && c % 2 == b)
            .collect();
        let m = trained.select_rows(&rows);
        Mat::from_rows(&[m.row_sum() / rows.len() as f32])
    };
    let tv = |a: &Mat, b: &Mat| (a - b).abs().sum() * 0.5;
    let by_batch = (tv(&mean_of(0, 0), &mean_of(0, 1)) + tv(&mean_of(1, 0), &mean_of(1, 1))) / 2.0;
    let by_group = (tv(&mean_of(0, 0), &mean_of(1, 0)) + tv(&mean_of(0, 1), &mean_of(1, 1))) / 2.0;
    let meta: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(format!("{model}.model.json")).unwrap())
            .unwrap();
    BatchFit {
        known: mean_tv(&known, &trained),
        unknown: mean_tv(&unknown, &trained),
        by_batch,
        by_group,
        trained_on_delta: meta["batch_null_in_training"] == true,
        one_batch,
    }
}

// Bar: under `--adj-method batch` the encoder trains on the per-batch δ, so
// predict reproduces the fit on its own cells from the TRAINED δ alone
// (within 0.05 mean TV), and the latent splits by group far more than by
// batch. One held-out batch is enough: its cells land where the fit put them
// though every other cell was mislabelled. Batches the model never saw fall
// back to a null estimated from the cells.
fn check_batch_fit(family: &str, f: &BatchFit) {
    eprintln!(
        "{family} (batch): known {:.4}, unknown {:.4}, one batch {:.4}, by batch {:.4}, \
         by group {:.4}",
        f.known, f.unknown, f.one_batch, f.by_batch, f.by_group
    );
    assert!(
        f.trained_on_delta,
        "{family}: the fit did not record its null"
    );
    assert!(f.known < 0.05, "{family}: with the trained δ {}", f.known);
    // Looser than the in-sample bar: a fit is not bit-reproducible under
    // load, and the masked one moves this by up to ~0.05 between runs.
    assert!(f.one_batch < 0.1, "{family}: one batch {}", f.one_batch);
    assert!(
        f.unknown < 0.05,
        "{family}: unseen batch names {}",
        f.unknown
    );
    assert!(
        f.by_group > 3.0 * f.by_batch,
        "{family}: by group {} vs by batch {}",
        f.by_group,
        f.by_batch
    );
}

#[test]
fn a_batch_topic_fit_is_reproduced_from_its_trained_delta() {
    check_batch_fit("topic", &batch_fit(crate::fit_topic_model));
}

#[test]
fn a_batch_vae_fit_is_reproduced_from_its_trained_delta() {
    check_batch_fit("vae", &batch_fit(crate::vae::fit_vae_model));
}

#[test]
fn a_batch_masked_fit_is_reproduced_from_its_trained_delta() {
    check_batch_fit(
        "masked",
        &batch_fit(crate::masked_topic::fit_masked_topic_model),
    );
}

// Bar: a default (residual) fit rebuilds its encoder null from the held-out
// cells, which one held-out batch cannot give. When that batch is one the
// model was trained on, the encoder reads its trained δ instead, so its
// cells land nearer the fit than with no null at all.
#[test]
fn a_residual_fit_reads_its_trained_delta_for_one_batch() {
    let dir = tempfile::tempdir().unwrap();
    let n = 240;
    let (data, batches) = planted_batched_zarr(dir.path(), n, 2);
    let at = |name: &str| dir.path().join(name).to_string_lossy().into_owned();
    let model = at("model");
    let argv = [
        data.as_str(),
        "-b",
        &batches,
        "-o",
        &model,
        "-t",
        "3",
        "-i",
        "20",
    ];
    let more = ["--minibatch-size", "60", "--seed", "7"];
    crate::fit_topic_model(&parse(&[&argv[..], &more[..]].concat())).unwrap();
    let trained = proportions(&model);
    // Batch B1 carries the planted fold; its cells are the odd ones.
    let b1: Vec<usize> = (1..n).step_by(2).collect();
    let all_b1 = at("b1.batch.txt");
    std::fs::write(&all_b1, "B1\n".repeat(n)).unwrap();
    let to_fit = |out: &str, b: Option<&str>| -> f32 {
        let out = at(out);
        let mut argv = vec![data.as_str(), "--model", &model, "-o", &out];
        if let Some(b) = b {
            argv.extend_from_slice(&["-b", b]);
        }
        predict_model(&parse::<PredictArgs>(&argv)).unwrap();
        mean_tv(
            &proportions(&out).select_rows(&b1),
            &trained.select_rows(&b1),
        )
    };
    let (with_delta, no_null) = (to_fit("one", Some(&all_b1)), to_fit("none", None));
    eprintln!("residual, one batch: with the trained δ {with_delta:.4}, no null {no_null:.4}");
    assert!(with_delta < no_null, "with {with_delta}, without {no_null}");
}
