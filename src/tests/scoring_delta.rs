//! `senna predict --scoring-delta`: which δ the held-out score is taken
//! under, end to end on a fit with a planted batch effect.

use crate::fit_cli::parse;
use crate::planted::planted_batched_zarr;
use crate::predict::{predict_model, PredictArgs};
use legume_numeric::matrix::traits::IoOps;
use senna::embed_common::Mat;

/// Mean of a column of `{prefix}.predictive.parquet` over its finite rows.
fn mean_of(prefix: &str, column: &str) -> f32 {
    let t = Mat::from_parquet_with_row_names(&format!("{prefix}.predictive.parquet"), Some(0))
        .expect("predictive");
    let j = t
        .cols
        .iter()
        .position(|c| &**c == column)
        .unwrap_or_else(|| panic!("no column {column}"));
    let v: Vec<f32> = t
        .mat
        .column(j)
        .iter()
        .copied()
        .filter(|x| x.is_finite())
        .collect();
    v.iter().sum::<f32>() / v.len() as f32
}

fn fit(data: &str, batches: Option<&str>, model: &str) {
    let mut argv = vec![data, "-o", model, "-t", "3", "-i", "20"];
    argv.extend_from_slice(&["--minibatch-size", "60", "--seed", "7"]);
    if let Some(b) = batches {
        argv.extend_from_slice(&["-b", b]);
    } else {
        argv.push("--ignore-batch");
    }
    crate::fit_topic_model(&parse(&argv)).unwrap();
}

// Bar: `auto` is `train` on the training batches and a cross-fit on batches
// the model never saw; the in-sample δ scores at least as well as the
// cross-fitted one; `train` refuses an unseen batch; and a model trained
// without batches is scored with no δ, whatever batches the query brings.
#[test]
fn the_scoring_delta_is_the_one_asked_for() {
    let dir = tempfile::tempdir().unwrap();
    let (data, batches) = planted_batched_zarr(dir.path(), 240, 2);
    let at = |name: &str| dir.path().join(name).to_string_lossy().into_owned();
    let renamed = at("renamed.batch.txt");
    let labels: String = (0..240).map(|c| format!("X{}\n", c % 2)).collect();
    std::fs::write(&renamed, labels).unwrap();

    let model = at("model");
    fit(&data, Some(&batches), &model);
    let predict_with =
        |model: &str, out: &str, mode: &str, b: &str, extra: &[&str]| -> anyhow::Result<f32> {
            let out = at(out);
            let argv = [data.as_str(), "--model", model, "-o", &out, "-b", b];
            let argv = [&argv[..], &["--scoring-delta", mode], extra].concat();
            predict_model(&parse::<PredictArgs>(&argv))?;
            let llik = mean_of(&out, "eval_llik_per_count");
            let null = mean_of(&out, "eval_null_llik_per_count");
            assert!(
                llik.is_finite() && null.is_finite(),
                "{mode}: {llik} {null}"
            );
            Ok(llik)
        };

    let predict =
        |model: &str, out: &str, mode: &str, b: &str| predict_with(model, out, mode, b, &[]);

    let train = predict(&model, "train", "train", &batches).unwrap();
    let auto = predict(&model, "auto", "auto", &batches).unwrap();
    assert_eq!(train, auto, "auto on training batches is the trained δ");

    let in_sample = predict(&model, "heldout", "heldout", &batches).unwrap();
    let crossfit = predict(&model, "crossfit", "crossfit", &batches).unwrap();
    let none = predict(&model, "none", "none", &batches).unwrap();
    eprintln!(
        "train {train:.4}, in-sample {in_sample:.4}, cross-fitted {crossfit:.4}, none {none:.4}"
    );
    assert!(
        in_sample > crossfit + 2e-4,
        "in-sample {in_sample} vs cross-fitted {crossfit}"
    );
    assert!(
        crossfit > none,
        "a 3x batch effect: δ must help ({crossfit} vs {none})"
    );
    assert!(train > none, "the trained δ must help ({train} vs {none})");

    // Unseen batch names: `auto` cross-fits them, `train` refuses.
    let auto_new = predict(&model, "auto_new", "auto", &renamed).unwrap();
    // Not bit-equal: the block sums are added in thread order.
    assert!(
        (auto_new - crossfit).abs() < 1e-4,
        "auto on unseen batches is the cross-fit: {auto_new} vs {crossfit}"
    );
    let err = predict(&model, "train_new", "train", &renamed).unwrap_err();
    assert!(err.to_string().contains("not trained on"), "{err}");

    // Hidden from the encoder and scored on: the genes that carry the batch
    // effect. Their δ is read from the counts, not through the encoder's view
    // (where they are all zero and δ cancels), so it still helps.
    let hidden = at("hidden.txt");
    let names: String = (0..60).step_by(3).map(|g| format!("GENE{g}\n")).collect();
    std::fs::write(&hidden, names).unwrap();
    let ablate = ["--ablate-features", hidden.as_str()];
    let hid_fit = predict_with(&model, "hid_fit", "crossfit", &batches, &ablate).unwrap();
    let hid_none = predict_with(&model, "hid_none", "none", &batches, &ablate).unwrap();
    assert!(
        (hid_fit - hid_none).abs() > 1e-3,
        "hidden genes are scored with no δ: {hid_fit} vs {hid_none}"
    );

    // A model trained without batches has no batch term to score under.
    let plain = at("plain");
    fit(&data, None, &plain);
    let plain_auto = predict(&plain, "plain_auto", "auto", &batches).unwrap();
    let plain_none = predict(&plain, "plain_none", "none", &batches).unwrap();
    assert_eq!(plain_auto, plain_none);
}

/// The held-out score of one fit under each way of getting δ.
struct Modes {
    train: f32,
    auto: f32,
    in_sample: f32,
    crossfit: f32,
    none: f32,
}

/// Fit with `fit_argv` (data, batches and `-o model` are added) and score the
/// same cells under every mode.
fn modes(family: &str, fit: impl Fn(&[&str]), fit_argv: &[&str]) -> Modes {
    let dir = tempfile::tempdir().unwrap();
    let (data, batches) = planted_batched_zarr(dir.path(), 240, 2);
    let at = |name: &str| dir.path().join(name).to_string_lossy().into_owned();
    let model = at("model");
    let argv = [data.as_str(), "-b", &batches, "-o", &model];
    fit(&[&argv[..], fit_argv].concat());
    let predict = |mode: &str| -> f32 {
        let out = at(mode);
        let argv = [data.as_str(), "--model", &model, "-o", &out, "-b", &batches];
        let argv = [&argv[..], &["--scoring-delta", mode]].concat();
        predict_model(&parse::<PredictArgs>(&argv)).unwrap();
        mean_of(&out, "eval_llik_per_count")
    };
    let m = Modes {
        train: predict("train"),
        auto: predict("auto"),
        in_sample: predict("heldout"),
        crossfit: predict("crossfit"),
        none: predict("none"),
    };
    eprintln!(
        "{family}: train {:.4}, in-sample {:.4}, cross-fitted {:.4}, none {:.4}",
        m.train, m.in_sample, m.crossfit, m.none
    );
    m
}

// Bar: every family that predicts a rate takes its scoring δ the same way.
// `auto` on training batches is the trained δ; the in-sample δ scores above
// the cross-fitted one; and a δ moves the score, so it is applied.
fn check_modes(family: &str, m: &Modes) {
    assert_eq!(m.train, m.auto, "{family}: auto is the trained δ");
    assert!(
        m.in_sample > m.crossfit,
        "{family}: in-sample {} vs cross-fitted {}",
        m.in_sample,
        m.crossfit
    );
    for (name, v) in [("trained", m.train), ("cross-fitted", m.crossfit)] {
        assert!(
            (v - m.none).abs() > 1e-4,
            "{family}: the {name} δ did not move the score ({v} vs {})",
            m.none
        );
    }
}

const TOPIC_FIT: [&str; 8] = [
    "-t",
    "3",
    "-i",
    "20",
    "--minibatch-size",
    "60",
    "--seed",
    "7",
];

#[test]
fn a_masked_fit_is_scored_under_the_delta_asked_for() {
    let fit = |argv: &[&str]| crate::masked_topic::fit_masked_topic_model(&parse(argv)).unwrap();
    check_modes("masked", &modes("masked", fit, &TOPIC_FIT));
}

#[test]
fn a_vae_fit_is_scored_under_the_delta_asked_for() {
    let fit = |argv: &[&str]| crate::vae::fit_vae_model(&parse(argv)).unwrap();
    check_modes("vae", &modes("vae", fit, &TOPIC_FIT));
}

#[test]
fn a_bge_fit_is_scored_under_the_delta_asked_for() {
    let fit = |argv: &[&str]| crate::bge::fit_bge(&parse(argv)).unwrap();
    let argv = ["--epochs", "2", "--skip-etm", "--no-emit-pb-reference"];
    let m = modes("bge", fit, &[&argv[..], &["--embedding-dim", "4"]].concat());
    check_modes("bge", &m);
}

// Bar: a bge run divides each batch's training counts by a per-batch gene
// fold; predict divides the held-out counts of those batches by the same
// fold before projecting them, so the two batches of one group land closer
// together than when the batches are not given.
#[test]
fn a_bge_projection_adjusts_cells_by_the_trained_fold() {
    let dir = tempfile::tempdir().unwrap();
    let n = 240;
    let (data, batches) = planted_batched_zarr(dir.path(), n, 2);
    let at = |name: &str| dir.path().join(name).to_string_lossy().into_owned();
    let model = at("model");
    let argv = [data.as_str(), "-b", &batches, "-o", &model, "--epochs", "2"];
    let more = [
        "--skip-etm",
        "--no-emit-pb-reference",
        "--embedding-dim",
        "4",
    ];
    crate::bge::fit_bge(&parse(&[&argv[..], &more[..]].concat())).unwrap();
    // Mean distance between the two batches' mean latent, within a group.
    let batch_gap = |out: &str, b: Option<&str>| -> f32 {
        let out = at(out);
        let mut argv = vec![data.as_str(), "--model", &model, "-o", &out];
        if let Some(b) = b {
            argv.extend_from_slice(&["-b", b]);
        }
        predict_model(&parse::<PredictArgs>(&argv)).unwrap();
        let z = crate::fit_cli::latent(&out);
        let mean_of = |g: usize, b: usize| {
            let rows: Vec<usize> = (0..n)
                .filter(|c| usize::from(*c >= n / 2) == g && c % 2 == b)
                .collect();
            z.select_rows(&rows).row_sum() / rows.len() as f32
        };
        ((mean_of(0, 0) - mean_of(0, 1)).norm() + (mean_of(1, 0) - mean_of(1, 1)).norm()) / 2.0
    };
    let (with, without) = (
        batch_gap("with", Some(&batches)),
        batch_gap("without", None),
    );
    eprintln!("bge: batch gap with the trained fold {with:.4}, without {without:.4}");
    assert!(with < without, "with {with}, without {without}");

    // One batch known, one the run never saw: the known batch's cells are
    // still divided by its fold, so they land where they did when both were
    // known; the adjustment is per batch, not all or nothing.
    let mixed = at("mixed.batch.txt");
    let labels: String = (0..n)
        .map(|c| if c % 2 == 0 { "B0\n" } else { "NEW\n" })
        .collect();
    std::fs::write(&mixed, labels).unwrap();
    let out = at("mixed");
    let argv = [data.as_str(), "--model", &model, "-o", &out, "-b", &mixed];
    predict_model(&parse::<PredictArgs>(&argv)).unwrap();
    let b0: Vec<usize> = (0..n).step_by(2).collect();
    let rows = |name: &str| crate::fit_cli::latent(&at(name)).select_rows(&b0);
    let (got, known, raw) = (rows("mixed"), rows("with"), rows("without"));
    let (to_known, to_raw) = ((&got - &known).norm(), (&got - &raw).norm());
    eprintln!("bge: mixed batches, B0 cells to adjusted {to_known:.4}, to unadjusted {to_raw:.4}");
    assert!(
        to_known < to_raw,
        "to adjusted {to_known}, to unadjusted {to_raw}"
    );
}
