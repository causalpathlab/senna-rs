//! `senna predict` gives the encoder the null it was trained with: on a fit's
//! own cells it reproduces the fit's latent, and it would not without the
//! null.

use crate::fit_cli::{latent, parse};
use crate::planted::planted_batched_zarr;
use crate::predict::{predict_model, PredictArgs};
use senna::embed_common::{softmax_rows_inplace, Mat};
use std::sync::OnceLock;

/// The planted cells.
const N: usize = 240;

/// A fit on the planted batched cells, trained once and read by every test
/// that needs it. Tests only read it: each writes its predictions into a
/// temp dir of its own.
pub(crate) struct Planted {
    _dir: tempfile::TempDir,
    pub data: String,
    pub batches: String,
    pub model: String,
}

/// The shared fit's training flags (data, batches and `-o` are added).
pub(crate) const TOPIC_FIT: [&str; 8] = [
    "-t",
    "3",
    "-i",
    "20",
    "--minibatch-size",
    "60",
    "--seed",
    "7",
];

/// Fit with `fit` on freshly planted batched cells, `extra` after the data,
/// batches and `-o model`.
pub(crate) fn fit_planted(fit: impl FnOnce(&[&str]), extra: &[&str]) -> Planted {
    let dir = tempfile::tempdir().unwrap();
    let (data, batches) = planted_batched_zarr(dir.path(), N, 2);
    let model = dir.path().join("model").to_string_lossy().into_owned();
    let argv = [data.as_str(), "-b", &batches, "-o", &model];
    fit(&[&argv[..], extra].concat());
    Planted {
        _dir: dir,
        data,
        batches,
        model,
    }
}

fn shared<A: clap::Args>(
    cell: &'static OnceLock<Planted>,
    fit: fn(&A) -> anyhow::Result<()>,
) -> &'static Planted {
    cell.get_or_init(|| fit_planted(|argv| fit(&parse::<A>(argv)).unwrap(), &TOPIC_FIT))
}

/// The default (residual) topic fit with [`TOPIC_FIT`], shared.
pub(crate) fn residual_topic() -> &'static Planted {
    static FIT: OnceLock<Planted> = OnceLock::new();
    shared(&FIT, crate::fit_topic_model)
}

/// The default (residual) vae fit with [`TOPIC_FIT`], shared.
pub(crate) fn residual_vae() -> &'static Planted {
    static FIT: OnceLock<Planted> = OnceLock::new();
    shared(&FIT, crate::vae::fit_vae_model)
}

/// The default (residual) masked fit with [`TOPIC_FIT`], shared.
pub(crate) fn residual_masked() -> &'static Planted {
    static FIT: OnceLock<Planted> = OnceLock::new();
    shared(&FIT, crate::masked_topic::fit_masked_topic_model)
}

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

/// Predict `fit`'s own cells into `out`, under batch labels `b` (none: no
/// batches given), and read the prediction as proportions.
fn predict(fit: &Planted, out: &str, b: Option<&str>) -> Mat {
    let mut argv = vec![fit.data.as_str(), "--model", &fit.model, "-o", out];
    if let Some(b) = b {
        argv.extend_from_slice(&["-b", b]);
    }
    predict_model(&parse::<PredictArgs>(&argv)).unwrap();
    proportions(out)
}

// Bar: with its null, predict on the fit's own cells stays within 0.05 mean
// TV of the fit's latent; without the null (no batches given), it does not.
fn check_in_sample(family: &str, fit: &Planted) {
    let dir = tempfile::tempdir().unwrap();
    let at = |name: &str| dir.path().join(name).to_string_lossy().into_owned();
    let trained = proportions(&fit.model);
    let with_null = mean_tv(&predict(fit, &at("with"), Some(&fit.batches)), &trained);
    let without = mean_tv(&predict(fit, &at("without"), None), &trained);
    eprintln!("{family}: with the null {with_null:.4}, without {without:.4}");
    assert!(with_null < 0.05, "{family}: with the null: {with_null}");
    assert!(
        without > with_null,
        "{family}: without {without}, with {with_null}"
    );
}

// Bar: [`check_in_sample`]; and a default (residual) fit rebuilds its encoder
// null from the held-out cells, which one held-out batch cannot give. When
// that batch is one the model was trained on, the encoder reads its trained
// δ instead, so its cells land nearer the fit than with no null at all.
#[test]
fn a_topic_fit_is_reproduced_on_its_own_cells_and_reads_its_trained_delta() {
    let fit = residual_topic();
    check_in_sample("topic", fit);

    let dir = tempfile::tempdir().unwrap();
    let at = |name: &str| dir.path().join(name).to_string_lossy().into_owned();
    let trained = proportions(&fit.model);
    // Batch B1 carries the planted fold; its cells are the odd ones.
    let b1: Vec<usize> = (1..N).step_by(2).collect();
    let all_b1 = at("b1.batch.txt");
    std::fs::write(&all_b1, "B1\n".repeat(N)).unwrap();
    let to_fit = |out: &str, b: Option<&str>| -> f32 {
        mean_tv(
            &predict(fit, &at(out), b).select_rows(&b1),
            &trained.select_rows(&b1),
        )
    };
    let (with_delta, no_null) = (to_fit("one", Some(&all_b1)), to_fit("none", None));
    eprintln!("residual, one batch: with the trained δ {with_delta:.4}, no null {no_null:.4}");
    assert!(with_delta < no_null, "with {with_delta}, without {no_null}");
}

#[test]
fn a_vae_fit_is_reproduced_on_its_own_cells() {
    check_in_sample("vae", residual_vae());
}

#[test]
fn a_masked_topic_fit_is_reproduced_on_its_own_cells() {
    check_in_sample("masked", residual_masked());
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
    let argv = [&TOPIC_FIT[..], &["--adj-method", "batch"]].concat();
    let f = fit_planted(|argv| fit(&parse::<A>(argv)).unwrap(), &argv);
    let dir = tempfile::tempdir().unwrap();
    let at = |name: &str| dir.path().join(name).to_string_lossy().into_owned();
    let trained = proportions(&f.model);

    let renamed = at("renamed.batch.txt");
    let labels: String = (0..N).map(|c| format!("X{}\n", c % 2)).collect();
    std::fs::write(&renamed, labels).unwrap();
    let known = predict(&f, &at("known"), Some(&f.batches));
    let unknown = predict(&f, &at("unknown"), Some(&renamed));
    let all_b0 = at("b0.batch.txt");
    std::fs::write(&all_b0, "B0\n".repeat(N)).unwrap();
    let one = predict(&f, &at("one"), Some(&all_b0));
    let b0: Vec<usize> = (0..N).step_by(2).collect();
    let one_batch = mean_tv(&one.select_rows(&b0), &trained.select_rows(&b0));

    // Mean latent of the cells of group `g` in batch `b`.
    let mean_of = |g: usize, b: usize| -> Mat {
        let rows: Vec<usize> = (0..N)
            .filter(|c| usize::from(*c >= N / 2) == g && c % 2 == b)
            .collect();
        let m = trained.select_rows(&rows);
        Mat::from_rows(&[m.row_sum() / rows.len() as f32])
    };
    let tv = |a: &Mat, b: &Mat| (a - b).abs().sum() * 0.5;
    let by_batch = (tv(&mean_of(0, 0), &mean_of(0, 1)) + tv(&mean_of(1, 0), &mean_of(1, 1))) / 2.0;
    let by_group = (tv(&mean_of(0, 0), &mean_of(1, 0)) + tv(&mean_of(0, 1), &mean_of(1, 1))) / 2.0;
    let meta: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(format!("{}.model.json", f.model)).unwrap())
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
