//! Regression guard for the bge driver: `senna bge` writes its output-file
//! set and a manifest of kind `bge`.

use clap::Parser;
use nalgebra::DMatrix;

use super::ensure_finite;
use crate::bge::{fit_bge, BgeArgs};
use senna::embed_common::candle_core;
use senna::run_manifest::{self, RunKind};
use senna::sparse_output::{write_backend, Entries};

const N_GENES: usize = 12;
const N_CELLS: usize = 40;

/// A tiny synthetic count backend: `GENE1..GENE12` rows × `C1..C40` columns,
/// one batch, every entry a small positive integer so no cell or gene is
/// empty (near-empty cell QC would otherwise drop it before it reaches the
/// fit).
fn synthetic_backend(dir: &std::path::Path) -> Box<str> {
    let path = dir.join("fixture.zarr.zip");
    let path: Box<str> = path.to_string_lossy().into_owned().into();
    let m = DMatrix::<f32>::from_fn(N_GENES, N_CELLS, |g, c| {
        (((g * 7 + c * 11 + 3) % 9) + 1) as f32
    });
    let genes: Vec<Box<str>> = (1..=N_GENES).map(|i| format!("GENE{i}").into()).collect();
    let cells: Vec<Box<str>> = (1..=N_CELLS).map(|i| format!("C{i}").into()).collect();
    write_backend(&path, Entries::Dense(&m), &genes, &cells).expect("create synthetic backend");
    path
}

/// `BgeArgs` is an `Args` group, not a `Parser`; wrap it to parse standalone
/// (matches the pattern already used for other arg-group tests, e.g.
/// `masked_topic_tests.rs`).
#[derive(Parser)]
struct Cli {
    #[command(flatten)]
    args: BgeArgs,
}

fn parse(data: &str, out: &str) -> BgeArgs {
    Cli::try_parse_from([
        "senna-bge",
        data,
        "-o",
        out,
        "--epochs",
        "2",
        "--skip-etm",
        "--no-emit-pb-reference",
        "--embedding-dim",
        "4",
    ])
    .expect("BgeArgs parses")
    .args
}

#[test]
fn bge_writes_its_documented_output_set_and_manifest_kind() {
    let dir = tempfile::tempdir().expect("tempdir");
    let data = synthetic_backend(dir.path());
    let out = dir.path().join("run").to_string_lossy().into_owned();

    let args = parse(&data, &out);
    fit_bge(&args).expect("fit_bge on the synthetic fixture");

    // The exact suffix set `fit_bge` writes for `--skip-etm
    // --no-emit-pb-reference`, confirmed by running this test against the
    // pre-extraction `fit_bge` (see the task report for the RED/GREEN
    // evidence): the plain hier phase 1 composes its dictionary through an
    // internal hard gene partition but never builds the *learned* module
    // layer (`model.modules` stays `None` on this path — see
    // `FitConfig::gene_modules`'s doc), so `write_module_tables` is a no-op
    // regardless of `--feature-modules`; no ETM tables (`--skip-etm`); no
    // `pb_reference.zarr` (`--no-emit-pb-reference`).
    let mut actual: Vec<String> = std::fs::read_dir(dir.path())
        .expect("read the run directory")
        .filter_map(|e| e.ok())
        .map(|e| e.file_name().to_string_lossy().into_owned())
        // Keep only the run's outputs, all named `run.<suffix>`; the input
        // fixture is not.
        .filter(|name| name.starts_with("run."))
        .collect();
    actual.sort();

    let mut expected: Vec<String> = [
        "cell_embedding",
        "cell_bias",
        "feature_embedding",
        "feature_coembedding",
        "feature_bias",
        "pb_embedding",
        "pb_batch",
        // the partition, for `--pb-from` / `--from` and `senna donor-deviation`
        "cell_to_pb",
    ]
    .into_iter()
    .map(|suffix| format!("run.{suffix}.parquet"))
    .collect();
    expected.push("run.cell_encoder.safetensors".into());
    expected.push("run.senna.json".into());
    expected.sort();

    assert_eq!(
        actual, expected,
        "fit_bge's output-file set changed under --skip-etm --no-emit-pb-reference"
    );

    let (manifest, _dir) = run_manifest::load_for(&out).expect("load the manifest back");
    assert_eq!(manifest.kind, RunKind::Bge, "manifest kind must be bge");
    assert_eq!(
        manifest.outputs.cell_to_pb.as_deref(),
        Some("run.cell_to_pb.parquet"),
        "the manifest must name the partition bge wrote"
    );
}

/// A pinned table wider than the data: its unmatched rows come out after the
/// data's genes in `feature_embedding` (ρ), unchanged, with a types table over
/// every row; the co-embed keeps the data's genes.
#[test]
fn bge_carries_the_unmatched_rows_of_a_pinned_table_through() {
    use crate::feature_preset::test_support::{assert_carried, widen};
    use legume_numeric::matrix::traits::IoOps;
    let dir = tempfile::tempdir().expect("tempdir");
    let data = synthetic_backend(dir.path());
    let first = dir.path().join("first").to_string_lossy().into_owned();
    let plus = dir.path().join("plus").to_string_lossy().into_owned();
    let second = dir.path().join("second").to_string_lossy().into_owned();
    fit_bge(&parse(&data, &first)).expect("first run");
    let extra = widen(&format!("{first}.feature_embedding.parquet"), &plus);
    let args = Cli::try_parse_from([
        "senna-bge",
        &data,
        "-o",
        &second,
        "--epochs",
        "2",
        "--skip-etm",
        "--no-emit-pb-reference",
        "--embedding-dim",
        "auto",
        "--freeze-feature-embedding",
        &plus,
    ])
    .expect("BgeArgs parses")
    .args;
    fit_bge(&args).expect("second run");
    assert_carried(
        &second,
        &format!("{second}.feature_embedding.parquet"),
        N_GENES,
        &extra,
    );
    let co =
        DMatrix::<f32>::from_parquet(&format!("{second}.feature_coembedding.parquet")).unwrap();
    assert_eq!(
        co.rows.len(),
        N_GENES,
        "the co-embed stays on the data's genes"
    );
}

/// The device-side probe passes a finite embedding and catches a NaN or an
/// infinity anywhere in it.
#[test]
fn a_non_finite_embedding_is_refused() {
    let dev = candle_core::Device::Cpu;
    let t = |v: Vec<f32>| candle_core::Tensor::from_vec(v, (2, 2), &dev).unwrap();
    assert!(ensure_finite(&t(vec![1.0, -2.0, 3e30, 0.0]), "x").is_ok());
    for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        let err = ensure_finite(&t(vec![1.0, bad, 3.0, 4.0]), "the cell embedding").unwrap_err();
        assert!(err.to_string().contains("the cell embedding"), "{err}");
    }
}

/// bge never builds the marginal sketch-sign partition, so asking for it is
/// refused before any data is read rather than quietly fitting the refined one.
#[test]
fn bge_refuses_the_marginal_pb_tree() {
    let mut args = parse("unused.zarr", "unused");
    args.collapse.pb_tree = crate::refine_weighting::PbTreeArg::Marginal;
    let err = fit_bge(&args).unwrap_err().to_string();
    assert!(err.contains("--pb-tree marginal"), "{err}");
}
