//! End-to-end guards for `run_tde` on a tiny synthetic spliced + unspliced
//! axis. GENE1..GENE3 carry both count channels; GENE4 only `spliced`, so it
//! sits on the gene axis but outside the unspliced support; GENE5 only
//! `unspliced`, so it has no place in the spliced space and is left out. A tiny fit
//! (`--epochs 2 --phase1-cells-per-pb 0 --embedding-dim 4`) must write bge's
//! own output set on gene names, plus the divergence tables, under a `tde`
//! manifest.

use super::run_tde;
use crate::tde::args::TdeArgs;
use crate::tde::test_fixtures::{boxes, synth};
use clap::Parser;
use legume_numeric::matrix::traits::IoOps;
use senna::embed_common::*;
use senna::run_manifest::{self, RunKind};

#[derive(Parser)]
struct Cli {
    #[command(flatten)]
    args: TdeArgs,
}

const CELLS: [&str; 12] = [
    "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9", "C10", "C11", "C12",
];

const ROWS: [&str; 8] = [
    "GENE1/count/spliced",
    "GENE1/count/unspliced",
    "GENE2/count/spliced",
    "GENE2/count/unspliced",
    "GENE3/count/spliced",
    "GENE3/count/unspliced",
    "GENE4/count/spliced",
    "GENE5/count/unspliced",
];

fn tiny_fit(genes: &str, out: &str, extra: &[&str]) -> TdeArgs {
    let mut argv = vec![
        "senna-tde",
        genes,
        "--epochs",
        "2",
        "--skip-etm",
        "--no-emit-pb-reference",
        "--phase1-cells-per-pb",
        "0",
        "--divergence-epochs",
        "2",
        "-o",
        out,
    ];
    argv.extend_from_slice(extra);
    Cli::try_parse_from(argv).expect("TdeArgs parses").args
}

/// The output-file suffixes `fit_embed_family` writes on the plain (no ETM,
/// no pb reference) path, shared with `bge::driver::tests`'s own guard.
const BGE_PARQUET_SUFFIXES: [&str; 7] = [
    "cell_embedding",
    "cell_bias",
    "feature_embedding",
    "feature_coembedding",
    "feature_bias",
    "pb_embedding",
    "pb_batch",
];

#[test]
fn tde_writes_bges_outputs_on_genes_plus_the_divergence_tables() {
    let dir = tempfile::tempdir().expect("tempdir");
    let genes = synth(dir.path(), "S1_count", &ROWS, &CELLS);
    let out = dir.path().join("run").to_string_lossy().into_owned();
    run_tde(&tiny_fit(&genes, &out, &["--embedding-dim", "4"])).expect("a tde run");

    let mut actual: Vec<String> = std::fs::read_dir(dir.path())
        .expect("read the run directory")
        .filter_map(|e| e.ok())
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .filter(|name| name.starts_with("run."))
        .collect();
    actual.sort();
    let mut expected: Vec<String> = BGE_PARQUET_SUFFIXES
        .into_iter()
        .chain(["cell_velocity", "feature_divergence", "divergence_loading"])
        .map(|suffix| format!("run.{suffix}.parquet"))
        .collect();
    expected.extend([
        "run.cell_encoder.safetensors".into(),
        "run.senna.json".into(),
    ]);
    expected.sort();
    assert_eq!(actual, expected);

    let (manifest, _dir) = run_manifest::load_for(&out).expect("load the manifest back");
    assert_eq!(manifest.kind, RunKind::Tde);
    let slots = manifest.outputs.divergence.expect("divergence slots");
    assert_eq!(slots.track, "count/unspliced");
    assert!(slots.cell.ends_with("run.cell_velocity.parquet"));
    assert!(slots.loading.ends_with("run.divergence_loading.parquet"));

    // The gene table is on gene names, the spliced rows only.
    let rho = Mat::from_parquet(&format!("{out}.feature_embedding.parquet")).unwrap();
    assert_eq!(rho.rows, boxes(&["GENE1", "GENE2", "GENE3", "GENE4"]));

    // Per gene: the unspliced support only (GENE4 has no unspliced row).
    let feat = Mat::from_parquet(&format!("{out}.feature_divergence.parquet")).unwrap();
    assert_eq!(feat.rows, boxes(&["GENE1", "GENE2", "GENE3"]));
    assert_eq!(feat.cols, boxes(&["ratio", "steady_anchor", "log_gamma"]));
    assert!(feat.mat.iter().all(|x| x.is_finite()));

    // Per gene: its direction in the cell space.
    let loading = Mat::from_parquet(&format!("{out}.divergence_loading.parquet")).unwrap();
    assert_eq!(loading.rows, boxes(&["GENE1", "GENE2", "GENE3"]));
    assert_eq!(loading.cols, boxes(&["h0", "h1", "h2", "h3"]));

    // Per cell: its velocity in the cell space, and κ; the same cells, in the
    // same order, as the cell embedding.
    let cell = Mat::from_parquet(&format!("{out}.cell_velocity.parquet")).unwrap();
    let z = Mat::from_parquet(&format!("{out}.cell_embedding.parquet")).unwrap();
    assert_eq!(cell.rows, z.rows);
    assert_eq!(cell.cols, boxes(&["h0", "h1", "h2", "h3", "kappa"]));
    assert!(cell.mat.iter().all(|x| x.is_finite()));
}

#[test]
fn an_input_without_unspliced_rows_is_refused() {
    let dir = tempfile::tempdir().expect("tempdir");
    let genes = synth(
        dir.path(),
        "S1_count",
        &["GENE1/count/spliced", "GENE2/count/spliced"],
        &CELLS,
    );
    let out = dir.path().join("run").to_string_lossy().into_owned();
    let e = run_tde(&tiny_fit(&genes, &out, &[]))
        .unwrap_err()
        .to_string();
    assert!(e.contains("count/unspliced"), "{e}");
}

#[test]
fn modality_rows_are_refused() {
    let dir = tempfile::tempdir().expect("tempdir");
    let genes = synth(
        dir.path(),
        "S1_count",
        &[
            "GENE1/count/spliced",
            "GENE1/count/unspliced",
            "GENE1/m6a/methylated",
            "GENE1/m6a/unmethylated",
        ],
        &CELLS,
    );
    let out = dir.path().join("run").to_string_lossy().into_owned();
    let e = run_tde(&tiny_fit(&genes, &out, &[]))
        .unwrap_err()
        .to_string();
    assert!(e.contains("m6a"), "{e}");
}

#[test]
fn pb_reference_flags_are_refused() {
    let dir = tempfile::tempdir().expect("tempdir");
    let genes = synth(dir.path(), "S1_count", &ROWS, &CELLS);
    let out = dir.path().join("run").to_string_lossy().into_owned();
    let refused = run_tde(&tiny_fit(&genes, &out, &["--mixture-batch", "b1"]));
    assert!(refused.is_err(), "--mixture-batch must be refused");
}

/// A gene table as the frozen base: the matched genes' rows come out
/// verbatim, `--embedding-dim auto` takes the table's width, and a row the
/// data lacks is carried through.
#[test]
fn a_frozen_gene_table_pins_the_genes() {
    let dir = tempfile::tempdir().expect("tempdir");
    let genes = synth(dir.path(), "S1_count", &ROWS, &CELLS);
    let h = 4;
    let table = Mat::from_fn(5, h, |i, k| (i as f32 + 1.0) * 0.25 - k as f32 * 0.1);
    let names = ["GENE1", "GENE2", "GENE3", "GENE4", "EXTRA1"];
    let given = dir.path().join("given").to_string_lossy().into_owned();
    table
        .to_parquet_with_names(
            &format!("{given}.feature_embedding.parquet"),
            (Some(&boxes(&names)), Some("gene")),
            None,
        )
        .unwrap();
    let out = dir.path().join("run").to_string_lossy().into_owned();
    run_tde(&tiny_fit(
        &genes,
        &out,
        &[
            "--embedding-dim",
            "auto",
            "--freeze-feature-embedding",
            &given,
        ],
    ))
    .expect("a tde run on a frozen table");
    let rho = Mat::from_parquet(&format!("{out}.feature_embedding.parquet")).unwrap();
    assert_eq!(rho.mat.ncols(), h, "auto takes the table's width");
    for (i, g) in names.iter().enumerate() {
        let r = rho
            .rows
            .iter()
            .position(|n| n.as_ref() == *g)
            .unwrap_or_else(|| panic!("no row {g} among {:?}", rho.rows));
        let got: Vec<f32> = rho.mat.row(r).iter().copied().collect();
        let want: Vec<f32> = table.row(i).iter().copied().collect();
        assert_eq!(got, want, "{g}");
    }
}
