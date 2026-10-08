use super::{load_tde_data, sample_ids};
use data_beans::sparse_io::{create_sparse_from_dmatrix, SparseIoBackend};
use nalgebra::DMatrix;

fn boxes(names: &[&str]) -> Vec<Box<str>> {
    names.iter().map(|&s| s.into()).collect()
}

/// Write a tiny synthetic zarr backend at `dir/{stem}.zarr` with the given
/// row and column names, one small positive integer per cell (so nothing is
/// near-empty). Mirrors `bge::driver::tests::synthetic_backend`.
fn synth(dir: &std::path::Path, stem: &str, rows: &[&str], cols: &[&str]) -> Box<str> {
    let path = dir.join(format!("{stem}.zarr"));
    let path: Box<str> = path.to_string_lossy().into_owned().into();
    let m = DMatrix::<f32>::from_fn(rows.len(), cols.len(), |r, c| {
        (((r * 7 + c * 11 + 3) % 9) + 1) as f32
    });
    let mut b = create_sparse_from_dmatrix(&m, Some(&path), Some(&SparseIoBackend::Zarr))
        .expect("create synthetic backend");
    b.register_row_names_vec(&boxes(rows));
    b.register_column_names_vec(&boxes(cols));
    path
}

#[test]
fn sample_ids_strip_the_default_suffix() {
    let dir = tempfile::tempdir().expect("tempdir");
    let a = synth(
        dir.path(),
        "s1_count",
        &["GENE1/count/spliced"],
        &["C1", "C2"],
    );
    let b = synth(
        dir.path(),
        "s2_count",
        &["GENE1/count/spliced"],
        &["C1", "C2"],
    );
    let ids = sample_ids(&[a, b], "").expect("sample_ids");
    assert_eq!(ids, vec![Box::from("s1"), Box::from("s2")]);
}

#[test]
fn sample_ids_honour_an_explicit_strip() {
    let dir = tempfile::tempdir().expect("tempdir");
    let genes = synth(
        dir.path(),
        "sample1_batch",
        &["GENE1/count/spliced"],
        &["C1"],
    );
    let ids = sample_ids(&[genes], "_batch").expect("sample_ids");
    assert_eq!(&*ids[0], "sample1");
}

#[test]
fn no_gene_files_is_an_error() {
    let err = sample_ids(&[], "").unwrap_err().to_string();
    assert!(err.contains("no gene matrices"), "{err}");
}

const CELLS: [&str; 6] = ["C1", "C2", "C3", "C4", "C5", "C6"];
const ROWS: [&str; 3] = [
    "GENE1/count/spliced",
    "GENE1/count/unspliced",
    "GENE2/count/spliced",
];

#[test]
fn two_samples_load_onto_one_axis_with_tagged_barcodes() {
    let dir = tempfile::tempdir().expect("tempdir");
    let a = synth(dir.path(), "S1_count", &ROWS, &CELLS);
    let b = synth(dir.path(), "S2_count", &ROWS, &CELLS[..3]);
    let files = [a, b];
    let ids = sample_ids(&files, "").expect("sample_ids");
    let (unified, plan) = load_tde_data(&files, &ids, None, false).expect("load_tde_data");

    assert_eq!(unified.n_features(), 3, "the row itself is the join key");
    assert_eq!(
        unified.n_cells(),
        9,
        "cells of different samples stay apart"
    );
    assert!(unified.barcodes.iter().any(|b| b.ends_with("@S1")));
    assert!(unified.barcodes.iter().any(|b| b.ends_with("@S2")));
    let pairs = plan.pair_rows().expect("pair_rows");
    assert_eq!(pairs.spliced, vec![0, 2]);
    assert_eq!(pairs.unspliced, vec![Some(1), None]);
}

/// `per_file_barcode_suffix` (the `@sample` tagging) must not be skipped just
/// because `--batch-files` was also given: under `Union` alignment the batch
/// file names one label per merged cell, not per input file, so the two are
/// independent and both apply together.
#[test]
fn sample_tagging_still_applies_with_explicit_batch_files() {
    let dir = tempfile::tempdir().expect("tempdir");
    let a = synth(dir.path(), "S1_count", &ROWS, &CELLS[..3]);
    let b = synth(dir.path(), "S2_count", &ROWS, &CELLS[3..]);
    let batch_path = dir.path().join("batches.txt");
    std::fs::write(&batch_path, "b0\nb0\nb0\nb1\nb1\nb1\n").expect("write batch file");
    let batch_path: Box<str> = batch_path.to_string_lossy().into_owned().into();
    let batch_files = [batch_path];

    let files = [a, b];
    let ids = sample_ids(&files, "").expect("sample_ids");
    let (unified, _plan) = load_tde_data(&files, &ids, Some(&batch_files), false)
        .expect("load_tde_data with --batch-files");

    assert_eq!(unified.n_cells(), 6);
    assert!(unified.barcodes.iter().all(|b| b.contains('@')));
    assert_eq!(unified.n_batches(), 2, "both batch labels round-trip");
}

/// `path`'s backend with `sample` recorded in its metadata, as faba does.
fn with_sample(path: &str, sample: &str) {
    use data_beans::sparse_io::{meta, open_sparse_matrix_by_path};
    let mut b = open_sparse_matrix_by_path(path).expect("open backend");
    b.set_meta(meta::SAMPLE, sample).expect("set sample");
}

#[test]
fn sample_ids_take_the_sample_recorded_in_each_file() {
    let dir = tempfile::tempdir().expect("tempdir");
    // A name that follows no convention: the metadata alone names it.
    let genes = synth(dir.path(), "lane3", &["GENE1/count/spliced"], &["C1", "C2"]);
    with_sample(&genes, "s1");
    let ids = sample_ids(std::slice::from_ref(&genes), "").expect("sample_ids");
    assert_eq!(&*ids[0], "s1");
    // An explicit strip still wins over the metadata.
    let ids = sample_ids(&[genes], "3").expect("sample_ids");
    assert_eq!(&*ids[0], "lane");
}
