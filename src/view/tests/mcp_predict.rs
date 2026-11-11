use super::*;
use crate::view::mcp::jobs::senna_parses;
use senna::run_manifest::RunKind;
use std::path::PathBuf;

fn predict(v: Value) -> Result<Task> {
    predict_task(&parse(v).unwrap(), "p")
}

/// `v` with the fields of `extra` set too.
fn with(mut v: Value, extra: Value) -> Value {
    v.as_object_mut()
        .unwrap()
        .extend(extra.as_object().unwrap().clone());
    v
}

#[test]
fn predict_runs_senna_predict_and_makes_the_out_s_latent() {
    let t = predict(
        json!({"model": "ref", "data": ["a.zarr", "b.zarr"], "args": ["--refine-steps", "5"]}),
    )
    .unwrap();
    assert_eq!(
        t.argv,
        [
            "predict",
            "a.zarr",
            "b.zarr",
            "--model",
            "ref",
            "-o",
            "p",
            "--refine-steps",
            "5"
        ]
    );
    assert_eq!(
        t.made,
        [
            ("latent", PathBuf::from("p.latent.parquet")),
            ("predictive", PathBuf::from("p.predictive.parquet"))
        ]
    );
}

#[test]
fn predict_gives_each_option_its_flag_and_senna_reads_the_line() {
    let t = predict(json!({
        "model": "fits/m.senna.json",
        "data": ["t1.zarr", "t2.zarr"],
        "batch_files": ["b1.txt", "b2.txt"],
        "null_from": ["train.zarr"],
        "eval_features": "panel.txt",
        "latent": "refine",
        "refine_steps": 20,
    }))
    .unwrap();
    assert_eq!(
        t.argv,
        [
            "predict",
            "t1.zarr",
            "t2.zarr",
            "--model",
            "fits/m",
            "-o",
            "p",
            "-b",
            "b1.txt",
            "-b",
            "b2.txt",
            "--null-from",
            "train.zarr",
            "--eval-features",
            "panel.txt",
            "--refine-steps",
            "20"
        ]
    );
    senna_parses(&t.argv).unwrap();

    let t = predict(json!({"model": "m", "data": ["t.zarr"], "latent": "decoder"})).unwrap();
    assert_eq!(t.argv[6..], ["--decoder-only"]);
    senna_parses(&t.argv).unwrap();

    let t = predict(json!({
        "model": "m", "bulk": ["c1.parquet", "c2.tsv"], "null_from": ["a.zarr", "b.zarr"],
    }))
    .unwrap();
    assert_eq!(
        t.argv,
        [
            "predict",
            "--bulk",
            "c1.parquet",
            "c2.tsv",
            "--model",
            "m",
            "-o",
            "p",
            "--null-from",
            "a.zarr",
            "b.zarr"
        ]
    );
    senna_parses(&t.argv).unwrap();
}

#[test]
fn predict_checks_its_flags_before_anything_runs() {
    let m = |extra: Value| with(json!({"model": "m", "data": ["t.zarr"]}), extra);
    for bad in [
        json!({"model": "m"}),
        m(json!({"bulk": ["c.parquet"]})),
        m(json!({"latent": "refine"})),
        m(json!({"refine_steps": 5})),
        m(json!({"batch_files": ["b1.txt", "b2.txt"]})),
    ] {
        assert!(predict(bad.clone()).is_err(), "{bad}");
    }
    let decoder: PredictCall = parse(m(json!({"latent": "decoder"}))).unwrap();
    for kind in [
        RunKind::Svd,
        RunKind::Bge,
        RunKind::Simba,
        RunKind::Vae,
        RunKind::Itopic,
    ] {
        let e = check_latent(&decoder, kind).unwrap_err().to_string();
        assert!(e.contains("encoder"), "{kind}: {e}");
    }
    check_latent(&decoder, RunKind::Topic).unwrap();
    assert!(parse::<PredictCall>(m(json!({"delta_iters": 0}))).is_err());
}

/// A predictive table as `senna predict` writes it, with or without the
/// cross-family columns.
fn write_predictive(path: &Path, eval: bool) {
    use legume_numeric::matrix::traits::IoOps;
    let cells: Vec<Box<str>> = ["c1", "c2", "c3"].map(Into::into).to_vec();
    let mut cols: Vec<Box<str>> = vec!["llik".into(), "total".into(), "llik_per_count".into()];
    let mut rows = [
        vec![-10.0, 5.0, -2.0],
        vec![-20.0, 10.0, -2.0],
        vec![0.0, 0.0, f32::NAN],
    ];
    if eval {
        cols.extend([
            "eval_llik_per_count".into(),
            "eval_null_llik_per_count".into(),
        ]);
        rows[0].extend([-1.0, -3.0]);
        rows[1].extend([-2.0, -4.0]);
        rows[2].extend([f32::NAN, f32::NAN]);
    }
    let m = senna::embed_common::Mat::from_fn(3, cols.len(), |i, j| rows[i][j]);
    m.to_parquet_with_names(
        path.to_str().unwrap(),
        (Some(&cells), Some("cell")),
        Some(&cols),
    )
    .unwrap();
}

#[test]
fn a_prediction_is_summed_up_by_its_held_out_and_null_fit() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("p.predictive.parquet");
    write_predictive(&path, true);
    let s = predict_summary(&path).unwrap();
    assert_eq!(s["cells"], 3);
    assert_eq!(s["mean_eval_llik_per_count"], -1.5);
    assert_eq!(s["mean_eval_null_llik_per_count"], -3.5);

    write_predictive(&path, false);
    let s = predict_summary(&path).unwrap();
    assert!(s.get("mean_eval_llik_per_count").is_none());
    assert!(s["note"].as_str().unwrap().contains("eval_llik_per_count"));
}

#[test]
fn probe_runs_senna_probe_and_makes_its_verdict() {
    let a: ProbeCall = parse(json!({
        "model": "m.senna.json",
        "data": ["q1.zarr", "q2.zarr"],
        "calibration": "ref.zarr",
        "alpha": 0.01,
        "counterfactual": true,
        "args": ["--cf-seed", "7"],
    }))
    .unwrap();
    let t = probe_task(&a, "pr");
    assert_eq!(
        t.argv,
        [
            "probe",
            "q1.zarr",
            "q2.zarr",
            "--model",
            "m",
            "--calibration",
            "ref.zarr",
            "-o",
            "pr",
            "--alpha",
            "0.01",
            "--counterfactual",
            "--cf-seed",
            "7"
        ]
    );
    senna_parses(&t.argv).unwrap();
    assert_eq!(
        t.made,
        [
            ("verdict", PathBuf::from("pr.probe.json")),
            ("cells", PathBuf::from("pr.probe.tsv"))
        ]
    );
    // The counterfactual is the masked models' only.
    for kind in [RunKind::Topic, RunKind::Vae, RunKind::Bge, RunKind::Simba] {
        assert!(check_probe(&a, kind).is_err(), "{kind}");
    }
    check_probe(&a, RunKind::Itopic).unwrap();
    check_probe(&a, RunKind::MaskedVae).unwrap();
}

#[test]
fn a_probe_s_verdict_is_read_from_the_file_it_writes() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("pr.probe.json");
    let written = json!({
        "verdict": "novel", "novel": true, "flagged": 40, "cells": 100,
        "counterfactual": {"benefit": 0.5, "forgetting": 0.1},
    });
    std::fs::write(&path, written.to_string()).unwrap();
    assert_eq!(probe_summary(&path).unwrap(), written);
}
