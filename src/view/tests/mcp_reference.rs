use super::*;
use crate::view::mcp::jobs::senna_parses;
use std::path::PathBuf;

#[test]
fn impute_pools_the_reference_s_counts_into_the_out() {
    let a: ImputeCall = parse(json!({
        "model": "m.senna.json",
        "data": ["panel.zarr"],
        "reference": "atlas.senna.json",
        "batch_files": ["b.txt"],
        "knn": 30,
    }))
    .unwrap();
    let t = impute_task(&a, "i").unwrap();
    assert_eq!(
        t.argv,
        [
            "impute",
            "panel.zarr",
            "--model",
            "m",
            "-o",
            "i",
            "--reference",
            "atlas.senna.json",
            "-b",
            "b.txt",
            "--knn",
            "30"
        ]
    );
    senna_parses(&t.argv).unwrap();
    assert_eq!(t.made, [("imputed", PathBuf::from("i.imputed.parquet"))]);
    let none: ImputeCall = parse(json!({"model": "m", "data": []})).unwrap();
    assert!(impute_task(&none, "i").is_err());
}

#[test]
fn deconvolve_splits_the_bulk_and_names_its_fractions_first() {
    let a: DeconvolveCall = parse(json!({
        "from": "r.senna.json",
        "bulk": ["b1.parquet", "b2.tsv"],
        "annotation": "ann.parquet",
        "args": ["--draws", "200"],
    }))
    .unwrap();
    let t = deconvolve_task(&a).unwrap();
    assert_eq!(
        t.argv,
        [
            "deconvolve",
            "--from",
            "r.senna.json",
            "--bulk",
            "b1.parquet",
            "b2.tsv",
            "-o",
            "r.deconv",
            "--annotation",
            "ann.parquet",
            "--draws",
            "200"
        ]
    );
    senna_parses(&t.argv).unwrap();
    assert_eq!(
        t.made[0],
        ("fractions", PathBuf::from("r.deconv.fractions.tsv"))
    );
    assert_eq!(t.made.len(), 4);
    let none: DeconvolveCall = parse(json!({"from": "r.senna.json", "bulk": []})).unwrap();
    assert!(deconvolve_task(&none).is_err());
}
