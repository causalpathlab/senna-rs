use super::*;
use std::path::PathBuf;

#[test]
fn predict_runs_senna_predict_and_makes_the_out_s_latent() {
    let a = parse(
        json!({"model": "ref", "data": ["a.zarr", "b.zarr"], "args": ["--refine-steps", "5"]}),
    )
    .unwrap();
    let t = predict_task(&a, "p");
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
fn update_runs_senna_update_and_makes_the_out_s_run() {
    let a = parse(json!({"model": "m1", "data": ["new.zarr"], "args": ["-b", "b.txt"]})).unwrap();
    let t = update_task(&a, "m2");
    assert_eq!(
        t.argv,
        ["update", "new.zarr", "--model", "m1", "-o", "m2", "-b", "b.txt"]
    );
    assert_eq!(t.made, [("run", PathBuf::from("m2.senna.json"))]);
    // No new data is a round.
    let round = update_task(&parse(json!({"model": "m1"})).unwrap(), "m2");
    assert_eq!(round.argv, ["update", "--model", "m1", "-o", "m2"]);
}
