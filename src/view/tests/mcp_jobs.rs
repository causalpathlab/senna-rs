use super::*;

#[test]
fn a_command_line_is_built_flag_by_flag_and_skips_what_is_not_given() {
    let none: Option<&str> = None;
    let argv = Argv::new("predict")
        .args(["a.zarr", "b.zarr"])
        .many("--bulk", &[])
        .flag("--model", Some("m"))
        .flag("--eval-features", none)
        .each("-b", &["b1.txt".into(), "b2.txt".into()])
        .many("--null-from", &["t1.zarr".into(), "t2.zarr".into()])
        .switch("--decoder-only", false)
        .switch("--counterfactual", true)
        .rest(&["--seed".into(), "3".into()]);
    assert_eq!(
        argv,
        [
            "predict",
            "a.zarr",
            "b.zarr",
            "--model",
            "m",
            "-b",
            "b1.txt",
            "-b",
            "b2.txt",
            "--null-from",
            "t1.zarr",
            "t2.zarr",
            "--counterfactual",
            "--seed",
            "3"
        ]
    );
}

#[test]
fn a_result_there_already_is_reported_and_half_of_one_is_refused() {
    let dir = tempfile::tempdir().unwrap();
    let a = dir.path().join("p.latent.parquet");
    let b = dir.path().join("p.predictive.parquet");
    let both = [a.clone(), b.clone()];
    assert!(!already(&both).unwrap());
    std::fs::write(&b, b"keep").unwrap();
    let e = already(&both).unwrap_err().to_string();
    assert!(
        e.contains("p.predictive.parquet") && e.contains("p.latent.parquet"),
        "{e}"
    );
    std::fs::write(&a, b"keep").unwrap();
    assert!(already(&both).unwrap());
    assert_eq!(std::fs::read(&b).unwrap(), b"keep");
}
