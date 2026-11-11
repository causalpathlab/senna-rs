use super::*;
use clap::Parser;

fn args(out: &str) -> ProbeArgs {
    let argv = [
        "senna",
        "probe",
        "q.zarr",
        "--model",
        "m",
        "--calibration",
        "c.zarr",
        "-o",
        out,
    ];
    match crate::Cli::try_parse_from(argv).unwrap().commands {
        crate::Commands::Probe(a) => a,
        other => panic!("not probe: {other:?}"),
    }
}

#[test]
fn the_verdict_and_the_counterfactual_are_written_beside_the_cells() {
    let dir = tempfile::tempdir().unwrap();
    let out = dir.path().join("pr").to_string_lossy().into_owned();
    let a = args(&out);
    // Every query cell below the calibration's whole range: all flagged.
    let cal_fit: Vec<f32> = (0..100).map(|i| -1.0 - i as f32 / 100.0).collect();
    let q_fit = vec![-5.0; 20];
    let q_names: Vec<Box<str>> = (0..20).map(|i| format!("c{i}").into()).collect();
    write_verdict(Verdict {
        args: &a,
        model_type: "M1",
        cal_fit,
        q_fit,
        q_names,
        counterfactual: Some(serde_json::json!({"benefit": 0.5, "forgetting": 0.1})),
    })
    .unwrap();
    let text = std::fs::read_to_string(format!("{out}.probe.json")).unwrap();
    let v: serde_json::Value = serde_json::from_str(&text).unwrap();
    assert_eq!(v["verdict"], "novel");
    assert_eq!(v["cells"], 20);
    assert_eq!(v["flagged"], 20);
    assert_eq!(v["counterfactual"]["benefit"], 0.5);
    let tsv = std::fs::read_to_string(format!("{out}.probe.tsv")).unwrap();
    assert_eq!(tsv.lines().count(), 21);
}
