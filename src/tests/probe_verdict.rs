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
fn the_verdict_is_written_beside_the_cells() {
    let dir = tempfile::tempdir().unwrap();
    let out = dir.path().join("pr").to_string_lossy().into_owned();
    let a = args(&out);
    // Every query cell below the calibration's whole range: all flagged.
    // Cells with no counts (NaN): neither a perfect fit nor a flag.
    let mut cal_fit: Vec<f32> = (0..100).map(|i| -1.0 - i as f32 / 100.0).collect();
    cal_fit.push(f32::NAN);
    // A cell the model gives no chance at all (-inf) is scored, and flagged.
    let mut q_fit = vec![-5.0; 19];
    q_fit.push(f32::NEG_INFINITY);
    q_fit.extend([f32::NAN; 2]);
    let q_names: Vec<Box<str>> = (0..22).map(|i| format!("c{i}").into()).collect();
    write_verdict(Verdict {
        args: &a,
        model_type: "M1",
        cal_fit,
        q_fit,
        q_names,
    })
    .unwrap();
    let text = std::fs::read_to_string(format!("{out}.probe.json")).unwrap();
    let v: serde_json::Value = serde_json::from_str(&text).unwrap();
    assert_eq!(v["verdict"], "novel");
    assert_eq!(v["cells"], 20);
    assert_eq!(v["cells_without_counts"], 2);
    assert_eq!(v["flagged"], 20);
    assert_eq!(v["flag_rate"], 1.0);
    let tsv = std::fs::read_to_string(format!("{out}.probe.tsv")).unwrap();
    assert_eq!(tsv.lines().count(), 23);
    assert!(tsv.ends_with("c21\tNaN\t0\n"), "{tsv}");
}

#[test]
fn a_cell_with_no_counts_has_no_fit() {
    let fit = per_cell_fit(&[-3.0, -1.0], &[3.0, 0.0]);
    assert_eq!(fit[0], -1.0);
    assert!(fit[1].is_nan());
}
