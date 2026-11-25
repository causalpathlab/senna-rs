use super::*;
use crate::view::mcp::jobs::senna_parses;
use std::path::PathBuf;

fn step(v: Value) -> Result<Task> {
    step_task(&parse(v).unwrap())
}

#[test]
fn each_step_runs_its_command_on_the_run_and_names_what_it_makes() {
    let cases = [
        (
            json!({"command": "clustering", "from": "d/r.senna.json", "out": "d/k8", "flags": {"m": "kmeans", "num-clusters": 8}}),
            vec![
                "clustering",
                "--from",
                "d/r.senna.json",
                "-o",
                "d/k8",
                "--method=kmeans",
                "--num-clusters=8",
            ],
            vec![("clusters", "d/k8.clusters.parquet")],
        ),
        (
            json!({"command": "clustering", "from": "r.senna.json", "out": "k", "target": "features"}),
            vec![
                "clustering",
                "--from",
                "r.senna.json",
                "-o",
                "k",
                "--target",
                "features",
            ],
            vec![("clusters", "k.feature_clusters.parquet")],
        ),
        (
            json!({"command": "resolve-embedding-space", "from": "r.senna.json", "out": "rest"}),
            vec![
                "resolve-embedding-space",
                "--from",
                "r.senna.json",
                "-o",
                "rest",
            ],
            vec![("run", "rest.senna.json")],
        ),
        (
            json!({"command": "joint-topic", "data": ["rna.zarr", "atac.zarr"], "out": "j", "flags": {"modalities": 2}}),
            vec![
                "joint-topic",
                "rna.zarr",
                "atac.zarr",
                "-o",
                "j",
                "--modalities=2",
            ],
            vec![("run", "j.senna.json")],
        ),
        (
            json!({"command": "fne", "data": ["ppi.tsv"], "out": "g"}),
            vec!["fne", "ppi.tsv", "-o", "g"],
            vec![("run", "g.senna.json")],
        ),
    ];
    for (call, argv, made) in cases {
        let t = step(call.clone()).unwrap();
        assert_eq!(t.argv, argv, "{call}");
        senna_parses(&t.argv).unwrap();
        let made: Vec<(&str, PathBuf)> = made.into_iter().map(|(k, p)| (k, p.into())).collect();
        assert_eq!(t.made, made, "{call}");
    }
}

#[test]
fn a_step_missing_its_input_or_writing_over_its_run_is_refused() {
    for bad in [
        json!({"command": "clustering"}),
        json!({"command": "clustering", "from": "r.senna.json"}),
        json!({"command": "clustering", "from": "r.senna.json", "data": ["x.zarr"]}),
        json!({"command": "joint-svd", "out": "j"}),
        json!({"command": "joint-svd", "data": ["a.zarr"]}),
        json!({"command": "resolve-embedding-space", "from": "r.senna.json"}),
        json!({"command": "resolve-embedding-space", "from": "r.senna.json", "out": "rest", "target": "features"}),
    ] {
        assert!(step(bad.clone()).is_err(), "{bad}");
    }
    assert!(parse::<StepCall>(json!({"command": "view"})).is_err());
}
