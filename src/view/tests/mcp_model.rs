use super::*;
use crate::view::mcp::jobs::senna_parses;
use std::path::PathBuf;

fn update(v: Value, out: &str) -> Result<Task> {
    update_task(&parse(v).unwrap(), out)
}

#[test]
fn update_runs_senna_update_and_makes_the_out_s_run() {
    let t = update(
        json!({"model": "m1", "data": ["new.zarr"], "args": ["-b", "b.txt"]}),
        "m2",
    )
    .unwrap();
    assert_eq!(
        t.argv,
        ["update", "new.zarr", "--model", "m1", "-o", "m2", "-b", "b.txt"]
    );
    assert_eq!(t.made, [("run", PathBuf::from("m2.senna.json"))]);
    // No new data is a round.
    let round = update(json!({"model": "m1"}), "m2").unwrap();
    assert_eq!(round.argv, ["update", "--model", "m1", "-o", "m2"]);
}

#[test]
fn update_absorbs_with_its_batches_or_runs_a_round_on_peer_labels() {
    let t = update(
        json!({"model": "m1.senna.json", "data": ["n1.zarr", "n2.zarr"], "batch_files": ["b1.txt", "b2.txt"]}),
        "m2",
    )
    .unwrap();
    assert_eq!(
        t.argv,
        [
            "update", "n1.zarr", "n2.zarr", "--model", "m1", "-o", "m2", "-b", "b1.txt", "-b",
            "b2.txt"
        ]
    );
    senna_parses(&t.argv).unwrap();
    let t = update(
        json!({"model": "m1", "pb_from": "p.senna.json", "peer_labels": "c.critique.labels.m1.parquet"}),
        "m2",
    )
    .unwrap();
    assert_eq!(
        t.argv,
        [
            "update",
            "--model",
            "m1",
            "-o",
            "m2",
            "--pb-from",
            "p.senna.json",
            "--peer-labels",
            "c.critique.labels.m1.parquet"
        ]
    );
    senna_parses(&t.argv).unwrap();
    // A round replays the parent's batches, and its partition cannot
    // cover new cells; the parent is kept.
    for bad in [
        json!({"model": "m1", "batch_files": ["b.txt"]}),
        json!({"model": "m1", "data": ["n.zarr"], "pb_from": "p.senna.json"}),
    ] {
        assert!(update(bad.clone(), "m2").is_err(), "{bad}");
    }
    assert!(update(json!({"model": "m1.senna.json"}), "m1").is_err());
}

#[test]
fn critique_compares_the_runs_and_makes_its_tables() {
    let a: CritiqueCall = parse(json!({
        "runs": ["a.senna.json", "b.senna.json", "c.senna.json"],
        "out": "c",
        "partition": "a.senna.json",
        "questions": 50,
        "cell_labels": "labels.tsv",
        "label_column": "label",
    }))
    .unwrap();
    let t = critique_task(&a).unwrap();
    assert_eq!(
        t.argv,
        [
            "critique",
            "a.senna.json",
            "b.senna.json",
            "c.senna.json",
            "-o",
            "c",
            "--partition",
            "a.senna.json",
            "--questions",
            "50",
            "--cell-labels",
            "labels.tsv",
            "--label-column",
            "label"
        ]
    );
    senna_parses(&t.argv).unwrap();
    assert_eq!(
        t.made,
        [
            ("summary", PathBuf::from("c.critique.summary.parquet")),
            ("pairs", PathBuf::from("c.critique.pairs.parquet")),
            ("record", PathBuf::from("c.critique.json"))
        ]
    );
    let one: CritiqueCall = parse(json!({"runs": ["a.senna.json"], "out": "c"})).unwrap();
    assert!(critique_task(&one).is_err());
}
