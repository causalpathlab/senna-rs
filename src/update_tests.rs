//! What `senna update` decides before it dispatches: whether to substitute the
//! parent's carried pseudobulks, and what it does when it cannot.

use super::{carried_reference_among, multiome_in_args, round_inputs, UpdateArgs};
use clap::Parser;

#[derive(clap::Parser)]
struct Cli {
    #[command(flatten)]
    args: UpdateArgs,
}

fn parse(extra: &[&str]) -> Result<UpdateArgs, clap::Error> {
    let base = ["senna-update", "new.zarr", "--model", "m", "-o", "out"];
    Cli::try_parse_from(base.iter().copied().chain(extra.iter().copied())).map(|c| c.args)
}

/// Carrying the pseudobulks forward is what makes a round cost the new data
/// rather than the whole history, so it is on unless refused.
/// `--use-pb-reference` is redundant for the behaviour, but it turns a silent
/// fallback into an error when the parent carries nothing; asking for both is
/// a contradiction, not a precedence puzzle.
#[test]
fn the_carried_reference_is_used_unless_refused() {
    let a = parse(&[]).expect("bare update parses");
    assert!(!a.no_pb_reference && !a.use_pb_reference);
    let a = parse(&["--no-pb-reference"]).expect("opt-out parses");
    assert!(a.no_pb_reference);
    let a = parse(&["--use-pb-reference"]).expect("explicit request parses");
    assert!(a.use_pb_reference && !a.no_pb_reference);
    assert!(parse(&["--use-pb-reference", "--no-pb-reference"]).is_err());
}

/// The recorded fit arguments say whether the parent loads under multiome
/// alignment: a flag on the topic families, a suffix list on bge, and absent
/// on a run that predates the option.
#[test]
fn a_multiome_parent_is_read_off_its_recorded_arguments() {
    assert!(multiome_in_args(&serde_json::json!({ "multiome": true })));
    assert!(!multiome_in_args(&serde_json::json!({ "multiome": false })));
    assert!(multiome_in_args(
        &serde_json::json!({ "multiome": ["rna", "atac"] })
    ));
    assert!(!multiome_in_args(&serde_json::json!({ "multiome": [] })));
    assert!(!multiome_in_args(&serde_json::json!({ "epochs": 10 })));
}

/// A lineage that substituted once holds the carried reference among its
/// inputs, under the name `update` gave it.
#[test]
fn a_substituted_lineage_is_recognised_by_its_carried_reference() {
    let plain: Vec<Box<str>> = vec!["a.zarr".into(), "b.zarr".into()];
    assert_eq!(carried_reference_among(&plain), None);
    let substituted: Vec<Box<str>> = vec!["c.zarr".into(), "runs/r1.pb_reference.zarr.zip".into()];
    assert_eq!(
        carried_reference_among(&substituted),
        Some("runs/r1.pb_reference.zarr.zip")
    );
}

/// Manifests store data paths relative to the run directory; they resolve
/// there, never against the caller's cwd, and absolute paths pass through.
#[test]
fn recorded_paths_resolve_against_the_run_directory() {
    let run = std::path::Path::new("/runs/r1");
    let mut m =
        senna::run_manifest::RunManifest::new(senna::run_manifest::RunKind::Svd, "/runs/r1/p");
    m.data.input = vec!["rna.zarr.zip".into(), "/abs/atac.zarr".into()];
    let got = m.data_inputs(run);
    assert_eq!(
        got,
        vec![
            Box::from("/runs/r1/rna.zarr.zip"),
            Box::from("/abs/atac.zarr")
        ]
    );
}

// ---- Rounds: continue on the same cells, with critique labels -------------------

/// A round continues the parent on its own cells: no new data file is needed.
#[test]
fn a_round_needs_no_new_data() {
    let round = ["senna-update", "--model", "m", "-o", "out"];
    let a = Cli::try_parse_from(round).expect("a round parses").args;
    assert!(a.data_files.is_empty());
    assert!(a.is_round());
    let a = parse(&[]).expect("an update with new data parses");
    assert!(!a.is_round());
}

/// What a round may set: the partition to collapse on and the labels to train
/// on. What it may not: batch files for new data it does not have, and the
/// carried-reference request, which stands pseudobulks in for new cells.
#[test]
fn a_round_takes_a_partition_and_labels_but_no_new_cell_flags() {
    let round = |extra: &[&str]| {
        let base = ["senna-update", "--model", "m", "-o", "out"];
        Cli::try_parse_from(base.iter().copied().chain(extra.iter().copied())).map(|c| c.args)
    };
    let a = round(&["--pb-from", "p.senna.json", "--peer-labels", "l.parquet"])
        .expect("round flags parse");
    assert_eq!(a.pb_from.as_deref(), Some("p.senna.json"));
    assert_eq!(a.peer_labels.as_deref(), Some("l.parquet"));
    assert!(a.check_round().is_ok());
    let a = round(&["--batch-files", "b.tsv"]).expect("parses");
    assert!(
        a.check_round().is_err(),
        "batch files describe new data a round has none of"
    );
    let a = round(&["--use-pb-reference"]).expect("parses");
    assert!(
        a.check_round().is_err(),
        "a round never substitutes its only cells"
    );
    let a = parse(&["--pb-from", "p.senna.json"]).expect("parses");
    assert!(
        a.check_round().is_err(),
        "a partition over the parent's cells cannot cover new cells"
    );
}

/// A round replays the recorded inputs and batches, untouched.
#[test]
fn a_round_replays_the_recorded_inputs() {
    let recorded: Vec<Box<str>> = vec!["a.zarr".into(), "b.zarr".into()];
    let batches: Vec<Box<str>> = vec!["a.tsv".into(), "b.tsv".into()];
    let (d, b) = round_inputs(recorded.clone(), batches.clone());
    assert_eq!(d, recorded);
    assert_eq!(b.as_deref(), Some(batches.as_slice()));
    let (_, b) = round_inputs(recorded, Vec::new());
    assert!(b.is_none());
}
