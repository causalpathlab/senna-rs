//! What `senna update` decides before it dispatches: whether to substitute the
//! parent's carried pseudobulks, and what it does when it cannot.

use super::{
    carried_reference_among, check_round_inputs, multiome_in_args, round_partition, union_batches,
    UpdateArgs,
};
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

/// `update` always absorbs new data, and takes no partition: rounds on a
/// fit's own cells belong to `senna revise`.
#[test]
fn an_update_needs_new_data_and_takes_no_round_flags() {
    let bare = ["senna-update", "--model", "m", "-o", "out"];
    assert!(Cli::try_parse_from(bare).is_err(), "no new data");
    parse(&[]).expect("new data parses");
    for flag in [
        ["--pb-from", "p.senna.json"],
        ["--peer-labels", "l.parquet"],
    ] {
        assert!(parse(&flag).is_err(), "{} belongs to senna revise", flag[0]);
    }
}

/// `senna revise` sets a fit up on its own cells: no new data, nothing else
/// changed.
#[test]
fn a_revision_sets_up_the_parent_on_its_own_cells() {
    let a = UpdateArgs::own_cells("m".into(), "out".into());
    assert!(a.data_files.is_empty());
    assert!(a.epochs.is_none() && a.batch_files.is_none());
    assert_eq!((a.add_topics, a.add_embedding_dim), (0, 0));
}

/// With no new data, the recorded batches are replayed as they are; with new
/// data, the new files still need their own.
#[test]
fn a_round_replays_the_recorded_batches() {
    let recorded: Vec<Box<str>> = vec!["a.tsv".into(), "b.tsv".into()];
    let b = union_batches(recorded.clone(), None, 0).expect("a round replays");
    assert_eq!(b.as_deref(), Some(recorded.as_slice()));
    assert!(union_batches(recorded, None, 1).is_err());
    assert!(union_batches(Vec::new(), None, 0)
        .expect("no batches")
        .is_none());
}

/// A parent cut by --cnv-clones built its own strata; a round cannot collapse
/// it on another run's partition. The flag is read off the recorded arguments.
#[test]
fn a_cnv_cut_parent_is_read_off_its_recorded_arguments() {
    use crate::refine_weighting::cut_by_cnv_clones;
    assert!(cut_by_cnv_clones(
        &serde_json::json!({ "collapse": { "cnv_clones": "c.tsv" } })
    ));
    assert!(!cut_by_cnv_clones(
        &serde_json::json!({ "collapse": { "cnv_clones": null } })
    ));
    assert!(!cut_by_cnv_clones(&serde_json::json!({ "epochs": 10 })));
}

/// A round keeps the pseudobulks it is critiqued on: without --pb-from it
/// collapses on the parent's own partition. A parent cut by --cnv-clones built
/// strata no partition can stand for, so it rebuilds; non-collapsing kinds
/// have no partition at all; with new data nothing is inherited.
#[test]
fn a_round_without_pb_from_keeps_the_parents_partition() {
    use senna::run_manifest::RunKind;
    let p = |given: Option<&str>, kind, round, cnv| {
        round_partition(given.map(Box::from), "runs/m", kind, round, cnv)
    };
    assert_eq!(
        p(None, RunKind::Vae, true, false).as_deref(),
        Some("runs/m")
    );
    assert_eq!(
        p(Some("t"), RunKind::Vae, true, false).as_deref(),
        Some("t")
    );
    assert_eq!(p(None, RunKind::Vae, true, true), None);
    assert_eq!(p(None, RunKind::Svd, true, false), None);
    assert_eq!(p(None, RunKind::Topic, false, false), None);
}

/// A lineage absorbed through carried pseudobulks holds them among its inputs;
/// a round would replay them as cells, so it is refused with a reason that fits.
#[test]
fn a_round_refuses_a_lineage_holding_carried_pseudobulks() {
    let cells: Vec<Box<str>> = vec!["a.zarr".into()];
    assert!(check_round_inputs(&cells).is_ok());
    let carried: Vec<Box<str>> = vec!["a.zarr".into(), "r1.pb_reference.zarr.zip".into()];
    let e = check_round_inputs(&carried).unwrap_err().to_string();
    assert!(e.contains("round") && !e.contains("--batch-files"), "{e}");
}
