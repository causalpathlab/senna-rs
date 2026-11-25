use super::*;

/// A cell fit leads to its clustering, then to lupin's annotation; a fit
/// with no cells to cluster, nowhere; an ODE with untyped cells, to the
/// annotation; one whose root it chose and lupin typed, or with a question
/// open, to a root of the user's choosing; else nowhere.
#[test]
fn a_fit_s_next_call_is_the_next_step_toward_its_cell_types() {
    let run = Path::new("/nowhere/r.senna.json");
    let tool = |m: &RunManifest, said: Said| -> Vec<String> {
        let types = Types::new(run, m);
        next_of(m, run, &types, &said)
            .into_iter()
            .map(|n| n.tool)
            .collect()
    };
    let nothing = Said::default;
    let mut m = RunManifest::new(RunKind::Topic, "/nowhere/r");
    assert!(tool(&m, nothing()).is_empty());
    m.outputs.latent = Some("r.latent.parquet".into());
    let first = &next_after(&m, run)[0];
    assert_eq!(first.tool, "run_step");
    assert_eq!(first.args["out"], "/nowhere/r_clusters");
    m.cluster.clusters = Some("k.clusters.parquet".into());
    assert_eq!(tool(&m, nothing()), ["lupin annotate_run"]);

    let mut ode = RunManifest::new(RunKind::Ode, "/nowhere/r");
    ode.cluster.clusters = Some("k.clusters.parquet".into());
    assert!(tool(&ode, nothing()).is_empty());
    let untyped = Said {
        untyped: true,
        ..nothing()
    };
    assert_eq!(tool(&ode, untyped), ["lupin annotate_run"]);
    let chose = Said {
        chose_typed_root: true,
        ..nothing()
    };
    assert_eq!(tool(&ode, chose), ["view_cells"]);
    let open = Said {
        open: true,
        ..nothing()
    };
    assert_eq!(tool(&ode, open), ["view_cells"]);
}

/// A named root no tree grew from is said unused, with why and, as a used
/// one would be, what its cells are by the groups view_cells wrote it from;
/// lineages asked are told apart from those holding cells.
#[test]
fn an_unused_root_says_why_and_what_its_cells_are() {
    let dir = tempfile::tempdir().unwrap();
    let run = dir.path().join("r.senna.json");
    std::fs::write(dir.path().join("root.txt"), "a\nb\n").unwrap();
    std::fs::write(
        dir.path().join("root.txt.source.json"),
        r#"{"total":2,"groups":[{"group":"HSC","cells":2}],"grouping":"cell type","round":"r.L1.senna.json"}"#,
    )
    .unwrap();
    let mut m = RunManifest::new(RunKind::Ode, "r");
    m.outputs.root_cells_unused = Some("root.txt".into());
    m.outputs.root_cells_unused_why = Some("one lineage".into());
    m.outputs.cell_time_lineages = Some(8);
    m.train_args = Some(run_manifest::TrainArgsRecord {
        senna_version: "0".into(),
        args: json!({"lineages": 10, "root_cells": "root.txt"}),
    });
    let mut out = json!({});
    decision_fields(&m, &Types::new(&run, &m), &mut out);
    let root = &out["root"];
    assert_eq!(
        (&root["used"], &root["note"]),
        (&json!(false), &json!("one lineage"))
    );
    assert_eq!(root["mostly"], "HSC (100%)");
    assert_eq!(root["typed_by"]["grouping"], "cell type");
    assert_eq!(out["lineages"], json!({"asked": 10, "with_cells": 8}));
}
