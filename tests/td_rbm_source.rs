//! A td-rbm run chains only into td-rbm: its feature table is in genes
//! while its data rows are tracks, so no other fit takes it as a source.
//! And a run of a retired kind says so instead of failing to parse.

use senna::run_manifest::{inherit_data_from, inherit_from, RunKind, RunManifest};

#[test]
fn only_td_rbm_takes_a_td_rbm_run_as_its_source() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("td.senna.json");
    let mut m = RunManifest::new(RunKind::TdRbm, "td");
    m.data.input = vec!["a.zarr".into(), "b.zarr".into()];
    m.save(&path).unwrap();
    let path = path.to_string_lossy();
    let Err(e) = inherit_from(&path) else {
        panic!("a td-rbm run became another fit's source");
    };
    assert!(e.to_string().contains("not a source"), "{e}");
    let got = inherit_data_from(&path).unwrap();
    assert_eq!(got.data_files.len(), 2);
    assert_eq!(got.source_kind, RunKind::TdRbm);
}

#[test]
fn a_gem_run_is_named_as_retired() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("old.senna.json");
    let mut v = serde_json::to_value(RunManifest::new(RunKind::Bge, "old")).unwrap();
    v["kind"] = "gem".into();
    std::fs::write(&path, v.to_string()).unwrap();
    let Err(e) = RunManifest::load(&path) else {
        panic!("a gem run opened");
    };
    assert!(
        e.to_string()
            .contains("a `senna gem` run, a kind this senna does not open"),
        "{e}"
    );
}
