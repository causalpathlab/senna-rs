//! Saving a run manifest: whole or not at all, through a link, keeping its
//! permissions; and a run that records more than its description writing
//! it once.

use super::*;

#[test]
fn a_manifest_is_saved_whole_and_leaves_nothing_beside_it() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("r.senna.json");
    RunManifest::new(RunKind::Svd, "r").save(&path).unwrap();
    RunManifest::new(RunKind::Topic, "r").save(&path).unwrap();
    let (m, _) = RunManifest::load(&path).unwrap();
    assert_eq!(m.kind, RunKind::Topic);
    let names: Vec<_> = fs::read_dir(dir.path())
        .unwrap()
        .map(|e| e.unwrap().file_name())
        .collect();
    assert_eq!(names, ["r.senna.json"]);
}

#[test]
fn an_edit_is_in_the_manifest_from_its_one_write() {
    let dir = tempfile::tempdir().unwrap();
    let prefix = dir.path().join("r").to_string_lossy().into_owned();
    let desc = RunDescription {
        kind: RunKind::TdRbm,
        prefix: &prefix,
        train_args: None,
        data_input: &[],
        data_batch: &[],
        data_input_null: &[],
        data_multiome: None,
        dictionary_suffix: None,
        has_model: false,
        has_cell_proj: false,
        pb_gene_suffix: None,
        pb_latent_suffix: None,
        pb_reference_suffix: None,
        dictionary_empirical_suffix: None,
        feature_embedding_suffix: None,
        feature_coembedding_suffix: None,
        carried: None,
        module_membership_suffix: None,
        module_dictionary_suffix: None,
        softmax_dictionary_suffix: None,
        cell_embedding_suffix: None,
        cell_encoder_suffix: None,
        default_colour_by: "cluster",
        has_latent: false,
        has_cell_to_pb: false,
        has_pb_tree: false,
    };
    write_run_manifest_with(&desc, |m, base| {
        m.outputs.cell_time = Some(format!("{base}.cell_time.parquet"));
    })
    .unwrap();
    let (m, _) = RunManifest::load(Path::new(&default_path(&prefix))).unwrap();
    assert_eq!(m.outputs.cell_time.as_deref(), Some("r.cell_time.parquet"));
}

#[cfg(unix)]
#[test]
fn a_linked_manifest_is_written_through_and_keeps_its_permissions() {
    use std::os::unix::fs::PermissionsExt;
    let dir = tempfile::tempdir().unwrap();
    let shared = dir.path().join("shared.senna.json");
    RunManifest::new(RunKind::Svd, "r").save(&shared).unwrap();
    fs::set_permissions(&shared, fs::Permissions::from_mode(0o664)).unwrap();
    let link = dir.path().join("r.senna.json");
    std::os::unix::fs::symlink(&shared, &link).unwrap();

    RunManifest::new(RunKind::Topic, "r").save(&link).unwrap();
    assert!(fs::symlink_metadata(&link)
        .unwrap()
        .file_type()
        .is_symlink());
    assert_eq!(RunManifest::load(&shared).unwrap().0.kind, RunKind::Topic);
    let mode = fs::metadata(&shared).unwrap().permissions().mode() & 0o777;
    assert_eq!(mode, 0o664);
}

#[cfg(unix)]
#[test]
fn a_new_manifest_gets_the_mode_a_plain_write_gives() {
    use std::os::unix::fs::PermissionsExt;
    let dir = tempfile::tempdir().unwrap();
    let plain = dir.path().join("plain");
    fs::write(&plain, "x").unwrap();
    let path = dir.path().join("r.senna.json");
    RunManifest::new(RunKind::Svd, "r").save(&path).unwrap();
    let mode = |p: &Path| fs::metadata(p).unwrap().permissions().mode() & 0o777;
    assert_eq!(mode(&path), mode(&plain));
}

#[cfg(unix)]
#[test]
fn a_link_to_a_manifest_not_written_yet_is_written_through() {
    let dir = tempfile::tempdir().unwrap();
    fs::create_dir(dir.path().join("shared")).unwrap();
    let link = dir.path().join("r.senna.json");
    std::os::unix::fs::symlink("shared/r.senna.json", &link).unwrap();
    RunManifest::new(RunKind::Svd, "r").save(&link).unwrap();
    assert!(fs::symlink_metadata(&link)
        .unwrap()
        .file_type()
        .is_symlink());
    let shared = dir.path().join("shared/r.senna.json");
    assert_eq!(RunManifest::load(&shared).unwrap().0.kind, RunKind::Svd);
}
