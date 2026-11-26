//! Run-manifest round trips and lookups: run kinds, path resolution, moved
//! data, unknown keys, artifact scale and where a run's ρ is.
use super::paths::rel_to_manifest_from;
use super::*;
use legume_numeric::matrix::traits::IoOps;

#[test]
fn every_kind_says_how_its_tables_predict_expression() {
    use ExpressionModel as E;
    assert_eq!(RunKind::Topic.expression_model(), E::Topic);
    assert_eq!(RunKind::MaskedVae.expression_model(), E::TopicOfLogits);
    assert_eq!(RunKind::Vae.expression_model(), E::LatentLoadings);
    assert_eq!(RunKind::Bge.expression_model(), E::Embedding);
    assert_eq!(RunKind::Svd.expression_model(), E::Unmodelled);
    // The log-θ latent rule follows the model.
    assert!(RunKind::Topic.latent_is_log_simplex());
    assert!(!RunKind::MaskedVae.latent_is_log_simplex());
}

#[test]
fn normalize_folds_dots_without_losing_leading_parents() {
    assert_eq!(
        normalize(Path::new("../../a/b")),
        PathBuf::from("../../a/b")
    );
    assert_eq!(normalize(Path::new("a/./b/../c")), PathBuf::from("a/c"));
    assert_eq!(normalize(Path::new("/../a")), PathBuf::from("/a"));
    assert_eq!(normalize(Path::new("a/../../b")), PathBuf::from("../b"));
}

#[test]
fn a_file_of_the_same_name_alone_is_not_taken_for_moved_data() {
    let tmp = tempfile::tempdir().unwrap();
    let root = tmp.path();
    let out = root.join("proj/out");
    fs::create_dir_all(&out).unwrap();
    // Only the bare name exists above the run, in another folder.
    fs::create_dir_all(root.join("other/x.zarr")).unwrap();
    let m = RunManifest::new(RunKind::Svd, "/elsewhere/proj/out/run");
    let recorded = "/elsewhere/proj/data/x.zarr";
    assert_eq!(m.data_file(&out, recorded), PathBuf::from(recorded));
}

#[test]
fn a_moved_runs_data_is_found_where_it_now_sits() {
    let tmp = tempfile::tempdir().unwrap();
    let root = tmp.path().canonicalize().unwrap();
    // Trained at /elsewhere/proj/out/run with data at /elsewhere/proj/data/x.zarr;
    // the project now sits under `root`.
    let out = root.join("proj/out");
    let data = root.join("proj/data/x.zarr");
    fs::create_dir_all(&out).unwrap();
    fs::create_dir_all(&data).unwrap();
    let m = RunManifest::new(RunKind::Svd, "/elsewhere/proj/out/run");

    // Same place relative to the training prefix.
    assert_eq!(m.data_file(&out, "/elsewhere/proj/data/x.zarr"), data);
    // A relative path that no longer resolves: its tail under an ancestor.
    assert_eq!(m.data_file(&out, "../../old/data/x.zarr"), data);
    // As recorded when it exists, and when nothing is found.
    assert_eq!(
        m.data_file(&out, "../data/x.zarr"),
        out.join("../data/x.zarr")
    );
    assert_eq!(
        m.data_file(&out, "/nowhere/y.zarr"),
        PathBuf::from("/nowhere/y.zarr")
    );
}

/// A round's version names what it continued from and trained against,
/// and a version that is no round carries no history at all.
#[test]
fn history_round_trips_and_is_absent_by_default() {
    let mut m = RunManifest::new(RunKind::Vae, "/tmp/run2");
    assert!(serde_json::to_value(&m).unwrap().get("history").is_none());
    m.history = Some(RunHistory {
        parent: "/tmp/run1".into(),
        peer_labels: Some("c.critique.labels.vae.parquet".into()),
        pb_from: Some("topic.senna.json".into()),
        unknown: Unknown::default(),
    });
    let json = serde_json::to_string(&m).unwrap();
    let back: RunManifest = serde_json::from_str(&json).unwrap();
    let h = back.history.expect("history survives");
    assert_eq!(h.parent, "/tmp/run1");
    assert_eq!(
        h.peer_labels.as_deref(),
        Some("c.critique.labels.vae.parquet")
    );
    assert_eq!(h.pb_from.as_deref(), Some("topic.senna.json"));
}

#[test]
fn round_trip() {
    let mut m = RunManifest::new(RunKind::Topic, "/tmp/run1");
    m.data.input = vec!["a.zarr".into(), "b.zarr".into()];
    m.outputs.latent = Some("run1.latent.parquet".into());
    m.layout.cell_coords = Some("run1.cell_coords.parquet".into());
    m.defaults.colour_by = Some("topic".into());
    let json = serde_json::to_string(&m).unwrap();
    let back: RunManifest = serde_json::from_str(&json).unwrap();
    assert_eq!(back.kind, RunKind::Topic);
    assert_eq!(back.data.input.len(), 2);
    assert_eq!(back.outputs.latent.as_deref(), Some("run1.latent.parquet"));
    assert_eq!(
        back.layout.cell_coords.as_deref(),
        Some("run1.cell_coords.parquet")
    );
}

/// A root list's path names its run and root, and no other file does.
#[test]
fn a_root_lists_path_names_its_run() {
    for i in [None, Some(3)] {
        assert_eq!(
            root_list_of(&root_list_path("d/run", i)),
            Some(("d/run", i))
        );
    }
    for other in [
        "d/run.root_.txt",
        "d/run.root_+1.txt",
        "d/run.roots.txt",
        "d/run.txt",
    ] {
        assert_eq!(root_list_of(other), None, "{other}");
    }
}

/// An ODE's root list round trips, and a run with none writes no key.
#[test]
fn an_odes_root_cells_round_trip_and_are_absent_by_default() {
    let mut m = RunManifest::new(RunKind::Ode, "/tmp/ode");
    let json = serde_json::to_value(&m).unwrap();
    assert!(json["outputs"].get("root_cells").is_none());
    m.outputs.root_cells = Some("ode.root.txt".into());
    let back: RunManifest = serde_json::from_str(&serde_json::to_string(&m).unwrap()).unwrap();
    assert_eq!(back.outputs.root_cells.as_deref(), Some("ode.root.txt"));
}

/// Another tool (or a newer senna) may add keys this version does not
/// know, or an older one may carry keys it no longer reads; saving must not
/// drop them, at the top level or inside a section.
#[test]
fn unknown_keys_survive_a_round_trip() {
    let json = r#"{
        "version": 2, "kind": "bge", "prefix": "r",
        "layout": {"cell_coords": "r.umap.cell_coords.parquet", "novel": {"a": 1}},
        "annotate": {"argmax": "a.argmax.tsv", "extra_table": "a.x.parquet"},
        "something_new": [1, 2],
        "pseudotime": {"pseudotime": "pt.pseudotime.parquet", "root_node": 3}
    }"#;
    let m: RunManifest = serde_json::from_str(json).unwrap();
    let back: serde_json::Value =
        serde_json::from_str(&serde_json::to_string(&m).unwrap()).unwrap();
    assert_eq!(back["layout"]["novel"]["a"], 1);
    assert_eq!(back["layout"]["cell_coords"], "r.umap.cell_coords.parquet");
    assert_eq!(back["annotate"]["extra_table"], "a.x.parquet");
    assert_eq!(back["something_new"][1], 2);
    // A block from the retired `lupin pseudotime` is kept as an unknown key.
    assert_eq!(back["pseudotime"]["pseudotime"], "pt.pseudotime.parquet");
    assert_eq!(back["pseudotime"]["root_node"], 3);
}

/// Composition views read the latent first and only fall back to the
/// embedding; geometry consumers do the opposite. The dictionary side
/// falls through to the gene table in the same order.
#[test]
fn structure_tables_prefer_topics_then_fall_back_to_the_embedding() {
    let mut o = RunOutputs::default();
    assert_eq!(o.structure_latent(), None);
    assert_eq!(o.structure_dictionary(), None);
    o.cell_embedding = Some("r.cell_embedding.parquet".into());
    o.feature_embedding = Some("r.feature_embedding.parquet".into());
    assert_eq!(o.structure_latent(), Some("r.cell_embedding.parquet"));
    assert_eq!(
        o.structure_dictionary(),
        Some("r.feature_embedding.parquet")
    );
    o.latent = Some("r.latent.parquet".into());
    o.softmax_dictionary = Some("r.dictionary.parquet".into());
    assert_eq!(o.structure_latent(), Some("r.latent.parquet"));
    assert_eq!(o.structure_dictionary(), Some("r.dictionary.parquet"));
    assert_eq!(o.geometry_latent(), Some("r.cell_embedding.parquet"));
}

#[test]
fn resolve_respects_absolute_and_relative() {
    let dir = Path::new("/tmp/runs");
    assert_eq!(
        resolve(dir, "x.parquet"),
        PathBuf::from("/tmp/runs/x.parquet")
    );
    assert_eq!(
        resolve(dir, "/abs/y.parquet"),
        PathBuf::from("/abs/y.parquet")
    );
}

/// An input path is typed relative to the shell's working directory, but
/// every reader resolves manifest paths against the manifest's directory.
/// Recording must bridge the two, whatever the working directory was.
#[test]
fn a_recorded_path_resolves_back_to_the_same_file_from_the_manifest() {
    let root = tempfile::tempdir().unwrap();
    let root = root.path().canonicalize().unwrap();
    let (cwd, run_dir) = (root.join("work"), root.join("work/out/run"));
    std::fs::create_dir_all(cwd.join("data")).unwrap();
    std::fs::create_dir_all(&run_dir).unwrap();
    std::fs::write(cwd.join("data/x.zarr.zip"), b"").unwrap();

    // Outside the run directory: still relative, so the tree can move
    // or be copied to another machine.
    let input = rel_to_manifest_from(&cwd, &run_dir, "data/x.zarr.zip");
    assert_eq!(input, "../../data/x.zarr.zip");
    assert_eq!(
        normalize(&resolve(&run_dir, &input)),
        cwd.join("data/x.zarr.zip")
    );
    let absolute = cwd.join("data/x.zarr.zip");
    assert_eq!(
        rel_to_manifest_from(&cwd, &run_dir, &absolute.to_string_lossy()),
        "../../data/x.zarr.zip"
    );

    // Inside it (a path that need not exist yet): stored relative to it,
    // so the run directory can be moved.
    let inside = rel_to_manifest_from(&cwd, &run_dir, "out/run/y.zarr.zip");
    assert_eq!(inside, "y.zarr.zip");
    assert_eq!(resolve(&run_dir, &inside), run_dir.join("y.zarr.zip"));
}

/// A run folder and a data folder beside it, with a manifest recording
/// `input` and `batch` as data files.
fn moved_run(input: &[&str], batch: &[&str]) -> (tempfile::TempDir, PathBuf, PathBuf) {
    let root = tempfile::tempdir().unwrap();
    let (r, d) = (root.path().join("run"), root.path().join("data"));
    std::fs::create_dir_all(&r).unwrap();
    std::fs::create_dir_all(&d).unwrap();
    let mut m = RunManifest::new(RunKind::Svd, "run");
    m.data.input = input.iter().map(|s| s.to_string()).collect();
    m.data.batch = batch.iter().map(|s| s.to_string()).collect();
    m.save(&r.join("run.senna.json")).unwrap();
    (root, r, d)
}

#[test]
fn a_found_data_file_and_its_neighbours_are_recorded_relative() {
    let (_root, r, d) = moved_run(
        &[
            "/elsewhere/x/rna.zarr",
            "/elsewhere/x/atac.zarr.zip",
            "kept.h5",
            "/elsewhere/x/gone.h5",
        ],
        &["/elsewhere/x/batch.txt"],
    );
    std::fs::create_dir(d.join("rna.zarr")).unwrap();
    std::fs::write(d.join("atac.zarr.zip"), b"").unwrap();
    std::fs::write(d.join("batch.txt"), b"").unwrap();
    std::fs::write(r.join("kept.h5"), b"").unwrap();

    // Chosen as an absolute path; recorded relative to the manifest.
    let (mut m, dir) = RunManifest::load(&r.join("run.senna.json")).unwrap();
    assert_eq!(m.relocate_data(&dir, 0, &d.join("rna.zarr")).unwrap(), 2);
    assert_eq!(
        m.data.input,
        [
            "../data/rna.zarr",
            "../data/atac.zarr.zip",
            "kept.h5",
            "/elsewhere/x/gone.h5"
        ]
    );
    assert_eq!(m.data.batch, ["../data/batch.txt"]);
}

#[test]
fn files_of_one_name_in_sibling_folders_keep_their_own_folders() {
    let (_root, r, d) = moved_run(&["/old/rna/data.zarr.zip", "/old/atac/data.zarr.zip"], &[]);
    for sub in ["rna", "atac"] {
        std::fs::create_dir(d.join(sub)).unwrap();
        std::fs::write(d.join(sub).join("data.zarr.zip"), b"").unwrap();
    }
    let (mut m, dir) = RunManifest::load(&r.join("run.senna.json")).unwrap();
    assert_eq!(
        m.relocate_data(&dir, 0, &d.join("rna/data.zarr.zip"))
            .unwrap(),
        1
    );
    assert_eq!(
        m.data.input,
        ["../data/rna/data.zarr.zip", "../data/atac/data.zarr.zip"]
    );
}

#[test]
fn unknown_fields_are_ignored() {
    let raw = r#"{"version":1,"kind":"topic","prefix":"r","extra_future_field":42}"#;
    let m: RunManifest = serde_json::from_str(raw).unwrap();
    assert_eq!(m.prefix, "r");
}

/// The three feature-side scales must be distinguishable from contents alone
/// — slot names and filenames have both moved between versions, so content is
/// the only stable signal. Confusing log-simplex with probability-simplex is
/// what silently produced NaN references.
#[test]
fn artifact_scale_distinguishes_the_three_forms() {
    let (d, k) = (64usize, 4usize);
    // probability simplex: non-negative columns summing to 1
    let prob = Mat::from_fn(d, k, |_, _| 1.0 / d as f32);
    assert_eq!(
        ArtifactScale::detect(&prob),
        ArtifactScale::ProbabilitySimplexColumns
    );
    // log simplex: ln of the above
    let logp = prob.map(f32::ln);
    assert_eq!(
        ArtifactScale::detect(&logp),
        ArtifactScale::LogSimplexColumns
    );
    // signed embedding: neither
    let emb = Mat::from_fn(d, k, |g, j| (((g * 7 + j * 13) % 11) as f32 / 11.0) - 0.5);
    assert_eq!(ArtifactScale::detect(&emb), ArtifactScale::Signed);
    // and the guard rejects a mismatch rather than proceeding
    assert!(ArtifactScale::ensure(&logp, ArtifactScale::Signed, "x").is_err());
    assert!(ArtifactScale::ensure(&emb, ArtifactScale::Signed, "x").is_ok());
}

/// The table's own path, or the run's manifest, stands in for the
/// prefix: all three resolve to the same file.
#[test]
fn feature_embedding_resolves_from_prefix_table_path_or_manifest() {
    let dir = tempfile::tempdir().unwrap();
    let prefix = dir.path().join("run").to_string_lossy().into_owned();
    let rho = format!("{prefix}.feature_embedding.parquet");
    write_signed(&rho);
    // A manifest that records no ρ, and one not yet written: both probe
    // the prefix.
    let manifest = default_path(&prefix);
    for given in [prefix.as_str(), rho.as_str(), manifest.as_str()] {
        let (got, bias) = resolve_feature_embedding(given).unwrap();
        assert_eq!(got, rho, "given {given}");
        assert!(bias.is_none());
    }
    write_kind_only(&prefix, RunKind::Topic).unwrap();
    assert_eq!(resolve_feature_embedding(&manifest).unwrap().0, rho);

    // A renamed manifest goes by the ρ it records, not by its own name.
    let renamed = format!("{}/best.senna.json", dir.path().display());
    let mut recorded = RunManifest::new(RunKind::Topic, &prefix);
    recorded.outputs.feature_embedding = Some(rho.clone());
    recorded.save(Path::new(&renamed)).unwrap();
    assert_eq!(resolve_feature_embedding(&renamed).unwrap().0, rho);
    assert!(resolve_feature_embedding(&format!("{prefix}.dictionary.parquet")).is_err());
}

/// A signed 2×2 table at `path`.
fn write_signed(path: &str) {
    let m = Mat::from_fn(2, 2, |i, j| if (i + j) % 2 == 0 { 0.5 } else { -0.5 });
    let genes: Vec<Box<str>> = vec!["A".into(), "B".into()];
    m.to_parquet_with_names(path, (Some(&genes), Some("gene")), None)
        .unwrap();
}

/// `run` in a fresh directory, with a signed table under each suffix.
fn run_with(dir: &Path, suffixes: &[&str]) -> String {
    let prefix = dir.join("run").to_string_lossy().into_owned();
    for suf in suffixes {
        write_signed(&format!("{prefix}{suf}"));
    }
    prefix
}

/// Where ρ is: the run's manifest says when there is one; without it the
/// table under the prefix, and a table named directly is taken as given.
#[test]
fn the_manifest_decides_where_rho_is() {
    let fe = RHO_TABLE_SUFFIX;

    // No manifest: the table under the prefix, or named directly.
    let dir = tempfile::tempdir().unwrap();
    let prefix = run_with(dir.path(), &[fe]);
    let e = format!("{prefix}{fe}");
    assert_eq!(resolve_feature_embedding(&prefix).unwrap().0, e);
    assert_eq!(resolve_feature_embedding(&e).unwrap().0, e);

    // A manifest recording ρ elsewhere: a table named beside it that is
    // not that ρ is refused, naming the one it records.
    let dir = tempfile::tempdir().unwrap();
    let prefix = run_with(dir.path(), &[fe]);
    let other = format!("{}/other{fe}", dir.path().display());
    write_signed(&other);
    let mut fne = RunManifest::new(RunKind::Fne, &prefix);
    fne.outputs.feature_embedding = Some(other.clone());
    fne.save(Path::new(&default_path(&prefix))).unwrap();
    assert_eq!(resolve_feature_embedding(&prefix).unwrap().0, other);
    let err = resolve_feature_embedding(&format!("{prefix}{fe}"))
        .unwrap_err()
        .to_string();
    assert!(err.contains(&other), "{err}");

    // A moved run whose manifest records no ρ: probed where it is now,
    // not under the prefix it was trained at.
    let dir = tempfile::tempdir().unwrap();
    let prefix = run_with(dir.path(), &[fe]);
    RunManifest::new(RunKind::Bge, "/nowhere/run")
        .save(Path::new(&default_path(&prefix)))
        .unwrap();
    assert_eq!(
        resolve_feature_embedding(&prefix).unwrap().0,
        format!("{prefix}{fe}")
    );

    // A `dictionary` is never ρ, whatever the kind.
    let dir = tempfile::tempdir().unwrap();
    let prefix = run_with(dir.path(), &[".dictionary.parquet"]);
    write_kind_only(&prefix, RunKind::Bge).unwrap();
    assert!(resolve_feature_embedding(&prefix).is_err());
    assert!(resolve_feature_embedding(&format!("{prefix}.dictionary.parquet")).is_err());

    // A manifest this binary cannot read is gone by the tables beside it.
    let dir = tempfile::tempdir().unwrap();
    let prefix = run_with(dir.path(), &[fe]);
    std::fs::write(default_path(&prefix), "{\"kind\": \"from-the-future\"}").unwrap();
    let e = format!("{prefix}{fe}");
    assert_eq!(resolve_feature_embedding(&prefix).unwrap().0, e);
    assert_eq!(resolve_feature_embedding(&e).unwrap().0, e);
}

/// A v1 manifest kept its co-embed where v2 keeps ρ; read as v2 it would hand
/// the co-embed out as the gene embedding, so it is refused by version.
#[test]
fn a_manifest_before_v2_is_refused() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("r.senna.json");
    std::fs::write(&path, r#"{"version":1,"kind":"bge","prefix":"r"}"#).unwrap();
    let err = RunManifest::load(&path).unwrap_err().to_string();
    assert!(err.contains("v1") && err.contains("Refit"), "{err}");
}

#[test]
fn outputs_beside_a_run_are_only_those_in_its_directory() {
    let dir = tempfile::tempdir().unwrap();
    let run = dir.path().join("fig").join("run.senna.json");
    std::fs::create_dir_all(run.parent().unwrap()).unwrap();
    let at = |p: &std::path::Path| p.to_string_lossy().into_owned();
    // Beside it, under any name, and through `..` that lands back there.
    assert!(outputs_beside(
        &run,
        &at(&dir.path().join("fig").join("run"))
    ));
    assert!(outputs_beside(
        &run,
        &at(&dir.path().join("fig").join("other"))
    ));
    assert!(outputs_beside(
        &run,
        &at(&dir.path().join("fig").join("sub").join("..").join("alt"))
    ));
    // Anywhere else leaves the run alone.
    assert!(!outputs_beside(
        &run,
        &at(&dir.path().join("scratch").join("run"))
    ));
    assert!(!outputs_beside(
        &run,
        &at(&dir.path().join("fig").join("sub").join("run"))
    ));
}
