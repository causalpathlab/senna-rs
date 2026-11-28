use super::*;
use clap::Parser;
use legume_numeric::matrix::traits::IoOps;
use senna::run_manifest::RunKind;

#[derive(Parser)]
struct Cli {
    #[command(flatten)]
    args: ClusteringArgs,
}

#[test]
fn from_a_run_alone_clusters_its_cell_table_into_its_own_prefix() {
    let dir = tempfile::tempdir().unwrap();
    let prefix = dir.path().join("run").to_string_lossy().into_owned();
    let cells: Vec<Box<str>> = (0..6).map(|i| format!("c{i}").into()).collect();
    let z = Mat::from_fn(6, 2, |i, k| if (i < 3) == (k == 0) { 5.0 } else { 0.0 });
    z.to_parquet_with_names(
        &format!("{prefix}.cell_embedding.parquet"),
        (Some(&cells), Some("cell")),
        None,
    )
    .unwrap();
    // The cell table is the embedding, not the latent (not even written).
    let mut m = RunManifest::new(RunKind::Bge, &prefix);
    m.outputs.cell_embedding = Some("run.cell_embedding.parquet".into());
    m.outputs.latent = Some("run.latent.parquet".into());
    let from = senna::run_manifest::default_path(&prefix);
    m.save(Path::new(&from)).unwrap();

    let cli = Cli::try_parse_from(["clustering", "--from", &from, "-k", "2"]).unwrap();
    run_clustering(&cli.args).unwrap();

    let got =
        Mat::from_parquet_with_row_names(&format!("{prefix}.clusters.parquet"), Some(0)).unwrap();
    assert_eq!(got.rows, cells);
    let (m, _) = RunManifest::load(Path::new(&from)).unwrap();
    assert_eq!(m.cluster.clusters.as_deref(), Some("run.clusters.parquet"));
}

#[test]
fn degree_correction_is_on_by_default_and_can_be_turned_off() {
    let dc = |extra: &[&str]| {
        let argv = [&["clustering", "--latent", "z.parquet", "-o", "o"], extra].concat();
        Cli::try_parse_from(argv).unwrap().args.degree_corrected
    };
    assert!(dc(&[]));
    assert!(dc(&["--degree-corrected"]));
    assert!(!dc(&["--degree-corrected", "false"]));
}
