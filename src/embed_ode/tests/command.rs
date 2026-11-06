use crate::embed_ode::command::{run_embed_ode, EmbedOdeArgs};
use crate::ode_mixture::command::{run_ode, OdeArgs};
use crate::ode_mixture::sim::{simulate, Sim, SimConfig};
use clap::Parser;
use data_beans::sparse_io::{create_sparse_from_dmatrix, SparseIoBackend};
use legume_numeric::matrix::parquet::read_table_columns;
use legume_numeric::matrix::traits::{IoOps, MatWithNames};
use nalgebra::DMatrix;
use senna::embed_common::Mat;
use senna::run_manifest::{RunKind, RunManifest};

#[derive(Parser)]
struct OdeCli {
    #[command(flatten)]
    args: OdeArgs,
}

#[derive(Parser)]
struct Cli {
    #[command(flatten)]
    args: EmbedOdeArgs,
}

fn parse_ode(argv: &[&str]) -> OdeArgs {
    let mut full = vec!["ode"];
    full.extend_from_slice(argv);
    OdeCli::try_parse_from(full).expect("valid argv").args
}

fn parse(argv: &[&str]) -> EmbedOdeArgs {
    let mut full = vec!["embed-ode"];
    full.extend_from_slice(argv);
    Cli::try_parse_from(full).expect("valid argv").args
}

/// The simulation's pseudobulks as cells of a two-track backend: rows
/// `GENEj/count/spliced` then `GENEj/count/unspliced`, columns `cI`.
fn backend(sim: &Sim, dir: &std::path::Path) -> String {
    let path = dir.join("tracks.zarr").to_string_lossy().into_owned();
    let (p, g) = (sim.n_pb, sim.n_genes);
    let m = DMatrix::<f32>::from_fn(2 * g, p, |r, c| {
        let x = if r < g {
            sim.s[c * g + r]
        } else {
            sim.u[c * g + r - g]
        };
        x.round() as f32
    });
    let mut b = create_sparse_from_dmatrix(&m, Some(&path), Some(&SparseIoBackend::Zarr)).unwrap();
    let rows: Vec<Box<str>> = (0..g)
        .map(|j| format!("GENE{j}/count/spliced").into())
        .chain((0..g).map(|j| format!("GENE{j}/count/unspliced").into()))
        .collect();
    let cols: Vec<Box<str>> = (0..p).map(|c| format!("c{c}").into()).collect();
    b.register_row_names_vec(&rows);
    b.register_column_names_vec(&cols);
    path
}

/// A small ode run on `data` at `ode`: a trunk, two lineages, `extra`
/// flags besides.
fn small_ode(data: &str, ode: &str, extra: &[&str]) {
    let mut argv = vec![
        data,
        "-o",
        ode,
        "--lineages",
        "2",
        "--points",
        "9",
        "--sub",
        "2",
        "--steps",
        "300",
        "--sweeps",
        "30",
        "--burn",
        "15",
        "--chains",
        "2",
        "--snapshot-every",
        "5",
        "--map-pcs",
        "3",
        "--no-direction",
    ];
    argv.extend_from_slice(extra);
    run_ode(&parse_ode(&argv)).unwrap();
}

fn table(path: &str) -> MatWithNames<Mat> {
    Mat::from_parquet_with_row_names(path, Some(0)).unwrap()
}

/// A small ode run on a trunk and two lineages, then embed-ode from it:
/// every output written, of the right shapes, finite, and a manifest of
/// its own kind naming them.
#[test]
fn a_run_from_an_ode_run_writes_every_table_and_a_manifest() {
    let sim = simulate(&SimConfig {
        n_genes: 40,
        n_pb: 500,
        depth: 3000.0,
        lineages: 2,
        branch_at: Some(0.3),
        seed: 6,
        ..SimConfig::default()
    })
    .unwrap();
    let dir = tempfile::tempdir().unwrap();
    let data = backend(&sim, dir.path());
    let ode = dir.path().join("ode").to_string_lossy().into_owned();
    small_ode(&data, &ode, &[]);

    let out = dir.path().join("emb").to_string_lossy().into_owned();
    let h = 4;
    run_embed_ode(&parse(&[
        "--from",
        &ode,
        "-o",
        &out,
        "--dim",
        "4",
        "--epochs",
        "200",
        "--lr",
        "0.05",
        "--block-size",
        "128",
    ]))
    .unwrap();

    let (m, _) = RunManifest::load(std::path::Path::new(&format!("{out}.senna.json"))).unwrap();
    assert_eq!(m.kind, RunKind::EmbedOde);
    for (slot, file) in [
        (&m.outputs.cell_embedding, "cell_embedding.parquet"),
        (&m.outputs.cell_velocity, "cell_velocity.parquet"),
        (&m.outputs.cell_time, "cell_time.parquet"),
        (&m.outputs.feature_embedding, "feature_embedding.parquet"),
        (&m.outputs.trajectory, "trajectory.parquet"),
        (&m.outputs.cell_to_pb, "cell_to_pb.parquet"),
    ] {
        assert_eq!(slot.as_deref(), Some(format!("emb.{file}").as_str()));
    }
    // Its data the ode run's, found from the manifest.
    assert_eq!(m.data.input.len(), 1);

    // Every cell the ode run trained on, placed, its velocity beside it.
    let emb = table(&format!("{out}.cell_embedding.parquet"));
    let vel = table(&format!("{out}.cell_velocity.parquet"));
    assert!(emb.rows.len() * 100 >= sim.n_pb * 99, "{}", emb.rows.len());
    assert_eq!(emb.mat.ncols(), h);
    assert_eq!(vel.rows, emb.rows);
    assert_eq!(vel.mat.shape(), emb.mat.shape());
    assert!(emb.mat.iter().chain(vel.mat.iter()).all(|v| v.is_finite()));
    assert!(vel.mat.iter().any(|v| *v != 0.0));
    assert_eq!(
        emb.cols,
        vec![
            Box::<str>::from("h0"),
            "h1".into(),
            "h2".into(),
            "h3".into()
        ]
    );

    // Every pseudobulk, named by its ode id, every one in the partition.
    let pb = table(&format!("{out}.pb_embedding.parquet"));
    assert_eq!(pb.mat.ncols(), h);
    assert!(pb.mat.iter().all(|v| v.is_finite()));
    let member = table(&format!("{out}.cell_to_pb.parquet"));
    assert_eq!(member.rows, emb.rows);
    let finest = member.mat.ncols() - 1;
    let ids: std::collections::HashSet<String> = (0..member.mat.nrows())
        .map(|c| (member.mat[(c, finest)] as usize).to_string())
        .collect();
    assert_eq!(ids.len(), pb.rows.len());
    assert!(pb.rows.iter().all(|r| ids.contains(r.as_ref())));

    // Every gene once, by its name, with its bias.
    let genes = table(&format!("{out}.feature_embedding.parquet"));
    assert_eq!(genes.rows.len(), sim.n_genes);
    assert!(genes
        .rows
        .iter()
        .all(|g| g.starts_with("GENE") && !g.contains('/')));
    assert_eq!(genes.mat.ncols(), h);
    let bias = table(&format!("{out}.feature_bias.parquet"));
    assert_eq!(bias.rows, genes.rows);
    assert!(bias
        .mat
        .iter()
        .chain(genes.mat.iter())
        .all(|v| v.is_finite()));

    // The curves, every lineage on the ode run's grid.
    let n = 9 * 2 - 1;
    let (keys, nums) = read_table_columns(
        &format!("{out}.trajectory.parquet"),
        &["lineage"],
        &["position", "phase", "h0", "h1", "h2", "h3"],
    )
    .unwrap();
    assert_eq!(keys[0].len(), 2 * n);
    assert_eq!(keys[0].iter().filter(|l| l.as_ref() == "L2").count(), n);
    assert!(nums[1].iter().all(|u| (0.0..=1.0).contains(u)));
    assert!(nums.iter().flatten().all(|v| v.is_finite()));
    // On the ode run's trunk: both lineages' curves one at its root.
    let root: Vec<usize> = (0..keys[0].len()).filter(|&r| nums[0][r] == 0.0).collect();
    assert_eq!(root.len(), 2);
    for e in &nums[2..] {
        assert!(
            (e[root[0]] - e[root[1]]).abs() < 1e-5,
            "{} vs {}",
            e[root[0]],
            e[root[1]]
        );
    }

    // The ode run's times, as they were.
    assert_eq!(
        std::fs::read(format!("{out}.cell_time.parquet")).unwrap(),
        std::fs::read(format!("{ode}.cell_time.parquet")).unwrap()
    );

    // What `senna view` reads of it besides the velocity (`cell_velocity`
    // row for row with `cell_embedding`, above): the gene table with its
    // bias beside it, for the expected expression.
    let (m, mdir) = RunManifest::load(std::path::Path::new(&format!("{out}.senna.json"))).unwrap();
    let (rho, bias) = senna::run_manifest::resolve_feature_embedding_for(&m, &mdir).unwrap();
    assert!(rho.ends_with("emb.feature_embedding.parquet"), "{rho}");
    assert!(bias.is_some_and(|b| b.ends_with("emb.feature_bias.parquet")));
}

/// An ode run on batches writes the collapse's batch effects, and
/// embed-ode reads them on its data: the effects the cells and
/// pseudobulks are summed over.
#[test]
fn an_ode_run_on_batches_leaves_its_batch_effects_for_embed_ode() {
    let sim = simulate(&SimConfig {
        n_genes: 30,
        n_pb: 300,
        depth: 3000.0,
        lineages: 2,
        branch_at: Some(0.3),
        seed: 7,
        ..SimConfig::default()
    })
    .unwrap();
    let dir = tempfile::tempdir().unwrap();
    let data = backend(&sim, dir.path());
    let batches = dir.path().join("batch.txt");
    let labels: Vec<&str> = (0..sim.n_pb)
        .map(|c| if c % 2 == 0 { "b0" } else { "b1" })
        .collect();
    std::fs::write(&batches, labels.join("\n") + "\n").unwrap();
    let ode = dir.path().join("ode").to_string_lossy().into_owned();
    small_ode(&data, &ode, &["-b", &batches.to_string_lossy()]);

    // Its own prefix as -o is refused, the run left as it was.
    let before = std::fs::read(format!("{ode}.cell_time.parquet")).unwrap();
    let e = run_embed_ode(&parse(&["--from", &ode, "-o", &ode]))
        .unwrap_err()
        .to_string();
    assert!(e.contains("own prefix"), "{e}");
    assert_eq!(
        std::fs::read(format!("{ode}.cell_time.parquet")).unwrap(),
        before
    );

    let run = crate::embed_ode::inputs::OdeRun::open(&ode).unwrap();
    let path = run.delta_path().expect("the ode run's batch effects");
    let (files, batch_files) = (
        run.manifest.data_inputs(&run.dir),
        run.manifest.data_batches(&run.dir),
    );
    let loaded = crate::embed_ode::inputs::load_data(&files, Some(&batch_files), false).unwrap();
    let delta = crate::embed_ode::inputs::read_delta(&path, &loaded.data).unwrap();
    assert_eq!(delta.ncols(), 2);
    assert!(delta.iter().all(|v| v.is_finite() && *v > 0.0));
}
