use super::init::spearman;
use crate::ode_mixture::command::{run_ode, OdeArgs};
use crate::ode_mixture::sim::{simulate, Sim, SimConfig};
use clap::Parser;
use data_beans::sparse_io::{create_sparse_from_dmatrix, SparseIoBackend};
use legume_numeric::matrix::parquet::read_table_columns;
use legume_numeric::matrix::traits::IoOps;
use nalgebra::DMatrix;
use senna::run_manifest::{RunKind, RunManifest};

#[derive(Parser)]
struct Cli {
    #[command(flatten)]
    args: OdeArgs,
}

fn parse(argv: &[&str]) -> OdeArgs {
    let mut full = vec!["ode"];
    full.extend_from_slice(argv);
    Cli::try_parse_from(full).expect("valid argv").args
}

/// The simulation's pseudobulks as cells of a two-track backend: rows
/// `GENEj/count/spliced` then `GENEj/count/unspliced`, columns `cI`;
/// `GENE0` has no unspliced reads at all.
fn backend(sim: &Sim, dir: &std::path::Path) -> String {
    let path = dir.join("tracks.zarr").to_string_lossy().into_owned();
    let (p, g) = (sim.n_pb, sim.n_genes);
    let m = DMatrix::<f32>::from_fn(2 * g, p, |r, c| {
        let x = if r < g {
            sim.s[c * g + r]
        } else {
            if r == g {
                0.0
            } else {
                sim.u[c * g + r - g]
            }
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

/// A run writes every cell's time, map place and velocity, every gene's
/// rates with their chains' diagnostics, and a manifest the viewer reads;
/// the cells' times rank with the truth's.
#[test]
fn a_run_writes_the_cells_tables_and_a_manifest() {
    let sim = simulate(&SimConfig {
        n_genes: 80,
        n_pb: 1000,
        depth: 3000.0,
        seed: 4,
        ..SimConfig::default()
    })
    .unwrap();
    let dir = tempfile::tempdir().unwrap();
    let data = backend(&sim, dir.path());
    let out = dir.path().join("run").to_string_lossy().into_owned();
    run_ode(&parse(&[
        data.as_str(),
        "-o",
        &out,
        "--lineages",
        "1",
        "--points",
        "25",
        "--steps",
        "1500",
        "--sweeps",
        "80",
        "--burn",
        "40",
        "--chains",
        "2",
        "--snapshot-every",
        "10",
        "--map-pcs",
        "5",
    ]))
    .unwrap();

    let (strs, nums) = read_table_columns(
        &format!("{out}.cell_time.parquet"),
        &["cell", "lineage"],
        &["tau", "tau_sd", "tau_lo", "tau_hi", "p_L1"],
    )
    .unwrap();
    let cells = &strs[0];
    // QC may leave a few near-empty cells out.
    assert!(cells.len() <= sim.n_pb && cells.len() * 100 >= sim.n_pb * 99);
    assert!(strs[1].iter().all(|l| l.as_ref() == "L1"));
    for (name, col) in ["tau", "tau_sd", "tau_lo", "tau_hi", "p_L1"]
        .iter()
        .zip(&nums)
    {
        assert!(col.iter().all(|x| x.is_finite()), "{name} not finite");
    }
    // Rows are named; put the truth in their order.
    let truth: Vec<f64> = cells
        .iter()
        .map(|c| sim.truth.time[c[1..].parse::<usize>().unwrap()])
        .collect();
    let tau = nums[0].clone();
    let rho = spearman(&tau, &truth).abs();
    eprintln!("senna ode: time Spearman with the truth {rho:.3}");
    assert!(rho > 0.9, "Spearman {rho}");

    let emb = senna::embed_common::Mat::from_parquet_with_row_names(
        &format!("{out}.cell_embedding.parquet"),
        Some(0),
    )
    .unwrap();
    let vel = senna::embed_common::Mat::from_parquet_with_row_names(
        &format!("{out}.cell_velocity.parquet"),
        Some(0),
    )
    .unwrap();
    assert_eq!(emb.rows, vel.rows);
    assert_eq!((emb.mat.nrows(), emb.mat.ncols()), (cells.len(), 5));
    assert_eq!(vel.mat.shape(), emb.mat.shape());
    assert!(vel.mat.iter().all(|x| x.is_finite()));

    let (genes, stats) = read_table_columns(
        &format!("{out}.gene_rates.parquet"),
        &["gene"],
        &[
            "log_gamma",
            "log_gamma_sd",
            "log_gamma_rhat",
            "log_gamma_ess",
            "log_ratio",
            "log_ratio_sd",
            "log_ratio_rhat",
            "log_ratio_ess",
            "flagged",
        ],
    )
    .unwrap();
    // Every gene but the one with no unspliced reads: the pseudobulks'
    // reads are the observed ones, no prior mass to pass `--min-reads`.
    assert_eq!(genes[0].len(), sim.n_genes - 1);
    assert!(genes[0]
        .iter()
        .all(|g| g.starts_with("GENE") && g.as_ref() != "GENE0"));
    assert!(stats[8].iter().all(|&f| f == 0.0 || f == 1.0));

    // Every lineage's direction evidence; a lineage with kinetics is called.
    let lineages = std::fs::read_to_string(format!("{out}.lineages.tsv")).unwrap();
    let mut lines = lineages.lines();
    assert_eq!(
        lines.next(),
        Some("lineage\tevidence\tper_pseudobulk\tdirection\tturned")
    );
    let row: Vec<&str> = lines.next().unwrap().split('\t').collect();
    // Called either way; a lone lineage (no trunk) the evidence reverses
    // is turned round, one it does not is left as climbed.
    assert_eq!(row[0], "L1");
    assert!(["forward", "reversed"].contains(&row[3]), "{row:?}");
    assert_eq!(row[4], if row[3] == "reversed" { "yes" } else { "no" });
    assert!(row[1].parse::<f64>().unwrap().is_finite());
    assert!(lines.next().is_none());

    let (m, _) = RunManifest::load(std::path::Path::new(&format!("{out}.senna.json"))).unwrap();
    assert_eq!(m.kind, RunKind::Ode);
    for (slot, file) in [
        (&m.outputs.cell_time, "cell_time"),
        (&m.outputs.cell_velocity, "cell_velocity"),
        (&m.outputs.cell_embedding, "cell_embedding"),
    ] {
        assert_eq!(
            slot.as_deref(),
            Some(format!("run.{file}.parquet").as_str())
        );
    }
}
