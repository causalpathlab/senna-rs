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
/// rates with their sds, and a manifest the viewer reads;
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
    // No direction is claimed: which end a lineage starts at is not
    // identifiable from the reads.
    assert!(
        read_table_columns(&format!("{out}.cell_time.parquet"), &["direction"], &[]).is_err(),
        "a direction column"
    );
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
        &["log_gamma", "log_gamma_sd", "log_ratio", "log_ratio_sd"],
    )
    .unwrap();
    // Every gene but the one with no unspliced reads: the pseudobulks'
    // reads are the observed ones, no prior mass to pass `--min-reads`.
    assert_eq!(genes[0].len(), sim.n_genes - 1);
    assert!(genes[0]
        .iter()
        .all(|g| g.starts_with("GENE") && g.as_ref() != "GENE0"));
    assert!(stats.iter().flatten().all(|x| x.is_finite()));
    assert!(stats[1].iter().chain(&stats[3]).all(|&sd| sd > 0.0));

    // Nor a table of direction calls.
    assert!(!std::path::Path::new(&format!("{out}.lineages.tsv")).exists());

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

    // Every pseudobulk's posterior over the grid, the cells' pseudobulks
    // among them; no branch with one lineage.
    assert_eq!(m.outputs.pb_time.as_deref(), Some("run.pb_time.parquet"));
    assert_eq!(m.outputs.branch_time, None);
    let (keys, nums) = read_table_columns(
        &format!("{out}.pb_time.parquet"),
        &["pb", "lineage"],
        &["position", "phase", "q"],
    )
    .unwrap();
    assert!(keys[1].iter().all(|l| l.as_ref() == "L1"));
    let mut mass = std::collections::HashMap::<&str, f64>::new();
    for (pb, q) in keys[0].iter().zip(&nums[2]) {
        *mass.entry(pb.as_ref()).or_default() += q;
    }
    assert!(
        mass.values().all(|m| (m - 1.0).abs() < 1e-3),
        "{:?}",
        mass.values().take(3).collect::<Vec<_>>()
    );
    // The phase on the grid, each lineage's own time.
    assert!(nums[1].iter().all(|u| (0.0..=1.0).contains(u)));
    let member = senna::embed_common::Mat::from_parquet_with_row_names(
        &format!("{out}.cell_to_pb.parquet"),
        Some(0),
    )
    .unwrap();
    let finest = member.mat.ncols() - 1;
    for c in 0..member.mat.nrows() {
        let pb = (member.mat[(c, finest)] as usize).to_string();
        assert!(mass.contains_key(pb.as_str()), "cell {c}'s pseudobulk {pb}");
    }
}

/// The defaults: lineages apart, each on its own time, no trunk, no root.
#[test]
fn the_defaults_put_the_lineages_apart() {
    let args = parse(&["tracks.zarr", "-o", "out"]);
    assert!(!args.trunk);
    assert!(args.root_cells.is_none());
}

/// A run on groups of genes (`--max-coarse-features`) still writes every
/// gene's own rates, records its groups, and its cells' times rank with
/// the truth's.
#[test]
fn a_coarsened_run_writes_every_genes_rates() {
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
        "--map-pcs",
        "5",
        "--max-coarse-features",
        "20",
    ]))
    .unwrap();

    let levels = crate::topic::model_metadata::load_coarsening_levels(&out)
        .unwrap()
        .expect("the groups recorded");
    let fc = levels.last().unwrap().as_ref().expect("a coarsening");
    assert_eq!(fc.fine_to_coarse.len(), sim.n_genes - 1);
    assert!(fc.num_coarse <= 20, "{} groups", fc.num_coarse);

    let (genes, stats) = read_table_columns(
        &format!("{out}.gene_rates.parquet"),
        &["gene"],
        &["log_gamma", "log_ratio"],
    )
    .unwrap();
    assert_eq!(genes[0].len(), sim.n_genes - 1);
    assert!(stats.iter().flatten().all(|x| x.is_finite()));
    // Genes in one group are drawn on their own.
    let distinct: std::collections::HashSet<u64> = stats[0].iter().map(|x| x.to_bits()).collect();
    assert!(
        distinct.len() > fc.num_coarse,
        "{} distinct rates",
        distinct.len()
    );

    let (strs, nums) =
        read_table_columns(&format!("{out}.cell_time.parquet"), &["cell"], &["tau"]).unwrap();
    let truth: Vec<f64> = strs[0]
        .iter()
        .map(|c| sim.truth.time[c[1..].parse::<usize>().unwrap()])
        .collect();
    let rho = spearman(&nums[0], &truth).abs();
    eprintln!("senna ode on groups: time Spearman with the truth {rho:.3}");
    assert!(rho > 0.9, "Spearman {rho}");
}
