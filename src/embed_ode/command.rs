//! `senna embed-ode`: read an ode run and its count backends (every gene's
//! spliced reads), sum the cells into the ode run's own finest
//! pseudobulks, fit the embedding with the curves on them
//! ([`super::model`]), then place every cell with the genes and the curves
//! fixed ([`super::place`]), a block at a time.

use super::curves::{grid_velocity, CurveSpec, Curves};
use super::inputs::{
    for_count_blocks, load_data, partition_on, pb_sums, read_branch_times, read_delta,
    read_pb_time, OdeRun,
};
use super::model::{Model, Pseudobulks, QMass};
use super::place::{expect_on_grid, grid_prior, Placer};
use crate::ode_mixture::model::{lineage_label, lineage_p_column};
use legume_numeric::matrix::parquet::{read_table_columns, write_named_table, Column};
use log::info;
use nalgebra::DMatrix;
use senna::embed_common::*;
use std::collections::HashMap;

#[derive(Args, Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(default = "senna::embed_common::clap_defaults")]
pub struct EmbedOdeArgs {
    #[arg(
        value_delimiter = ',',
        help = "Count backends (default: the ode run's own)",
        long_help = "Count backends, read on every gene's `{gene}/count/spliced` row (a\n\
                     backend without tracks, on its rows as they are). Default: the\n\
                     files the ode run read. Cells are matched to the ode run's\n\
                     pseudobulks by name."
    )]
    pub(crate) data_files: Vec<Box<str>>,

    #[arg(
        long,
        required = true,
        help = "The ode run (its prefix or .senna.json)"
    )]
    pub(crate) from: Box<str>,

    #[arg(short, long, required = true, help = "Output prefix")]
    pub(crate) out: Box<str>,

    #[arg(
        long,
        short,
        value_delimiter(','),
        help = "Batch membership files, one per data file (default: the ode run's)"
    )]
    pub(crate) batch_files: Option<Vec<Box<str>>>,

    #[arg(long, default_value_t = 10, help = "Dimension of the embedding")]
    pub dim: usize,

    #[arg(
        long,
        default_value_t = 2000,
        help = "AdamW epochs of the pseudobulk fit"
    )]
    pub epochs: usize,

    #[arg(long, default_value_t = 0.02, help = "AdamW's learning rate")]
    pub lr: f64,

    #[arg(
        long,
        help = "Pseudobulks a step (default: every one, a step an epoch)"
    )]
    pub pb_per_step: Option<usize>,

    #[arg(long, help = "Most variable genes to embed (default: every gene)")]
    pub max_features: Option<usize>,

    #[arg(long, default_value_t = 1, help = "Random seed")]
    pub seed: u64,

    #[arg(long, default_value_t = ComputeDevice::Cpu, value_enum, help = "Compute device")]
    pub device: ComputeDevice,

    #[arg(long, default_value_t = 0, help = "CUDA/Metal device index")]
    pub device_no: usize,

    #[arg(long, default_value_t = 1024, help = "Cells per block")]
    pub(crate) block_size: usize,
}

/// `[rows × h]` row-major as a table keyed `key`, columns `h0…`.
fn write_embedding(
    path: &str,
    key: &str,
    rows: &[Box<str>],
    values: &[f32],
    h: usize,
) -> anyhow::Result<()> {
    Mat::from_row_slice(rows.len(), h, values).to_parquet_with_names(
        path,
        (Some(rows), Some(key)),
        Some(&axis_id_names("h", h)),
    )
}

/// Every cell's lineage posteriors and time, from the ode run's
/// `cell_time.parquet`, by name.
struct CellTimes {
    at: HashMap<Box<str>, usize>,
    /// `[L][C]`.
    p: Vec<Vec<f32>>,
    tau: Vec<f32>,
    tau_sd: Vec<f32>,
}

impl CellTimes {
    fn read(path: &str, lineages: usize) -> anyhow::Result<Self> {
        let p_cols: Vec<Box<str>> = (0..lineages).map(lineage_p_column).collect();
        let mut nums: Vec<&str> = vec!["tau", "tau_sd"];
        nums.extend(p_cols.iter().map(AsRef::as_ref));
        let (keys, cols) = read_table_columns(path, &["cell"], &nums)?;
        let f32s = |v: &[f64]| v.iter().map(|&x| x as f32).collect::<Vec<_>>();
        Ok(Self {
            at: keys[0]
                .iter()
                .enumerate()
                .map(|(i, c)| (c.clone(), i))
                .collect(),
            tau: f32s(&cols[0]),
            tau_sd: f32s(&cols[1]),
            p: cols[2..].iter().map(|c| f32s(c)).collect(),
        })
    }

    /// The cell's posterior on the grid, if it has one.
    fn prior(&self, cell: &str, duration: &[f64], n: usize) -> Option<Vec<(usize, usize, f32)>> {
        let &i = self.at.get(cell)?;
        let p: Vec<f32> = self.p.iter().map(|col| col[i]).collect();
        grid_prior(&p, self.tau[i], self.tau_sd[i], duration, n)
    }

    /// The largest time, the duration of a lineage `pb_time` gives none.
    fn longest(&self) -> f64 {
        let t = self
            .tau
            .iter()
            .filter(|t| t.is_finite())
            .fold(0f32, |a, &b| a.max(b));
        if t > 0.0 {
            f64::from(t)
        } else {
            1.0
        }
    }
}

/// The run's data and batch files: as given, or the ode run's.
fn inputs(args: &EmbedOdeArgs, ode: &OdeRun) -> (Vec<Box<str>>, Option<Vec<Box<str>>>) {
    if args.data_files.is_empty() {
        let batches = ode.manifest.data_batches(&ode.dir);
        (
            ode.manifest.data_inputs(&ode.dir),
            args.batch_files
                .clone()
                .or_else(|| (!batches.is_empty()).then_some(batches)),
        )
    } else {
        (args.data_files.clone(), args.batch_files.clone())
    }
}

/// The curves' grid and trunk, the ode run's.
fn curve_spec(ode: &OdeRun) -> anyhow::Result<CurveSpec> {
    let a = &ode.args;
    let branch = match ode.path(ode.manifest.outputs.branch_time.as_deref()) {
        Some(path) if a.lineages > 1 && !a.no_trunk => Some(read_branch_times(&path, a.lineages)?),
        _ => None,
    };
    Ok(CurveSpec {
        lineages: a.lineages,
        points: a.points,
        sub: a.sub,
        kappa: a.kappa,
        branch,
    })
}

pub fn run_embed_ode(args: &EmbedOdeArgs) -> anyhow::Result<()> {
    mkdir_parent(&args.out)?;
    anyhow::ensure!(
        args.dim > 0 && args.block_size > 0,
        "--dim and --block-size must be positive"
    );
    let dev = args.device.to_device(args.device_no)?;
    let ode = OdeRun::open(&args.from)?;
    anyhow::ensure!(
        !ode.is_at(&args.out),
        "-o {}: the ode run's own prefix; its outputs would be overwritten",
        args.out
    );
    let (files, batch_files) = inputs(args, &ode);

    // The data on the counted rows, every gene's spliced reads.
    let mut loaded = load_data(
        &files,
        batch_files.as_deref(),
        ode.args.collapse.ignore_batch,
    )?;
    if let Some(k) = args.max_features.filter(|&k| k < loaded.genes.len()) {
        let sel = data_beans::alg::hvg::select_hvg_streaming(
            &loaded.data,
            Some(k),
            None,
            None,
            Some(args.block_size),
        )?;
        let mut keep = vec![false; loaded.genes.len()];
        sel.selected_indices.iter().for_each(|&i| keep[i] = true);
        loaded.data.mask_rows(&keep)?;
        loaded.genes = sel
            .selected_indices
            .iter()
            .map(|&i| loaded.genes[i].clone())
            .collect();
    }
    let (data, genes) = (loaded.data, loaded.genes);
    let cells = data.column_names()?;

    // The ode run's finest pseudobulks on these cells.
    let member_path = ode
        .manifest
        .cell_to_pb_path(&ode.dir)
        .ok_or_else(|| anyhow::anyhow!("{}: no cell_to_pb in the ode run", args.from))?;
    let (levels, src_cells) = senna::run_manifest::load_cell_to_pb_raw(&member_path)?;
    let finest = levels.last().expect("at least one level");
    let part = partition_on(finest, &src_cells, &cells)?;
    let delta = ode
        .delta_path()
        .map(|p| read_delta(&p, &data))
        .transpose()?;
    let (p_n, g) = (part.ids.len(), genes.len());
    let y = pb_sums(&data, &part.of_cell, p_n, args.block_size, delta.as_ref())?;

    // The grid and the ode run's posteriors on it.
    let spec = curve_spec(&ode)?;
    let n = spec.positions();
    let pb_time = read_pb_time(
        &ode.need(ode.manifest.outputs.pb_time.as_deref(), "pb_time")?,
        &part.ids,
        spec.lineages,
        n,
    )?;
    let pbs = Pseudobulks::new(y, p_n, g, pb_time.q)?;
    let with_prior = pbs.has_prior().iter().filter(|&&h| h).count();
    anyhow::ensure!(
        with_prior > 0,
        "no pseudobulk of the ode run's partition has a time in its pb_time"
    );
    info!(
        "embed-ode: {} cells, {p_n} pseudobulks ({with_prior} with a prior), {g} genes",
        part.of_cell.iter().flatten().count()
    );

    // The fit on the pseudobulks.
    let curves = Curves::new(&spec, &dev)?;
    if let Some(tb) = curves.branch_times() {
        info!("embed-ode: the ode run's branch times {tb:.3?} (on phase)");
    }
    if curves.on_nodes() {
        info!("embed-ode: the curves on the grid's nodes, straight between them");
    }
    let model = Model::init(&pbs, curves, args.dim, args.seed, &dev)?;
    let trace = model.fit(args.epochs, args.lr, args.pb_per_step, args.seed)?;
    if let (Some(first), Some(last)) = (trace.first(), trace.last()) {
        info!(
            "embed-ode: loss {first:.1} → {last:.1} over {} epochs",
            trace.len()
        );
    }
    let h = args.dim;
    let (x, mu) = (model.x()?, model.mu()?);
    let sigma = model.sigma()?;
    info!(
        "embed-ode: σ {sigma:.4}, curve amplitude {:.4}",
        model.amplitude()?
    );
    let velocity = grid_velocity(&mu, spec.lineages, n, h);

    // Every cell placed, its curve's slope at it.
    let cell_time = ode.need(ode.manifest.outputs.cell_time.as_deref(), "cell_time")?;
    let times = CellTimes::read(&cell_time, spec.lineages)?;
    let duration: Vec<f64> = pb_time
        .duration
        .iter()
        .map(|d| d.unwrap_or_else(|| times.longest()))
        .collect();
    let placer = Placer::new(&model.w()?, &model.b()?, h, &dev)?;
    let inv_var = 1.0 / (sigma * sigma);
    let mut placed = Placed::default();
    for_count_blocks(&data, args.block_size, delta.as_ref(), |start, reads| {
        let rows: Vec<usize> = (0..reads.nrows())
            .filter(|&i| part.of_cell[start + i].is_some())
            .collect();
        if rows.is_empty() {
            return Ok(());
        }
        let mut centre = Vec::with_capacity(rows.len() * h);
        for &i in &rows {
            let c = start + i;
            let p = part.of_cell[c].expect("in the partition");
            // Its own posterior, else its pseudobulk's, else its
            // pseudobulk's point.
            let own = times.prior(&cells[c], &duration, n);
            let of_pb = pbs.of(p);
            let on = |table: &[f32]| match &own {
                Some(q) => expect_on_grid(q.iter().copied(), table, n, h),
                None => expect_on_grid(of_pb.iter().map(QMass::on_grid), table, n, h),
            };
            if own.is_some() || !of_pb.is_empty() {
                centre.extend(on(&mu));
                placed.velocity.extend(on(&velocity));
            } else {
                centre.extend_from_slice(&x[p * h..(p + 1) * h]);
                placed.velocity.extend(std::iter::repeat_n(0f32, h));
            }
            placed.cells.push(c);
        }
        let block: DMatrix<f32> = reads.select_rows(&rows);
        placed.theta.extend(placer.place(&block, &centre, inv_var)?);
        Ok(())
    })?;

    let out = &args.out;
    let names: Vec<Box<str>> = placed.cells.iter().map(|&c| cells[c].clone()).collect();
    let pb_names: Vec<Box<str>> = part.ids.iter().map(|p| p.to_string().into()).collect();
    write_embeddings(out, &model, &placed, &names, &pb_names, &genes)?;
    write_trajectory(out, &model)?;
    // The ode run's times, and its partition on the cells placed.
    std::fs::copy(&cell_time, format!("{out}.cell_time.parquet"))?;
    write_partition(out, &levels, &src_cells, &names)?;
    write_manifest(args, &files, batch_files.as_deref())?;
    info!(
        "Wrote {0}.{{cell_embedding,cell_velocity,pb_embedding,feature_embedding,feature_bias,trajectory,cell_time,cell_to_pb}}.parquet and {0}.senna.json",
        args.out
    );
    Ok(())
}

/// The cells placed, in the data's order.
#[derive(Default)]
struct Placed {
    /// Data column of every one.
    cells: Vec<usize>,
    /// `[C × H]` row-major.
    theta: Vec<f32>,
    velocity: Vec<f32>,
}

/// The cells', pseudobulks' and genes' tables: `names` the cells placed,
/// `pb_names` the pseudobulks.
fn write_embeddings(
    out: &str,
    model: &Model,
    placed: &Placed,
    names: &[Box<str>],
    pb_names: &[Box<str>],
    genes: &[Box<str>],
) -> anyhow::Result<()> {
    let h = model.dim();
    write_embedding(
        &format!("{out}.cell_embedding.parquet"),
        "cell",
        names,
        &placed.theta,
        h,
    )?;
    write_embedding(
        &format!("{out}.cell_velocity.parquet"),
        "cell",
        names,
        &placed.velocity,
        h,
    )?;
    write_embedding(
        &format!("{out}.pb_embedding.parquet"),
        "pb",
        pb_names,
        &model.x()?,
        h,
    )?;
    write_embedding(
        &format!("{out}.feature_embedding.parquet"),
        "feature",
        genes,
        &model.w()?,
        h,
    )?;
    let b = model.b()?;
    Mat::from_column_slice(b.len(), 1, &b).to_parquet_with_names(
        &format!("{out}.feature_bias.parquet"),
        (Some(genes), Some("feature")),
        Some(&[Box::from("bias")]),
    )
}

/// The curves, a row per lineage and position.
fn write_trajectory(out: &str, model: &Model) -> anyhow::Result<()> {
    let (curves, h) = (model.curves(), model.dim());
    let (n, mu) = (curves.positions(), model.mu()?);
    let rows = curves.lineages() * n;
    let lineage: Vec<Box<str>> = (0..rows).map(|r| lineage_label(r / n)).collect();
    let position: Vec<i32> = (0..rows).map(|r| (r % n) as i32).collect();
    let phase: Vec<f32> = (0..rows).map(|r| curves.phase(r % n) as f32).collect();
    let coords: Vec<(Box<str>, Vec<f32>)> = axis_id_names("h", h)
        .into_iter()
        .enumerate()
        .map(|(d, name)| (name, (0..rows).map(|r| mu[r * h + d]).collect()))
        .collect();
    let mut cols = vec![
        ("position".into(), Column::I32(&position)),
        ("phase".into(), Column::F32(&phase)),
    ];
    cols.extend(coords.iter().map(|(k, v)| (k.clone(), Column::F32(v))));
    write_named_table(
        &format!("{out}.trajectory.parquet"),
        "lineage",
        &lineage,
        &cols,
    )
}

/// The ode run's partition, every level of `levels` (on its cells
/// `src_cells`), on the cells `names`.
fn write_partition(
    out: &str,
    levels: &[Vec<usize>],
    src_cells: &[Box<str>],
    names: &[Box<str>],
) -> anyhow::Result<()> {
    let at: HashMap<&str, usize> = src_cells
        .iter()
        .enumerate()
        .map(|(i, c)| (c.as_ref(), i))
        .collect();
    let src_row: Vec<usize> = names.iter().map(|c| at[c.as_ref()]).collect();
    let member: Vec<Vec<usize>> = levels
        .iter()
        .map(|lvl| src_row.iter().map(|&i| lvl[i]).collect())
        .collect();
    crate::postprocess::viz_prep::write_cell_to_pb(out, &member, names, None)?;
    Ok(())
}

/// `{out}.senna.json`.
fn write_manifest(
    args: &EmbedOdeArgs,
    files: &[Box<str>],
    batch_files: Option<&[Box<str>]>,
) -> anyhow::Result<()> {
    use senna::run_manifest::{
        record_train_args, write_run_manifest_with, RunDescription, RunKind,
    };
    let input: Vec<String> = files.iter().map(ToString::to_string).collect();
    let batch: Vec<String> = batch_files
        .map(|v| v.iter().map(ToString::to_string).collect())
        .unwrap_or_default();
    write_run_manifest_with(
        &RunDescription {
            train_args: Some(record_train_args(args)?),
            kind: RunKind::EmbedOde,
            prefix: &args.out,
            data_input: &input,
            data_multiome: None,
            data_batch: &batch,
            data_input_null: &[],
            dictionary_suffix: None,
            has_model: false,
            has_cell_proj: false,
            pb_gene_suffix: None,
            pb_latent_suffix: None,
            pb_reference_suffix: None,
            dictionary_empirical_suffix: None,
            feature_embedding_suffix: Some("feature_embedding.parquet"),
            feature_coembedding_suffix: None,
            carried: None,
            module_membership_suffix: None,
            module_dictionary_suffix: None,
            softmax_dictionary_suffix: None,
            cell_embedding_suffix: Some("cell_embedding.parquet"),
            cell_encoder_suffix: None,
            default_colour_by: "cluster",
            has_latent: false,
            has_cell_to_pb: true,
            has_pb_tree: false,
        },
        |m, base| {
            m.outputs.cell_time = Some(format!("{base}.cell_time.parquet"));
            m.outputs.cell_velocity = Some(format!("{base}.cell_velocity.parquet"));
            m.outputs.trajectory = Some(format!("{base}.trajectory.parquet"));
        },
    )
}
