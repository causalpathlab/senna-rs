//! What stage 2 reads from a finished stage-1 run: each cell's place in the
//! run's cell table (in a Euclidean view of its kind), its finest pseudobulk, its batch, and its donor; and the raw
//! counts the run was fitted on.

use super::DonorDeviationArgs;
use data_beans::aux::data_loading::{read_data_on_shared_rows, ReadSharedRowsArgs};
use rustc_hash::FxHashMap as HashMap;
use senna::embed_common::*;
use senna::run_manifest::{self, CellSpace, RunKind};

/// The run's cells, in its embedding table's row order.
pub struct Stage1Cells {
    pub cell_names: Vec<Box<str>>,
    /// `[cells × dims]`, the run's cell table in a Euclidean view
    /// ([`euclidean_view`]): where cells are compared and the state axes are
    /// taken.
    pub embedding: Mat,
    /// Whether the view's rows have unit length (an embedding, or √θ), so a
    /// mean of rows is renormalized to stay in it.
    pub unit_rows: bool,
    /// Each cell's pseudobulk at the run's finest level.
    pub finest_pb_of_cell: Vec<usize>,
    pub batch_of_cell: Vec<Box<str>>,
    /// The donors, sorted, and each cell's donor as an index into them.
    pub donor_names: Vec<Box<str>>,
    pub donor_index_of_cell: Vec<usize>,
}

/// The stage-1 cells and the raw counts (every cell of the data, the run's
/// cells among them).
pub struct Stage1Inputs {
    pub cells: Stage1Cells,
    pub data: SparseIoVec,
    /// Each data column's row in `cells`, `None` for a column the run did
    /// not embed (a cell its QC held back from the outputs).
    pub cell_of_column: Vec<Option<usize>>,
}

pub fn read_stage1_inputs(args: &DonorDeviationArgs) -> anyhow::Result<Stage1Inputs> {
    let (manifest, dir) = run_manifest::load_for(&args.from)?;
    anyhow::ensure!(
        manifest.kind != RunKind::Ode,
        "{}: an ode run's counts are two tracks (spliced and unspliced), not one gene axis",
        args.from
    );
    let resolve = |rel: &str| {
        run_manifest::resolve(&dir, rel)
            .to_string_lossy()
            .into_owned()
    };
    let embedding_path = manifest
        .outputs
        .geometry_latent()
        .map(resolve)
        .ok_or_else(|| anyhow::anyhow!("{}: the run records no cell embedding", args.from))?;
    let partition_path = manifest.cell_to_pb_path(&dir).ok_or_else(|| {
        anyhow::anyhow!(
            "{}: the run records no cell → pseudobulk partition (refit it with this senna, \
             which writes {{out}}.cell_to_pb.parquet)",
            args.from
        )
    })?;

    let named = Mat::from_parquet_with_row_names(&embedding_path, Some(0))?;
    let cell_names = named.rows;
    let cell_space = manifest.kind.cell_space();
    let (embedding, unit_rows) = euclidean_view(named.mat, cell_space);
    info!(
        "Stage-1 cells {embedding_path}: {} cells × {} dims ({} run, {cell_space:?} space)",
        embedding.nrows(),
        embedding.ncols(),
        manifest.kind.as_str()
    );

    let (levels, partition_cells) = run_manifest::load_cell_to_pb_raw(&partition_path)?;
    let finest = levels.last().expect("a partition has a level");
    let pb_of_name: HashMap<&str, usize> = partition_cells
        .iter()
        .zip(finest)
        .map(|(name, &pb)| (name.as_ref(), pb))
        .collect();
    let finest_pb_of_cell = cell_names
        .iter()
        .map(|name| {
            pb_of_name.get(name.as_ref()).copied().ok_or_else(|| {
                anyhow::anyhow!("cell {name} of the embedding is not in {partition_path}")
            })
        })
        .collect::<anyhow::Result<Vec<usize>>>()?;

    let data_files = match &args.data {
        Some(files) => files.clone(),
        None => manifest.data_inputs(&dir),
    };
    anyhow::ensure!(
        !data_files.is_empty(),
        "{}: the run records no data files; pass --data",
        args.from
    );
    let batch_files = match &args.batch_files {
        Some(files) => files.clone(),
        None => manifest.data_batches(&dir),
    };
    anyhow::ensure!(
        !batch_files.is_empty(),
        "{}: the run records no batch files; pass --batch-files (stage 2 needs each cell's \
         batch)",
        args.from
    );
    // The run's multiome layout applies to the files it recorded, not to
    // files given with --data.
    let multiome = match args.data {
        Some(_) => None,
        None => manifest.data.multiome.as_ref(),
    };
    let reload = senna::multiome_layout::recorded_layout(multiome, data_files.len())?;
    let loaded = read_data_on_shared_rows(reload.apply(ReadSharedRowsArgs {
        data_files,
        batch_files: Some(batch_files),
        ..Default::default()
    })?)?;
    let data = loaded.data;
    let column_names = data.column_names()?;
    anyhow::ensure!(
        loaded.batch.len() == column_names.len(),
        "{} batch labels for {} cells",
        loaded.batch.len(),
        column_names.len()
    );
    let cell_of_name: HashMap<&str, usize> = cell_names
        .iter()
        .enumerate()
        .map(|(i, name)| (name.as_ref(), i))
        .collect();
    let cell_of_column: Vec<Option<usize>> = column_names
        .iter()
        .map(|name| cell_of_name.get(name.as_ref()).copied())
        .collect();
    let mut batch_of_cell: Vec<Option<Box<str>>> = vec![None; cell_names.len()];
    for (column, cell) in cell_of_column.iter().enumerate() {
        if let Some(cell) = *cell {
            batch_of_cell[cell] = Some(loaded.batch[column].clone());
        }
    }
    let batch_of_cell = batch_of_cell
        .into_iter()
        .zip(&cell_names)
        .map(|(batch, name)| {
            batch.ok_or_else(|| anyhow::anyhow!("cell {name} of the embedding is not in the data"))
        })
        .collect::<anyhow::Result<Vec<Box<str>>>>()?;

    let donor_of_batch = match &args.donor_of_batch {
        Some(path) => Some(read_donor_of_batch(path)?),
        None => {
            info!("No --donor-of-batch: each batch is its own donor");
            None
        }
    };
    let donor_of_cell = batch_of_cell
        .iter()
        .map(|batch| match &donor_of_batch {
            Some(map) => map
                .get(batch)
                .cloned()
                .ok_or_else(|| anyhow::anyhow!("batch {batch} has no donor in --donor-of-batch")),
            None => Ok(batch.clone()),
        })
        .collect::<anyhow::Result<Vec<Box<str>>>>()?;
    let mut donor_names = donor_of_cell.clone();
    donor_names.sort_unstable();
    donor_names.dedup();
    anyhow::ensure!(
        donor_names.len() >= 3,
        "{} donors; stage 2 needs at least three (a donor is contrasted with the others)",
        donor_names.len()
    );
    let donor_index_of_cell = donor_of_cell
        .iter()
        .map(|d| donor_names.binary_search(d).expect("a listed donor"))
        .collect();

    Ok(Stage1Inputs {
        cells: Stage1Cells {
            cell_names,
            embedding,
            unit_rows,
            finest_pb_of_cell,
            batch_of_cell,
            donor_names,
            donor_index_of_cell,
        },
        data,
        cell_of_column,
    })
}

/// The run's cell table in a space where Euclidean distance compares cells
/// the way the run's kind calls for (as `senna critique` compares them), and
/// whether its rows have unit length:
///
/// * an embedding (bge): rows scaled to unit length, the angular distance
///   (an embedding's magnitude follows depth);
/// * log θ (topic family): √θ, whose Euclidean distance is the Hellinger
///   distance, rows of unit length by construction;
/// * signed scores (svd, vae, masked-vae, rbm): columns z-scored, so no
///   one component's scale decides the neighbours.
pub fn euclidean_view(mut table: Mat, cell_space: CellSpace) -> (Mat, bool) {
    match cell_space {
        CellSpace::Embedding => {
            l2_normalize_rows_inplace(&mut table);
            (table, true)
        }
        CellSpace::LogSimplex => {
            table.iter_mut().for_each(|x| *x = (0.5 * *x).exp());
            (table, true)
        }
        CellSpace::Signed => {
            let n = table.nrows().max(2) as f32;
            for mut column in table.column_iter_mut() {
                let mean = column.sum() / n;
                column.add_scalar_mut(-mean);
                let sd = (column.norm_squared() / (n - 1.0)).sqrt().max(1e-12);
                column /= sd;
            }
            (table, false)
        }
    }
}

/// A two-column file: each batch and its donor (tab, comma or spaces; a
/// `batch donor` header is skipped).
pub fn read_donor_of_batch(path: &str) -> anyhow::Result<HashMap<Box<str>, Box<str>>> {
    let text = std::fs::read_to_string(path)
        .map_err(|e| anyhow::anyhow!("--donor-of-batch {path}: {e}"))?;
    let mut donor_of_batch = HashMap::default();
    for (line_number, line) in text.lines().enumerate() {
        let fields: Vec<&str> = line
            .split(|c: char| c == ',' || c.is_whitespace())
            .filter(|f| !f.is_empty())
            .collect();
        match fields.as_slice() {
            [] | ["batch", "donor"] => continue,
            [batch, donor] => {
                if let Some(earlier) =
                    donor_of_batch.insert(Box::<str>::from(*batch), Box::<str>::from(*donor))
                {
                    anyhow::ensure!(
                        earlier.as_ref() == *donor,
                        "--donor-of-batch {path} line {}: batch {batch} is given donor {donor}, \
                         and donor {earlier} on an earlier line",
                        line_number + 1
                    );
                }
            }
            _ => anyhow::bail!(
                "--donor-of-batch {path} line {}: expected two fields (batch, donor), got {}",
                line_number + 1,
                fields.len()
            ),
        }
    }
    Ok(donor_of_batch)
}
