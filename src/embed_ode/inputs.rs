//! What an embed-ode run reads: the ode run (its manifest, arguments,
//! partition, grid posteriors, branch times and batch effects) and the
//! count backends, every gene's spliced reads alone (a backend without
//! tracks, its rows as they are), streamed a block of cells at a time.

use super::model::QMass;
use crate::ode_mixture::command::OdeArgs;
use crate::ode_mixture::model::lineage_of;
use data_beans::aux::data_loading::{read_data_on_shared_rows, ReadSharedRowsArgs};
use data_beans::aux::feature_rows::split_count_row;
use data_beans::sparse_io_vector::SparseIoVec;
use legume_numeric::matrix::parquet::read_table_columns;
use nalgebra::DMatrix;
use senna::run_manifest::{RunKind, RunManifest};
use std::collections::HashMap;
use std::path::PathBuf;

/// An ode run as embed-ode reads it.
pub struct OdeRun {
    pub manifest: RunManifest,
    /// The manifest's directory, every output resolved against it.
    pub dir: PathBuf,
    pub args: OdeArgs,
}

impl OdeRun {
    /// The run at `from` (a prefix or its `.senna.json`); an error unless
    /// it is an ode run.
    pub fn open(from: &str) -> anyhow::Result<Self> {
        let (manifest, dir) = senna::run_manifest::load_for(from)?;
        anyhow::ensure!(
            manifest.kind == RunKind::Ode,
            "{from}: a {} run, not an ode run",
            manifest.kind
        );
        let args = manifest.train_args_as::<OdeArgs>(&manifest.prefix)?;
        Ok(Self {
            manifest,
            dir,
            args,
        })
    }

    /// An output slot's file, resolved; `None` when the run has none.
    pub fn path(&self, slot: Option<&str>) -> Option<String> {
        slot.map(|s| {
            senna::run_manifest::resolve(&self.dir, s)
                .to_string_lossy()
                .into_owned()
        })
    }

    /// A slot's file, or an error naming `what`.
    pub fn need(&self, slot: Option<&str>, what: &str) -> anyhow::Result<String> {
        self.path(slot).ok_or_else(|| {
            anyhow::anyhow!("{}: the ode run records no {what}", self.manifest.prefix)
        })
    }

    /// `{prefix}.delta.parquet`, the collapse's batch effects, when the run
    /// wrote one.
    pub fn delta_path(&self) -> Option<String> {
        let prefix = self.manifest.prefix_in(&self.dir);
        let path = format!("{}.delta.parquet", prefix.to_string_lossy());
        std::path::Path::new(&path).is_file().then_some(path)
    }

    /// Whether `out` names this run's own prefix.
    pub fn is_at(&self, out: &str) -> bool {
        use senna::run_manifest::{default_path, same_file};
        let own = self.manifest.prefix_in(&self.dir);
        same_file(
            std::path::Path::new(&default_path(out)),
            std::path::Path::new(&default_path(&own.to_string_lossy())),
        )
    }
}

/// The rows counted: every `{gene}/count/spliced` row if any row is one,
/// named by its gene; otherwise every row as it is.
pub struct CountRows {
    pub keep: Vec<bool>,
    pub genes: Vec<Box<str>>,
}

#[must_use]
pub fn count_rows(rows: &[Box<str>]) -> CountRows {
    let spliced: Vec<Option<&str>> = rows
        .iter()
        .map(|r| split_count_row(r).and_then(|(g, u)| (!u).then_some(g)))
        .collect();
    if spliced.iter().all(Option::is_none) {
        return CountRows {
            keep: vec![true; rows.len()],
            genes: rows.to_vec(),
        };
    }
    CountRows {
        keep: spliced.iter().map(Option::is_some).collect(),
        genes: spliced.iter().flatten().map(|&g| g.into()).collect(),
    }
}

/// The backends on the counted rows, every cell's batch registered.
pub struct Loaded {
    /// Its rows the counted ones, one per gene.
    pub data: SparseIoVec,
    pub genes: Vec<Box<str>>,
}

/// `files` read as the ode run reads them (rows by their full names, on
/// the rows every file has; every file's cells apart), no QC (cells are
/// matched to the ode run's partition by name), then cut to the counted
/// rows ([`count_rows`]).
pub fn load_data(
    files: &[Box<str>],
    batch_files: Option<&[Box<str>]>,
    ignore_batch: bool,
) -> anyhow::Result<Loaded> {
    use data_beans::sparse_io_vector::{ColumnAlignment, RowAlignment};
    let loaded = read_data_on_shared_rows(ReadSharedRowsArgs {
        data_files: files.to_vec(),
        batch_files: batch_files.map(<[_]>::to_vec),
        feature_kind: Some(data_beans::aux::feature_names::FeatureNameKind::Exact),
        row_alignment: RowAlignment::Intersect,
        column_alignment: ColumnAlignment::Disjoint,
        ..ReadSharedRowsArgs::default()
    })?;
    let (mut data, mut batch) = (loaded.data, loaded.batch);
    if ignore_batch {
        senna::senna_input::collapse_to_single_batch(&mut batch);
    }
    let rows = count_rows(&data.row_names()?);
    anyhow::ensure!(!rows.genes.is_empty(), "no row to count");
    if rows.keep.iter().any(|k| !k) {
        data.mask_rows(&rows.keep)?;
    }
    data.register_batch_membership(&batch);
    Ok(Loaded {
        data,
        genes: rows.genes,
    })
}

/// `[D × batches]` the batch effects of `path` (the collapse's melted
/// `delta.parquet`: `gene`, `batch`, `mean`) on `data`'s rows and batches;
/// one where it has none.
pub fn read_delta(path: &str, data: &SparseIoVec) -> anyhow::Result<DMatrix<f32>> {
    let (keys, nums) = read_table_columns(path, &["gene", "batch"], &["mean"])?;
    let rows: HashMap<Box<str>, usize> = data
        .row_names()?
        .into_iter()
        .enumerate()
        .map(|(i, r)| (r, i))
        .collect();
    let batches: HashMap<Box<str>, usize> = data
        .batch_names()
        .unwrap_or_default()
        .into_iter()
        .enumerate()
        .map(|(i, b)| (b, i))
        .collect();
    let mut delta = DMatrix::<f32>::from_element(rows.len(), batches.len().max(1), 1.0);
    let mut matched = 0usize;
    for ((g, b), &v) in keys[0].iter().zip(&keys[1]).zip(&nums[0]) {
        if let (Some(&r), Some(&c)) = (rows.get(g), batches.get(b)) {
            if v.is_finite() && v > 0.0 {
                delta[(r, c)] = v as f32;
                matched += 1;
            }
        }
    }
    anyhow::ensure!(
        matched > 0,
        "{path}: no batch effect on this data's rows and batches"
    );
    Ok(delta)
}

/// The cells of `data` a block of columns at a time: `f(first column,
/// reads)`, `reads` `[B × D]` on every row
/// ([`for_column_blocks`](crate::ode_mixture::counts::for_column_blocks)).
pub fn for_count_blocks(
    data: &SparseIoVec,
    block: usize,
    delta: Option<&DMatrix<f32>>,
    f: impl FnMut(usize, DMatrix<f32>) -> anyhow::Result<()>,
) -> anyhow::Result<()> {
    let d = data.num_rows();
    let rows: Vec<Option<usize>> = (0..d).map(Some).collect();
    crate::ode_mixture::counts::for_column_blocks(data, &rows, d, block, delta, f)
}

/// `[P × D]` row-major, every pseudobulk's reads: the sum of its cells'
/// ([`for_count_blocks`]), `of_cell` every cell's pseudobulk.
pub fn pb_sums(
    data: &SparseIoVec,
    of_cell: &[Option<usize>],
    n_pb: usize,
    block: usize,
    delta: Option<&DMatrix<f32>>,
) -> anyhow::Result<Vec<f32>> {
    let d = data.num_rows();
    anyhow::ensure!(
        of_cell.len() == data.num_columns(),
        "{} cells' pseudobulks for {} cells",
        of_cell.len(),
        data.num_columns()
    );
    let mut y = vec![0f32; n_pb * d];
    for_count_blocks(data, block, delta, |start, reads| {
        for i in 0..reads.nrows() {
            if let Some(p) = of_cell[start + i] {
                anyhow::ensure!(p < n_pb, "pseudobulk {p} of {n_pb}");
                for (j, v) in reads.row(i).iter().enumerate() {
                    y[p * d + j] += v;
                }
            }
        }
        Ok(())
    })?;
    Ok(y)
}

/// The ode run's finest pseudobulks on the data's cells.
#[derive(Debug)]
pub struct Partition {
    /// Every local pseudobulk's ode id, ascending.
    pub ids: Vec<usize>,
    /// Every data cell's local pseudobulk; `None` off the partition.
    pub of_cell: Vec<Option<usize>>,
}

/// `finest` (the ode run's finest level, its cells `src`) on the cells
/// `data`, matched by name: the pseudobulks some data cell is in,
/// renumbered in the order of their ode ids. A data cell the run did not
/// train on has none; an error if none has one.
pub fn partition_on(
    finest: &[usize],
    src: &[Box<str>],
    data: &[Box<str>],
) -> anyhow::Result<Partition> {
    anyhow::ensure!(
        finest.len() == src.len(),
        "a partition of {} cells named {}",
        finest.len(),
        src.len()
    );
    let at: HashMap<&str, usize> = src
        .iter()
        .zip(finest)
        .map(|(n, &p)| (n.as_ref(), p))
        .collect();
    let ode: Vec<Option<usize>> = data.iter().map(|n| at.get(n.as_ref()).copied()).collect();
    let mut ids: Vec<usize> = ode.iter().flatten().copied().collect();
    ids.sort_unstable();
    ids.dedup();
    anyhow::ensure!(
        !ids.is_empty(),
        "none of the {} data cells is in the ode run's partition (e.g. {:?}); \
         was the ode run trained on these files?",
        data.len(),
        data.iter().take(3).collect::<Vec<_>>()
    );
    let local: HashMap<usize, usize> = ids.iter().enumerate().map(|(i, &p)| (p, i)).collect();
    let of_cell = ode.iter().map(|o| o.map(|p| local[&p])).collect();
    Ok(Partition { ids, of_cell })
}

/// The ode run's grid posteriors on the local pseudobulks.
pub struct PbTime {
    /// On the local pseudobulks (`ids`' order), not renormalised.
    pub q: Vec<QMass>,
    /// Every lineage's duration, `time / phase` on its grid (the draws'
    /// mean time at a position is its phase times the mean duration);
    /// `None` for a lineage with no row past phase 0.
    pub duration: Vec<Option<f64>>,
}

/// `pb_time.parquet` (`pb`, `lineage`, `position`, `phase`, `time`, `q`)
/// on the pseudobulks whose ode ids are `ids`; rows of others are left
/// out. An error at a lineage or position off a grid of `lineages ×
/// positions`.
pub fn read_pb_time(
    path: &str,
    ids: &[usize],
    lineages: usize,
    positions: usize,
) -> anyhow::Result<PbTime> {
    let (keys, nums) = read_table_columns(
        path,
        &["pb", "lineage"],
        &["position", "phase", "time", "q"],
    )?;
    let local: HashMap<usize, usize> = ids.iter().enumerate().map(|(i, &p)| (p, i)).collect();
    let mut q = Vec::new();
    let mut ratio = vec![(0f64, 0usize); lineages];
    for (r, (pb, label)) in keys[0].iter().zip(&keys[1]).enumerate() {
        let lineage = lineage_of(label).filter(|&l| l < lineages);
        let position = nums[0][r];
        let (Some(lineage), true) = (lineage, position >= 0.0 && (position as usize) < positions)
        else {
            anyhow::bail!(
                "{path}: lineage {label} position {position} off a grid of {lineages} × {positions}"
            );
        };
        let (phase, time) = (nums[1][r], nums[2][r]);
        if phase > 0.0 && time.is_finite() {
            ratio[lineage].0 += time / phase;
            ratio[lineage].1 += 1;
        }
        let Some(&p) = pb.parse::<usize>().ok().and_then(|id| local.get(&id)) else {
            continue;
        };
        q.push(QMass {
            pb: p,
            lineage,
            position: position as usize,
            q: nums[3][r] as f32,
        });
    }
    let duration = ratio
        .iter()
        .map(|&(s, n)| (n > 0).then(|| s / n as f64))
        .collect();
    Ok(PbTime { q, duration })
}

/// `branch_time.tsv`'s `branch_time` of lineages `L2..L{lineages}`, in
/// that order, on the phase scale.
pub fn read_branch_times(path: &str, lineages: usize) -> anyhow::Result<Vec<f64>> {
    let text = std::fs::read_to_string(path)?;
    let mut at: HashMap<usize, f64> = HashMap::new();
    for line in text.lines().skip(1) {
        let mut cols = line.split('\t');
        let (Some(label), Some(value)) = (cols.next(), cols.next()) else {
            continue;
        };
        if let Some(l) = lineage_of(label) {
            at.insert(l, value.parse()?);
        }
    }
    (1..lineages)
        .map(|l| {
            at.get(&l)
                .copied()
                .ok_or_else(|| anyhow::anyhow!("{path}: no branch time of lineage L{}", l + 1))
        })
        .collect()
}
