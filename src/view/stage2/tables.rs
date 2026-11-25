//! Reading what `senna donor-deviation` recorded in the run's manifest
//! (`stage2`): the per-(gene, donor) deviations and their batch jackknife,
//! and each cell's donor, state coordinates and sample's matched distance.

use legume_numeric::matrix::parquet::{
    read_parquet_string_columns_by_name, read_table_columns, ParquetReader,
};
use senna::run_manifest::{resolve, RunManifest};
use std::collections::HashMap;
use std::path::Path;

/// Stage 2 of a run, held for the viewer.
pub(crate) struct Stage2 {
    pub donors: Vec<Box<str>>,
    donor_index: HashMap<Box<str>, usize>,
    pub genes: Vec<Box<str>>,
    gene_index: HashMap<Box<str>, usize>,
    /// Coefficients per deviation: the intercept, then one per state axis.
    width: usize,
    /// `[gene][donor][width]`: the full fits.
    coefficients: Vec<f32>,
    /// Batches left out in turn, one jackknife refit each.
    batches: usize,
    /// `[gene][donor][batch][width]`: the refits, one deviation's together.
    refits: Vec<f32>,
    /// Each cell's row in the per-cell tables.
    cell_index: HashMap<Box<str>, usize>,
    donor_of_cell: Vec<u32>,
    /// `[cell][width]`: 1 (the intercept's), then the cell's state axes.
    coordinates_of_cell: Vec<f32>,
    /// The nearest matched distance of each cell's sample: large where no
    /// other donor has the sample's state.
    exclusivity_of_cell: Vec<f32>,
}

/// One deviation read out at a point: the log gain and its jackknife z.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Gain {
    pub log_gain: f32,
    /// `log_gain / se`, the standard error from the batch jackknife; 0 when
    /// both are 0 (a deviation the group lasso set to 0).
    pub z: f32,
}

/// The index of `key` in `names`, added at the end when new.
fn intern(names: &mut Vec<Box<str>>, index: &mut HashMap<Box<str>, usize>, key: &str) -> usize {
    if let Some(&i) = index.get(key) {
        return i;
    }
    index.insert(key.into(), names.len());
    names.push(key.into());
    names.len() - 1
}

/// The four tables' paths, as the manifest records them.
struct Paths {
    genes: String,
    refits: String,
    samples: String,
    cells: String,
}

impl Paths {
    fn of(m: &RunManifest, dir: &Path) -> anyhow::Result<Self> {
        let slot = &m.stage2;
        let [genes, refits, samples, cells] = [
            &slot.donor_gene_deviation,
            &slot.donor_gene_deviation_without_batch,
            &slot.donor_sample_states,
            &slot.donor_cell_states,
        ]
        .map(|p| {
            p.as_deref()
                .map(|p| resolve(dir, p).to_string_lossy().into_owned())
        });
        let missing = || anyhow::anyhow!("{}", super::NO_STAGE2);
        Ok(Self {
            genes: genes.ok_or_else(missing)?,
            refits: refits.ok_or_else(missing)?,
            samples: samples.ok_or_else(missing)?,
            cells: cells.ok_or_else(missing)?,
        })
    }
}

/// The deviations' part of [`Stage2`].
struct Deviations {
    donors: Vec<Box<str>>,
    donor_index: HashMap<Box<str>, usize>,
    genes: Vec<Box<str>>,
    gene_index: HashMap<Box<str>, usize>,
    /// `intercept`, then the state axes' names.
    coefficient_names: Vec<Box<str>>,
    coefficients: Vec<f32>,
    batches: usize,
    refits: Vec<f32>,
}

/// The full fits and the jackknife refits, each (gene, donor) row placed
/// by name, so the two tables need not list rows in one order.
fn read_deviations(p: &Paths) -> anyhow::Result<Deviations> {
    let full = GeneTable::read(&p.genes)?;
    anyhow::ensure!(
        full.columns.first().map(AsRef::as_ref) == Some("intercept"),
        "{}: no intercept column",
        p.genes
    );
    let width = full.columns.len();
    let (mut donors, mut donor_index) = (Vec::new(), HashMap::new());
    let (mut genes, mut gene_index) = (Vec::new(), HashMap::new());
    let rows: Vec<(usize, usize)> = full
        .gene
        .iter()
        .zip(&full.donor)
        .map(|(g, d)| {
            (
                intern(&mut genes, &mut gene_index, g),
                intern(&mut donors, &mut donor_index, d),
            )
        })
        .collect();
    let d_n = donors.len();
    let mut coefficients = vec![0f32; genes.len() * d_n * width];
    for (r, (g, d)) in rows.into_iter().enumerate() {
        let at = (g * d_n + d) * width;
        for (c, &v) in coefficients[at..at + width]
            .iter_mut()
            .zip(&full.values[r * width..(r + 1) * width])
        {
            *c = v as f32;
        }
    }

    let refit = GeneTable::read(&p.refits)?;
    // `{coefficient}.without_{batch}`: each column's batch and coefficient.
    let mut batch_names: Vec<Box<str>> = Vec::new();
    let mut batch_index = HashMap::new();
    let slots = refit
        .columns
        .iter()
        .map(|name| {
            let (coefficient, batch) = name
                .split_once(".without_")
                .ok_or_else(|| anyhow::anyhow!("{}: unexpected column {name}", p.refits))?;
            let k = full
                .columns
                .iter()
                .position(|c| c.as_ref() == coefficient)
                .ok_or_else(|| anyhow::anyhow!("{}: unknown coefficient {name}", p.refits))?;
            Ok((intern(&mut batch_names, &mut batch_index, batch), k))
        })
        .collect::<anyhow::Result<Vec<_>>>()?;
    let batches = batch_names.len();
    let cols = refit.columns.len();
    let mut refits = vec![0f32; genes.len() * d_n * batches * width];
    for (r, (gene, donor)) in refit.gene.iter().zip(&refit.donor).enumerate() {
        let (Some(&g), Some(&d)) = (gene_index.get(gene), donor_index.get(donor)) else {
            anyhow::bail!("{}: ({gene}, {donor}) is not in {}", p.refits, p.genes);
        };
        let row = (g * d_n + d) * batches;
        for (j, &(b, k)) in slots.iter().enumerate() {
            refits[(row + b) * width + k] = refit.values[r * cols + j] as f32;
        }
    }
    Ok(Deviations {
        donors,
        donor_index,
        genes,
        gene_index,
        coefficient_names: full.columns,
        coefficients,
        batches,
        refits,
    })
}

impl Stage2 {
    /// Stage 2 of the run at `dir`, from its manifest's `stage2` slot.
    pub(crate) fn load(m: &RunManifest, dir: &Path) -> anyhow::Result<Self> {
        let p = Paths::of(m, dir)?;
        let dev = read_deviations(&p)?;
        let width = dev.coefficient_names.len();

        // Each sample's nearest matched distance, then each cell's.
        let (samples, distances) =
            read_table_columns(&p.samples, &["sample"], &["nearest_matched_distance"])?;
        let nearest_of_sample: HashMap<&str, f32> = samples[0]
            .iter()
            .zip(&distances[0])
            .map(|(n, &d)| (n.as_ref(), d as f32))
            .collect();
        let axes = &dev.coefficient_names[1..];
        let cell = ParquetReader::new(&p.cells, Some(0), None, Some(axes))?;
        anyhow::ensure!(
            cell.column_names == axes,
            "{}: its state axes are not those of {}",
            p.cells,
            p.genes
        );
        let [sample_of_cell, donor_name_of_cell] = <[Vec<Box<str>>; 2]>::try_from(
            read_parquet_string_columns_by_name(&p.cells, &["sample", "donor"])?,
        )
        .map_err(|_| anyhow::anyhow!("{}: no sample or donor column", p.cells))?;
        let n = cell.row_names.len();
        let mut donor_of_cell = Vec::with_capacity(n);
        let mut coordinates_of_cell = Vec::with_capacity(n * width);
        let mut exclusivity_of_cell = Vec::with_capacity(n);
        for (i, (donor, sample)) in donor_name_of_cell.iter().zip(&sample_of_cell).enumerate() {
            let d = dev
                .donor_index
                .get(donor)
                .ok_or_else(|| anyhow::anyhow!("{}: donor {donor} has no deviations", p.cells))?;
            donor_of_cell.push(*d as u32);
            coordinates_of_cell.push(1.0);
            coordinates_of_cell.extend(
                cell.row_major_data[i * (width - 1)..(i + 1) * (width - 1)]
                    .iter()
                    .map(|&x| x as f32),
            );
            exclusivity_of_cell.push(
                nearest_of_sample
                    .get(sample.as_ref())
                    .copied()
                    .unwrap_or(f32::NAN),
            );
        }
        let cell_index = cell
            .row_names
            .into_iter()
            .enumerate()
            .map(|(i, c)| (c, i))
            .collect();
        Ok(Self {
            donors: dev.donors,
            donor_index: dev.donor_index,
            genes: dev.genes,
            gene_index: dev.gene_index,
            width,
            coefficients: dev.coefficients,
            batches: dev.batches,
            refits: dev.refits,
            cell_index,
            donor_of_cell,
            coordinates_of_cell,
            exclusivity_of_cell,
        })
    }

    /// The row of `gene`, matched as spelled, else ignoring case.
    pub(crate) fn gene(&self, name: &str) -> Result<usize, String> {
        self.gene_index
            .get(name)
            .copied()
            .or_else(|| self.genes.iter().position(|g| g.eq_ignore_ascii_case(name)))
            .ok_or_else(|| format!("{name} has no stage-2 deviation"))
    }

    /// The index of donor `name`, when one is named.
    pub(crate) fn donor(&self, name: Option<&str>) -> Result<Option<usize>, String> {
        name.map(|n| {
            self.donor_index
                .get(n)
                .copied()
                .ok_or_else(|| format!("no donor {n} in stage 2"))
        })
        .transpose()
    }

    /// Each named cell's row in the per-cell tables (`None`: not in stage 2).
    pub(crate) fn cells(&self, names: &[Box<str>]) -> Vec<Option<u32>> {
        use rayon::prelude::*;
        names
            .par_iter()
            .map(|n| self.cell_index.get(n).map(|&i| i as u32))
            .collect()
    }

    pub(crate) fn donor_of(&self, cell: u32) -> usize {
        self.donor_of_cell[cell as usize] as usize
    }

    pub(crate) fn coordinates_of(&self, cell: u32) -> &[f32] {
        let c = cell as usize;
        &self.coordinates_of_cell[c * self.width..(c + 1) * self.width]
    }

    pub(crate) fn exclusivity_of(&self, cell: u32) -> f32 {
        self.exclusivity_of_cell[cell as usize]
    }

    /// Whether there are refits to take a z from (two batches or more).
    pub(crate) fn has_jackknife(&self) -> bool {
        self.batches >= 2
    }

    pub(crate) fn width(&self) -> usize {
        self.width
    }

    /// Donor `d`'s deviation for `gene` at state coordinates `x` (intercept
    /// first): its log gain and jackknife z, the variance
    /// `(B − 1) / B · Σ_b (q_b − mean)²` over the `B` refits.
    pub(crate) fn gain(&self, gene: usize, d: usize, x: &[f32]) -> Gain {
        let w = self.width;
        let row = gene * self.donors.len() + d;
        let dot = |c: &[f32]| -> f32 { c.iter().zip(x).map(|(c, x)| c * x).sum() };
        let log_gain = dot(&self.coefficients[row * w..(row + 1) * w]);
        let b_n = self.batches;
        let (sum, squares) = self.refits[row * b_n * w..(row + 1) * b_n * w]
            .chunks(w)
            .fold((0f64, 0f64), |(s, ss), refit| {
                let q = f64::from(dot(refit));
                (s + q, ss + q * q)
            });
        let b = b_n as f64;
        let se = if b_n < 2 {
            f32::NAN
        } else {
            let mean = sum / b;
            ((b - 1.0) / b * (squares - b * mean * mean).max(0.0)).sqrt() as f32
        };
        let z = if log_gain == 0.0 { 0.0 } else { log_gain / se };
        Gain { log_gain, z }
    }
}

/// A gene table as read: a row per (gene, donor), its numeric columns
/// row-major, as read.
struct GeneTable {
    gene: Vec<Box<str>>,
    donor: Vec<Box<str>>,
    columns: Vec<Box<str>>,
    values: Vec<f64>,
}

impl GeneTable {
    fn read(path: &str) -> anyhow::Result<Self> {
        let numbers = ParquetReader::new(path, Some(0), None, None)?;
        let donor = read_parquet_string_columns_by_name(path, &["donor"])?
            .into_iter()
            .next()
            .unwrap_or_default();
        anyhow::ensure!(
            donor.len() == numbers.row_names.len(),
            "{path}: {} donors for {} rows",
            donor.len(),
            numbers.row_names.len()
        );
        Ok(Self {
            gene: numbers.row_names,
            donor,
            columns: numbers.column_names,
            values: numbers.row_major_data,
        })
    }
}
