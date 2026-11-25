//! Reading what `senna donor-deviation` recorded in the run's manifest
//! (`stage2`): the per-(gene, donor) deviations and their batch jackknife,
//! and each cell's donor, state coordinates and sample's matched distance.

use senna::run_manifest::{resolve, RunManifest};
use std::collections::HashMap;
use std::path::Path;

/// Stage 2 of a run, held for the viewer.
pub(crate) struct Stage2 {
    pub donors: Vec<Box<str>>,
    pub genes: Vec<Box<str>>,
    gene_index: HashMap<Box<str>, usize>,
    /// Coefficients per deviation: the intercept, then one per state axis.
    width: usize,
    /// `[gene][donor][width]`: the full fits.
    coefficients: Vec<f32>,
    /// Batches left out in turn, one jackknife refit each.
    batches: usize,
    /// `[batch][gene][donor][width]`: the refits.
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

impl Stage2 {
    /// Stage 2 of the run at `dir`, from its manifest's `stage2` slot.
    pub(crate) fn load(m: &RunManifest, dir: &Path) -> anyhow::Result<Self> {
        let slot = &m.stage2;
        let at = |p: &Option<String>, what: &str| {
            p.as_deref()
                .map(|p| resolve(dir, p).to_string_lossy().into_owned())
                .ok_or_else(|| {
                    anyhow::anyhow!(
                        "this run records no {what}; run `senna donor-deviation --from` it first"
                    )
                })
        };
        let genes_path = at(&slot.donor_gene_deviation, "stage 2")?;
        let refits_path = at(
            &slot.donor_gene_deviation_without_batch,
            "stage-2 batch jackknife",
        )?;
        let cells_path = at(&slot.donor_cell_states, "stage-2 cell states")?;
        let samples_path = at(&slot.donor_sample_states, "stage-2 sample states")?;

        let full = GeneTable::read(&genes_path)?;
        let width = full.columns.len();
        anyhow::ensure!(
            full.columns.first().map(AsRef::as_ref) == Some("intercept"),
            "{genes_path}: no intercept column"
        );
        let refit = GeneTable::read(&refits_path)?;
        // `{coefficient}.without_{batch}`, batch by batch.
        let mut batch_names: Vec<&str> = Vec::new();
        let mut refit_slot = Vec::with_capacity(refit.columns.len());
        for name in &refit.columns {
            let (coefficient, batch) = name
                .split_once(".without_")
                .ok_or_else(|| anyhow::anyhow!("{refits_path}: unexpected column {name}"))?;
            let b = batch_names
                .iter()
                .position(|x| *x == batch)
                .unwrap_or_else(|| {
                    batch_names.push(batch);
                    batch_names.len() - 1
                });
            let k = full
                .columns
                .iter()
                .position(|c| c.as_ref() == coefficient)
                .ok_or_else(|| anyhow::anyhow!("{refits_path}: unknown coefficient {name}"))?;
            refit_slot.push((b, k));
        }
        let batches = batch_names.len();

        let mut donors: Vec<Box<str>> = Vec::new();
        let mut donor_index: HashMap<Box<str>, usize> = HashMap::new();
        let mut genes: Vec<Box<str>> = Vec::new();
        let mut gene_index: HashMap<Box<str>, usize> = HashMap::new();
        for (gene, donor) in full.gene.iter().zip(&full.donor) {
            if !gene_index.contains_key(gene) {
                gene_index.insert(gene.clone(), genes.len());
                genes.push(gene.clone());
            }
            if !donor_index.contains_key(donor) {
                donor_index.insert(donor.clone(), donors.len());
                donors.push(donor.clone());
            }
        }
        let (g_n, d_n) = (genes.len(), donors.len());
        let row_of = |gene: &str, donor: &str| -> anyhow::Result<usize> {
            let g = gene_index.get(gene).copied();
            let d = donor_index.get(donor).copied();
            g.zip(d).map(|(g, d)| g * d_n + d).ok_or_else(|| {
                anyhow::anyhow!("{refits_path}: ({gene}, {donor}) is not in {genes_path}")
            })
        };
        let mut coefficients = vec![0f32; g_n * d_n * width];
        for (r, (gene, donor)) in full.gene.iter().zip(&full.donor).enumerate() {
            let at = row_of(gene, donor)? * width;
            coefficients[at..at + width].copy_from_slice(&full.values[r * width..(r + 1) * width]);
        }
        let mut refits = vec![0f32; batches * g_n * d_n * width];
        let cols = refit.columns.len();
        for (r, (gene, donor)) in refit.gene.iter().zip(&refit.donor).enumerate() {
            let row = row_of(gene, donor)?;
            for (j, &(b, k)) in refit_slot.iter().enumerate() {
                refits[((b * g_n * d_n) + row) * width + k] = refit.values[r * cols + j];
            }
        }

        // Each sample's nearest matched distance, then each cell's.
        let sample = legume_numeric::matrix::parquet::ParquetReader::new(
            &samples_path,
            Some(0),
            None,
            Some(&["nearest_matched_distance".into()]),
        )?;
        let nearest_of_sample: HashMap<&str, f32> = sample
            .row_names
            .iter()
            .zip(&sample.row_major_data)
            .map(|(n, &d)| (n.as_ref(), d as f32))
            .collect();
        let axes: Vec<Box<str>> = full.columns[1..].to_vec();
        let cell = legume_numeric::matrix::parquet::ParquetReader::new(
            &cells_path,
            Some(0),
            None,
            Some(&axes),
        )?;
        anyhow::ensure!(
            cell.column_names == axes,
            "{cells_path}: its state axes are not those of {genes_path}"
        );
        let strings = legume_numeric::matrix::parquet::read_parquet_string_columns_by_name(
            &cells_path,
            &["sample", "donor"],
        )?;
        let [sample_of_cell, donor_name_of_cell] = <[Vec<Box<str>>; 2]>::try_from(strings)
            .map_err(|_| anyhow::anyhow!("{cells_path}: no sample or donor column"))?;
        let n = cell.row_names.len();
        let mut donor_of_cell = Vec::with_capacity(n);
        let mut coordinates_of_cell = Vec::with_capacity(n * width);
        let mut exclusivity_of_cell = Vec::with_capacity(n);
        for i in 0..n {
            let d = donor_index.get(&donor_name_of_cell[i]).ok_or_else(|| {
                anyhow::anyhow!(
                    "{cells_path}: donor {} has no deviations",
                    donor_name_of_cell[i]
                )
            })?;
            donor_of_cell.push(*d as u32);
            coordinates_of_cell.push(1.0);
            coordinates_of_cell.extend(
                cell.row_major_data[i * (width - 1)..(i + 1) * (width - 1)]
                    .iter()
                    .map(|&x| x as f32),
            );
            exclusivity_of_cell.push(
                nearest_of_sample
                    .get(sample_of_cell[i].as_ref())
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
            donors,
            genes,
            gene_index,
            width,
            coefficients,
            batches,
            refits,
            cell_index,
            donor_of_cell,
            coordinates_of_cell,
            exclusivity_of_cell,
        })
    }

    /// The row of `gene`, matched as spelled, else ignoring case.
    pub(crate) fn gene(&self, name: &str) -> Option<usize> {
        self.gene_index
            .get(name)
            .copied()
            .or_else(|| self.genes.iter().position(|g| g.eq_ignore_ascii_case(name)))
    }

    pub(crate) fn donor(&self, name: &str) -> Option<usize> {
        self.donors.iter().position(|d| d.as_ref() == name)
    }

    /// Each named cell's row in the per-cell tables (`None`: not in stage 2).
    pub(crate) fn cells(&self, names: &[Box<str>]) -> Vec<Option<usize>> {
        names
            .iter()
            .map(|n| self.cell_index.get(n).copied())
            .collect()
    }

    pub(crate) fn donor_of(&self, cell: usize) -> usize {
        self.donor_of_cell[cell] as usize
    }

    pub(crate) fn coordinates_of(&self, cell: usize) -> &[f32] {
        &self.coordinates_of_cell[cell * self.width..(cell + 1) * self.width]
    }

    pub(crate) fn exclusivity_of(&self, cell: usize) -> f32 {
        self.exclusivity_of_cell[cell]
    }

    pub(crate) fn width(&self) -> usize {
        self.width
    }

    /// Donor `d`'s deviation for `gene` at state coordinates `x` (intercept
    /// first): its log gain and jackknife z, the variance
    /// `(B − 1) / B · Σ_b (q_b − mean)²` over the `B` refits.
    pub(crate) fn gain(&self, gene: usize, d: usize, x: &[f32]) -> Gain {
        let row = gene * self.donors.len() + d;
        let at = |coefficients: &[f32]| -> f32 {
            coefficients[row * self.width..(row + 1) * self.width]
                .iter()
                .zip(x)
                .map(|(c, x)| c * x)
                .sum()
        };
        let log_gain = at(&self.coefficients);
        let block = self.genes.len() * self.donors.len() * self.width;
        let (sum, squares) = self
            .refits
            .chunks(block)
            .fold((0f64, 0f64), |(s, ss), refit| {
                let q = f64::from(at(refit));
                (s + q, ss + q * q)
            });
        let b = self.batches as f64;
        let se = if self.batches < 2 {
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
/// row-major.
struct GeneTable {
    gene: Vec<Box<str>>,
    donor: Vec<Box<str>>,
    columns: Vec<Box<str>>,
    values: Vec<f32>,
}

impl GeneTable {
    fn read(path: &str) -> anyhow::Result<Self> {
        let numbers =
            legume_numeric::matrix::parquet::ParquetReader::new(path, Some(0), None, None)?;
        let donor =
            legume_numeric::matrix::parquet::read_parquet_string_columns_by_name(path, &["donor"])?
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
            values: numbers
                .row_major_data
                .into_iter()
                .map(|v| v as f32)
                .collect(),
        })
    }
}
