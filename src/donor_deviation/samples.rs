//! The units stage 2 is fitted on: pseudobulk samples, the cells of one
//! finest pseudobulk in one batch.
//!
//! A finest pseudobulk can span batches, so the run's pseudobulks are not
//! samples themselves; a sample is their intersection with a batch (as
//! data-beans' pb samples are), so it lies in one batch and one donor.
//!
//! The samples are not grown to a minimum size. The run's coarser levels
//! come from a hash partition and mix states, so pooling up them would break
//! the "same state, elsewhere" a matched set relies on (on a planted sim a
//! climb to ≥ 10 cells per batch left the planted cells in nodes 65% their
//! own state). Small samples are pure instead, and the matching does the
//! pooling: each anchor is contrasted with its nearest samples of several
//! other donors.
//!
//! A sample's counts are its cells' raw counts AVERAGED per cell, so a large
//! and a small sample are on one scale and neither dominates a contrast set
//! by its size. Its embedding for matching is its cells' mean direction
//! (renormalized), and its state coordinates its cells' mean coordinates:
//! the deviation is linear in them, so a fit on samples reads out on every
//! cell at its own coordinates with the same coefficients.

use super::inputs::Stage1Cells;
use rayon::prelude::*;
use rustc_hash::FxHashMap as HashMap;
use senna::embed_common::*;

/// Cells read per block when pooling counts.
const CELLS_PER_READ_BLOCK: usize = 20_000;

pub struct PbSamples {
    /// `pb{finest pseudobulk}@{batch}`.
    pub sample_names: Vec<Box<str>>,
    pub batch_of_sample: Vec<Box<str>>,
    pub donor_index_of_sample: Vec<usize>,
    pub cells_in_sample: Vec<usize>,
    pub sample_of_cell: Vec<usize>,
    /// `[samples × dims]`: where matches are looked up.
    pub embedding_of_sample: Mat,
    /// `[samples × width]`, row-major: the cells' mean state coordinates.
    pub state_coordinates_of_sample: Vec<f32>,
}

impl PbSamples {
    pub fn number_of_samples(&self) -> usize {
        self.sample_names.len()
    }
}

/// Group the cells into samples; `state_coordinates_of_cell` is
/// `[cells × width]`, row-major.
pub fn pb_samples(
    cells: &Stage1Cells,
    state_coordinates_of_cell: &[f32],
    width: usize,
) -> PbSamples {
    let mut sample_of_key: HashMap<(usize, &str), usize> = HashMap::default();
    let mut sample_names = Vec::new();
    let mut batch_of_sample = Vec::new();
    let mut donor_index_of_sample = Vec::new();
    let mut cells_in_sample: Vec<usize> = Vec::new();
    let sample_of_cell: Vec<usize> = (0..cells.cell_names.len())
        .map(|cell| {
            let pb = cells.finest_pb_of_cell[cell];
            let batch = cells.batch_of_cell[cell].as_ref();
            let sample = *sample_of_key.entry((pb, batch)).or_insert_with(|| {
                sample_names.push(format!("pb{pb}@{batch}").into_boxed_str());
                batch_of_sample.push(batch.into());
                donor_index_of_sample.push(cells.donor_index_of_cell[cell]);
                cells_in_sample.push(0);
                sample_names.len() - 1
            });
            cells_in_sample[sample] += 1;
            sample
        })
        .collect();
    let number_of_samples = sample_names.len();
    let dims = cells.embedding.ncols();
    let mut embedding_of_sample = Mat::zeros(number_of_samples, dims);
    let mut state_coordinates_of_sample = vec![0f32; number_of_samples * width];
    for (cell, &sample) in sample_of_cell.iter().enumerate() {
        let mut row = embedding_of_sample.row_mut(sample);
        row += cells.embedding.row(cell);
        let coordinates = &state_coordinates_of_cell[cell * width..(cell + 1) * width];
        for (s, c) in state_coordinates_of_sample[sample * width..(sample + 1) * width]
            .iter_mut()
            .zip(coordinates)
        {
            *s += c;
        }
    }
    if cells.unit_rows {
        l2_normalize_rows_inplace(&mut embedding_of_sample);
    } else {
        for (mut row, &n) in embedding_of_sample.row_iter_mut().zip(&cells_in_sample) {
            row /= n as f32;
        }
    }
    for (coordinates, &n) in state_coordinates_of_sample
        .chunks_mut(width)
        .zip(&cells_in_sample)
    {
        coordinates.iter_mut().for_each(|c| *c /= n as f32);
    }
    PbSamples {
        sample_names,
        batch_of_sample,
        donor_index_of_sample,
        cells_in_sample,
        sample_of_cell,
        embedding_of_sample,
        state_coordinates_of_sample,
    }
}

/// Each gene's counts over the samples: (sample, mean count per cell) for
/// every sample with any, sorted by sample.
pub struct SampleCountsByGene {
    pub gene_names: Vec<Box<str>>,
    pub nonzero_samples_of_gene: Vec<Vec<(u32, f32)>>,
}

/// Sort a gene's `(sample, count)` entries by sample and sum each sample's
/// into one, in place. The sort is stable (a merge sort that finds runs): the
/// sums already held are one sorted run, so adding a block's entries costs
/// near-linear time, not a fresh sort of everything held.
pub(super) fn merge_by_sample(entries: &mut Vec<(u32, f32)>) {
    entries.sort_by_key(|&(sample, _)| sample);
    entries.dedup_by(|later, kept| {
        let same = later.0 == kept.0;
        if same {
            kept.1 += later.1;
        }
        same
    });
}

/// Pool the raw counts of every embedded cell into its sample, gene by gene,
/// reading the data once in blocks of cells.
pub fn pool_sample_counts_by_gene(
    data: &SparseIoVec,
    cell_of_column: &[Option<usize>],
    samples: &PbSamples,
) -> anyhow::Result<SampleCountsByGene> {
    let gene_names = data.row_names()?;
    let number_of_columns = cell_of_column.len();
    // per gene, (sample, summed count), merged after every block, so what is
    // held is the distinct (gene, sample) sums plus one block's entries, not
    // every nonzero of the data
    let mut pooled: Vec<Vec<(u32, f32)>> = vec![Vec::new(); gene_names.len()];
    for block_start in (0..number_of_columns).step_by(CELLS_PER_READ_BLOCK) {
        let block_end = (block_start + CELLS_PER_READ_BLOCK).min(number_of_columns);
        // genes × cells by row, so each gene's entries are appended on its own
        let by_gene =
            nalgebra_sparse::CsrMatrix::from(&data.read_columns_csc(block_start..block_end)?);
        pooled
            .par_iter_mut()
            .enumerate()
            .for_each(|(gene, entries)| {
                let row = by_gene.row(gene);
                entries.extend(row.col_indices().iter().zip(row.values()).filter_map(
                    |(&offset, &count)| {
                        cell_of_column[block_start + offset]
                            .map(|cell| (samples.sample_of_cell[cell] as u32, count))
                    },
                ));
                merge_by_sample(entries);
            });
    }
    let nonzero_samples_of_gene = pooled
        .into_par_iter()
        .map(|mut merged| {
            merged.retain(|&(_, count)| count > 0.0);
            for (sample, count) in &mut merged {
                *count /= samples.cells_in_sample[*sample as usize] as f32;
            }
            merged.shrink_to_fit();
            merged
        })
        .collect();
    Ok(SampleCountsByGene {
        gene_names,
        nonzero_samples_of_gene,
    })
}
