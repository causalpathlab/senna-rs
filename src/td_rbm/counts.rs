//! Pseudobulk reads on both tracks: the count backends' `{gene}/count/spliced`
//! and `{gene}/count/unspliced` rows summed over the cells of each pseudobulk
//! of every level, the genes the kinetics are fitted on, and the cells' own
//! reads block by block.

use crate::logging::new_progress_bar;
use data_beans::aux::feature_rows::split_count_row;
use data_beans::sparse_io_vector::SparseIoVec;
use nalgebra::DMatrix;
use std::collections::HashMap;

/// Columns read per call.
const COLUMNS_PER_READ: usize = 4096;

/// Every pseudobulk's reads of every gene that has both tracks.
pub struct PbTracks {
    /// Gene names, one per column; rows are the pseudobulks in the order
    /// asked for.
    pub genes: Vec<Box<str>>,
    /// `[P × G]`.
    pub unspliced: DMatrix<f32>,
    pub spliced: DMatrix<f32>,
    /// Cells per pseudobulk.
    pub cells: Vec<f32>,
}

/// A backend row's (gene column, unspliced?), if it is a track of a gene.
type Role = Option<(usize, bool)>;

/// The genes with both tracks among backend rows `rows`, sorted, and every
/// row's role among them.
fn track_rows(rows: &[Box<str>]) -> (Vec<&str>, Vec<Role>) {
    let mut tracks: HashMap<&str, [Option<usize>; 2]> = HashMap::new();
    for (r, name) in rows.iter().enumerate() {
        if let Some((gene, unspliced)) = split_count_row(name) {
            tracks.entry(gene).or_default()[usize::from(unspliced)] = Some(r);
        }
    }
    let mut genes: Vec<&str> = tracks
        .iter()
        .filter(|(_, t)| t[0].is_some() && t[1].is_some())
        .map(|(g, _)| *g)
        .collect();
    genes.sort_unstable();
    let mut role = vec![None; rows.len()];
    for (j, g) in genes.iter().enumerate() {
        let [s, u] = tracks[g];
        role[s.expect("both tracks")] = Some((j, false));
        role[u.expect("both tracks")] = Some((j, true));
    }
    (genes, role)
}

/// How many genes of `rows` have both an unspliced and a spliced row
/// (`{gene}/count/unspliced`, `{gene}/count/spliced`): the genes td-rbm
/// can fit.
pub fn two_track_genes(rows: &[Box<str>]) -> usize {
    track_rows(rows).0.len()
}

/// The cells of `data` one block of columns at a time: `f(first column,
/// unspliced, spliced)` with each block's `[B × G]` reads on the genes
/// named `genes`, in that order.
pub fn for_cell_blocks(
    data: &SparseIoVec,
    genes: &[Box<str>],
    block: usize,
    mut f: impl FnMut(usize, DMatrix<f32>, DMatrix<f32>) -> anyhow::Result<()>,
) -> anyhow::Result<()> {
    let rows = data.row_names()?;
    let (all, role) = track_rows(&rows);
    let column: HashMap<&str, usize> = genes
        .iter()
        .enumerate()
        .map(|(c, g)| (g.as_ref(), c))
        .collect();
    // Backend row → (fitted gene column, unspliced?).
    let role: Vec<Role> = role
        .iter()
        .map(|r| r.and_then(|(j, u)| column.get(all[j]).map(|&c| (c, u))))
        .collect();
    let n = data.num_columns();
    let bar = new_progress_bar(n.div_ceil(block.max(1)) as u64).with_message("reading cells");
    for start in (0..n).step_by(block.max(1)) {
        bar.inc(1);
        let end = (start + block).min(n);
        let mut unspliced = DMatrix::<f32>::zeros(end - start, genes.len());
        let mut spliced = DMatrix::<f32>::zeros(end - start, genes.len());
        data.for_each_triplet(start..end, block, |row, col, v| match role[row as usize] {
            Some((c, true)) => unspliced[(col as usize, c)] += v,
            Some((c, false)) => spliced[(col as usize, c)] += v,
            None => {}
        })?;
        f(start, unspliced, spliced)?;
    }
    bar.finish_and_clear();
    Ok(())
}

/// Sum `data` into its pseudobulks: `member[c]` is cell `c`'s pseudobulk,
/// in `data`'s column order. Genes with only one track are dropped.
pub fn pb_tracks(data: &SparseIoVec, member: &[usize]) -> anyhow::Result<PbTracks> {
    let rows = data.row_names()?;
    let (genes, role) = track_rows(&rows);
    anyhow::ensure!(
        !genes.is_empty(),
        "no gene has both `/count/spliced` and `/count/unspliced` rows in every file"
    );
    let n = data.num_columns();
    anyhow::ensure!(
        member.len() == n,
        "the partition must place each of the {n} cells"
    );
    let (g, p) = (genes.len(), member.iter().max().map_or(0, |&m| m + 1));
    let mut cells = vec![0f32; p];
    member.iter().for_each(|&pb| cells[pb] += 1.0);
    let mut out = PbTracks {
        genes: genes.iter().map(|&s| Box::from(s)).collect(),
        unspliced: DMatrix::<f32>::zeros(p, g),
        spliced: DMatrix::<f32>::zeros(p, g),
        cells,
    };
    let bar = new_progress_bar(n.div_ceil(COLUMNS_PER_READ) as u64).with_message("pseudobulks");
    for start in (0..n).step_by(COLUMNS_PER_READ) {
        bar.inc(1);
        let end = (start + COLUMNS_PER_READ).min(n);
        data.for_each_triplet(start..end, COLUMNS_PER_READ, |row, col, v| {
            if let Some((j, u)) = role[row as usize] {
                let m = if u {
                    &mut out.unspliced
                } else {
                    &mut out.spliced
                };
                m[(member[start + col as usize], j)] += v;
            }
        })?;
    }
    bar.finish_and_clear();
    Ok(out)
}

/// The genes to fit: at least `min_reads` reads on each track over all
/// pseudobulks, then the `max_genes` most variable by the variance of their
/// log total per 10⁴ reads across pseudobulks. Ascending gene indices.
#[must_use]
pub fn select_genes(t: &PbTracks, min_reads: f32, max_genes: usize) -> Vec<usize> {
    let total = &t.unspliced + &t.spliced;
    let depth: Vec<f32> = total.row_iter().map(|r| r.sum().max(1.0)).collect();
    let mut scored: Vec<(usize, f32)> = (0..t.genes.len())
        .filter(|&j| {
            t.unspliced.column(j).sum() >= min_reads && t.spliced.column(j).sum() >= min_reads
        })
        .map(|j| {
            let x: Vec<f32> = (0..total.nrows())
                .map(|i| (total[(i, j)] / depth[i] * 1e4).ln_1p())
                .collect();
            let mean = x.iter().sum::<f32>() / x.len() as f32;
            let var = x.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / x.len() as f32;
            (j, var)
        })
        .collect();
    scored.sort_by(|a, b| b.1.total_cmp(&a.1));
    scored.truncate(max_genes);
    let mut out: Vec<usize> = scored.into_iter().map(|(j, _)| j).collect();
    out.sort_unstable();
    out
}

#[cfg(test)]
#[path = "tests/counts.rs"]
mod tests;
