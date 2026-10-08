//! Pseudobulk reads on both tracks: a count backend's `{gene}/count/spliced`
//! and `{gene}/count/unspliced` rows summed over the cells of each pseudobulk
//! of a base fit, and the genes the kinetics are fitted on.

use data_beans::aux::feature_rows::split_count_row;
use data_beans::sparse_io::open_sparse_matrix_by_path;
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

/// Sum the backend at `path` into the pseudobulks of `membership` (cell name
/// → pseudobulk name). Cells without a pseudobulk are skipped; genes with
/// only one track are dropped. Pseudobulks are the rows, in the order of
/// `pb_order`.
pub fn pb_tracks(
    path: &str,
    membership: &HashMap<Box<str>, Box<str>>,
    pb_order: &[Box<str>],
) -> anyhow::Result<PbTracks> {
    let data = open_sparse_matrix_by_path(path)?;
    let pb_index: HashMap<&str, usize> = pb_order
        .iter()
        .enumerate()
        .map(|(i, p)| (p.as_ref(), i))
        .collect();

    // Backend row → (gene column, unspliced?), for genes with both tracks.
    let rows = data.row_names()?;
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
    anyhow::ensure!(
        !genes.is_empty(),
        "{path}: no gene has both `/count/spliced` and `/count/unspliced` rows"
    );
    let mut role = vec![None; rows.len()];
    for (j, g) in genes.iter().enumerate() {
        let [s, u] = tracks[g];
        role[s.expect("both tracks")] = Some((j, false));
        role[u.expect("both tracks")] = Some((j, true));
    }

    let columns = data.column_names()?;
    let col_pb: Vec<Option<usize>> = columns
        .iter()
        .map(|c| {
            membership
                .get(c)
                .and_then(|p| pb_index.get(p.as_ref()).copied())
        })
        .collect();
    let missing = col_pb.iter().filter(|p| p.is_none()).count();
    if missing > 0 {
        log::warn!(
            "{path}: {missing} of {} cells have no pseudobulk and are skipped",
            columns.len()
        );
    }
    let (p, g) = (pb_order.len(), genes.len());
    let mut unspliced = DMatrix::<f32>::zeros(p, g);
    let mut spliced = DMatrix::<f32>::zeros(p, g);
    let mut cells = vec![0f32; p];
    for &pb in col_pb.iter().flatten() {
        cells[pb] += 1.0;
    }
    for start in (0..columns.len()).step_by(COLUMNS_PER_READ) {
        let end = (start + COLUMNS_PER_READ).min(columns.len());
        let (_, _, triplets) = data.read_triplets_by_columns((start..end).collect())?;
        for (row, col, v) in triplets {
            let (Some(pb), Some((j, u))) = (col_pb[start + col as usize], role[row as usize])
            else {
                continue;
            };
            if u {
                unspliced[(pb, j)] += v;
            } else {
                spliced[(pb, j)] += v;
            }
        }
    }
    Ok(PbTracks {
        genes: genes.into_iter().map(Box::from).collect(),
        unspliced,
        spliced,
        cells,
    })
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
#[path = "counts/tests.rs"]
mod tests;
