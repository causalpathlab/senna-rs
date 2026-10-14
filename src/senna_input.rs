use crate::embed_common::*;
use data_beans::convert::try_open_or_convert;
use data_beans::sparse_io::COLUMN_SEP;
use rustc_hash::FxHashMap as HashMap;

pub use data_beans::aux::data_loading::{
    read_data_on_shared_rows, ReadSharedRowsArgs, SparseDataWithBatch,
};

/// Collapse a batch-membership slice into a single batch ("all"), neutralizing
/// any downstream per-batch correction. Use when `--ignore-batch` is set.
pub fn collapse_to_single_batch(membership: &mut [Box<str>]) {
    let label: Box<str> = "all".into();
    for tag in membership.iter_mut() {
        *tag = label.clone();
    }
}

/// The batch-label files a fit should actually use, given `--ignore-batch`.
///
/// `--ignore-batch` drops the labels entirely, so the projection and multilevel
/// collapse run as if every cell shared one batch. Warning only when labels were
/// supplied keeps a bare `--ignore-batch` quiet.
///
/// This is the file-list counterpart of [`collapse_to_single_batch`], which does
/// the same thing to per-cell labels already in memory.
pub fn effective_batch_files(
    ignore_batch: bool,
    batch_files: Option<&[Box<str>]>,
) -> Option<&[Box<str>]> {
    if ignore_batch {
        if batch_files.is_some() {
            log::info!("--ignore-batch: dropping batch labels; treating all cells as one batch");
        }
        None
    } else {
        batch_files
    }
}

///////////////////////////////////////////////
// read data stack (vector of `SparseIoVec`) //
///////////////////////////////////////////////

pub struct ReadSharedColumnsArgs {
    pub data_files: Vec<Box<str>>,
    pub batch_files: Option<Vec<Box<str>>>,
    pub num_types: usize,
    pub preload: bool,
    /// Optional shared cell QC (modality-agnostic; see
    /// [`data_beans::qc_lib::compute_qc_stack`]). `None` = no QC.
    pub qc: Option<QcConfig>,
    pub qc_block_size: Option<usize>,
    pub qc_report_out: Option<Box<str>>,
}

pub struct SparseStackWithBatch {
    pub data_stack: SparseIoStack,
    pub batch_stack: Vec<Vec<Box<str>>>,
    /// Near-empty output keep-mask (post-`mask_columns_all` order).
    /// `None` when no QC ran.
    pub output_keep_idx: Option<Vec<usize>>,
}

pub fn read_data_on_shared_columns(
    args: ReadSharedColumnsArgs,
) -> anyhow::Result<SparseStackWithBatch> {
    let nfiles = args.data_files.len();

    let nfiles_per_type = nfiles.div_ceil(args.num_types);

    let mut data_stack = SparseIoStack::new();

    if nfiles_per_type * args.num_types != nfiles {
        return Err(anyhow::anyhow!(
            "Found fewer data sets: {} vs. {}",
            nfiles_per_type * args.num_types,
            nfiles
        ));
    }

    if let Some(batch_files) = &args.batch_files {
        if batch_files.len() != nfiles {
            return Err(anyhow::anyhow!(
                "data files {} vs. batch files {}",
                nfiles,
                batch_files.len()
            ));
        }
    }

    use legume_numeric::matrix::common_io::read_lines;

    for files in args.data_files.chunks(nfiles_per_type) {
        let mut data_vec = SparseIoVec::new();
        let attach_data_name = files.len() > 1;

        for data_file in files {
            info!("Importing data file: {data_file}");

            let mut data = try_open_or_convert(data_file)?;

            if args.preload || data_beans::sparse_io::auto_preload_enabled() {
                data.preload_columns()?;
            }
            // Use basename (not file index) so the `@<basename>` suffix
            // appended to disambiguate barcodes matches what
            // `data_beans::aux::data_loading::read_data_on_shared_rows`
            // produces (used by gbe / topic / svd loaders). Without
            // this, multi-file annotate runs see `AAA@0` while the
            // training pipeline wrote `AAA@Control0004`, every cell
            // misses the latent lookup, and downstream cluster /
            // enrichment outputs are empty.
            let data_name = attach_data_name
                .then(|| legume_numeric::matrix::common_io::basename(data_file))
                .transpose()?;
            data_vec.push(Arc::from(data), data_name)?;
        }

        data_stack.push(data_vec)?;
    }

    let mut batch_stack = vec![];

    if let Some(batch_files) = &args.batch_files {
        for (data_vec, batch_file) in data_stack.stack.iter_mut().zip(batch_files) {
            let ntot = data_vec.num_columns();
            let mut batch_membership = Vec::with_capacity(ntot);

            info!("Reading batch file: {batch_file}");
            for s in read_lines(batch_file)? {
                batch_membership.push(s.to_string().into_boxed_str());
            }
            batch_stack.push(batch_membership);
        }
    } else {
        for data_vec in &mut data_stack.stack {
            let mut batch_membership = Vec::with_capacity(data_vec.num_columns());
            for (id, &nn) in data_vec.num_columns_by_data()?.iter().enumerate() {
                batch_membership.extend(vec![id.to_string().into_boxed_str(); nn]);
            }
            batch_stack.push(batch_membership);
        }
    }

    // Optional shared cell QC (modality-agnostic), before any batch
    // registration. MAD outliers dropped across all members in lockstep;
    // batch_stack filtered to match; near-empty output mask returned.
    let output_keep_idx = if let Some(cfg) = args.qc.as_ref() {
        let report = data_beans::qc_lib::compute_qc_stack(&data_stack, cfg, args.qc_block_size)?;
        if let Some(path) = args.qc_report_out.as_deref() {
            data_beans::qc_lib::write_qc_report(path, &data_stack.column_names()?, &report)?;
        }
        let n_near_empty = report.near_empty.iter().filter(|&&e| e).count();
        info!(
            "QC (stacked): dropped {}/{} cells from training, {} near-empty masked at output",
            report.n_cells_dropped,
            report.train_keep.len(),
            n_near_empty,
        );
        let keep_idx = report.output_keep_idx();
        if report.n_cells_dropped > 0 {
            data_stack.mask_columns_all(&report.train_keep)?;
            for bm in &mut batch_stack {
                *bm = data_beans::qc_lib::filter_by_keep(bm, &report.train_keep);
            }
        }
        Some(keep_idx)
    } else {
        None
    };

    Ok(SparseStackWithBatch {
        data_stack,
        batch_stack,
        output_keep_idx,
    })
}

// /// Build an affine transformation matrix that will help reduce
// /// dimensions in training
// ///
// /// * `collapsed`: data matrices derived from collapsing operations
// /// * `target_size`: targeting size
// pub fn build_row_aggregator(collapsed: &CollapsedOut, target_size: usize) -> anyhow::Result<Mat> {
//     if collapsed.mu_observed.nrows() > target_size {
//         let log_x_nd = collapsed.mu_adjusted.as_ref().map_or_else(
//             || {
//                 collapsed
//                     .mu_observed
//                     .posterior_log_mean()
//                     .transpose()
//                     .clone()
//             },
//             |x| x.posterior_log_mean().transpose().clone(),
//         );

//         let kk = target_size.ilog2() as usize;
//         info!(
//             "reduce data features: {} -> {}",
//             log_x_nd.ncols(),
//             target_size,
//         );

//         let membership = row_membership_matrix(binary_sort_columns(&log_x_nd, kk)?)?;

//         if membership.ncols() != target_size {
//             let d_available = membership.ncols().min(target_size);
//             let mut ret = Mat::zeros(membership.nrows(), target_size);
//             ret.columns_range_mut(0..d_available)
//                 .copy_from(&membership.columns_range(0..d_available));
//             Ok(ret)
//         } else {
//             Ok(membership)
//         }
//     } else {
//         Ok(Mat::identity(collapsed.mu_observed.nrows(), target_size))
//     }
// }

/// The column of each of the run's `cells` among a file's `columns`: by
/// name, as senna matches cells everywhere; when that finds fewer than nine
/// in ten, by barcode with any `@` tag left off both sides (a run fitted on
/// several files tags each cell with its file), whichever finds more. A
/// barcode is used only where it names one cell on each side.
pub fn match_cells<'a>(cells: &'a [Box<str>], columns: &'a [Box<str>]) -> HashMap<Box<str>, u32> {
    let by_name: HashMap<&str, u32> = columns
        .iter()
        .enumerate()
        .map(|(i, n)| (n.as_ref(), i as u32))
        .collect();
    let exact: HashMap<Box<str>, u32> = cells
        .iter()
        .filter_map(|n| Some((n.clone(), *by_name.get(n.as_ref())?)))
        .collect();
    if exact.len() * 10 >= cells.len() * 9 {
        return exact;
    }
    let bare = |n: &'a str| n.split(COLUMN_SEP).next().unwrap_or_default();
    // A barcode's column, `None` once it names a second one.
    let mut column: HashMap<&str, Option<u32>> = HashMap::default();
    for (i, n) in columns.iter().enumerate() {
        column
            .entry(bare(n))
            .and_modify(|c| *c = None)
            .or_insert(Some(i as u32));
    }
    let mut seen: HashMap<&str, usize> = HashMap::default();
    for n in cells {
        *seen.entry(bare(n)).or_default() += 1;
    }
    let by_barcode: HashMap<Box<str>, u32> = cells
        .iter()
        .filter_map(|n| {
            let b = bare(n);
            (seen[b] == 1).then_some(())?;
            Some((n.clone(), (*column.get(b)?)?))
        })
        .collect();
    if by_barcode.len() > exact.len() {
        by_barcode
    } else {
        exact
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Names as boxed strings.
    fn boxed(names: &[&str]) -> Vec<Box<str>> {
        names.iter().map(|&n| n.into()).collect()
    }

    #[test]
    fn cells_match_by_name_then_by_barcode_where_one_cell_has_it() {
        // By name, when nine in ten match.
        let m = match_cells(&boxed(&["a", "b"]), &boxed(&["b", "a", "z"]));
        assert_eq!((m["a"], m["b"]), (1, 0));
        // A run over several files tags its cells; the file does not.
        let run = boxed(&["a@lib1", "b@lib1", "c@lib2", "c@lib1"]);
        let m = match_cells(&run, &boxed(&["a", "b", "c"]));
        assert_eq!((m["a@lib1"], m["b@lib1"]), (0, 1));
        // `c` names two of the run's cells: neither is guessed.
        assert_eq!(m.len(), 2);
        // The other way round: the file tags, the run does not; a barcode
        // twice in the file is not used either.
        let m = match_cells(&boxed(&["a", "b"]), &boxed(&["a@x", "b@x", "b@y"]));
        assert_eq!(m.len(), 1);
        assert_eq!(m["a"], 0);
        // Nothing in common.
        assert!(match_cells(&boxed(&["a"]), &boxed(&["q"])).is_empty());
    }
}
