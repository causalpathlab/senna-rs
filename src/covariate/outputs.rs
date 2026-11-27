//! What a covariate fit writes beside its run: the model, each cell's
//! scores and label, one batch file per data file, and each gene's
//! covariate effect.

use data_beans::sparse_io_vector::SparseIoVec;
use genomic_data::covariate::plain_batch;
use legume_numeric::matrix::common_io::{read_lines, write_lines};
use log::info;
use nalgebra::DMatrix;

/// One line of a batch file: a loaded cell (its index among the scored
/// cells) or a column QC set aside, with its plain batch label.
pub enum Line {
    Cell(usize),
    Plain(Box<str>),
}

/// Every column of every data file as loaded before QC, file by file, so
/// batch files can be written whole: the loaded cells of `data` at their
/// places, and each column QC set aside with its plain label (`all` under
/// `ignore_batch`, else its line of the file's batch file, else, when the
/// labels were inferred from names or file names, the data read again by
/// `reread`, called only when some column was set aside).
///
/// # Errors
/// A batch file that cannot be read or does not match its data file.
pub fn column_layout(
    data: &SparseIoVec,
    batch_files: Option<&[Box<str>]>,
    ignore_batch: bool,
    reread: impl FnOnce() -> anyhow::Result<data_beans::aux::data_loading::SparseDataWithBatch>,
) -> anyhow::Result<Vec<Vec<Line>>> {
    let per_file = data.num_columns_by_data()?;
    let mut at: Vec<Vec<Option<usize>>> = per_file.iter().map(|&n| vec![None; n]).collect();
    for col in 0..data.num_columns() {
        for loc in data.column_locations(col) {
            at[loc.backend as usize][loc.local_col as usize] = Some(col);
        }
    }
    let set_aside = at.iter().flatten().any(Option::is_none);
    // With nothing set aside no plain label is written.
    let plain: Vec<Vec<Box<str>>> = if !set_aside || ignore_batch {
        per_file.iter().map(|&n| vec!["all".into(); n]).collect()
    } else if let Some(files) = batch_files {
        anyhow::ensure!(
            files.len() == per_file.len(),
            "{} batch files for {} data files",
            files.len(),
            per_file.len()
        );
        files
            .iter()
            .map(|f| read_lines(f))
            .collect::<anyhow::Result<_>>()?
    } else {
        let whole = reread()?;
        let mut batch = whole.batch.into_iter();
        per_file
            .iter()
            .map(|&n| batch.by_ref().take(n).collect())
            .collect()
    };
    at.into_iter()
        .zip(plain)
        .enumerate()
        .map(|(f, (cols, plain))| {
            anyhow::ensure!(
                plain.len() == cols.len(),
                "--covariate: data file {f} has {} columns but {} plain batch labels",
                cols.len(),
                plain.len()
            );
            cols.into_iter()
                .zip(&plain)
                .map(|(c, b)| match c {
                    Some(c) => Ok(Line::Cell(c)),
                    None => plain_batch(b).map(Line::Plain),
                })
                .collect()
        })
        .collect()
}

/// One batch file per data file (`{out}.covariate.batch.{i}.txt`, every
/// column a line: its `batch|cvK` label, or its plain one for a column QC
/// set aside). Returns the batch files, in data-file order.
///
/// # Errors
/// A file that cannot be written.
pub fn write_batch_files(
    out: &str,
    labels: &[Box<str>],
    layout: &[Vec<Line>],
) -> anyhow::Result<Vec<Box<str>>> {
    let mut files: Vec<Box<str>> = Vec::with_capacity(layout.len());
    for (i, lines) in layout.iter().enumerate() {
        let path = genomic_data::covariate::files::batch(out, i);
        let text: Vec<Box<str>> = lines
            .iter()
            .map(|line| match line {
                Line::Cell(c) => labels[*c].clone(),
                Line::Plain(b) => b.clone(),
            })
            .collect();
        write_lines(&text, &path)?;
        files.push(path.into());
    }
    Ok(files)
}

/// `{out}.covariate_effect.parquet`: each gene's log δ by bin, centred per
/// gene ([`genomic_data::covariate::write_covariate_effect`]).
///
/// # Errors
/// A table that cannot be written.
pub fn write_covariate_effect(
    out: &str,
    genes: &[Box<str>],
    levels: &[Box<str>],
    log_delta: &DMatrix<f32>,
) -> anyhow::Result<()> {
    let path = genomic_data::covariate::files::effect(out);
    if genomic_data::covariate::write_covariate_effect(
        std::path::Path::new(&path),
        genes,
        levels,
        log_delta,
    )? {
        info!("--covariate: wrote {path} (log δ by bin, centred per gene)");
    }
    Ok(())
}
