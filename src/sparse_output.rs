//! Writing a fresh gene × cell sparse backend: a `.zarr.zip` archive (built
//! in data-beans' hidden stage beside it, then zipped into place) or, with
//! the `hdf5` feature, an `.h5` file. A loose `.zarr` directory is never
//! written.

use data_beans::sparse_io::{
    create_sparse_from_dmatrix, create_sparse_from_triplets_owned, SparseIoBackend,
};
use data_beans::zarr_io::{finalize_output, leave_stage, prepare_output};
use legume_numeric::matrix::common_io::remove_file;
use nalgebra::DMatrix;

/// The entries of a new backend. Its shape is the length of the row and
/// column names it is written with.
pub enum Entries<'a> {
    /// `(row, column, value)` nonzeros, handed over so the writer sorts them
    /// in place instead of copying a list that can be the size of the data.
    Triplets(Vec<(u64, u64, f32)>),
    /// A dense matrix, rows × columns.
    Dense(&'a DMatrix<f32>),
}

/// `path` if it names a backend [`write_backend`] can write: a `.zarr.zip`
/// archive, or `.h5` in a build with the `hdf5` feature (without it, data-beans
/// would quietly write a loose `.zarr` directory instead). Fits clap's
/// `value_parser`, so a bad target is refused before any work is done.
pub fn check_target(path: &str) -> Result<Box<str>, String> {
    let h5 = cfg!(feature = "hdf5");
    if path.ends_with(".zarr.zip") || (h5 && path.ends_with(".h5")) {
        return Ok(path.into());
    }
    let other = if h5 {
        " or .h5"
    } else {
        " (.h5 needs a build with the `hdf5` feature)"
    };
    Err(format!("'{path}' must end in .zarr.zip{other}"))
}

/// Write `entries` with `rows` and `cols` as a new backend at `path`, and
/// return the path the backend is found at. Whatever was at `path` is
/// removed first, so a failed write never leaves an older run's backend
/// looking like this one's, and a failed write leaves no stage behind.
pub fn write_backend(
    path: &str,
    entries: Entries,
    rows: &[Box<str>],
    cols: &[Box<str>],
) -> anyhow::Result<Box<str>> {
    check_target(path).map_err(anyhow::Error::msg)?;
    let (effective, backend, working) = prepare_output(path, SparseIoBackend::Zarr, true)?;
    // `prepare_output` clears only the stage it builds in, not an archive
    // already at the target.
    remove_file(&effective)?;
    let written = (|| {
        {
            let mut b = match entries {
                Entries::Triplets(t) => {
                    let shape = (rows.len(), cols.len(), t.len());
                    create_sparse_from_triplets_owned(t, shape, Some(&working), Some(&backend))?
                }
                Entries::Dense(m) => create_sparse_from_dmatrix(m, Some(&working), Some(&backend))?,
            };
            b.register_row_names_vec(rows);
            b.register_column_names_vec(cols);
        }
        finalize_output(&working, &effective).map(Box::from)
    })();
    if written.is_err() {
        let _ = remove_file(&working);
        leave_stage(&working);
    }
    written
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::embed_common::axis_id_names as names;
    use data_beans::sparse_io::open_sparse_matrix_by_path;

    fn listing(dir: &std::path::Path) -> Vec<std::ffi::OsString> {
        let mut v: Vec<_> = std::fs::read_dir(dir)
            .unwrap()
            .map(|e| e.unwrap().file_name())
            .collect();
        v.sort();
        v
    }

    #[test]
    fn an_archive_is_written_with_its_names_and_no_stage_is_left() -> anyhow::Result<()> {
        let dir = tempfile::tempdir()?;
        let path = dir.path().join("x.zarr.zip");
        let path = path.to_str().unwrap();
        let written = write_backend(
            path,
            Entries::Triplets(vec![(0, 1, 2.0), (1, 0, 3.0)]),
            &names("GENE", 2),
            &names("C", 3),
        )?;
        assert_eq!(&*written, path);
        let b = open_sparse_matrix_by_path(&written)?;
        assert_eq!((b.num_rows(), b.num_columns()), (Some(2), Some(3)));
        assert_eq!(b.row_names()?, names("GENE", 2));
        assert_eq!(listing(dir.path()), ["x.zarr.zip"]);
        Ok(())
    }

    #[test]
    fn a_failed_write_leaves_neither_an_older_archive_nor_a_stage() -> anyhow::Result<()> {
        let dir = tempfile::tempdir()?;
        let path = dir.path().join("x.zarr.zip");
        let path = path.to_str().unwrap();
        std::fs::write(path, b"an older run's archive")?;
        // Two row names for a nonzero in row 5: the write fails.
        let failed = write_backend(
            path,
            Entries::Triplets(vec![(5, 0, 1.0)]),
            &names("GENE", 2),
            &names("C", 1),
        );
        assert!(failed.is_err());
        assert!(listing(dir.path()).is_empty());
        Ok(())
    }

    #[test]
    fn a_target_is_an_archive_or_h5_never_a_loose_zarr() {
        assert!(check_target("r.zarr.zip").is_ok());
        assert_eq!(check_target("r.h5").is_ok(), cfg!(feature = "hdf5"));
        for bad in ["r.zarr", "r.csv", "r"] {
            assert!(check_target(bad).is_err(), "{bad}");
        }
        let dir = tempfile::tempdir().unwrap();
        let loose = dir.path().join("x.zarr");
        let refused = write_backend(
            loose.to_str().unwrap(),
            Entries::Triplets(vec![]),
            &names("GENE", 1),
            &names("C", 1),
        );
        assert!(refused.is_err());
        assert!(listing(dir.path()).is_empty());
    }
}
