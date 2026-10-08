use super::*;
use data_beans::sparse_io::{create_sparse_from_dmatrix, SparseIoBackend};

/// Rows: A spliced, B spliced, A unspliced, C unspliced (C has one track
/// only); cells c1..c4.
fn backend(dir: &std::path::Path) -> String {
    let path = dir.join("tracks.zarr").to_string_lossy().into_owned();
    #[rustfmt::skip]
    let m = DMatrix::<f32>::from_row_slice(4, 4, &[
        1.0, 2.0, 3.0, 4.0,
        5.0, 6.0, 7.0, 8.0,
        0.0, 1.0, 0.0, 2.0,
        9.0, 9.0, 9.0, 9.0,
    ]);
    let mut b = create_sparse_from_dmatrix(&m, Some(&path), Some(&SparseIoBackend::Zarr)).unwrap();
    let rows: Vec<Box<str>> = [
        "A/count/spliced",
        "B/count/spliced",
        "A/count/unspliced",
        "C/count/unspliced",
    ]
    .into_iter()
    .map(Box::from)
    .collect();
    let cols: Vec<Box<str>> = ["c1", "c2", "c3", "c4"]
        .into_iter()
        .map(Box::from)
        .collect();
    b.register_row_names_vec(&rows);
    b.register_column_names_vec(&cols);
    path
}

fn membership(pairs: &[(&str, &str)]) -> HashMap<Box<str>, Box<str>> {
    pairs
        .iter()
        .map(|&(c, p)| (Box::from(c), Box::from(p)))
        .collect()
}

#[test]
fn sums_cells_into_their_pseudobulks_on_both_tracks() {
    let dir = tempfile::tempdir().unwrap();
    let path = backend(dir.path());
    // c4 has no pseudobulk.
    let m = membership(&[("c1", "x"), ("c2", "x"), ("c3", "y")]);
    let t = pb_tracks(&path, &m, &[Box::from("y"), Box::from("x")]).unwrap();
    assert_eq!(
        t.genes,
        vec![Box::<str>::from("A")],
        "only A has both tracks"
    );
    assert_eq!(t.cells, vec![1.0, 2.0]);
    // y = c3; x = c1 + c2.
    assert_eq!(t.spliced[(0, 0)], 3.0);
    assert_eq!(t.unspliced[(0, 0)], 0.0);
    assert_eq!(t.spliced[(1, 0)], 3.0);
    assert_eq!(t.unspliced[(1, 0)], 1.0);
}

#[test]
fn select_genes_drops_thin_tracks_and_keeps_the_most_variable() {
    let genes: Vec<Box<str>> = ["flat", "varied", "thin"]
        .into_iter()
        .map(Box::from)
        .collect();
    #[rustfmt::skip]
    let spliced = DMatrix::<f32>::from_row_slice(4, 3, &[
        50.0, 10.0, 50.0,
        50.0, 90.0, 50.0,
        50.0, 10.0, 50.0,
        50.0, 90.0, 50.0,
    ]);
    #[rustfmt::skip]
    let unspliced = DMatrix::<f32>::from_row_slice(4, 3, &[
        10.0, 10.0, 1.0,
        10.0, 10.0, 1.0,
        10.0, 10.0, 1.0,
        10.0, 10.0, 1.0,
    ]);
    let t = PbTracks {
        genes,
        unspliced,
        spliced,
        cells: vec![1.0; 4],
    };
    assert_eq!(
        select_genes(&t, 20.0, 10),
        vec![0, 1],
        "thin has 4 unspliced reads"
    );
    assert_eq!(select_genes(&t, 20.0, 1), vec![1], "varied beats flat");
}
