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

#[test]
fn sums_cells_into_every_levels_pseudobulks_on_both_tracks() {
    let dir = tempfile::tempdir().unwrap();
    let path = backend(dir.path());
    // Coarse: every cell in one; fine: c3 alone, c1 + c2, c4 in none.
    let levels = vec![vec![Some(0); 4], vec![Some(1), Some(1), Some(0), None]];
    let t = pb_tracks(&open_tracks(&[path.into()]).unwrap(), &levels).unwrap();
    assert_eq!(
        t[1].genes,
        vec![Box::<str>::from("A")],
        "only A has both tracks"
    );
    assert_eq!(t[0].cells, vec![4.0]);
    assert_eq!(t[0].spliced[(0, 0)], 10.0);
    assert_eq!(t[0].unspliced[(0, 0)], 3.0);
    assert_eq!(t[1].cells, vec![1.0, 2.0]);
    assert_eq!(t[1].spliced[(0, 0)], 3.0);
    assert_eq!(t[1].unspliced[(0, 0)], 0.0);
    assert_eq!(t[1].spliced[(1, 0)], 3.0);
    assert_eq!(t[1].unspliced[(1, 0)], 1.0);
}

/// Every cell's reads on the named genes, block by block, rows in the
/// backend's column order.
#[test]
fn cell_blocks_read_each_cells_tracks_on_the_named_genes() {
    let dir = tempfile::tempdir().unwrap();
    let path = backend(dir.path());
    let mut seen = Vec::new();
    for_cell_blocks(
        &open_tracks(&[path.into()]).unwrap(),
        &[Box::from("A")],
        |start, u, s| {
            assert_eq!((u.ncols(), s.ncols()), (1, 1));
            for c in 0..u.nrows() {
                seen.push((start + c, u[(c, 0)], s[(c, 0)]));
            }
            Ok(())
        },
    )
    .unwrap();
    assert_eq!(
        seen,
        vec![(0, 0.0, 1.0), (1, 1.0, 2.0), (2, 0.0, 3.0), (3, 2.0, 4.0)]
    );
}

/// A second file with its rows in another order, one gene the first lacks
/// and cells c1, c2.
fn second_backend(dir: &std::path::Path) -> String {
    let path = dir.join("more.zarr").to_string_lossy().into_owned();
    #[rustfmt::skip]
    let m = DMatrix::<f32>::from_row_slice(4, 2, &[
        1.0, 0.0,
        10.0, 20.0,
        5.0, 5.0,
        7.0, 7.0,
    ]);
    let mut b = create_sparse_from_dmatrix(&m, Some(&path), Some(&SparseIoBackend::Zarr)).unwrap();
    let rows: Vec<Box<str>> = [
        "A/count/unspliced",
        "A/count/spliced",
        "D/count/spliced",
        "D/count/unspliced",
    ]
    .into_iter()
    .map(Box::from)
    .collect();
    let cols: Vec<Box<str>> = ["c1", "c2"].into_iter().map(Box::from).collect();
    b.register_row_names_vec(&rows);
    b.register_column_names_vec(&cols);
    path
}

/// Several files read as one: every file's cells in turn, tagged by file,
/// on the genes every file has (rows matched by name, not position).
#[test]
fn several_files_read_together_on_their_shared_genes() {
    let dir = tempfile::tempdir().unwrap();
    let files: Vec<Box<str>> = vec![
        backend(dir.path()).into(),
        second_backend(dir.path()).into(),
    ];
    let data = open_tracks(&files).unwrap();
    let cells = data.column_names().unwrap();
    assert_eq!(cells.len(), 6);
    assert!(
        cells[0].ends_with("@tracks") && cells[4].ends_with("@more"),
        "{cells:?}"
    );
    // Every cell alone; the second file's two cells together.
    let levels = vec![vec![Some(0), Some(1), Some(2), Some(3), Some(4), Some(4)]];
    let t = pb_tracks(&data, &levels).unwrap();
    assert_eq!(t[0].genes, vec![Box::<str>::from("A")], "D is in one file");
    assert_eq!(t[0].spliced[(4, 0)], 30.0);
    assert_eq!(t[0].unspliced[(4, 0)], 1.0);
    assert_eq!(t[0].spliced[(3, 0)], 4.0);
    let mut seen = Vec::new();
    for_cell_blocks(&data, &[Box::from("A")], |start, u, s| {
        for c in 0..u.nrows() {
            seen.push((start + c, u[(c, 0)], s[(c, 0)]));
        }
        Ok(())
    })
    .unwrap();
    assert_eq!(seen[4..], [(4, 1.0, 10.0), (5, 0.0, 20.0)]);
    // The same name twice could not tell the cells apart.
    let Err(e) = open_tracks(&[files[0].clone(), files[0].clone()]) else {
        panic!("a name twice opened");
    };
    assert!(e.to_string().contains("two files are named"), "{e}");
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

/// Only genes with both an unspliced and a spliced row count.
#[test]
fn two_track_genes_counts_genes_with_both_rows() {
    let rows: Vec<Box<str>> = [
        "GENE1/count/spliced",
        "GENE1/count/unspliced",
        "GENE2/count/spliced",
        "GENE3/count/unspliced",
        "GENE4",
    ]
    .iter()
    .map(|r| (*r).into())
    .collect();
    assert_eq!(two_track_genes(&rows), 1);
    assert_eq!(two_track_genes(&rows[2..]), 0);
}
