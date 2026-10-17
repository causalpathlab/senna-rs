use super::*;
use data_beans::sparse_io::{create_sparse_from_triplets, SparseIoBackend};
use senna::run_manifest::RunKind;

/// A backend `name` in `dir` with two genes and `n` cells `c0`, `c1`, ….
fn backend(dir: &Path, name: &str, n: usize) {
    let triplets: Vec<(u64, u64, f32)> = (0..n as u64).map(|c| (0, c, 1.0)).collect();
    let mut b = create_sparse_from_triplets(
        &triplets,
        (2, n, triplets.len()),
        Some(dir.join(name).to_str().unwrap()),
        Some(&SparseIoBackend::Zarr),
    )
    .unwrap();
    b.register_row_names_vec(&["GENE1".into(), "GENE2".into()]);
    let cells: Vec<Box<str>> = (0..n).map(|i| format!("c{i}").into()).collect();
    b.register_column_names_vec(&cells);
}

/// The level each cell has, by name.
fn level(l: &Labels, cell: &str) -> String {
    l.levels[l.by_name[cell] as usize].to_string()
}

#[test]
fn each_cell_gets_the_batch_the_fit_gave_it() {
    let dir = tempfile::tempdir().unwrap();
    let d = dir.path();
    backend(d, "lib1.zarr", 2);
    backend(d, "lib2.zarr", 3);
    let mut m = RunManifest::new(RunKind::Svd, "r");
    m.data.input = vec!["lib1.zarr".into(), "lib2.zarr".into()];

    // No batch files: each file is its own batch.
    let batches = |m: &RunManifest| {
        crate::view::activity::read_run_data(m, d)
            .unwrap()
            .batches
            .unwrap()
    };
    let l = batches(&m);
    assert_eq!(l.kind, LabelKind::Batch);
    assert_eq!(l.levels.len(), 2);
    assert_eq!(l.by_name.len(), 5);
    assert_eq!(level(&l, "c0@lib1"), "lib1");
    assert_eq!(level(&l, "c2@lib2"), "lib2");

    // Batch files: their labels, one per cell.
    std::fs::write(d.join("b1.txt"), "x\ny\n").unwrap();
    std::fs::write(d.join("b2.txt"), "y\ny\nz\n").unwrap();
    m.data.batch = vec!["b1.txt".into(), "b2.txt".into()];
    let l = batches(&m);
    assert_eq!(l.levels.len(), 3);
    assert_eq!(level(&l, "c0@lib1"), "x");
    assert_eq!(level(&l, "c1@lib1"), "y");
    assert_eq!(level(&l, "c2@lib2"), "z");
}

#[test]
fn missing_data_skips_the_batch_colouring() {
    let dir = tempfile::tempdir().unwrap();
    let mut m = RunManifest::new(RunKind::Svd, "r");
    m.data.input = vec!["gone.zarr".into()];
    let Err(e) = crate::view::activity::read_run_data(&m, dir.path()) else {
        panic!("labels from data that is not there");
    };
    assert!(e.to_string().contains("not here"));
}

#[test]
fn a_scene_takes_the_batches_once_its_run_s_data_files_are_read() {
    let dir = tempfile::tempdir().unwrap();
    let d = dir.path();
    backend(d, "lib1.zarr", 2);
    backend(d, "lib2.zarr", 3);
    let mut m = RunManifest::new(RunKind::Svd, "r");
    m.data.input = vec!["lib1.zarr".into(), "lib2.zarr".into()];
    let mut s = crate::view::tests::scene();
    s.data.loads = crate::view::activity::RunLoads::new(&m, d);
    assert!(s.data.loads.files.get().is_ok());
    assert!(s.take_batches());
    let batch = s.label_index(LabelKind::Batch).unwrap();
    assert_eq!(s.data.labels[batch].levels.len(), 2);
    // Taken once.
    assert!(!s.take_batches());
}

/// `annotate.argmax` as tsv, tsv.gz or parquet (cell, cell_type, probability)
/// gives the same labels; a NaN probability, as a marker score writes, is fine,
/// and a cell without a label is left out.
#[test]
fn argmax_reads_from_text_or_parquet() {
    use legume_numeric::matrix::common_io::open_buf_writer;
    use legume_numeric::matrix::parquet::{write_table, Column};
    use std::io::Write;

    let dir = tempfile::tempdir().unwrap();
    let at = |f: &str| dir.path().join(f);
    let tsv = "cell\tcell_type\tprobability\nc1\tCT1\t0.9\nc2\tCT2\tNaN\nc3\t\tNaN\n";
    for f in ["a.argmax.tsv", "a.argmax.tsv.gz"] {
        let mut w = open_buf_writer(at(f).to_str().unwrap()).unwrap();
        w.write_all(tsv.as_bytes()).unwrap();
        w.flush().unwrap();
    }
    let s = |v: &[&str]| v.iter().map(|&x| Box::from(x)).collect::<Vec<Box<str>>>();
    write_table(
        at("a.argmax.parquet").to_str().unwrap(),
        &[
            ("cell".into(), Column::Str(&s(&["c1", "c2", "c3"]))),
            ("cell_type".into(), Column::Str(&s(&["CT1", "CT2", ""]))),
            (
                "probability".into(),
                Column::F32(&[0.9, f32::NAN, f32::NAN]),
            ),
        ],
    )
    .unwrap();

    let want: HashMap<Box<str>, Box<str>> = [("c1", "CT1"), ("c2", "CT2")]
        .map(|(c, l)| (c.into(), l.into()))
        .into_iter()
        .collect();
    for f in ["a.argmax.tsv", "a.argmax.tsv.gz", "a.argmax.parquet"] {
        assert_eq!(
            super::super::rounds::read_argmax(&at(f)).unwrap(),
            want,
            "{f}"
        );
    }
}

/// A tab-separated marker panel keeps a comma inside a cell type's name, and
/// the marker lookup finds that type's genes under the annotation's spelling,
/// across every spelling the panel uses for it.
#[test]
fn marker_types_keep_their_commas() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("markers.tsv");
    std::fs::write(
        &path,
        "gene\tcell_type\nMS4A1\tB cell, naive\nCD79A\tB cell naive\nCD3E\tT cell\n",
    )
    .unwrap();
    let l = read_marker_labels(&path).unwrap();
    assert_eq!(l.by_name.len(), 3, "the header is no marker");
    assert_eq!(level(&l, "MS4A1"), "B cell, naive");
    assert_eq!(
        l.members_like("B_cell_naive"),
        [Box::from("CD79A"), "MS4A1".into()]
    );
    assert_eq!(l.members_like("T cell"), [Box::<str>::from("CD3E")]);
    assert!(
        l.members_like("t cell").is_empty(),
        "case is part of a type"
    );
}
