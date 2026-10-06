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
