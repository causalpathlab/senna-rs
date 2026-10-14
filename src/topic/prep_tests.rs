use super::*;

/// A kept preprocessing reads back as it was written.
#[test]
fn a_kept_preprocessing_reads_back() {
    let dir = tempfile::tempdir().unwrap();
    let prefix = dir.path().join("run").to_string_lossy().into_owned();
    assert!(Prep::read(&prefix).unwrap().is_none(), "none kept yet");
    let prep = Prep {
        cells: vec!["c1".into(), "c2".into(), "c3".into()],
        rows: vec!["GENE1".into(), "GENE2".into()],
        hvg: Some(vec!["GENE2".into()]),
        output: Some(vec![0, 2]),
        proj_kn: Mat::from_fn(2, 3, |i, j| (i * 3 + j) as f32),
    };
    prep.write(&prefix).unwrap();
    let back = Prep::read(&prefix).unwrap().unwrap();
    assert_eq!(back.cells, prep.cells);
    assert_eq!(back.rows, prep.rows);
    assert_eq!(back.hvg, prep.hvg);
    assert_eq!(back.output, prep.output);
    assert_eq!(back.proj_kn, prep.proj_kn);
}
