use super::*;
use legume_numeric::matrix::parquet::{write_named_table, Column};

fn write(path: &str, with_extra: bool) {
    let cell: Vec<Box<str>> = ["c1@d1", "c2@d1", "c3@d2"].map(Box::from).to_vec();
    let donor: Vec<Box<str>> = ["d1", "d1", "d2"].map(Box::from).to_vec();
    let stratum = [0i32, 1, 0];
    let purity = [0.0f32, 0.9, 0.0];
    let p_malig = [0.1f32, 0.8, 0.2];
    let mut cols = vec![
        ("donor".into(), Column::Str(&donor)),
        ("stratum".into(), Column::I32(&stratum)),
    ];
    if with_extra {
        cols.push(("purity".into(), Column::F32(&purity)));
        cols.push(("p_malig".into(), Column::F32(&p_malig)));
    }
    write_named_table(path, "cell", &cell, &cols).unwrap();
}

#[test]
fn reads_the_table_and_aligns_cells_by_name() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("x.clones.parquet");
    let path = path.to_str().unwrap();
    write(path, true);
    let t = read(path).unwrap();
    assert_eq!(t.stratum, [0, 1, 0]);
    assert_eq!(&*t.donor[2], "d2");
    assert_eq!(t.p_malig.as_deref(), Some(&[0.1f32, 0.8, 0.2][..]));
    assert_eq!(t.purity.as_deref(), Some(&[0.0f32, 0.9, 0.0][..]));
    let cells: Vec<Box<str>> = ["c2@d1", "other", "c1@d1"].map(Box::from).to_vec();
    assert_eq!(strata_of(path, &cells).unwrap(), [1, 0, 0]);
}

#[test]
fn purity_and_malignancy_are_optional() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("y.clones.parquet");
    let path = path.to_str().unwrap();
    write(path, false);
    let t = read(path).unwrap();
    assert_eq!(t.stratum.len(), 3);
    assert!(t.purity.is_none() && t.p_malig.is_none());
}
