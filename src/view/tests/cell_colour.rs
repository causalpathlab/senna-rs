use super::*;

#[test]
fn a_table_s_values_colour_the_cells_it_matches_and_leave_the_rest_grey() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("sites.tsv");
    std::fs::write(
        &path,
        "cell\tm6a_sites\tumi\tsites_per_1k_umi\tnote\n\
         c3\t4\t200\t20\tx\nc1\t2\t100\t20.5\ty\nc2\tNA\t50\tNA\tz\nzz\t1\t10\t100\tw\n",
    )
    .unwrap();
    // By barcode, as an extra file's cells match; the last numeric column.
    let run: Vec<Box<str>> = ["c1@lib", "c2@lib", "c3@lib", "c4@lib"]
        .map(Into::into)
        .to_vec();
    let (file, column) = file_and_column(path.to_str().unwrap());
    assert_eq!(column, None);
    let v = CellValues::read(&file, column, &run).unwrap();
    assert_eq!(v.by_cell.len(), 2);
    assert_eq!(v.by_cell["c1@lib"], 20.5);
    assert!(
        v.said().contains("2 of the run's 4 cells matched"),
        "{}",
        v.said()
    );

    let mut s = crate::view::tests::scene();
    let arg = format!("{}:umi", path.display());
    let (file, column) = file_and_column(&arg);
    assert_eq!(column, Some("umi"));
    let v = CellValues::read(&file, column, run_cells(&s.data)).unwrap();
    assert!(v.title.starts_with("umi"), "{}", v.title);
    s.show_values(Arc::new(v));
    assert_eq!(s.pick, Some(Pick::Values));
    let levels = &s.shown.as_ref().expect("values shown").levels;
    // c1 c2 c3 c4: c4 is not in the table, grey.
    assert_eq!(&levels.values[..3], &[100.0, 50.0, 200.0]);
    assert!(levels.values[3].is_nan());
    assert!(levels.t(0) >= 0.0 && levels.t(3) < 0.0);
    // The values close the `u` cycle once open.
    s.pick = Some(Pick::Stat(Stat::Detected));
    s.cycle_stat();
    assert_eq!(s.pick, Some(Pick::Values));
    s.cycle_stat();
    assert_eq!(s.pick, None);
    assert!(CellValues::read(&file, Some("note"), &run).is_err());
    // R's write.table header, one name short, is refused.
    let r = dir.path().join("r.tsv");
    std::fs::write(&r, "score\nc1\t1\n").unwrap();
    assert!(CellValues::read(&r, None, &run).is_err());
}

#[test]
fn a_values_table_is_the_same_until_its_file_changes() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("v.tsv");
    std::fs::write(&path, "cell\tumi\tscore\nc1\t3\t1.5\n").unwrap();
    let run: Vec<Box<str>> = ["c1".into()].to_vec();
    let v = CellValues::read(&path, None, &run).unwrap();
    assert!(v.stamp.file.is_absolute());
    // No column named: the one read; another column is another table.
    assert!(v.is(&path, None));
    assert!(v.is(&path, Some("score")));
    assert!(!v.is(&path, Some("umi")));
    let later = std::time::SystemTime::now() + std::time::Duration::from_secs(5);
    std::fs::File::options()
        .write(true)
        .open(&path)
        .unwrap()
        .set_modified(later)
        .unwrap();
    assert!(!v.is(&path, None));
}
