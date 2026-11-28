use super::*;
use crate::view::cell_colour::{file_and_column, run_cells};
use crate::view::data::Axis;
use std::path::PathBuf;

fn table(dir: &tempfile::TempDir, text: &str) -> PathBuf {
    let path = dir.path().join("groups.tsv");
    std::fs::write(&path, text).unwrap();
    path
}

#[test]
fn a_table_s_groups_match_the_run_s_cells_by_barcode_and_leave_na_out() {
    let dir = tempfile::tempdir().unwrap();
    let path = table(
        &dir,
        "cell\tcell_type\tscore\nc1\tB_cell\t1\nc2\tNA\t2\nc3\tacinar_cell\t3\nzz\tB_cell\t4\n",
    );
    let run: Vec<Box<str>> = ["c1@s1", "c2@s1", "c3@s1", "c4@s1"]
        .map(Into::into)
        .to_vec();
    let t = GroupTable::read(&path, None, Vec::new(), &run).unwrap();
    assert_eq!(t.labels.kind, LabelKind::Groups);
    assert_eq!(t.labels.levels, ["B_cell", "acinar_cell"].map(Into::into));
    assert_eq!(t.labels.by_name.len(), 2);
    assert!(
        t.said().contains("2 of the run's 4 cells matched"),
        "{}",
        t.said()
    );
    // Another column by name.
    let arg = format!("{}:score", path.display());
    let (file, column) = file_and_column(&arg);
    let by_score = GroupTable::read(&file, column, Vec::new(), &run).unwrap();
    assert_eq!(by_score.labels.levels.len(), 3);
}

#[test]
fn a_table_naming_none_of_the_cells_or_a_colour_for_no_group_is_refused() {
    let dir = tempfile::tempdir().unwrap();
    let path = table(&dir, "cell\tg\nx\tA\n");
    let run: Vec<Box<str>> = ["c1".into()].to_vec();
    assert!(GroupTable::read(&path, None, Vec::new(), &run).is_err());
    let path = table(&dir, "cell\tg\nc1\tA\n");
    let err = GroupTable::read(&path, None, vec![("B".into(), [0, 0, 0])], &run)
        .err()
        .unwrap();
    assert!(err.to_string().contains("group B"), "{err}");
}

#[test]
fn colours_are_read_as_pairs_or_from_a_table() {
    let hex = |c| colours([("g", c)]).map(|v| v[0].1);
    assert_eq!(hex("#006d86").unwrap(), [0x00, 0x6d, 0x86]);
    assert_eq!(hex("9f85e5").unwrap(), [0x9f, 0x85, 0xe5]);
    assert!(hex("#12345").is_err());
    assert_eq!(
        parse_colours("PDAC_classical=#006d86, B_cell=#c34e54").unwrap(),
        vec![
            ("PDAC_classical".into(), [0x00, 0x6d, 0x86]),
            ("B_cell".into(), [0xc3, 0x4e, 0x54])
        ]
    );
    assert!(parse_colours("B_cell").is_err());
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("colours.tsv");
    std::fs::write(&path, "group\tcolour\nB_cell\t#c34e54\n").unwrap();
    assert_eq!(
        parse_colours(path.to_str().unwrap()).unwrap(),
        vec![("B_cell".into(), [0xc3, 0x4e, 0x54])]
    );
}

#[test]
fn shown_groups_take_their_colours_and_stay_for_another_round_of_the_run() {
    let dir = tempfile::tempdir().unwrap();
    let path = table(&dir, "cell\tg\nc1\tA\nc2\tB\nc3\tA\n");
    let mut s = crate::view::tests::scene();
    s.data.run = Some((
        senna::run_manifest::RunManifest::new(senna::run_manifest::RunKind::Svd, "r"),
        dir.path().to_path_buf(),
    ));
    let t = GroupTable::read(
        &path,
        None,
        vec![("B".into(), [0xc3, 0x4e, 0x54])],
        run_cells(&s.data),
    )
    .unwrap();
    s.open_groups(Arc::new(t), true);
    let at = s.label_index(LabelKind::Groups).expect("groups taken up");
    assert_eq!(s.colour, Some(at));
    assert_eq!(s.levels(), ["A", "B"].map(Into::<Box<str>>::into));
    // c1 c2 c3 c4: c4 has no group.
    assert_eq!(s.groups().unwrap(), [0, 1, 0, crate::view::data::NONE]);
    assert_eq!(s.group_sizes().unwrap(), [2, 1]);
    // Drawn in the table's colour, never written to the run's style book.
    let red = crate::view::color::linear_rgb([0xc3, 0x4e, 0x54]);
    assert_eq!(s.resolved(1).unwrap().colour, red);
    assert_ne!(s.resolved(0).unwrap().colour, red);
    assert_eq!(s.style_of(1).colour, None);
    // The same run read again (with no table of its own) keeps the table,
    // coloured by it still.
    let mut again = s.data.clone();
    again.labels.retain(|l| l.kind != LabelKind::Groups);
    s.replace_data(again);
    let at = s.label_index(LabelKind::Groups).expect("kept");
    assert_eq!(s.colour, Some(at));
    assert_eq!(
        s.data
            .labels
            .iter()
            .filter(|l| l.kind == LabelKind::Groups)
            .count(),
        1
    );
    // A round with fewer groupings: the table's index moves, and is found.
    let mut fewer = s.data.clone();
    fewer
        .labels
        .retain(|l| !matches!(l.kind, LabelKind::Groups | LabelKind::Annotation));
    s.replace_data(fewer);
    let at = s.label_index(LabelKind::Groups).expect("kept");
    assert_eq!(s.colour, Some(at));
    assert_eq!(s.group_sizes().unwrap(), [2, 1]);
    // Another run's cells are not the table's.
    let other = crate::view::tests::scene().data;
    s.replace_data(other);
    assert!(s.group_table.is_none());
    assert_eq!(s.label_index(LabelKind::Groups), None);
}

#[test]
fn another_table_unfocuses_drops_the_old_colours_and_stays_off_a_feature_map() {
    let dir = tempfile::tempdir().unwrap();
    let path = table(&dir, "cell\tg\nc1\tA\nc2\tB\nc3\tC\n");
    let mut s = crate::view::tests::scene();
    let read = |s: &Scene, colours| {
        Arc::new(GroupTable::read(&path, None, colours, run_cells(&s.data)).unwrap())
    };
    let t = read(&s, vec![("C".into(), [0xc3, 0x4e, 0x54])]);
    s.open_groups(t, true);
    s.focus = Some(2);
    let other = dir.path().join("two.tsv");
    std::fs::write(&other, "cell\tg\nc1\tX\nc2\tY\n").unwrap();
    let t = Arc::new(GroupTable::read(&other, None, Vec::new(), run_cells(&s.data)).unwrap());
    s.open_groups(t, false);
    assert_eq!(s.focus, None);
    assert_eq!(s.levels(), ["X", "Y"].map(Into::<Box<str>>::into));
    // The first table again with no colours: none of its colours stay.
    s.open_groups(read(&s, Vec::new()), true);
    let red = crate::view::color::linear_rgb([0xc3, 0x4e, 0x54]);
    assert_ne!(s.resolved(2).unwrap().colour, red);
    // On the features map the table opens but colours nothing.
    let genes = s
        .data
        .spaces
        .iter()
        .position(|sp| sp.axis() == Axis::Features)
        .unwrap();
    s.set_space(genes);
    let before = s.colour;
    s.open_groups(read(&s, Vec::new()), true);
    assert_eq!(s.colour, before);
}

#[test]
fn a_table_is_the_same_until_its_file_changes() {
    let dir = tempfile::tempdir().unwrap();
    let path = table(&dir, "cell\tg\nc1\tA\n");
    let run: Vec<Box<str>> = ["c1".into()].to_vec();
    let t = GroupTable::read(&path, None, Vec::new(), &run).unwrap();
    assert!(t.stamp.file.is_absolute());
    assert!(t.is(&path, None, &[]));
    assert!(!t.is(&path, Some("g"), &[]));
    let later = std::time::SystemTime::now() + std::time::Duration::from_secs(5);
    std::fs::File::options()
        .write(true)
        .open(&path)
        .unwrap()
        .set_modified(later)
        .unwrap();
    assert!(!t.is(&path, None, &[]));
}

#[test]
fn a_table_whose_header_is_one_name_short_is_refused_as_values_are() {
    let dir = tempfile::tempdir().unwrap();
    // R's write.table without col.names = NA: no name for the cell column.
    let path = table(&dir, "cell_type\nc1\tA\nc2\tB\n");
    let run: Vec<Box<str>> = ["c1".into(), "c2".into()].to_vec();
    let err = GroupTable::read(&path, None, Vec::new(), &run)
        .err()
        .unwrap();
    assert!(err.to_string().contains("one more field"), "{err}");
}

#[test]
fn another_table_s_labels_sit_at_its_own_groups() {
    let dir = tempfile::tempdir().unwrap();
    let mut s = crate::view::tests::scene();
    let open = |s: &mut Scene, name: &str, text: &str| {
        let path = dir.path().join(name);
        std::fs::write(&path, text).unwrap();
        let t = GroupTable::read(&path, None, Vec::new(), run_cells(&s.data)).unwrap();
        s.open_groups(Arc::new(t), true);
    };
    open(&mut s, "one.tsv", "cell\tg\nc1\tA\nc2\tB\n");
    let first = s.group_centre(0).expect("A labelled");
    s.layers(); // draw once, caching the orders too
    open(&mut s, "two.tsv", "cell\tg\nc3\tA\nc4\tB\n");
    let second = s.group_centre(0).expect("A labelled");
    // One cell per group: its label sits on that cell (c1, then c3).
    assert_eq!(first, s.current().points.xy[0]);
    assert_eq!(second, s.current().points.xy[2]);
}
