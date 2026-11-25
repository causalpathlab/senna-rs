use super::*;

/// A group's cells are those the map shows in it, by name or cluster id,
/// of the grouping shown or another; a group the grouping lacks is an
/// error naming those it has.
#[test]
fn the_cells_of_a_group_are_its_cells_on_the_map() {
    let o = Open::of(crate::view::tests::scene(), PathBuf::from("r.senna.json"));
    fn names(p: &Picked) -> Vec<&str> {
        p.names.iter().map(AsRef::as_ref).collect()
    }
    let clusters = o.pick_cells(&["K1".into()], Some("cluster")).unwrap();
    assert_eq!(names(&clusters), ["c2", "c3"]);
    assert_eq!(clusters.grouping, "cluster");
    assert_eq!(clusters.groups, [("C1".to_string(), 2)]);
    let both = o
        .pick_cells(&["0".into(), "C1".into(), "c0".into()], Some("cluster"))
        .unwrap();
    assert_eq!(names(&both), ["c1", "c2", "c3"]);
    assert_eq!(both.groups.len(), 2);
    let typed = o.pick_cells(&["ct1".into()], Some("annotation")).unwrap();
    assert_eq!(names(&typed), ["c1"]);
    let e = o
        .pick_cells(&["CT9".into()], Some("annotation"))
        .unwrap_err();
    assert!(format!("{e}").contains("CT1"), "{e}");
    assert!(o.pick_cells(&["C0".into()], Some("topic")).is_err());
}
