use super::assign_tracks;

fn names(rows: &[&str]) -> Vec<Box<str>> {
    rows.iter().map(|&s| s.into()).collect()
}

#[test]
fn rows_get_their_track_and_gene() {
    let axis = names(&[
        "GENE1/count/spliced",
        "GENE1/count/unspliced",
        "GENE2/count/spliced",
    ]);
    let plan = assign_tracks(&axis).expect("assign_tracks");
    assert_eq!(plan.row_unspliced, vec![false, true, false]);
    assert_eq!(plan.gene_names, names(&["GENE1", "GENE2"]));
    assert_eq!(plan.row_gene, vec![0, 0, 1]);
    assert_eq!(plan.rows(false), vec![0, 2]);
    assert_eq!(plan.rows(true), vec![1]);
}

#[test]
fn any_other_row_is_an_error_naming_it() {
    for row in [
        "GENE1/count/total",
        "GENE1/m6a/methylated",
        "GENE1/m6a/site1/methylated",
        "not_a_feature_row",
    ] {
        let axis = names(&["GENE1/count/spliced", row]);
        let err = assign_tracks(&axis).unwrap_err().to_string();
        assert!(err.contains(row), "{row}: {err}");
    }
}

#[test]
fn missing_base_track_errors() {
    let axis = names(&["GENE1/count/unspliced"]);
    let err = assign_tracks(&axis).unwrap_err().to_string();
    assert!(err.contains("base"), "{err}");
}

#[test]
fn more_than_ten_offending_rows_are_capped_in_the_message() {
    let mut rows: Vec<String> = vec!["GENE1/count/spliced".to_string()];
    for i in 0..15 {
        rows.push(format!("bad{i}"));
    }
    let axis: Vec<Box<str>> = rows.iter().map(|s| s.as_str().into()).collect();
    let err = assign_tracks(&axis).unwrap_err().to_string();
    assert!(err.contains("15"), "{err}");
    assert!(err.contains("+5 more"), "{err}");
}

#[test]
fn pair_rows_gives_each_spliced_genes_unspliced_row_and_name() {
    let axis = names(&[
        "GENE1/count/spliced",
        "GENE1/count/unspliced",
        "GENE2/count/spliced",
        "GENE3/count/unspliced",
    ]);
    let plan = assign_tracks(&axis).expect("assign_tracks");
    // GENE3 has no spliced row, so it is not on the base axis.
    let pairs = plan.pair_rows().expect("pair_rows");
    assert_eq!(pairs.spliced, vec![0, 2]);
    assert_eq!(pairs.unspliced, vec![Some(1), None]);
    assert_eq!(pairs.genes, names(&["GENE1", "GENE2"]));
}
