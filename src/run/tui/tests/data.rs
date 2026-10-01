use super::*;

fn pair(name: &str) -> Pair {
    Pair {
        data: PathBuf::from(name),
        batch: None,
        info: String::new(),
        gene_counts: None,
    }
}

#[test]
fn stems_drop_data_and_label_endings() {
    assert_eq!(stem(Path::new("/d/s1.zarr.zip")), "s1");
    assert_eq!(stem(Path::new("s1_batch.tsv.gz")), "s1_batch");
    assert_eq!(stem(Path::new("s1.h5")), "s1");
}

fn paths(v: &[&str]) -> Vec<PathBuf> {
    v.iter().map(PathBuf::from).collect()
}

fn batches_of(pairs: &[Pair]) -> Vec<Option<PathBuf>> {
    pairs.iter().map(|p| p.batch.clone()).collect()
}

#[test]
fn batch_files_pair_by_name_whatever_their_order() {
    let mut pairs = vec![pair("d/s1.zarr.zip"), pair("d/s2.zarr.zip")];
    let got = assign(&mut pairs, &paths(&["d/s2_batch.tsv", "d/s1_batch.tsv"]));
    assert_eq!(got, Paired::ByName(2));
    assert_eq!(
        batches_of(&pairs),
        [Some("d/s1_batch.tsv".into()), Some("d/s2_batch.tsv".into())]
    );
}

#[test]
fn names_alike_only_in_their_start_are_not_crossed() {
    // Data listed rep2 first: a shared `rep` start must not decide.
    let mut pairs = vec![pair("rep2.zarr"), pair("rep1.zarr")];
    let got = assign(&mut pairs, &paths(&["rep_batch_1.txt", "rep_batch_2.txt"]));
    assert_eq!(got, Paired::ByName(2));
    assert_eq!(
        batches_of(&pairs),
        [
            Some("rep_batch_2.txt".into()),
            Some("rep_batch_1.txt".into())
        ]
    );
}

#[test]
fn a_number_that_runs_on_is_another_sample() {
    let mut pairs = vec![pair("s1.zarr"), pair("s2.zarr")];
    let got = assign(&mut pairs, &paths(&["s10_batch.tsv", "s2_batch.tsv"]));
    assert_eq!(got, Paired::Partly(1));
    assert_eq!(batches_of(&pairs), [None, Some("s2_batch.tsv".into())]);
}

#[test]
fn unrelated_names_go_in_order_only_when_nothing_matches() {
    let mut pairs = vec![pair("a.zarr"), pair("b.zarr")];
    assert_eq!(
        assign(&mut pairs, &paths(&["x.tsv", "y.tsv"])),
        Paired::InOrder
    );
    assert_eq!(
        batches_of(&pairs),
        [Some("x.tsv".into()), Some("y.tsv".into())]
    );
    let mut pairs = vec![pair("a.zarr"), pair("b.zarr")];
    assert_eq!(assign(&mut pairs, &paths(&["x.tsv"])), Paired::Partly(0));
    assert_eq!(batches_of(&pairs), [None, None]);
}

#[test]
fn a_label_file_beside_the_data_is_found() {
    let dir = tempfile::tempdir().unwrap();
    let d = dir.path().join("s1.zarr.zip");
    std::fs::write(&d, "").unwrap();
    assert_eq!(beside(&d, &label_files_in(dir.path())), None);
    std::fs::write(dir.path().join("s10_batch.txt"), "a\n").unwrap();
    assert_eq!(
        beside(&d, &label_files_in(dir.path())),
        None,
        "s10's labels are not s1's"
    );
    std::fs::write(dir.path().join("s1.batch.tsv"), "a\n").unwrap();
    std::fs::write(dir.path().join("s2.batch.tsv"), "a\n").unwrap();
    assert_eq!(
        beside(&d, &label_files_in(dir.path())),
        Some(dir.path().join("s1.batch.tsv"))
    );
    std::fs::write(dir.path().join("s1_batch.txt"), "a\n").unwrap();
    assert_eq!(
        beside(&d, &label_files_in(dir.path())),
        None,
        "two candidates: none is guessed"
    );
}

#[test]
fn batch_files_are_all_or_none() {
    let mut pairs = vec![pair("a.zarr"), pair("b.zarr")];
    assert_eq!(batch_problem(&pairs), None);
    pairs[0].batch = Some("a.tsv".into());
    assert!(batch_problem(&pairs).is_some());
    pairs[1].batch = Some("b.tsv".into());
    assert_eq!(batch_problem(&pairs), None);
}
