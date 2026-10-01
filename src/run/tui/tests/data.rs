use super::*;

#[test]
fn stems_drop_data_and_label_endings() {
    assert_eq!(stem(Path::new("/d/s1.zarr.zip")), "s1");
    assert_eq!(stem(Path::new("s1_batch.tsv.gz")), "s1_batch");
    assert_eq!(stem(Path::new("s1.h5")), "s1");
}

fn paths(v: &[&str]) -> Vec<PathBuf> {
    v.iter().map(PathBuf::from).collect()
}

fn some(v: &[Option<&str>]) -> Vec<Option<PathBuf>> {
    v.iter().map(|p| p.map(PathBuf::from)).collect()
}

#[test]
fn batch_files_pair_by_name_whatever_their_order() {
    let data = paths(&["d/s1.zarr.zip", "d/s2.zarr.zip"]);
    let got = assign(&data, &paths(&["d/s2_batch.tsv", "d/s1_batch.tsv"]));
    assert_eq!(
        got,
        (
            some(&[Some("d/s1_batch.tsv"), Some("d/s2_batch.tsv")]),
            Paired::ByName(2)
        )
    );
}

#[test]
fn names_alike_only_in_their_start_are_not_crossed() {
    // Data listed lib2 first: a shared `lib` start must not decide.
    let data = paths(&["lib2.zarr", "lib1.zarr"]);
    let got = assign(&data, &paths(&["lib_batch_1.txt", "lib_batch_2.txt"]));
    assert_eq!(
        got,
        (
            some(&[Some("lib_batch_2.txt"), Some("lib_batch_1.txt")]),
            Paired::ByName(2)
        )
    );
}

#[test]
fn a_number_that_runs_on_is_another_sample() {
    let data = paths(&["s1.zarr", "s2.zarr"]);
    let got = assign(&data, &paths(&["s10_batch.tsv", "s2_batch.tsv"]));
    assert_eq!(
        got,
        (some(&[None, Some("s2_batch.tsv")]), Paired::Partly(1))
    );
}

#[test]
fn unrelated_names_go_in_order_only_when_nothing_matches() {
    let data = paths(&["a.zarr", "b.zarr"]);
    assert_eq!(
        assign(&data, &paths(&["x.tsv", "y.tsv"])),
        (some(&[Some("x.tsv"), Some("y.tsv")]), Paired::InOrder)
    );
    assert_eq!(
        assign(&data, &paths(&["x.tsv"])),
        (some(&[None, None]), Paired::Partly(0))
    );
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
