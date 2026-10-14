use super::*;

/// A run recorded with one file (a string) replays as a list of one; a
/// list replays as it is.
#[test]
fn recorded_files_read_as_one_or_many() {
    let files = |v: serde_json::Value| -> Vec<Box<str>> { one_or_many(v).unwrap() };
    assert_eq!(
        files(serde_json::json!("a.zarr")),
        vec![Box::from("a.zarr")]
    );
    assert_eq!(
        files(serde_json::json!(["a.zarr", "b.zarr"])),
        vec![Box::from("a.zarr"), Box::from("b.zarr")]
    );
}
