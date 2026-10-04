//! Reading `senna critique`'s label files into per-level training pairs.

use super::read_level_pairs;
use legume_numeric::matrix::parquet::{write_named_table, Column};

fn labels(dir: &std::path::Path, channel: &str, rows: &[(i32, i32, i32, f32)]) -> String {
    let path = dir.join("l.critique.labels.topic.parquet");
    let path = path.to_string_lossy().into_owned();
    let key: Vec<Box<str>> = (0..rows.len()).map(|i| i.to_string().into()).collect();
    let channel: Vec<Box<str>> = vec![channel.into(); rows.len()];
    let level: Vec<i32> = rows.iter().map(|r| r.0).collect();
    let a: Vec<i32> = rows.iter().map(|r| r.1).collect();
    let b: Vec<i32> = rows.iter().map(|r| r.2).collect();
    let w: Vec<f32> = rows.iter().map(|r| r.3).collect();
    let cols: Vec<(Box<str>, Column)> = vec![
        ("channel".into(), Column::Str(&channel)),
        ("level".into(), Column::I32(&level)),
        ("pb_a".into(), Column::I32(&a)),
        ("pb_b".into(), Column::I32(&b)),
        ("weight".into(), Column::F32(&w)),
    ];
    write_named_table(&path, "pair", &key, &cols).unwrap();
    path
}

/// A label's level is its `cell_to_pb` column, which is also the training
/// level: coarsest first. Its pseudobulk ids are that level's data rows.
#[test]
fn labels_land_on_their_training_level() {
    let dir = tempfile::tempdir().unwrap();
    let path = labels(
        dir.path(),
        "A",
        &[(0, 3, 7, 1.0), (2, 1, 0, 0.5), (0, 2, 4, 2.0)],
    );
    let per_level = read_level_pairs(&path, 3).unwrap();
    assert_eq!(per_level.len(), 3);
    assert_eq!(per_level[0], vec![(3, 7, 1.0), (2, 4, 2.0)]);
    assert!(per_level[1].is_empty());
    assert_eq!(per_level[2], vec![(1, 0, 0.5)]);
}

/// Labels from another partition, or a channel no trainer reads, are refused
/// rather than trained on in the wrong place.
#[test]
fn foreign_labels_are_refused() {
    let dir = tempfile::tempdir().unwrap();
    let deep = labels(dir.path(), "A", &[(3, 0, 1, 1.0)]);
    assert!(read_level_pairs(&deep, 3).is_err(), "level 3 of 3");
    let genes = labels(dir.path(), "B", &[(0, 0, 1, 1.0)]);
    assert!(read_level_pairs(&genes, 3).is_err(), "gene-pair channel");
    let negative = labels(dir.path(), "A", &[(0, -1, 1, 1.0)]);
    assert!(read_level_pairs(&negative, 3).is_err(), "negative id");
}
