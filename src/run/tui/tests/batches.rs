use super::*;
use std::path::PathBuf;

fn pair(name: &str, batch: Batch, cells: Option<usize>) -> Pair {
    Pair {
        data: PathBuf::from(name),
        batch,
        info: String::new(),
        cells,
        tags: None,
        gene_counts: None,
    }
}

fn lines(v: &[&str]) -> Vec<String> {
    v.iter().map(ToString::to_string).collect()
}

#[test]
fn labels_are_counted_in_the_order_first_seen() {
    assert_eq!(
        count(&lines(&["b", "a", "b", "c", "a", "b"])),
        [("b".into(), 3), ("a".into(), 2), ("c".into(), 1)]
    );
}

#[test]
fn a_file_is_named_after_its_base_name_or_its_barcode_tags() {
    assert_eq!(own_name(Path::new("/d/s1.zarr.zip")), "s1");
    let mut p = pair("/d/s1.h5", Batch::Own, Some(3));
    assert_eq!(of(&p), [("s1".to_string(), Some(3))]);
    p.tags = Some(lines(&["t1", "t2", "t1"]));
    assert_eq!(
        of(&p),
        [("t1".to_string(), Some(2)), ("t2".to_string(), Some(1))]
    );
}

#[test]
fn only_a_departure_from_sennas_rule_writes_label_files() {
    let dir = Path::new("/run");
    let own = [pair("/d/s1.zarr", Batch::Own, Some(2))];
    assert_eq!(files(&own, dir, "o"), Ok((Vec::new(), Vec::new())));

    let mut tagged = pair("/e/s1.zarr", Batch::Own, None);
    tagged.tags = Some(lines(&["t1", "t2"]));
    let mixed = [
        pair("/d/s1.zarr", Batch::Named("b".into()), Some(2)),
        tagged,
    ];
    let (passed, written) = files(&mixed, dir, "o").unwrap();
    // Two files of one stem do not share a label file.
    assert_eq!(
        passed,
        [
            PathBuf::from("/run/o.batches/s1.txt"),
            PathBuf::from("/run/o.batches/s1-2.txt")
        ]
    );
    assert_eq!(
        written[0].content,
        Content::Repeat {
            label: "b".into(),
            cells: 2
        }
    );
    assert_eq!(written[1].content, Content::Each(lines(&["t1", "t2"])));
}

#[test]
fn a_named_batch_waits_for_the_cell_count() {
    let p = [pair("/d/s1.zarr", Batch::Named("b".into()), None)];
    assert!(files(&p, Path::new("/run"), "o")
        .unwrap_err()
        .contains("not counted"));
}

#[test]
fn label_files_are_written_once_with_renames() {
    let dir = tempfile::tempdir().unwrap();
    let from = dir.path().join("b.tsv");
    std::fs::write(&from, "x\ny\nx\n").unwrap();
    let w = Written {
        path: dir.path().join("o.batches/s1.txt"),
        content: Content::Renamed {
            from,
            renamed: [("x".to_string(), "z".to_string())].into(),
        },
    };
    write(&w).unwrap();
    assert_eq!(std::fs::read_to_string(&w.path).unwrap(), "z\ny\nz\n");
    assert!(write(&w).is_err(), "never written over");
    let w = Written {
        path: dir.path().join("o.batches/s2.txt"),
        content: Content::Repeat {
            label: "b".into(),
            cells: 2,
        },
    };
    write(&w).unwrap();
    assert_eq!(std::fs::read_to_string(&w.path).unwrap(), "b\nb\n");
}
