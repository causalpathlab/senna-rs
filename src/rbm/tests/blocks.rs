use super::*;

fn names(v: &[&str]) -> Vec<Box<str>> {
    v.iter().map(|&s| s.into()).collect()
}

#[test]
fn a_plain_load_is_one_block_whatever_its_names_look_like() {
    let rows = names(&["ID1/GENE1", "ID2/GENE2", "ID3/GENE3"]);
    let b = Blocks::from_names(&rows, false);
    assert_eq!(b.len(), 1);
    assert_eq!(b.ranges, vec![0..3]);
    assert_eq!(b.feature_names(&rows), rows);
}

#[test]
fn a_multiome_load_splits_by_modality() {
    let rows = names(&["GENE1/rna", "chr1:1-2/atac", "GENE2/rna"]);
    let b = Blocks::from_names(&rows, true);
    assert_eq!(b.len(), 2);
    let sizes: Vec<usize> = b.ranges.iter().map(|r| r.len()).collect();
    assert_eq!(sizes.iter().sum::<usize>(), 3);
    assert!(sizes.contains(&2) && sizes.contains(&1));
}
