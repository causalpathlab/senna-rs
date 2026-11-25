//! Stage 2 read back from its tables: a gain at a cell's state and its
//! batch-jackknife z, by hand.

use super::tables::Stage2;
use legume_numeric::matrix::parquet::{write_named_table, Column};
use senna::run_manifest::{RunKind, RunManifest};

fn names(v: &[&str]) -> Vec<Box<str>> {
    v.iter().map(|&s| s.into()).collect()
}

/// Two genes × three donors, one state axis, two batches left out in turn;
/// two cells of donor B. Only (g1, B) deviates: 1 + 2·state, and 0.5 + 2·state
/// and 1.5 + 2·state without each batch.
fn tables(dir: &std::path::Path) -> RunManifest {
    let at = |f: &str| dir.join(f).to_string_lossy().into_owned();
    let gene = names(&["g0", "g0", "g0", "g1", "g1", "g1"]);
    let donor = names(&["A", "B", "C", "A", "B", "C"]);
    let only_g1_b = |v: f32| -> Vec<f32> { (0..6).map(|r| if r == 4 { v } else { 0.0 }).collect() };
    let (i, s) = (only_g1_b(1.0), only_g1_b(2.0));
    write_named_table(
        &at("s2.genes.parquet"),
        "gene",
        &gene,
        &[
            ("donor".into(), Column::Str(&donor)),
            ("intercept".into(), Column::F32(&i)),
            ("state_axis_1".into(), Column::F32(&s)),
        ],
    )
    .unwrap();
    let (i1, i2) = (only_g1_b(0.5), only_g1_b(1.5));
    write_named_table(
        &at("s2.refits.parquet"),
        "gene",
        &gene,
        &[
            ("donor".into(), Column::Str(&donor)),
            ("intercept.without_b1".into(), Column::F32(&i1)),
            ("state_axis_1.without_b1".into(), Column::F32(&s)),
            ("intercept.without_b2".into(), Column::F32(&i2)),
            ("state_axis_1.without_b2".into(), Column::F32(&s)),
        ],
    )
    .unwrap();
    write_named_table(
        &at("s2.samples.parquet"),
        "sample",
        &names(&["p1"]),
        &[("nearest_matched_distance".into(), Column::F32(&[0.25]))],
    )
    .unwrap();
    write_named_table(
        &at("s2.cells.parquet"),
        "cell",
        &names(&["c1", "c2"]),
        &[
            ("sample".into(), Column::Str(&names(&["p1", "p1"]))),
            ("donor".into(), Column::Str(&names(&["B", "B"]))),
            ("state_axis_1".into(), Column::F32(&[0.0, 1.0])),
        ],
    )
    .unwrap();
    let mut m = RunManifest::new(RunKind::Bge, "r");
    m.stage2.donor_gene_deviation = Some("s2.genes.parquet".into());
    m.stage2.donor_gene_deviation_without_batch = Some("s2.refits.parquet".into());
    m.stage2.donor_sample_states = Some("s2.samples.parquet".into());
    m.stage2.donor_cell_states = Some("s2.cells.parquet".into());
    m
}

#[test]
fn a_gain_is_the_donor_s_deviation_at_the_cell_s_state_with_its_jackknife_z() {
    let dir = tempfile::tempdir().unwrap();
    let s = Stage2::load(&tables(dir.path()), dir.path()).unwrap();
    let rows = s.cells(&names(&["c2", "absent"]));
    assert_eq!(rows[1], None);
    let c2 = rows[0].unwrap();
    assert_eq!(s.donors[s.donor_of(c2)].as_ref(), "B");
    let g1 = s.gene("G1").unwrap();
    let gain = s.gain(g1, s.donor_of(c2), s.coordinates_of(c2));
    // 1 + 2·1 = 3; replicates 2.5 and 3.5: variance (2 − 1)/2 · 0.5 = 0.25.
    assert!((gain.log_gain - 3.0).abs() < 1e-6);
    assert!((gain.z - 6.0).abs() < 1e-4);
    // A deviation the fit set to 0 has z 0, not NaN.
    assert_eq!(
        s.gain(s.gene("g0").unwrap(), 0, s.coordinates_of(c2)).z,
        0.0
    );
    assert!((s.exclusivity_of(c2) - 0.25).abs() < 1e-6);
}
