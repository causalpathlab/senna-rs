//! Stage 2 read back from its tables: a gain at a cell's state and its
//! batch-jackknife z, by hand.

use super::tables::Stage2;
use crate::view::{Pick, Scene};
use legume_numeric::matrix::parquet::{write_named_table, Column};
use senna::run_manifest::{RunKind, RunManifest};

fn names(v: &[&str]) -> Vec<Box<str>> {
    v.iter().map(|&s| s.into()).collect()
}

/// Two genes × three donors, one state axis, two batches left out in turn;
/// two cells of donor B. Only (g1, B) deviates: 1 + 2·state, and 0.5 + 2·state
/// and 1.5 + 2·state without each batch.
pub(crate) fn tables(dir: &std::path::Path) -> RunManifest {
    tables_of_genes(dir, ["g0", "g1"])
}

/// [`tables`] with the two genes named `genes`.
fn tables_of_genes(dir: &std::path::Path, [g0, g1]: [&str; 2]) -> RunManifest {
    let at = |f: &str| dir.join(f).to_string_lossy().into_owned();
    let gene = names(&[g0, g0, g0, g1, g1, g1]);
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

/// The test scene (cells c1..c4) with the stage 2 of [`tables`] read in:
/// c1 and c2 are donor B's, at states 0 and 1.
pub(crate) fn scene_with_stage2(dir: &std::path::Path) -> Scene {
    let m = tables(dir);
    let s = Stage2::load(&m, dir).unwrap();
    let mut scene = crate::view::tests::scene();
    scene.data.loads.stage2 =
        std::sync::Arc::new(crate::view::activity::Stage2Tables::ready(Ok(s)));
    scene.data.run = Some((m, dir.to_path_buf()));
    scene
}

#[test]
fn the_gain_map_shows_each_cell_s_own_donor_then_steps_through_the_donors() {
    let dir = tempfile::tempdir().unwrap();
    let mut scene = scene_with_stage2(dir.path());
    scene.set_pick(Pick::One("g1".into()));
    scene.toggle_gain();
    let value = |s: &Scene, cell: usize| s.current_shown().unwrap().levels.values[cell];
    // c1, c2 are B's: 1 + 2·0 and 1 + 2·1; c3, c4 are not in stage 2.
    assert_eq!((value(&scene, 0), value(&scene, 1)), (1.0, 3.0));
    assert!(value(&scene, 2).is_nan());
    // Own donor → A (no deviation) → B → C → own donor.
    scene.step_donor(1);
    assert_eq!(value(&scene, 1), 0.0);
    scene.step_donor(1);
    assert_eq!(value(&scene, 1), 3.0);
    scene.step_donor(2);
    assert_eq!(
        scene.pick,
        Some(Pick::Gain {
            gene: "g1".into(),
            donor: None
        })
    );
    // Again: back to the feature's activity.
    scene.toggle_gain();
    assert_eq!(scene.pick, Some(Pick::One("g1".into())));
}

#[test]
fn u_steps_through_the_counts_statistics_then_stage_2_s_exclusivity_then_off() {
    use crate::view::Stat;
    let dir = tempfile::tempdir().unwrap();
    let mut scene = scene_with_stage2(dir.path());
    let mut seen = Vec::new();
    for _ in 0..4 {
        scene.cycle_stat();
        seen.push(scene.pick.clone());
    }
    assert_eq!(
        seen,
        [
            Some(Pick::Stat(Stat::Total)),
            Some(Pick::Stat(Stat::Detected)),
            Some(Pick::Exclusive),
            None
        ]
    );
}

#[test]
fn a_gene_is_found_as_any_feature_is_by_its_symbol() {
    let dir = tempfile::tempdir().unwrap();
    let m = tables_of_genes(
        dir.path(),
        ["ENSG00000000001_CD3E", "ENSG00000000002_MS4A1"],
    );
    let s = Stage2::load(&m, dir.path()).unwrap();
    assert_eq!(s.gene("ENSG00000000001_CD3E"), Ok(0));
    assert_eq!(s.gene("ms4a1"), Ok(1));
    assert_eq!(s.gene("ENSG00000000002"), Ok(1));
    assert!(s.gene("CD19").is_err());
}
