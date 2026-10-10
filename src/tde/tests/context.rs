use super::*;

/// Each unit's context from its cells' time bins: [`profiles`], then
/// [`context_from_profiles`].
///
/// Units are given by `level` and `source` (a pseudobulk index at a
/// pseudobulk level, `0..cell_to_pb_per_level.len()`; a cell index at the
/// level after the last) and `total` reads; `cell_to_pb_per_level[l][c]` is
/// cell `c`'s pseudobulk at level `l`, the last level the finest;
/// `cell_bin[c]` is cell `c`'s bin in `0..n_bins`, `None` when unlabelled.
fn time_context(
    level: &[u8],
    source: &[u32],
    cell_to_pb_per_level: &[Vec<usize>],
    cell_bin: &[Option<u32>],
    n_bins: usize,
    total: &[f32],
) -> anyhow::Result<graph_embedding_util::UnitContext> {
    let p = profiles(level, source, cell_to_pb_per_level, cell_bin, n_bins)?;
    context_from_profiles(&p, level, source, cell_to_pb_per_level.len(), total)
}

/// Six cells, two pseudobulk levels. Bins: cells 0,1 early (0); 2,3,5 late
/// (1); cell 4 unlabelled.
///   level 0: pb0 = {0,1,2}, pb1 = {3,4,5}
///   level 1 (finest): pb0 = {0,1}, pb1 = {2,3}, pb2 = {4,5}
/// Units: the five pseudobulks (level order), then cells 1 and 4.
#[allow(clippy::type_complexity)]
fn fixture() -> (
    Vec<u8>,
    Vec<u32>,
    Vec<Vec<usize>>,
    Vec<Option<u32>>,
    Vec<f32>,
) {
    let level = vec![0, 0, 1, 1, 1, 2, 2];
    let source = vec![0, 1, 0, 1, 2, 1, 4];
    let cell_to_pb = vec![vec![0, 0, 0, 1, 1, 1], vec![0, 0, 1, 1, 2, 2]];
    let bins = vec![Some(0), Some(0), Some(1), Some(1), None, Some(1)];
    // Read totals per unit; the finest pseudobulks' are 10, 20 and 30.
    let total = vec![30.0, 30.0, 10.0, 20.0, 30.0, 5.0, 5.0];
    (level, source, cell_to_pb, bins, total)
}

fn row(ctx: &ge::UnitContext, u: usize) -> Vec<(u32, f32)> {
    let mut r = ctx.rows[u].clone();
    r.sort_by_key(|&(v, _)| v);
    r
}

#[test]
fn targets_are_the_finest_pseudobulks_weighted_by_shared_time_and_reads() {
    let (level, source, cell_to_pb, bins, total) = fixture();
    let ctx = time_context(&level, &source, &cell_to_pb, &bins, 2, &total).unwrap();
    assert_eq!(ctx.n_targets, 3);
    ctx.validate(level.len()).unwrap();
    // Cell 1 is early; only finest pb0 holds early cells.
    assert_eq!(row(&ctx, 5), vec![(0, 1.0)]);
    // Level-0 pb1 is all late; late cells sit in finest pb1 (all) and pb2
    // (its one labelled cell): shares ∝ N_v · p_v(late) = 20 and 30.
    let r = row(&ctx, 1);
    assert_eq!(r.len(), 2);
    assert_eq!((r[0].0, r[1].0), (1, 2));
    assert!(
        (r[0].1 - 0.4).abs() < 1e-6 && (r[1].1 - 0.6).abs() < 1e-6,
        "{r:?}"
    );
    // Finest pb1 is late too: the same targets, itself included.
    assert_eq!(row(&ctx, 3), r);
}

#[test]
fn a_unit_with_no_labelled_cell_observes_no_context() {
    let (level, source, cell_to_pb, bins, total) = fixture();
    let ctx = time_context(&level, &source, &cell_to_pb, &bins, 2, &total).unwrap();
    assert!(ctx.rows[6].is_empty(), "cell 4 has no time");
}

#[test]
fn mixed_units_spread_over_both_times() {
    let (level, source, cell_to_pb, bins, total) = fixture();
    let ctx = time_context(&level, &source, &cell_to_pb, &bins, 2, &total).unwrap();
    // Level-0 pb0 = {early, early, late}: p = (2/3, 1/3). Weights to the
    // finest pbs: pb0 2/3·10, pb1 1/3·20, pb2 1/3·30.
    let r = row(&ctx, 0);
    let w = [2.0 / 3.0 * 10.0, 1.0 / 3.0 * 20.0, 1.0 / 3.0 * 30.0];
    let z: f32 = w.iter().sum();
    for (v, q) in r {
        assert!((q - w[v as usize] / z).abs() < 1e-6, "target {v}: {q}");
    }
}

/// Continuous time: the same six cells with τ = 0, 0.1, 0.4, 0.5, ∅, 0.9.
fn continuous_tau() -> Vec<Option<f32>> {
    vec![Some(0.0), Some(0.1), Some(0.4), Some(0.5), None, Some(0.9)]
}

#[test]
fn the_kernel_width_is_the_pooled_spread_of_time_within_the_finest_pseudobulks() {
    let (_, _, cell_to_pb, _, _) = fixture();
    // Finest pbs: {0, 0.1}, {0.4, 0.5}, {∅, 0.9}: deviations ±0.05, ±0.05, 0
    // over 5 timed cells in 3 pbs.
    let h = kernel_width(cell_to_pb.last().unwrap(), &continuous_tau()).unwrap();
    let want = ((4.0 * 0.05f32 * 0.05) / (5.0 - 3.0)).sqrt();
    assert!((h - want).abs() < 1e-6, "{h} vs {want}");
}

#[test]
fn continuous_time_weights_targets_by_a_kernel_on_mean_times_and_reads() {
    let (level, source, cell_to_pb, _, total) = fixture();
    let tau = continuous_tau();
    let ctx = continuous_context(&level, &source, &cell_to_pb, &tau, &total).unwrap();
    ctx.validate(level.len()).unwrap();
    let h = kernel_width(cell_to_pb.last().unwrap(), &tau).unwrap();
    let k = |d: f32| (-(d * d) / (2.0 * h * h)).exp();
    // Cell 1 (τ 0.1) against the finest pbs' means 0.05, 0.45, 0.9 and
    // reads 10, 20, 30.
    let w = [k(0.05) * 10.0, k(0.35) * 20.0, k(0.8) * 30.0];
    let z: f32 = w.iter().sum();
    for (v, q) in row(&ctx, 5) {
        assert!((q - w[v as usize] / z).abs() < 1e-5, "target {v}: {q}");
    }
    // Nearer in time is heavier, read for read.
    let r = row(&ctx, 5);
    assert!(r[0].1 / 10.0 > r[1].1 / 20.0);
    assert!(ctx.rows[6].is_empty(), "cell 4 has no time");
}
