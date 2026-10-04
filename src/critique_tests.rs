use super::*;

fn line_view(x: &[f32], valid: &[bool]) -> View {
    View::compact(Mat::from_column_slice(x.len(), 1, x), valid)
}

#[test]
fn tolerant_alignment_marks_missing_names() {
    let src: Vec<Box<str>> = vec!["a".into(), "x".into(), "c".into()];
    let tgt: Vec<Box<str>> = vec!["c".into(), "b".into(), "a".into()];
    assert_eq!(tolerant_align(&src, &tgt), vec![2, usize::MAX, 0]);
}

#[test]
fn rank_counts_the_strictly_closer_points() {
    let sorted = [0.5, 1.0, 1.0, 2.0];
    assert_eq!(rank_of(&sorted, 0.5), 1);
    assert_eq!(rank_of(&sorted, 1.0), 2);
    assert_eq!(rank_of(&sorted, 2.0), 4);
}

/// Four points on a line, 0 1 2 10: the pair (0, 3) is far from both ends.
#[test]
fn pair_rank_is_the_smaller_directional_rank() {
    let view = line_view(&[0.0, 1.0, 2.0, 10.0], &[true; 4]);
    // (2,3): 3 is 2's third neighbour, but 2 is 3's first.
    assert_eq!(view.pair_ranks(&[(0, 1), (2, 3), (0, 3)]), vec![1, 1, 3]);
}

#[test]
fn pair_rank_is_unknown_without_cells() {
    let view = line_view(&[0.0, 1.0, 2.0], &[true, false, true]);
    assert_eq!(view.pair_ranks(&[(0, 1), (0, 2)]), vec![u32::MAX, 1]);
}

#[test]
fn top_k_pairs_are_each_points_nearest() {
    let view = line_view(&[0.0, 1.0, 2.0, 10.0], &[true; 4]);
    let mut p = view.top_k_pairs(1);
    p.sort_unstable();
    p.dedup();
    assert_eq!(p, vec![(0, 1), (1, 2), (2, 3)]);
}

/// A pseudobulk without cells is skipped, and indices stay those of the level.
#[test]
fn top_k_pairs_skip_pseudobulks_without_cells() {
    let view = line_view(&[0.0, 99.0, 1.0, 5.0], &[true, false, true, true]);
    let mut p = view.top_k_pairs(1);
    p.sort_unstable();
    p.dedup();
    assert_eq!(p, vec![(0, 2), (2, 3)]);
}

#[test]
fn level_keeps_pseudobulks_with_enough_cells() {
    let lv = Level::new(&[0, 0, 1, 2, 2, 2, usize::MAX], 2);
    assert_eq!(lv.pb_id, vec![0, 2]);
    assert_eq!(lv.n_cells, vec![2, 3]);
    assert_eq!(lv.pb_of_cell, vec![0, 0, usize::MAX, 1, 1, 1, usize::MAX]);
}

// ---- Bounds -----------------------------------------------------------------------

/// "Far" must be well clear of "near": beyond max(2K, P/4), never just K + 1.
/// Swept against expert cell types on two donors: rules set by K alone do not
/// carry over between datasets; the P/4 part does.
#[test]
fn far_leaves_a_gap_after_near() {
    let far = |p, k| Bounds::for_level(p, k, 0.25).map(|b| b.far);
    assert_eq!(far(106, 15), Some(30));
    assert_eq!(far(420, 15), Some(105));
    assert_eq!(far(1000, 15), Some(250));
    assert_eq!(far(40, 5), Some(10));
    assert_eq!(Bounds::for_level(106, 15, 0.25).map(|b| b.near), Some(15));
}

/// Ranks run to P − 1, so a level too small to hold a rank beyond far has no bounds.
#[test]
fn a_level_too_small_for_far_has_no_bounds() {
    assert!(Bounds::for_level(32, 15, 0.25).is_some());
    assert!(Bounds::for_level(31, 15, 0.25).is_none());
    assert!(Bounds::for_level(20, 15, 0.25).is_none());
}

#[test]
fn parameters_are_checked() {
    assert!(check_params(15, 0.25, 10).is_ok());
    assert!(check_params(0, 0.25, 10).is_err());
    assert!(check_params(15, 0.0, 10).is_err());
    assert!(check_params(15, 1.0, 10).is_err());
    assert!(check_params(15, f64::NAN, 10).is_err());
    assert!(check_params(15, 0.25, 0).is_err());
}

// ---- Consensus ------------------------------------------------------------------

#[test]
fn known_ranks_skip_unknown_and_sort() {
    let ranks = vec![vec![9], vec![u32::MAX], vec![3], vec![5]];
    assert_eq!(sorted_known(&ranks, 0), vec![3, 5, 9]);
}

#[test]
fn median_averages_the_middle_pair() {
    assert_eq!(median(&[3, 5, 9]), Some(5.0));
    assert_eq!(median(&[3, 9]), Some(6.0));
    assert_eq!(median(&[]), None);
}

/// A model is answered by its committee only: its own rank never votes.
#[test]
fn the_answer_is_the_committees_median() {
    let ranks = vec![vec![1, 5], vec![60, u32::MAX], vec![70, 9], vec![10, 11]];
    assert_eq!(answers(&ranks, &[1, 2]), vec![Some(65.0), Some(9.0)]);
    assert_eq!(answers(&ranks, &[1, 2, 3]), vec![Some(60.0), Some(10.0)]);
    assert_eq!(answers(&ranks, &[1]), vec![Some(60.0), None]);
}

/// All other models by default; a random subset of a given size otherwise,
/// never the model itself, the same for the same seed.
#[test]
fn committees_are_random_subsets_of_the_others() {
    let rng = |s| rand::rngs::SmallRng::seed_from_u64(s);
    assert_eq!(committee(4, 1, 0, &mut rng(7)), vec![0, 2, 3]);
    assert_eq!(committee(4, 1, 9, &mut rng(7)), vec![0, 2, 3]);
    let c = committee(6, 2, 3, &mut rng(7));
    assert_eq!(c.len(), 3);
    assert!(!c.contains(&2));
    assert!(c.windows(2).all(|w| w[0] < w[1]));
    assert_eq!(c, committee(6, 2, 3, &mut rng(7)));
}

// ---- Merges -----------------------------------------------------------------------

const B: Bounds = Bounds { near: 15, far: 45 };

fn merges_of(ranks: &[Vec<u32>], model: usize) -> Vec<bool> {
    let others: Vec<usize> = (0..ranks.len()).filter(|&o| o != model).collect();
    merges(&ranks[model], &answers(ranks, &others), B)
}

/// Two identical models never disagree, so neither merges anything.
#[test]
fn agreeing_models_do_not_merge() {
    let ranks = vec![vec![1, 2, 60], vec![1, 2, 60]];
    assert_eq!(merges_of(&ranks, 0), vec![false; 3]);
    assert_eq!(near_count(&ranks[0], B), 2);
}

/// A model merges a pair it keeps near while the others hold it far. Holding
/// far what the others keep near is not a critique: that model is usually right.
#[test]
fn only_keeping_near_what_the_others_hold_far_is_a_merge() {
    let ranks = vec![vec![3, 50], vec![60, 2]];
    assert_eq!(merges_of(&ranks, 0), vec![true, false]);
    assert_eq!(merges_of(&ranks, 1), vec![false, true]);
}

/// With three models the majority decides: only the lone model merges.
#[test]
fn the_majority_does_not_merge() {
    let ranks = vec![vec![3], vec![60], vec![70]];
    assert_eq!(merges_of(&ranks, 0), vec![true]);
    assert_eq!(merges_of(&ranks, 1), vec![false]);
    assert_eq!(merges_of(&ranks, 2), vec![false]);
}

#[test]
fn a_model_without_cells_does_not_merge() {
    let ranks = vec![vec![u32::MAX], vec![2]];
    assert_eq!(merges_of(&ranks, 0), vec![false]);
    assert_eq!(merges_of(&ranks, 1), vec![false]);
}

/// The report card: how often a pair the model keeps near is one the others
/// hold far.
#[test]
fn merge_rate_is_merges_over_near_pairs() {
    assert!((merge_rate(5, 20) - 0.25).abs() < 1e-6);
    assert!(merge_rate(0, 0).is_nan());
}

#[test]
fn names_stay_unique_after_suffixing() {
    let names = unique_names(["a", "a", "a_2", "b"].map(String::from).to_vec());
    let mut sorted = names.clone();
    sorted.sort();
    sorted.dedup();
    assert_eq!(sorted.len(), 4, "{names:?}");
    assert_eq!(names[0], "a");
    assert_eq!(names[3], "b");
}

/// A non-finite averaged latent says nothing about distance: its pseudobulk
/// leaves the view instead of becoming everyone's nearest neighbour.
#[test]
fn non_finite_rows_leave_the_view() {
    let view = line_view(&[0.0, f32::NAN, 1.0, 5.0], &[true; 4]);
    assert_eq!(
        view.pair_ranks(&[(0, 1), (1, 3), (0, 2)]),
        vec![u32::MAX, u32::MAX, 1]
    );
    assert!(view.top_k_pairs(3).iter().all(|&(a, b)| a != 1 && b != 1));
}

// ---- Cell labels: a known answer to check the critique against -----------------

#[test]
fn cell_labels_read_the_named_column() {
    let tsv = "barcode\tDonor\tCellType\nc1\tBM1\tB\nc2\tBM1\tNA\nc3\tBM1\tT\nc4\tBM1\t\n";
    let m = parse_cell_labels(tsv.as_bytes(), "CellType").unwrap();
    assert_eq!(m.get("c1").map(AsRef::as_ref), Some("B"));
    assert_eq!(m.get("c3").map(AsRef::as_ref), Some("T"));
    assert_eq!(m.len(), 2, "NA and empty labels are left out");
    assert!(parse_cell_labels(tsv.as_bytes(), "Missing").is_err());
}

/// pb 0 holds B, B, T; pb 1 holds T; pb 2's one cell has no label.
#[test]
fn composition_is_the_fraction_of_each_label() {
    let pb_of_cell = [0, 0, 0, 1, 2, usize::MAX];
    let label_of_cell = [Some(0), Some(0), Some(1), Some(1), None, Some(0)];
    let comp = label_composition(&pb_of_cell, &label_of_cell, 3, 2);
    let close = |a: &[f32], b: &[f32]| a.iter().zip(b).all(|(x, y)| (x - y).abs() < 1e-6);
    assert!(close(comp[0].as_ref().unwrap(), &[2.0 / 3.0, 1.0 / 3.0]));
    assert!(close(comp[1].as_ref().unwrap(), &[0.0, 1.0]));
    assert!(comp[2].is_none());
}

/// The shared mass of two compositions: 1 for the same mix, 0 for no shared label.
#[test]
fn overlap_is_the_shared_mass() {
    assert!((overlap(&[0.5, 0.5, 0.0], &[0.5, 0.0, 0.5]) - 0.5).abs() < 1e-6);
    assert_eq!(overlap(&[1.0, 0.0], &[0.0, 1.0]), 0.0);
    assert!((overlap(&[0.25, 0.75], &[0.25, 0.75]) - 1.0).abs() < 1e-6);
}

#[test]
fn mean_over_skips_unknown_overlaps() {
    let ov = [0.0, f32::NAN, 0.5, 1.0];
    let pick = [true, true, true, false];
    assert!((mean_over(&ov, &pick) - 0.25).abs() < 1e-6);
    assert!(mean_over(&ov, &[false; 4]).is_nan());
}

// ---- Question sampling: informative, diverse, random ----------------------------

/// A merge's weight is the share of the committee holding the pair far.
#[test]
fn label_weight_is_the_share_of_far_votes() {
    let ranks = vec![vec![3], vec![60], vec![20], vec![u32::MAX]];
    assert!((far_share(&ranks, &[1, 2, 3], 0, B) - 0.5).abs() < 1e-6);
    assert!((far_share(&ranks, &[1], 0, B) - 1.0).abs() < 1e-6);
}

/// Two pairs are as far apart as their endpoints, matched the better way round.
#[test]
fn pair_distance_matches_endpoints_the_better_way() {
    let pos = [0.0f32, 1.0, 10.0, 11.0];
    let d = |i: usize, j: usize| (pos[i] - pos[j]).abs();
    assert_eq!(pair_distance(&d, (0, 2), (1, 3)), 2.0);
    assert_eq!(pair_distance(&d, (0, 2), (3, 1)), 2.0);
    assert_eq!(pair_distance(&d, (0, 1), (0, 1)), 0.0);
}

/// k-means++ seeding draws only items with weight, each at most once, and the
/// same items for the same seed.
#[test]
fn seeding_draws_weighted_items_once() {
    let w = [1.0, 0.0, 2.0, 3.0, 0.0];
    let d = |i: usize, j: usize| (i as f64 - j as f64).abs();
    let picked = kmeanspp(&w, &d, 10, &mut rand::rngs::SmallRng::seed_from_u64(3));
    let mut sorted = picked.clone();
    sorted.sort_unstable();
    assert_eq!(sorted, vec![0, 2, 3]);
    let again = kmeanspp(&w, &d, 10, &mut rand::rngs::SmallRng::seed_from_u64(3));
    assert_eq!(picked, again);
}

/// Two tight clusters of equal weight: a draw of two almost always takes one
/// from each, because the second draw is weighted by distance to the first.
#[test]
fn seeding_spreads_draws_across_clusters() {
    let pos = [0.0f64, 0.1, 0.2, 100.0, 100.1, 100.2];
    let w = [1.0; 6];
    let d = |i: usize, j: usize| (pos[i] - pos[j]).abs();
    let spread = (0..200)
        .filter(|&s| {
            let p = kmeanspp(&w, &d, 2, &mut rand::rngs::SmallRng::seed_from_u64(s));
            (p[0] < 3) != (p[1] < 3)
        })
        .count();
    assert!(
        spread >= 195,
        "only {spread} of 200 draws spread across clusters"
    );
}
