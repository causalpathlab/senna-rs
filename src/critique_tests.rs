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

/// "Far" must be well clear of "near": beyond max(3k, P/4), never just k + 1.
#[test]
fn far_leaves_a_gap_after_near() {
    let far = |p, k| Bounds::for_level(p, k, 0.25).map(|b| b.far);
    assert_eq!(far(106, 15), Some(45));
    assert_eq!(far(1000, 15), Some(250));
    assert_eq!(far(40, 5), Some(15));
    assert_eq!(Bounds::for_level(106, 15, 0.25).map(|b| b.near), Some(15));
}

/// Ranks run to P − 1, so a level too small to hold a rank beyond far has no bounds.
#[test]
fn a_level_too_small_for_far_has_no_bounds() {
    assert!(Bounds::for_level(47, 15, 0.25).is_some());
    assert!(Bounds::for_level(46, 15, 0.25).is_none());
    assert!(Bounds::for_level(40, 15, 0.25).is_none());
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

/// A model is judged against the others only: its own rank never votes.
#[test]
fn median_without_leaves_one_rank_out() {
    assert_eq!(median_without(&[1, 60, 70], 1), Some(65.0));
    assert_eq!(median_without(&[1, 60, 70], 60), Some(35.5));
    assert_eq!(median_without(&[5, 5, 80], 5), Some(42.5));
    assert_eq!(median_without(&[7], 7), None);
}

#[test]
fn pair_label_reads_the_all_model_median() {
    let b = Bounds { near: 15, far: 45 };
    assert_eq!(PairLabel::of(Some(4.0), b), PairLabel::Similar);
    assert_eq!(PairLabel::of(Some(30.0), b), PairLabel::Ambiguous);
    assert_eq!(PairLabel::of(Some(46.0), b), PairLabel::Different);
    assert_eq!(PairLabel::of(None, b), PairLabel::Ambiguous);
}

// ---- Charges ----------------------------------------------------------------------

const B: Bounds = Bounds { near: 15, far: 45 };

fn charges_of(ranks: &[Vec<u32>], model: usize) -> Vec<Charge> {
    let sorted: Vec<Vec<u32>> = (0..ranks[0].len())
        .map(|c| sorted_known(ranks, c))
        .collect();
    charges(&ranks[model], &sorted, B)
}

/// Two identical models never disagree, so neither is charged.
#[test]
fn agreeing_models_are_not_charged() {
    let ranks = vec![vec![1, 2, 60], vec![1, 2, 60]];
    assert!(charges_of(&ranks, 0).iter().all(|&c| c == Charge::None));
    assert_eq!(near_count(&ranks[0], B), 2);
}

/// With two models, a pair one keeps near and the other holds far is a merge
/// for the first and a split for the second.
#[test]
fn two_models_charge_each_other() {
    let ranks = vec![vec![3, 50], vec![60, 2]];
    assert_eq!(charges_of(&ranks, 0), vec![Charge::Merge, Charge::Split]);
    assert_eq!(charges_of(&ranks, 1), vec![Charge::Split, Charge::Merge]);
}

/// With three models the majority decides: the lone model keeping the pair near
/// is charged a merge; the two that agree on far are not charged.
#[test]
fn the_majority_is_not_charged() {
    let ranks = vec![vec![3], vec![60], vec![70]];
    assert_eq!(charges_of(&ranks, 0), vec![Charge::Merge]);
    assert_eq!(charges_of(&ranks, 1), vec![Charge::None]);
    assert_eq!(charges_of(&ranks, 2), vec![Charge::None]);
}

#[test]
fn a_model_without_cells_is_not_charged() {
    let ranks = vec![vec![u32::MAX], vec![2]];
    assert_eq!(charges_of(&ranks, 0), vec![Charge::None]);
    assert_eq!(charges_of(&ranks, 1), vec![Charge::None]);
}

#[test]
fn pseudobulk_labels_combine_their_charges() {
    let pairs = [(0, 1), (1, 2), (3, 4)];
    let charges = [Charge::Merge, Charge::Split, Charge::None];
    assert_eq!(
        pb_labels(5, &pairs, &charges),
        ["merge", "merge+split", "split", "consistent", "consistent"]
    );
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
