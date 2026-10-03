use super::*;

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
    let view = Mat::from_column_slice(4, 1, &[0.0, 1.0, 2.0, 10.0]);
    let valid = vec![true; 4];
    let pairs = vec![(0, 1), (2, 3), (0, 3)];
    // (2,3): 3 is 2's third neighbour, but 2 is 3's first.
    assert_eq!(pair_ranks(&view, &valid, &pairs), vec![1, 1, 3]);
}

#[test]
fn pair_rank_is_unknown_without_cells() {
    let view = Mat::from_column_slice(3, 1, &[0.0, 1.0, 2.0]);
    let valid = vec![true, false, true];
    assert_eq!(
        pair_ranks(&view, &valid, &[(0, 1), (0, 2)]),
        vec![u32::MAX, 1]
    );
}

#[test]
fn top_k_pairs_are_each_points_nearest() {
    let view = Mat::from_column_slice(4, 1, &[0.0, 1.0, 2.0, 10.0]);
    let mut p = top_k_pairs(&view, &[true; 4], 1);
    p.sort_unstable();
    p.dedup();
    assert_eq!(p, vec![(0, 1), (1, 2), (2, 3)]);
}

#[test]
fn level_keeps_pseudobulks_with_enough_cells() {
    let lv = Level::new(&[0, 0, 1, 2, 2, 2, usize::MAX], 2);
    assert_eq!(lv.pb_id, vec![0, 2]);
    assert_eq!(lv.n_cells, vec![2, 3]);
    assert_eq!(lv.pb_of_cell, vec![0, 0, usize::MAX, 1, 1, 1, usize::MAX]);
}

/// "Far" must be well clear of "near": beyond max(3k, P/4), never just k + 1.
#[test]
fn far_rank_leaves_a_gap_after_near() {
    assert_eq!(far_rank(106, 15, 0.25), 45);
    assert_eq!(far_rank(1000, 15, 0.25), 250);
    assert_eq!(far_rank(40, 5, 0.25), 15);
}

// ---- Consensus ------------------------------------------------------------------

#[test]
fn median_rank_skips_unknown_and_averages_the_middle_pair() {
    assert_eq!(median_rank([3, u32::MAX, 9, 5].into_iter()), Some(5.0));
    assert_eq!(median_rank([3, 9].into_iter()), Some(6.0));
    assert_eq!(median_rank([u32::MAX].into_iter()), None);
}

/// A model is judged against the others only: its own rank never votes.
#[test]
fn consensus_leaves_the_model_out() {
    let ranks = vec![vec![1], vec![60], vec![70]];
    assert_eq!(others_consensus(&ranks, 0, 0), Some(65.0));
    assert_eq!(others_consensus(&ranks, 1, 0), Some(35.5));
}

#[test]
fn pair_label_reads_the_all_model_median() {
    assert_eq!(PairLabel::of(Some(4.0), 15, 45), PairLabel::Similar);
    assert_eq!(PairLabel::of(Some(30.0), 15, 45), PairLabel::Ambiguous);
    assert_eq!(PairLabel::of(Some(46.0), 15, 45), PairLabel::Different);
    assert_eq!(PairLabel::of(None, 15, 45), PairLabel::Ambiguous);
}

// ---- Charges ----------------------------------------------------------------------

/// Two identical models never disagree, so neither is charged.
#[test]
fn agreeing_models_are_not_charged() {
    let ranks = vec![vec![1, 2, 60], vec![1, 2, 60]];
    let (t, ch) = tally(&ranks, 0, 15, 45);
    assert_eq!(t.near, 2);
    assert_eq!((t.merges, t.splits), (0, 0));
    assert!(ch.iter().all(|&c| c == Charge::None));
}

/// With two models, a pair one keeps near and the other holds far is a merge
/// for the first and a split for the second.
#[test]
fn two_models_charge_each_other() {
    let ranks = vec![vec![3, 50], vec![60, 2]];
    let (t0, ch0) = tally(&ranks, 0, 15, 45);
    assert_eq!(ch0, vec![Charge::Merge, Charge::Split]);
    assert_eq!((t0.merges, t0.splits), (1, 1));
    let (_, ch1) = tally(&ranks, 1, 15, 45);
    assert_eq!(ch1, vec![Charge::Split, Charge::Merge]);
}

/// With three models the majority decides: the lone model keeping the pair near
/// is charged a merge; the two that agree on far are not charged.
#[test]
fn the_majority_is_not_charged() {
    let ranks = vec![vec![3], vec![60], vec![70]];
    assert_eq!(tally(&ranks, 0, 15, 45).1, vec![Charge::Merge]);
    assert_eq!(tally(&ranks, 1, 15, 45).1, vec![Charge::None]);
    assert_eq!(tally(&ranks, 2, 15, 45).1, vec![Charge::None]);
}

#[test]
fn a_model_without_cells_is_not_charged() {
    let ranks = vec![vec![u32::MAX], vec![2]];
    assert_eq!(tally(&ranks, 0, 15, 45).1, vec![Charge::None]);
    assert_eq!(tally(&ranks, 1, 15, 45).1, vec![Charge::None]);
}

#[test]
fn pseudobulk_labels_combine_their_charges() {
    let pairs = [(0, 1), (1, 2), (3, 4)];
    let charges = [Charge::Merge, Charge::Split, Charge::None];
    let l = pb_labels(5, &pairs, &charges);
    let l: Vec<&str> = l.iter().map(AsRef::as_ref).collect();
    assert_eq!(
        l,
        ["merge", "merge+split", "split", "consistent", "consistent"]
    );
}
