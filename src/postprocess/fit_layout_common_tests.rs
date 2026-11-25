//! Outlier cuts on pseudobulk isolation scores, and how an embedding cell is
//! laid out.

use super::*;
use senna::run_manifest::RunKind;

/// `n` scores spread evenly over `[lo, hi]`.
fn spread(n: usize, lo: f64, hi: f64) -> Vec<f64> {
    (0..n)
        .map(|i| lo + (hi - lo) * i as f64 / (n - 1) as f64)
        .collect()
}

/// One mode with no tail: `auto` keeps every PB.
#[test]
fn auto_keeps_a_single_mode_whole() {
    let score = spread(500, 0.0, 1.0);
    assert!(!outlier_pbs(&score, OutlierCut::Auto).iter().any(|&o| o));
}

/// One long, right-skewed mode (a log-normal's quantiles): two-means
/// splits it, but no empty space parts the halves, so nothing is dropped.
#[test]
fn auto_keeps_a_skewed_single_mode_whole() {
    let score: Vec<f64> = (1..1000)
        .map(|i| {
            // The standard normal's quantile at i/1000, by its logit approximation.
            let p = f64::from(i) / 1000.0;
            (0.551_328_895 * (p / (1.0 - p)).ln() * 0.6).exp()
        })
        .collect();
    assert!(!outlier_pbs(&score, OutlierCut::Auto).iter().any(|&o| o));
}

/// A small group well clear of the rest is what `auto` drops, and only it.
#[test]
fn auto_drops_a_small_group_standing_clear() {
    let mut score = spread(475, 0.0, 1.0);
    score.extend(spread(25, 4.0, 4.5));
    let out = outlier_pbs(&score, OutlierCut::Auto);
    assert!(out[..475].iter().all(|&o| !o));
    assert!(out[475..].iter().all(|&o| o));
}

/// A clear group too large to be outliers (here 40%) is kept: it is a
/// population, not a tail.
#[test]
fn auto_keeps_a_group_too_large_to_be_outliers() {
    let mut score = spread(300, 0.0, 1.0);
    score.extend(spread(200, 4.0, 4.5));
    assert!(!outlier_pbs(&score, OutlierCut::Auto).iter().any(|&o| o));
}

#[test]
fn a_mad_cut_and_off_do_as_they_say() {
    let mut score = spread(99, 0.0, 1.0);
    score.push(10.0);
    let out = outlier_pbs(&score, OutlierCut::Mads(5.0));
    assert_eq!(out.iter().filter(|&&o| o).count(), 1);
    assert!(out[99]);
    assert!(!outlier_pbs(&score, OutlierCut::Off).iter().any(|&o| o));
    assert_eq!("auto".parse::<OutlierCut>(), Ok(OutlierCut::Auto));
    assert_eq!("off".parse::<OutlierCut>(), Ok(OutlierCut::Off));
    assert_eq!("3.5".parse::<OutlierCut>(), Ok(OutlierCut::Mads(3.5)));
    assert!("-1".parse::<OutlierCut>().is_err());
}

#[test]
fn one_pb_has_an_isolation_score_and_no_panic() {
    let pbs = Coarsened {
        membership: vec![0, 0],
        pb_edges: Vec::new(),
        pb_features: Mat::zeros(2, 1),
        pb_size: vec![2],
    };
    assert_eq!(isolation(&pbs, 15), vec![0.0]);
}

/// A MAD cut works on a few PBs too (only auto needs twenty), and finds
/// nothing where most scores tie, rather than every PB above the tie.
#[test]
fn a_mad_cut_works_on_few_pbs_and_not_on_ties() {
    let out = outlier_pbs(&[0.0, 0.1, 0.2, 0.1, 0.0, 9.0], OutlierCut::Mads(5.0));
    assert_eq!(out, [false, false, false, false, false, true]);
    assert!(
        !outlier_pbs(&[0.0, 0.1, 0.2, 0.1, 0.0, 9.0], OutlierCut::Auto)
            .iter()
            .any(|&o| o)
    );
    let mut tied = vec![1.0; 30];
    tied[0] = 1.5;
    assert!(!outlier_pbs(&tied, OutlierCut::Mads(3.0)).iter().any(|&o| o));
}

#[test]
fn an_embedding_cell_is_laid_out_by_its_direction_alone() {
    // Cells as rows at norms 1 to 10 and these angles: the first two
    // alike, the last along the first axis, far enough out on it that
    // clipping would shorten it.
    let angles = [
        0.52, 0.52, 0.54, 0.56, 0.58, 0.60, 0.62, 0.64, 0.66, 0.0_f32,
    ];
    let latent = Mat::from_fn(10, 2, |i, d| {
        (i + 1) as f32 * [angles[i].cos(), angles[i].sin()][d]
    });
    for kind in [RunKind::Bge, RunKind::Simba] {
        let (cells, _) = latent_layout_features(kind, &latent, 1.0, 0.5);
        assert!((cells.column(0) - cells.column(1)).norm() < 1e-6, "{cells}");
        for c in cells.column_iter() {
            assert!((c.norm() - 1.0).abs() < 1e-6, "{cells}");
        }
    }
}
