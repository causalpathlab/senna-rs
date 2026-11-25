//! The scoring δ: the sums against the dense formula, cross-fitting, trained
//! and fitted batches, and the projection fold.

use super::*;
use crate::topic::eval::{build_gene_remap_with, QueryNameOpts};

/// The formula as it stood before this module: scatter the block dense
/// on the training axis, then sum `w·x` and `w·μ` over every gene.
fn reference(
    csc: &CscMatrix<f32>,
    theta_nk: &Mat,
    remap: Option<&GeneRemap>,
    beta_dk: &Mat,
    phi: Option<&[f32]>,
    batch_ids: &[usize],
    n_batches: usize,
) -> Mat {
    let d_train = beta_dk.nrows();
    let mut x = Mat::zeros(d_train, csc.ncols());
    for (j, col) in csc.col_iter().enumerate() {
        for (&row, &v) in col.row_indices().iter().zip(col.values()) {
            match remap {
                Some(r) => {
                    if let Some(t) = r.new_to_train[row] {
                        x[(t, j)] += v;
                    }
                }
                None => x[(row, j)] = v,
            }
        }
    }
    let pred_dn = beta_dk.map(f32::exp) * theta_nk.transpose();
    let mut sums = DeltaSums {
        pb_obs: Mat::zeros(d_train, n_batches),
        pb_pred: Mat::zeros(d_train, n_batches),
    };
    for j in 0..csc.ncols() {
        let b = batch_ids[j];
        // The cell's counts on the model's genes.
        let lib: f32 = x.column(j).sum();
        for d in 0..d_train {
            let mu = lib * pred_dn[(d, j)];
            let w = nb_fisher_weight(phi, mu, d);
            sums.pb_obs[(d, b)] += w * x[(d, j)];
            sums.pb_pred[(d, b)] += w * mu;
        }
    }
    solve_delta_from_sums(&sums)
}

/// δ by this module, the cells split into two blocks.
fn by_blocks(
    csc: &CscMatrix<f32>,
    theta_nk: &Mat,
    remap: Option<&GeneRemap>,
    beta_dk: &Mat,
    phi: Option<&[f32]>,
    batch_ids: &[usize],
    n_batches: usize,
) -> Mat {
    let exp_beta_dk = beta_dk.map(f32::exp);
    let inp = DeltaInputs {
        theta_nk,
        gene_remap: remap,
        rate: RateModel::Topic {
            exp_beta_dk: &exp_beta_dk,
        },
        phi,
    };
    let n = csc.ncols();
    let mut total = Acc::zeros(&inp, n_batches);
    for (lb, ub) in [(0, n / 2), (n / 2, n)] {
        let mut acc = Acc::zeros(&inp, n_batches);
        let block = column_block(csc, lb, ub);
        let groups: Vec<Option<usize>> = batch_ids[lb..ub].iter().map(|&b| Some(b)).collect();
        accumulate(&mut acc, &inp, &block, theta_nk.rows(lb, ub - lb), &groups);
        total = total.merge(acc);
    }
    solve_delta_from_sums(&total.finish(&inp).sums)
}

fn column_block(csc: &CscMatrix<f32>, lb: usize, ub: usize) -> CscMatrix<f32> {
    let mut coo = nalgebra_sparse::CooMatrix::new(csc.nrows(), ub - lb);
    for (j, col) in csc.col_iter().enumerate().take(ub).skip(lb) {
        for (&r, &v) in col.row_indices().iter().zip(col.values()) {
            coo.push(r, j - lb, v);
        }
    }
    CscMatrix::from(&coo)
}

struct Example {
    csc: CscMatrix<f32>,
    theta: Mat,
    remap: GeneRemap,
    beta: Mat,
    batches: Vec<usize>,
}

/// Five query rows (one unmatched, two on one training gene) × six
/// cells in two batches, three topics over four training genes.
fn example() -> Example {
    let (rows, cells) = (5, 6);
    let mut coo = nalgebra_sparse::CooMatrix::new(rows, cells);
    for j in 0..cells {
        for r in 0..rows {
            let v = ((3 * j + 5 * r) % 7) as f32;
            if v > 1.0 {
                coo.push(r, j, v - 1.0);
            }
        }
    }
    let theta = Mat::from_fn(cells, 3, |j, k| 1.0 + ((j + 2 * k) % 4) as f32);
    let theta = Mat::from_rows(
        &theta
            .row_iter()
            .map(|r| r.unscale(r.sum()))
            .collect::<Vec<_>>(),
    );
    let beta = Mat::from_fn(4, 3, |d, k| -1.0 - 0.3 * ((d * 3 + k) % 5) as f32);
    let query: Vec<Box<str>> = ["GENE1", "GENE9", "GENE2", "GENE3", "GENE3"]
        .map(Box::from)
        .to_vec();
    let train: Vec<Box<str>> = ["GENE0", "GENE1", "GENE2", "GENE3"].map(Box::from).to_vec();
    let remap = build_gene_remap_with(&train, &query, &QueryNameOpts::default());
    let batches = (0..cells).map(|j| j % 2).collect();
    Example {
        csc: CscMatrix::from(&coo),
        theta,
        remap,
        beta,
        batches,
    }
}

#[test]
fn delta_matches_the_dense_formula() {
    let Example {
        csc,
        theta,
        remap,
        beta,
        batches,
    } = example();
    let phi = [0.5_f32, 2.0, 7.0, 0.1];
    assert_eq!(remap.new_to_train[4], Some(3));
    for phi in [None, Some(&phi[..])] {
        let want = reference(&csc, &theta, Some(&remap), &beta, phi, &batches, 2);
        let got = by_blocks(&csc, &theta, Some(&remap), &beta, phi, &batches, 2);
        let gap = (&want - &got).abs().max();
        assert!(gap < 1e-4, "phi {phi:?}: gap {gap}\n{want}\n{got}");
        assert!(want.iter().any(|&v| (v - 1.0).abs() > 0.05));
    }
    // No remap: the query rows are the training rows.
    let csc4 = {
        let mut coo = nalgebra_sparse::CooMatrix::new(4, csc.ncols());
        for (j, col) in csc.col_iter().enumerate() {
            for (&r, &v) in col.row_indices().iter().zip(col.values()) {
                coo.push(r % 4, j, v);
            }
        }
        CscMatrix::from(&coo)
    };
    for phi in [None, Some(&phi[..])] {
        let want = reference(&csc4, &theta, None, &beta, phi, &batches, 2);
        let got = by_blocks(&csc4, &theta, None, &beta, phi, &batches, 2);
        assert!((&want - &got).abs().max() < 1e-4, "phi {phi:?}");
    }
}

/// Counts drawn from `δ*·exp(β)·θ` in two batches, scored as the held-out
/// multinomial per count under: the true δ (the oracle), no δ, δ fitted
/// on the scored cells, and δ cross-fitted.
struct Scores {
    oracle: f64,
    none: f64,
    in_sample: f64,
    crossfit: f64,
}

fn planted_scores(phi: Option<&[f32]>) -> Scores {
    use rand::{rngs::SmallRng, SeedableRng};
    use rand_distr::{Distribution, Poisson};
    let (d, k, n, n_batches, folds, lib) = (150usize, 3usize, 300usize, 2usize, 5usize, 40.0);
    let mut rng = SmallRng::seed_from_u64(11);
    let exp_beta = {
        let raw = Mat::from_fn(
            d,
            k,
            |g, t| if g % k == t { 6.0 } else { 1.0 } + (g % 7) as f32,
        );
        Mat::from_columns(
            &raw.column_iter()
                .map(|c| c.unscale(c.sum()))
                .collect::<Vec<_>>(),
        )
    };
    let theta = {
        let raw = Mat::from_fn(n, k, |j, t| 1.0 + ((j * 5 + t * 3) % 6) as f32);
        Mat::from_rows(
            &raw.row_iter()
                .map(|r| r.unscale(r.sum()))
                .collect::<Vec<_>>(),
        )
    };
    let delta_true = Mat::from_fn(d, n_batches, |g, b| match (b, g % 3) {
        (1, 0) => 3.0,
        (1, 1) => 0.4,
        _ => 1.0,
    });
    let batch_of = |j: usize| (j / 3) % n_batches;
    // The normalised rate of each cell under a δ read by `delta_of(cell)`.
    let rate = |delta_of: &dyn Fn(usize) -> DVec| -> Mat {
        let mut r = &exp_beta * theta.transpose();
        for (j, mut col) in r.column_iter_mut().enumerate() {
            col.component_mul_assign(&delta_of(j));
            let z = col.sum();
            col /= z;
        }
        r
    };
    let truth = rate(&|j| delta_true.column(batch_of(j)).into_owned());
    let mut coo = nalgebra_sparse::CooMatrix::new(d, n);
    for j in 0..n {
        for g in 0..d {
            let x: f64 = Poisson::new(f64::from(lib * truth[(g, j)]))
                .expect("rate")
                .sample(&mut rng);
            if x > 0.0 {
                coo.push(g, j, x as f32);
            }
        }
    }
    let csc = CscMatrix::from(&coo);
    let score = |delta_of: &dyn Fn(usize) -> DVec| -> f64 {
        let r = rate(delta_of);
        let (mut llik, mut count) = (0f64, 0f64);
        for (j, col) in csc.col_iter().enumerate() {
            for (&g, &x) in col.row_indices().iter().zip(col.values()) {
                llik += f64::from(x) * f64::from(r[(g, j)]).ln();
                count += f64::from(x);
            }
        }
        llik / count
    };

    let inp = DeltaInputs {
        theta_nk: &theta,
        gene_remap: None,
        rate: RateModel::Topic {
            exp_beta_dk: &exp_beta,
        },
        phi,
    };
    let plans = vec![BatchPlan::Fitted; n_batches];
    let layout = Layout::new(&plans, folds);
    let batches: Vec<usize> = (0..n).map(batch_of).collect();
    let groups: Vec<usize> = assign_groups(&batches, &layout)
        .into_iter()
        .map(|g| g as usize)
        .collect();
    let mut acc = Acc::zeros(&inp, layout.n_groups());
    let some: Vec<Option<usize>> = groups.iter().map(|&g| Some(g)).collect();
    accumulate(&mut acc, &inp, &csc, theta.rows(0, n), &some);
    let total = acc.finish(&inp);
    let fit = |crossfit: bool| -> Mat {
        let mut delta = Mat::from_element(d, layout.n_groups(), 1.0);
        let mut fit_obs = Mat::zeros(d, layout.n_groups());
        let mut batch_delta = Mat::from_element(d, n_batches, 1.0);
        fill_fitted(
            &total,
            &plans,
            &layout,
            crossfit,
            &mut FittedOut {
                delta: &mut delta,
                fit_obs: &mut fit_obs,
                batch_delta: &mut batch_delta,
            },
        );
        // The raw counts beside a group's δ are those of the cells it was
        // fitted on: the batch's, less the group's own when cross-fitted.
        let all: f32 = csc.values().iter().sum();
        let per_fold = if crossfit { folds - 1 } else { folds } as f32;
        assert!((fit_obs.sum() - per_fold * all).abs() < 1e-2 * all);
        if !crossfit {
            // One δ per batch, in every one of its columns.
            for b in 0..n_batches {
                for g in layout.columns(b) {
                    assert_eq!(delta.column(g), batch_delta.column(b));
                }
            }
        }
        delta
    };
    let (in_sample, crossfit) = (fit(false), fit(true));
    Scores {
        oracle: score(&|j| delta_true.column(batch_of(j)).into_owned()),
        none: score(&|_| DVec::from_element(d, 1.0)),
        in_sample: score(&|j| in_sample.column(groups[j]).into_owned()),
        crossfit: score(&|j| crossfit.column(groups[j]).into_owned()),
    }
}

// Bar: scored under a δ fitted on the scored cells, the held-out
// likelihood beats the oracle's (the true δ), which nothing honest can;
// cross-fitted it does not, and it still recovers most of what the true
// δ gains over no δ.
#[test]
fn a_cross_fitted_delta_is_not_optimistic() {
    let phi = vec![5.0_f32; 150];
    for phi in [None, Some(&phi[..])] {
        let s = planted_scores(phi);
        eprintln!(
            "phi {}: oracle {:.4}, none {:.4}, in-sample {:.4}, cross-fitted {:.4}",
            phi.is_some(),
            s.oracle,
            s.none,
            s.in_sample,
            s.crossfit
        );
        assert!(s.in_sample > s.oracle + 1e-3, "in-sample is not optimistic");
        assert!(s.crossfit < s.oracle, "cross-fitted beats the oracle");
        let (gain, kept) = (s.oracle - s.none, s.crossfit - s.none);
        assert!(gain > 0.05, "the planted δ gains only {gain}");
        assert!(kept > 0.8 * gain, "cross-fitted keeps {kept} of {gain}");
    }
}

// Bar: a batch's cells are dealt to its folds in turn, so a batch of
// two cells is never left with a fold that has nothing to be fitted on;
// a batch that is not cross-fitted has one group.
#[test]
fn folds_are_dealt_within_each_batch() {
    let plans = [BatchPlan::Fitted, BatchPlan::Trained(0), BatchPlan::Fitted];
    let layout = Layout::new(&plans, 5);
    assert_eq!(layout.n_groups(), 11);
    let batches = [0, 1, 2, 0, 1, 0, 0, 0, 0, 2, 1];
    let groups = assign_groups(&batches, &layout);
    let of = |b: usize| -> Vec<u32> {
        let at = batches.iter().zip(&groups).filter(|(x, _)| **x == b);
        at.map(|(_, g)| *g).collect()
    };
    // Batch 0: its first five cells take its five folds, once each.
    let mut round = of(0)[..5].to_vec();
    round.sort_unstable();
    assert_eq!(round, [0, 1, 2, 3, 4]);
    assert_eq!(of(1), [5, 5, 5]);
    // Batch 2 has two cells: in different folds, so each has the other.
    assert!(of(2)[0] != of(2)[1] && of(2).iter().all(|g| (6..11).contains(g)));
    // A repeating pattern of the folds' period is not dealt one kind to
    // a fold: the rounds start at different folds.
    let one = Layout::new(&[BatchPlan::Fitted], 5);
    let dealt = assign_groups(&[0; 50], &one);
    let kind0: std::collections::HashSet<u32> = dealt.iter().step_by(5).copied().collect();
    assert!(kind0.len() > 1, "{kind0:?}");
}

// Bar: the null of a group has the batch term its δ has. A trained δ
// multiplies a batch-free null and leaves a held-out one alone; a fitted
// δ's null is the fitting cells' composition, on the scored genes only.
#[test]
fn the_null_takes_the_batch_term_of_its_group() {
    let sd = ScoringDelta {
        delta: Mat::from_row_slice(3, 2, &[2.0, 1.0, 0.5, 1.0, 1.0, 1.0]),
        group: vec![0, 1],
        fit_obs: Mat::from_row_slice(3, 2, &[0.0, 7.0, 0.0, 3.0, 0.0, 9.0]),
        fitted: vec![false, true],
        batch: vec![0, 1],
        batch_delta: Mat::from_element(3, 2, 1.0),
        label: "trained and cross-fitted, by batch",
    };
    let null = [0.5, 0.0, 0.5];
    let scored = [0, 1];
    assert_eq!(
        sd.null_composition(0, &null, &scored, true),
        [1.0, 0.0, 0.5]
    );
    assert_eq!(sd.null_composition(0, &null, &scored, false), null);
    // Gene 1 is scored with no pooled mass and still reads its counts;
    // gene 2 is not scored.
    assert_eq!(
        sd.null_composition(1, &null, &scored, true),
        [8.0, 3.0, 0.0]
    );
}

// Bar: the cross-fit prior shrinks to the batch's own scale, so genes
// the model lacks (which the predicted sum spans and the observed one
// does not) do not pull every δ down.
#[test]
fn the_prior_shrinks_to_the_batch_scale() {
    assert!((delta_ratio(0.0, 0.0, 1.0, 0.25) - 0.25).abs() < 1e-6);
    assert!((delta_ratio(25.0, 100.0, 1.0, 0.25) - 0.25).abs() < 1e-6);
    assert!((delta_ratio(3.0, 2.0, 0.0, 9.0) - 1.5).abs() < 1e-6);
}

// Bar: with no batch effect, cells that differ in how much of their
// library sits on genes the model lacks still give one δ for every gene.
// The predicted total is the cell's counts on the model's genes; under the
// whole library, each cell's prediction would be scaled by its unmapped
// share, and the genes of the cells with more of it would read a lower δ.
#[test]
fn unmapped_counts_do_not_bend_delta() {
    // Two model genes; a third query row the model lacks. Cells 0-1 are
    // all gene A with nothing unmapped, cells 2-3 all gene B with three
    // quarters of their counts unmapped.
    let exp_beta = Mat::from_row_slice(2, 2, &[1.0, 0.0, 0.0, 1.0]);
    let theta = Mat::from_row_slice(4, 2, &[1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0]);
    let mut coo = nalgebra_sparse::CooMatrix::new(3, 4);
    for j in 0..2 {
        coo.push(0, j, 8.0);
    }
    for j in 2..4 {
        coo.push(1, j, 2.0);
        coo.push(2, j, 6.0);
    }
    let csc = CscMatrix::from(&coo);
    let query: Vec<Box<str>> = ["A", "B", "X"].map(Box::from).to_vec();
    let train: Vec<Box<str>> = ["A", "B"].map(Box::from).to_vec();
    let remap = build_gene_remap_with(&train, &query, &QueryNameOpts::default());
    let inp = DeltaInputs {
        theta_nk: &theta,
        gene_remap: Some(&remap),
        rate: RateModel::Topic {
            exp_beta_dk: &exp_beta,
        },
        phi: None,
    };
    let mut acc = Acc::zeros(&inp, 1);
    accumulate(&mut acc, &inp, &csc, theta.rows(0, 4), &[Some(0); 4]);
    let delta = solve_delta_from_sums(&acc.finish(&inp).sums);
    assert!((delta[(0, 0)] - 1.0).abs() < 1e-6, "{delta}");
    assert!((delta[(1, 0)] - 1.0).abs() < 1e-6, "{delta}");
}
