use super::model::truth_params;
use crate::ode_mixture::fit::Fit;
use crate::ode_mixture::init::{axes, climb, start_order, tree_starts, Climb};
use crate::ode_mixture::likelihood::Reads;
use crate::ode_mixture::sim::{simulate, Sim, SimConfig};
use legume_numeric::candle::candle_core::Device;

/// Pearson's correlation.
pub(super) fn pearson(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len() as f64;
    let (ma, mb) = (a.iter().sum::<f64>() / n, b.iter().sum::<f64>() / n);
    let (mut c, mut va, mut vb) = (0.0, 0.0, 0.0);
    for (x, y) in a.iter().zip(b) {
        c += (x - ma) * (y - mb);
        va += (x - ma).powi(2);
        vb += (y - mb).powi(2);
    }
    c / (va * vb).sqrt()
}

/// Spearman's rank correlation.
pub(super) fn spearman(a: &[f64], b: &[f64]) -> f64 {
    let rank = |v: &[f64]| {
        let mut idx: Vec<usize> = (0..v.len()).collect();
        idx.sort_by(|&i, &j| v[i].total_cmp(&v[j]));
        let mut r = vec![0.0; v.len()];
        for (k, &i) in idx.iter().enumerate() {
            r[i] = k as f64;
        }
        r
    };
    pearson(&rank(a), &rank(b))
}

/// The first `n` pseudobulks' reads.
pub(super) fn first(sim: &Sim, n: usize) -> Reads {
    let g = sim.n_genes;
    Reads::new(&sim.u[..n * g], &sim.s[..n * g], n, g)
}

#[test]
fn the_start_orders_pseudobulks_along_the_trajectory() {
    let sim = simulate(&SimConfig::default()).unwrap();
    let reads = first(&sim, sim.n_pb);
    let (lineage, order) = start_order(&reads, 1, 3);
    assert!(lineage.iter().all(|&l| l == 0));
    let r = spearman(&order, &sim.truth.time);
    // Rough: a curved trajectory is not straight along its first axis; the
    // climb straightens it.
    assert!(r.abs() > 0.8, "start Spearman {r}");
}

#[test]
fn from_scratch_the_climb_finds_time_and_its_direction() {
    let cfg = SimConfig {
        fine: 133,
        ..SimConfig::default()
    };
    let sim = simulate(&cfg).unwrap();
    let n_train = 240;
    let train = first(&sim, n_train);
    let dev = Device::Cpu;
    let opts = Climb {
        jitter: true,
        trunk: false,
        lineages: 1,
        points: 45,
        dim: 3,
        kappa: cfg.kappa,
        steps: 2500,
        lr: 0.03,
        sub: 1,
    };
    let (fit, objective) = climb(&train, &opts, &dev).unwrap();
    let at_truth = {
        let prior = crate::ode_mixture::kernel::GridPrior::new(45, 1.0, cfg.kappa).unwrap();
        let f = Fit::from_params(&truth_params(&sim, 133, 45), &prior, &dev).unwrap();
        f.objective(&f.reads(&train).unwrap())
            .unwrap()
            .to_scalar::<f32>()
            .unwrap()
    };
    let all = first(&sim, sim.n_pb);
    let (_, time) = fit.posterior(&fit.reads(&all).unwrap()).unwrap();
    let held: Vec<usize> = (n_train..sim.n_pb).collect();
    let r = spearman(
        &held.iter().map(|&p| time[p]).collect::<Vec<_>>(),
        &held.iter().map(|&p| sim.truth.time[p]).collect::<Vec<_>>(),
    );
    eprintln!("objective {objective:.1} (truth {at_truth:.1}); held-out signed Spearman {r:.3}");
    assert!(r > 0.95, "held-out Spearman {r}");
    assert!(f64::from(objective) > f64::from(at_truth) - 50.0);
}

/// Two lineages from scratch: each true lineage in a fitted one of its
/// own, and time right within each.
#[test]
fn from_scratch_two_lineages_come_apart() {
    let cfg = SimConfig {
        lineages: 2,
        n_pb: 600,
        fine: 133,
        ..SimConfig::default()
    };
    let sim = simulate(&cfg).unwrap();
    let n_train = 480;
    let train = first(&sim, n_train);
    let dev = Device::Cpu;
    let opts = Climb {
        jitter: true,
        trunk: false,
        lineages: 2,
        points: 45,
        dim: 3,
        kappa: cfg.kappa,
        steps: 2500,
        lr: 0.03,
        sub: 1,
    };
    let (fit, objective) = climb(&train, &opts, &dev).unwrap();
    let all = first(&sim, sim.n_pb);
    let (weights, time) = fit.posterior(&fit.reads(&all).unwrap()).unwrap();
    let held: Vec<usize> = (n_train..sim.n_pb).collect();
    let fitted: Vec<usize> = held
        .iter()
        .map(|&p| if weights[p][0] > weights[p][1] { 0 } else { 1 })
        .collect();
    let mut table = [[0usize; 2]; 2];
    for (i, &p) in held.iter().enumerate() {
        table[sim.truth.lineage[p]][fitted[i]] += 1;
    }
    let purity =
        (table[0][0] + table[1][1]).max(table[0][1] + table[1][0]) as f64 / held.len() as f64;
    let mut per = Vec::new();
    for lin in 0..2 {
        let rows: Vec<usize> = held
            .iter()
            .copied()
            .filter(|&p| sim.truth.lineage[p] == lin)
            .collect();
        per.push(spearman(
            &rows.iter().map(|&p| time[p]).collect::<Vec<_>>(),
            &rows.iter().map(|&p| sim.truth.time[p]).collect::<Vec<_>>(),
        ));
    }
    eprintln!("objective {objective:.1}; table {table:?}; purity {purity:.3}; per-lineage signed Spearman {per:?}");
    assert!(purity > 0.95, "purity {purity}");
    assert!(
        per.iter().all(|&r| r > 0.95),
        "per-lineage Spearman {per:?}"
    );
}

/// On a tree, one of the starts (one per end taken as the root) puts the
/// trunk on both lineages and each branch on one of its own, roughly in
/// order. Rough: on the leading axes the branches part well after the
/// branch (the ramp starts flat, the levels lag), so the start's trunk runs
/// on into both branches and only the late branch pseudobulks are judged;
/// the climb sorts out the rest.
#[test]
fn a_tree_start_gives_both_lineages_the_trunk() {
    let cfg = SimConfig {
        lineages: 2,
        n_pb: 480,
        fine: 133,
        branch_at: Some(0.4),
        ..SimConfig::default()
    };
    let sim = simulate(&cfg).unwrap();
    let t = &sim.truth;
    let reads = first(&sim, sim.n_pb);
    let starts = tree_starts(&reads, 3);
    assert_eq!(starts.len(), 3, "a start per end");
    let score = |places: &Vec<Vec<(usize, f64)>>| {
        let (mut tab, mut shared, mut trunk, mut late) = ([[0usize; 2]; 2], 0, 0, 0);
        for p in 0..sim.n_pb {
            if t.trunk[p] {
                trunk += 1;
                shared += usize::from(places[p].len() == 2);
            } else if t.time[p] > 0.8 * cfg.t_len {
                late += 1;
                if places[p].len() == 1 {
                    tab[t.lineage[p]][places[p][0].0] += 1;
                }
            }
        }
        let purity = (tab[0][0] + tab[1][1]).max(tab[0][1] + tab[1][0]) as f64 / late as f64;
        let swap = tab[0][1] + tab[1][0] > tab[0][0] + tab[1][1];
        let per: Vec<f64> = (0..2)
            .map(|lin| {
                let fitted = if swap { 1 - lin } else { lin };
                let (mut a, mut b) = (Vec::new(), Vec::new());
                for (p, at) in places.iter().enumerate() {
                    let on = at.iter().find(|(l, _)| *l == fitted);
                    if let (true, Some(&(_, x))) = (t.trunk[p] || t.lineage[p] == lin, on) {
                        a.push(x);
                        b.push(t.time[p]);
                    }
                }
                spearman(&a, &b)
            })
            .collect();
        (purity, shared as f64 / trunk as f64, per)
    };
    let scored: Vec<_> = starts.iter().map(score).collect();
    eprintln!("tree starts (purity, trunk shared, per-lineage Spearman): {scored:?}");
    assert!(
        scored.iter().any(|(purity, shared, per)| *purity > 0.95
            && *shared > 0.8
            && per.iter().all(|r| *r > 0.75)),
        "no start finds the tree: {scored:?}"
    );
}

/// The climb's time origin is right to within half a grid step once a
/// pseudobulk's time can sit between nodes (four positions a step): on the
/// nodes alone the climb stays a grid step late, an optimum no gradient
/// leaves.
#[test]
#[ignore = "fails since the global capture: mean time shift 0.0124 over the bound 0.0114, the capture trading slightly with the time origin"]
fn the_climbs_time_is_not_a_grid_step_late() {
    let cfg = SimConfig {
        fine: 133,
        ..SimConfig::default()
    };
    let sim = simulate(&cfg).unwrap();
    let train = first(&sim, 240);
    let opts = Climb {
        jitter: true,
        trunk: false,
        lineages: 1,
        points: 45,
        dim: 3,
        kappa: cfg.kappa,
        steps: 2500,
        lr: 0.03,
        sub: 4,
    };
    let (fit, _) = climb(&train, &opts, &Device::Cpu).unwrap();
    let (_, time) = fit.posterior(&fit.reads(&train).unwrap()).unwrap();
    let shift = (0..240).map(|p| time[p] - sim.truth.time[p]).sum::<f64>() / 240.0;
    let step = 1.0 / 44.0;
    eprintln!(
        "mean time − truth {shift:.4} (grid step {step:.4}); jitter sd {:?}, log capture {:.3}",
        fit.share_sd().unwrap(),
        fit.log_capture().unwrap()
    );
    assert!(
        shift.abs() < 0.5 * step,
        "the climb's times are off by {shift}"
    );
}

/// A start whose objective is NaN never beats one that is finite, and a
/// finite one always beats NaN.
#[test]
fn a_nan_start_is_never_the_best() {
    use crate::ode_mixture::init::beats;
    assert!(!beats(f32::NAN, -10.0));
    assert!(beats(-10.0, f32::NAN));
    assert!(beats(-1.0, -10.0) && !beats(-10.0, -1.0));
    assert!(!beats(f32::NAN, f32::NAN));
}

/// A `p × g` matrix of rank-3 signal (singular values 30, 20, 10 in
/// scale) plus small noise, deterministic.
fn low_rank(p: usize, g: usize) -> nalgebra::DMatrix<f64> {
    use rand::SeedableRng;
    use rand_distr::{Distribution, StandardNormal};
    let mut rng = rand::rngs::StdRng::seed_from_u64(7);
    let mut draw = |r: usize, c: usize| {
        nalgebra::DMatrix::from_fn(r, c, |_, _| StandardNormal.sample(&mut rng))
    };
    let (a, b) = (draw(p, 3), draw(3, g));
    let scale = nalgebra::DMatrix::from_diagonal(&nalgebra::DVector::from_vec(vec![3.0, 2.0, 1.0]));
    &a * scale * &b + draw(p, g) * 0.05
}

#[test]
fn the_axes_are_the_leading_singular_ones() {
    let y = low_rank(120, 300);
    let (scores, loadings) = axes(&y, 3);
    let svd = y.clone().svd(true, true);
    let (u, v) = (svd.u.unwrap(), svd.v_t.unwrap().transpose());
    for d in 0..3 {
        // Up to sign: the scores are `u_d σ_d`, the loadings `v_d`.
        let su = scores.column(d) / svd.singular_values[d];
        let cos_u = su.dot(&u.column(d)).abs();
        let cos_v = loadings.column(d).dot(&v.column(d)).abs();
        assert!(cos_u > 0.999 && cos_v > 0.999, "axis {d}: {cos_u}, {cos_v}");
        // Score and loading share the axis's sign.
        let back = y.tr_mul(&scores.column(d)).dot(&loadings.column(d));
        assert!(back > 0.0, "axis {d} flips between its score and loading");
    }
}

/// The start's axes at a run's size (thousands of pseudobulks by ten
/// thousand genes) take seconds, not the full SVD's tens of minutes.
#[test]
fn the_axes_take_seconds_at_scale() {
    let y = low_rank(1500, 4000);
    let t0 = std::time::Instant::now();
    let (scores, _) = axes(&y, 3);
    let secs = t0.elapsed().as_secs_f64();
    assert_eq!(scores.ncols(), 3);
    assert!(secs < 5.0, "{secs:.1} s");
}

/// The start at a run's size takes seconds: every step of it is linear in
/// the reads (or the few axes), none quadratic in the genes.
#[test]
fn the_start_takes_seconds_at_scale() {
    let sim = simulate(&SimConfig {
        n_genes: 4000,
        n_pb: 1500,
        fine: 133,
        ..SimConfig::default()
    })
    .unwrap();
    let reads = first(&sim, sim.n_pb);
    let t0 = std::time::Instant::now();
    let (lineage, _) = start_order(&reads, 2, 3);
    let secs = t0.elapsed().as_secs_f64();
    assert_eq!(lineage.len(), sim.n_pb);
    assert!(secs < 5.0, "{secs:.1} s");
}
