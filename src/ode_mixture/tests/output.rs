use super::model::truth_params;
use crate::ode_mixture::output::MapBasis;
use crate::ode_mixture::readout::Snapshot;
use crate::ode_mixture::sim::{simulate, Sim, SimConfig};
use nalgebra::DMatrix;
use rand::rngs::StdRng;
use rand::SeedableRng;
use rand_distr::{Distribution, Normal};

/// Rows of reads `[P × G]` with two directions of variation.
fn rows() -> DMatrix<f64> {
    DMatrix::from_fn(40, 6, |i, j| {
        let t = i as f64 / 39.0;
        let a = [5.0, 1.0, 3.0, 0.5, 2.0, 4.0][j];
        let b = [0.0, 2.0, -1.0, 1.0, 0.5, -2.0][j];
        50.0 * (a + b * t + 0.3 * (i % 3) as f64).exp()
    })
}

/// The basis places its own rows at scores of zero mean, the first
/// component the most spread.
#[test]
fn the_map_places_its_rows_centred() {
    let x = rows();
    let map = MapBasis::new(&x, 3).unwrap();
    assert_eq!(map.dim(), 3);
    let placed: Vec<Vec<f64>> = x
        .row_iter()
        .map(|r| map.place(&r.iter().copied().collect::<Vec<_>>()))
        .collect();
    let var = |d: usize| {
        let m = placed.iter().map(|p| p[d]).sum::<f64>() / placed.len() as f64;
        assert!(m.abs() < 1e-6, "component {d} mean {m}");
        placed.iter().map(|p| (p[d] - m).powi(2)).sum::<f64>()
    };
    assert!(var(0) >= var(1) && var(1) >= var(2));
}

/// A cell's velocity on the map is the rate its place moves at when every
/// gene's level changes at its relative rate `d log x_g/dt`.
#[test]
fn the_maps_velocity_is_the_places_rate_of_change() {
    let x = rows();
    let map = MapBasis::new(&x, 3).unwrap();
    let cell: Vec<f64> = x.row(17).iter().copied().collect();
    let dlog: [f64; 6] = [0.4, -0.2, 0.1, 0.9, -0.5, 0.0];
    let h = 1e-6;
    let ahead: Vec<f64> = cell
        .iter()
        .zip(&dlog)
        .map(|(c, v)| c * (v * h).exp())
        .collect();
    let (now, next) = (map.place(&cell), map.place(&ahead));
    let v = map.velocity(&cell, &dlog);
    for d in 0..3 {
        let fd = (next[d] - now[d]) / h;
        assert!(
            (v[d] - fd).abs() < 1e-4 * (1.0 + fd.abs()),
            "{d}: {} vs {fd}",
            v[d]
        );
    }
}

/// Every point's share of its `k` nearest (Euclidean) that carry its label.
fn knn_purity(points: &[Vec<f64>], label: &[usize], k: usize) -> f64 {
    let d2 = |a: &[f64], b: &[f64]| a.iter().zip(b).map(|(x, y)| (x - y).powi(2)).sum::<f64>();
    let mut same = 0usize;
    for (i, p) in points.iter().enumerate() {
        let mut near: Vec<(f64, usize)> = points
            .iter()
            .enumerate()
            .filter(|&(j, _)| j != i)
            .map(|(j, q)| (d2(p, q), j))
            .collect();
        near.sort_by(|a, b| a.0.total_cmp(&b.0));
        same += near[..k]
            .iter()
            .filter(|&&(_, j)| label[j] == label[i])
            .count();
    }
    same as f64 / (points.len() * k) as f64
}

/// The simulation's pseudobulks' reads, both tracks summed, `[P × G]`.
fn sim_reads(sim: &Sim) -> DMatrix<f64> {
    let g = sim.n_genes;
    DMatrix::from_fn(sim.n_pb, g, |i, j| sim.u[i * g + j] + sim.s[i * g + j])
}

/// The truth as one draw, on `grid` of the simulator's `fine` points, the
/// lineages equally weighted.
fn truth_snap(sim: &Sim, fine: usize, grid: usize) -> Snapshot {
    let l = sim.truth.t_len.len();
    Snapshot {
        params: truth_params(sim, fine, grid),
        log_w: vec![-(l as f64).ln(); l],
        share_sd: None,
        rows: Vec::new(),
        lambda: Vec::new(),
    }
}

/// Rows `rows` of `x` placed on `map`.
fn places(map: &MapBasis, x: &DMatrix<f64>, rows: &[usize]) -> Vec<Vec<f64>> {
    rows.iter()
        .map(|&i| map.place(&x.row(i).iter().copied().collect::<Vec<_>>()))
        .collect()
}

/// On the trajectories' own axes the lineages part past the branch, where
/// the reads' own leading components follow three programs, each on a
/// quarter of the genes, that no trajectory drives.
#[test]
fn the_trajectories_axes_part_the_lineages_a_nuisance_hides() {
    let cfg = SimConfig {
        lineages: 2,
        n_pb: 400,
        fine: 81,
        branch_at: Some(0.3),
        seed: 3,
        ..SimConfig::default()
    };
    let sim = simulate(&cfg).unwrap();
    let (p, g) = (sim.n_pb, sim.n_genes);
    let mut rng = StdRng::seed_from_u64(9);
    let z = Normal::new(0.0f64, 1.5).unwrap();
    let nuisance: Vec<[f64; 3]> = (0..p)
        .map(|_| std::array::from_fn(|_| z.sample(&mut rng).exp()))
        .collect();
    let clean = sim_reads(&sim);
    let x = DMatrix::from_fn(p, g, |i, j| match 4 * j / g {
        b @ 0..=2 => (clean[(i, j)] * nuisance[i][b]).round(),
        _ => clean[(i, j)],
    });
    let traj = MapBasis::of_trajectories(&[truth_snap(&sim, cfg.fine, 21)], 2).unwrap();
    assert_eq!(traj.dim(), 2);
    // Off the trunk only: on it the lineages are one.
    let off: Vec<usize> = (0..p).filter(|&i| !sim.truth.trunk[i]).collect();
    let label: Vec<usize> = off.iter().map(|&i| sim.truth.lineage[i]).collect();
    let purity = |x: &DMatrix<f64>, map: &MapBasis| knn_purity(&places(map, x, &off), &label, 10);

    let (reads, model) = (
        purity(&x, &MapBasis::new(&x, 2).unwrap()),
        purity(&x, &traj),
    );
    eprintln!("lineage purity off the trunk: reads' axes {reads:.3}, trajectories' {model:.3}");
    assert!(model > reads + 0.05, "{model} vs {reads}");
    // With no program besides the trajectories, no worse than the reads'.
    let (reads, model) = (
        purity(&clean, &MapBasis::new(&clean, 2).unwrap()),
        purity(&clean, &traj),
    );
    assert!(model > reads - 0.02, "{model} vs {reads}");
}

/// Two lineages apart, as the truth and as one draw of the map.
fn two_lineages() -> (Sim, Snapshot) {
    let sim = simulate(&SimConfig {
        lineages: 2,
        fine: 81,
        seed: 5,
        ..SimConfig::default()
    })
    .unwrap();
    let snap = truth_snap(&sim, 81, 21);
    (sim, snap)
}

/// The maps of draws `a` and of draws `b` place the same pseudobulks at the
/// same places, each component up to its sign.
fn assert_same_map(sim: &Sim, a: &[Snapshot], b: &[Snapshot]) {
    let x = sim_reads(sim);
    let some: Vec<usize> = (0..sim.n_pb).step_by(37).collect();
    let on = |snaps| places(&MapBasis::of_trajectories(snaps, 3).unwrap(), &x, &some);
    let (a, b) = (on(a), on(b));
    for d in 0..a[0].len() {
        let sign = a
            .iter()
            .zip(&b)
            .map(|(x, y)| x[d] * y[d])
            .sum::<f64>()
            .signum();
        for (x, y) in a.iter().zip(&b) {
            assert!(
                (x[d] - sign * y[d]).abs() < 1e-6,
                "component {d}: {} vs {}",
                x[d],
                y[d]
            );
        }
    }
}

/// The levels' scale is a gauge (the multinomial sees shares only): every
/// level a thousandth as large, the map is the same.
#[test]
fn the_trajectories_map_is_the_same_at_any_scale_of_the_levels() {
    let (sim, snap) = two_lineages();
    let mut small = snap.clone();
    for b in &mut small.params.base {
        *b += 1e-3f64.ln();
    }
    assert_same_map(&sim, &[snap], &[small]);
}

/// A lineage of no weight does not shape the map: it is the map of the
/// other lineage alone.
#[test]
fn a_lineage_of_no_weight_does_not_shape_the_map() {
    let (sim, mut lopsided) = two_lineages();
    let mut alone = lopsided.clone();
    lopsided.log_w = vec![0.0, -60.0];
    alone.params.path.truncate(1);
    alone.params.t_len.truncate(1);
    alone.log_w = vec![0.0];
    assert_same_map(&sim, &[lopsided], &[alone]);
}

/// A draw with a level not finite (a NaN rate; an infinite one saturates
/// at the clamp) is left out of the rows it spoils, not the run: the map is
/// the other draws'.
#[test]
fn a_draw_not_finite_is_left_out_of_the_map() {
    let (sim, snap) = two_lineages();
    let mut spoiled = snap.clone();
    spoiled.params.base[3] = f64::NAN;
    assert_same_map(&sim, &[snap.clone(), spoiled], &[snap]);
}
