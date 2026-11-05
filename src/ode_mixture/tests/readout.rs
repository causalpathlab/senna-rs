use super::init::{first, pearson, spearman};
use super::model::truth_params;
use crate::ode_mixture::likelihood::Reads;
use crate::ode_mixture::readout::{read_cells, CellReadout, Snapshot};
use crate::ode_mixture::sim::{simulate, Sim, SimConfig};

/// The simulator's grid, the model's and the positions a step.
const FINE: usize = 133;
const GRID: usize = 45;
const SUB: usize = 4;

/// Cells: shallow pseudobulks of one, on two lineages apart.
fn cells() -> Sim {
    simulate(&SimConfig {
        lineages: 2,
        n_pb: 600,
        depth: 800.0,
        fine: FINE,
        seed: 3,
        ..SimConfig::default()
    })
    .unwrap()
}

/// The truth as one draw: lineages equally weighted, `share_sd` the jitter.
fn truth_snap(sim: &Sim, share_sd: Option<f64>) -> Snapshot {
    let mut params = truth_params(sim, FINE, GRID);
    params.sub = SUB;
    let lineages = params.path.len();
    Snapshot {
        params,
        log_w: vec![-(lineages as f64).ln(); lineages],
        share_sd,
    }
}

/// The share of cells whose 90% interval covers their true time `t`,
/// widened by half a position (the truth sits between positions).
fn coverage(out: &CellReadout, t: &[f64]) -> f64 {
    let half = 0.5 / ((GRID - 1) * SUB) as f64;
    let covered = (0..t.len())
        .filter(|&c| out.time_lo[c] - half <= t[c] && t[c] <= out.time_hi[c] + half)
        .count();
    covered as f64 / t.len() as f64
}

/// At the truth, every cell's lineage is called, its time ranks with the
/// truth's and its 90% interval covers it about as often as it should.
#[test]
fn at_the_truth_cells_are_read_out() {
    let sim = cells();
    let out = read_cells(&first(&sim, sim.n_pb), &[truth_snap(&sim, None)]);
    let t = &sim.truth;
    let called = (0..sim.n_pb)
        .filter(|&c| {
            let p = &out.lineage[c];
            (0..p.len()).max_by(|&a, &b| p[a].total_cmp(&p[b])) == Some(t.lineage[c])
        })
        .count() as f64
        / sim.n_pb as f64;
    let rho = spearman(&out.time_mean, &t.time);
    let covered = coverage(&out, &t.time);
    eprintln!("lineage called {called:.3}, time Spearman {rho:.4}, 90% coverage {covered:.3}");
    assert!(called > 0.95, "lineage {called}");
    assert!(rho > 0.95, "Spearman {rho}");
    assert!((0.85..=0.97).contains(&covered), "coverage {covered}");
}

/// With a jitter of the unspliced share in the cells, the readout at the
/// truth integrates it out: the 90% intervals still cover.
#[test]
fn with_a_jitter_the_cells_intervals_still_cover() {
    let sim = simulate(&SimConfig {
        n_pb: 600,
        depth: 800.0,
        fine: FINE,
        share_jitter: 1.0,
        seed: 4,
        ..SimConfig::default()
    })
    .unwrap();
    let out = read_cells(&first(&sim, sim.n_pb), &[truth_snap(&sim, Some(1.0))]);
    let covered = coverage(&out, &sim.truth.time);
    eprintln!("with the jitter: 90% coverage {covered:.3}");
    assert!((0.85..=0.97).contains(&covered), "coverage {covered}");
}

/// At the truth, every cell's posterior mean velocities track the truth's at
/// its true time, on the simulator's own grid.
#[test]
fn at_the_truth_cells_velocities_track_the_truths() {
    let sim = cells();
    let out = read_cells(&first(&sim, sim.n_pb), &[truth_snap(&sim, None)]);
    let fine = truth_params(&sim, FINE, FINE);
    let (du, ds) = fine.velocities();
    let t = &sim.truth;
    for (name, got, truth) in [
        ("unspliced", &out.velocity_u, &du),
        ("spliced", &out.velocity_s, &ds),
    ] {
        let (mut a, mut b) = (Vec::new(), Vec::new());
        for c in 0..sim.n_pb {
            let row = t.lineage[c] * FINE + (t.time[c] * (FINE - 1) as f64).round() as usize;
            for g in 0..sim.n_genes {
                a.push(got[(c, g)]);
                b.push(truth[(row, g)]);
            }
        }
        let r = pearson(&a, &b);
        eprintln!("{name} velocity: Pearson with the truth's {r:.4}");
        assert!(r > 0.9, "{name}: Pearson {r}");
    }
}

/// At the truth, every cell's rate of change of its log level (both
/// tracks, `d log(ū + s̄)/dt`) tracks the truth's at the cell's place.
#[test]
fn at_the_truth_cells_log_velocities_track_the_truths() {
    let sim = cells();
    let out = read_cells(&first(&sim, sim.n_pb), &[truth_snap(&sim, None)]);
    let fine = truth_params(&sim, FINE, FINE);
    let lv = fine.levels();
    let (du, ds) = fine.velocities_of(&lv);
    let t = &sim.truth;
    let (mut a, mut b) = (Vec::new(), Vec::new());
    for c in 0..sim.n_pb {
        let row = t.lineage[c] * FINE + (t.time[c] * (FINE - 1) as f64).round() as usize;
        for g in 0..sim.n_genes {
            a.push(out.velocity_log[(c, g)]);
            b.push((du[(row, g)] + ds[(row, g)]) / (lv.u[(row, g)] + lv.s[(row, g)]));
        }
    }
    let r = pearson(&a, &b);
    eprintln!("log velocity: Pearson with the truth's {r:.4}");
    assert!(r > 0.9, "Pearson {r}");
}

/// The intervals are the mixture's own 5% and 95% points, to within a
/// position of the lineages' own grid: here with one draw, from every
/// cell's posterior over the rows, on lineages of the unit duration and of
/// a quarter of it.
#[test]
fn the_intervals_are_the_posteriors_quantiles() {
    intervals_are_quantiles(1.0);
    intervals_are_quantiles(0.25);
}

fn intervals_are_quantiles(t_len: f64) {
    use crate::ode_mixture::blocks::log_posterior_row;
    use crate::ode_mixture::likelihood::table;
    let sim = cells();
    let mut snap = truth_snap(&sim, None);
    snap.params.t_len = vec![t_len; 2];
    let reads = first(&sim, 50);
    let (params, log_w) = (snap.params.clone(), snap.log_w.clone());
    let lv = params.levels();
    let tab = table(&reads, &lv);
    let out = read_cells(&reads, &[snap]);
    let position = t_len / (lv.points - 1) as f64;
    assert!(
        (out.time_bin - position).abs() < 1e-12,
        "bin {}",
        out.time_bin
    );
    for c in 0..50 {
        let (lp, _) = log_posterior_row(&tab, &log_w, lv.points, c);
        let mut m: Vec<(f64, f64)> = lp
            .iter()
            .enumerate()
            .map(|(r, x)| (params.time_at(r / lv.points, r % lv.points), x.exp()))
            .collect();
        m.sort_by(|a, b| a.0.total_cmp(&b.0));
        let point = |q: f64| {
            let mut acc = 0.0;
            m.iter()
                .find(|x| {
                    acc += x.1;
                    acc >= q
                })
                .map_or(1.0, |x| x.0)
        };
        let (lo, hi) = (point(0.05), point(0.95));
        assert!(
            (out.time_lo[c] - lo).abs() <= position,
            "cell {c}: lo {} vs {lo}",
            out.time_lo[c]
        );
        assert!(
            (out.time_hi[c] - hi).abs() <= position,
            "cell {c}: hi {} vs {hi}",
            out.time_hi[c]
        );
    }
}

/// A cell no row can explain (its reads not finite) reads out as unknown,
/// NaN throughout, not as a confident call; the other cells as before.
#[test]
fn a_cell_no_row_explains_is_unknown() {
    let sim = cells();
    let g = sim.n_genes;
    let mut u = sim.u[..3 * g].to_vec();
    u[g] = f64::NAN;
    let reads = Reads::new(&u, &sim.s[..3 * g], 3, g);
    let out = read_cells(&reads, &[truth_snap(&sim, None)]);
    assert!(out.time_mean[1].is_nan() && out.time_lo[1].is_nan() && out.time_sd[1].is_nan());
    assert!(out.lineage[1].iter().all(|x| x.is_nan()));
    assert!(out.velocity_s.row(1).iter().all(|x| x.is_nan()));
    assert!(out.time_mean[0].is_finite() && out.time_mean[2].is_finite());
}

/// The readout's time at a larger size: 20k cells, 1000 genes, 4 draws.
#[test]
#[ignore = "benchmark"]
fn benchmark_the_readout() {
    let sim = simulate(&SimConfig {
        lineages: 2,
        n_genes: 1000,
        n_pb: 20_000,
        depth: 800.0,
        fine: FINE,
        seed: 5,
        ..SimConfig::default()
    })
    .unwrap();
    let reads = first(&sim, sim.n_pb);
    let snaps = vec![truth_snap(&sim, Some(0.1)); 4];
    let t0 = std::time::Instant::now();
    let out = read_cells(&reads, &snaps);
    eprintln!(
        "readout of {} cells × {} genes over {} draws: {:.2} s",
        sim.n_pb,
        sim.n_genes,
        snaps.len(),
        t0.elapsed().as_secs_f64()
    );
    assert!(out.time_mean.iter().all(|t| t.is_finite()));
}
