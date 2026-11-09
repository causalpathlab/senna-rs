//! A gene's conditional: its coordinates, the bounded prior and the
//! Laplace approximation at its mode.

use super::model::truth_params;
use crate::ode_mixture::blocks::{from_theta, gene_mode, to_theta, GenePrior};
use crate::ode_mixture::likelihood::{GeneRows, Reads, RowLayout};
use crate::ode_mixture::model::Params;
use crate::ode_mixture::sim::{simulate, Sim, SimConfig};

/// Gene `g`'s mode `(log γ, log γ/β)` and the Laplace sds of both, from
/// `anchor` (`None`: the prior's mean), the rows at `times`.
fn mode_of(
    params: &Params,
    reads: &Reads,
    times: &[f64],
    g: usize,
    prior: &GenePrior,
    anchor: Option<&[f64]>,
) -> ((f64, f64), (f64, f64)) {
    let lv = params.levels();
    let step = params.t_len[0] / (params.positions() - 1) as f64;
    let rows: Vec<usize> = times.iter().map(|t| (t / step).round() as usize).collect();
    let lambda = reads.depths(&rows, &lv);
    let layout = RowLayout::new(&rows, &lambda);
    let z_bar = params.path_mean();
    let by_row = GeneRows::new(&layout, reads, g);
    let own = to_theta(
        params.beta[g],
        params.gamma[g],
        &params.rho[g],
        params.base[g],
        &z_bar,
    );
    let prop = gene_mode(params, &by_row, &z_bar, prior, anchor.unwrap_or(&own));
    let (sd_gamma, sd_ratio) = prop.rate_sds(prior);
    ((prop.mode[0], prop.mode[1]), (sd_gamma, sd_ratio))
}

/// From a wrong start (the prior's mean), bounded Newton finds every
/// gene's conditional mode: the ratio `γ/β` (the steady state) close to
/// the truth, and each rate within its Laplace spread of it (a gene near
/// steady state pins the ratio, not the speed).
#[test]
fn a_genes_rates_are_found_from_a_wrong_start() {
    let cfg = SimConfig {
        fine: 133,
        ..SimConfig::default()
    };
    let sim = simulate(&cfg).unwrap();
    let truth = truth_params(&sim, 133, 45);
    let reads = Reads::new(&sim.u, &sim.s, sim.n_pb, sim.n_genes);
    let prior = GenePrior::bounded(&truth);
    for g in [0, 3, 11, 25, 40, 77] {
        let anchor = prior.mean(truth.rho[g].len());
        let ((lg, lr), (sg, sr)) =
            mode_of(&truth, &reads, &sim.truth.time, g, &prior, Some(&anchor));
        let (tg, tr) = (truth.gamma[g].ln(), (truth.gamma[g] / truth.beta[g]).ln());
        let (zg, zr) = ((lg - tg) / sg.max(1e-3), (lr - tr) / sr.max(1e-3));
        eprintln!(
            "gene {g}: log γ {lg:.3} ± {sg:.3} (truth {tg:.3}, z {zg:.2}); log γ/β {lr:.3} ± {sr:.3} (truth {tr:.3}, z {zr:.2})"
        );
        assert!(
            zr.abs() < 3.0,
            "gene {g}: log γ/β off by {} (z {zr})",
            lr - tr
        );
        assert!(
            zg.abs() < 3.0,
            "gene {g}: log γ off by {} (z {zg})",
            lg - tg
        );
        // A well-read gene pins its steady share tightly.
        let reads_g: f64 = (0..sim.n_pb)
            .map(|p| reads.u[(p, g)] + reads.s[(p, g)])
            .sum();
        if reads_g > 20_000.0 {
            assert!(
                (lr - tr).abs() < 0.05,
                "gene {g} ({reads_g} reads): log γ/β off by {}",
                lr - tr
            );
        }
    }
}

#[test]
fn the_genes_coordinates_round_trip() {
    let z_bar = [0.3, -2.0];
    let t = to_theta(3.0, 0.7, &[0.1, -0.4], 1.2, &z_bar);
    let (b, y, r, base) = from_theta(&t, &z_bar);
    assert!((b - 3.0).abs() < 1e-12 && (y - 0.7).abs() < 1e-12 && (base - 1.2).abs() < 1e-12);
    assert_eq!(r, &[0.1, -0.4]);
}

/// The rates' link carries the prior exactly: `link` and `unlink` are
/// inverse inside the bounds, every rate it gives is inside them, and the
/// log-Jacobian is the map's (by finite differences of `log r` in `u`).
#[test]
fn the_rate_link_round_trips_and_carries_its_jacobian() {
    let params = Params {
        beta: vec![1.0],
        gamma: vec![1.0],
        rho: vec![vec![0.0]],
        base: vec![0.0],
        t_len: vec![1.0],
        path: vec![vec![vec![0.0]; 45]],
        sub: 4,
        branch: None,
        log_capture: 0.0,
    };
    let prior = GenePrior::bounded(&params);
    let (lo, hi) = prior.log_rate.unwrap();
    eprintln!("bounds on log γ and log β at 45 points, 4 a step: [{lo:.3}, {hi:.3}]");
    for theta in [
        [0.3, -0.5, 0.2, 1.0],
        [hi - 0.1, 0.05, -1.0, 0.0],
        [lo + 0.2, 0.1, 0.0, 2.0],
    ] {
        let back = prior.unlink(&prior.link(&theta));
        assert!(
            theta.iter().zip(&back).all(|(a, b)| (a - b).abs() < 1e-9),
            "{theta:?} {back:?}"
        );
    }
    for u in [-30.0, -5.0, 0.0, 3.0, 30.0] {
        let t = prior.unlink(&[u, -u, 0.0, 0.0]);
        let (lg, lb) = (t[0], t[0] - t[1]);
        assert!(
            (lo..=hi).contains(&lg) && (lo..=hi).contains(&lb),
            "u {u}: {lg} {lb}"
        );
    }
    let eps = 1e-6;
    for u in [-3.0, -0.4, 0.0, 1.1, 4.0] {
        let lg = |u: f64| prior.unlink(&[u, 0.0, 0.0, 0.0])[0];
        let slope = (lg(u + eps) - lg(u - eps)) / (2.0 * eps);
        let at = |a: f64| prior.linked_log_density(&[a, 0.5, 0.0, 0.0]);
        let jac = at(u) - prior.log_density(&prior.unlink(&[u, 0.5, 0.0, 0.0]));
        let want = slope.ln() + {
            let lb = |v: f64| prior.unlink(&[0.0, v, 0.0, 0.0]);
            let lb = |v: f64| lb(v)[0] - lb(v)[1];
            ((lb(0.5 + eps) - lb(0.5 - eps)) / (2.0 * eps)).ln()
        };
        assert!((jac - want).abs() < 1e-6, "u {u}: {jac} vs {want}");
    }
}

/// The simulation's reads with gene `g`'s replaced by their expectation
/// under `params` at the truth's times: no noise, so where a rate is
/// unidentified only the prior (and the walls) hold it.
fn expected_reads(sim: &Sim, params: &Params, g: usize) -> Reads {
    let reads = Reads::new(&sim.u, &sim.s, sim.n_pb, sim.n_genes);
    let lv = params.levels();
    let step = params.t_len[0] / (params.positions() - 1) as f64;
    let rows: Vec<usize> = sim
        .truth
        .time
        .iter()
        .map(|t| (t / step).round() as usize)
        .collect();
    let lambda = reads.depths(&rows, &lv);
    let (mut u, mut s) = (reads.u.clone(), reads.s.clone());
    for p in 0..sim.n_pb {
        u[(p, g)] = lambda[p][0] * lv.u[(rows[p], g)];
        s[(p, g)] = lambda[p][1] * lv.s[(rows[p], g)];
    }
    Reads::from_matrices(u, s)
}

/// A gene whose rates are far faster (or slower) than the grid resolves has
/// them on a flat ridge: with the bounds ([`GenePrior::bounded`]) its mode
/// stays inside them, exactly.
#[test]
fn an_unresolved_rate_stays_inside_its_bounds() {
    let sim = simulate(&SimConfig {
        fine: 133,
        ..SimConfig::default()
    })
    .unwrap();
    let mut params = truth_params(&sim, 133, 45);
    let bounded = GenePrior::bounded(&params);
    let (lo, hi) = bounded.log_rate.unwrap();
    eprintln!("bounds on log γ and log β: [{lo:.3}, {hi:.3}]");
    let g = 0;
    for (name, log_gamma) in [("fast", hi + 3.0), ("slow", lo - 3.0)] {
        let ratio = params.gamma[g] / params.beta[g];
        params.gamma[g] = log_gamma.exp();
        params.beta[g] = params.gamma[g] / ratio;
        let reads = expected_reads(&sim, &params, g);
        let ((free_g, free_r), _) = mode_of(
            &params,
            &reads,
            &sim.truth.time,
            g,
            &GenePrior::default(),
            None,
        );
        let ((lg, lr), _) = mode_of(&params, &reads, &sim.truth.time, g, &bounded, None);
        let lb = lg - lr;
        eprintln!(
            "{name} gene: log γ {free_g:.3} unbounded (log β {:.3}), {lg:.3} bounded (log β {lb:.3})",
            free_g - free_r
        );
        assert!(
            (lo..=hi).contains(&lg) && (lo..=hi).contains(&lb),
            "{name}: the mode outside the bounds"
        );
    }
}

/// For a gene whose rates the reads pin, well inside the bounds, the bounds
/// change next to nothing.
#[test]
fn the_rate_bounds_leave_an_ordinary_gene_alone() {
    let sim = simulate(&SimConfig {
        fine: 133,
        ..SimConfig::default()
    })
    .unwrap();
    let params = truth_params(&sim, 133, 45);
    let reads = Reads::new(&sim.u, &sim.s, sim.n_pb, sim.n_genes);
    let bounded = GenePrior::bounded(&params);
    for g in [3, 11] {
        let ((fg, fr), _) = mode_of(
            &params,
            &reads,
            &sim.truth.time,
            g,
            &GenePrior::default(),
            None,
        );
        let ((wg, wr), _) = mode_of(&params, &reads, &sim.truth.time, g, &bounded, None);
        let (dg, db) = (wg - fg, (wg - wr) - (fg - fr));
        eprintln!(
            "gene {g}: log γ {fg:.3} ({dg:+.4} bounded), log β {:.3} ({db:+.4})",
            fg - fr
        );
        assert!(dg.abs() < 0.05 && db.abs() < 0.05, "gene {g}: {dg} {db}");
    }
}

/// The bounds are fixed for a run: the climb starts every lineage at the
/// unit duration and moves the others, and every stage keeps the same
/// bounds.
#[test]
fn the_rate_bounds_do_not_move_with_the_durations() {
    let sim = simulate(&SimConfig::default()).unwrap();
    let mut params = truth_params(&sim, 133, 45);
    params.t_len = vec![1.0; params.t_len.len().max(2)];
    params
        .path
        .resize(params.t_len.len(), params.path[0].clone());
    let at_start = GenePrior::bounded(&params).log_rate.unwrap();
    for t in [0.3, 3.0] {
        let mut moved = params.clone();
        moved.t_len[1] = t;
        assert_eq!(
            GenePrior::bounded(&moved).log_rate.unwrap(),
            at_start,
            "a duration of {t}"
        );
    }
}
