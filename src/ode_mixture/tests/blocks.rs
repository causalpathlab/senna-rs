use super::model::truth_params;
use crate::ode_mixture::blocks::{
    capture_step, draw_latents, draw_share_sd, draw_weights, from_theta, gene_mode, gene_step,
    share_step, to_theta, GenePrior,
};
use crate::ode_mixture::likelihood::{GeneRows, Reads, RowLayout};
use crate::ode_mixture::sim::{simulate, SimConfig};
use nalgebra::DMatrix;
use rand::rngs::StdRng;
use rand::SeedableRng;

#[test]
fn latents_are_drawn_in_proportion_to_their_posterior() {
    // Two pseudobulks over two lineages of three points.
    let tab = DMatrix::from_row_slice(
        2,
        6,
        &[0.0, 1.0, 2.0, 0.0, 0.0, 0.0, 5.0, 0.0, 0.0, 0.0, 0.0, 9.0],
    );
    let log_w = [0.5f64.ln(), 0.5f64.ln()];
    let mut count = [[0usize; 6]; 2];
    let draws = 60_000;
    for sweep in 0..draws {
        let rows = draw_latents(&tab, &log_w, 3, 11, sweep as u64);
        for (p, &r) in rows.iter().enumerate() {
            count[p][r] += 1;
        }
    }
    for p in 0..2 {
        let w: Vec<f64> = (0..6).map(|r| tab[(p, r)].exp()).collect();
        let z: f64 = w.iter().sum();
        for r in 0..6 {
            let got = count[p][r] as f64 / draws as f64;
            assert!(
                (got - w[r] / z).abs() < 0.01,
                "p {p} r {r}: {got} vs {}",
                w[r] / z
            );
        }
    }
}

#[test]
fn the_weights_follow_the_lineages_counts() {
    let rows: Vec<usize> = (0..1000).map(|p| if p < 750 { 1 } else { 4 }).collect();
    let mut mean = [0.0; 2];
    let n = 4000;
    for sweep in 0..n {
        let w = draw_weights(&rows, 2, 3, 1.0, 5, sweep);
        assert!((w.iter().map(|x| x.exp()).sum::<f64>() - 1.0).abs() < 1e-9);
        mean[0] += w[0].exp() / n as f64;
        mean[1] += w[1].exp() / n as f64;
    }
    assert!((mean[0] - 751.0 / 1002.0).abs() < 0.005, "{mean:?}");
}

/// From a wrong start, every gene's rates come out as the data allow: the
/// ratio `γ/β` (the steady state) close, and each rate within its own
/// posterior spread (a gene near steady state pins the ratio, not the speed).
#[test]
fn a_genes_rates_are_found_from_a_wrong_start() {
    let cfg = SimConfig {
        fine: 133,
        ..SimConfig::default()
    };
    let sim = simulate(&cfg).unwrap();
    let truth = truth_params(&sim, 133, 45);
    let reads = Reads::new(&sim.u, &sim.s, sim.n_pb, sim.n_genes);
    let lv = truth.levels();
    let step = cfg.t_len / 44.0;
    let rows: Vec<usize> = (0..sim.n_pb)
        .map(|p| (sim.truth.time[p] / step).round() as usize)
        .collect();
    let lambda = reads.depths(&rows, &lv);
    let prior = GenePrior::bounded(&truth);
    let layout = RowLayout::new(&rows, &lambda);
    let z_bar = truth.path_mean();
    for g in [0, 3, 11, 25, 40, 77] {
        // From the prior mean, bounded Newton finds the conditional's mode;
        // four chains start there, the proposal built on it once.
        let by_row = GeneRows::new(&layout, &reads, g);
        let prop = gene_mode(
            &truth,
            &by_row,
            &z_bar,
            &prior,
            &prior.mean(truth.rho[g].len()),
        );
        let (mut b, mut y, mut r) = (Vec::new(), Vec::new(), Vec::new());
        let (mut ess_r, mut ess_b) = (0.0, 0.0);
        for chain in 0..4 {
            let mut rng = StdRng::seed_from_u64(9 + 1000 * chain + g as u64);
            let (mut theta, mut ll) = (prop.mode.clone(), f64::NEG_INFINITY);
            let (mut kept, mut kept_b) = (Vec::new(), Vec::new());
            for it in 0..4000 {
                (theta, ll) =
                    gene_step(&truth, &theta, ll, &by_row, &z_bar, &prior, &prop, &mut rng);
                if it >= 300 {
                    let (beta, gamma, _, _) = from_theta(&theta, &z_bar);
                    kept.push((gamma / beta).ln());
                    kept_b.push(beta.ln());
                    b.push(beta.ln());
                    y.push(gamma.ln());
                    r.push((gamma / beta).ln());
                }
            }
            let f32s = |v: &[f64]| v.iter().map(|&x| x as f32).collect::<Vec<_>>();
            ess_r += legume_numeric::mcmc::engine::diagnostics::ess(&f32s(&kept));
            ess_b += legume_numeric::mcmc::engine::diagnostics::ess(&f32s(&kept_b));
        }
        let z = |v: &[f64], truth: f64| {
            let m = v.iter().sum::<f64>() / v.len() as f64;
            let sd = (v.iter().map(|x| (x - m).powi(2)).sum::<f64>() / v.len() as f64).sqrt();
            ((m - truth) / sd.max(1e-3), m - truth)
        };
        // The identified coordinate mixes: `log γ/β`, the steady share. The
        // rates' common scale runs along a ridge the reads barely pin (weak
        // identifiability, reported per gene by R̂): it mixes slowly, and is
        // gated only against a sampler that stops moving it.
        eprintln!(
            "gene {g}: effective draws of log γ/β {ess_r:.0}, of log β {ess_b:.0}, of {} over four chains",
            r.len()
        );
        assert!(
            ess_r > 400.0,
            "gene {g}: {ess_r:.0} effective draws of log γ/β"
        );
        assert!(
            ess_b > 100.0,
            "gene {g}: {ess_b:.0} effective draws of log β"
        );
        let (zb, _) = z(&b, truth.beta[g].ln());
        let (zy, _) = z(&y, truth.gamma[g].ln());
        let (zr, er) = z(&r, (truth.gamma[g] / truth.beta[g]).ln());
        eprintln!("gene {g}: z log β {zb:.2}, z log γ {zy:.2}, log γ/β error {er:.3} (z {zr:.2})");
        assert!(zr.abs() < 3.0, "gene {g}: log γ/β off by {er} (z {zr})");
        // A well-read gene pins its steady share tightly.
        let reads_g: f64 = (0..sim.n_pb)
            .map(|p| reads.u[(p, g)] + reads.s[(p, g)])
            .sum();
        if reads_g > 20_000.0 {
            assert!(
                er.abs() < 0.05,
                "gene {g} ({reads_g} reads): log γ/β off by {er}"
            );
        }
        assert!(zb.abs() < 3.0 && zy.abs() < 3.0, "gene {g}: z {zb} {zy}");
    }
}

#[test]
fn the_samplers_coordinates_round_trip() {
    let z_bar = [0.3, -2.0];
    let t = to_theta(3.0, 0.7, &[0.1, -0.4], 1.2, &z_bar);
    let (b, y, r, base) = from_theta(&t, &z_bar);
    assert!((b - 3.0).abs() < 1e-12 && (y - 0.7).abs() < 1e-12 && (base - 1.2).abs() < 1e-12);
    assert_eq!(r, &[0.1, -0.4]);
}

/// From the posterior mode (Adam from the truth), the path block alone,
/// genes, latents and depths held: it mixes. The move is the generalized
/// elliptical slice on the banded Laplace approximation
/// (`path_mode`, `path_gess_step`); `PLAIN=1` runs plain ESS on the
/// prior instead, for comparison. Followed are the invariants a
/// path sets, log transcription `ρ_gᵀz(t_k) + b_g` of a few genes at a few
/// times, and the reads' log-likelihood; their effective draws are reported
/// and must not be few. (A start from a flat path, as this test once
/// tried, is the climb's job, not the sampler's.)
#[test]
fn a_lineages_path_mixes_from_the_mode() {
    use crate::ode_mixture::blocks::{path_gess_step, path_ll, path_mode, path_step};
    use crate::ode_mixture::fit::Fit;
    use crate::ode_mixture::kernel::GridPrior;
    use crate::ode_mixture::likelihood::table;
    use legume_numeric::candle::candle_core::Device;
    use legume_numeric::mcmc::engine::diagnostics::ess;
    let cfg = SimConfig {
        fine: 133,
        ..SimConfig::default()
    };
    let sim = simulate(&cfg).unwrap();
    let reads = Reads::new(&sim.u, &sim.s, sim.n_pb, sim.n_genes);
    let grid = GridPrior::new(45, 1.0, cfg.kappa).unwrap();
    let fit = Fit::from_params(&truth_params(&sim, 133, 45), &grid, &Device::Cpu).unwrap();
    fit.adam(&reads, 2000, 0.01).unwrap();
    let mut params = fit.params().unwrap();
    // Every pseudobulk at its most likely grid point at the mode.
    let lv = params.levels();
    let tab = table(&reads, &lv);
    let rows: Vec<usize> = (0..sim.n_pb)
        .map(|p| {
            (0..tab.ncols())
                .max_by(|&a, &b| tab[(p, a)].total_cmp(&tab[(p, b)]))
                .unwrap()
        })
        .collect();
    let lambda = reads.depths(&rows, &lv);
    let h = params.rho[0].len();
    let mut eta: Vec<nalgebra::DVector<f64>> = (0..h)
        .map(|d| grid.whiten(&nalgebra::DVector::from_fn(45, |k, _| params.path[0][k][d])))
        .collect();
    let followed: Vec<(usize, usize)> = [0, 3, 11]
        .iter()
        .flat_map(|&g| [2usize, 22, 42].map(|k| (g, k)))
        .collect();
    let log_alpha = |p: &crate::ode_mixture::model::Params, (g, k): (usize, usize)| {
        p.rho[g]
            .iter()
            .zip(&p.path[0][k])
            .map(|(a, b)| a * b)
            .sum::<f64>()
            + p.base[g]
    };
    let prior = GenePrior::default();
    let mut rng = StdRng::seed_from_u64(4);
    let mut ll = f64::NEG_INFINITY;
    let moves = 1500;
    let mut traces = vec![Vec::with_capacity(moves); followed.len() + 1];
    let t0 = std::time::Instant::now();
    let plain = std::env::var("PLAIN").is_ok();
    let prop = path_mode(&params, 0, &eta, &grid, &reads, &rows, &lambda, &prior);
    for _ in 0..moves {
        (eta, ll) = if plain {
            path_step(
                &mut params,
                0,
                &eta,
                ll,
                &grid,
                &reads,
                &rows,
                &lambda,
                &prior,
                &mut rng,
            )
        } else {
            path_gess_step(
                &mut params,
                0,
                &eta,
                ll,
                &grid,
                &reads,
                &rows,
                &lambda,
                &prior,
                &prop,
                &mut rng,
            )
        };
        for (i, &at) in followed.iter().enumerate() {
            traces[i].push(log_alpha(&params, at) as f32);
        }
        traces[followed.len()].push(path_ll(&params, 0, &reads, &rows, &lambda) as f32);
    }
    let eff: Vec<f32> = traces.iter().map(|t| ess(t).round()).collect();
    eprintln!(
        "{moves} path moves in {:.1} s; effective draws: log α at (gene, point) {followed:?}: {:?}; reads' log-likelihood {}",
        t0.elapsed().as_secs_f64(),
        &eff[..followed.len()],
        eff[followed.len()]
    );
    let worst = eff.iter().cloned().fold(f32::INFINITY, f32::min);
    assert!(worst > 100.0, "{worst} effective draws of {moves}");
}

/// The path's conditional carries the genes' prior: `b̃ = b + ρᵀz̄ − log γ`
/// moves with the path mean `z̄`. With no reads, one gene and one
/// dimension, `z̄ = cᵀη` (`c = Cᵀ1/n`) and the conditional of `η` is normal:
///
/// ```text
/// E z̄ = −ρ d q / (s² + ρ² q),   var z̄ = q s² / (s² + ρ² q),   q = cᵀc
/// ```
///
/// `d = b − log γ − m`, `(m, s)` the prior on `b̃`.
#[test]
fn the_path_block_carries_the_genes_prior_through_the_path_mean() {
    use crate::ode_mixture::blocks::{path_gess_step, path_mode, path_step};
    use crate::ode_mixture::kernel::GridPrior;
    use crate::ode_mixture::model::Params;
    let n = 5;
    let grid = GridPrior::new(n, 1.0, 0.25).unwrap();
    let gene_prior = GenePrior::default();
    let (rho, base) = (2.0, 3.0);
    let mut params = Params {
        beta: vec![1.0],
        gamma: vec![1.0],
        rho: vec![vec![rho]],
        base: vec![base],
        t_len: vec![1.0],
        path: vec![vec![vec![0.0]; n]],
        sub: 1,
        branch: None,
        log_capture: 0.0,
    };
    let reads = Reads::new(&[], &[], 0, 1);
    let c: Vec<f64> = (0..n)
        .map(|j| (0..n).map(|k| grid.chol()[(k, j)]).sum::<f64>() / n as f64)
        .collect();
    let q: f64 = c.iter().map(|x| x * x).sum();
    let (m, s) = gene_prior.log_spliced;
    let d = base - m;
    let want_mean = -rho * d * q / (s * s + rho * rho * q);
    let want_var = q * s * s / (s * s + rho * rho * q);

    // The plain move, then the preconditioned one, its proposal built on an
    // anchor away from the mode (the move is exact whatever it is built on).
    let anchor = vec![nalgebra::DVector::from_element(n, 0.7)];
    let prop = path_mode(&params, 0, &anchor, &grid, &reads, &[], &[], &gene_prior);
    for preconditioned in [false, true] {
        let mut eta = vec![nalgebra::DVector::<f64>::zeros(n)];
        let mut ll = f64::NEG_INFINITY;
        let mut rng = StdRng::seed_from_u64(3);
        let mut z_bar = Vec::new();
        for _ in 0..40_000 {
            (eta, ll) = if preconditioned {
                path_gess_step(
                    &mut params,
                    0,
                    &eta,
                    ll,
                    &grid,
                    &reads,
                    &[],
                    &[],
                    &gene_prior,
                    &prop,
                    &mut rng,
                )
            } else {
                path_step(
                    &mut params,
                    0,
                    &eta,
                    ll,
                    &grid,
                    &reads,
                    &[],
                    &[],
                    &gene_prior,
                    &mut rng,
                )
            };
            z_bar.push(params.path_mean()[0]);
        }
        let mean = z_bar.iter().sum::<f64>() / z_bar.len() as f64;
        let var = z_bar.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / z_bar.len() as f64;
        eprintln!(
            "preconditioned {preconditioned}: z̄ mean {mean:.3} (want {want_mean:.3}), \
             var {var:.3} (want {want_var:.3})"
        );
        assert!(
            (mean - want_mean).abs() < 0.03,
            "mean {mean} vs {want_mean}"
        );
        assert!(
            (var / want_var - 1.0).abs() < 0.1,
            "var {var} vs {want_var}"
        );
    }
}

/// With times between the nodes, the path block's likelihood is the genes'
/// summed: both read the levels at the same positions.
#[test]
fn with_time_between_nodes_the_path_likelihood_is_the_genes() {
    use crate::ode_mixture::blocks::path_ll;
    use crate::ode_mixture::likelihood::gene_log_likelihood;
    let sim = simulate(&SimConfig {
        lineages: 2,
        fine: 133,
        ..SimConfig::default()
    })
    .unwrap();
    let mut params = truth_params(&sim, 133, 23);
    params.sub = 4;
    let reads = Reads::new(&sim.u, &sim.s, sim.n_pb, sim.n_genes);
    let lv = params.levels();
    let positions = params.positions();
    let rows: Vec<usize> = (0..sim.n_pb).map(|p| (p * 37) % (2 * positions)).collect();
    let lambda = reads.depths(&rows, &lv);
    let blocks: f64 = (0..2)
        .map(|l| path_ll(&params, l, &reads, &rows, &lambda))
        .sum();
    let genes: f64 = (0..sim.n_genes)
        .map(|g| {
            let (u, s) = params.gene_levels(g);
            gene_log_likelihood(&reads, g, &rows, &lambda, (&u, &s))
        })
        .sum();
    assert!(
        (blocks - genes).abs() < 1e-8 * genes.abs(),
        "{blocks} vs {genes}"
    );
}

/// The jitter's conditional, `∝ e^{Δ(ω)} N(ω; 0, σ²)`, its mean and sd by
/// quadrature.
fn share_posterior(n_u: f64, n: f64, z_u: f64, z_s: f64, sd: f64) -> (f64, f64) {
    let (lo, hi, m) = (-8.0 * sd - 4.0, 8.0 * sd + 4.0, 200_001);
    let h = (hi - lo) / (m - 1) as f64;
    let f =
        |w: f64| n_u * w - n * ((w.exp() * z_u + z_s) / (z_u + z_s)).ln() - 0.5 * (w / sd).powi(2);
    let top = (0..m)
        .map(|i| f(lo + i as f64 * h))
        .fold(f64::MIN, f64::max);
    let (mut z, mut m1, mut m2) = (0.0, 0.0, 0.0);
    for i in 0..m {
        let w = lo + i as f64 * h;
        let d = (f(w) - top).exp();
        (z, m1, m2) = (z + d, m1 + d * w, m2 + d * w * w);
    }
    let mean = m1 / z;
    (mean, (m2 / z - mean * mean).sqrt())
}

/// [`share_step`] leaves the jitter's conditional invariant: the chain's
/// mean and sd are the conditional's, at a deep and a shallow pseudobulk.
#[test]
fn the_jitter_step_draws_its_conditional() {
    for (n_u, n, z_u, z_s, sd) in [(1200.0, 5000.0, 0.3, 0.7, 0.5), (3.0, 40.0, 0.3, 0.7, 0.5)] {
        let (want_m, want_s) = share_posterior(n_u, n, z_u, z_s, sd);
        let mut rng = StdRng::seed_from_u64(4);
        let (mut w, steps) = (0.0, 100_000);
        let (mut m1, mut m2) = (0.0, 0.0);
        for _ in 0..steps {
            w = share_step(n_u, n, z_u, z_s, w, sd, &mut rng);
            m1 += w / steps as f64;
            m2 += w * w / steps as f64;
        }
        let got_s = (m2 - m1 * m1).sqrt();
        assert!(
            (m1 - want_m).abs() < 0.03 * want_s.max(0.01) + 0.002
                && (got_s / want_s).ln().abs() < 0.03,
            "N {n}: mean {m1} vs {want_m}, sd {got_s} vs {want_s}"
        );
    }
}

/// Given the jitters, the sd's draws centre on their spread.
#[test]
fn the_jitter_sd_is_drawn_from_its_conditional() {
    let omega: Vec<f64> = (0..300).map(|p| 0.5 * (p as f64 * 1.3).sin()).collect();
    let spread = (omega.iter().map(|w| w * w).sum::<f64>() / omega.len() as f64).sqrt();
    let draws: Vec<f64> = (0..4000).map(|it| draw_share_sd(&omega, 9, it)).collect();
    let mean = draws.iter().sum::<f64>() / draws.len() as f64;
    assert!(
        (mean / spread).ln().abs() < 0.02,
        "mean {mean} vs spread {spread}"
    );
}

/// The capture's conditional,
/// `∝ Π_p e^{N_u c} (1 + e^{a_p + c})^{−N} N(c; CAPTURE_PRIOR)`, its mean
/// and sd by quadrature.
fn capture_posterior(n_u: &[f64], n: &[f64], a: &[f64]) -> (f64, f64) {
    use crate::ode_mixture::likelihood::softplus;
    let (pm, psd) = crate::ode_mixture::model::CAPTURE_PRIOR;
    let f = |c: f64| {
        (0..a.len())
            .map(|p| n_u[p] * c - n[p] * softplus(a[p] + c))
            .sum::<f64>()
            - 0.5 * ((c - pm) / psd).powi(2)
    };
    let (lo, hi, m) = (-10.0, 10.0, 400_001);
    let h = (hi - lo) / (m - 1) as f64;
    let top = (0..m)
        .map(|i| f(lo + i as f64 * h))
        .fold(f64::MIN, f64::max);
    let (mut z, mut m1, mut m2) = (0.0, 0.0, 0.0);
    for i in 0..m {
        let c = lo + i as f64 * h;
        let d = (f(c) - top).exp();
        (z, m1, m2) = (z + d, m1 + d * c, m2 + d * c * c);
    }
    let mean = m1 / z;
    (mean, (m2 / z - mean * mean).sqrt())
}

/// [`capture_step`] leaves the capture's conditional invariant: the
/// chain's mean and sd are the conditional's, with many reads and with few.
#[test]
fn the_capture_step_draws_its_conditional() {
    let a: Vec<f64> = (0..30)
        .map(|p| -0.5 + 0.8 * (p as f64 * 0.9).sin())
        .collect();
    for scale in [1.0, 0.002] {
        let n: Vec<f64> = (0..30)
            .map(|p| (2000.0 + 50.0 * p as f64) * scale)
            .collect();
        let n_u: Vec<f64> = n.iter().map(|x| (0.2 * x).round()).collect();
        let (want_m, want_s) = capture_posterior(&n_u, &n, &a);
        let mut rng = StdRng::seed_from_u64(4);
        let (mut c, steps) = (0.0, 100_000);
        let (mut m1, mut m2) = (0.0, 0.0);
        for _ in 0..steps {
            c = capture_step(&n_u, &n, &a, c, &mut rng);
            m1 += c / steps as f64;
            m2 += c * c / steps as f64;
        }
        let got_s = (m2 - m1 * m1).sqrt();
        assert!(
            (m1 - want_m).abs() < 0.03 * want_s && (got_s / want_s).ln().abs() < 0.03,
            "scale {scale}: mean {m1} vs {want_m}, sd {got_s} vs {want_s}"
        );
    }
}

/// The branching simulation's truth put on a shared trunk at the truth's
/// branch by Adam (the mode the sampler starts from), its reads, their rows
/// at the truth and depths.
fn trunk_mode() -> (
    crate::ode_mixture::model::Params,
    Reads,
    Vec<usize>,
    Vec<crate::ode_mixture::likelihood::Depth>,
) {
    use crate::ode_mixture::fit::Fit;
    use crate::ode_mixture::kernel::GridPrior;
    use legume_numeric::candle::candle_core::Device;
    let sim = simulate(&SimConfig {
        lineages: 2,
        fine: 133,
        branch_at: Some(0.4),
        ..SimConfig::default()
    })
    .unwrap();
    let reads = Reads::new(&sim.u, &sim.s, sim.n_pb, sim.n_genes);
    let grid = GridPrior::new(45, 1.0, 0.25).unwrap();
    let truth = truth_params(&sim, 133, 45);
    let fit = Fit::from_params(&truth, &grid, &Device::Cpu)
        .unwrap()
        .with_trunk(&truth, &grid, &[0.4])
        .unwrap();
    fit.adam(&reads, 1000, 0.01).unwrap();
    let params = fit.params().unwrap();
    let rows: Vec<usize> = (0..sim.n_pb)
        .map(|p| {
            let k = (sim.truth.time[p] * 44.0).round() as usize;
            sim.truth.lineage[p] * 45 + k
        })
        .collect();
    let lambda = reads.depths(&rows, &params.levels());
    (params, reads, rows, lambda)
}

/// On a trunk, a move of the trunk (lineage 0's block) moves every lineage
/// and keeps every departure; a move of a departure moves its lineage
/// alone and keeps it on the trunk before the branch.
#[test]
fn on_a_trunk_a_trunk_move_moves_every_lineage_and_a_departure_its_own() {
    use crate::ode_mixture::blocks::{path_eta, path_step};
    use crate::ode_mixture::kernel::GridPrior;
    let (params, reads, rows, lambda) = trunk_mode();
    let grid = GridPrior::new(45, 1.0, 0.25).unwrap();
    let prior = GenePrior::default();
    let tb = params.branch.as_ref().unwrap()[0];
    let gap = |p: &crate::ode_mixture::model::Params, k: usize| -> f64 {
        p.path[1][k]
            .iter()
            .zip(&p.path[0][k])
            .map(|(a, b)| (a - b).abs())
            .sum()
    };
    let mut rng = StdRng::seed_from_u64(3);
    let mut moved = params.clone();
    let eta0 = path_eta(&params, &grid, 0);
    path_step(
        &mut moved,
        0,
        &eta0,
        f64::NEG_INFINITY,
        &grid,
        &reads,
        &rows,
        &lambda,
        &prior,
        &mut rng,
    );
    let mut changed = 0f64;
    for k in 0..45 {
        changed = changed.max((moved.path[0][k][0] - params.path[0][k][0]).abs());
        for d in 0..3 {
            let (a, b) = (
                moved.path[1][k][d] - moved.path[0][k][d],
                params.path[1][k][d] - params.path[0][k][d],
            );
            assert!((a - b).abs() < 1e-9, "departure at {k} moved");
        }
    }
    assert!(changed > 1e-6, "the trunk did not move");
    let mut moved = params.clone();
    let eta1 = path_eta(&params, &grid, 1);
    path_step(
        &mut moved,
        1,
        &eta1,
        f64::NEG_INFINITY,
        &grid,
        &reads,
        &rows,
        &lambda,
        &prior,
        &mut rng,
    );
    assert_eq!(moved.path[0], params.path[0]);
    for k in (0..45).filter(|&k| k as f64 / 44.0 <= tb) {
        assert!(gap(&moved, k) < 1e-9, "off the trunk at {k}");
    }
    assert!((0..45).any(|k| (gap(&moved, k) - gap(&params, k)).abs() > 1e-6));
}

/// On a trunk the Laplace mode of the trunk's block and of a departure's is
/// a stationary point of the density they approximate: along any
/// direction it rises no more either way than its curvature allows.
#[test]
fn on_a_trunk_a_paths_laplace_mode_is_the_densitys() {
    use crate::ode_mixture::blocks::{path_eta, path_log_density, path_mode, place_path};
    use crate::ode_mixture::kernel::GridPrior;
    use nalgebra::DVector;
    let (params, reads, rows, lambda) = trunk_mode();
    let grid = GridPrior::new(45, 1.0, 0.25).unwrap();
    let prior = GenePrior::default();
    for block in [0, 1] {
        let anchor = path_eta(&params, &grid, block);
        let prop = path_mode(
            &params, block, &anchor, &grid, &reads, &rows, &lambda, &prior,
        );
        let mode = prop.mode(3);
        // At the mode's branch time too (a departure's block moves it).
        let mut at_mode = params.clone();
        if let Some(t) = prop.branch_time(3, 45) {
            at_mode.branch.as_mut().unwrap()[0] = t;
        }
        let f = |eta: &[DVector<f64>]| {
            let mut p = at_mode.clone();
            place_path(&mut p, &at_mode, &grid, block, eta);
            path_log_density(&p, block, &reads, &rows, &lambda, &prior)
                - 0.5 * eta.iter().map(|e| e.norm_squared()).sum::<f64>()
        };
        let mut rng = StdRng::seed_from_u64(7);
        for _ in 0..5 {
            let v: Vec<DVector<f64>> = (0..3)
                .map(|_| {
                    DVector::from_fn(45, |_, _| {
                        <rand_distr::StandardNormal as rand_distr::Distribution<f64>>::sample(
                            &rand_distr::StandardNormal,
                            &mut rng,
                        )
                    })
                })
                .collect();
            let eps = 1e-3;
            let at = |s: f64| -> Vec<DVector<f64>> {
                mode.iter().zip(&v).map(|(m, d)| m + d * s).collect()
            };
            let (up, mid, down) = (f(&at(eps)), f(&at(0.0)), f(&at(-eps)));
            let slope = (up - down) / (2.0 * eps);
            let curve = -(up - 2.0 * mid + down) / (eps * eps);
            // A Newton step along v from the mode would be slope/curve.
            assert!(
                (slope / curve).abs() < 0.02,
                "block {block}: slope {slope}, curvature {curve}"
            );
        }
    }
}

/// With no reads and no genes, a departure's block follows its own prior:
/// its whitened path `N(0, I)` and its branch time uniform.
#[test]
fn with_no_reads_a_departure_and_its_branch_time_follow_their_prior() {
    use crate::ode_mixture::blocks::{path_eta, path_gess_step, path_mode};
    use crate::ode_mixture::kernel::GridPrior;
    let (params, reads, _, _) = trunk_mode();
    let grid = GridPrior::new(45, 1.0, 0.25).unwrap();
    let prior = GenePrior::default();
    let mut p = params.clone();
    p.beta.clear();
    p.gamma.clear();
    p.rho.clear();
    p.base.clear();
    let mut eta = path_eta(&params, &grid, 1);
    let prop = path_mode(&p, 1, &eta, &grid, &reads, &[], &[], &prior);
    let mut rng = StdRng::seed_from_u64(5);
    let n = 20_000;
    let (mut m, mut v, mut e2) = (0.0, 0.0, 0.0);
    let mut ll = f64::NEG_INFINITY;
    for _ in 0..n {
        (eta, ll) = path_gess_step(
            &mut p,
            1,
            &eta,
            ll,
            &grid,
            &reads,
            &[],
            &[],
            &prior,
            &prop,
            &mut rng,
        );
        let t = p.branch.as_ref().unwrap()[0];
        m += t / n as f64;
        v += t * t / n as f64;
        e2 += eta[0][10] * eta[0][10] / n as f64;
    }
    let var = v - m * m;
    assert!(
        (m - 0.5).abs() < 0.03 && (var - 1.0 / 12.0).abs() < 0.008,
        "branch time: mean {m} var {var}"
    );
    assert!(
        (e2 - 1.0).abs() < 0.08,
        "a whitened coordinate's second moment {e2}"
    );
}

/// The trajectory's response to the transcription at every node, built
/// from a few trajectories (a pulse inside the grid is a shift of the
/// first one), is the definition, one trajectory a node, exactly.
#[test]
fn the_response_to_every_node_is_one_trajectory_a_node() {
    use crate::ode_mixture::blocks::response;
    use crate::ode_mixture::kinetics::fine_trajectory;
    for (beta, gamma, h, n, sub) in [
        (2.0, 1.0, 0.05, 23, 4),
        (1.0, 1.0, 0.1, 9, 1),
        (5.0, 0.3, 0.02, 45, 3),
    ] {
        let (a_u, a_s) = response(beta, gamma, h, n, sub);
        for j in 0..n {
            let mut e = vec![0.0; n];
            e[j] = 1.0;
            let (u, s) = fine_trajectory(&e, beta, gamma, h, sub);
            for k in 0..u.len() {
                assert_eq!(
                    (a_u[(k, j)], a_s[(k, j)]),
                    (u[k], s[k]),
                    "node {j}, position {k}"
                );
            }
        }
    }
}

/// A lineage's reads' gradient and banded Fisher information (its
/// responses cut to the band) in its path, as one
/// gene's contribution after another summed (the definition, written out
/// here), however they are accumulated.
#[test]
fn a_lineages_information_is_every_genes_summed() {
    use crate::ode_mixture::blocks::{lineage_information, response};
    use nalgebra::DVector;
    let sim = simulate(&SimConfig {
        lineages: 2,
        fine: 133,
        ..SimConfig::default()
    })
    .unwrap();
    let mut params = truth_params(&sim, 133, 23);
    params.sub = 4;
    let reads = Reads::new(&sim.u, &sim.s, sim.n_pb, sim.n_genes);
    let lv = params.levels();
    let positions = params.positions();
    let rows: Vec<usize> = (0..sim.n_pb).map(|p| (p * 13) % (2 * positions)).collect();
    let lambda = reads.depths(&rows, &lv);
    let l = 1;
    let (n, h) = (params.path[l].len(), params.rho[0].len());
    let step = params.t_len[l] / (n - 1) as f64;
    let on: Vec<(usize, usize)> = rows
        .iter()
        .enumerate()
        .filter(|&(_, &r)| r / positions == l)
        .map(|(p, &r)| (p, r - l * positions))
        .collect();
    let (mut want_g, mut want_f) = (DMatrix::zeros(n, h), DMatrix::zeros(h * n, h * n));
    for g in 0..sim.n_genes {
        let rho = &params.rho[g];
        let alpha: Vec<f64> = params.path[l]
            .iter()
            .map(|z| (z.iter().zip(rho).map(|(a, b)| a * b).sum::<f64>() + params.base[g]).exp())
            .collect();
        let (a_u, a_s) = response(params.beta[g], params.gamma[g], step, n, params.sub);
        let av = DVector::from_column_slice(&alpha);
        let (u, s) = (&a_u * &av, &a_s * &av);
        let (mut eu, mut es) = (DVector::zeros(positions), DVector::zeros(positions));
        let (mut wu, mut ws) = (DVector::zeros(positions), DVector::zeros(positions));
        for &(p, k) in &on {
            eu[k] += reads.u[(p, g)] / u[k] - lambda[p][0];
            es[k] += reads.s[(p, g)] / s[k] - lambda[p][1];
            wu[k] += lambda[p][0] / u[k];
            ws[k] += lambda[p][1] / s[k];
        }
        let d_alpha = a_u.transpose() * eu + a_s.transpose() * es;
        // The information's responses cut to the nodes from `BAND − 1`
        // before a position's to the one after.
        let band = crate::ode_mixture::kernel::BAND;
        let cut = |a: &DMatrix<f64>| {
            DMatrix::from_fn(a.nrows(), n, |k, j| {
                let node = k / params.sub;
                if j + band > node && j <= node + 1 {
                    a[(k, j)]
                } else {
                    0.0
                }
            })
        };
        let (c_u, c_s) = (cut(&a_u), cut(&a_s));
        let fa = c_u.transpose() * DMatrix::from_diagonal(&wu) * &c_u
            + c_s.transpose() * DMatrix::from_diagonal(&ws) * &c_s;
        for j in 0..n {
            for d in 0..h {
                want_g[(j, d)] += alpha[j] * rho[d] * d_alpha[j];
            }
        }
        for a in 0..h * n {
            for b in 0..h * n {
                let (d, i, e, j) = (a / n, a % n, b / n, b % n);
                want_f[(a, b)] += rho[d] * rho[e] * alpha[i] * alpha[j] * fa[(i, j)];
            }
        }
    }
    let (got_g, got_f) = lineage_information(&params, l, &reads, &rows, &lambda);
    let rel = |a: &DMatrix<f64>, b: &DMatrix<f64>| (a - b).abs().max() / b.abs().max();
    assert!(
        rel(&got_g, &want_g) < 1e-10,
        "gradient {}",
        rel(&got_g, &want_g)
    );
    // Node-major `k H + d`.
    let got_f = DMatrix::from_fn(h * n, h * n, |a, b| got_f.get(a, b));
    let want_f = DMatrix::from_fn(h * n, h * n, |a, b| {
        let (i, d, j, e) = (a / h, a % h, b / h, b % h);
        want_f[(d * n + i, e * n + j)]
    });
    assert!(
        rel(&got_f, &want_f) < 1e-10,
        "Fisher {}",
        rel(&got_f, &want_f)
    );
}

/// A gene whose rate is not a number leaves the reads' gradient non-finite: the
/// path's mode search stops at its last finite point instead of failing,
/// its mode finite.
#[test]
fn a_paths_mode_survives_a_gradient_that_is_not_finite() {
    use crate::ode_mixture::blocks::path_mode;
    use crate::ode_mixture::kernel::GridPrior;
    use crate::ode_mixture::likelihood::table;
    let cfg = SimConfig::default();
    let sim = simulate(&cfg).unwrap();
    let reads = Reads::new(&sim.u, &sim.s, sim.n_pb, sim.n_genes);
    let grid = GridPrior::new(45, 1.0, cfg.kappa).unwrap();
    let mut params = truth_params(&sim, 133, 45);
    let lv = params.levels();
    let tab = table(&reads, &lv);
    let rows: Vec<usize> = (0..sim.n_pb)
        .map(|p| {
            (0..tab.ncols())
                .max_by(|&a, &b| tab[(p, a)].total_cmp(&tab[(p, b)]))
                .unwrap()
        })
        .collect();
    let lambda = reads.depths(&rows, &lv);
    // A NaN rate (an infinite one saturates at the clamp, finite).
    params.base[0] = f64::NAN;
    let h = params.rho[0].len();
    let eta: Vec<nalgebra::DVector<f64>> = (0..h)
        .map(|d| grid.whiten(&nalgebra::DVector::from_fn(45, |k, _| params.path[0][k][d])))
        .collect();
    let prop = path_mode(
        &params,
        0,
        &eta,
        &grid,
        &reads,
        &rows,
        &lambda,
        &GenePrior::default(),
    );
    assert!(prop.mode(h).iter().flatten().all(|v| v.is_finite()));
}

/// An extreme baseline or embedding saturates the transcription at the
/// clamp ([`crate::ode_mixture::model::LOG_CLAMP`]) instead of
/// overflowing: the likelihood, the levels and the reads' gradient and
/// information stay finite.
#[test]
fn an_extreme_rate_gives_a_finite_likelihood_and_gradient() {
    use crate::ode_mixture::blocks::{lineage_information, path_ll};
    let sim = simulate(&SimConfig::default()).unwrap();
    let reads = Reads::new(&sim.u, &sim.s, sim.n_pb, sim.n_genes);
    let mut params = truth_params(&sim, 133, 45);
    params.base[0] = 800.0;
    params.base[1] = -800.0;
    params.rho[2] = vec![1e4; params.rho[2].len()];
    let lv = params.levels();
    assert!(lv.u.iter().chain(lv.s.iter()).all(|v| v.is_finite()));
    let rows: Vec<usize> = (0..sim.n_pb).map(|p| p % params.positions()).collect();
    let lambda = reads.depths(&rows, &lv);
    assert!(path_ll(&params, 0, &reads, &rows, &lambda).is_finite());
    let (grad, fisher) = lineage_information(&params, 0, &reads, &rows, &lambda);
    assert!(grad.iter().all(|v| v.is_finite()));
    let hn = 45 * params.rho[0].len();
    assert!((0..hn).all(|i| (0..hn).all(|j| fisher.get(i, j).is_finite())));
}

/// The rates' link carries the prior exactly: `link` and `unlink` are
/// inverse inside the bounds, every rate it gives is inside them, and the
/// log-Jacobian is the map's (by finite differences of `log r` in `u`).
#[test]
fn the_rate_link_round_trips_and_carries_its_jacobian() {
    use crate::ode_mixture::model::Params;
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

/// Gene `g`'s `(log γ, log β)` draws from four chains of [`gene_step`] on
/// `prior`, from its mode (rows at the truth's times).
fn rate_draws(
    params: &crate::ode_mixture::model::Params,
    reads: &Reads,
    times: &[f64],
    g: usize,
    prior: &GenePrior,
) -> (Vec<f64>, Vec<f64>) {
    let lv = params.levels();
    let step = params.t_len[0] / (params.positions() - 1) as f64;
    let rows: Vec<usize> = times.iter().map(|t| (t / step).round() as usize).collect();
    let lambda = reads.depths(&rows, &lv);
    let layout = RowLayout::new(&rows, &lambda);
    let z_bar = params.path_mean();
    let by_row = GeneRows::new(&layout, reads, g);
    let h = params.rho[g].len();
    let anchor = to_theta(
        params.beta[g],
        params.gamma[g],
        &params.rho[g],
        params.base[g],
        &z_bar,
    );
    let prop = gene_mode(params, &by_row, &z_bar, prior, &anchor);
    let (mut lg, mut lb) = (Vec::new(), Vec::new());
    for chain in 0..4 {
        let mut rng = StdRng::seed_from_u64(31 + chain);
        let (mut theta, mut ll) = (prop.mode.clone(), f64::NEG_INFINITY);
        for it in 0..2000 {
            (theta, ll) = gene_step(params, &theta, ll, &by_row, &z_bar, prior, &prop, &mut rng);
            if it >= 300 {
                lg.push(theta[0]);
                lb.push(theta[0] - theta[1]);
            }
        }
    }
    let _ = h;
    (lg, lb)
}

/// The simulation's reads with gene `g`'s replaced by their expectation
/// under `params` at the truth's times: no noise, so where a rate is
/// unidentified only the prior (and the walls) hold it.
fn expected_reads(
    sim: &crate::ode_mixture::sim::Sim,
    params: &crate::ode_mixture::model::Params,
    g: usize,
) -> Reads {
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

fn mean(v: &[f64]) -> f64 {
    v.iter().sum::<f64>() / v.len() as f64
}

/// A gene whose rates are far faster (or slower) than the grid resolves has
/// them on a flat ridge: with the bounds ([`GenePrior::bounded`]) its draws
/// stay inside them, exactly.
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
    for (name, log_gamma, past) in [("fast", hi + 3.0, 1.0), ("slow", lo - 3.0, -1.0)] {
        let ratio = params.gamma[g] / params.beta[g];
        params.gamma[g] = log_gamma.exp();
        params.beta[g] = params.gamma[g] / ratio;
        let reads = expected_reads(&sim, &params, g);
        let (free_g, _) = rate_draws(&params, &reads, &sim.truth.time, g, &GenePrior::default());
        let (lg, lb) = rate_draws(&params, &reads, &sim.truth.time, g, &bounded);
        let inside = |v: &[f64]| v.iter().all(|x| (lo..=hi).contains(x));
        let free_past = free_g
            .iter()
            .filter(|&&x| if past > 0.0 { x > hi } else { x < lo })
            .count() as f64
            / free_g.len() as f64;
        eprintln!(
            "{name} gene: mean log γ {:.3} unbounded ({:.3} of draws past the bound), {:.3} bounded (log β {:.3})",
            mean(&free_g),
            free_past,
            mean(&lg),
            mean(&lb)
        );
        assert!(
            inside(&lg) && inside(&lb),
            "{name}: a draw outside the bounds"
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
        let (fg, fb) = rate_draws(&params, &reads, &sim.truth.time, g, &GenePrior::default());
        let (wg, wb) = rate_draws(&params, &reads, &sim.truth.time, g, &bounded);
        let (dg, db) = (mean(&wg) - mean(&fg), mean(&wb) - mean(&fb));
        eprintln!(
            "gene {g}: log γ {:.3} ({dg:+.4} bounded), log β {:.3} ({db:+.4})",
            mean(&fg),
            mean(&fb)
        );
        assert!(dg.abs() < 0.05 && db.abs() < 0.05, "gene {g}: {dg} {db}");
    }
}

/// The bounds are fixed for a run: the climb starts every lineage at the
/// unit duration, the chains move the durations, and every stage keeps
/// the same bounds, so a gene the climb left at one is at the same one in
/// the chains.
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
