use super::model::truth_params;
use crate::ode_mixture::blocks::{
    draw_latents, draw_weights, from_theta, gene_laplace, gene_step, to_theta, GenePrior,
};
use crate::ode_mixture::likelihood::Reads;
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
    let prior = GenePrior::default();
    let mut rng = StdRng::seed_from_u64(9);
    for g in [0, 3, 11, 25, 40, 77] {
        let mut theta = prior.mean(truth.rho[g].len());
        let mut ll = f64::NEG_INFINITY;
        let (mut b, mut y, mut r) = (Vec::new(), Vec::new(), Vec::new());
        // From the prior mean, Newton finds the conditional's mode; the
        // chain starts there, the proposal built on it once.
        let prop = gene_laplace(&truth, g, &reads, &rows, &lambda, &prior, &theta);
        theta = prop.mode.clone();
        for it in 0..4000 {
            (theta, ll) = gene_step(
                &truth, g, &theta, ll, &reads, &rows, &lambda, &prior, &prop, &mut rng,
            );
            if it >= 300 {
                let (beta, gamma, _, _) = from_theta(&theta, &truth.path_mean());
                b.push(beta.ln());
                y.push(gamma.ln());
                r.push((gamma / beta).ln());
            }
        }
        let z = |v: &[f64], truth: f64| {
            let m = v.iter().sum::<f64>() / v.len() as f64;
            let sd = (v.iter().map(|x| (x - m).powi(2)).sum::<f64>() / v.len() as f64).sqrt();
            ((m - truth) / sd.max(1e-3), m - truth)
        };
        let ess_b = legume_numeric::mcmc::engine::diagnostics::ess(
            &b.iter().map(|&x| x as f32).collect::<Vec<_>>(),
        );
        eprintln!(
            "gene {g}: effective draws of log β {ess_b:.0} of {}",
            b.len()
        );
        assert!(ess_b > 100.0, "gene {g}: {ess_b:.0} effective draws");
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
#[ignore = "diagnostic"]
fn diagnostic_how_far_genes_move_off_steady_state() {
    let sim = simulate(&SimConfig {
        fine: 133,
        ..SimConfig::default()
    })
    .unwrap();
    let t = &sim.truth;
    for g in [0, 3, 11, 25] {
        let steady = t.gamma[g] / t.beta[g];
        let dev: Vec<f64> = (0..sim.n_pb)
            .map(|p| (t.u_mean[p * sim.n_genes + g] / t.s_mean[p * sim.n_genes + g] / steady).ln())
            .collect();
        let max = dev.iter().fold(0f64, |a, b| a.max(b.abs()));
        let reads: f64 = (0..sim.n_pb)
            .map(|p| sim.u[p * sim.n_genes + g] + sim.s[p * sim.n_genes + g])
            .sum();
        eprintln!(
            "gene {g}: γ {:.2} β {:.2} max |log ratio / steady| {max:.3}, reads {reads}",
            t.gamma[g], t.beta[g]
        );
    }
}

#[test]
#[ignore = "diagnostic"]
fn diagnostic_gene_chain_trace() {
    let cfg = SimConfig {
        fine: 133,
        ..SimConfig::default()
    };
    let sim = simulate(&cfg).unwrap();
    let truth = truth_params(&sim, 133, 45);
    let reads = Reads::new(&sim.u, &sim.s, sim.n_pb, sim.n_genes);
    let lv = truth.levels();
    let rows: Vec<usize> = (0..sim.n_pb)
        .map(|p| (sim.truth.time[p] / (cfg.t_len / 44.0)).round() as usize)
        .collect();
    let lambda = reads.depths(&rows, &lv);
    let prior = GenePrior::default();
    let g: usize = std::env::var("GENE").map_or(11, |v| v.parse().unwrap());
    let mut rng = StdRng::seed_from_u64(9);
    let mut theta = prior.mean(truth.rho[g].len());
    let mut ll = f64::NEG_INFINITY;
    let prop = gene_laplace(&truth, g, &reads, &rows, &lambda, &prior, &theta);
    eprintln!(
        "truth log β {:.3} log γ {:.3}",
        truth.beta[g].ln(),
        truth.gamma[g].ln()
    );
    let mut sum = [0.0; 2];
    for it in 1..=20_000 {
        (theta, ll) = gene_step(
            &truth, g, &theta, ll, &reads, &rows, &lambda, &prior, &prop, &mut rng,
        );
        let (beta, gamma, _, _) = from_theta(&theta, &truth.path_mean());
        sum[0] += beta.ln();
        sum[1] += gamma.ln();
        if it % 2000 == 0 {
            eprintln!(
                "{it}: block mean log β {:.3} log γ {:.3}; now ll {ll:.1}",
                sum[0] / 2000.0,
                sum[1] / 2000.0
            );
            sum = [0.0; 2];
        }
    }
    let at_truth = {
        let mut t = truth.gene_levels(g);
        let _ = &mut t;
        crate::ode_mixture::likelihood::gene_log_likelihood(&reads, g, &rows, &lambda, (&t.0, &t.1))
    };
    eprintln!("log-likelihood at the truth {at_truth:.1}");
}

#[test]
fn the_samplers_coordinates_round_trip() {
    let z_bar = [0.3, -2.0];
    let t = to_theta(3.0, 0.7, &[0.1, -0.4], 1.2, &z_bar);
    let (b, y, r, base) = from_theta(&t, &z_bar);
    assert!((b - 3.0).abs() < 1e-12 && (y - 0.7).abs() < 1e-12 && (base - 1.2).abs() < 1e-12);
    assert_eq!(r, &[0.1, -0.4]);
}

#[test]
#[ignore = "diagnostic"]
fn diagnostic_gene_laplace() {
    let cfg = SimConfig {
        fine: 133,
        ..SimConfig::default()
    };
    let sim = simulate(&cfg).unwrap();
    let truth = truth_params(&sim, 133, 45);
    let reads = Reads::new(&sim.u, &sim.s, sim.n_pb, sim.n_genes);
    let lv = truth.levels();
    let rows: Vec<usize> = (0..sim.n_pb)
        .map(|p| (sim.truth.time[p] / (cfg.t_len / 44.0)).round() as usize)
        .collect();
    let lambda = reads.depths(&rows, &lv);
    let prior = GenePrior::default();
    let g: usize = std::env::var("GENE").map_or(3, |v| v.parse().unwrap());
    let t0 = to_theta(
        truth.beta[g],
        truth.gamma[g],
        &truth.rho[g],
        truth.base[g],
        &truth.path_mean(),
    );
    for (name, anchor) in [
        ("prior mean", prior.mean(truth.rho[g].len())),
        ("truth", t0.clone()),
    ] {
        let prop = gene_laplace(&truth, g, &reads, &rows, &lambda, &prior, &anchor);
        eprintln!(
            "from {name}: mode {:?}",
            prop.mode
                .iter()
                .map(|x| (x * 1000.0).round() / 1000.0)
                .collect::<Vec<_>>()
        );
    }
    eprintln!(
        "truth θ {:?}",
        t0.iter()
            .map(|x| (x * 1000.0).round() / 1000.0)
            .collect::<Vec<_>>()
    );
    let prop = gene_laplace(&truth, g, &reads, &rows, &lambda, &prior, &t0);
    let mut rng = StdRng::seed_from_u64(1);
    let (mut theta, mut ll) = (prop.mode.clone(), f64::NEG_INFINITY);
    let mut lls = Vec::new();
    for it in 0..3000 {
        (theta, ll) = gene_step(
            &truth, g, &theta, ll, &reads, &rows, &lambda, &prior, &prop, &mut rng,
        );
        if it % 500 == 499 {
            eprintln!(
                "{it}: θ {:?} ll {ll:.2}",
                theta
                    .iter()
                    .map(|x| (x * 1000.0).round() / 1000.0)
                    .collect::<Vec<_>>()
            );
        }
        lls.push(ll);
    }
    let path_mean: Vec<f64> = (0..truth.rho[g].len())
        .map(|d| truth.path[0].iter().map(|z| z[d]).sum::<f64>() / truth.path[0].len() as f64)
        .collect();
    eprintln!("path mean {path_mean:?}");
    let _ = lls;
    let mut rng = StdRng::seed_from_u64(1);
    let (mut theta, mut ll) = (prior.mean(truth.rho[g].len()), f64::NEG_INFINITY);
    let prop = gene_laplace(&truth, g, &reads, &rows, &lambda, &prior, &theta);
    let mut lb = Vec::new();
    for it in 0..4000 {
        (theta, ll) = gene_step(
            &truth, g, &theta, ll, &reads, &rows, &lambda, &prior, &prop, &mut rng,
        );
        if it >= 300 {
            lb.push(from_theta(&theta, &truth.path_mean()).0.ln() as f32);
        }
        if it < 5 || it % 1000 == 999 {
            eprintln!(
                "start-from-prior {it}: log β {:.3} ll {ll:.1}",
                from_theta(&theta, &truth.path_mean()).0.ln()
            );
        }
    }
    eprintln!(
        "effective draws {}",
        legume_numeric::mcmc::engine::diagnostics::ess(&lb)
    );
}

/// From the posterior mode (Adam from the truth), the path block alone,
/// genes, latents and depths held: it mixes. The move is the generalized
/// elliptical slice on the Gauss–Newton Laplace approximation
/// (`path_laplace`, `path_gess_step`); `PLAIN=1` runs plain ESS on the
/// prior instead, for comparison. Followed are the invariants a
/// path sets, log transcription `ρ_gᵀz(t_k) + b_g` of a few genes at a few
/// times, and the reads' log-likelihood; their effective draws are reported
/// and must not be few. (A start from a flat path, as this test once
/// tried, is the climb's job, not the sampler's.)
#[test]
fn a_lineages_path_mixes_from_the_mode() {
    use crate::ode_mixture::blocks::{path_gess_step, path_laplace, path_ll, path_step};
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
    let prop = path_laplace(&params, 0, &eta, &grid, &reads, &rows, &lambda, &prior);
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
    use crate::ode_mixture::blocks::{path_gess_step, path_laplace, path_step};
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
    let prop = path_laplace(&params, 0, &anchor, &grid, &reads, &[], &[], &gene_prior);
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

/// The collapsed path density's gradient (every pseudobulk's lineage and
/// grid point summed out) is its finite difference.
#[test]
fn the_collapsed_paths_gradient_is_its_finite_difference() {
    use crate::ode_mixture::blocks::{collapsed_newton, collapsed_path_log_density};
    use crate::ode_mixture::kernel::GridPrior;
    let cfg = SimConfig {
        lineages: 2,
        n_pb: 120,
        fine: 133,
        ..SimConfig::default()
    };
    let sim = simulate(&cfg).unwrap();
    let truth = truth_params(&sim, 133, 23);
    let reads = Reads::new(&sim.u, &sim.s, sim.n_pb, sim.n_genes);
    let grid = GridPrior::new(23, 1.0, cfg.kappa).unwrap();
    let prior = GenePrior::default();
    let log_w = vec![0.5f64.ln(); 2];
    let h = truth.rho[0].len();
    let l = 1;
    let eta: Vec<nalgebra::DVector<f64>> = (0..h)
        .map(|d| grid.whiten(&nalgebra::DVector::from_fn(23, |k, _| truth.path[l][k][d])))
        .collect();
    let density = |eta: &[nalgebra::DVector<f64>]| {
        let mut p = truth.clone();
        let dims: Vec<_> = eta.iter().map(|e| grid.colour(e)).collect();
        p.path[l] = (0..23)
            .map(|k| dims.iter().map(|d| d[k]).collect())
            .collect();
        collapsed_path_log_density(&p, &reads, &log_w, &prior)
            - 0.5 * eta.iter().map(|e| e.norm_squared()).sum::<f64>()
    };
    let (grad, _) = collapsed_newton(&truth, l, &grid, &reads, &log_w, &prior);
    for (d, k) in [(0, 0), (1, 5), (2, 11), (0, 22)] {
        let eps = 1e-5;
        let (mut up, mut down) = (eta.clone(), eta.clone());
        up[d][k] += eps;
        down[d][k] -= eps;
        let fd = (density(&up) - density(&down)) / (2.0 * eps);
        let an = grad[d * 23 + k];
        assert!(
            (fd - an).abs() < 1e-3 * fd.abs().max(1.0),
            "({d}, {k}): analytic {an} vs {fd}"
        );
    }
}

#[test]
#[ignore = "diagnostic: timing of the collapsed path pieces"]
fn diagnostic_collapsed_timing() {
    use crate::ode_mixture::blocks::{
        collapsed_laplace, collapsed_newton, collapsed_path_log_density,
    };
    use crate::ode_mixture::kernel::GridPrior;
    use std::time::Instant;
    let cfg = SimConfig {
        fine: 133,
        ..SimConfig::default()
    };
    let sim = simulate(&cfg).unwrap();
    let truth = truth_params(&sim, 133, 45);
    let reads = Reads::new(
        &sim.u[..240 * sim.n_genes],
        &sim.s[..240 * sim.n_genes],
        240,
        sim.n_genes,
    );
    let grid = GridPrior::new(45, 1.0, cfg.kappa).unwrap();
    let prior = GenePrior::default();
    let log_w = vec![0.0];
    let t = Instant::now();
    for _ in 0..20 {
        collapsed_path_log_density(&truth, &reads, &log_w, &prior);
    }
    eprintln!("density {:.1} ms", t.elapsed().as_secs_f64() * 50.0);
    let t = Instant::now();
    for _ in 0..3 {
        collapsed_newton(&truth, 0, &grid, &reads, &log_w, &prior);
    }
    eprintln!("newton {:.1} ms", t.elapsed().as_secs_f64() * 1000.0 / 3.0);
    let h = truth.rho[0].len();
    let eta: Vec<nalgebra::DVector<f64>> = (0..h)
        .map(|d| grid.whiten(&nalgebra::DVector::from_fn(45, |k, _| truth.path[0][k][d])))
        .collect();
    let t = Instant::now();
    collapsed_laplace(&truth, 0, &eta, &grid, &reads, &log_w, &prior, 20);
    eprintln!(
        "laplace from the truth {:.1} ms",
        t.elapsed().as_secs_f64() * 1000.0
    );
}
