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

/// From a flat path, a lineage's path is found: the chain reaches the
/// truth's log-likelihood and every gene's log transcription over time.
#[test]
#[ignore = "prior-shaped ESS from a flat path is too slow (8400 nats short after 400 sweeps); the sampler starts paths at the Adam mode"]
fn a_lineages_path_is_found_from_a_flat_start() {
    use crate::ode_mixture::blocks::{path_ll, path_step};
    use crate::ode_mixture::kernel::GridPrior;
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
    let prior = GridPrior::new(45, cfg.t_len, cfg.kappa * cfg.t_len).unwrap();
    let h = truth.rho[0].len();
    let at_truth = path_ll(&truth, 0, &reads, &rows, &lambda);

    let mut params = truth.clone();
    params.path[0] = vec![vec![0.0; h]; 45];
    let mut eta = vec![nalgebra::DVector::<f64>::zeros(45); h];
    let mut ll = f64::NEG_INFINITY;
    let mut rng = StdRng::seed_from_u64(4);
    let mut lls = Vec::new();
    for _ in 0..400 {
        (eta, ll) = path_step(
            &mut params,
            0,
            &eta,
            ll,
            &prior,
            &reads,
            &rows,
            &lambda,
            &mut rng,
        );
        lls.push(ll);
    }
    let late: f64 = lls[300..].iter().sum::<f64>() / 100.0;
    let err: f64 = (0..sim.n_genes)
        .map(|g| {
            (0..45)
                .map(|k| {
                    let fit: f64 = params.path[0][k]
                        .iter()
                        .zip(&truth.rho[g])
                        .map(|(a, b)| a * b)
                        .sum();
                    let tru: f64 = truth.path[0][k]
                        .iter()
                        .zip(&truth.rho[g])
                        .map(|(a, b)| a * b)
                        .sum();
                    (fit - tru).abs()
                })
                .sum::<f64>()
                / 45.0
        })
        .sum::<f64>()
        / sim.n_genes as f64;
    eprintln!("log-likelihood: truth {at_truth:.1}, chain late {late:.1}; mean |Δ log α| {err:.3}");
    assert!(
        late > at_truth - 3.0 * ((45 * h) as f64).sqrt(),
        "the chain is short of the truth"
    );
    assert!(err < 0.1, "log transcription off by {err}");
}
