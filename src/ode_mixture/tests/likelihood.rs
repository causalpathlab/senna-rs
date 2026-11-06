use super::model::truth_params;
use crate::ode_mixture::likelihood::{
    gene_log_likelihood, jittered_table, share_term, table, Reads,
};
use crate::ode_mixture::sim::{simulate, SimConfig};

#[test]
fn a_table_entry_is_the_multinomial_log_likelihood() {
    let cfg = SimConfig {
        fine: 81,
        ..SimConfig::default()
    };
    let sim = simulate(&cfg).unwrap();
    let lv = truth_params(&sim, 81, 21).levels();
    let reads = Reads::new(&sim.u, &sim.s, sim.n_pb, sim.n_genes);
    let tab = table(&reads, &lv);
    assert_eq!(tab.shape(), (sim.n_pb, 21));
    for (p, r) in [(0, 0), (9, 20), (100, 7)] {
        let z: f64 = (0..sim.n_genes).map(|g| lv.u[(r, g)] + lv.s[(r, g)]).sum();
        let by_hand: f64 = (0..sim.n_genes)
            .map(|g| {
                sim.u[p * sim.n_genes + g] * (lv.u[(r, g)] / z).ln()
                    + sim.s[p * sim.n_genes + g] * (lv.s[(r, g)] / z).ln()
            })
            .sum();
        assert!(
            (tab[(p, r)] - by_hand).abs() < 1e-8 * by_hand.abs(),
            "{p} {r}"
        );
    }
}

#[test]
fn the_genes_terms_add_up_to_the_table_at_the_depth_it_implies() {
    let sim = simulate(&SimConfig {
        fine: 81,
        ..SimConfig::default()
    })
    .unwrap();
    let lv = truth_params(&sim, 81, 41).levels();
    let reads = Reads::new(&sim.u, &sim.s, sim.n_pb, sim.n_genes);
    let tab = table(&reads, &lv);
    let rows: Vec<usize> = (0..sim.n_pb).map(|p| (p * 7) % 41).collect();
    let lambda = reads.depths(&rows, &lv);
    let genes: f64 = (0..sim.n_genes)
        .map(|g| {
            let (u, s): (Vec<f64>, Vec<f64>) = (
                lv.u.column(g).iter().copied().collect(),
                lv.s.column(g).iter().copied().collect(),
            );
            gene_log_likelihood(&reads, g, &rows, &lambda, (&u, &s))
        })
        .sum();
    let n: Vec<f64> = (0..sim.n_pb).map(|p| reads.total(p)).collect();
    let want: f64 = (0..sim.n_pb)
        .map(|p| tab[(p, rows[p])] + n[p] * n[p].ln() - n[p])
        .sum();
    assert!(
        (genes - want).abs() < 1e-8 * want.abs(),
        "{genes} vs {want}"
    );
}

#[test]
fn the_truth_puts_every_pseudobulk_near_its_time() {
    let cfg = SimConfig::default();
    let sim = simulate(&cfg).unwrap();
    let n = 134; // (400 − 1) / 3 + 1
    let lv = truth_params(&sim, cfg.fine, n).levels();
    let tab = table(&Reads::new(&sim.u, &sim.s, sim.n_pb, sim.n_genes), &lv);
    let step = cfg.t_len / (n - 1) as f64;
    let near = (0..sim.n_pb)
        .filter(|&p| {
            let best = (0..n)
                .max_by(|&a, &b| tab[(p, a)].total_cmp(&tab[(p, b)]))
                .unwrap();
            (best as f64 * step - sim.truth.time[p]).abs() < 0.03
        })
        .count();
    assert!(near * 100 >= 95 * sim.n_pb, "{near} of {}", sim.n_pb);
}

/// `log ∫ e^{Δ(ω)} N(ω; 0, σ²) dω`, `Δ(ω) = N_u ω − N log((e^ω Z_u + Z_s)/(Z_u + Z_s))`,
/// by quadrature on a fine grid.
fn share_by_quadrature(n_u: f64, n: f64, z_u: f64, z_s: f64, sd: f64) -> f64 {
    let (lo, hi, m) = (-12.0 * sd - 6.0, 12.0 * sd + 6.0, 400_001);
    let h = (hi - lo) / (m - 1) as f64;
    let f = |w: f64| {
        n_u * w
            - n * ((w.exp() * z_u + z_s) / (z_u + z_s)).ln()
            - 0.5 * (w / sd).powi(2)
            - (sd * (2.0 * std::f64::consts::PI).sqrt()).ln()
    };
    let top = (0..m)
        .map(|i| f(lo + i as f64 * h))
        .fold(f64::MIN, f64::max);
    top + ((0..m)
        .map(|i| (f(lo + i as f64 * h) - top).exp())
        .sum::<f64>()
        * h)
        .ln()
}

/// The share term is the jitter integrated out, to a tenth of a nat at a
/// pseudobulk's depth whether the reads' share agrees with the levels' or
/// not (Laplace is worst with no unspliced reads and a wide jitter, where
/// the integrand flattens out on one side).
#[test]
fn the_share_term_integrates_the_jitter_out() {
    for (n_u, n, z_u, z_s) in [
        (1200.0, 5000.0, 0.3, 0.7),
        (300.0, 5000.0, 0.3, 0.7),
        (0.0, 800.0, 0.2, 0.8),
        (4000.0, 5000.0, 0.1, 0.9),
    ] {
        for sd in [0.05, 0.3, 1.0, 3.0] {
            let (got, want) = (
                share_term(n_u, n, z_u, z_s, sd),
                share_by_quadrature(n_u, n, z_u, z_s, sd),
            );
            assert!(
                (got - want).abs() < 0.1,
                "N_u {n_u} N {n} Z_u {z_u} sd {sd}: {got} vs {want}"
            );
        }
    }
}

/// With no jitter there is no share term: the shared model.
#[test]
fn with_no_jitter_the_share_term_is_nothing() {
    assert_eq!(share_term(1200.0, 5000.0, 0.3, 0.7, 0.0), 0.0);
}

/// Every pseudobulk's jitter, a pattern of no meaning.
fn some_jitter(p_n: usize) -> Vec<f64> {
    (0..p_n).map(|p| 0.4 * (p as f64 * 0.7).sin()).collect()
}

/// With a jitter `ω_p` a table entry is the multinomial of the levels with
/// the unspliced scaled by `e^{ω_p}`; with none it is the shared table.
#[test]
fn with_a_jitter_a_table_entry_is_the_multinomial_of_the_jittered_levels() {
    let sim = simulate(&SimConfig {
        fine: 81,
        ..SimConfig::default()
    })
    .unwrap();
    let lv = truth_params(&sim, 81, 21).levels();
    let reads = Reads::new(&sim.u, &sim.s, sim.n_pb, sim.n_genes);
    let omega = some_jitter(sim.n_pb);
    let tab = jittered_table(&reads, &lv, &omega);
    for (p, r) in [(0, 0), (9, 20), (100, 7)] {
        let e = omega[p].exp();
        let z: f64 = (0..sim.n_genes)
            .map(|g| e * lv.u[(r, g)] + lv.s[(r, g)])
            .sum();
        let by_hand: f64 = (0..sim.n_genes)
            .map(|g| {
                sim.u[p * sim.n_genes + g] * (e * lv.u[(r, g)] / z).ln()
                    + sim.s[p * sim.n_genes + g] * (lv.s[(r, g)] / z).ln()
            })
            .sum();
        assert!(
            (tab[(p, r)] - by_hand).abs() < 1e-8 * by_hand.abs(),
            "{p} {r}"
        );
    }
    let shared = table(&reads, &lv);
    let none = jittered_table(&reads, &lv, &vec![0.0; sim.n_pb]);
    assert!((shared - none).abs().max() < 1e-8);
}

#[test]
fn the_genes_terms_add_up_to_the_jittered_table_at_its_depths() {
    let sim = simulate(&SimConfig {
        fine: 81,
        ..SimConfig::default()
    })
    .unwrap();
    let lv = truth_params(&sim, 81, 41).levels();
    let reads = Reads::new(&sim.u, &sim.s, sim.n_pb, sim.n_genes);
    let omega = some_jitter(sim.n_pb);
    let tab = jittered_table(&reads, &lv, &omega);
    let rows: Vec<usize> = (0..sim.n_pb).map(|p| (p * 7) % 41).collect();
    let lambda = reads.jittered_depths(&rows, &lv, &omega);
    let genes: f64 = (0..sim.n_genes)
        .map(|g| {
            let (u, s): (Vec<f64>, Vec<f64>) = (
                lv.u.column(g).iter().copied().collect(),
                lv.s.column(g).iter().copied().collect(),
            );
            gene_log_likelihood(&reads, g, &rows, &lambda, (&u, &s))
        })
        .sum();
    let n: Vec<f64> = (0..sim.n_pb).map(|p| reads.total(p)).collect();
    let want: f64 = (0..sim.n_pb)
        .map(|p| tab[(p, rows[p])] + n[p] * n[p].ln() - n[p])
        .sum();
    assert!(
        (genes - want).abs() < 1e-8 * want.abs(),
        "{genes} vs {want}"
    );
}

/// A gene's likelihood from its reads summed by row ([`GeneRows`]) is the
/// per-pseudobulk one up to a constant in the gene's parameters: the same
/// offset at its levels and at other levels.
#[test]
fn a_genes_likelihood_by_rows_is_the_per_pseudobulk_one() {
    use crate::ode_mixture::likelihood::{GeneRows, RowLayout};
    let sim = simulate(&SimConfig {
        fine: 133,
        ..SimConfig::default()
    })
    .unwrap();
    let mut params = truth_params(&sim, 133, 23);
    params.sub = 4;
    let reads = Reads::new(&sim.u, &sim.s, sim.n_pb, sim.n_genes);
    let lv = params.levels();
    // Many pseudobulks to a row, some rows empty.
    let rows: Vec<usize> = (0..sim.n_pb).map(|p| (p * 7) % 40).collect();
    let omega: Vec<f64> = (0..sim.n_pb)
        .map(|p| 0.1 * ((p % 5) as f64 - 2.0))
        .collect();
    let lambda = reads.jittered_depths(&rows, &lv, &omega);
    let layout = RowLayout::new(&rows, &lambda);
    for g in 0..sim.n_genes {
        let (u, s) = params.gene_levels(g);
        let (u2, s2): (Vec<f64>, Vec<f64>) = (
            u.iter().map(|x| 1.3 * x).collect(),
            s.iter().map(|x| 0.8 * x).collect(),
        );
        let by_row = GeneRows::new(&layout, &reads, g);
        let offset = |u: &[f64], s: &[f64]| {
            gene_log_likelihood(&reads, g, &rows, &lambda, (u, s)) - by_row.log_likelihood((u, s))
        };
        let (a, b) = (offset(&u, &s), offset(&u2, &s2));
        let want = gene_log_likelihood(&reads, g, &rows, &lambda, (&u, &s)).abs();
        assert!((a - b).abs() < 1e-9 * want, "gene {g}: offsets {a} vs {b}");
    }
}
