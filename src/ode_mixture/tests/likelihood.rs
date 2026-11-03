use super::model::truth_params;
use crate::ode_mixture::likelihood::{gene_log_likelihood, table, Reads};
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
