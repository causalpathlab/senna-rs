//! Genes sharing kinetics coarsen exactly into a group of their summed
//! reads ([`coarse_reads`], [`coarse_params`]).

use super::model::truth_params;
use crate::ode_mixture::blocks::draw_latents;
use crate::ode_mixture::coarsen::{coarse_params, coarse_reads, coarsen_genes, gene_params};
use crate::ode_mixture::likelihood::{table, GeneRows, Reads, RowLayout};
use crate::ode_mixture::model::Params;
use crate::ode_mixture::sim::{simulate, SimConfig};
use data_beans::alg::feature_coarsening::FeatureCoarsening;

const GENES: usize = 40;
const GROUPS: usize = 6;

/// Two lineages, part-steps; every module's genes on its first member's
/// rates and embedding, each on its own baseline.
fn setup() -> (Params, Reads, FeatureCoarsening) {
    let cfg = SimConfig {
        lineages: 2,
        n_genes: GENES,
        fine: 133,
        ..SimConfig::default()
    };
    let sim = simulate(&cfg).unwrap();
    let mut params = truth_params(&sim, 133, 45);
    params.t_len[1] = 1.3;
    params.sub = 3;
    let groups = FeatureCoarsening::from_fine_to_coarse(
        (0..GENES).map(|g| (g * 5) % GROUPS).collect(),
        GROUPS,
    )
    .unwrap();
    for g in 0..GENES {
        let first = groups.coarse_to_fine[groups.fine_to_coarse[g]][0];
        params.beta[g] = params.beta[first];
        params.gamma[g] = params.gamma[first];
        params.rho[g] = params.rho[first].clone();
    }
    let reads = Reads::new(&sim.u, &sim.s, sim.n_pb, sim.n_genes);
    (params, reads, groups)
}

/// Every pseudobulk's log-likelihood at every grid point is its groups'
/// up to a constant of the pseudobulk: its latents' posterior is the same.
#[test]
fn the_groups_tables_are_the_genes_up_to_a_pseudobulks_constant() {
    let (params, reads, groups) = setup();
    let (lv, lv_m) = (params.levels(), coarse_params(&groups, &params).levels());
    let (tab, tab_m) = (
        table(&reads, &lv),
        table(&coarse_reads(&groups, &reads), &lv_m),
    );
    assert_eq!(tab_m.ncols(), tab.ncols());
    let mut worst = 0f64;
    for p in 0..tab.nrows() {
        let shift = tab[(p, 0)] - tab_m[(p, 0)];
        for r in 0..tab.ncols() {
            let gap = tab[(p, r)] - tab_m[(p, r)] - shift;
            worst = worst.max(gap.abs() / tab[(p, r)].abs().max(1.0));
        }
    }
    eprintln!("worst relative gap {worst:.2e}");
    assert!(worst < 1e-10, "worst relative gap {worst}");
    let log_w = [0.5f64.ln(); 2];
    let draw = |t| draw_latents(t, &log_w, params.positions(), 5, 0);
    assert_eq!(draw(&tab), draw(&tab_m));
}

/// Every pseudobulk's depth is the same from the groups' levels.
#[test]
fn the_groups_depths_are_the_genes() {
    let (params, reads, groups) = setup();
    let (lv, lv_m) = (params.levels(), coarse_params(&groups, &params).levels());
    let rows: Vec<usize> = (0..reads.u.nrows())
        .map(|p| (p * 7) % lv.u.nrows())
        .collect();
    let (d, d_m) = (
        reads.depths(&rows, &lv),
        coarse_reads(&groups, &reads).depths(&rows, &lv_m),
    );
    for (a, b) in d.iter().zip(&d_m) {
        for (x, y) in a.iter().zip(b) {
            assert!((x - y).abs() <= 1e-10 * x.abs(), "{x} vs {y}");
        }
    }
}

/// A group's log-likelihood given the rows is its genes' summed, up to a
/// constant of the reads alone: moving the shared kinetics or the paths
/// changes both alike.
#[test]
fn a_groups_log_likelihood_is_its_genes_up_to_a_constant() {
    let (params, reads, groups) = setup();
    let lv = params.levels();
    let rows: Vec<usize> = (0..reads.u.nrows())
        .map(|p| (p * 7) % lv.u.nrows())
        .collect();
    let lambda = reads.depths(&rows, &lv);
    let layout = RowLayout::new(&rows, &lambda);
    let reads_m = coarse_reads(&groups, &reads);
    let gap_at = |params: &Params, k: usize| {
        let p_m = coarse_params(&groups, params);
        let genes: f64 = groups.coarse_to_fine[k]
            .iter()
            .map(|&g| {
                let (u, s) = params.gene_levels(g);
                GeneRows::new(&layout, &reads, g).log_likelihood((&u, &s))
            })
            .sum();
        let (u, s) = p_m.gene_levels(k);
        genes - GeneRows::new(&layout, &reads_m, k).log_likelihood((&u, &s))
    };
    let mut moved = params.clone();
    for p in &mut moved.path {
        for z in p.iter_mut() {
            z[0] += 0.2;
        }
    }
    for k in 0..GROUPS {
        for &g in &groups.coarse_to_fine[k] {
            moved.beta[g] *= 1.3;
            moved.gamma[g] *= 0.8;
            moved.rho[g][0] += 0.1;
        }
        let (a, b) = (gap_at(&params, k), gap_at(&moved, k));
        assert!(
            (a - b).abs() <= 1e-9 * a.abs().max(1.0),
            "group {k}: {a} vs {b}"
        );
    }
}

/// A group's state back on the genes: every gene on its group's rates
/// and embedding, the group's levels split among its genes.
#[test]
fn a_groups_genes_sum_to_its_levels() {
    let (params, reads, groups) = setup();
    let group = coarse_params(&groups, &params);
    let genes = gene_params(&groups, &group, &reads);
    assert_eq!(genes.beta.len(), GENES);
    let (lv_k, lv_g) = (group.levels(), genes.levels());
    let mut worst = 0f64;
    for (k, members) in groups.coarse_to_fine.iter().enumerate() {
        for r in 0..lv_k.u.nrows() {
            for (p, c) in [(&lv_k.u, &lv_g.u), (&lv_k.s, &lv_g.s)] {
                let sum: f64 = members.iter().map(|&g| c[(r, g)]).sum();
                worst = worst.max((sum - p[(r, k)]).abs() / p[(r, k)].abs());
            }
        }
        for &g in members {
            assert_eq!(genes.beta[g], group.beta[k]);
            assert_eq!(genes.rho[g], group.rho[k]);
        }
    }
    assert!(worst < 1e-12, "worst relative gap {worst}");
}

/// A group's genes split its baseline by their reads.
#[test]
fn a_groups_genes_baselines_follow_their_reads() {
    let (params, reads, groups) = setup();
    let genes = gene_params(&groups, &coarse_params(&groups, &params), &reads);
    let total = |g: usize| reads.u.column(g).sum() + reads.s.column(g).sum();
    for members in &groups.coarse_to_fine {
        for &a in members {
            for &b in members {
                let (want, got) = ((total(a) / total(b)).ln(), genes.base[a] - genes.base[b]);
                assert!((want - got).abs() < 1e-2, "{a}, {b}: {got} vs {want}");
            }
        }
    }
}

/// Genes from `programs` distinct programs (embeddings) × two kinetic classes
/// (slow, fast), `each` genes per pair on their own baselines, read with
/// Poisson noise at pseudobulks spread over one lineage: the reads, and
/// every gene's (program, class).
fn programs_and_classes(programs: usize, each: usize) -> (Reads, Vec<f32>, Vec<(usize, usize)>) {
    use rand::SeedableRng;
    use rand_distr::{Distribution, Poisson};
    let sim = simulate(&SimConfig {
        lineages: 1,
        n_genes: programs,
        fine: 133,
        ..SimConfig::default()
    })
    .unwrap();
    let mut params = truth_params(&sim, 133, 45);
    let rates = [(0.7, 0.4), (6.0, 10.0)];
    let h = sim.truth.rho[0].len();
    let mut labels = Vec::new();
    let (mut beta, mut gamma, mut rho, mut base) = (vec![], vec![], vec![], vec![]);
    for m in 0..programs {
        for (c, &(b, y)) in rates.iter().enumerate() {
            for i in 0..each {
                labels.push((m, c));
                beta.push(b);
                gamma.push(y);
                // Program m on its own path axis, signed: programs apart.
                let mut r = vec![0.0; h];
                r[m % h] = if m < h { 1.5 } else { -1.5 };
                rho.push(r);
                base.push(sim.truth.base[m] + (i as f64 / each as f64 - 0.5) * 2.0);
            }
        }
    }
    (params.beta, params.gamma, params.rho, params.base) = (beta, gamma, rho, base);
    let lv = params.levels();
    let (pbs, g) = (400, labels.len());
    let mut rng = rand::rngs::StdRng::seed_from_u64(3);
    let mut draw = |m: &nalgebra::DMatrix<f64>, p: usize, j: usize| {
        let r = p * (lv.points - 1) / (pbs - 1);
        let mean: f64 = 5000.0 * m[(r, j)] / lv.u.row(r).sum();
        Poisson::new(mean.max(1e-6)).unwrap().sample(&mut rng)
    };
    let u = nalgebra::DMatrix::from_fn(pbs, g, |p, j| draw(&lv.u, p, j));
    let s = nalgebra::DMatrix::from_fn(pbs, g, |p, j| draw(&lv.s, p, j));
    (Reads::from_matrices(u, s), vec![1.0; pbs], labels)
}

/// The share of genes whose group's most common label is their own.
fn purity(fc: &FeatureCoarsening, label: &[usize]) -> f64 {
    let mut agree = 0;
    for members in &fc.coarse_to_fine {
        let mut count = std::collections::HashMap::new();
        for &g in members {
            *count.entry(label[g]).or_insert(0) += 1;
        }
        agree += count.values().max().copied().unwrap_or(0);
    }
    agree as f64 / label.len() as f64
}

/// Genes are grouped by program and by kinetics: a group shares its rates
/// in the coarse fit, so no group mixes programs or a fast class with a
/// slow one.
#[test]
fn genes_are_grouped_by_program_and_kinetics() {
    let (programs, each) = (4, 10);
    let (reads, sizes, labels) = programs_and_classes(programs, each);
    // Twice the pairs: a run has more groups than programs, and splitting a
    // program costs speed, mixing one accuracy.
    let cap = 4 * programs;
    let fc = coarsen_genes(&reads, &sizes, cap, 1).unwrap();
    assert!(fc.num_coarse <= cap, "{} groups", fc.num_coarse);
    let program: Vec<usize> = labels.iter().map(|l| l.0).collect();
    let class: Vec<usize> = labels.iter().map(|l| l.1).collect();
    let (p, c) = (purity(&fc, &program), purity(&fc, &class));
    eprintln!(
        "{} groups; program purity {p:.3}, kinetic class purity {c:.3}",
        fc.num_coarse
    );
    assert!(p >= 0.95, "program purity {p}");
    assert!(c >= 0.95, "kinetic class purity {c}");
}
