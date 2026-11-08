//! Genes sharing kinetics coarsen exactly into a group of their summed
//! reads ([`coarse_reads`], [`coarse_params`]).

use super::model::truth_params;
use crate::ode_mixture::blocks::draw_latents;
use crate::ode_mixture::coarsen::{coarse_params, coarse_reads, uncoarsen_params};
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

/// Two nested coarsenings: every fine group of `GROUPS` inside one of
/// `PARENTS` coarse ones.
const PARENTS: usize = 2;

fn nested() -> (FeatureCoarsening, FeatureCoarsening) {
    let fine: Vec<usize> = (0..GENES).map(|g| (g * 5) % GROUPS).collect();
    let coarse = fine.iter().map(|&j| j % PARENTS).collect();
    (
        FeatureCoarsening::from_fine_to_coarse(coarse, PARENTS).unwrap(),
        FeatureCoarsening::from_fine_to_coarse(fine, GROUPS).unwrap(),
    )
}

/// A coarse state uncoarsened: every fine group on its parent's rates and
/// embedding, the parent's levels split among its children.
#[test]
fn uncoarsened_groups_sum_to_their_parents_levels() {
    let (params, reads, _) = setup();
    let (coarse, fine) = nested();
    let parent = coarse_params(&coarse, &params);
    let child = uncoarsen_params(&coarse, &fine, &parent, &reads);
    assert_eq!(child.beta.len(), GROUPS);
    let (lv_p, lv_c) = (parent.levels(), child.levels());
    let mut worst = 0f64;
    for k in 0..PARENTS {
        let kids: Vec<usize> = (0..GROUPS).filter(|&j| j % PARENTS == k).collect();
        for r in 0..lv_p.u.nrows() {
            for (p, c) in [(&lv_p.u, &lv_c.u), (&lv_p.s, &lv_c.s)] {
                let sum: f64 = kids.iter().map(|&j| c[(r, j)]).sum();
                worst = worst.max((sum - p[(r, k)]).abs() / p[(r, k)].abs());
            }
        }
        for &j in &kids {
            assert_eq!(child.beta[j], parent.beta[k]);
            assert_eq!(child.rho[j], parent.rho[k]);
        }
    }
    assert!(worst < 1e-12, "worst relative gap {worst}");
}

/// The children's baselines split their parent's by their reads.
#[test]
fn uncoarsened_baselines_follow_the_reads() {
    let (params, reads, _) = setup();
    let (coarse, fine) = nested();
    let parent = coarse_params(&coarse, &params);
    let child = uncoarsen_params(&coarse, &fine, &parent, &reads);
    let fine_reads = coarse_reads(&fine, &reads);
    let total = |j: usize| fine_reads.u.column(j).sum() + fine_reads.s.column(j).sum();
    for a in 0..GROUPS {
        for b in 0..GROUPS {
            if a % PARENTS == b % PARENTS {
                let (want, got) = ((total(a) / total(b)).ln(), child.base[a] - child.base[b]);
                assert!((want - got).abs() < 1e-2, "{a}, {b}: {got} vs {want}");
            }
        }
    }
}
