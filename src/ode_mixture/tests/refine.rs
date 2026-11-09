//! Every gene on its own from a coarse run's mode ([`genes_at_mode`]).

use super::model::truth_params;
use crate::ode_mixture::coarsen::coarse_params;
use crate::ode_mixture::likelihood::Reads;
use crate::ode_mixture::readout::mode_snapshot;
use crate::ode_mixture::refine::genes_at_mode;
use crate::ode_mixture::sim::{simulate, Sim, SimConfig};
use data_beans::alg::feature_coarsening::FeatureCoarsening;

const GENES: usize = 40;
const GROUPS: usize = 6;

fn sim() -> Sim {
    simulate(&SimConfig {
        lineages: 2,
        n_genes: GENES,
        fine: 133,
        ..SimConfig::default()
    })
    .unwrap()
}

/// From its group's rates and embedding, every gene's Laplace interval at
/// the truth's rows covers its own true rates: the start is far from the
/// gene, the mode is found from there.
#[test]
fn from_its_groups_start_every_genes_interval_covers_its_rates() {
    let sim = sim();
    let reads = Reads::new(&sim.u, &sim.s, sim.n_pb, sim.n_genes);
    let fc = FeatureCoarsening::from_fine_to_coarse(
        (0..GENES).map(|g| (g * 5) % GROUPS).collect(),
        GROUPS,
    )
    .unwrap();
    let mut genes = truth_params(&sim, 133, 45);
    genes.sub = 3;
    let log_w = vec![0.5f64.ln(); 2];
    // The rows at the genes' truth, the parameters the groups'.
    let mut snap = mode_snapshot(&reads, &genes, &log_w, None);
    snap.params = coarse_params(&fc, &genes);
    let at_mode = genes_at_mode(&reads, Some(&fc), &snap);
    assert_eq!(at_mode.log_gamma.len(), GENES);
    let t = &sim.truth;
    let covered = |est: &[(f64, f64)], truth: &dyn Fn(usize) -> f64| -> f64 {
        (0..GENES)
            .filter(|&g| {
                let (m, s) = est[g];
                s.is_finite() && s > 0.0 && (truth(g) - m).abs() <= 2.0 * s + 0.05
            })
            .count() as f64
            / GENES as f64
    };
    let gamma = covered(&at_mode.log_gamma, &|g| t.gamma[g].ln());
    let ratio = covered(&at_mode.log_ratio, &|g| (t.gamma[g] / t.beta[g]).ln());
    eprintln!("coverage from the groups' start: log γ {gamma:.3}, log γ/β {ratio:.3}");
    assert!(gamma >= 0.8, "log γ coverage {gamma}");
    assert!(ratio >= 0.8, "log γ/β coverage {ratio}");
}
