//! Every gene on its own from a coarse run's mode ([`genes_at_mode`]).

use super::init::first;
use super::model::{coverage, truth_params, uniform_log_w};
use crate::ode_mixture::coarsen::coarse_params;
use crate::ode_mixture::readout::{grid_posterior, place, Mode};
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
    let reads = first(&sim, sim.n_pb);
    let fc = FeatureCoarsening::from_fine_to_coarse(
        (0..GENES).map(|g| (g * 5) % GROUPS).collect(),
        GROUPS,
    )
    .unwrap();
    let mut genes = truth_params(&sim, 133, 45);
    genes.sub = 3;
    // The rows at the genes' truth, the parameters the groups'.
    let truth = Mode {
        log_w: uniform_log_w(2),
        params: genes,
        share_sd: None,
    };
    let placed = place(&reads, &truth, &grid_posterior(&reads, &truth));
    let groups = Mode {
        params: coarse_params(&fc, &truth.params),
        ..truth.clone()
    };
    let at_mode = genes_at_mode(&reads, Some(&fc), &groups, &placed);
    assert_eq!(at_mode.log_gamma.len(), GENES);
    let t = &sim.truth;
    let gamma = coverage(&at_mode.log_gamma, |g| t.gamma[g].ln(), 0.05);
    let ratio = coverage(&at_mode.log_ratio, |g| (t.gamma[g] / t.beta[g]).ln(), 0.05);
    eprintln!("coverage from the groups' start: log γ {gamma:.3}, log γ/β {ratio:.3}");
    assert!(gamma >= 0.8, "log γ coverage {gamma}");
    assert!(ratio >= 0.8, "log γ/β coverage {ratio}");
}
