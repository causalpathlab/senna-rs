//! Every gene drawn on its own from a coarse run's snapshots
//! ([`refine_genes`]).

use super::model::truth_params;
use crate::ode_mixture::blocks::draw_latents;
use crate::ode_mixture::coarsen::coarse_params;
use crate::ode_mixture::likelihood::{table, Reads};
use crate::ode_mixture::readout::Snapshot;
use crate::ode_mixture::refine::refine_genes;
use crate::ode_mixture::sim::{simulate, Sim, SimConfig};
use data_beans::alg::feature_coarsening::FeatureCoarsening;

const GENES: usize = 40;
const GROUPS: usize = 6;
const SNAPS: usize = 60;

fn sim() -> Sim {
    simulate(&SimConfig {
        lineages: 2,
        n_genes: GENES,
        fine: 133,
        ..SimConfig::default()
    })
    .unwrap()
}

/// `SNAPS` draws at the truth, each its own rows drawn at the truth, the
/// genes collapsed into `GROUPS` groups: every gene starts far from itself.
fn coarse_snaps(sim: &Sim, reads: &Reads, fc: &FeatureCoarsening) -> Vec<Snapshot> {
    let mut genes = truth_params(sim, 133, 45);
    genes.sub = 3;
    let lv = genes.levels();
    let tab = table(reads, &lv);
    let log_w = vec![0.5f64.ln(); 2];
    (0..SNAPS)
        .map(|s| {
            let rows = draw_latents(&tab, &log_w, genes.positions(), 11, s as u64);
            let lambda = reads.depths(&rows, &lv);
            Snapshot {
                params: coarse_params(fc, &genes),
                log_w: log_w.clone(),
                share_sd: None,
                rows,
                lambda,
            }
        })
        .collect()
}

/// A 90% interval of the draws, widened by `slack` each side.
fn covers(mut draws: Vec<f64>, truth: f64, slack: f64) -> bool {
    draws.sort_by(f64::total_cmp);
    let n = draws.len();
    let (lo, hi) = (draws[n / 20], draws[n - 1 - n / 20]);
    lo - slack <= truth && truth <= hi + slack
}

/// From groups' rates, every gene's intervals cover its own true rates.
#[test]
fn from_its_groups_start_every_genes_intervals_cover_its_rates() {
    let sim = sim();
    let reads = Reads::new(&sim.u, &sim.s, sim.n_pb, sim.n_genes);
    let fc = FeatureCoarsening::from_fine_to_coarse(
        (0..GENES).map(|g| (g * 5) % GROUPS).collect(),
        GROUPS,
    )
    .unwrap();
    let snaps = coarse_snaps(&sim, &reads, &fc);
    let draws = refine_genes(&reads, &fc, &snaps, 3, 7);
    assert_eq!(draws.log_gamma.len(), SNAPS);
    let t = &sim.truth;
    let share = |of: &dyn Fn(usize) -> (Vec<f64>, f64)| {
        (0..GENES)
            .filter(|&g| {
                let (d, truth) = of(g);
                covers(d, truth, 0.05)
            })
            .count() as f64
            / GENES as f64
    };
    let gamma = share(&|g| {
        let d = draws.log_gamma.iter().map(|s| s[g]).collect();
        (d, t.gamma[g].ln())
    });
    let ratio = share(&|g| {
        let d = draws.log_ratio.iter().map(|s| s[g]).collect();
        (d, (t.gamma[g] / t.beta[g]).ln())
    });
    eprintln!("coverage: log γ {gamma:.3}, log γ/β {ratio:.3}");
    assert!(gamma >= 0.8, "log γ coverage {gamma}");
    assert!(ratio >= 0.8, "log γ/β coverage {ratio}");
}
