//! The cost of a run at the target size.

use crate::ode_mixture::coarsen::coarse_reads;
use crate::ode_mixture::init::{climb, Climb};
use crate::ode_mixture::likelihood::Reads;
use crate::ode_mixture::readout::{grid_posterior, place, Mode};
use crate::ode_mixture::refine::genes_at_mode;
use crate::ode_mixture::sim::{simulate, SimConfig};
use data_beans::alg::feature_coarsening::coarsen_features;
use legume_numeric::candle::candle_core::Device;
use nalgebra::DMatrix;
use std::time::Instant;

/// 10,000 genes × 3,000 pseudobulks in 300 groups: the climb's cost a
/// step, the gene pass's at the mode, and the default run's total from
/// them.
#[test]
#[ignore]
fn a_run_on_groups_at_the_target_size() {
    let (genes, pbs, groups) = (10_000, 3000, 300);
    let sim = simulate(&SimConfig {
        lineages: 2,
        n_genes: genes,
        n_pb: pbs,
        ..SimConfig::default()
    })
    .unwrap();
    let reads = Reads::new(&sim.u, &sim.s, sim.n_pb, sim.n_genes);
    let t = Instant::now();
    let counts: DMatrix<f32> = (&reads.u + &reads.s).transpose().cast();
    let fc = coarsen_features(&counts, &vec![1.0; pbs], &[groups], 1)
        .unwrap()
        .pop()
        .unwrap();
    let grouped = coarse_reads(&fc, &reads);
    let coarsen = t.elapsed().as_secs_f64();

    #[cfg(feature = "cuda")]
    let dev = Device::new_cuda(0).unwrap();
    #[cfg(not(feature = "cuda"))]
    let dev = Device::Cpu;
    let steps = 100;
    let t = Instant::now();
    let (fit, _) = climb(
        &grouped,
        &Climb {
            lineages: 2,
            points: 45,
            dim: 3,
            kappa: 0.25,
            steps,
            lr: 0.03,
            sub: 4,
            jitter: true,
            trunk: false,
            roots: Vec::new(),
        },
        &dev,
    )
    .unwrap();
    let climb_secs = t.elapsed().as_secs_f64();
    let start = fit.params().unwrap();
    let share = fit.share_sd().unwrap();
    let log_w = fit.log_weights().unwrap();

    let t = Instant::now();
    let mode = Mode {
        params: start,
        log_w,
        share_sd: share,
    };
    let placed = place(&grouped, &mode, &grid_posterior(&grouped, &mode));
    let at_mode = genes_at_mode(&reads, Some(&fc), &mode, &placed);
    assert_eq!(at_mode.log_gamma.len(), genes);
    let gene_secs = t.elapsed().as_secs_f64();

    let per_step = climb_secs / steps as f64;
    let total = coarsen + 2500.0 * per_step + gene_secs;
    eprintln!(
        "{} groups in {coarsen:.1} s; climb {:.1} ms a step; gene pass {gene_secs:.1} s; \
         the default run ≈ {:.1} min",
        fc.num_coarse,
        1e3 * per_step,
        total / 60.0
    );
}
