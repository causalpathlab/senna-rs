//! The run on groups at the target size, timed (ignored: minutes).

use crate::ode_mixture::coarsen::coarse_reads;
use crate::ode_mixture::init::{climb, Climb};
use crate::ode_mixture::likelihood::Reads;
use crate::ode_mixture::refine::refine_genes;
use crate::ode_mixture::sampler;
use crate::ode_mixture::sim::{simulate, SimConfig};
use data_beans::alg::feature_coarsening::coarsen_features;
use legume_numeric::candle::candle_core::Device;
use nalgebra::DMatrix;
use std::time::Instant;

/// 10,000 genes × 3,000 pseudobulks in 300 groups: the climb's and the
/// chains' cost a step and a sweep, the gene pass's a snapshot, and the
/// default run's total from them.
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
            trunk: true,
        },
        &dev,
    )
    .unwrap();
    let climb_secs = t.elapsed().as_secs_f64();
    let start = fit.params().unwrap();
    let share = fit.share_sd().unwrap();

    let (sweeps, burn) = (30, 10);
    let t = Instant::now();
    let draws = sampler::run(
        &grouped,
        &start,
        &sampler::Options {
            chains: 4,
            sweeps,
            burn,
            share,
            snapshot_every: 4,
            ..sampler::Options::default()
        },
    )
    .unwrap();
    let chain_secs = t.elapsed().as_secs_f64();

    let t = Instant::now();
    let snaps = &draws[0].snapshots;
    let refined = refine_genes(&reads, &fc, snaps, 3, 7);
    assert_eq!(refined.log_gamma[0].len(), genes);
    let per_snap = t.elapsed().as_secs_f64() / snaps.len() as f64;

    let per_step = climb_secs / steps as f64;
    let per_sweep = chain_secs / sweeps as f64;
    let total = coarsen + 2500.0 * per_step + 1500.0 * per_sweep + 4.0 * 25.0 * per_snap;
    eprintln!(
        "{} groups in {coarsen:.1} s; climb {:.1} ms a step; chains {:.2} s a sweep (4 at once); \
         gene pass {per_snap:.2} s a snapshot; the default run ≈ {:.1} min",
        fc.num_coarse,
        1e3 * per_step,
        per_sweep,
        total / 60.0
    );
}
