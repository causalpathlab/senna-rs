//! The fit, on pseudobulk samples: every epoch, every level's pseudobulks
//! (coarse to fine) are drawn again at a cell's depth ([`resample`]). The
//! lineages are the root-to-leaf paths of a forest over the finest
//! pseudobulks' clusters ([`crate::td_rbm::tree`]). One file per step, run
//! in this order by [`fit`]:
//!
//! 1. **Lineages and time** ([`exact`]), by the exact likelihood (with the
//!    states off the machine is a mixture of multinomials): every pseudobulk
//!    sample's lineage drawn by collapsed Gibbs under a truncated Dirichlet
//!    process (rich get richer), every lineage's curve fitted on its rows,
//!    time summed out. First at steady state (the velocity off, blind to
//!    direction), later with the velocity on.
//! 2. **Roots** ([`roots`]): every tree's root, its lineages' one direction,
//!    from the splicing along its edges ([`direction`]); the lineages
//!    rebuilt from the chosen roots and the steady fit settled again.
//! 3. **Direction report** ([`direction`]): every lineage's splicing
//!    evidence as it now runs, and its rates' posterior means carried on.
//! 4. **States** ([`states`]), by persistent contrastive divergence, every
//!    parameter with them.

mod direction;
mod exact;
mod gamma;
mod roots;
mod states;

pub use direction::Direction;
pub use roots::RootCall;

use crate::rbm_core::adam;
use crate::td_rbm::curve::{grid_times, Spline};
use crate::td_rbm::model::{Energy, Init0, Shape, Tables, EMPTY};
use crate::td_rbm::rbm::{posterior, Reads};
use crate::td_rbm::start::Start;
use crate::td_rbm::tree::{Forest, Path};
use legume_numeric::candle::candle_core::{Device, Result, Tensor};
use log::info;
use nalgebra::DMatrix;
use rand::rngs::StdRng;
use rand::SeedableRng;

#[derive(Clone, Debug)]
pub struct TrainConfig {
    /// Grid times over `(0, 1)` per lineage.
    pub grid: usize,
    /// Basis functions of every curve.
    pub knots: usize,
    /// Width of a cell's state.
    pub states: usize,
    /// Width of the gene embedding `ρ`.
    pub embedding: usize,
    /// The most lineages, and the Dirichlet process's concentration.
    pub lineages: usize,
    pub concentration: f64,
    pub epochs: usize,
    /// The first epochs fit lineages and time alone; then the states join.
    pub time_epochs: usize,
    /// Epochs first fitting every lineage at steady state (velocity off),
    /// which shapes the curves and rates the same either way in time.
    pub steady_epochs: usize,
    /// Draws of every pseudobulk's time behind a lineage's direction.
    pub direction_draws: usize,
    /// The depth a pseudobulk sample's lineage is chosen at: its
    /// log-likelihood per read times this, so the Dirichlet process's
    /// rich-get-richer prior weighs in between lineages that explain it about
    /// equally well.
    pub assign_depth: f64,
    /// Reads every pseudobulk sample holds in the exact phases.
    pub depth: u64,
    /// Reads every sample and every chain holds in the states phase: the
    /// time unit's evidence grows with the reads, so a smaller depth
    /// tempers its posterior and the chains move between times and
    /// lineages; the states read per read and are the same at any depth.
    pub state_depth: u64,
    pub batch: usize,
    /// Neighbours per pseudobulk in a lineage's diffusion order.
    pub knn: usize,
    /// Prior weight on every curve's roughness ([`Energy::roughness`]), in
    /// nats per unit roughness over one pass through every level's samples.
    pub smooth: f64,
    /// Persistent chains and Gibbs sweeps per step.
    pub chains: usize,
    pub sweeps: usize,
    pub lr: f64,
    pub seed: u64,
}

impl Default for TrainConfig {
    fn default() -> Self {
        Self {
            grid: 32,
            knots: 8,
            states: 4,
            embedding: 8,
            lineages: 1,
            concentration: 1.0,
            epochs: 80,
            time_epochs: 40,
            steady_epochs: 40,
            direction_draws: 16,
            assign_depth: 300.0,
            depth: 1000,
            state_depth: 1000,
            knn: 10,
            smooth: 1e4,
            batch: 128,
            chains: 128,
            sweeps: 1,
            lr: 1e-2,
            seed: 1,
        }
    }
}

/// The splicing and degradation rates every gene starts at.
pub const BETA0: f64 = 4.0;
pub const GAMMA0: f64 = 1.0;

/// One level's pseudobulks.
pub struct Level {
    /// `[P × F]` row-major counts, unspliced genes first.
    pub x: Vec<f32>,
    /// `[P]` cells per pseudobulk.
    pub cells: Vec<f32>,
}

/// A fitted machine.
pub struct Model {
    pub energy: Energy,
    pub spline: Spline,
    /// `[L]` every lineage's log weight ([`EMPTY`] where empty).
    pub log_weight: Vec<f32>,
    pub directions: Vec<Direction>,
    /// The lineages (paths from every tree's chosen root) and the roots.
    pub paths: Vec<Path>,
    pub roots: Vec<RootCall>,
    /// Every finest pseudobulk's starting lineage and order, along the
    /// chosen roots.
    pub start_lineage: Vec<usize>,
    pub start_order: Vec<f32>,
}

/// Every row's lineage, time, its spread, state and velocity.
pub struct Summary {
    /// The likeliest lineage, and the posterior over lineages `[N × L]`.
    pub lineage: Vec<usize>,
    pub p_lineage: Vec<Vec<f32>>,
    /// Time within the likeliest lineage, and its sd.
    pub tau: Vec<f32>,
    pub tau_sd: Vec<f32>,
    /// `[N × d]` the state's posterior mean.
    pub state: Vec<Vec<f32>>,
    /// `[N × H]` the velocity `ż`, averaged over the row's posterior.
    pub velocity: Vec<Vec<f32>>,
}

/// Rows read at a time at the end.
const BLOCK: usize = 4096;

impl Model {
    /// The tables at the fitted weights.
    pub fn tables(&self, states: bool) -> Result<Tables> {
        self.energy.tables(&self.spline, states, &self.log_weight)
    }

    /// Every row of `x` (`[N × F]` counts on the host), read under the
    /// fitted machine a block at a time. A row deeper than `depth` (the
    /// training depth) is read as if scaled down to it, as the machine's
    /// samples were; a shallower one at its own depth, never inflated, so its
    /// posteriors stay as wide as its reads allow.
    pub fn summarise(&self, x: &[f32], f: usize, depth: f32, dev: &Device) -> Result<Summary> {
        let t = self.tables(true)?.detach();
        let (l, k) = (t.lineages, t.grid_len()?);
        let n = x.len() / f;
        let times = grid_times(k);
        let mut out = Summary {
            lineage: Vec::with_capacity(n),
            p_lineage: Vec::with_capacity(n),
            tau: Vec::with_capacity(n),
            tau_sd: Vec::with_capacity(n),
            state: Vec::with_capacity(n),
            velocity: Vec::with_capacity(n),
        };
        for start in (0..n).step_by(BLOCK) {
            let w = BLOCK.min(n - start);
            let mut rows = x[start * f..(start + w) * f].to_vec();
            for row in rows.chunks_mut(f) {
                let own: f32 = row.iter().sum();
                if own > depth {
                    row.iter_mut().for_each(|v| *v *= depth / own);
                }
            }
            let block = Reads::from_features(Tensor::from_vec(rows, (w, f), dev)?)?;
            let (p, state) = posterior(&block, &t)?;
            out.velocity
                .extend(p.matmul(&t.velocity)?.to_vec2::<f32>()?);
            out.state.extend(state.to_vec2::<f32>()?);
            for row in p.to_vec2::<f32>()? {
                let pl: Vec<f32> = (0..l)
                    .map(|i| row[i * k..(i + 1) * k].iter().sum())
                    .collect();
                let best = (0..l)
                    .max_by(|&a, &b| pl[a].total_cmp(&pl[b]))
                    .expect("a lineage");
                let q = &row[best * k..(best + 1) * k];
                let z = pl[best].max(1e-30);
                let mean: f32 = q.iter().zip(&times).map(|(p, t)| p * t).sum::<f32>() / z;
                let second: f32 = q.iter().zip(&times).map(|(p, t)| p * t * t).sum::<f32>() / z;
                out.tau.push(mean);
                out.tau_sd.push((second - mean * mean).max(0.0).sqrt());
                out.lineage.push(best);
                out.p_lineage.push(pl);
            }
        }
        Ok(out)
    }
}

/// Every lineage's log weight from its cells: the Dirichlet process's
/// posterior mean, empty lineages at [`EMPTY`].
fn log_weights(counts: &[f64], concentration: f64) -> Vec<f32> {
    let total: f64 = counts.iter().sum::<f64>() + concentration;
    counts
        .iter()
        .map(|&n| {
            if n > 0.0 {
                ((n + concentration / counts.len() as f64) / total).ln() as f32
            } else {
                EMPTY
            }
        })
        .collect()
}

/// Reads in one pass through every level's samples at `depth`: what the
/// per-read loss averages over, so a prior in nats is divided by it.
fn pass_reads(levels: &[Level], depth: u64) -> f64 {
    levels.iter().map(|lv| lv.cells.len()).sum::<usize>() as f64 * depth as f64
}

/// Fit the machine on pseudobulks of every level (`levels`, coarse to fine),
/// the genes' embedding, baselines and curves starting at `init` and the
/// finest pseudobulks where `start` puts them (both built along `forest`
/// from `roots` on the finest pseudobulks' expression `states`,
/// [`crate::td_rbm::start::start_along`]); `given[t]` fixes tree `t`'s
/// root. `cfg.lineages` is the number of paths.
#[allow(clippy::too_many_arguments)]
pub fn fit(
    levels: &[Level],
    f: usize,
    init: &Init0,
    start: Start,
    forest: &Forest,
    states: &DMatrix<f32>,
    roots: &[usize],
    given: &[bool],
    cfg: &TrainConfig,
    dev: &Device,
) -> anyhow::Result<Model> {
    anyhow::ensure!(!levels.is_empty(), "no level to fit");
    anyhow::ensure!(f.is_multiple_of(2), "both tracks of every gene");
    let finest = levels.len() - 1;
    let mut paths = forest.paths(roots);
    let l = cfg.lineages;
    anyhow::ensure!(paths.len() == l, "a lineage per path");
    anyhow::ensure!(
        forest.cluster.len() == levels[finest].cells.len(),
        "a cluster per finest pseudobulk"
    );
    let spline = Spline::new(cfg.knots, &grid_times(cfg.grid), dev)?;
    let energy = Energy::new(
        Shape {
            genes: f / 2,
            embedding: cfg.embedding,
            knots: cfg.knots,
            states: cfg.states,
            lineages: l,
        },
        init,
        cfg.seed,
        dev,
    )?;
    let mut rng = StdRng::seed_from_u64(cfg.seed);

    // Every finest pseudobulk's lineage and the cells every lineage holds;
    // the coarser levels' rows follow them.
    let mut assigned: Vec<Vec<usize>> = levels.iter().map(|lv| vec![0; lv.cells.len()]).collect();
    assigned[finest] = start.lineage.clone();
    let mut counts = vec![0f64; l];
    for (&a, &c) in start.lineage.iter().zip(&levels[finest].cells) {
        counts[a] += f64::from(c);
    }
    let mut phase = exact::Phase {
        energy: &energy,
        spline: &spline,
        levels,
        f,
        cfg,
        dev,
        assigned: &mut assigned,
        counts: &mut counts,
    };
    let mut roots = roots.to_vec();

    // 1. Lineages and curves at steady state (the velocity off), blind to
    // direction. 2. Every tree's root from its splicing; the lineages again
    // from the chosen roots where one moved.
    energy.set_motion(false);
    phase.epochs(cfg.steady_epochs, "steady", &mut rng)?;
    energy.set_motion(true);
    let calls = phase.choose_roots(forest, &paths, &roots, given, &mut rng)?;
    for c in &calls {
        info!(
            "td-rbm tree {} ({} clusters): root cluster {} (measured from {}), p {:.4}{}",
            c.tree + 1,
            forest.trees[c.tree].len(),
            c.chosen,
            c.provisional,
            c.p.iter().cloned().fold(0.0, f64::max),
            match (c.given, c.determined, c.mirrored) {
                (true, ..) => ", given",
                (false, false, _) => ", undetermined",
                (false, true, true) => ", mirrored",
                (false, true, false) => "",
            }
        );
    }
    let mut start = start;
    // A mirrored one-cluster lineage starts the other way round too.
    for c in calls.iter().filter(|c| c.mirrored) {
        for (o, &l) in start.order.iter_mut().zip(&start.lineage) {
            if paths[l].tree == c.tree {
                *o = 1.0 - *o;
            }
        }
    }
    let chosen: Vec<usize> = calls.iter().map(|c| c.chosen).collect();
    if chosen != roots {
        let moved: Vec<bool> = chosen.iter().zip(&roots).map(|(a, b)| a != b).collect();
        roots = chosen;
        start = phase.rebuild(forest, states, &roots, &moved)?;
        paths = start.paths.clone();
        energy.set_motion(false);
        phase.epochs(cfg.steady_epochs, "steady (rerooted)", &mut rng)?;
        energy.set_motion(true);
    }
    // 3. Every lineage's splicing as it runs; the rates carried on. Then
    // lineages and time with the velocity on.
    let directions = direction::orient_all(
        &energy,
        &spline,
        &levels[finest].x,
        f,
        &phase.assigned[finest],
        cfg,
        dev,
        &mut rng,
    )?;
    phase.epochs(cfg.time_epochs, "time", &mut rng)?;
    let log_weight = log_weights(&counts, cfg.concentration);

    // 4. States.
    if cfg.epochs > cfg.time_epochs {
        states::fit_states(&energy, &spline, levels, f, cfg, dev, &log_weight, &mut rng)?;
    }
    Ok(Model {
        energy,
        spline,
        log_weight,
        directions,
        paths,
        roots: calls,
        start_lineage: start.lineage,
        start_order: start.order,
    })
}

#[cfg(test)]
#[path = "tests/train.rs"]
mod tests;
