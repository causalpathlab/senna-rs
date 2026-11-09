//! The blocked Gibbs sampler, chains started at the posterior mode.
//!
//! One sweep, every block a move that leaves its conditional invariant:
//!
//! 1. the levels on every lineage's grid ([`Params::levels`]);
//! 2. every pseudobulk's `(ℓ, k)` drawn exactly ([`draw_latents`]), its
//!    depth `λ_p = N_p / Z` at the drawn row;
//! 3. every gene: [`gene_step`] a few times on a Student-t shaped like the
//!    Laplace approximation of its conditional by bounded Newton
//!    ([`gene_mode`]); in
//!    parallel over genes, a stream per gene and sweep;
//! 4. every lineage's whitened path, one lineage after another (they meet
//!    in the genes' prior through the paths' mean): generalized elliptical
//!    slice moves ([`path_gess_step`]) on its Laplace approximation, banded
//!    in the path and factored once a refit ([`path_mode`]; plain ESS on
//!    the prior gives a handful of effective draws in a thousand moves even
//!    from the mode);
//! 5. every other lineage's log duration `log T_ℓ` (lineage 0's is the
//!    unit) by a scalar elliptical slice move on its `N(0, 1)` prior, the
//!    path's whitened coordinates held. The length scale stays a fixed share
//!    of the duration (`κ/T` held, as in the climb): the path on its grid
//!    does not move with `T`, only the grid's step does;
//! 6. the lineages' weights ([`draw_weights`]).
//!
//! The Laplace approximations (the moves' proposals) are refitted, each
//! from the last one's mode, every [`Options::laplace_every`]-th burn-in
//! sweep and at the last; after burn-in they are kept as they are, so the
//! kept sweeps are one fixed kernel leaving the posterior invariant (a
//! homogeneous Markov chain, not an adaptive one).
//!
//! On a shared trunk ([`Params::branch`]) the trunk's path block (lineage
//! 0's) moves every lineage and the others are departures from it
//! ([`super::blocks::place_path`]), each departure's block moving its
//! branch time with it (alone the two trade off along a ridge;
//! [`super::blocks::Space`]); the clock is shared, step 5 skipped.
//!
//! With a jitter of the unspliced share ([`Options::share`]), every
//! pseudobulk's `ω_p` is a latent too: the rows are drawn given it
//! ([`jittered_table`]), then `ω_p` given its row ([`share_step`]), then
//! the jitter's sd given every `ω` ([`draw_share_sd`]); the depths are the
//! jittered ones ([`Reads::jittered_depths`]). The unspliced reads' global
//! capture ([`Params::log_capture`]) is drawn given the rows and jitters
//! ([`capture_step`]) before the depths; the jitters are every
//! pseudobulk's departure from it.
//!
//! The parameters are kept natural, `(β, γ, ρ, b)`; a gene's sampler
//! coordinates ([`to_theta`]) are taken at the current paths' mean each
//! sweep. Chains run in parallel, each from the same start (the Adam mode)
//! with a seed of its own; only invariants are kept (the gauges, rotation of
//! `z` against `ρ` and the baseline's shared shift, do not show in them).

use super::blocks::{
    capture_step, draw_latents, draw_share_sd, draw_weights, from_theta, gene_mode, gene_step,
    path_eta, path_gess_step, path_ll, path_mode, share_step, stream, to_theta, GenePrior,
    GeneProposal, PathProposal,
};
use super::kernel::GridPrior;
use super::likelihood::{jittered_table, table, Depth, GeneRows, Reads, RowLayout};
use super::model::{clamped_exp, Params};
use super::readout::Snapshot;
use legume_numeric::mcmc::engine::elliptical_slice_step;
use nalgebra::DVector;
use rand_distr::{Distribution, StandardNormal};
use rayon::prelude::*;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

#[derive(Clone, Debug)]
pub struct Options {
    /// The prior's length scale over the duration (the climb's).
    pub kappa: f64,
    pub chains: usize,
    /// Sweeps per chain, the first `burn` not kept.
    pub sweeps: usize,
    pub burn: usize,
    /// Moves per sweep of every gene (on one Laplace approximation) and of
    /// every path.
    pub gene_moves: usize,
    pub path_moves: usize,
    /// The weights' Dirichlet prior.
    pub a0: f64,
    pub seed: u64,
    /// Set (as Ctrl+C sets it, [`crate::topic::common::setup_stop_handler`])
    /// to end every chain after its current sweep, keeping what it drew.
    pub stop: Option<Arc<AtomicBool>>,
    /// The jitter of the unspliced share's sd to start from (the climb's,
    /// [`super::fit::Fit::share_sd`]); `None`: no jitter, the shared model.
    pub share: Option<f64>,
    /// Keep every this many kept sweeps' parameters for the cells' readout
    /// ([`super::readout::read_cells`]); `0`: none.
    pub snapshot_every: usize,
    /// During burn-in, refit every gene's and path's Laplace approximation
    /// (the moves' proposals) every this many sweeps (`0` as `1`: every
    /// sweep), and at its last sweep, reusing it between. After burn-in none
    /// is refitted: the kept sweeps are one fixed kernel that leaves the
    /// posterior invariant (a homogeneous Markov chain), not an adaptive one.
    pub laplace_every: usize,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            kappa: 0.25,
            chains: 4,
            sweeps: 400,
            burn: 100,
            gene_moves: 5,
            path_moves: 1,
            a0: 1.0,
            seed: 1,
            stop: None,
            share: None,
            snapshot_every: 0,
            laplace_every: 5,
        }
    }
}

/// One chain's kept draws of the invariants, a row per kept sweep.
pub struct Draws {
    /// `[S][P]` every pseudobulk's lineage and its time on it, `k T_ℓ/(n − 1)`.
    pub lineage: Vec<Vec<usize>>,
    pub time: Vec<Vec<f64>>,
    /// `[S][G]` `log γ` and `log γ/β`.
    pub log_gamma: Vec<Vec<f64>>,
    pub log_ratio: Vec<Vec<f64>>,
    /// `[S][L]` every lineage's log duration (lineage 0's is 0).
    pub log_t: Vec<Vec<f64>>,
    /// `[S]` the reads' log-likelihood at the drawn latents and depths, a
    /// summary of the paths and genes together.
    pub path_ll: Vec<f64>,
    /// `[S]` the jitter's sd and `[S][P]` every pseudobulk's jitter (empty
    /// with no jitter).
    pub share_sd: Vec<f64>,
    pub omega: Vec<Vec<f64>>,
    /// `[S]` the unspliced reads' log capture.
    pub log_capture: Vec<f64>,
    /// `[S][L − 1]` every other lineage's branch time off the trunk (empty
    /// with the lineages apart).
    pub branch: Vec<Vec<f64>>,
    /// Every [`Options::snapshot_every`]-th kept sweep's parameters.
    pub snapshots: Vec<Snapshot>,
    /// The sweeps whose proposals were refitted (all in burn-in, the first
    /// and the last always; [`Options::laplace_every`]).
    pub refits: Vec<usize>,
    /// Seconds the chain spent in every block of its sweeps ([`BLOCKS`]).
    pub block_secs: Vec<f64>,
}

/// [`BLOCKS`]' indices.
const LATENTS_BLOCK: usize = 0;
const JITTER_BLOCK: usize = 1;
const GENES_BLOCK: usize = 2;
const PATH_LAPLACE: usize = 3;
const PATH_MOVES: usize = 4;
const DURATIONS_BLOCK: usize = 5;
const LL_BLOCK: usize = 6;

/// The sweep's blocks as [`Draws::block_secs`] times them.
pub const BLOCKS: [&str; 7] = [
    "levels and latents",
    "jitter, capture and depths",
    "genes",
    "paths' Laplace",
    "paths' moves",
    "durations and weights",
    "reads' log-likelihood",
];

/// A chain's state.
struct State {
    params: Params,
    /// `[L][H]` the whitened paths.
    eta: Vec<Vec<DVector<f64>>>,
    log_w: Vec<f64>,
    /// `[G]` the start's `θ` and `[L][H]` its whitened paths: the first
    /// Laplace fits' anchors (later ones start from the last fit's mode).
    modes: Vec<Vec<f64>>,
    path_modes: Vec<Vec<DVector<f64>>>,
    /// The proposals last fitted (`None` before the first sweep), reused
    /// until the next refit ([`refit_at`]).
    gene_props: Vec<Option<GeneProposal>>,
    path_props: Vec<Option<PathProposal>>,
    /// `[P]` every pseudobulk's jitter and their sd, `None` with no jitter.
    omega: Vec<f64>,
    share_sd: Option<f64>,
}

impl State {
    /// What the cells' readout needs of this state.
    fn snapshot(&self, rows: &[usize], lambda: &[Depth]) -> Snapshot {
        Snapshot {
            params: self.params.clone(),
            log_w: self.log_w.clone(),
            share_sd: self.share_sd,
            rows: rows.to_vec(),
            lambda: lambda.to_vec(),
        }
    }
}

/// What the readout needs of the climb's mode when no chain runs: every
/// pseudobulk at its most probable row under `params` and the lineages'
/// `log_w` (the jitter of its unspliced share integrated out at
/// `share_sd`), and its depths there.
pub fn mode_snapshot(
    reads: &Reads,
    params: &Params,
    log_w: &[f64],
    share_sd: Option<f64>,
) -> Snapshot {
    let lv = params.levels();
    let tab = match share_sd {
        Some(sd) => super::likelihood::integrated_table(reads, &lv, sd),
        None => super::likelihood::table(reads, &lv),
    };
    let n = params.positions();
    let rows: Vec<usize> = (0..tab.nrows())
        .into_par_iter()
        .map(|p| {
            let (lp, _) = super::blocks::log_posterior_row(&tab, log_w, n, p);
            (0..lp.len())
                .max_by(|&a, &b| lp[a].total_cmp(&lp[b]))
                .expect("a row")
        })
        .collect();
    let lambda = reads.depths(&rows, &lv);
    Snapshot {
        params: params.clone(),
        log_w: log_w.to_vec(),
        share_sd,
        rows,
        lambda,
    }
}

/// Every block's seed of its own, from the chain's.
fn block_seed(seed: u64, block: u64) -> u64 {
    seed.wrapping_mul(0x9e37_79b9_7f4a_7c15)
        .wrapping_add(block.wrapping_mul(0xd1b5_4a32_d192_ed03))
}

const LATENTS: u64 = 1;
const GENES: u64 = 2;
const PATHS: u64 = 3;
const DURATIONS: u64 = 4;
const WEIGHTS: u64 = 5;
const JITTERS: u64 = 6;
const JITTER_SD: u64 = 7;
const CAPTURE: u64 = 8;

fn start(params: &Params, grid: &GridPrior, pseudobulks: usize, share: Option<f64>) -> State {
    let lineages = params.path.len();
    let eta: Vec<Vec<DVector<f64>>> = (0..lineages).map(|l| path_eta(params, grid, l)).collect();
    let z_bar = params.path_mean();
    let modes: Vec<Vec<f64>> = (0..params.beta.len())
        .map(|g| {
            to_theta(
                params.beta[g],
                params.gamma[g],
                &params.rho[g],
                params.base[g],
                &z_bar,
            )
        })
        .collect();
    State {
        params: params.clone(),
        path_modes: eta.clone(),
        eta,
        log_w: vec![-(lineages as f64).ln(); lineages],
        gene_props: vec![None; modes.len()],
        path_props: vec![None; lineages],
        modes,
        omega: vec![0.0; pseudobulks],
        share_sd: share,
    }
}

/// One scalar elliptical slice move of lineage `l`'s log duration on its
/// `N(0, 1)` prior (as the climb's), the reads of the pseudobulks on `l` at
/// their rows and depths, the path on its grid held (`κ/T` fixed: only the
/// grid's step moves).
pub fn duration_step(
    params: &mut Params,
    l: usize,
    reads: &Reads,
    rows: &[usize],
    lambda: &[Depth],
    rng: &mut impl rand::Rng,
) {
    let probe = params.clone();
    let ll = |log_t: f64| {
        let mut p = probe.clone();
        p.t_len[l] = clamped_exp(log_t);
        path_ll(&p, l, reads, rows, lambda)
    };
    let x0 = probe.t_len[l].ln();
    let base = ll(x0);
    let e: f64 = StandardNormal.sample(rng);
    let (next, _) = elliptical_slice_step(
        &DVector::from_element(1, x0 as f32),
        &DVector::from_element(1, e as f32),
        &|y: &DVector<f32>| (ll(f64::from(y[0])) - base) as f32,
        0.0,
        rng,
    );
    params.t_len[l] = clamped_exp(f64::from(next[0]));
}

/// Whether sweep `it` refits the proposals: the first, every
/// [`Options::laplace_every`]-th of burn-in and its last; none after.
pub fn refit_at(opts: &Options, it: usize) -> bool {
    it == 0
        || it + 1 == opts.burn
        || (it < opts.burn && it.is_multiple_of(opts.laplace_every.max(1)))
}

/// One sweep of a chain; the drawn rows, the depths and the reads'
/// log-likelihood there.
#[allow(clippy::too_many_arguments)]
fn sweep(
    state: &mut State,
    reads: &Reads,
    grid: &GridPrior,
    prior: &GenePrior,
    opts: &Options,
    seed: u64,
    it: u64,
    refit: bool,
    secs: &mut [f64],
) -> (Vec<usize>, Vec<Depth>, f64) {
    let mut clock = std::time::Instant::now();
    let mut lap = |block: usize| {
        secs[block] += clock.elapsed().as_secs_f64();
        clock = std::time::Instant::now();
    };
    let lineages = state.params.path.len();
    // Rows are positions (a pseudobulk's time may sit between the nodes).
    let n = state.params.positions();
    // 1–2. Levels, latents (and jitters), the capture and depths.
    let mut lv = state.params.levels();
    let tab = match state.share_sd {
        Some(_) => jittered_table(reads, &lv, &state.omega),
        None => table(reads, &lv),
    };
    let rows = draw_latents(&tab, &state.log_w, n, block_seed(seed, LATENTS), it);
    lap(LATENTS_BLOCK);
    if let Some(sd) = state.share_sd {
        let jitter_seed = block_seed(seed, JITTERS);
        let z = lv.totals();
        state.omega = rows
            .par_iter()
            .zip(&state.omega)
            .enumerate()
            .map(|(p, (&r, &w))| {
                let mut rng = stream(jitter_seed, p as u64, it);
                let (z_u, z_s) = z[r];
                share_step(
                    reads.unspliced(p),
                    reads.total(p),
                    z_u,
                    z_s,
                    w,
                    sd,
                    &mut rng,
                )
            })
            .collect();
        state.share_sd = Some(draw_share_sd(&state.omega, block_seed(seed, JITTER_SD), it));
    }
    let c = state.params.log_capture;
    let z = lv.totals();
    let a: Vec<f64> = rows
        .iter()
        .zip(&state.omega)
        .map(|(&r, &w)| (z[r].0 / z[r].1).ln() - c + w)
        .collect();
    let n_u: Vec<f64> = (0..rows.len()).map(|p| reads.unspliced(p)).collect();
    let n_all: Vec<f64> = (0..rows.len()).map(|p| reads.total(p)).collect();
    let mut rng = stream(block_seed(seed, CAPTURE), 0, it);
    let next = capture_step(&n_u, &n_all, &a, c, &mut rng);
    lv.u *= clamped_exp(next - c);
    state.params.log_capture = next;
    let lambda = reads.jittered_depths(&rows, &lv, &state.omega);
    lap(JITTER_BLOCK);

    // 3. Genes, every other parameter held, each on its reads by row; the
    // proposal refitted (from the last one's mode) when the schedule says.
    let z_bar = state.params.path_mean();
    let layout = RowLayout::new(&rows, &lambda);
    let (params, starts) = (&state.params, &state.modes);
    let moved: Vec<Vec<f64>> = state
        .gene_props
        .par_iter_mut()
        .enumerate()
        .map(|(g, slot)| {
            let mut rng = stream(block_seed(seed, GENES), g as u64, it);
            let by_row = GeneRows::new(&layout, reads, g);
            if refit {
                let anchor = slot.as_ref().map_or(&starts[g], |p| &p.mode);
                *slot = Some(gene_mode(params, &by_row, &z_bar, prior, anchor));
            }
            let prop = slot.as_ref().expect("fitted at the first sweep");
            let mut theta = to_theta(
                params.beta[g],
                params.gamma[g],
                &params.rho[g],
                params.base[g],
                &z_bar,
            );
            let mut ll = f64::NEG_INFINITY;
            for _ in 0..opts.gene_moves {
                (theta, ll) = gene_step(params, &theta, ll, &by_row, &z_bar, prior, prop, &mut rng);
            }
            theta
        })
        .collect();
    for (g, theta) in moved.into_iter().enumerate() {
        let (beta, gamma, rho, base) = from_theta(&theta, &z_bar);
        state.params.beta[g] = beta;
        state.params.gamma[g] = gamma;
        state.params.rho[g] = rho.to_vec();
        state.params.base[g] = base;
    }
    lap(GENES_BLOCK);

    // 4. Paths, one lineage after another: generalized ESS on the Laplace
    // approximation (banded in the path), refitted as the genes' are.
    for l in 0..lineages {
        let mut rng = stream(block_seed(seed, PATHS), l as u64, it);
        if refit {
            let h = state.eta[l].len();
            let anchor = state.path_props[l]
                .as_ref()
                .map_or_else(|| state.path_modes[l].clone(), |p| p.mode(h));
            state.path_props[l] = Some(path_mode(
                &state.params,
                l,
                &anchor,
                grid,
                reads,
                &rows,
                &lambda,
                prior,
            ));
        }
        let prop = state.path_props[l]
            .as_ref()
            .expect("fitted at the first sweep");
        lap(PATH_LAPLACE);
        let mut cur = f64::NEG_INFINITY;
        for _ in 0..opts.path_moves {
            let (eta, ll) = path_gess_step(
                &mut state.params,
                l,
                &state.eta[l],
                cur,
                grid,
                reads,
                &rows,
                &lambda,
                prior,
                prop,
                &mut rng,
            );
            state.eta[l] = eta;
            cur = ll;
        }
        lap(PATH_MOVES);
    }

    // 5. Durations, lineage 0's the unit (on a trunk, one clock).
    let apart = state.params.branch.is_none();
    for l in (1..lineages).filter(|_| apart) {
        let mut rng = stream(block_seed(seed, DURATIONS), l as u64, it);
        duration_step(&mut state.params, l, reads, &rows, &lambda, &mut rng);
    }

    // 6. Weights.
    state.log_w = draw_weights(&rows, lineages, n, opts.a0, block_seed(seed, WEIGHTS), it);

    lap(DURATIONS_BLOCK);
    let total: f64 = (0..lineages)
        .map(|l| path_ll(&state.params, l, reads, &rows, &lambda))
        .sum();
    lap(LL_BLOCK);
    (rows, lambda, total)
}

/// `opts.chains` chains from `start` (the Adam mode, [`super::fit::Fit::params`]),
/// in parallel; every chain's kept draws.
pub fn run(reads: &Reads, start_at: &Params, opts: &Options) -> anyhow::Result<Vec<Draws>> {
    let n = start_at.positions();
    if opts.burn == 0 {
        log::warn!(
            "ode-mixture sampler: no burn-in, so the moves keep the proposals fitted at the start"
        );
    }
    let grid = GridPrior::new(start_at.path[0].len(), 1.0, opts.kappa)?;
    let prior = GenePrior::bounded(start_at);
    let bar = crate::logging::new_progress_bar((opts.chains * opts.sweeps) as u64)
        .with_message("ode-mixture sampler");
    let stopped = || {
        opts.stop
            .as_ref()
            .is_some_and(|s| s.load(Ordering::Relaxed))
    };
    let draws = (0..opts.chains)
        .into_par_iter()
        .map(|c| {
            let seed = block_seed(opts.seed, 1000 + c as u64);
            let mut state = start(start_at, &grid, reads.u.nrows(), opts.share);
            let kept = opts.sweeps.saturating_sub(opts.burn);
            let mut d = Draws {
                lineage: Vec::with_capacity(kept),
                time: Vec::with_capacity(kept),
                log_gamma: Vec::with_capacity(kept),
                log_ratio: Vec::with_capacity(kept),
                log_t: Vec::with_capacity(kept),
                path_ll: Vec::with_capacity(kept),
                share_sd: Vec::new(),
                omega: Vec::new(),
                log_capture: Vec::with_capacity(kept),
                branch: Vec::new(),
                snapshots: Vec::new(),
                block_secs: vec![0.0; BLOCKS.len()],
                refits: Vec::new(),
            };
            for it in 0..opts.sweeps {
                if stopped() {
                    break;
                }
                let refit = refit_at(opts, it);
                if refit {
                    d.refits.push(it);
                }
                let (rows, lambda, ll) = sweep(
                    &mut state,
                    reads,
                    &grid,
                    &prior,
                    opts,
                    seed,
                    it as u64,
                    refit,
                    &mut d.block_secs,
                );
                bar.inc(1);
                if it < opts.burn {
                    continue;
                }
                let p = &state.params;
                d.lineage.push(rows.iter().map(|r| r / n).collect());
                d.time
                    .push(rows.iter().map(|r| p.time_at(r / n, r % n)).collect());
                d.log_gamma.push(p.gamma.iter().map(|y| y.ln()).collect());
                d.log_ratio.push(
                    p.gamma
                        .iter()
                        .zip(&p.beta)
                        .map(|(y, b)| (y / b).ln())
                        .collect(),
                );
                d.log_t.push(p.t_len.iter().map(|t| t.ln()).collect());
                d.path_ll.push(ll);
                if let Some(b) = &state.params.branch {
                    d.branch.push(b.clone());
                }
                if opts.snapshot_every > 0
                    && (it - opts.burn + 1).is_multiple_of(opts.snapshot_every)
                {
                    d.snapshots.push(state.snapshot(&rows, &lambda));
                }
                d.log_capture.push(state.params.log_capture);
                if let Some(sd) = state.share_sd {
                    d.share_sd.push(sd);
                    d.omega.push(state.omega.clone());
                }
            }
            d
        })
        .collect::<Vec<_>>();
    bar.finish_and_clear();
    if stopped() {
        log::info!("ode-mixture sampler: stopped; the draws so far are kept");
    }
    Ok(draws)
}
