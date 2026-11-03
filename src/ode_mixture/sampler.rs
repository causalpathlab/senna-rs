//! The blocked Gibbs sampler, chains started at the posterior mode.
//!
//! One sweep, every block a move that leaves its conditional invariant:
//!
//! 0. every lineage's whitened path with the latents summed out
//!    ([`collapsed_path_step`] on [`collapsed_laplace`]): the path and every
//!    pseudobulk's time move together, which no conditional block can (a
//!    shared warp of time the reads barely see); the latents are drawn next;
//! 1. the levels on every lineage's grid ([`Params::levels`]);
//! 2. every pseudobulk's `(ℓ, k)` drawn exactly ([`draw_latents`]), its
//!    depth `λ_p = N_p / Z` at the drawn row;
//! 3. every gene: the Laplace approximation of its conditional by Newton
//!    from the previous sweep's mode (never the current `θ`), then
//!    [`gene_step`] a few times on it (one Laplace costs many moves); in
//!    parallel over genes, a stream per gene and sweep;
//! 4. every lineage's whitened path, one lineage after another (they meet
//!    in the genes' prior through the paths' mean): the Laplace
//!    approximation by Gauss–Newton from the previous sweep's mode
//!    ([`path_laplace`]), then generalized elliptical slice moves on it
//!    ([`path_gess_step`]; plain ESS on the prior gives a handful of
//!    effective draws in a thousand moves even from the mode);
//! 5. every other lineage's log duration `log T_ℓ` (lineage 0's is the
//!    unit) by a scalar elliptical slice move on its `N(0, 1)` prior, the
//!    path's whitened coordinates held. The length scale stays a fixed share
//!    of the duration (`κ/T` held, as in the climb): the path on its grid
//!    does not move with `T`, only the grid's step does;
//! 6. the lineages' weights ([`draw_weights`]).
//!
//! The parameters are kept natural, `(β, γ, ρ, b)`; a gene's sampler
//! coordinates ([`to_theta`]) are taken at the current paths' mean each
//! sweep. Chains run in parallel, each from the same start (the Adam mode)
//! with a seed of its own; only invariants are kept (the gauges, rotation of
//! `z` against `ρ` and the baseline's shared shift, do not show in them).

use super::blocks::{
    collapsed_laplace, collapsed_path_step, draw_latents, draw_weights, from_theta, gene_laplace,
    gene_step, path_gess_step, path_laplace, path_ll, stream, to_theta, GenePrior, PathProposal,
};
use super::kernel::GridPrior;
use super::likelihood::{table, Reads};
use super::model::Params;
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
    /// Moves per sweep of every path with the latents summed out (the path
    /// and every time together; off by default: it does not leave the
    /// climb's grid-step-late optimum, which runs through the genes too, and
    /// costs mixing elsewhere), on a proposal rebuilt every
    /// `collapsed_refresh` sweeps during burn-in and frozen after it (so the
    /// kept sweeps run one fixed kernel).
    pub collapsed_moves: usize,
    pub collapsed_refresh: usize,
    /// The weights' Dirichlet prior.
    pub a0: f64,
    pub seed: u64,
    /// Set (as Ctrl+C sets it, [`crate::topic::common::setup_stop_handler`])
    /// to end every chain after its current sweep, keeping what it drew.
    pub stop: Option<Arc<AtomicBool>>,
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
            collapsed_moves: 0,
            collapsed_refresh: 10,
            a0: 1.0,
            seed: 1,
            stop: None,
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
}

/// A chain's state.
struct State {
    params: Params,
    /// `[L][H]` the whitened paths.
    eta: Vec<Vec<DVector<f64>>>,
    log_w: Vec<f64>,
    /// `[G]` the previous sweep's Laplace modes, the next Newton's anchors;
    /// `[L][H]` the paths' likewise.
    modes: Vec<Vec<f64>>,
    path_modes: Vec<Vec<DVector<f64>>>,
    /// `[L]` every path's collapsed proposal, and the mode it was built at.
    collapsed: Vec<Option<PathProposal>>,
    collapsed_modes: Vec<Vec<DVector<f64>>>,
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
const COLLAPSED: u64 = 6;
/// Fisher-scoring steps per rebuild of a collapsed proposal (from the last).
const COLLAPSED_STEPS: usize = 5;

fn start(params: &Params, grid: &GridPrior) -> State {
    let lineages = params.path.len();
    let h = params.rho.first().map_or(0, Vec::len);
    let eta: Vec<Vec<DVector<f64>>> = params
        .path
        .iter()
        .map(|path| {
            (0..h)
                .map(|d| grid.whiten(&DVector::from_fn(path.len(), |k, _| path[k][d])))
                .collect()
        })
        .collect();
    let z_bar = params.path_mean();
    let modes = (0..params.beta.len())
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
        collapsed: vec![None; lineages],
        collapsed_modes: eta.clone(),
        eta,
        log_w: vec![-(lineages as f64).ln(); lineages],
        modes,
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
    lambda: &[f64],
    rng: &mut impl rand::Rng,
) {
    let probe = params.clone();
    let ll = |log_t: f64| {
        let mut p = probe.clone();
        p.t_len[l] = log_t.exp();
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
    params.t_len[l] = f64::from(next[0]).exp();
}

/// One sweep of a chain; the drawn rows and the reads' log-likelihood there.
fn sweep(
    state: &mut State,
    reads: &Reads,
    grid: &GridPrior,
    prior: &GenePrior,
    opts: &Options,
    seed: u64,
    it: u64,
) -> (Vec<usize>, f64) {
    let lineages = state.params.path.len();
    // Rows are positions (a pseudobulk's time may sit between the nodes).
    let n = state.params.positions();
    // 0. Every path with the latents summed out, so the path and every time
    // move together; the latents are drawn next, as a partially collapsed
    // sampler must (van Dyk and Park, *JASA* (2008)).
    for l in (0..lineages).filter(|_| opts.collapsed_moves > 0) {
        let mut rng = stream(block_seed(seed, COLLAPSED), l as u64, it);
        let refresh = (it as usize) < opts.burn
            && (it as usize).is_multiple_of(opts.collapsed_refresh.max(1));
        if refresh || state.collapsed[l].is_none() {
            let prop = collapsed_laplace(
                &state.params,
                l,
                &state.collapsed_modes[l],
                grid,
                reads,
                &state.log_w,
                prior,
                COLLAPSED_STEPS,
            );
            state.collapsed_modes[l] = prop.mode(state.eta[l].len());
            state.collapsed[l] = Some(prop);
        }
        let prop = state.collapsed[l].as_ref().expect("built above");
        let mut cur = f64::NEG_INFINITY;
        for _ in 0..opts.collapsed_moves {
            let (eta, ll) = collapsed_path_step(
                &mut state.params,
                l,
                &state.eta[l],
                cur,
                grid,
                reads,
                &state.log_w,
                prior,
                prop,
                &mut rng,
            );
            state.eta[l] = eta;
            cur = ll;
        }
    }

    // 1–2. Levels, latents and depths.
    let lv = state.params.levels();
    let rows = draw_latents(
        &table(reads, &lv),
        &state.log_w,
        n,
        block_seed(seed, LATENTS),
        it,
    );
    let lambda = reads.depths(&rows, &lv);

    // 3. Genes, every other parameter held.
    let z_bar = state.params.path_mean();
    let params = &state.params;
    let moved: Vec<(Vec<f64>, Vec<f64>)> = (0..params.beta.len())
        .into_par_iter()
        .map(|g| {
            let mut rng = stream(block_seed(seed, GENES), g as u64, it);
            let theta = to_theta(
                params.beta[g],
                params.gamma[g],
                &params.rho[g],
                params.base[g],
                &z_bar,
            );
            let prop = gene_laplace(params, g, reads, &rows, &lambda, prior, &state.modes[g]);
            let (mut theta, mut ll) = (theta, f64::NEG_INFINITY);
            for _ in 0..opts.gene_moves {
                (theta, ll) = gene_step(
                    params, g, &theta, ll, reads, &rows, &lambda, prior, &prop, &mut rng,
                );
            }
            (theta, prop.mode)
        })
        .collect();
    for (g, (theta, mode)) in moved.into_iter().enumerate() {
        let (beta, gamma, rho, base) = from_theta(&theta, &z_bar);
        state.params.beta[g] = beta;
        state.params.gamma[g] = gamma;
        state.params.rho[g] = rho.to_vec();
        state.params.base[g] = base;
        state.modes[g] = mode;
    }

    // 4. Paths, one lineage after another: the Laplace approximation by
    // Gauss–Newton from the previous sweep's mode, then generalized ESS.
    for l in 0..lineages {
        let mut rng = stream(block_seed(seed, PATHS), l as u64, it);
        let prop = path_laplace(
            &state.params,
            l,
            &state.path_modes[l],
            grid,
            reads,
            &rows,
            &lambda,
            prior,
        );
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
                &prop,
                &mut rng,
            );
            state.eta[l] = eta;
            cur = ll;
        }
        state.path_modes[l] = prop.mode(state.eta[l].len());
    }

    // 5. Durations, lineage 0's the unit.
    for l in 1..lineages {
        let mut rng = stream(block_seed(seed, DURATIONS), l as u64, it);
        duration_step(&mut state.params, l, reads, &rows, &lambda, &mut rng);
    }

    // 6. Weights.
    state.log_w = draw_weights(&rows, lineages, n, opts.a0, block_seed(seed, WEIGHTS), it);

    let total: f64 = (0..lineages)
        .map(|l| path_ll(&state.params, l, reads, &rows, &lambda))
        .sum();
    (rows, total)
}

/// `opts.chains` chains from `start` (the Adam mode, [`super::fit::Fit::params`]),
/// in parallel; every chain's kept draws.
pub fn run(reads: &Reads, start_at: &Params, opts: &Options) -> anyhow::Result<Vec<Draws>> {
    let n = start_at.positions();
    let grid = GridPrior::new(start_at.path[0].len(), 1.0, opts.kappa)?;
    let prior = GenePrior::default();
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
            let mut state = start(start_at, &grid);
            let kept = opts.sweeps.saturating_sub(opts.burn);
            let mut d = Draws {
                lineage: Vec::with_capacity(kept),
                time: Vec::with_capacity(kept),
                log_gamma: Vec::with_capacity(kept),
                log_ratio: Vec::with_capacity(kept),
                log_t: Vec::with_capacity(kept),
                path_ll: Vec::with_capacity(kept),
            };
            for it in 0..opts.sweeps {
                if stopped() {
                    break;
                }
                let (rows, ll) = sweep(&mut state, reads, &grid, &prior, opts, seed, it as u64);
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
