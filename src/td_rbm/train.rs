//! The fit, on pseudobulk samples: every epoch, every level's pseudobulks
//! (coarse to fine) are drawn again at a cell's depth ([`resample`]).
//!
//! 1. **Lineages and time**, by the exact likelihood (with the states off the
//!    machine is a mixture of multinomials): every pseudobulk sample is
//!    assigned a lineage by collapsed Gibbs under a truncated Dirichlet
//!    process — `p(ℓ) ∝ (n_ℓ + α/L) · (1/K) Σ_k e^{v·ψ_{ℓ,k}}`, `n_ℓ` the cells
//!    the other finest pseudobulks hold there, rich get richer — and every
//!    lineage's curve is fitted on its rows, time summed out. Lineages left
//!    empty are dropped from the weights.
//! 2. **Direction**, per lineage: the lineage run the other way (exactly,
//!    [`Energy::mirror`]), the shared rates refitted, the two orientations'
//!    exact log-likelihoods compared; the likelier kept. `Λ_ℓ`, their gap, is
//!    the log odds of the kept direction.
//! 3. **States**, by persistent contrastive divergence: the positive phase the
//!    rows' free energy, the negative phase that of persistent Gibbs chains
//!    ([`Chains`]).
//!
//! A starting order and lineage split only set where the curves start
//! ([`super::start::curve_start`]).

use super::curve::{grid_times, Spline};
use super::model::{Energy, Init0, Shape, Tables};
use super::rbm::{free_energy, lineage_log_likelihood, posterior, Chains, Reads};
use super::start::{resample, STATE_DIMS};
use legume_numeric::candle::candle_core::{Device, Result, Tensor, D};
use legume_numeric::candle::candle_nn::{AdamW, Optimizer, ParamsAdamW};
use log::info;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::{RngExt, SeedableRng};
use rand_distr::{Binomial, Distribution};

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
    /// Epochs fitting each lineage both ways round, from that steady fit, to
    /// choose its direction.
    pub orient_epochs: usize,
    /// Bootstrap resamples of a lineage's pseudobulks behind its direction
    /// call (and half as many draws without splicing dynamics).
    pub orient_draws: usize,
    /// The depth a pseudobulk sample's lineage is chosen at: its
    /// log-likelihood per read times this, so the Dirichlet process's
    /// rich-get-richer prior weighs in between lineages that explain it about
    /// equally well.
    pub assign_depth: f64,
    /// Reads every pseudobulk sample (and every chain) holds.
    pub depth: u64,
    pub batch: usize,
    /// Neighbours per pseudobulk in a lineage's diffusion order.
    pub knn: usize,
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
            orient_epochs: 20,
            orient_draws: 40,
            assign_depth: 300.0,
            depth: 1000,
            knn: 10,
            batch: 128,
            chains: 256,
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

/// What a lineage's direction rests on.
#[derive(Clone, Debug)]
pub struct Direction {
    pub lineage: usize,
    /// Cells the lineage holds (finest pseudobulks' cells).
    pub cells: f64,
    /// `Λ`: the forward fit's which-track log-likelihood over the reversed
    /// one's, in nats over the lineage's pooled reads, forward meaning the
    /// orientation the lineage was started in; and per read.
    pub lambda: f64,
    pub lambda_per_read: f64,
    /// The mean of `Λ` over draws with no splicing dynamics (every gene's
    /// unspliced share constant in time), the bias `Λ` is measured from.
    pub lambda_null: f64,
    /// The sd of `Λ` over bootstrap resamples of the lineage's pseudobulks.
    pub lambda_sd: f64,
    /// `(Λ − null) / sd`; the call is made when `|z|` and `|Λ| / sd` both
    /// pass `Z_CALL`, with the same sign.
    pub z: f64,
    /// Whether the data decide the direction; when not, it stays as started.
    pub determined: bool,
    /// Whether the lineage now runs against the curve it was started on.
    pub reversed: bool,
    /// `[G]` every gene's share of `Λ`, signed for the kept orientation.
    pub gene_votes: Vec<f32>,
}

/// A fitted machine.
pub struct Model {
    pub energy: Energy,
    pub spline: Spline,
    /// `[L]` every lineage's log weight (`−∞` where empty).
    pub log_weight: Vec<f32>,
    pub directions: Vec<Direction>,
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
/// A lineage's log weight when it is empty.
const EMPTY: f32 = -1e4;

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

fn adam(vars: Vec<legume_numeric::candle::candle_core::Var>, lr: f64) -> Result<AdamW> {
    AdamW::new(
        vars,
        ParamsAdamW {
            lr,
            weight_decay: 0.0,
            ..Default::default()
        },
    )
}

/// An index drawn with log weights `w`.
fn draw(w: &[f64], rng: &mut StdRng) -> usize {
    let top = w.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let p: Vec<f64> = w.iter().map(|x| (x - top).exp()).collect();
    let mut u = rng.random::<f64>() * p.iter().sum::<f64>();
    for (i, &pi) in p.iter().enumerate() {
        if u < pi {
            return i;
        }
        u -= pi;
    }
    p.len() - 1
}

/// The rows' summed log-likelihood under their assigned lineages, `[]`,
/// with the graph: `ll` `[N × L]`, `assigned` `[N]`.
fn assigned_log_likelihood(ll: &Tensor, assigned: &[usize]) -> Result<Tensor> {
    let (n, l) = ll.dims2()?;
    let mut mask = vec![0f32; n * l];
    for (r, &a) in assigned.iter().enumerate() {
        mask[r * l + a] = 1.0;
    }
    (ll * Tensor::from_vec(mask, (n, l), ll.device())?)?.sum_all()
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

/// Fit the machine on pseudobulks of every level (`levels`, coarse to fine),
/// the genes' embedding, baselines and curves starting at `init`, the finest
/// pseudobulks starting in lineages `start_lineage`.
pub fn fit(
    levels: &[Level],
    f: usize,
    init: &Init0,
    start_lineage: &[usize],
    cfg: &TrainConfig,
    dev: &Device,
) -> anyhow::Result<Model> {
    anyhow::ensure!(!levels.is_empty(), "no level to fit");
    anyhow::ensure!(f.is_multiple_of(2), "both tracks of every gene");
    let l = cfg.lineages;
    let finest = levels.len() - 1;
    anyhow::ensure!(
        start_lineage.len() == levels[finest].cells.len(),
        "a starting lineage per finest pseudobulk"
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

    // Every finest pseudobulk's lineage and the cells every lineage holds,
    // from the start; the coarser levels' rows follow them.
    let mut assigned: Vec<Vec<usize>> = levels
        .iter()
        .enumerate()
        .map(|(i, lv)| {
            if i == finest {
                start_lineage.to_vec()
            } else {
                vec![0; lv.cells.len()]
            }
        })
        .collect();
    let mut counts = vec![0f64; l];
    for (&a, &c) in assigned[finest].iter().zip(&levels[finest].cells) {
        counts[a] += f64::from(c);
    }

    // 0. Lineages and curves at steady state (the velocity off): every
    // timeline's shape and membership, blind to its direction, so lineages
    // that start as pieces of one trajectory merge before any piece is
    // oriented. 1. Every lineage's direction, from there. 2. Lineages and
    // time with the velocity on.
    let mut phase = Phase {
        energy: &energy,
        spline: &spline,
        levels,
        f,
        cfg,
        dev,
        assigned: &mut assigned,
        counts: &mut counts,
        alive: vec![true; l],
    };
    energy.set_motion(false);
    phase.epochs(cfg.steady_epochs, "steady", &mut rng)?;
    // Two halves of one path become one lineage (branch points stay). Then
    // every lineage's curve starts again from a diffusion order of the
    // pseudobulks it now holds (a lineage that grew by taking in others
    // still has the curve of the piece it began as), and settles.
    phase.stitch()?;
    phase.reseed()?;
    phase.epochs(cfg.steady_epochs, "steady (reseeded)", &mut rng)?;
    energy.set_motion(true);
    let mut directions = orient_all(
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
    for d in &mut directions {
        d.cells = counts[d.lineage];
    }
    let log_weight = log_weights(&counts, cfg.concentration);

    // 3. States.
    if cfg.epochs > cfg.time_epochs {
        // Everything, the states with it: with the states taking up what lies
        // off every trajectory, the curves and rates refit cleaner (training
        // the states alone leaves the velocity markedly worse in simulation).
        let mut vars = energy.vars();
        vars.push(energy.load_var());
        let mut opt = adam(vars, cfg.lr)?;
        let p = levels[finest].cells.len();
        let drawn = resample(&levels[finest].x, f, cfg.depth, &mut rng);
        let x0: Vec<f32> = (0..cfg.chains)
            .flat_map(|j| drawn[(j % p) * f..(j % p + 1) * f].to_vec())
            .collect();
        let mut chains = Chains::new(x0, f, cfg.seed.wrapping_add(1));
        for epoch in cfg.time_epochs..cfg.epochs {
            let (mut gap, mut steps) = (0f64, 0usize);
            for lv in levels {
                let rows = lv.cells.len();
                let all = Reads::from_features(Tensor::from_vec(
                    resample(&lv.x, f, cfg.depth, &mut rng),
                    (rows, f),
                    dev,
                )?)?;
                let mut order: Vec<u32> = (0..rows as u32).collect();
                order.shuffle(&mut rng);
                for chunk in order.chunks(cfg.batch) {
                    let batch = all.select(&Tensor::from_slice(chunk, chunk.len(), dev)?)?;
                    let t = energy.tables(&spline, true, &log_weight)?;
                    let fantasy = chains.reads(dev)?;
                    let loss = ((free_energy(&batch, &t)?.mean_all()?
                        - free_energy(&fantasy, &t)?.mean_all()?)?
                        / cfg.depth as f64)?;
                    opt.backward_step(&loss)?;
                    gap += f64::from(loss.to_scalar::<f32>()?);
                    steps += 1;
                    let after = energy.tables(&spline, true, &log_weight)?.detach();
                    for _ in 0..cfg.sweeps {
                        chains.sweep(&after)?;
                    }
                }
            }
            if epoch % 10 == 0 || epoch + 1 == cfg.epochs {
                info!(
                    "td-rbm epoch {epoch} (states): free-energy gap per read {:.5}",
                    gap / steps.max(1) as f64
                );
            }
        }
    }
    Ok(Model {
        energy,
        spline,
        log_weight,
        directions,
    })
}

/// The lineage-and-time loop shared by the steady and the moving phases.
struct Phase<'a> {
    energy: &'a Energy,
    spline: &'a Spline,
    levels: &'a [Level],
    f: usize,
    cfg: &'a TrainConfig,
    dev: &'a Device,
    /// Every level's rows' lineages, and the cells every lineage holds (the
    /// finest level's).
    assigned: &'a mut Vec<Vec<usize>>,
    counts: &'a mut Vec<f64>,
    /// Lineages still in play (a merged-away lineage is retired).
    alive: Vec<bool>,
}

impl Phase<'_> {
    /// `epochs` epochs: every level drawn again, every row's lineage chosen by
    /// collapsed Gibbs under the truncated Dirichlet process (the finest
    /// level's rows move the counts; rich get richer), every lineage's curve
    /// fitted on its rows by the exact likelihood, time summed out.
    fn epochs(&mut self, epochs: usize, label: &str, rng: &mut StdRng) -> anyhow::Result<()> {
        let (cfg, f, dev) = (self.cfg, self.f, self.dev);
        let l = cfg.lineages;
        let finest = self.levels.len() - 1;
        let prior = cfg.concentration / l as f64;
        let temper = cfg.assign_depth / cfg.depth as f64;
        let zero = vec![0f32; l];
        let mut opt = adam(self.energy.vars(), cfg.lr)?;
        for epoch in 0..epochs {
            let (mut nll, mut reads) = (0f64, 0f64);
            let drawn: Vec<Reads> = self
                .levels
                .iter()
                .map(|lv| {
                    let rows = lv.cells.len();
                    Reads::from_features(Tensor::from_vec(
                        resample(&lv.x, f, cfg.depth, rng),
                        (rows, f),
                        dev,
                    )?)
                })
                .collect::<Result<_>>()?;
            let t = self.energy.tables(self.spline, false, &zero)?.detach();
            for (i, r) in drawn.iter().enumerate() {
                let ll = lineage_log_likelihood(r, &t)?.to_vec2::<f32>()?;
                let mut order: Vec<usize> = (0..ll.len()).collect();
                order.shuffle(rng);
                for row in order {
                    let cells = f64::from(self.levels[i].cells[row]);
                    if i == finest {
                        self.counts[self.assigned[i][row]] -= cells;
                    }
                    let w: Vec<f64> = (0..l)
                        .map(|j| {
                            if self.alive[j] {
                                (self.counts[j].max(0.0) + prior).ln()
                                    + f64::from(ll[row][j]) * temper
                            } else {
                                f64::NEG_INFINITY
                            }
                        })
                        .collect();
                    let a = draw(&w, rng);
                    self.assigned[i][row] = a;
                    if i == finest {
                        self.counts[a] += cells;
                    }
                }
            }
            for (i, r) in drawn.iter().enumerate() {
                let mut order: Vec<u32> = (0..r.x.dim(0)? as u32).collect();
                order.shuffle(rng);
                for chunk in order.chunks(cfg.batch) {
                    let batch = r.select(&Tensor::from_slice(chunk, chunk.len(), dev)?)?;
                    let ids: Vec<usize> = chunk
                        .iter()
                        .map(|&c| self.assigned[i][c as usize])
                        .collect();
                    let t = self.energy.tables(self.spline, false, &zero)?;
                    let ll = assigned_log_likelihood(&lineage_log_likelihood(&batch, &t)?, &ids)?;
                    let loss = (ll.neg()? / (chunk.len() as f64 * cfg.depth as f64))?;
                    opt.backward_step(&loss)?;
                    nll += f64::from(loss.to_scalar::<f32>()?) * chunk.len() as f64;
                    reads += chunk.len() as f64;
                }
            }
            if epoch % 10 == 0 || epoch + 1 == epochs {
                let used: Vec<String> = self
                    .counts
                    .iter()
                    .enumerate()
                    .filter(|(_, &n)| n > 0.0)
                    .map(|(j, n)| format!("L{}:{:.0}", j + 1, n))
                    .collect();
                info!(
                    "td-rbm {label} epoch {epoch}: negative log-likelihood per read {:.5}; lineages (cells) {}",
                    nll / reads.max(1.0),
                    used.join(" ")
                );
            }
        }
        Ok(())
    }

    /// Every finest pseudobulk's posterior mean time within its lineage,
    /// at steady state, read at the training depth.
    fn times_within(&self) -> anyhow::Result<Vec<f32>> {
        let finest = self.levels.len() - 1;
        let (f, k) = (self.f, self.cfg.grid);
        let p = self.levels[finest].cells.len();
        let mut fixed = self.levels[finest].x.clone();
        super::start::scale_rows(&mut fixed, f, self.cfg.depth as f32);
        let eval = Reads::from_features(Tensor::from_vec(fixed, (p, f), self.dev)?)?;
        let zero = vec![0f32; self.cfg.lineages];
        let t = self.energy.tables(self.spline, false, &zero)?.detach();
        let logits = eval.x.matmul(&t.psi.t()?)?.to_vec2::<f32>()?;
        let times = grid_times(k);
        Ok((0..p)
            .map(|r| {
                let a = self.assigned[finest][r];
                let row = &logits[r][a * k..(a + 1) * k];
                let top = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let w: Vec<f32> = row.iter().map(|v| (v - top).exp()).collect();
                let z: f32 = w.iter().sum();
                w.iter().zip(&times).map(|(w, t)| w * t).sum::<f32>() / z
            })
            .collect())
    }

    /// Every live lineage's curve from a fresh diffusion order of the finest
    /// pseudobulks it holds (fitted by least squares, [`super::start::curve_start`]).
    fn reseed(&mut self) -> anyhow::Result<()> {
        let (cfg, f) = (self.cfg, self.f);
        let finest = self.levels.len() - 1;
        let x = &self.levels[finest].x;
        let p = self.levels[finest].cells.len();
        let g = f / 2;
        let (k, m) = (cfg.grid, cfg.knots);
        let h = self.energy.rho().dim(1)?;
        let values = self.spline.values.flatten_all()?.to_vec1::<f32>()?;
        let rho = self.energy.rho().flatten_all()?.to_vec1::<f32>()?;
        let tau = self.times_within()?;
        for l in 0..cfg.lineages {
            if !self.alive[l] || self.counts[l] <= 0.0 {
                continue;
            }
            let rows: Vec<usize> = (0..p).filter(|&r| self.assigned[finest][r] == l).collect();
            if rows.len() < 3 {
                continue;
            }
            let ux: Vec<f32> = rows
                .iter()
                .flat_map(|&r| x[r * f..(r + 1) * f].to_vec())
                .collect();
            let mut order = super::init::diffusion_order(
                &super::start::row_states(&ux, f, STATE_DIMS),
                cfg.knn.min(rows.len() - 1),
            );
            // A diffusion order's sign is arbitrary: keep the lineage's
            // orientation so far.
            let now: Vec<f32> = rows.iter().map(|&r| tau[r]).collect();
            if super::start::rank_correlation(&order, &now) < 0.0 {
                order.iter_mut().for_each(|t| *t = 1.0 - *t);
            }
            let tot: Vec<f32> = ux
                .chunks(f)
                .flat_map(|r| (0..g).map(|j| r[j] + r[g + j]).collect::<Vec<_>>())
                .collect();
            let curve = super::start::curve_start(&tot, &order, &rho, &values, (g, h, k, m));
            self.energy.set_lineage_curve(l, &curve)?;
        }
        Ok(())
    }

    /// Join lineages that are two halves of one path. Every live lineage's
    /// two ends are the expression centroids (`log1p` reads per 10⁴, both
    /// tracks summed) of its earliest and latest [`END_SHARE`] of finest
    /// pseudobulks by their posterior time; two ends meet when their
    /// correlation distance is under [`MEET`] times the typical step between
    /// neighbouring slices within a lineage and the finest pseudobulks'
    /// neighbour graph joins them (so an island never stitches). A joint (ends that meet,
    /// transitively) of exactly two ends of two lineages is a path cut in two:
    /// those lineages become one curve, fitted through their pseudobulks in
    /// the concatenated order (each turned so the meeting ends sit at
    /// `τ = ½`). Three or more ends meeting are a branch point and stay
    /// apart. Repeats until no such joint is left; whether anything stitched.
    fn stitch(&mut self) -> anyhow::Result<bool> {
        let (cfg, f) = (self.cfg, self.f);
        let finest = self.levels.len() - 1;
        let x = &self.levels[finest].x;
        let p = self.levels[finest].cells.len();
        let g = f / 2;
        let (k, m) = (cfg.grid, cfg.knots);
        let h = self.energy.rho().dim(1)?;
        let values = self.spline.values.flatten_all()?.to_vec1::<f32>()?;
        // Every finest pseudobulk's log expression per 10⁴.
        let logx: Vec<Vec<f32>> = x
            .chunks(f)
            .map(|r| {
                let tot: f32 = r.iter().sum::<f32>().max(1.0);
                (0..g)
                    .map(|j| ((r[j] + r[g + j]) / tot * super::start::SCALE).ln_1p())
                    .collect()
            })
            .collect();
        let centroid = |rows: &[usize]| -> Vec<f32> {
            let mut c = vec![0f32; g];
            for &r in rows {
                c.iter_mut().zip(&logx[r]).for_each(|(a, b)| *a += b);
            }
            c.iter_mut().for_each(|v| *v /= rows.len().max(1) as f32);
            c
        };
        let distance = |a: &[f32], b: &[f32]| -> f64 {
            let n = a.len() as f64;
            let (ma, mb) = (
                a.iter().map(|&v| f64::from(v)).sum::<f64>() / n,
                b.iter().map(|&v| f64::from(v)).sum::<f64>() / n,
            );
            let (mut c, mut va, mut vb) = (0.0, 0.0, 0.0);
            for (&x, &y) in a.iter().zip(b) {
                let (x, y) = (f64::from(x) - ma, f64::from(y) - mb);
                c += x * y;
                va += x * x;
                vb += y * y;
            }
            1.0 - c / (va * vb).sqrt().max(1e-12)
        };
        // The finest pseudobulks' neighbour graph: two ends meet only where
        // it joins them (an island never stitches onto anything).
        let edges: std::collections::HashSet<(usize, usize)> = super::init::knn_edges(
            &super::start::row_states(x, f, STATE_DIMS),
            STITCH_KNN.min(p.saturating_sub(1)).max(1),
        )
        .into_iter()
        .flat_map(|(a, b, _)| [(a, b), (b, a)])
        .collect();
        let mut stitched = false;
        loop {
            let live: Vec<usize> = (0..cfg.lineages)
                .filter(|&j| self.alive[j] && self.counts[j] > 0.0)
                .collect();
            if live.len() < 2 {
                break;
            }
            let tau = self.times_within()?;
            // Ends and the within-lineage step.
            let mut ends: Vec<(usize, bool, Vec<f32>, Vec<usize>)> = Vec::new(); // (lineage, late end?, centroid, rows)
            let mut steps = Vec::new();
            for &l in &live {
                let mut rows: Vec<usize> =
                    (0..p).filter(|&r| self.assigned[finest][r] == l).collect();
                if rows.len() < 4 {
                    continue;
                }
                rows.sort_by(|&a, &b| tau[a].total_cmp(&tau[b]));
                let n = ((rows.len() as f64 * END_SHARE).ceil() as usize).max(1);
                let first = centroid(&rows[..n]);
                let second = centroid(&rows[n..(2 * n).min(rows.len())]);
                let last = centroid(&rows[rows.len() - n..]);
                steps.push(distance(&first, &second));
                ends.push((l, false, first, rows[..n].to_vec()));
                ends.push((l, true, last, rows[rows.len() - n..].to_vec()));
            }
            if ends.len() < 4 || steps.is_empty() {
                break;
            }
            steps.sort_by(f64::total_cmp);
            let threshold = MEET * steps[steps.len() / 2];
            // Joints: ends of different lineages that meet, transitively.
            let e = ends.len();
            let mut parent: Vec<usize> = (0..e).collect();
            fn root(p: &mut [usize], i: usize) -> usize {
                let mut i = i;
                while p[i] != i {
                    p[i] = p[p[i]];
                    i = p[i];
                }
                i
            }
            for i in 0..e {
                for j in i + 1..e {
                    let joined = ends[i]
                        .3
                        .iter()
                        .any(|&a| ends[j].3.iter().any(|&b| edges.contains(&(a, b))));
                    if ends[i].0 != ends[j].0
                        && joined
                        && distance(&ends[i].2, &ends[j].2) < threshold
                    {
                        let (a, b) = (root(&mut parent, i), root(&mut parent, j));
                        parent[a] = b;
                    }
                }
            }
            let mut joints: std::collections::BTreeMap<usize, Vec<usize>> =
                std::collections::BTreeMap::new();
            for i in 0..e {
                joints.entry(root(&mut parent, i)).or_default().push(i);
            }
            let pair = joints
                .values()
                .find(|j| j.len() == 2 && ends[j[0]].0 != ends[j[1]].0)
                .cloned();
            let Some(pair) = pair else { break };
            let ((a, a_late, ..), (b, b_late, ..)) = (ends[pair[0]].clone(), ends[pair[1]].clone());
            // One path: a's free end at τ = 0, the joint at ½, b's free end at 1.
            let mut rows = Vec::new();
            let mut order = Vec::new();
            for (r, &tr) in tau.iter().enumerate() {
                let l = self.assigned[finest][r];
                if l == a {
                    let t = if a_late { tr } else { 1.0 - tr };
                    rows.push(r);
                    order.push(0.5 * t);
                } else if l == b {
                    let t = if b_late { 1.0 - tr } else { tr };
                    rows.push(r);
                    order.push(0.5 + 0.5 * t);
                }
            }
            // The joined lineage keeps a's orientation.
            if !a_late {
                order.iter_mut().for_each(|t| *t = 1.0 - *t);
            }
            let tot: Vec<f32> = rows
                .iter()
                .flat_map(|&r| (0..g).map(move |j| x[r * f + j] + x[r * f + g + j]))
                .collect();
            let rho = self.energy.rho().flatten_all()?.to_vec1::<f32>()?;
            let curve = super::start::curve_start(&tot, &order, &rho, &values, (g, h, k, m));
            self.energy.set_lineage_curve(a, &curve)?;
            for i in 0..self.levels.len() {
                for v in self.assigned[i].iter_mut() {
                    if *v == b {
                        *v = a;
                    }
                }
            }
            self.counts[a] += self.counts[b];
            self.counts[b] = 0.0;
            self.alive[b] = false;
            stitched = true;
            info!(
                "td-rbm stitched L{} onto L{}: one path, met at a joint of two ends",
                b + 1,
                a + 1
            );
            #[cfg(test)]
            eprintln!("stitched L{} onto L{}", b + 1, a + 1);
        }
        Ok(stitched)
    }
}

/// Share of a lineage's pseudobulks (by time) that make up each of its ends.
const END_SHARE: f64 = 0.15;
/// Ends meet within this many typical within-lineage steps.
const MEET: f64 = 1.5;
/// Neighbours per finest pseudobulk in the graph that must join two ends.
const STITCH_KNN: usize = 10;

/// A direction is called when `Λ` sits this many bootstrap sds both from its
/// null and from zero, on the same side of each.
const Z_CALL: f64 = 2.0;

/// Every lineage's direction from the splicing alone ([`orient`]).
#[allow(clippy::too_many_arguments)]
fn orient_all(
    energy: &Energy,
    spline: &Spline,
    finest: &[f32],
    f: usize,
    lineage: &[usize],
    cfg: &TrainConfig,
    dev: &Device,
    rng: &mut StdRng,
) -> anyhow::Result<Vec<Direction>> {
    let mut out = Vec::new();
    let mut rate_sum: Option<(Tensor, Tensor, f64)> = None;
    for lin in 0..cfg.lineages {
        let rows: Vec<usize> = (0..lineage.len()).filter(|&r| lineage[r] == lin).collect();
        if rows.len() < 2 {
            continue;
        }
        let x: Vec<f32> = rows
            .iter()
            .flat_map(|&r| finest[r * f..(r + 1) * f].to_vec())
            .collect();
        let (d, (lb, lg)) = orient(energy, spline, &x, f, lin, cfg, dev, rng)?;
        let w = rows.len() as f64;
        rate_sum = Some(match rate_sum {
            None => ((&lb * w)?, (&lg * w)?, w),
            Some((sb, sg, sw)) => ((sb + (&lb * w)?)?, (sg + (&lg * w)?)?, sw + w),
        });
        info!(
            "td-rbm lineage L{} ({} pseudobulks): {} — Λ {:.1} nats, {:.1} with no splicing dynamics, bootstrap sd {:.1}, z {:+.2}; genes for {} against {}",
            lin + 1,
            rows.len(),
            match (d.determined, d.reversed) {
                (false, _) => "direction undetermined, kept as started",
                (true, false) => "direction as started",
                (true, true) => "direction reversed",
            },
            d.lambda,
            d.lambda_null,
            d.lambda_sd,
            d.z,
            d.gene_votes.iter().filter(|&&v| v > 0.0).count(),
            d.gene_votes.iter().filter(|&&v| v < 0.0).count(),
        );
        out.push(d);
    }
    // The model carries on with every gene's rates as fitted for the kept
    // directions, averaged over the lineages by their pseudobulks.
    if let Some((sb, sg, sw)) = rate_sum {
        energy.set_log_rates(&(sb / sw)?, &(sg / sw)?)?;
    }
    Ok(out)
}

/// Lineage `lin`'s direction from the splicing alone. Every pseudobulk of
/// `x` (`[P × F]` counts) keeps its time posterior `q` under the lineage at
/// steady state (the velocity off, so blind to direction); with those weights
/// every gene's unspliced and spliced reads are pooled per grid time, `U_kg`,
/// `S_kg`; then, for the velocity read forward and reversed (`±x_kg`, the
/// curve's `d log s/dτ`), every gene's `β_g, γ_g` are fitted to its
/// which-track log-likelihood `Σ_k U log σ(η) + S log σ(−η)`,
/// `η = log m⁺ − log β`, `m = γ ± x`. `Λ_g` is the gene's forward minus
/// reversed best fit; `Λ` their sum.
///
/// `Λ` is not centred at zero without dynamics (the floor and the curve's
/// shape favour one sign), so it is read against draws of `U` at every gene's
/// overall unspliced share (no dynamics, the same totals), and its spread
/// comes from bootstrap resamples of the pseudobulks (`q` with them). All the
/// draws are fitted together. The lineage is mirrored only when the call is
/// made and goes against the start. Returns the call and the rates fitted for
/// the kept orientation.
#[allow(clippy::too_many_arguments)]
fn orient(
    energy: &Energy,
    spline: &Spline,
    x: &[f32],
    f: usize,
    lin: usize,
    cfg: &TrainConfig,
    dev: &Device,
    rng: &mut StdRng,
) -> anyhow::Result<(Direction, (Tensor, Tensor))> {
    let g = f / 2;
    let p = x.len() / f;
    let l = cfg.lineages;
    let mut fixed = x.to_vec();
    super::start::scale_rows(&mut fixed, f, cfg.depth as f32);
    let rows = Reads::from_features(Tensor::from_vec(fixed, (p, f), dev)?)?;
    // Time posteriors at steady state, under this lineage alone.
    energy.set_motion(false);
    let t = energy.tables(spline, false, &vec![0f32; l])?.detach();
    energy.set_motion(true);
    let k = t.grid_len()?;
    let logits = rows.x.matmul(&t.psi.narrow(0, lin * k, k)?.t()?)?; // [P × K]
    let q = legume_numeric::candle::candle_nn::ops::softmax(&logits, D::Minus1)?;

    // Replicate 0 is the data; then the bootstrap; then the null.
    let boot = cfg.orient_draws;
    let nulls = cfg.orient_draws.div_ceil(2);
    let mut pooled = vec![q.t()?.matmul(&rows.x)?]; // [K × F]
    for _ in 0..boot {
        let pick: Vec<u32> = (0..p).map(|_| rng.random_range(0..p as u32)).collect();
        let pick = Tensor::from_vec(pick, p, dev)?;
        pooled.push(
            q.index_select(&pick, 0)?
                .t()?
                .matmul(&rows.x.index_select(&pick, 0)?)?,
        );
    }
    let data = pooled[0].flatten_all()?.to_vec1::<f32>()?;
    let mut share = vec![0f64; g];
    for j in 0..g {
        let (u, n) = (0..k).fold((0f64, 0f64), |(u, n), kk| {
            let (a, b) = (f64::from(data[kk * f + j]), f64::from(data[kk * f + g + j]));
            (u + a, n + a + b)
        });
        share[j] = u / n.max(1e-12);
    }
    for _ in 0..nulls {
        let mut draw = data.clone();
        for kk in 0..k {
            for j in 0..g {
                let n = f64::from(data[kk * f + j] + data[kk * f + g + j]);
                let whole = n.round() as u64;
                let u = if whole == 0 {
                    n * share[j]
                } else {
                    let b =
                        Binomial::new(whole, share[j].clamp(0.0, 1.0)).map_or(0, |b| b.sample(rng));
                    n * b as f64 / whole as f64
                };
                draw[kk * f + j] = u as f32;
                draw[kk * f + g + j] = (n - u) as f32;
            }
        }
        pooled.push(Tensor::from_vec(draw, (k, f), dev)?);
    }
    let pooled = Tensor::stack(&pooled, 0)?; // [R × K × F]
    let (u, s) = (pooled.narrow(2, 0, g)?, pooled.narrow(2, g, g)?);

    let readout = energy.velocity_readout(spline, lin)?.detach(); // [K × G]
                                                                  // Every gene starts with γ above its largest |x|, so neither sign starts
                                                                  // on the floor, and β at its overall unspliced share: u/s = γ/β.
    let gamma0 = (readout.abs()?.max_keepdim(0)? * 2.0)?
        .affine(1.0, 0.1)?
        .log()?
        .unsqueeze(0)?; // [1 × 1 × G]
    let (u0, s0) = (u.narrow(0, 0, 1)?, s.narrow(0, 0, 1)?);
    let ratio = ((s0.sum_keepdim(1)? + 0.5)?.log()? - (u0.sum_keepdim(1)? + 0.5)?.log()?)?; // log s/u
    let beta0 = (ratio + &gamma0)?;
    let r = u.dim(0)?;
    let steps = cfg.orient_epochs.max(1) * 100;
    // The data fitted alone from the start; then every replicate (the data
    // too, so all are fitted alike) from the data's optimum, briefly.
    let best = |sign: f64| -> anyhow::Result<(Tensor, (Tensor, Tensor))> {
        let xs = (&readout * sign)?.unsqueeze(0)?;
        let (_, lb, lg) = fit_track_rates(&u0, &s0, &xs, &beta0, &gamma0, steps)?;
        let (lb, lg) = (lb.repeat((r, 1, 1))?, lg.repeat((r, 1, 1))?);
        let (per_gene, lb, lg) = fit_track_rates(&u, &s, &xs, &lb, &lg, (steps / REFIT).max(1))?;
        Ok((per_gene, (lb.get(0)?, lg.get(0)?)))
    };
    let (fwd, rates_fwd) = best(1.0)?;
    let (bwd, rates_bwd) = best(-1.0)?;
    let votes = (fwd - bwd)?; // [R × G]
    let lambdas: Vec<f64> = votes
        .sum(1)?
        .to_vec1::<f32>()?
        .into_iter()
        .map(f64::from)
        .collect();
    let lambda = lambdas[0];
    let mean = |v: &[f64]| v.iter().sum::<f64>() / v.len().max(1) as f64;
    let null = mean(&lambdas[1 + boot..]);
    let spread = &lambdas[1..1 + boot];
    let centre = mean(spread);
    let sd = (spread.iter().map(|v| (v - centre).powi(2)).sum::<f64>() / (boot.max(2) - 1) as f64)
        .sqrt();
    let z = (lambda - null) / sd.max(1e-9);
    // The null's bias is measured where there are no dynamics; a `Λ` near
    // zero is no evidence either way, whatever the bias there.
    let bare = lambda / sd.max(1e-9);
    let determined =
        boot > 1 && z.abs() > Z_CALL && bare.abs() > Z_CALL && z.signum() == bare.signum();
    let reversed = determined && z < 0.0;
    if reversed {
        energy.mirror(lin)?;
    }
    let sign = if reversed { -1.0 } else { 1.0 };
    let reads = p as f64 * cfg.depth as f64;
    let d = Direction {
        lineage: lin,
        cells: 0.0,
        lambda,
        lambda_per_read: lambda / reads,
        lambda_null: null,
        lambda_sd: sd,
        z,
        determined,
        reversed,
        gene_votes: votes
            .get(0)?
            .to_vec1::<f32>()?
            .into_iter()
            .map(|v| sign * v)
            .collect(),
    };
    Ok((d, if reversed { rates_bwd } else { rates_fwd }))
}

/// The replicates' steps from the data's optimum, as a share of its own.
const REFIT: usize = 5;

/// Every replicate's genes' which-track log-likelihood `[R × G]` at their
/// best rates, from `u`, `s` `[R × K × G]` pooled reads and the velocity
/// `xs` `[1 × K × G]`, fitted by `steps` of Adam from the log rates `lb`,
/// `lg` (`[R × 1 × G]`); and those log rates.
fn fit_track_rates(
    u: &Tensor,
    s: &Tensor,
    xs: &Tensor,
    lb: &Tensor,
    lg: &Tensor,
    steps: usize,
) -> anyhow::Result<(Tensor, Tensor, Tensor)> {
    let total = (u.sum_all()? + s.sum_all()?)?.to_scalar::<f32>()? as f64;
    let lb = legume_numeric::candle::candle_core::Var::from_tensor(&lb.contiguous()?)?;
    let lg = legume_numeric::candle::candle_core::Var::from_tensor(&lg.contiguous()?)?;
    let mut opt = adam(vec![lb.clone(), lg.clone()], 0.05)?;
    let per_gene = || -> Result<Tensor> {
        let (beta, gamma) = (
            lb.as_tensor().clamp(-6.0, 12.0)?.exp()?,
            lg.as_tensor().clamp(-6.0, 12.0)?.exp()?,
        );
        let m = xs.broadcast_add(&gamma)?;
        let soft = (&gamma * 0.05)?;
        let m_plus = super::model::softplus(&m.broadcast_div(&soft)?)?.broadcast_mul(&soft)?;
        let eta = m_plus.maximum(1e-30)?.log()?.broadcast_sub(&beta.log()?)?;
        let up = legume_numeric::candle::loss::log_sigmoid(&eta)?;
        let down = legume_numeric::candle::loss::log_sigmoid(&eta.neg()?)?;
        (u.broadcast_mul(&up)? + s.broadcast_mul(&down)?)?.sum(1)
    };
    for _ in 0..steps {
        let loss = (per_gene()?.sum_all()?.neg()? / total.max(1.0))?;
        opt.backward_step(&loss)?;
    }
    Ok((
        per_gene()?,
        lb.as_tensor().clamp(-6.0, 12.0)?.detach(),
        lg.as_tensor().clamp(-6.0, 12.0)?.detach(),
    ))
}

#[cfg(test)]
#[path = "tests/train.rs"]
mod tests;
