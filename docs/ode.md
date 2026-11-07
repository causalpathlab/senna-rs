# `senna ode` — every cell's lineage, time and velocity from a mixture of splicing ODEs

Every pseudobulk sits on one lineage's trajectory at one time; the trajectory
is the splicing ODE solved exactly, driven by the lineage's transcription.
Fitted on the finest pseudobulks and the most variable genes by a climb to a
mode, then blocked Gibbs chains from it; every cell is read out from the
chains' draws. It builds on cell2fate (Aivazidis et al., *Nature Methods*
2025) and Chronocell (Fang, Gorin and Pachter, *PLOS Computational Biology*
2025).

```bash
senna ode tracks.zarr -o out                     # two lineages on a shared trunk
senna ode tracks.zarr -o out --lineages 1        # one trajectory
senna ode tracks.zarr -o out --no-trunk          # lineages apart from the start
senna view -f out.senna.json                     # y: time, v: velocity field
```

The input is a two-track count backend: rows `{gene}/count/spliced` and
`{gene}/count/unspliced`.

## The model

Per gene `g`: a splicing rate `β_g` and a decay rate `γ_g`, shared by every
lineage, so all of them run on one clock; an embedding `ρ_g` and a baseline
`b_g`. Per lineage `ℓ`: a path `z_ℓ(t)` on a grid of `--points` times in
`[0, 1]`, a Gaussian process (Matérn-5/2, length scale `--kappa`) in `--dim`
dimensions. Transcription is held constant on each grid step:

```text
α_gℓ(t) = exp(ρ_gᵀ z_ℓ(t) + b_g)
```

and the trajectory starts at steady state and steps exactly:

```text
ū(t+h) = α/β + (ū(t) − α/β) e^{−βh}
s̄(t+h) = s̄(t) e^{−γh} + (α/γ)(1 − e^{−γh}) + (βū(t) − α) E(h)
E(h)   = (e^{−βh} − e^{−γh}) / (γ − β)      (a series near β = γ)
```

A pseudobulk `p` picks a lineage and a time, uniformly on the grid, and its
reads are Poisson at its own depth:

```text
(ℓ_p, t_p) ~ π × Uniform,   u_pg ~ Pois(λ_p ū_g(ℓ_p, t_p)),   s_pg ~ Pois(λ_p s̄_g(ℓ_p, t_p))
```

The depth is the pseudobulk's total over the model's, so the likelihood is a
multinomial over both tracks: a change in total mRNA shared by every gene is
not seen.

- **Shared trunk** (the default with two or more lineages): every lineage
  follows one path until a branch time `t_b` and departs from it smoothly
  after, `z_ℓ = z_0 + √2 r(t; t_b) C η_ℓ`, with `r = 1 − exp(−((t − t_b)₊/κ)²)`.
  `--no-trunk` gives each lineage its own path from the start.
- **Unspliced jitter** (on unless `--no-jitter`): every pseudobulk's unspliced
  levels are scaled by `e^{ω_p}`, `ω_p ~ N(0, σ²)` with `σ` fitted, so a
  pseudobulk's nuclear fraction does not pass for kinetics.

Gauges: time is in units of the lineages' duration (`T = 1`); a shift of
every baseline cancels in the multinomial; the rotation of `z` against `ρ` is
left free. Only the invariants are reported: times, rates and levels.

## Inference

1. **Climb** (`--steps`, `--lr`): Adam on the marginal of every pseudobulk
   over its lineage and time, from several starts; a start whose objective is
   not finite never wins.
2. **Direction** (unless `--no-direction`): for each lineage, the pseudobulks'
   order held as climbed, refitted both ways round. The difference in
   evidence calls the lineage `forward`, `reversed`, `undetermined` (within
   5 nats a pseudobulk: steady state looks the same either way) or
   `diverged`. A reversed lineage is turned round only off a trunk; on a
   trunk both arms start at the branch, so the call is reported, not acted
   on.
3. **Chains** (`--chains`, `--sweeps`, `--burn`): blocked Gibbs from the
   climb's mode — the pseudobulks' lineage and time exactly, every gene's
   rates by generalized elliptical slice moves on its Laplace approximation
   (bounded Newton), the paths by slice moves on theirs (banded in the path,
   factored once a refit), the branch time and the jitter. Both rates are
   kept between a half-life of one fine time step and four time units (the
   first lineage's duration; a bounded logit link, the rates' prior carried
   by its Jacobian): past either end the reads cannot tell a rate apart and it would drift.
   Proposals are refitted every `--laplace-every` sweeps during burn-in and
   frozen after, so the kept chain is a homogeneous Markov chain.
4. **Cells**: every cell is scored against `--snapshot-every`'s draws of the
   chains, at its own depth, for its lineage and time posteriors and its
   velocity.

Along one direction the data say little: a warp of the clock traded against
every gene's speed. This is weak identifiability, not a sampler defect, so
every gene's rates are reported with their split R̂ and effective sample size
and flagged above R̂ 1.05.

## Outputs

| file | holds |
|---|---|
| `{out}.cell_time.parquet` | every cell's `lineage`, its `direction`, `tau` with `tau_sd` and the 90% interval `tau_lo`/`tau_hi`, and `p_L1…` lineage posteriors |
| `{out}.cell_embedding.parquet` | every cell on the map: its log1p reads per 10⁴ on the leading `--map-pcs` components of the fitted trajectories (every lineage's levels over the draws), so the axes follow the dynamics, not other variation among the cells |
| `{out}.cell_velocity.parquet` | every cell's velocity on that map, `d log level/dt` through the chain rule |
| `{out}.gene_rates.parquet` | every gene's `log_gamma` and `log_ratio` (log γ/β) with sd, R̂, ESS, and `flagged` |
| `{out}.pb_time.parquet` | every finest pseudobulk's posterior over lineage and time, a row per `(pb, lineage, position)` with its `phase` (the position's share of the lineage's grid, the same in every draw), the draws' mean `time` there and mass `q`; `pb` as `{out}.cell_to_pb.parquet`'s finest level |
| `{out}.branch_time.tsv` | with a shared trunk, every other lineage's branch time off it over the chains (mean and sd, R̂, ESS, `flagged`), on `pb_time`'s phase scale |
| `{out}.lineages.tsv` | every lineage's direction evidence, per pseudobulk, its call and whether it was turned |
| `{out}.senna.json` | the run manifest `senna view` and the MCP tools read |

In `senna view`, `y` colours the cells by time, `v` draws the velocity field
(streamlines, arrows, off) and `lineage` is a grouping; over MCP, `view_map`
takes `time` and `velocity`, and `view_cell` gives a cell's time.
