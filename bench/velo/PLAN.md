# senna ode: time and splicing kinetics, then dynamic embeddings: master plan

Status as of 2026-10-09. The branch is `tde-ode-pb` in senna-rs (provisional, work in progress).

## Goal

The goal is dynamic feature and cell embeddings: genes and cells placed by how they change over time, not only by static co-expression. The path there has three steps:

1. **Stage 1 (`senna ode`):** a time τ for every pseudobulk, plus splicing kinetics, from a finished base fit. Mostly done; see below.
2. **Stage 2:** a τ for every cell, each refining its finest pseudobulk's τ on its own reads.
3. **A dynamic embedding** that learns from τ (and the kinetics). This eventually moves into legume-graph-embedding, next to bge.

## Stage 1: the model (`src/ode/`)

- **Units.** Cells of a finished `senna bge` base fit are grouped into multilevel pseudobulks by their base states, with data-beans `collapse_columns_multilevel_with_hierarchy` (`levels.rs`). The fit runs coarse to fine.
- **Transcription is per module.** Genes are grouped by `coarsen_features` on pseudobulk totals; groups below `--min-module-genes` (default 20) are dropped, as is the background group (`modules.rs`). Each module has a switch-on, a switch-off, a relaxation rate λ and a basal fraction. The closed-form ODE is in `kinetics.rs`, with divided differences that stay stable at coincident rates.
- **Splicing and degradation are per module:** β_m and γ_m (as in be4af4f). Per-gene rates were tried and dropped: never earned by a failing test, and not what fixed the direction.
- **The likelihood factorises exactly:** P(m, g, k | p) = P(m | p) · P(g | m, p) · P(k | p, g); which gene within a module carries no time and is left out.
  - **Which track:** x^u_pg ~ Binom(x^u_pg + x^s_pg, σ(κ_p + b_g + log u_m(τ_p) − log s_m(τ_p))).
  - **Which module:** n_p· ~ Mult(softmax_m(c_m + log(u_m + s_m)(τ_p))).
- **Observation terms:** κ_p (per pseudobulk capture of unspliced, profiled by Newton steps, never a free parameter), b_g (per gene, intron structure), c_m (module level). κ_p was removed for a while on the theory that it ate the arrow; the opposite held (see findings) and it is back.
- **Priors on τ:** a kNN-graph smoothness term λ_g, and on finer levels a pull λ_a toward the parent's τ.
- **Fitting.** Per round:
  1. a timing grid step (switch-on/off pairs per module; this is a non-local move, because a switch-off after the last pseudobulk has no gradient);
  2. a τ grid step per pseudobulk;
  3. Adam with clipping (norm 5), lr 0.01 with cosine decay.

  Both orientations of the starting diffusion order are fitted to the end, and the one with the lower loss is kept. The direction is the likelihood's call; no labels are used.
- **Credit:** the module idea comes from cell2fate (Aivazidis et al., Nature Methods 2025). The training is ours: discriminative read classification with no count model, pseudobulks fitted coarse to fine, grid moves, and a label-free direction.

## Findings so far (`bench/velo`)

### 2026-10-09: κ_p decides the direction on pancreas (senna_astro, 2× RTX 3080, own bge bases)

- **Clean ablation.** be4af4f as is (per-module rates, profiled κ): pancreas correct at lr 0.05 and 0.01 (trunk Spearman ≈ +0.89 / +0.91 / +0.90). The same build with only κ ≡ 0: reversed at both (≈ −0.94 / −0.94 / −0.93). Every no-κ variant was reversed: per-gene or per-module rates, all genes, top-20 genes by reads removed, trunk cells only, lr 0.01 and 0.03. So it was neither overdispersion, the branches, nor per-gene rates.
- **Mechanism (model-free).** Within genes (318 genes with ≥ 20 reads per track in every cluster, log(u/s) centred per gene), the median log(u/s) rises monotonically along the trunk by about 0.3, with about 75% of genes up. A rise of every gene's unspliced share together can only be fitted by the ODE as relaxation after a common switch-on, which puts the high-unspliced end first. κ_p takes up that common trend, so the genes' own lags set the direction. Whether the trend is technical or biological, it is not the arrow.
- **Test.** `a_common_capture_trend_does_not_decide_the_direction`: the sim gets κ_p = trend · τ_p (`SimConfig::capture_trend`). It fails without κ at trend 4 (the sim's kinetic signal is cleaner than real data's, so it takes a larger trend) and passes with κ restored.
- **With κ restored on 5cee045 (lr 0.01 / 0.03), Spearman with the reference order at levels 0/1/2:**
  - pancreas (trunk cluster order): +0.88/+0.90/+0.90 and +0.87/+0.90/+0.90. Correct at both, margins 0.0077 and 0.0015 nats/read.
  - dentate gyrus (cluster order): −0.87/−0.87/−0.83 and −0.50/−0.58/−0.58. Wrong at both (be4af4f with κ at its default lr: +0.92/+0.87/+0.82).
  - erythroid (embryonic stage as clock): −0.91/−0.88/−0.86 and +0.68/+0.46/+0.32. Flips with lr; margins 0.0004 and 0.0015.
- **The starting order is the weak point off pancreas.** The level-0 diffusion order's Spearman with the reference is +0.94 on pancreas, but only +0.24 on dentate gyrus (branches) and +0.05 on erythroid. With a start that barely follows the trajectory, the two orientation runs rebuild the order themselves and the call is near a coin flip. Code review also found that `diffusion_order` does not check that the kNN graph is connected (with several components, v2 is a component indicator).
- **Margins are 0.0004–0.008 nats/read everywhere.** A single fit's margin is not a confidence.

### Earlier (senna_tb; ran on a machine with faulty RAM, treat as unconfirmed)

- **τ order is solid in every variant:** |Spearman with the trunk cluster order| is about 0.93 at every level.
- **The ratio-only likelihood cannot orient time.** Adding the which-gene (or which-module) magnitude term is what supplies the arrow.
- **Direction under the different rate settings:**
  - per-module rates: correct, but by only 0.0008 nats per read;
  - shared β/γ: the rates ran to the caps and the direction came out wrong;
  - per-gene rates: correct, by 0.0063 per read;
  - per-gene rates with lr decay: wrong.

  This last flip led to the diagnosis that κ_p was eating the arrow. κ has since been removed. **Pending:** the direction at lr 0.01 and at lr 0.03 without κ.
- **Rates (per-gene, since dropped).** With per-gene β/γ, about 88% of genes lie inside the bounds, and the cloud has structure. About 11% sit at the bounds; these are genes at the extremes of the unspliced share (intron-poor or intron-rich), most likely misfit rather than drift. The rate bounds are now a smooth sigmoid, but at the very edge the gradient is still negligible. The test `rates_started_past_their_bound_come_back` FAILS. The likely fix is plain log rates with no bound (clipping and the lower lr guard against runaways).
- **Runtime:** about 400 s on an RTX 3080 for 3 levels × 2 orientations. The per-gene timing step dominates; speed it up, e.g. by running it every k rounds.

## Next steps (in order)

1. **Starting order on branched or weak data** (dentate gyrus, erythroid): a connectivity check in `diffusion_order`, then a better start (the user's ideas: pseudotime with clustering, or an MST over pseudobulks, which also gives branches). Still no `--root`.
2. **Direction confidence:** refit on resampled pseudobulks (the user's posterior-sampling idea) and report how often the direction agrees, instead of one margin.
3. **Code-review fixes**, each test-first: level-0 `tau_start` stored unflipped when the reverse wins; b_g not written back after timing/init moves; `refine` resets b_g/c_g; Adam moments across grid jumps; cosine floor never reached.
4. Speed (the timing step).
5. Stage 2: cells. Each cell refines its finest pseudobulk's τ on its own reads, with the kinetics frozen and a pull toward the pseudobulk.
6. Benchmarks: a head-to-head with scVelo and cell2fate on CBDir and time order.
7. **The dynamic embedding (the user's idea, see below)**, then a move into graph-embedding.

## The embedding idea (the user's): time-displaced negatives

Given τ, for each pseudobulk at time t, draw negatives from pseudobulks or cells at t ± Δt. The contrastive (NCE) objective then learns gene features that change over time.

Notes from the discussion:
- Signed Δt, or a head that predicts sign(Δt), is needed to learn direction, not just "a different time".
- Take negatives from the same lineage (kNN in the base embedding with a displaced τ). Otherwise, at a fixed t, branch identity gets learned as time.
- Δt sets the difficulty: small Δt gives hard negatives, large Δt easy ones. Use it as a curriculum.
- Possible extension: positive pairs of "unspliced at t ↔ spliced at t + Δt" (the displaced-tracks idea), to tie the embedding to splicing.

An earlier attempt, θ(τ) = W φ(τ) with φ the log ODE curves, diverged: badly scaled coordinates and an ill-conditioned W. It was removed and is in git history (commit be4af4f).

## Working rules (from the user)

- **TDD.** Start minimal; add a parameter only when a failing test needs it, and keep only what is essential.
- **Tests** go in `tests/` folders (`#[path = "tests/<name>.rs"]`).
- **Commits and PRs carry no trailers.** Publishing only at the end of a session, after asking.
- **Patch-up order:** code-review → simplify → fmt → clippy.
- **Use CUDA** (`--features cuda`, `--device cuda`). senna_astro's machine has 2 GPUs: run orientations or datasets in parallel (`--device-no 0/1`).

## Reproducing the benchmark (`bench/velo/`)

```
bench/velo/fetch.sh                                                     # raw h5ad: pancreas, dentate gyrus, erythroid
uv run --with anndata,scipy,pandas,numpy python bench/velo/export.py pancreas dentategyrus
data-beans from-mtx mtx/pancreas_tracks/matrix.mtx.gz -r .../features.tsv.gz -c .../barcodes.tsv.gz \
  --row-name-columns 1 --select-row-type "" --backend zarr -o beans/pancreas_tracks_count   # same for _total
senna bge beans/pancreas_total_count.zarr.zip -o base/pancreas_total --device cuda --skip-etm
senna ode beans/pancreas_tracks_count.zarr.zip --base base/pancreas_total -o runs/X --device cuda
uv run --with pandas,pyarrow,numpy,scipy,scikit-learn,matplotlib python bench/velo/score_pb.py pancreas runs/X
python bench/velo/module_fit.py pancreas runs/X out.png    # observed vs fitted per module
python bench/velo/rates.py runs/X out.png                  # β vs γ per gene, by module
```

Data live in `$VELO_BENCH` (default `~/work/velo-bench`): `raw/`, `mtx/`, `beans/`, `base/`, `runs/`, `<name>.cells.tsv.gz`.
