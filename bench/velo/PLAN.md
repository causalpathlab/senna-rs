# senna ode: time and splicing kinetics, then dynamic embeddings: master plan

Status as of 2026-10-08. The branch is `tde-ode-pb` in senna-rs (provisional, work in progress).

## Goal

The goal is dynamic feature and cell embeddings: genes and cells placed by how they change over time, not only by static co-expression. The path there has three steps:

1. **Stage 1 (`senna ode`):** a time τ for every pseudobulk, plus splicing kinetics, from a finished base fit. Mostly done; see below.
2. **Stage 2:** a τ for every cell, each refining its finest pseudobulk's τ on its own reads.
3. **A dynamic embedding** that learns from τ (and the kinetics). This eventually moves into legume-graph-embedding, next to bge.

## Stage 1: the model (`src/ode/`)

- **Units.** Cells of a finished `senna bge` base fit are grouped into multilevel pseudobulks by their base states, with data-beans `collapse_columns_multilevel_with_hierarchy` (`levels.rs`). The fit runs coarse to fine.
- **Transcription is per module.** Genes are grouped by `coarsen_features` on pseudobulk totals; groups below `--min-module-genes` (default 20) are dropped, as is the background group (`modules.rs`). Each module has a switch-on, a switch-off, a relaxation rate λ and a basal fraction. The closed-form ODE is in `kinetics.rs`, with divided differences that stay stable at coincident rates.
- **Splicing and degradation are per gene:** β_g and γ_g. This is what cell2fate does. Shared rates ran to the caps and lost the direction.
- **The likelihood factorises exactly:** P(g, k | p) = P(g | p) · P(k | p, g).
  - **Which track:** x^u_pg ~ Binom(x^u_pg + x^s_pg, σ(b_g + log u_g(τ_p) − log s_g(τ_p))).
  - **Which gene:** n_p· ~ Mult(softmax_g(c_g + log(u_g + s_g)(τ_p))).
- **Observation terms are only b_g (capture of unspliced, i.e. intron structure) and c_g (level).** There is NO per-pseudobulk capture offset κ_p: it absorbed the common rise and fall of the unspliced share, which is the arrow of time. A shared κ is redundant with b_g.
- **Priors on τ:** a kNN-graph smoothness term λ_g, and on finer levels a pull λ_a toward the parent's τ.
- **Fitting.** Per round:
  1. a timing grid step (switch-on/off pairs per module; this is a non-local move, because a switch-off after the last pseudobulk has no gradient);
  2. a τ grid step per pseudobulk;
  3. Adam with clipping (norm 5), lr 0.01 with cosine decay.

  Both orientations of the starting diffusion order are fitted to the end, and the one with the lower loss is kept. The direction is the likelihood's call; no labels are used.
- **Credit:** the module idea comes from cell2fate (Aivazidis et al., Nature Methods 2025). The training is ours: discriminative read classification with no count model, pseudobulks fitted coarse to fine, grid moves, and a label-free direction.

## Findings so far, on pancreas (`bench/velo`)

- **τ order is solid in every variant:** |Spearman with the trunk cluster order| is about 0.93 at every level.
- **The ratio-only likelihood cannot orient time.** Adding the which-gene (or which-module) magnitude term is what supplies the arrow.
- **Direction under the different rate settings:**
  - per-module rates: correct, but by only 0.0008 nats per read;
  - shared β/γ: the rates ran to the caps and the direction came out wrong;
  - per-gene rates: correct, by 0.0063 per read;
  - per-gene rates with lr decay: wrong.

  This last flip led to the diagnosis that κ_p was eating the arrow. κ has since been removed. **Pending:** the direction at lr 0.01 and at lr 0.03 without κ.
- **Rates.** With per-gene β/γ, about 88% of genes lie inside the bounds, and the cloud has structure. About 11% sit at the bounds; these are genes at the extremes of the unspliced share (intron-poor or intron-rich), most likely misfit rather than drift. The rate bounds are now a smooth sigmoid, but at the very edge the gradient is still negligible. The test `rates_started_past_their_bound_come_back` FAILS. The likely fix is plain log rates with no bound (clipping and the lower lr guard against runaways).
- **Runtime:** about 400 s on an RTX 3080 for 3 levels × 2 orientations. The per-gene timing step dominates; speed it up, e.g. by running it every k rounds.

## Next steps (in order)

1. Confirm the direction is the same and correct at lr 0.01 and 0.03 without κ. If it still flips, find what else absorbs the arrow before going further. Do not add a `--root` workaround.
2. Fix the rate-edge test, test-first.
3. Speed (the timing step).
4. Stage 2: cells. Each cell refines its finest pseudobulk's τ on its own reads, with the kinetics frozen and a pull toward the pseudobulk.
5. Benchmarks: dentate gyrus (branches), erythroid. A head-to-head with scVelo and cell2fate on CBDir and time order.
6. **The dynamic embedding (the user's idea, see below)**, then a move into graph-embedding.

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
