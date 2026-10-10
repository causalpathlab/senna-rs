# `senna tde`: temporal divergence embedding (plan)

**Status:** a plan only; nothing is built yet. It depends on `senna ode` giving every pseudobulk a time τ together
with its posterior over ode's τ grid.

## Goal

A unit (pseudobulk or cell) at time τ should predict the expression of its neighbours in time. **Similarity shared
between distant time points should be explicitly penalised.** The aim is tree-like embedding structure, i.e. lineages
that diverge over time, in data with no clear cell-type clusters.

**One factorised likelihood:** no stacked losses with λ weights, and no tuned widths.

## Model

Read the data as events: "a read of gene g, observed at unit v, which shares its time with unit u". Then

  p(v, g | u) = p(v | u) · p(g | v)

- **p(g | v)** is `bge`'s conditional: the exact module → gene softmax on the unit embedding e_v.
- **p(v | u)** is an exact softmax over units, ∝ exp⟨e_u, c_v⟩. Here c is a context table, and a separate table leaves
  room for a forward (earlier → later) variant. The targets v are the finest pseudobulks only, so one exact softmax
  over them is cheap. It is weighted by the unit's own `bge` weight w⁰_u, like its counts, so the context is one more
  observed categorical per unit in the same likelihood.

**Observed weights, in reads, with no kernel:**

  P_uv = Σ_t p(t | x_u) · p(t | x_v),  w_uv = P_uv · 1[v ∈ kNN_expr(u) ∪ {u}] · N_v

- p(t | x) comes from ode's timing scores over its grid, normalised.
- A `bge` unit's posterior is the mean over its member cells' ode pseudobulks. ode's pseudobulks are not `bge`'s, so
  τ is carried through cells, never joined by pseudobulk name.
- Sharp τ makes P_uv narrow, and uncertain τ makes it wide, so the data set the scale.
- The kNN restriction keeps branches apart. It reuses the collapse's existing `--knn-cells`.

**Log-likelihood:** each unit u observes two things: its reads, and its time neighbours, as shares
q_uv = w_uv / Σ_v w_uv. Given e_u, the two are independent, and both are weighted by u's own `bge` weight w⁰_u:

  L = Σ_u w⁰_u [ Σ_g q_ug log p(g | u) + Σ_v q_uv log p(v | u) ]

- It is one likelihood with no λ. The context is scaled exactly like the counts.
- The chain p(v | u) · p(g | v) ("u predicts its neighbours' reads") differs from this only in how often each v's gene
  term is counted (Σ_u w_uv / N_v, ≈ constant per unit). The simpler form is what is built.

**Why this gives a tree:**
- p(v|u) is normalised over all units, so probability on units that share no time with u (w = 0) is pushed down by
  the normaliser. That is the penalty on distant time, with no negative sampling.
- Attraction exists only between units near in both expression and time. Everything else, including other branches
  at the same τ, is repelled.

**New hyperparameters:** none. The only new input is `--ode PREFIX`.

## Code

### legume-graph-embedding (`fit/hier`)

Built on branch `unit-context`, with 325 tests passing:

1. `UnitTable.context: Option<UnitContext>`: per unit, its (target, share) row, with shares summing to 1, plus
   `validate`.
2. `FitConfig.unit_context: Option<Arc<dyn UnitContextBuilder>>`. It is called once after the unit table is built,
   with the table (each unit's level and source index) and each level's cell → pseudobulk map.
3. `hier/step.rs::context_level`: one exact softmax of ⟨e_u, c_v⟩ + b_v over the targets, weighted by w⁰_u · q_uv.
   The target rows `c` and biases train with RowAdagrad like `r` and `b_g`. `FitOutput.unit_context` returns them.
4. With `context = None` every existing test is unchanged.
5. Tests:
   - an f64 reference for the term;
   - disjoint-context look-alikes;
   - the transient-pulse look-alikes (mirror pairs 0.18 → 1.56 of the mean distance, time neighbours 0.06);
   - the miscalibration record above.

### senna (`src/tde/`)

- **`TdeArgs`** = `bge`'s args plus `--ode PREFIX`.
- **The run** reuses `bge::fit_bge` → `driver::fit_embed_family`, with a `TimeContext` builder passed through
  `EmbedKnobs` (the way `gem` passes `after_fit`).
- **`TimeContext`:** unit posteriors from `{ode}.cell_pb.parquet` and `{ode}.pb_tau_post.parquet`; kNN among units of
  the same level, from an existing kNN utility; then w_uv.
- **Outputs:** `bge`'s own, plus `{out}.unit_time.parquet` and the `c` table.
- **Tree readout:** `legume_numeric::matrix::principal_graph` (SimplePPT, already used by `bge`'s scorer). It gives
  branch membership and `pseudotime_from_root`.

### Needed from `senna ode`

`{out}.pb_tau_post.parquet`: the finest level's normalised timing scores, one row per pseudobulk and one column per
grid point.

## Calibration of the τ posterior (a precondition)

P_uv is only as good as p(t | x).
- A binomial timing likelihood summed over 10⁴–10⁵ reads per pseudobulk makes p(t | x) essentially a δ.
- The engine test `a_miscalibrated_time_context_degrades_as_expected` (legume-graph-embedding, branch
  `unit-context`) puts the cost on record. Distances are relative to the mean pairwise distance:

  | posterior width | next-step distance | mirror-pair distance |
  |---|---|---|
  | calibrated | 0.06 | 1.56 |
  | δ-sharp | 0.83: the time line breaks apart | 0.52 |
  | 5× too wide | 0.02 | 0.54: the pulse blurs |

- So ode must ship a calibrated posterior:
  - a simulator test that the central 90% mass covers the true τ about 90% of the time;
  - overdispersion estimated in the likelihood (beta-binomial or NB per gene, by MLE).
  - Fallback: the empirical posterior over `bench/velo/resample.py --halves` replicates.
  - No temperature knob.
- The arrow stays out of tde (c is shared, not forward-only) until the disjoint-halves agreement supports it.

## GPU SGD design

**What one step costs today:**
- the host builds ids and targets: the gene level's `target_pos`/`target_val`, and the context's `pos`/`val`;
- it uploads them once per step;
- it runs one `[B × V]` matmul plus a log-softmax for the context.

The targets are the finest pseudobulks only (V ≈ 10³–10⁴), never cells.

**Target design, after profiling one epoch:**
1. **Device-resident CSR for both targets.** Build it once per fit: the gene counts per (unit, module) and the context
   shares per unit. Each step gathers by `plan.units` on the device, so the per-step host work is just the unit ids.
2. **A chunked log-sum-exp over targets.** The normaliser runs over V in tiles, so `[B × V]` is never materialised.
   The positive terms are a sparse gather. Memory stays at O(B · tile) as V grows.
3. **Scaling in V is the two-level softmax.**
   - With an exact normaliser, sampling positives saves almost nothing: the cost is the O(B·V) log-sum-exp, and the
     numerator is cheap. The chunked LSE in (2) cuts memory, not FLOPs.
   - If V grows, the FLOP saving comes from p(v|u) = p(parent(v) | u) · p(v | parent(v), u) over the pseudobulk
     hierarchy: an exact normaliser over parents, plus a child normaliser within the parent sampled from q (with
     multiplicity weights, as the module → gene level already does).
   - That costs O(B·(P + V/P)), and the module → gene code path is the template.
4. **ode:** batch the two orientations along a leading dimension, as one pass and not two fits back to back. The
   resample replicates share the GPU the same way.

## TDD steps

The simulator is a fork with continuous programs (no discrete clusters), a transient-pulse gene, and known τ.

1. With `ctx = None`, or w_uv = N_u δ_uv, `tde` reproduces `bge` exactly.
2. **Look-alikes:** units just before and just after the pulse are nearest neighbours under `bge`, but not under
   `tde`.
3. **The tree:** on the fork, units at equal τ on different branches are far apart. The principal tree recovers one
   branch point (ARI > 0.9 against the simulator's branches) and beats `bge`.
4. **Distant-time penalty:** the mean ⟨e_u, c_v⟩ over pairs with P_uv ≈ 0 is below that of near pairs, and below
   `bge`'s.

## Evaluation on real data

Run pancreas, then dentate gyrus, once ode is final. Compare the principal tree on `tde` vs `bge`, and check the
branch points against the known fates. Also report the held-out-τ likelihood.
