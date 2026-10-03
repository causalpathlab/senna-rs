# Peer critique — fits that correct each other (plan)

## 0. Status and summary

**Status: plan. None of this is built.** Code references describe what exists today and where
the new pieces would attach.

senna fits several latent models on the same cells: `topic`, `vae`, `svd`, the `masked-*`
family, `bge`, `simba`, and on the gene side `fne`. Each has a characteristic failure:

- a topic model merges fine states (its resolution limit);
- a Gaussian VAE can collapse neighbourhoods;
- SVD separates on directions that are not biology.

Their failures differ, so one model can point out another's mistakes. Instead of one joint model
holding every component, the fits stay separate and exchange **critique** in rounds. A model is
told "these two things you placed together are different", or "these two you placed apart are
the same". A **referee** confirms each claim against the counts before it reaches the model.
This is contrastive learning with peer-mined hard pairs, and the referee guards against false
negatives.

Critique runs on two channels, one on each side of the cell × gene matrix:

- **Channel A, pseudobulk pairs** (§3): are two pseudobulks really different?
- **Channel B, feature pairs** (§4): do two genes belong to the same program?

The channels feed each other (§5).

## 1. Concepts

| term | meaning |
|---|---|
| **model** | one fit (`topic`, `vae`, …). It publishes its view and consumes critique. |
| **channel** | one kind of pair: pseudobulk pairs (A) or feature pairs (B). |
| **view** | a model's neighbourhoods on one channel: top-*k* neighbours, by rank. |
| **candidate** | a pair on which two models' views disagree. |
| **referee** | the component that tests candidates against the counts and labels them. It never trusts a model. |
| **label** | +1 (the pair is the same, so pull it together) or −1 (the pair differs, so push it apart). Each label carries a weight. |
| **round** | publish views → referee → each model trains on its labels → next round. |
| **history** | a model's chain of versions, one per round. Each version's manifest names its parent version, the label file it trained on (by hash) and the partition it trained on. |

## 2. Shared indices

A pair can be compared across models only when every model indexes it the same way.

### 2.1 Pseudobulks

- Every family trains on `CollapsedOut` pseudobulks: `train_mixed` and `train_masked` consume
  `&[CollapsedOut]`, and `svd` runs `rsvd` on the pseudobulk posterior.
- `--pb-from` (#30) makes fits collapse on one shared cell→pseudobulk partition. `senna run`
  already passes it to fits whose collapse flags agree.
- The key is `(level, pb_id)`. Every message carries a **partition hash**, and a reader rejects
  any message whose hash does not match its own.
- bge and simba embed cells, not pseudobulks. Their pseudobulk view is the cell embedding
  averaged through the round's `cell_to_pb`.

### 2.2 Features

- The key is the gene name, reconciled through the shared canonicaliser. No shared partition is
  needed.
- Fits select features differently (HVG choice), so pairs are mined on the **intersection** of
  the fits' feature axes.
- Topic models that train on coarsened features (`FeatureCoarsening`) embed supergenes. Their
  view is mapped back to genes through the coarsening membership. Two genes in the same supergene
  are close by construction, so such pairs are excluded from mining for that model.

## 3. Channel A: pseudobulk pairs

### 3.1 Each model's view

| model | latent | distance |
|---|---|---|
| topic, masked-* | θ (pseudobulk-level) | Hellinger |
| vae | z | Euclidean |
| svd | right singular vectors | cosine |
| bge, simba | cell embedding averaged per pseudobulk | cosine |

A model's view can be obtained in two ways: encode the pseudobulk itself, or average its cells'
latents. Which one to use is an open question (§11).

### 3.2 Mining

Compare **ranks**, not distances; the geometries differ. For each pseudobulk, take its top-*k*
in model A and look up the rank of each neighbour in model B.

- Close in A, far in B: a suspected **merge** in A.
- Far in A, close in B: a suspected **split** in A, or noise in B.

### 3.3 Referee

- The test is a Poisson / NB likelihood ratio on the two pseudobulks' counts: one shared profile
  against two.
- **Level context.** Two pseudobulks under the same coarse parent are expected to be similar. A
  merge across parents is the stronger signal.
- **Caching.** The test depends only on the data, so each pair is tested once over the whole run.
- **Global collapse comes first.** Before mining, check each model's effective dimension,
  per-dimension KL and pseudobulk spread. Global collapse needs KL annealing or free bits, not
  pairs.

### 3.4 Labels

- **Negatives (−1)** come from a significant difference. These are the first to be used.
- **Positives (+1)** need a stricter criterion, such as an equivalence test or a small effect
  size. A test that fails to reject is not evidence of equality, and pulling pairs together on
  weak evidence over-smooths.
- The weight is the test statistic, capped.

## 4. Channel B: feature pairs

### 4.1 Each model's view

| model | gene embedding | distance |
|---|---|---|
| topic | dictionary β rows (gene × topic) | Hellinger on normalised rows |
| masked-* | ρ (D × H) | cosine |
| vae | decoder weight rows | cosine |
| svd | left singular vectors | cosine |
| bge, simba, fne | `feature_embedding.parquet` | cosine |

### 4.2 Mining

As in §3.2, on the intersected feature axis (§2.2), with supergene-internal pairs excluded.

- Close in A, far in B: A may have merged two programs. Typically a topic model folds them into
  one topic.
- Far in A, close in B: A may have split one program.

### 4.3 Referee

The test is co-expression of the two genes across the shared pseudobulks. Two requirements:

- **Residuals.** Remove library size and the dominant composition axis first. Otherwise most
  gene pairs look correlated simply because pseudobulks differ in size and cell-type mix.
- **Local as well as global.** Test within each coarse-level parent as well as across all
  pseudobulks. Two genes can co-vary in one lineage and not in another, and a global correlation
  averages that away. That is the case where a topic model folds two programs into one, so the
  local test is what addresses its resolution limit.

Results are cached as in §3.3. They depend on the partition, so the cache is keyed on it.

### 4.4 Labels

As in §3.4. A negative is a confirmed lack of co-expression (globally, or within some parent). A
positive is a strong, stable co-expression.

**External knowledge is a voice, not a referee.** fne's sources (GWAS, eQTL, ABC / ENCODE-rE2G,
ontologies) can *propose* gene pairs as candidates, but only the co-expression test labels them.
If prior knowledge could label pairs on its own, it would override what this dataset shows and
correct away programs specific to this data.

## 5. Coupling the channels

The two referees feed each other candidates. They do not feed each other labels: every
candidate is still tested on its own channel.

- **A → B, driver genes.** A confirmed pseudobulk negative `(i, j)` has per-gene contributions to
  its likelihood ratio. Driver genes that move in opposite directions between *i* and *j* become
  channel-B negative candidates.
- **B → A, program ratios.** A confirmed gene negative `(g, h)` marks two separate programs.
  Pseudobulks that differ mainly in the ratio of `g` to `h` become channel-A negative candidates.

Bipartite models already tie the two sides together internally: bge and simba co-embed cells
with genes, and the topic model has θ and β. A critique on one side moves their other side too.

## 6. Models

| model | sends | receives | how it receives | change needed |
|---|---|---|---|---|
| topic | A, B | A, B | extra loss term: a margin on θ (A) and on β rows (B) | `ExtraLossHook` in legume-numeric (below) |
| vae | A, B | A, B | extra loss term on z (A) and on decoder rows (B) | same hook |
| masked-* | A, B | A, B | extra loss term on θ (A) and on ρ (B) | the same hook on `train_masked` |
| svd | A, B | — | does not receive at first; later, pair weights in a weighted SVD | none at first |
| bge, simba | A, B | A, B | **as edges**: positives as a relation, negatives as explicit negatives | relation input for peer edges; explicit negatives in `graph-embedding-util` (to be checked) |
| fne | B | B | **as edges**: one relation per model's gene view, plus external knowledge | edge files from the gene views; it already consumes typed relations |

**fne as the hub of channel B.** fne fuses the models' gene views with external knowledge into
one feature embedding. That embedding goes back to bge, simba and the masked models through
`--{init,freeze,lora}-feature-embedding` (`src/feature_embedding_args.rs`). This hand-off
exists today.

**The extra-loss hook.** Today's `loss_hook(loss, level)` in `legume_numeric::candle::vae::topic`
sees a loss that has already been averaged. It sees no rows and no latent, and it cannot capture
the encoder, because `train_mixed` holds the encoder as `&mut`. The proposal is a hook that
receives the encoder at call time:

```rust
pub type ExtraLossHook<'a, Enc> =
    dyn Fn(&Enc, usize /*level*/, usize /*epoch*/) -> anyhow::Result<Option<Tensor>> + 'a;
```

- It is called once per minibatch, and its result is added to the loss before backward.
- Channel-A pairs are gathered by pseudobulk index from the level's resident data.
- Channel-B pairs index decoder or ρ rows.
- At an epoch boundary, the same hook can publish the model's views.
- The anchor prior (`src/topic/train.rs`) could move onto this hook.

The loss terms, with `d` the model's own distance from §3.1 / §4.1:

```
negative:  w · max(0, m − d(a, b))
positive:  w · d(a, b)
```

For channel A there is a variant that puts the margin on the **decoded profiles** instead of the
latents. A decoder can map separated latents back onto the same output, so pushing the latents
apart could be undone; a margin on the reconstructions targets the failure directly.

## 7. Protocol

Messages are files under `{run}.peer/`. Models never read each other's messages; only the
referee does. Each file is written to a temporary name and then renamed.

**Views, model → referee:** `{model}.r{round}.view.parquet`

| channel | level | a | b | rank | dist |
|---|---|---|---|---|---|

**Labels, referee → model:** `{model}.r{round}.labels.parquet`

| channel | level | a | b | label | weight | source_model |
|---|---|---|---|---|---|---|

Field notes:

- `a` and `b` are `pb_id`s on channel A and gene names on channel B.
- `level` is the partition level on channel A. On channel B it is the parent level of a local
  test, or null for a global one.
- A model receives only the labels on which *it* was judged wrong.
- Both files carry the partition hash and the feature-axis hash in the parquet metadata.

## 8. Execution

### 8.1 Rounds through `senna update`

`senna update` (`src/update.rs`) is a dispatcher. It replays a fit's recorded arguments through
`Rebase`, warm-starts from the parent and writes a new versioned artifact, for every family
through `Updatable`. A round has the same shape, with "continue with critique" in place of
"continue with new cells":

```
round r:
  senna critique M_a.r{r} M_b.r{r} … --partition P_r          → labels/{a,b,…}.r{r}.parquet
  senna update --model M_a.r{r} --out M_a.r{r+1} --peer-labels labels/a.r{r}.parquet \
               --pb-from P_r --epochs E                       (one per device, in parallel)
  …
```

- `senna critique` is new. It is the referee: it reads the models' manifests, produces or
  reads their views, mines, tests and writes the labels.
- `senna run` drives the loop.

What this provides:

- **History.** Each round writes a new version whose manifest records its parent version, its
  label file (by hash) and its partition (§1).
- **Rollback.** If a round makes a model worse (held-out likelihood drops, or its confirmed
  disagreements grow), the previous version stays the result.

Where `update` does not fit today:

1. `data_files` is a required argument. Rounds need a **no-new-cells** mode.
2. Update forces `from = None` and drops `--pb-from`, because new cells are absent from the
   parent's partition. With no new cells that reason does not apply, so a round **sets**
   `--pb-from P_r`.
3. **Carried pseudobulks** (`pb_reference`) are tied to the parent's partition. A
   fixed-partition round can keep that fast path. A redrawn-partition round (§8.2) must
   re-collapse on `P_r`.
4. **Each round is a fresh process**, so Adam state resets and data is reloaded. This is
   acceptable at tens of epochs per round. Otherwise the optimizer state can be saved next to
   the weights, or the fits can run as long-lived processes that synchronise at a barrier (§8.3).

### 8.2 Redrawing partitions (channel A)

The partition is **fixed within a round**, so that messages share an index. It is **redrawn
between rounds**, so that critique is not an artifact of one particular grouping.

- Models do not care about the partition. Their encoders are amortised, so a new partition only
  means new training rows. Their views are rebuilt each round.
- Channel-A evidence is kept at **cell-group** resolution, as must-link and cannot-link
  constraints over a stable fine grouping such as the round-0 finest level. Each round moves the
  constraints onto the new partition by overlap:

  ```
  w_ab = Σ_(i,j) frac(a ∩ S_i) · frac(b ∩ S_j) · w_ij
  ```

  Constraints that survive several partitions build up weight. Constraints that depend on one
  grouping fade.
- What to vary: `--pb-refine-seed`, the projection seed, `--sort-dim`, `--knn-cells`. Keep the
  number of levels fixed, because changing it changes scale rather than resampling.
- Channel B is keyed by gene, so its labels need no transport. Its co-expression cache is
  re-keyed per partition (§4.3).
- An optional later step lets the partition take critique as well. Cannot-links keep cells out
  of a shared pseudobulk, and pseudobulks the referee finds internally heterogeneous get split.

### 8.3 Parallel devices

- One fit per GPU, each its own process, with the device chosen as today
  (`src/embed_common.rs` `to_device`).
- The referee runs on the CPU, and so does collapsing `P_{r+1}` while round `r` trains.
- Rounds are **synchronous by default** (a barrier), so they are deterministic. An asynchronous
  mode (pick up the newest labels whenever ready, as in codistillation) can come later if waiting
  for the slowest fit hurts.
- Fits that share a GPU must pin `--minibatch-size` and skip the `gpu_mem_fraction` probe,
  which assumes it is alone on the device.

## 9. Safeguards

- **The data decides.** No label reaches a model without a referee test on the counts. This is
  what stops the models agreeing on a shared mistake, and it is why external knowledge only
  proposes candidates (§4.4).
- Partition and feature-axis hashes on every message (§7).
- A label budget per round per channel, and a weight λ that ramps up, so a model is not steered
  by peers that are still poor in early rounds.
- A log per round of the confirmed labels per model per channel. The count should fall across
  rounds; if it oscillates, stop.
- A held-out likelihood check per round, with rollback (§8.1).

## 10. Staged plan

Each stage can be checked on its own before the next one is built.

| stage | channel | builds | check |
|---|---|---|---|
| 0a | A | `senna critique`, report only: mine pseudobulk pairs (vae ↔ svd ↔ topic) on one shared partition, run the count test | Do confirmed negatives line up with known fine cell types? If they look random, stop channel A. |
| 0b | B | the same for gene pairs, with residual co-expression, global and local | Do confirmed negatives separate known programs that a topic merged? |
| 1 | — | no-new-cells mode for `senna update` with explicit `--pb-from` | Continuing without critique is neutral. |
| 2 | A, B | `ExtraLossHook`; `--peer-labels` through `Rebase` for topic / vae / masked | One round reduces the critiqued model's confirmed disagreements without hurting its likelihood. |
| 3 | A, B | peer edges into bge / simba; gene views into fne as relations | The same check, for the graph models. |
| 4 | A, B | the round loop in `senna run`, on parallel devices | Disagreements fall across rounds. |
| 5 | A | redrawn partitions with constraint transport | Persistent constraints agree with stage 0a. |
| 6 | A ↔ B | channel coupling (§5) | Coupled candidates are confirmed at a higher rate than mined ones. |
| 7 | A | (optional) the partition takes critique | — |

## 11. Open questions

- **§3.1** Pseudobulk-level views: encode the pseudobulk, or average its cells? The masked
  models' encoders see a visible mask.
- **§3.3** The test's thresholds at pseudobulk depth. The ambient term and batch residuals must
  not make every pair look different.
- **§3.4 / §4.4** What counts as "the same": an equivalence margin, and how strict.
- **§4.3** Which composition axes to remove before co-expression, and the minimum number of
  pseudobulks per parent for a local test.
- **§6** A margin on the latent or on the decoded profile. This may differ by family.
- **§6** Whether `graph-embedding-util` can take explicit negatives, or only samples its own.
- **§6** How far svd can take part beyond publishing.
- **§8.1** Whether Adam resets at round boundaries matter in practice.

## 12. Related work

- Peers teaching each other: Deep Mutual Learning (Zhang et al., CVPR 2018); codistillation
  (Anil et al., ICLR 2018 — stale peer snapshots suffice); Mutual Mean-Teaching (Ge et al.,
  ICLR 2020 — unsupervised, clustering pseudo-labels).
- Peers choosing each other's samples: Co-teaching (Han et al., NeurIPS 2018); Co-teaching+
  (Yu et al., ICML 2019 — train on disagreements); JoCoR (Wei et al., CVPR 2020 — the agreement
  counterpoint); DivideMix (Li et al., ICLR 2020).
- Contrastive learning: NCE (Gutmann & Hyvärinen, 2010); InfoNCE (van den Oord et al., 2018);
  hard negatives (Robinson et al., ICLR 2021); false negatives (Chuang et al., NeurIPS 2020).
- Graph-regularised factorisation, the static form of a channel-A critique: LapPLSA (Cai et al.,
  KDD 2008); GNMF (Cai et al., TPAMI 2011).
- Must-link / cannot-link constraints: Wagstaff et al., ICML 2001.

Citations were written from memory. Check each one before quoting it.
