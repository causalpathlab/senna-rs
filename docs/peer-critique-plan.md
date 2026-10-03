# Peer critique — fits that correct each other (plan)

## 0. Status and summary

**Status: plan. Only stage 0a is built:** `senna critique`, a read-only report (§10). Code
references describe what exists today and where the new pieces would attach.

senna fits several latent models on the same cells: `topic`, `vae`, `svd`, the `masked-*`
family, `bge`, `simba`, `gem`, and on the gene side `fne`. Each has a characteristic failure:

- a topic model merges fine states (its resolution limit);
- a Gaussian VAE can collapse neighbourhoods (mode collapse);
- SVD separates on directions that are not biology.

No model is perfect, and their failures differ, so one model can point out another's mistakes.
Instead of one joint model holding every component, the fits stay separate and exchange
**critique** in rounds. A model is told "these two things you keep together, the others hold
apart", or "these two you hold apart, the others keep together".

**Critique lives entirely in the models' latent spaces.** No gene counts are read: every model
has already been fitted to the counts, so its latent is a data-informed view, and a disagreement
between views means at least one model kept structure another lost. A model is judged against
the **consensus of the other models**. This is contrastive learning with peer-mined hard pairs.

Critique runs on two channels, one on each side of the cell × gene matrix:

- **Channel A, pseudobulk pairs** (§3): do the models agree on which pseudobulks are alike?
- **Channel B, feature pairs** (§4): do they agree on which genes belong together?

The channels feed each other (§5).

## 1. Concepts

| term | meaning |
|---|---|
| **model** | one fit (`topic`, `vae`, …). It publishes its view and consumes critique. |
| **channel** | one kind of pair: pseudobulk pairs (A) or feature pairs (B). |
| **view** | a model's neighbourhoods on one channel: every pair's rank (1 = nearest). |
| **near / far** | rank ≤ *k* / rank beyond `max(3k, P/4)`, well clear of near. |
| **candidate** | a pair in some model's top-*k*. |
| **consensus** | for one model, the median rank of a pair over the *other* models. A model never votes on itself. |
| **critic** | the component that builds the consensus and charges each model (`senna critique`). |
| **charge** | a **merge** (the model keeps near what the others hold far) or a **split** (the reverse). |
| **label** | what a charge becomes for training: −1 for a merge (push apart), +1 for a split (pull together), with a weight. |
| **round** | publish views → critic → each model trains on its labels → next round. |
| **history** | a model's chain of versions, one per round. Each version's manifest names its parent version, the label file it trained on (by hash) and the partition it trained on. |

## 2. Shared indices

A pair can be compared across models only when every model indexes it the same way.

### 2.1 Pseudobulks

- Every family trains on `CollapsedOut` pseudobulks: `train_mixed` and `train_masked` consume
  `&[CollapsedOut]`, and `svd` runs `rsvd` on the pseudobulk posterior.
- One run's cell → pseudobulk partition (`cell_to_pb.parquet`) defines the pseudobulks for every
  model. The models need not have trained on it: each model's view is its own cell latent
  averaged over the partition's pseudobulks (§3.1), so independently fitted runs compare as well.
  For training rounds, `--pb-from` (#30) gives the fits that partition as their own.
- The key is `(level, pb_id)`; level `0` is the coarsest. Every message carries a **partition
  hash**, and a reader rejects any message whose hash does not match its own.

### 2.2 Features

- The key is the gene name, reconciled through the shared canonicaliser. No shared partition is
  needed.
- Fits select features differently (HVG choice), so pairs are mined on the **intersection** of
  the fits' feature axes.
- Topic models that train on coarsened features (`FeatureCoarsening`) embed supergenes. Their
  view is mapped back to genes through the coarsening membership. Two genes in the same supergene
  are close by construction, so such pairs are excluded from that model's view.

## 3. Channel A: pseudobulk pairs

### 3.1 Each model's view

| model | latent | distance |
|---|---|---|
| topic, masked-topic, masked-sbp | θ | Hellinger |
| vae, masked-vae, svd | z, or component scores, z-scored per dimension | Euclidean |
| bge, simba, gem | cell embedding | cosine |

The view is the model's **cell latent averaged over each pseudobulk** of the partition. It needs
no model loading and works the same for every kind. The metric follows the run's `CellSpace`
(`RunKind::cell_space`). Pseudobulks under `--min-cells` cells are left out.

### 3.2 Mining

Compare **ranks**, not distances; the geometries differ. The candidates are the union of every
model's top-*k* pairs. A pair's rank in a model is the smaller of its two directional ranks.

### 3.3 Consensus

- For each model and pair, the **consensus** is the median rank over the other models.
- **Merge:** the model ranks the pair near while the consensus is far. **Split:** the model ranks
  it far while the consensus is near. Agreeing models are never charged.
- **Pair label:** the median over *all* models makes the pair `similar` (near), `different`
  (far) or `ambiguous`.
- **Merge share** = merges / (merges + splits), per model and level. Near 1, a model packs
  together states the others separate: the signature of mode collapse, or of a topic model's
  resolution limit. Near 0, it separates what the others keep together.
- **Level context.** At a coarse level the nearest pseudobulks are often different cell types;
  near/far are ranks within a level, so this is fine, but rates are compared within a level only.
- **Global collapse comes first.** Before mining, check each model's effective dimension,
  per-dimension KL and pseudobulk spread. Global collapse needs KL annealing or free bits, not
  pairs.

**Considered and set aside: a count-based referee.** Stage 0a was first built with a referee that
tested each pair on the summed counts (a two-group Poisson log-likelihood ratio, calibrated by
random halves of each pseudobulk). On BMMNC it showed three problems:

- abundant housekeeping genes (EEF1A1, TPT1, ACTB, MALAT1, MT-, RPL/RPS) piled up large
  statistics from small shifts, so pairs differing only in cell quality were called different;
- at pseudobulk depth a "same" verdict could almost never be earned: fine pseudobulks are small,
  so a low statistic was lack of power, not sameness;
- the counts had already been fitted by every model, so the test mostly re-asked a question the
  latents answer.

### 3.4 Labels

- A merge becomes a **negative** (−1) for the merging model; a split becomes a **positive** (+1)
  for the splitting model.
- The weight grows with agreement: the more other models agree, and the wider the rank gap
  between the model and the consensus, the larger.
- A model is never labelled where the others disagree among themselves (`ambiguous`).

## 4. Channel B: feature pairs

### 4.1 Each model's view

| model | gene embedding | distance |
|---|---|---|
| topic | dictionary β rows (gene × topic) | Hellinger on normalised rows |
| masked-* | ρ (D × H) | cosine |
| vae | decoder weight rows | cosine |
| svd | left singular vectors | cosine |
| bge, simba, gem, fne | `feature_embedding.parquet` | cosine |

### 4.2 Mining

As in §3.2, on the intersected feature axis (§2.2), with supergene-internal pairs excluded.

### 4.3 Consensus

As in §3.3. A **merge** on this channel is two genes a model keeps together that the others hold
apart; typically a topic model folding two programs into one topic. A **split** is one program a
model breaks up.

### 4.4 Labels

As in §3.4.

**External knowledge is one voice.** fne's view comes from its graph (GWAS, eQTL, ABC /
ENCODE-rE2G, ontologies) rather than from this dataset's counts. It joins the consensus as one
model, so prior knowledge can tip a close call but cannot outvote the fitted models.

## 5. Coupling the channels

Each channel can propose candidates for the other. Labels are not passed across: a candidate is
still judged by its own channel's consensus.

- **A → B, separating genes.** For a disputed pseudobulk pair, the model that separates it says
  which genes do the separating: its decoder or dictionary evaluated at the two pseudobulks'
  latents. Those genes' pairs become channel-B candidates.
- **B → A, program ratios.** For a disputed gene pair, pseudobulks that differ mainly in the
  ratio of the two genes' loadings become channel-A candidates.

Bipartite models already tie the two sides together internally: bge, simba and gem co-embed cells
with genes, and the topic model has θ and β. A critique on one side moves their other side too.

## 6. Models

| model | sends | receives | how it receives | change needed |
|---|---|---|---|---|
| topic | A, B | A, B | extra loss term: a margin on θ (A) and on β rows (B) | `ExtraLossHook` in legume-numeric (below) |
| vae | A, B | A, B | extra loss term on z (A) and on decoder rows (B) | same hook |
| masked-* | A, B | A, B | extra loss term on θ (A) and on ρ (B) | the same hook on `train_masked` |
| svd | A, B | — | does not receive at first; later, pair weights in a weighted SVD | none at first |
| bge, simba, gem | A, B | A, B | **as edges**: positives as a relation, negatives as explicit negatives | relation input for peer edges; explicit negatives in `graph-embedding-util` (to be checked) |
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

Messages are files under `{run}.peer/`. Models never read each other's messages; only the critic
does. Each file is written to a temporary name and then renamed.

**Views, model → critic:** `{model}.r{round}.view.parquet`

| channel | level | a | b | rank | dist |
|---|---|---|---|---|---|

**Labels, critic → model:** `{model}.r{round}.labels.parquet`

| channel | level | a | b | label | weight | consensus_rank |
|---|---|---|---|---|---|---|

Field notes:

- `a` and `b` are `pb_id`s on channel A and gene names on channel B.
- `level` is the partition level on channel A, and null on channel B.
- A model receives only the labels it was charged with.
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

- `senna critique` is the critic: it reads the models' manifests, builds their views and the
  consensus, and writes the labels.
- `senna run` drives the loop.

What this provides:

- **History.** Each round writes a new version whose manifest records its parent version, its
  label file (by hash) and its partition (§1). Before and after open side by side in `senna view`.
- **Rollback.** If a round makes a model worse (held-out likelihood drops, or its charges grow),
  the previous version stays the result.

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
- Channel-A charges are kept at **cell-group** resolution, as must-link and cannot-link
  constraints over a stable fine grouping such as the round-0 finest level. Each round moves the
  constraints onto the new partition by overlap:

  ```
  w_ab = Σ_(i,j) frac(a ∩ S_i) · frac(b ∩ S_j) · w_ij
  ```

  Constraints that survive several partitions build up weight. Constraints that depend on one
  grouping fade.
- What to vary: `--pb-refine-seed`, the projection seed, `--sort-dim`, `--knn-cells`. Keep the
  number of levels fixed, because changing it changes scale rather than resampling.
- Channel B is keyed by gene, so its labels need no transport.
- An optional later step lets the partition take critique as well. Cannot-links keep cells out
  of a shared pseudobulk, and a pseudobulk whose cells the models place far apart gets split.

### 8.3 Parallel devices

- One fit per GPU, each its own process, with the device chosen as today
  (`src/embed_common.rs` `to_device`).
- The critic runs on the CPU, and so does collapsing `P_{r+1}` while round `r` trains.
- Rounds are **synchronous by default** (a barrier), so they are deterministic. An asynchronous
  mode (pick up the newest labels whenever ready, as in codistillation) can come later if waiting
  for the slowest fit hurts.
- Fits that share a GPU must pin `--minibatch-size` and skip the `gpu_mem_fraction` probe,
  which assumes it is alone on the device.

## 9. Safeguards

Judging by consensus has a known failure: models that share a mistake are never charged for it,
and repeated rounds pull the models towards agreement. The safeguards target that.

- **Diverse voices.** The consensus is only as good as the spread of model families in it.
  Near-duplicates (topic and masked-topic) should count as one family, so that similar models
  cannot outvote different ones.
- **Each model's own likelihood.** Every round checks each model's held-out likelihood on the
  counts and rolls back a round that hurts it (§8.1). This is where the data still has the last
  word: a model may move towards the others only as far as its own fit to the counts allows.
- **Disagreement is kept, not erased.** A small λ that ramps up, a label budget per round per
  channel, and labels only where the others agree among themselves (§3.4).
- The label budget is where **k-means++-style sampling** belongs: pick labels with probability
  growing with the disagreement, and down-weight pairs near ones already picked, so one region
  does not steer a model. D² weighting favours outliers, so it needs a floor on cells per
  pseudobulk.
- Partition and feature-axis hashes on every message (§7).
- A log per round of the charges per model per channel. They should fall across rounds; if they
  oscillate, stop.

## 10. Staged plan

Each stage can be checked on its own before the next one is built.

| stage | channel | builds | check |
|---|---|---|---|
| 0a | A | **built:** `senna critique`, report only. Views averaged over one run's partition, top-*k* candidates, ranks per model, leave-one-out consensus, merges / splits / merge share per model, per-cell labels | Do the charges match known model behaviour (a topic model's merges, a VAE's collapse)? Do merged pairs fall on known fine cell types? |
| 0b | B | the same for gene pairs, from the models' gene embeddings | Do topic merges join genes of known separate programs? |
| 1 | — | no-new-cells mode for `senna update` with explicit `--pb-from` | Continuing without critique is neutral. |
| 2 | A, B | `ExtraLossHook`; `--peer-labels` through `Rebase` for topic / vae / masked | One round reduces the model's charges without hurting its held-out likelihood. |
| 3 | A, B | peer edges into bge / simba / gem; gene views into fne as relations | The same check, for the graph models. |
| 4 | A, B | the round loop in `senna run`, on parallel devices | Charges fall across rounds, and likelihoods hold. |
| 5 | A | redrawn partitions with constraint transport | Persistent constraints agree with stage 0a. |
| 6 | A ↔ B | channel coupling (§5) | Coupled candidates are charged at a higher rate than mined ones. |
| 7 | A | (optional) the partition takes critique | — |

**Stage 0a, first look (BMMNC, topic + vae + svd).** The topic model's merge share was above 0.9
at every level, the VAE's rose to 0.99 at level 1, and SVD's stayed near 0.5 with the most
charges in both directions.

## 11. Open questions

- **§3.3** *k* and the far threshold, and how both should scale with the number of pseudobulks.
- **§3.3** With three models, one model is half of every other model's consensus. How many
  families are needed before the consensus is trustworthy, and how to weight near-duplicates.
- **§3.4** The label weight: agreement count, rank gap, or both.
- **§4.1** Gene embeddings live on different axes and supports; whether ranks on the
  intersected axis are enough.
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
