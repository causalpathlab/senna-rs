# Peer critique — models that question each other (plan)

## 0. Status and summary

**Status: plan. Stage 0a is built:** `senna critique`, a read-only report (§10). Code references
describe what exists today and where the new pieces would attach.

senna fits several latent models on the same cells: `topic`, `vae`, `svd`, the `masked-*`
family, `bge`, `simba`, `gem`, and on the gene side `fne`. Each has a characteristic failure:

- a topic model merges fine states (its resolution limit);
- a Gaussian VAE can collapse neighbourhoods (mode collapse);
- SVD separates on directions that are not biology.

No model is perfect, and their failures differ. Instead of one joint model holding every
component, the fits stay separate and improve one another in rounds, as **active learning in
which the oracle is the other models**:

- every model is a **learner**: it asks about pairs of cell groups;
- the other models are its **oracle**: their consensus answers;
- answers are trusted in one direction only (§3.3), and questions are chosen to be informative,
  diverse and random (§8.2);
- the learner trains on the answers it can trust, and the next round asks again.

**Everything happens in the models' latent spaces.** No gene counts are read: every model has
already been fitted to the counts, so a disagreement between latents means at least one model
kept structure another lost.

Questions run on two channels, one on each side of the cell × gene matrix:

- **Channel A, pseudobulk pairs** (§3): do the models agree on which pseudobulks are alike?
- **Channel B, feature pairs** (§4): do they agree on which genes belong together?

### In plain terms

**A model asks the others about pairs of cell groups. When it keeps a pair close and the others
clearly hold it apart, it is told "push these two apart". Nothing else is taken as an answer.**

1. **Every model is a witness.** Each has its own picture of the cells; none is trusted alone.
2. **They describe the same groups.** All models are asked about the same pseudobulks, and each
   ranks the other groups from nearest to farthest.
3. **Close and far are ranks, not distances.** Each model measures in its own units, so the
   question is "is B among A's 15 nearest?" (close) or "is B in A's far quarter?" (far). The gap
   between the two means rank 15 against rank 16 is never a disagreement.
4. **A model is answered by the others, never by itself:** the median rank of the other models.
5. **Only "far" answers are taken.**

   | the asking model says | the others answer | outcome |
   |---|---|---|
   | close | clearly far | **merge**: "push these apart" |
   | far | close | **contentious**: no answer; queued as a question (§8.2) |

6. **Why.** Against expert cell types (§10), a merged pair is two different cell types almost
   every time. When a lone model holds a pair far that the others keep close, it is usually the
   one that is right; telling it to pull the pair together would teach it the majority's mistake.
   **Separation is evidence; closeness is not.**

Example: topic ranks pseudobulks 36 and 58 at 52, svd at 60, bge at 48, and the VAE at 15. The
VAE alone keeps them together and is told to push them apart. They are B cells and non-B cells.

A model's **merge rate**, merges over its near pairs, is its report card: high, and it lumps
together groups the others separate (a topic model's resolution limit, a VAE's mode collapse).

## 1. Concepts

| term | meaning |
|---|---|
| **model** | one fit (`topic`, `vae`, …). It is a learner, and an oracle for the others. |
| **channel** | one kind of pair: pseudobulk pairs (A) or feature pairs (B). |
| **view** | a model's ranks on one channel: for each item, every other item from nearest (1) on. |
| **near / far** | rank ≤ *K* / beyond `max(2K, P/4)` (§3.3), well clear of near. |
| **question** | a pair a learner asks about. |
| **answer** | the median rank of the pair over the other models (a random subset of them, §8.2). |
| **merge** | the learner keeps the pair near and the answer is far: the pair becomes a label. |
| **contentious pair** | the learner holds the pair far and the answer is near: no label; it may be asked again. |
| **label** | −1 on a merged pair (push apart), with a weight. There are no +1 labels. |
| **round** | publish views → ask and answer → each model trains on its labels → next round. |
| **history** | a model's chain of versions, one per round. Each version's manifest names its parent version, the label file it trained on (by hash), the partition it trained on, and the round's seed. |

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
- Fits select features differently (HVG choice), so pairs are taken on the **intersection** of
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
(`RunKind::cell_space`). Pseudobulks under `--min-cells` cells, and pseudobulks whose averaged
latent is not finite, are left out of the view.

### 3.2 Questions

- **Ranks, not distances;** the geometries differ. A pair's rank in a model is the smaller of its
  two directional ranks.
- **What a learner asks about:** its near pairs ("am I lumping these?"), the union of every
  model's top-*K* pairs. Stage 0a asks about all of them; from stage 2 on, a round asks a
  sample (§8.2).
- Later, a learner can also ask about its **grey zone**, pairs between near and far, where it is
  least sure.

### 3.3 Answers

- **The answer is the others' median rank.** The learner never answers itself.
- **Far is beyond `max(2K, P/4)`.** Swept against expert cell types on two donors (§10): a far
  set by *K* alone admits merges of one cell type on one donor and not the other, so far must
  scale with *P*; rules looser than about P/4 let same-type merges in; rules much stricter (P/2,
  the bottom *K*, all other models agreeing) are just as precise but find far fewer merges. The
  `2K` floor only guards tiny levels.
- **Only a far answer is taken.** A near pair answered far is a **merge**. A far pair answered
  near is **contentious**: no label, because a lone separating model is usually right (§10).
- **Level context.** At a coarse level the nearest pseudobulks are often different cell types;
  near and far are ranks within a level, and rates are compared within a level only.
- **Global collapse comes first.** Before any round, check each model's effective dimension,
  per-dimension KL and pseudobulk spread. Global collapse needs KL annealing or free bits, not
  pairs.

**Considered and set aside.**

- *A count-based referee* (a two-group Poisson log-likelihood ratio between pseudobulks,
  calibrated by random halves of each pseudobulk). Abundant housekeeping genes (EEF1A1, TPT1,
  ACTB, MALAT1, MT-, RPL/RPS) piled up large statistics from small shifts; at pseudobulk depth a
  "same" could almost never be earned; and the counts had already been fitted by every model.
- *Taking "near" answers* (pull together). Against expert labels they are mostly wrong (§10).
- *A concatenated joint latent* as the oracle. It merges whatever most models merge, so it hides
  exactly the merges worth finding, and near-duplicate models count twice.
- *Far as the bottom K.* Precise, but finds almost no merges.

### 3.4 Labels

- **A merge becomes a negative** (−1) for the learner: push the pair apart.
- **There are no positives.**
- The weight grows with agreement: the more of the answering models hold the pair far, and the
  wider the gap between the learner's rank and the answer, the larger.

## 4. Channel B: feature pairs

### 4.1 Each model's view

| model | gene embedding | distance |
|---|---|---|
| topic | dictionary β rows (gene × topic) | Hellinger on normalised rows |
| masked-* | ρ (D × H) | cosine |
| vae | decoder weight rows | cosine |
| svd | left singular vectors | cosine |
| bge, simba, gem, fne | `feature_embedding.parquet` | cosine |

### 4.2 Questions

As in §3.2, on the intersected feature axis (§2.2), with supergene-internal pairs excluded.

### 4.3 Answers

As in §3.3. A **merge** on this channel is two genes a model keeps together that the others hold
apart; typically a topic model folding two programs into one topic.

**External knowledge is one voice.** fne's view comes from its graph (GWAS, eQTL, ABC /
ENCODE-rE2G, ontologies) rather than from this dataset's counts. It answers as one model, so
prior knowledge can tip a close call but cannot outvote the fitted models.

### 4.4 Labels

As in §3.4. Whether the channel-B answers are as one-sided as channel A's is for stage 0b to show.

## 5. Coupling the channels

Each channel can suggest questions for the other. Labels are not passed across: a question is
still answered on its own channel.

- **A → B, separating genes.** For a merged pseudobulk pair, the models that separate it say which
  genes do the separating: their decoder or dictionary evaluated at the two pseudobulks' latents.
  Those genes' pairs become channel-B questions.
- **B → A, program ratios.** For a merged gene pair, pseudobulks that differ mainly in the ratio of
  the two genes' loadings become channel-A questions.

Bipartite models already tie the two sides together internally: bge, simba and gem co-embed cells
with genes, and the topic model has θ and β. A label on one side moves their other side too.

## 6. Models

| model | asks and answers | receives labels | how it receives | change needed |
|---|---|---|---|---|
| topic | A, B | A, B | extra loss term: a margin on θ (A) and on β rows (B) | `ExtraLossHook` in legume-numeric (below) |
| vae | A, B | A, B | extra loss term on z (A) and on decoder rows (B) | same hook |
| masked-* | A, B | A, B | extra loss term on θ (A) and on ρ (B) | the same hook on `train_masked` |
| svd | A, B | A | **as features**: the genes that separate its merged pairs in the pseudobulk counts are promoted into its HVG set, and `rsvd` is re-solved; the solver is untouched | a merge-score term in the HVG ranking (`--must-train-features` exists already) |
| bge, simba, gem | A, B | A, B | **as edges**: explicit negatives | explicit negatives in `graph-embedding-util` (to be checked) |
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

The loss term, with `d` the model's own distance from §3.1 / §4.1:

```
negative:  w · max(0, m − d(a, b))
```

For channel A there is a variant that puts the margin on the **decoded profiles** instead of the
latents. A decoder can map separated latents back onto the same output, so pushing the latents
apart could be undone; a margin on the reconstructions targets the failure directly.

## 7. Protocol

Messages are files under `{run}.peer/`. Models never read each other's messages; only
`senna critique` does. Each file is written to a temporary name and then renamed.

**Views, model → critique:** `{model}.r{round}.view.parquet`

| channel | level | a | b | rank | dist |
|---|---|---|---|---|---|

**Labels, critique → model:** `{model}.r{round}.labels.parquet` (stage 0c writes `{out}.critique.labels.{model}.parquet`)

| channel | level | pb_a | pb_b | weight | answer_rank | own_rank | answered_by |
|---|---|---|---|---|---|---|---|

Field notes:

- `pb_a` and `pb_b` are `pb_id`s on channel A; on channel B the pair is two gene names.
- `level` is the partition level on channel A, and null on channel B.
- `answered_by` names the models in the round's sub-committee.
- Both files carry the partition hash, the feature-axis hash and the round's seed in the parquet
  metadata.

## 8. Execution

### 8.1 Rounds through `senna update`

`senna update` (`src/update.rs`) is a dispatcher. It replays a fit's recorded arguments through
`Rebase`, warm-starts from the parent and writes a new versioned artifact, for every family
through `Updatable`. A round has the same shape, with "continue with labels" in place of
"continue with new cells":

```
round r (seed s_r):
  senna critique M_a.r{r} M_b.r{r} … --partition P_r --seed s_r   → labels/{a,b,…}.r{r}.parquet
  senna update --model M_a.r{r} --out M_a.r{r+1} --peer-labels labels/a.r{r}.parquet \
               --pb-from P_r --epochs E                          (one per device, in parallel)
  …
```

`senna run` drives the loop.

What this provides:

- **History.** Each round writes a new version whose manifest records its parent version, its
  label file (by hash), its partition and its seed (§1). Before and after open side by side in
  `senna view`.
- **Rollback.** If a round makes a model worse (held-out likelihood drops, or its merge rate
  grows), the previous version stays the result.

Where `update` does not fit today:

1. `data_files` is a required argument. Rounds need a **no-new-cells** mode.
2. Update forces `from = None` and drops `--pb-from`, because new cells are absent from the
   parent's partition. With no new cells that reason does not apply, so a round **sets**
   `--pb-from P_r`.
3. **Carried pseudobulks** (`pb_reference`) are tied to the parent's partition. A
   fixed-partition round can keep that fast path. A redrawn-partition round (§8.3) must
   re-collapse on `P_r`.
4. **Each round is a fresh process**, so Adam state resets and data is reloaded. This is
   acceptable at tens of epochs per round. Otherwise the optimizer state can be saved next to
   the weights, or the fits can run as long-lived processes that synchronise at a barrier (§8.4).

### 8.2 Choosing questions: informative, diverse, random

A round does not ask every question. It draws a batch, like a minibatch in stochastic gradient
descent, so no fixed list steers a model and, over rounds, every question gets asked.

- **Informative and diverse: k-means++ seeding.** Draw the batch one pair at a time, each with
  probability proportional to its disagreement (the gap between the learner's rank and the
  answer) squared, times its distance from the pairs already drawn. Disagreement makes a question
  informative; the distance spreads the batch over the space. This is BADGE's use of k-means++
  for batch active learning (Ash et al., ICLR 2020).
- **Stochastic.** Each round draws a fresh batch with a fresh seed. An outlier pair can be drawn
  but cannot dominate every round. The `--min-cells` floor keeps tiny pseudobulks, the likeliest
  outliers, out of the pool.
- **Random sub-committees.** Each round, the answer comes from a random subset of the other models
  (query by bagging; Abe & Mamitsuka, ICML 1998). An odd model only sometimes answers, and a
  merge that holds across sub-committees is the more trustworthy.
- **Contentious pairs stay in the pool.** No label is given, but they can be asked again; a pair
  that stays contentious across rounds and partitions marks a difference only some models see.

### 8.3 Redrawing partitions (channel A)

The partition is **fixed within a round**, so that messages share an index. It is **redrawn
between rounds**, so that a label is not an artifact of one particular grouping.

- Models do not care about the partition. Their encoders are amortised, so a new partition only
  means new training rows. Their views are rebuilt each round.
- Channel-A labels are kept at **cell-group** resolution, as cannot-link constraints over a stable
  fine grouping such as the round-0 finest level. Each round moves the constraints onto the new
  partition by overlap:

  ```
  w_ab = Σ_(i,j) frac(a ∩ S_i) · frac(b ∩ S_j) · w_ij
  ```

  Constraints that survive several partitions build up weight. Constraints that depend on one
  grouping fade.
- What to vary: `--pb-refine-seed`, the projection seed, `--sort-dim`, `--knn-cells`. Keep the
  number of levels fixed, because changing it changes scale rather than resampling.
- Channel B is keyed by gene, so its labels need no transport.

### 8.4 Parallel devices

- One fit per GPU, each its own process, with the device chosen as today
  (`src/embed_common.rs` `to_device`).
- `senna critique` runs on the CPU, and so does collapsing `P_{r+1}` while round `r` trains.
- Rounds are **synchronous by default** (a barrier), so they are deterministic given their seeds.
  An asynchronous mode (pick up the newest labels whenever ready, as in codistillation) can come
  later if waiting for the slowest fit hurts.
- Fits that share a GPU must pin `--minibatch-size` and skip the `gpu_mem_fraction` probe,
  which assumes it is alone on the device.

## 9. Safeguards

The oracle is the other models, so it can share their mistakes: a pair every model merges is
never questioned, and repeated rounds pull the models towards agreement. The safeguards target
that.

- **Only far answers.** A model is never told to pull a pair together (§3.3), so a correct lone
  separation is never trained away.
- **Diverse voices.** The answer is only as good as the spread of model families behind it.
  Near-duplicates (topic and masked-topic) should count as one family, so that similar models
  cannot outvote different ones.
- **Each model's own likelihood.** Every round checks each model's held-out likelihood on the
  counts and rolls back a round that hurts it (§8.1). This is where the data still has the last
  word.
- **Small steps.** A small weight λ that ramps up, and a batch of questions per round (§8.2),
  not the whole pool.
- Partition and feature-axis hashes, and the round's seed, on every message (§7).
- A log per round of each model's merge rate per channel. It should fall across rounds; if it
  oscillates, stop.

## 10. Staged plan and results

Each stage can be checked on its own before the next one is built.

| stage | channel | builds | check |
|---|---|---|---|
| 0a | A | **built:** `senna critique`, report only. Views averaged over one run's partition, top-*K* questions, ranks per model, leave-one-out answers, merges and merge rate per model, `merged_by` per pair; `--cell-labels` checks the merges against known labels | Are merged pairs different cell types? |
| 0c | A | **built:** random sub-committees (`--committee`) and k-means++ question draws (`--questions`, `--seed`), writing per-model label files (§7) | Sampled labels stay precise and spread wider than the top-disagreement pairs. |
| 0b | B | the same for gene pairs, from the models' gene embeddings | Do topic merges join genes of known separate programs? |
| 1 | — | no-new-cells mode for `senna update` with explicit `--pb-from` | Continuing without labels is neutral. |
| 2 | A, B | `ExtraLossHook`; `--peer-labels` through `Rebase` for topic / vae / masked | One round lowers the model's merge rate without hurting its held-out likelihood. |
| 3 | A, B | peer negatives into bge / simba / gem; gene views into fne as relations | The same check, for the graph models. |
| 4 | A, B | the round loop in `senna run`, on parallel devices | Merge rates fall across rounds, and likelihoods hold. |
| 5 | A | redrawn partitions with constraint transport | Persistent constraints agree with stage 0a. |
| 6 | A ↔ B | channel coupling (§5) | Coupled questions yield merges at a higher rate than sampled ones. |

**Evaluation** follows active learning: model quality (merge rate, and with known labels the
label overlap of what a model keeps near) against the number of labels it has trained on. Expert
labels act as a simulated perfect oracle; they never enter the loop.

**Stage 0a results.**

- *BMMNC, topic + vae + svd + bge.* The topic model and the VAE carry most of their charges as
  merges at the finer levels; SVD disagrees most in both directions.
- *HCA bone marrow, donor BM1, the same four models, with `--cell-labels` (55 fine and 24 broad
  types; outputs in `paper-senna/results/critique-hca-bm1/`).*
  - **Merges are right.** The label compositions of a merged pair's two pseudobulks overlap
    0.00–0.01, against 0.18–0.63 for the same model's other near pairs.
  - **Contentious pairs favour the lone model.** For topic, vae and bge, the pairs they alone hold
    far share 0–7 % of a broad type: the lone separating model was right. SVD's share 28–45 %
    (base 14–32 %): some of its separations are over-splits.
  - **Merge rate per near pair:** SVD 0.06–0.13; topic, vae and bge 0.015–0.05.
  - **Far rule.** The others' median beyond P/4 passes (merge overlap ≤ 0.03 of the base at both
    label granularities); beyond P/8, 3K or 4K it does not. `max(2K, P/4)` finds 1628 merges,
    `max(3K, P/4)` 1282, at the same precision.
  - **Seven models** (adding masked-topic, masked-vae and simba): every model's merges still
    overlap ≤ 0.009. Merge rates rank SVD highest (0.06–0.16), then masked-vae; bge, simba and
    masked-topic lowest (0.007–0.023). With more voices, even far beyond P/8 passes.
- *Question sampling, HCA BM1, committees of 2, 30 questions per model and level.* The labels'
  overlap is 0.00–0.035 of each model's near-pair base, they cover 25–51 % more distinct
  pseudobulks than the same number of top-disagreement pairs, and they are reproducible per seed.
- *HCA donor BM2, topic + vae + svd + bge.* `max(2K, P/4)` passes again (1698 merges; ratio to
  base 0.017 fine, 0.029 broad), `max(3K, P/4)` finds 1412. Rules set by K alone fail here
  (beyond 5K: ratio 0.29), though beyond 5K passed on BM1.
  - The labels come from clustering in a PCA-like space, so they share some bias with separating
    models.

## 11. Follow-ups and open questions

**Follow-ups**

- **SVD learns through its features, not its geometry.** A merged pair is one whose difference
  lies outside the span of SVD's chosen genes, so the feedback acts on the gene set: the
  separating genes of each merged pair (`|x_a − x_b|` over `pb_gene.parquet`, weighted by the
  label) are summed over the model's merges and folded into the HVG ranking, at a fixed feature
  budget. SVD stays closed-form and unweighted, and the committee cannot bend its geometry, only
  point at genes it ignores. The same hint can serve any model with an HVG step, beside its loss
  term. Set aside: appending weighted difference columns before the solve, and a cannot-link
  constrained PCA (Zhang et al., SDM 2007), both of which change what SVD is.

- **§3.1 A hybrid mixing view.** Tried: a view from each model's cell-level kNN graph (PAGA-style
  connectivity between pseudobulks; Wolf et al., Genome Biology 2019), built by streaming each
  cell's HNSW neighbours into pseudobulk-pair edge counts, so no graph is kept. On HCA BM1 and
  BM2 its near pairs were more coherent (label overlap 0.48 vs 0.43, 0.56 vs 0.50) but its
  merges were not precise (ratio to base 0.33–0.53 vs 0.008–0.03): with a few neighbours per
  cell most pseudobulk pairs share no edge, tie at the worst rank, and read as far. Absence of
  edges is not evidence of distance. Next to try: rank a pseudobulk's connected neighbours by
  connectivity and the rest by centroid distance. The code is on the unmerged branch
  `critique-mixing`.
- **The pseudobulk tree as a supervisor.** The collapse already grows a tree over the
  pseudobulks, and each split carries its own test against noise (the Marchenko–Pastur edge and a
  two-group likelihood ratio, `{out}.pb_tree.json`). Pseudobulks on opposite sides of a split
  that passed the edge are negative pairs the data vouches for; pseudobulks within one small
  subtree, below any passing split, are positive pairs. It is a supervisor outside every model's
  latent, so it can answer what the committee cannot: pairs every model merges, and contentious
  pairs only one model separates. The tree comes from one run's random projection, so it needs
  the same label check before it is trusted.

**Open questions**

- **§3.3** Per-model reliability: SVD's far answers include over-splits. Whether to weight each
  model's answer by a reliability estimated without labels.
- **§3.3** How many model families are needed before the answers are trustworthy, and how to
  weight near-duplicates.
- **§3.4** The label weight: agreement count, rank gap, or both.
- **§8.2** What contentious pairs that persist across rounds and partitions should become: a
  report of differences only some models see, or questions for an outside oracle.
- **§4.1** Gene embeddings live on different axes and supports; whether ranks on the
  intersected axis are enough.
- **§6** A margin on the latent or on the decoded profile. This may differ by family.
- **§6** Whether `graph-embedding-util` can take explicit negatives, or only samples its own.
- **§8.1** Whether Adam resets at round boundaries matter in practice.

## 12. Related work

- Active learning with a committee: query by committee (Seung, Opper & Sompolinsky, COLT 1992;
  Freund et al., Machine Learning 1997); multi-view contention points, Co-Testing (Muslea, Minton
  & Knoblock, JAIR 2006); query by bagging (Abe & Mamitsuka, ICML 1998); diverse batches by
  k-means++ seeding, BADGE (Ash et al., ICLR 2020).
- Peers teaching each other: Deep Mutual Learning (Zhang et al., CVPR 2018); codistillation
  (Anil et al., ICLR 2018 — stale peer snapshots suffice); Mutual Mean-Teaching (Ge et al.,
  ICLR 2020).
- Peers choosing each other's samples: Co-teaching (Han et al., NeurIPS 2018); Co-teaching+
  (Yu et al., ICML 2019 — train on disagreements); JoCoR (Wei et al., CVPR 2020).
- Contrastive learning: NCE (Gutmann & Hyvärinen, 2010); InfoNCE (van den Oord et al., 2018);
  hard negatives (Robinson et al., ICLR 2021); false negatives (Chuang et al., NeurIPS 2020).
- Cluster connectivity from a cell kNN graph: PAGA (Wolf et al., Genome Biology 2019).
- Must-link / cannot-link constraints: Wagstaff et al., ICML 2001.

Citations were written from memory. Check each one before quoting it.
