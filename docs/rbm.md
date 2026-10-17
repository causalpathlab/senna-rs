# `senna rbm` — a feature-embedded restricted Boltzmann machine

Cells as the hidden units of a restricted Boltzmann machine over their reads,
with the weights factored through a feature embedding. Fitted on pseudobulks,
read out exactly on every cell, with no encoder.

```bash
senna rbm counts.zarr -o out --hidden bernoulli          # K = 64 units by default
senna rbm counts.zarr -o out --no-feature-modules        # free rows of ρ
senna rbm rna.zarr atac.zarr -o out --tie-symbols         # multiome, detected
senna rbm counts.zarr -o out --freeze-feature-embedding prior_run
```

## The model

A cell's reads `v` are the visible units: one multinomial per modality block
`m`, at the cell's own depth `N_m`. The hidden units `h` are one of three kinds
(`--hidden`):

| kind | support | log-partition `A(a)` | `E[h | a]` |
|---|---|---|---|
| bernoulli | {0, 1} | softplus(a) | σ(a) |
| binomial | {0..n} (`--trials`) | n softplus(a) | n σ(a) |
| poisson | {0..M} (`--max-count`) | log Σ_{h≤M} e^{a h}/h! | its mean |

The energy, with `s_f = D_m / N_m` on block `m`'s features (a cell is read per
unit depth against the training depth `D`):

```text
E(v, h) = −v·b − (s ⊙ v − μ)ᵀ W (h − λ) − c·h − Σ_k log μ(h_k)
W       = ρ αᵀ           ρ: features × H,  α: hidden units × H (one per block)
```

`μ` and `λ` centre the reads and the hidden units (a centred RBM; same machines,
better conditioned training). Given the reads the hidden units are
independent with natural parameters

```text
a(v) = c + (s ⊙ v − μ)ᵀ ρ αᵀ = c + α u(v),     u(v) = ρᵀ (s ⊙ v − μ)
```

so a cell reaches the hidden units only through `u(v)`, where its reads sit in
the feature embedding. Given `h`, every block's reads are
`Multinomial(N_m, softmax(b + s ⊙ ρ αᵀ (h − λ)))`: the hidden state maps to the
point `αᵀ(h − λ)` in the same space, and the features whose embeddings line up
with it get the reads.

**A missing modality drops out exactly.** Given `h` the blocks are
independent, so a block a cell has no reads in leaves `a(v)` and the free
energy untouched: the exact marginal over that block.

## Training: the two phases

The free energy `F(v) = −v·b + (s ⊙ v − μ)ᵀ W λ − Σ_k A(a_k(v))` is exact.
The likelihood gradient is the data's `∂F` minus the model's:

```text
Δρ_f ∝ ⟨ s_f v_f z(v) ⟩_data − ⟨ s_f v_f z(v) ⟩_model,      z(v) = αᵀ (E[h | v] − λ)
Δα_k ∝ ⟨ E[h_k | v] u(v) ⟩_data − ⟨ E[h_k | v] u(v) ⟩_model
```

- **Positive phase:** `ρ_f` is pulled toward the embedding position `z` of the
  pseudobulks that read feature `f`; `α_k` toward the centroid `u` of those
  that switch unit `k` on.
- **Negative phase:** the same statistics on persistent Gibbs chains (the
  model's fantasies) push back, so the embedding cannot just grow. Within a
  block the visible softmax's candidates are that block's features: a feature
  the model over-produces is pushed out of the fantasies' direction.

None of this is written by hand: the loss is
`(mean F(data) − mean F(fantasies)) / D` plus the penalties below, its
gradient comes from candle's backward through `W = ρ αᵀ`, and Adam takes the
step. Every collapse level is visited each epoch, coarse to fine; the data
term reads a pseudobulk at its expected reads at depth `D` (probabilities
rather than samples wherever the statistics allow, as Hinton advises), while
the fantasies are whole-read Gibbs samples.

## Following Hinton's practical guide

The recipe follows Hinton, *A Practical Guide to Training Restricted
Boltzmann Machines* (2010), where it applies:

- **Distributed codes need every unit in use.** `--sparsity-target p`
  (default 0.1) and `--sparsity-cost` hold a running estimate of every unit's
  mean activation near `p` by a cross-entropy penalty, so no unit dies or
  saturates and a cell is a combination of units.
- **Many units.** `-t` defaults to 64.
- **Weight decay on weights, not biases:** an L2 on the rows of `ρ` (or the
  module vectors) and of `α`, per row so its weight means the same at any
  feature count (`--embedding-l2`).
- **Watch for overfitting:** `--holdout` (default 5%) of the finest
  pseudobulks are left out of that level's training; every epoch logs their
  mean free energy per read minus as many training ones', and the number of
  units in use (`log_likelihood.parquet`: `heldout_gap`, `live_units`). A gap
  that keeps rising is overfitting; coarser levels still contain the held-out
  cells, so the check is conservative.

## Feature modules

Feature modules are on by default (`--feature-modules 50`; `--no-feature-modules`
for free rows, and off when a feature embedding is given). They compose every
row of `ρ` as `senna bge` does: a sparse
mixture of `M` shared module vectors plus a small residual of its own,
`ρ_r = Σ_m sparsemax(logits_r)_m μ_m + r_r`. The partition starts from the
finest pseudobulks' counts (per modality under a multiome layout, unless
symbols are tied), the membership is held at its start for the first quarter
of the epochs, and `bge`'s load-balance prior and residual ridge apply. A
feature with few reads gets a trained row through its module's siblings.
The module layer trains at a tenth of the learning rate: one module vector
moves every member at once, and the background module of flat features can
hold most of the axis. Writes `module_membership`, `module_dictionary` and
`module_residual` tables beside the feature embedding, as `senna bge` does.

## The training depth `D`

`--depth` (default 20) tempers the likelihood. The energy barriers between the
machine's modes grow with `D`, and persistent chains only mix while they stay
low: at a few hundred reads the hidden units saturate early and training
stalls with several cell states sharing one code, or diverges. Cells are read
against the same `D` (`s = D / N`), so a cell's latent does not depend on its
depth, and larger `D` gives sharper posteriors.

Lower `--learning-rate` if the log shows the free-energy gap swinging
widely, most often with binomial or poisson units.

## Starting point

`ρ` starts on the leading singular directions of the centred pseudobulk reads
(module vectors at their members' mean), and every hidden unit on its own
orthogonal row of `α`, all scaled together so the units' inputs vary by one
on average across pseudobulks; a unit aimed at a weak direction starts small
rather than amplified onto noise. Factors started small and equal grow along
one direction together and every unit ends on it; this keeps them apart.

`--freeze-`, `--init-` and `--lora-feature-embedding` take `ρ` from an earlier
run's feature table instead (any run that writes one, `senna rbm` included);
the table sets `H`. `--tie-symbols` gives features `{name}/{modality}` of the
same name one row of `ρ`, each modality keeping its own `α` and bias.

## Batches

Cells are divided by their batch's `δ` per feature (from the collapse), then
every block is scaled back to the cell's own depth before it is read.

## Outputs

| file | contents |
|---|---|
| `latent.parquet` | cell × K, `E[h | v]` |
| `pb_latent.parquet` | finest pseudobulks × K |
| `cell_score.parquet` | free energy and reconstruction log-likelihood per read |
| `feature_embedding.parquet` | `ρ` |
| `hidden_embedding.parquet` | every block's `α` |
| `dictionary.parquet` | `W = ρ αᵀ`, feature × K |
| `feature_bias.parquet` | `b` |
| `log_likelihood.parquet` | per epoch: free-energy gap, reconstruction, held-out gap, units in use |
| `module_membership`, `module_dictionary`, `module_residual` | with `--feature-modules` |
| `model.json`, `safetensors` | the machine |
| `senna.json` | run manifest (`senna view` opens it) |

References: Smolensky (1986); Welling, Rosen-Zvi and Hinton, *NIPS* (2004)
(exponential-family harmoniums); Salakhutdinov and Hinton, *NIPS* (2009)
(replicated softmax); Tieleman, *ICML* (2008) (persistent contrastive
divergence); Montavon and Müller (2012) (centring); Hinton, *A Practical
Guide to Training Restricted Boltzmann Machines* (2010).
