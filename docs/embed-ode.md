# `senna embed-ode` — one embedding of pseudobulks, genes and cells, shaped by an ode run

An embedding in the manner of `senna bge` (every pseudobulk a point, every
gene a vector and a baseline, the reads a softmax over the genes) in which
every lineage of a `senna ode` run is a smooth curve, and every pseudobulk is
held near its curve at the lineage and time the ode run put it. The axes then
carry the trajectories and whatever else the reads hold that the
trajectories do not. Every cell is placed afterwards by its own reads, near
its curve, and its velocity is the curves' slope there.

```bash
senna ode tracks.zarr -o ode                     # the trajectories first
senna embed-ode --from ode -o emb                # on the ode run's own data
senna embed-ode tracks.zarr --from ode -o emb --dim 16
senna view -f emb.senna.json                     # y: time, v: velocity field
```

## Inputs from the ode run

`--from` names an ode run (its prefix or `.senna.json`); any other kind is
refused. From it:

- **Data and batches**: its data files and batch files, unless given.
- **Pseudobulks**: its finest level of `cell_to_pb`, by its ids. Cells are
  matched to it by name; a cell it did not train on is left out.
- **Batch effects**: its `delta.parquet`, when it wrote one. Every cell's
  reads are divided by its batch's effect, then scaled back to its own
  total, as the ode run reads them.
- **Grid and trunk**: `--points`, `--sub`, `--kappa` and the trunk from its
  arguments, and its branch times (`branch_time.tsv`, on phase), each kept
  off the ends of the grid.
- **Every pseudobulk's posterior** `q_p(ℓ, k)` over lineage and grid
  position (`pb_time.parquet`), renormalised within each pseudobulk. A
  pseudobulk with no rows there has no prior.
- **Every cell's posterior**: its `p_L…` and `tau`, `tau_sd`
  (`cell_time.parquet`).

**Spliced reads only.** On a two-track backend, every gene's
`{gene}/count/spliced` row, named by the gene; the unspliced rows are not
read. The kinetics are already in `q`, through the ode run; a cell's own
unspliced share is what the ode run's jitter treats as a nuisance, and would
only give every gene a second row. On
a backend without tracks, its rows as they are. `--max-features` keeps the
most variable genes; by default every gene is embedded.

## The model

Per pseudobulk `p`: a point `x_p` in `--dim` (`H`) dimensions and its reads
`y_p`. Per gene `g`: a vector `w_g` and a baseline `b_g`. Per lineage `ℓ`: a
curve `μ_ℓ(k)` on the ode run's grid, `n = (points − 1) sub + 1` positions at
phase `k/(n − 1)`. Every dimension of a curve is a Matérn-5/2 path on phase
of length scale `kappa`, carried whitened (`z = C η`, `η ~ N(0, I)`), with
one amplitude `a`:

```text
no trunk:  μ_ℓ = a z_ℓ
trunk:     μ_0 = a z_0,   μ_ℓ = μ_0 + a r_ℓ ⊙ z_ℓ,   r_ℓ(k) = √2 (1 − exp(−((k/(n−1) − t_b,ℓ)₊/κ)²))
```

with the branch times `t_b` fixed at the ode run's, so the lineages agree
before their branch and part smoothly after it. Should the prior on every
position have no Cholesky factor, the curves are carried on the `points`
nodes and drawn straight between them.

The fit minimises, every term at weight one,

```text
L = − Σ_p Σ_g y_pg log softmax_g(x_pᵀ w_g + b_g)
    + Σ_p [ (1/2σ²) Σ_{ℓ,k} q_p(ℓ,k) |x_p − μ_ℓ(k)|² + H log σ ]
    + ½|w|² + ½|η|² + ½ (log a)² + ½ G b̄²
```

- The softmax is over every gene, as in `senna bge`; scores are clamped to
  `±30`.
- The bracket sums over the pseudobulks with a prior. Each is one point under
  one Gaussian whatever its depth (not weighted by its reads), and the
  normaliser `H log σ` is kept, so `σ` (how far a pseudobulk may sit off its
  curve) is learned, not set.
- `½|w|²` also pins the shift every `w_g` can share (adding `x_pᵀ v` to every
  gene's score leaves the softmax as it is); `½ G b̄²` pins the shift every
  `b_g` can share.

AdamW (no weight decay) at `--lr` for `--epochs`, over every pseudobulk at
once, or with `--pb-per-step` over a shuffled partition of them, each part's
terms scaled so its mean over a partition is the full loss. The start: `x`
from a PCA of the pseudobulks' log1p CPM, `w` small and random, `b` every
gene's centred log share, every curve at the origin, `a = σ = 1`.

## Cells

With the genes and the curves fixed, every cell `c` is placed by its own
reads `n_c`:

```text
θ_c = argmin_θ  − Σ_g n_cg log softmax_g(θᵀ w_g + b_g) + (1/2σ²) |θ − M_c|²,
M_c = Σ_{ℓ,k} q_c(ℓ,k) μ_ℓ(k)
```

the pseudobulk fit's objective for one point. `q_c` is the cell's ode
posterior on the grid: its lineage posterior `p_ℓ` spread over the positions
as a Gaussian in phase about `τ / T_ℓ`, of sd `√((τ_sd/T_ℓ)² + 1/(n − 1)²)`,
`T_ℓ` the lineage's duration (`time / phase` over its `pb_time` rows, since
the time at a position is its phase times the mean duration). A cell the ode
run gave no time takes its pseudobulk's `q_p`; with neither, it sits at its
pseudobulk's point with no velocity. Solved by Newton's method from `M_c`,
each cell's own `H × H` system, a block of `--block-size` cells at a time.

A cell's velocity is `Σ_{ℓ,k} q_c(ℓ,k) dμ_ℓ/dphase (k)`, the curves' slope
(a central difference on the grid, second-order one-sided at the ends) where
the cell's posterior puts it, in the embedding's coordinates.

## Outputs

| file | holds |
|---|---|
| `{out}.cell_embedding.parquet` | every placed cell's `θ_c`, columns `h0…h{H−1}` |
| `{out}.cell_velocity.parquet` | every cell's velocity, the same rows and columns |
| `{out}.pb_embedding.parquet` | every pseudobulk's `x_p`, keyed by its ode pseudobulk id |
| `{out}.feature_embedding.parquet` | every gene's `w_g`, keyed `feature` by the gene's name, as `senna bge`'s |
| `{out}.feature_bias.parquet` | every gene's `b_g`, keyed `feature`, column `bias` |
| `{out}.trajectory.parquet` | every lineage's curve: `lineage`, `position`, `phase`, `h0…h{H−1}` |
| `{out}.cell_time.parquet` | the ode run's, as it was |
| `{out}.cell_to_pb.parquet` | the ode run's partition, every level, on the cells placed |
| `{out}.senna.json` | the run manifest (kind `embed-ode`) `senna view` and the MCP tools read |

## In the viewer and over MCP

The map is laid out from `cell_embedding` as an embedding (no z-scoring),
as for `senna bge`. In `senna view`, `y` colours the cells by the ode run's
time, `v` draws the velocity field (streamlines, arrows, off) and `lineage`
is a grouping; markers and a gene's expected level come from
`feature_embedding` and `feature_bias`, as for `senna bge`. Over MCP,
`run_fit` with `method: "embed-ode"` and `args: ["--from", "<ode run>"]`
fits one (pass the data files too, as `run_fit` asks for them), `view_open`
opens it, `view_map` takes `time` and `velocity`, and `view_cell` gives a
cell's time. The curves in `trajectory.parquet` are not drawn on the map.

`senna run` lists `embed-ode` after `ode`; its `--from` must be filled in,
and when it names an ode fit queued before it, it waits for that fit.
