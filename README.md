# senna

Stochastic embedding with nearest-neighbourhood adjustment.

**Install:** `cargo install senna-rs` (binary: `senna`).

Annotation, lineage, pseudotime, and plots live in [`lupin`](https://github.com/causalpathlab/lupin-rs); layout/clustering stay here.

**Set up fits in the terminal:** `senna run [DIR]` lets you pick data files and the batch labels of each, queue several methods (topic, masked-topic, masked-vae, masked-sbp, vae, svd, bge, simba, rbm, td-rbm), and change any of their flags. `O` sets an output header that every `--out` starts with (`exp1` gives `exp1_svd`; `results/` puts them all in that folder; any folder the header names, `~/` included, is made as soon as it is entered); `o` names one by hand. Every flag comes with its help, including the hidden ones (marked `·`; `a` hides them) and `--threads`. Ctrl+R shows the exact commands, checked as senna parses them, and Ctrl+R (or Enter) runs them. They run in turn with each one's progress and a scrollable log on screen, and `v` opens the results in `senna view`. Each command is saved first as `{out}.cmd.sh`, so `bash {out}.cmd.sh` runs it again. The script refuses to run over an existing `{out}.senna.json`, and `senna run` never writes over a script.

**Clones first, when donors differ:** with several donors, a donor-private CNV clone can be mixed into other donors by batch correction. `senna run` can queue `mung clones` ([mung-cnv](https://github.com/causalpathlab/mung-cnv), 0.2.7 or later, installed separately) before the fits. When it is done, you see its clones (cells per donor in clones; each clone's size, donor share, purity and malignancy) and choose whether the fits get `--cnv-clones`, run without it, or stop. mung is a separate program, as lupin is: senna reads its form from `mung describe clones`, starts `mung` (or `$SENNA_MUNG`), and reads only the `{out}.clones.parquet` it writes.

## Agents (MCP)

`senna mcp` serves senna's fits and its viewer over the
[Model Context Protocol](https://modelcontextprotocol.io) on stdin/stdout, so
any MCP client can drive them, with any model behind it (Claude, GPT, Gemini,
or a local one). senna fits and draws; annotating clusters with cell types is
[lupin](https://github.com/causalpathlab/lupin-rs)'s own server, whose rounds
senna shows as they are on disk.

### Connect a client

The server is one command, `senna mcp`. A client needs that, and where the
project is: the server keeps its records (below) in its working directory, or
in `LEGUME_MCP_PROJECT` when set. Set it whenever the client may start servers
somewhere else (desktop apps usually do). If the client cannot find `senna`,
give its full path (`which senna`).

Most clients (Claude Desktop, Cursor's `.cursor/mcp.json`, Gemini CLI's
`settings.json`, and others) take this shape:

```json
{
  "mcpServers": {
    "senna": {
      "command": "senna",
      "args": ["mcp"],
      "env": { "LEGUME_MCP_PROJECT": "/path/to/project" }
    }
  }
}
```

Claude Code:

```sh
claude mcp add senna -e LEGUME_MCP_PROJECT="$PWD" -- senna mcp
```

VS Code (`.vscode/mcp.json`):

```json
{ "servers": { "senna": { "type": "stdio", "command": "senna", "args": ["mcp"],
                          "env": { "LEGUME_MCP_PROJECT": "${workspaceFolder}" } } } }
```

Codex CLI (`~/.codex/config.toml`):

```toml
[mcp_servers.senna]
command = "senna"
args = ["mcp"]
env = { LEGUME_MCP_PROJECT = "/path/to/project" }
```

Key names differ between clients; the server needs only the command, its
argument and, optionally, the environment. Nothing is sent anywhere by senna:
the model sees what the tools return (text, and the maps as PNG images), and
where that goes is up to the client and model you choose (a local model keeps
it on the machine).

### What the agent does

The server tells the agent how to work when it connects (the MCP
`instructions`). Some clients do not pass those to the model; with one of them,
put this in the agent's own instructions (an `AGENTS.md`, a system prompt):

> Use senna's MCP tools to embed single-cell data and look at the result;
> annotation is lupin's own server. Open the day with `notebook` (kind `plan`):
> the question, the data, and how the result will be checked. `run_fit` runs a
> fit to its end (`run_help` lists a method's flags) and returns its run
> manifest; an existing run is not fitted again. A long fit runs with
> `background`, as a job to follow with `job_wait`, in a session of `senna run`
> the user can watch with `senna attach`. `view_open` opens a run (or a
> lupin round) and computes a missing layout or clustering first. `view_map`
> draws the map with its legend; its arguments stay set until changed, so pass
> only what changes. `view_markers` lists what sets a group apart, `view_cell`
> names a point and its groups, and `view_export` writes figures. World
> coordinates are the layout's, y up. Close with `notebook` (kind
> `conclusion`) and tell the user where the records are.

| Tool | Does |
|---|---|
| `run_fit` | a fit (`senna <method> <data> -o <out> <flags>`) as a child, its log as progress; returns the run manifest and its kind, or with `background` a job to follow, running on `senna run`'s screen in a session of its own (`senna attach`) that outlives the server |
| `run_help` | every flag of one method, with its help |
| `view_open` | opens a run or a lupin round: its maps, groupings and the groups shown, a missing layout or clustering computed first |
| `view_map` | the map as a PNG with its state and legend: layout, points, grouping, focus, a feature's level, window and zoom |
| `view_markers` | the features that set one group apart, by the model's expected fold change, and its markers from the run's table |
| `view_cell` | the point under a pixel of the last map, at a world point, or by name: its position and groups |
| `view_export` | the view as `.png`/`.pdf` figures; never replaces a file unless asked |
| `notebook` | (from legume-mcp) an entry in the day's lab notebook |
| `job_status`, `job_wait`, `job_cancel` | (from legume-mcp) a background fit's state, waiting for it, stopping it |

Fits and the layouts and clusterings a view needs run as child `senna`
processes. A fit with `background` runs as a job the agent follows with
`job_status` or `job_wait` (or a person stops from a watch view); jobs end with
the server, so a client that closes its session stops a background fit. An output the agent does not name goes to the day's results folder.

### Records

The server keeps the project's records, laid out after Noble's guide to
organizing computational biology projects [1] (see
[legume-mcp](https://github.com/causalpathlab/legume-mcp)):

- `results/<YYYY-MM-DD>/runall.sh`: every step that wrote something, in order,
  with paths relative to the project and steps skipped when their outputs
  exist, so `bash results/<day>/runall.sh` in a copy of the project redoes the
  work without the agent.
- `results/<YYYY-MM-DD>/notebook.md`: the agent's plan, observations and
  conclusions; the plan names the question, the data and the check, after [2].
- `results/INDEX.tsv`: one row per output with a one-line summary, for `grep`.

`LEGUME_MCP_RECORDS=off` keeps none.

### Watching

While an agent works, the server writes a live feed to
`$XDG_RUNTIME_DIR/legume-mcp/senna-<pid>/` and publishes the view it shows
after `view_open` and `view_map`. `senna view --watch` (or `--watch <pid>`)
follows it in senna's own viewer: the same run, round, map, grouping, focus and
window, with a pane of the agent's latest calls, what the latest is doing and
the last command it ran. Changing the view by hand stops following; `W` follows again.
legume-mcp's own watch view can also stop a call or leave the agent a note.
`LEGUME_MCP_WATCH=off` writes no feed.

[1] W. S. Noble, "A quick guide to organizing computational biology projects",
PLoS Comput Biol 5(7): e1000424 (2009). doi:10.1371/journal.pcbi.1000424

[2] W. S. Noble, "Ten simple rules for defining a computational biology
project", PLoS Comput Biol 19(1): e1010786 (2023). doi:10.1371/journal.pcbi.1010786

---

# Stochastic data Embedding with Nearest Neighbourhood Adjustment

## `senna indexed-topic` — Embedded Topic Model

The per-cell top-K feature window training (`indexed-topic` / `itopic`) factorizes the topic-feature distribution β through a shared per-gene embedding ρ — Dieng et al. (2020)'s ETM, on packed top-K inputs.

**Parameters**
- ρ ∈ ℝ^{D×H}: per-gene embedding. **Shared between encoder and decoder.**
- α ∈ ℝ^{K×H}: per-topic embedding (decoder).

**Model**

```text
encoder (input pooling):
    v_norm  = anscombe_lite(values, batch_null, μ_d)                   # [N, K]
    h       = Σ_k v_norm[k] · ρ[idx[k]]                                # [N, H]
    z       ∼ Gaussian(μ_z(h), σ_z(h)),    log θ = log_softmax(z)      # [N, K_topics]

decoder (ETM factorization):
    β_kd    = log_softmax_d(α_k · ρ_dᵀ)                                # [K_topics, D]
    log x̂_n = log(Σ_k θ_nk · exp β_k:)                                 # multinomial recon
```

**Per-batch slice (no [K, D] materialization).** For the union of top-K feature ids across the minibatch, only `ρ_S = ρ[S, :]` is gathered:

```text
β̃_kS = log_softmax_S(α · ρ_Sᵀ − log q_S)
```

where `log q_S` is the Jean et al. (2015) importance correction for sampled softmax (`q_s` = per-feature selection frequency). Per-batch cost is O(K·S·H) on the gradient path; the full β is only built on demand for output / evaluation.

**Why share ρ.** Each gradient step lands on the same `Var` from two paths — the encoder's bag-of-features pool *and* the decoder's α·ρᵀ logits — so feature embeddings are densely supervised even though only top-K features fire per cell. Parameter count drops from K·D to K·H + D·H (H ≪ D in typical configurations).

**Outputs.** In addition to the standard topic-model artifacts (`.dictionary.parquet`, `.latent.parquet`, `.safetensors`, etc.), `indexed-topic` writes:

- `{out}.feature_embedding.parquet` — D × H learned ρ. Directly usable for gene-gene similarity, clustering into programs, or initializing downstream models.

## `senna masked-vae` — masked imputation with an unconstrained latent (BERT-style)

The continuous-latent sibling of `masked-topic` (alias `bert`). Same masked-imputation
pipeline — PB-collapse training, shared per-gene embedding ρ, NB ETM head, encoder-only
cell inference — but the encoder emits a raw `z` with no simplex softmax. It is
deterministic and KL-free, like `masked-topic` and `masked-sbp`: the masking is the
regularizer.

**Model**

```text
encoder (masked, visible-pooled):
    h       = Σ_{k∈visible} v_norm[k] · ρ[idx[k]]            # bag-of-visible-genes
    z       = clamp(W_z · h)                                 # raw latent, NO softmax

decoder (NB ETM head, reused unchanged):
    θ       = softmax(z)                                     # gene-axis mixture weights
    β_kg    = softmax_d(α_k · ρ_dᵀ)                          # per-topic gene dist (full-vocab Zₖ)
    μ_g     = ℓ · Σ_t θ_t · β_{t,g}
    x_g     ∼ NB(μ_g, φ_g)        for masked g

loss = − Σ_{masked g} log NB(x_g | μ_g, φ_g)
```

**Latent vs masked-topic.** Both are deterministic and KL-free, and both hand the
decoder `log_softmax` of the encoder's logits, so they train the same model with the
same gradients. The difference is what is written out: `masked-topic` stores `log θ`,
`masked-vae` stores the raw pre-softmax `z`, which is the more natural input for a
downstream kNN or linear model. NB objective only. Outputs otherwise match
`masked-topic` (`.dictionary`, `.feature_embedding`, `.dispersion`, `.latent`, …).

> A KL bottleneck (`--kl-weight`) used to sit on top of the masked objective here. At
> its default weight it drove `z` to zero and every cell's θ to uniform, so it was
> removed along with the flag; a model recorded with `kl_weight` still replays, the
> value is ignored.

## `senna rbm` — feature-embedded restricted Boltzmann machine

Cells as the hidden units of a restricted Boltzmann machine over their reads,
with weights factored through a feature embedding, `W = ρ αᵀ`: a cell reaches
the hidden units only through where its reads sit in `ρ`. Visible units are a
multinomial per modality block at the cell's own depth; multiome inputs are
detected as `senna bge` detects them, and a cell missing a modality is read
exactly from the blocks it has. Hidden units are Bernoulli, binomial
or truncated Poisson (`--hidden`). Fitted on pseudobulks by
persistent contrastive divergence (Adam on candle's backward of the
free-energy gap), read out on every cell with no encoder. `senna docs rbm`
has the model, the two phases of the gradient and the role of `--depth`.

```bash
senna rbm counts.zarr -o out -t 10 --hidden bernoulli
```

## `senna vae` — scVI-style Gaussian VAE

The continuous-latent sibling of `senna topic`. Same data pipeline (batch-aware
pseudobulk collapse → multilevel hierarchy → dense VAE), but the latent is an
**unconstrained Gaussian** `z` instead of a simplex `θ`, paired with a gene-axis
softmax NB decoder (the scVI parameterization). Outputs are continuous **factors**
(cell × factor) and **loadings** (gene × factor) — not topic proportions and a
topic-gene dictionary.

**Model**

```text
encoder:
    z ∼ Gaussian(μ_z(x), σ_z(x))                       # [N, K] raw latent (no softmax)

decoder (scVI):
    π_nd = softmax_d(z_n · W + b)                       # gene distribution, sums to 1 over D
    μ_nd = library_n · π_nd
    x_nd ∼ NB(μ_nd, φ_d)                                # per-gene dispersion φ
```

**Why a separate decoder front.** A Gaussian `z` is not on the simplex, so it
cannot drive the topic decoders' mixture `logsumexp_k(log θ_k + log β_kd)` (that
assumes `θ` sums to 1). The gene-axis softmax `softmax_d(z·W)` is the matched
likelihood: it turns an arbitrary real vector into a valid gene distribution.

**Shared training loop.** The dense topic trainer is reused verbatim — with
`topic_smoothing = 0` the simplex smoothing step becomes a no-op, so the raw `z`
flows straight into the decoder's own NB likelihood. The topic-specific machinery
(anchor prior, NB-Fisher weighting, ambient mixture, empirical dictionary, feature
coarsening) does not apply to a continuous-factor model and is skipped. Inference
is encoder-only (no decoder refinement); `senna predict` recognizes the `vae`
model type and runs the encoder over held-out cells.

**Outputs.** `{out}.latent.parquet` (cell × factor `z`), `{out}.dictionary.parquet`
(gene × factor loadings `W`), `{out}.feature_mean.parquet` (per-gene mean rate
`μ_d`), plus the standard `.safetensors` / `.model.json` / `.senna.json`.
