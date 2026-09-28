# `senna` design & methods notes

Separated by **status**, because these are not the same kind of document and reading a plan as if
it were a description of the code is how people end up debugging things that were never built.

## Methods — describes code that exists

| doc | what it is |
|---|---|
| [`deconvolve.md`](deconvolve.md) | The `senna deconvolve` model and its inputs. |

## Moved to lupin

Cell-type annotation and lineage live in [`lupin`](https://github.com/causalpathlab/lupin-rs),
and so do their write-ups: `lupin docs annotation`, `lupin docs grouping`,
`lupin docs ontology-plan` and `lupin docs rooting-plan`.

## Plans — design notes for code that does **not** exist

| doc | what it is |
|---|---|
| [`text-anchored-gene-space-plan.md`](text-anchored-gene-space-plan.md) | Anchoring the gene space to text descriptions of genes. Design only; no code. |

---

**Before trusting any annotation output**, check that the marker panel is on the embedding's
trained feature axis: `n_live / n_markers` in `{out}.panel_null.tsv` (or the "marker liveness" log
line) should be near 100%. If it is not, the panel's genes were projected rather than fitted, and
every downstream statistic is uninterpretable — including the ones designed to detect exactly
that. At gem's default `--n-hvg 0` every gene is trained, so this holds by construction; if you
run `--n-hvg > 0`, pass `senna gem --markers` pointing at *the same marker file* the annotation
uses. See `lupin docs annotation`, §1.

The per-cell feature matrices these commands read are produced by
[`faba`](https://github.com/causalpathlab/faba); see its docs for how they are built.
