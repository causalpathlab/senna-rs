"""scVelo on a benchmark dataset, written in the form `compare.py` scores.

Standard pipeline on raw/DATASET.h5ad: filter_genes (20 shared counts),
normalize_per_cell over all genes, log1p, 2000 HVGs, PCA, neighbours, moments (30 PCs, 30 neighbours), velocity in MODE
(dynamical after recover_dynamics), velocity_graph, velocity_embedding on the
UMAP. Time is latent_time for dynamical, velocity_pseudotime otherwise.

Writes runs/scvelo_MODE_DATASET.cells.parquet: barcode, time, vx, vy (the
velocity on the UMAP).

usage: uv run --python 3.11 --with scvelo,"numpy<2","pyarrow<20" python baselines.py DATASET MODE [MODE ...]
  MODE: deterministic, stochastic, dynamical (stochastic needs numpy<2 on scvelo 0.3.4)
"""

import os
import sys

import anndata as ad
import pandas as pd
import scanpy as sc
import scvelo as scv

ROOT = os.environ.get("VELO_BENCH", os.path.expanduser("~/work/velo-bench"))


def run(dataset, mode):
    a = ad.read_h5ad(os.path.join(ROOT, "raw", f"{dataset}.h5ad"))
    a.var_names_make_unique()
    # normalise over all genes before the HVG subset: subsetting first changes
    # the size factors (pancreas CBDir +0.36 -> +0.18)
    scv.pp.filter_genes(a, min_shared_counts=20)
    scv.pp.normalize_per_cell(a)
    sc.pp.log1p(a)
    sc.pp.highly_variable_genes(a, n_top_genes=2000, subset=True)
    sc.pp.pca(a, n_comps=30)
    sc.pp.neighbors(a, n_pcs=30, n_neighbors=30)
    scv.pp.moments(a, n_pcs=None, n_neighbors=None)
    if mode == "dynamical":
        scv.tl.recover_dynamics(a, n_jobs=os.cpu_count())
    scv.tl.velocity(a, mode=mode)
    scv.tl.velocity_graph(a, n_jobs=os.cpu_count())
    scv.tl.velocity_embedding(a, basis="umap")
    if mode == "dynamical":
        scv.tl.latent_time(a)
        time = a.obs.latent_time
    else:
        scv.tl.velocity_pseudotime(a)
        time = a.obs.velocity_pseudotime
    v = a.obsm["velocity_umap"]
    out = pd.DataFrame({"barcode": a.obs_names, "time": time.values, "vx": v[:, 0], "vy": v[:, 1]})
    path = os.path.join(ROOT, "runs", f"scvelo_{mode}_{dataset}.cells.parquet")
    out.to_parquet(path, index=False)
    print(f"wrote {path}")


if __name__ == "__main__":
    for m in sys.argv[2:]:
        run(sys.argv[1], m)
