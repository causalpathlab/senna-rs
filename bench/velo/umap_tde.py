"""UMAPs of bge vs tde embeddings: cells by time and by cluster, and
pseudobulks by their mean time. One column per run.

Cells come from {run}.cell_embedding.parquet; pseudobulks (finest level) from
{run}.pb_embedding.parquet, timed by TIME_RUN's unit_time (same seed, same
pbs). Cosine UMAP, 15 neighbours, seed 0.

usage: umap_tde.py DATASET TIME_COLUMN TIME_RUN OUT.png RUN [RUN ...]
"""

import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import umap  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from score_pb import ROOT  # noqa: E402
from score_tde import time_of  # noqa: E402
from score_tde_pb import unit_times  # noqa: E402


def layout(z):
    return umap.UMAP(n_neighbors=15, metric="cosine", random_state=0).fit_transform(z)


def main():
    dataset, col, time_run, out, runs = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5:]
    cells = pd.read_csv(os.path.join(ROOT, f"{dataset}.cells.tsv.gz"), sep="\t").set_index("barcode")
    if col not in cells.columns:
        extra = pd.read_csv(os.path.join(ROOT, f"{dataset}.{col}.tsv"), sep="\t").set_index("barcode")
        cells = cells.join(extra[col])
    ut, t, _ = unit_times(time_run)
    on = (ut.level == ut.level.max()).to_numpy()
    fin, pb_t = ut[on], t[on]
    clusters = sorted(cells.clusters.unique())
    cmap = plt.get_cmap("tab10")
    fig, ax = plt.subplots(3, len(runs), figsize=(5 * len(runs), 14), squeeze=False)
    for j, run in enumerate(runs):
        e = pd.read_parquet(f"{run}.cell_embedding.parquet")
        e = e.set_index(e.columns[0]) if e.index.dtype != object else e
        e = e[e.index.isin(cells.index)]
        u = layout(e.select_dtypes("number").to_numpy())
        c = cells.loc[e.index]
        sc = ax[0, j].scatter(u[:, 0], u[:, 1], c=time_of(c[col]), s=2, cmap="viridis")
        ax[0, j].set_title(f"{os.path.basename(run)}: cells by {col}")
        fig.colorbar(sc, ax=ax[0, j])
        for k, cl in enumerate(clusters):
            q = (c.clusters == cl).to_numpy()
            ax[1, j].scatter(u[q, 0], u[q, 1], s=2, color=cmap(k % 10), label=cl)
        ax[1, j].set_title("cells by cluster")
        ax[1, j].legend(fontsize=6, markerscale=4)
        p = pd.read_parquet(f"{run}.pb_embedding.parquet").set_index("pb").loc[fin.index]
        v = layout(p.to_numpy())
        sc = ax[2, j].scatter(v[:, 0], v[:, 1], c=pb_t, s=8, cmap="viridis")
        ax[2, j].set_title("finest pseudobulks by mean time")
        fig.colorbar(sc, ax=ax[2, j])
    for a in ax.ravel():
        a.set_xticks([])
        a.set_yticks([])
    fig.tight_layout()
    fig.savefig(out, dpi=90)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
