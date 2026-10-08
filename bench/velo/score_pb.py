"""Score a `senna tde-pb` run at pseudobulk resolution, every level.

Per pseudobulk: majority cluster and mean UMAP position of its cells. Reports
  * median tau per cluster (in trunk order) and Spearman(tau, trunk order);
  * CBDir on the dataset's transition edges: each pseudobulk's velocity v
    (in the base space) is projected onto the UMAP scVelo-style, a softmax over
    its k neighbours in the base space of cos(v, h_j - h_i), weighted unit
    displacements in UMAP minus their uniform mean; for every pseudobulk of
    cluster A, the mean cosine of that projection with the UMAP displacement
    to its UMAP neighbours of cluster B (UniTVelo's cross-boundary direction).
Optionally writes an arrow plot of the finest level.

usage: score_pb.py DATASET RUN_PREFIX [--plot out.png]
"""

import argparse
import os

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.neighbors import NearestNeighbors

ROOT = os.environ.get("VELO_BENCH", os.path.expanduser("~/work/velo-bench"))

TRUNK = {
    "pancreas": ["Ductal", "Ngn3 low EP", "Ngn3 high EP", "Pre-endocrine"],
    "dentategyrus": ["nIPC", "Neuroblast", "Granule immature", "Granule mature"],
}
EDGES = {
    "pancreas": [
        ("Ngn3 low EP", "Ngn3 high EP"),
        ("Ngn3 high EP", "Pre-endocrine"),
        ("Pre-endocrine", "Alpha"),
        ("Pre-endocrine", "Beta"),
        ("Pre-endocrine", "Delta"),
        ("Pre-endocrine", "Epsilon"),
    ],
    "dentategyrus": [
        ("OPC", "OL"),
        ("Radial Glia-like", "Astrocytes"),
        ("Neuroblast", "Granule immature"),
        ("Granule immature", "Granule mature"),
    ],
}


def unit(x):
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.maximum(n, 1e-12)


def project(h, v, umap, k=15, scale=10.0):
    """scVelo-style projection of base-space velocities onto the UMAP."""
    nn = NearestNeighbors(n_neighbors=min(k + 1, len(h))).fit(h)
    idx = nn.kneighbors(h, return_distance=False)[:, 1:]
    out = np.zeros_like(umap)
    for i in range(len(h)):
        d = h[idx[i]] - h[i]
        cos = (unit(d) @ unit(v[i][None])[0]).ravel()
        t = np.exp(scale * cos)
        t /= t.sum()
        du = unit(umap[idx[i]] - umap[i])
        out[i] = t @ du - du.mean(0)
    return out


def cbdir(clusters, umap, vel_umap, edges, k=15):
    nn = NearestNeighbors(n_neighbors=min(k + 1, len(umap))).fit(umap)
    idx = nn.kneighbors(umap, return_distance=False)[:, 1:]
    res = {}
    for a, b in edges:
        scores = []
        for i in np.where(clusters == a)[0]:
            nb = [j for j in idx[i] if clusters[j] == b]
            if nb:
                d = unit(umap[nb] - umap[i])
                scores.append(float((d @ unit(vel_umap[i][None])[0]).mean()))
        res[f"{a}->{b}"] = (np.mean(scores) if scores else np.nan, len(scores))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset")
    ap.add_argument("run")
    ap.add_argument("--plot")
    a = ap.parse_args()
    cells = pd.read_csv(os.path.join(ROOT, f"{a.dataset}.cells.tsv.gz"), sep="\t")
    pb = pd.read_parquet(f"{a.run}.pb_ode.parquet")
    cell_pb = pd.read_parquet(f"{a.run}.cell_pb.parquet")
    m = cell_pb.merge(cells, left_on="cell", right_on="barcode")
    hcols = [c for c in pb.columns if c.startswith("h") and c[1:].isdigit()]
    vcols = [c for c in pb.columns if c.startswith("v") and c[1:].isdigit()]
    trunk = TRUNK[a.dataset]
    for level in sorted(pb.level.unique()):
        lv = f"l{level}"
        g = m[m[lv] != ""].groupby(lv)
        info = pd.DataFrame(
            {
                "cluster": g.clusters.agg(lambda s: s.value_counts().index[0]),
                "ux": g.umap_x.mean(),
                "uy": g.umap_y.mean(),
            }
        )
        p = pb[pb.level == level].set_index("pb").join(info, how="inner")
        med = p.groupby("cluster").tau.median()
        on = p[p.cluster.isin(trunk)]
        rank = on.cluster.map({c: i for i, c in enumerate(trunk)})
        rho = spearmanr(on.tau, rank)[0]
        rho0 = spearmanr(on.tau_start, rank)[0]
        umap = p[["ux", "uy"]].to_numpy()
        print(f"== level {level}: {len(p)} pseudobulks")
        print("   median tau: " + ", ".join(f"{c} {med.get(c, np.nan):.2f}" for c in med.sort_values().index))
        print(f"   trunk Spearman: tau {rho:+.3f} (start {rho0:+.3f})")
        if not vcols:
            continue
        vu = project(p[hcols].to_numpy(), p[vcols].to_numpy(), umap)
        cb = cbdir(p.cluster.to_numpy(), umap, vu, EDGES[a.dataset])
        mean_cb = np.nanmean([s for s, _ in cb.values()])
        print(f"   CBDir mean {mean_cb:+.3f}: " + "; ".join(f"{e} {s:+.2f} (n={n})" for e, (s, n) in cb.items()))
        if a.plot and level == pb.level.max():
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(1, 2, figsize=(14, 6))
            for c in p.cluster.unique():
                q = p.cluster == c
                ax[0].scatter(umap[q, 0], umap[q, 1], s=8, label=c)
            ax[0].quiver(umap[:, 0], umap[:, 1], vu[:, 0], vu[:, 1], angles="xy", width=0.002)
            ax[0].legend(fontsize=7, markerscale=2)
            ax[0].set_title("velocity on the UMAP")
            sc = ax[1].scatter(umap[:, 0], umap[:, 1], c=p.tau, s=8, cmap="viridis")
            fig.colorbar(sc, ax=ax[1])
            ax[1].set_title("tau")
            fig.tight_layout()
            fig.savefig(a.plot, dpi=110)


if __name__ == "__main__":
    main()
