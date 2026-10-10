"""Compare cell embeddings of `senna bge` and `senna tde` runs against known time.

Per run, on {run}.cell_embedding.parquet (cosine geometry, 15 neighbours):
  * knn |Δt|      mean time gap to a cell's neighbours (lower: time-coherent)
  * knn type      share of neighbours with the cell's own cluster
  * far / near    mean cosine of random cell pairs whose times differ by more
                  than half the time range, and of pairs within a tenth of it —
                  tde should push "far" down
  * geodesic ρ    Spearman of time with the kNN-graph geodesic distance from
                  the earliest bin's cells (a tree/path readout)
With --held-out FILE (the --time table a run was given), the scores are on the
cells whose time was hidden from that run only, plus
  * held-out MAE  each hidden cell's time predicted as the mean over its K
                  nearest LABELLED cells, against the truth; also per bin

usage: uv run --with pandas,pyarrow,numpy,scipy,scikit-learn,torch python score_tde.py
         DATASET TIME_COLUMN RUN [RUN ...] [--held-out FILE]
  kNN runs on CUDA through torch when available.
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.stats import spearmanr
from sklearn.neighbors import NearestNeighbors

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from score_pb import ROOT  # noqa: E402

K = 15


def knn(base, query, k):
    """Distances and indices of each query row's k nearest base rows
    (Euclidean): on CUDA through torch when there is one, else sklearn."""
    try:
        import torch

        if torch.cuda.is_available():
            b = torch.as_tensor(base, dtype=torch.float32, device="cuda")
            out_d, out_i = [], []
            for q in torch.as_tensor(query, dtype=torch.float32, device="cuda").split(4096):
                d, i = torch.cdist(q, b).topk(k, largest=False)
                out_d.append(d.cpu())
                out_i.append(i.cpu())
            return torch.cat(out_d).numpy(), torch.cat(out_i).numpy()
    except ImportError:
        pass
    return NearestNeighbors(n_neighbors=k).fit(base).kneighbors(query)


def time_of(values):
    s = pd.Series(values).astype(str).str.lstrip("E")
    return pd.to_numeric(s, errors="coerce").to_numpy()


def score(z, t, clusters, keep):
    z = z / np.maximum(np.linalg.norm(z, axis=1, keepdims=True), 1e-12)
    dist, idx = knn(z, z, K + 1)
    dist, idx = dist[:, 1:], idx[:, 1:]
    gap = np.abs(t[idx] - t[:, None]).mean(1)
    same = (clusters[idx] == clusters[:, None]).mean(1)
    rng = np.random.default_rng(0)
    a, b = rng.integers(0, len(z), (2, 200_000))
    dt = np.abs(t[a] - t[b])
    cos = (z[a] * z[b]).sum(1)
    span = np.nanmax(t) - np.nanmin(t)
    far = cos[dt > span / 2].mean()
    near = cos[(dt < span / 10) & (a != b)].mean()
    rows = np.repeat(np.arange(len(z)), K)
    g = csr_matrix((dist.ravel() + 1e-9, (rows, idx.ravel())), shape=(len(z), len(z)))
    root = np.where(t == np.nanmin(t))[0]
    d = dijkstra(g, directed=False, indices=root, min_only=True)
    ok = keep & np.isfinite(d)
    rho = spearmanr(d[ok], t[ok])[0]
    return gap[keep].mean(), same[keep].mean(), far, near, rho


def held_out_error(z, t, hidden):
    """|predicted − true| per hidden cell: the mean time of its K nearest
    labelled cells."""
    z = z / np.maximum(np.linalg.norm(z, axis=1, keepdims=True), 1e-12)
    _, idx = knn(z[~hidden], z[hidden], K)
    return np.abs(t[~hidden][idx].mean(1) - t[hidden])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset")
    ap.add_argument("time_column")
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--held-out")
    a = ap.parse_args()
    cells = pd.read_csv(os.path.join(ROOT, f"{a.dataset}.cells.tsv.gz"), sep="\t").set_index("barcode")
    if a.time_column not in cells.columns:
        # A time kept beside the cell table: $VELO_BENCH/DATASET.COLUMN.tsv.
        extra = pd.read_csv(os.path.join(ROOT, f"{a.dataset}.{a.time_column}.tsv"), sep="\t")
        cells = cells.join(extra.set_index("barcode")[a.time_column])
    hidden = None
    if a.held_out:
        given = pd.read_csv(a.held_out, sep="\t")
        hidden = set(cells.index) - set(given.iloc[:, 0][given[a.time_column].notna()])
    rows, per_bin = [], []
    for run in a.runs:
        e = pd.read_parquet(f"{run}.cell_embedding.parquet")
        e = e.set_index(e.columns[0]) if e.index.dtype != object else e
        e = e[e.index.isin(cells.index)]
        c = cells.loc[e.index]
        t = time_of(c[a.time_column])
        keep = np.isfinite(t)
        if hidden is not None:
            keep &= e.index.isin(list(hidden))
        z = e.select_dtypes("number").to_numpy()
        ok = np.isfinite(t)
        z, t, keep, cl = z[ok], t[ok], keep[ok], c.clusters.to_numpy()[ok]
        gap, same, far, near, rho = score(z, t, cl, keep)
        row = [os.path.basename(run), gap, same, far, near, rho, int(keep.sum())]
        if hidden is not None:
            err = held_out_error(z, t, keep)
            row.append(err.mean())
            per_bin.append(pd.Series(err).groupby(t[keep]).mean().rename(os.path.basename(run)))
        rows.append(row)
    cols = ["run", "knn |Δt|", "knn type", "far cos", "near cos", "geodesic ρ", "cells"]
    if hidden is not None:
        cols.append("held-out MAE")
    df = pd.DataFrame(rows, columns=cols)
    print(df.to_string(index=False, float_format=lambda x: f"{x:+.3f}"))
    if per_bin:
        print("\nheld-out MAE per time bin:")
        print(pd.concat(per_bin, axis=1).to_string(float_format=lambda x: f"{x:.3f}"))


if __name__ == "__main__":
    main()
