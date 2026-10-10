"""One table for every method on a dataset, same metrics for all.

  spearman  cell-level Spearman of time with the reference (trunk rank, or
            the embryonic stage on erythroid), over the cells that have one
  cbdir     cell-level mean CBDir on the dataset's transition edges, from
            each method's velocity on the UMAP
  cbdir_pb  (senna ode only) the same at the finest pseudobulk level

Baselines are runs/{scvelo_*,cell2fate}_DATASET.cells.parquet (barcode, time,
vx, vy) from baselines*.py. A senna ode run gives every cell its finest
pseudobulk's tau and that pseudobulk's velocity projected on the UMAP.

usage: compare.py DATASET [--senna RUN_PREFIX ...]
"""

import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from score_pb import ROOT, cbdir, edges, project, reference  # noqa: E402


def senna_cells(cells, run, edges):
    """Every cell's time and UMAP velocity from its finest pseudobulk, and the
    pseudobulk-level CBDir (NaN without velocities)."""
    pb = pd.read_parquet(f"{run}.pb_ode.parquet")
    finest = pb.level.max()
    pb = pb[pb.level == finest].set_index("pb")
    lv = f"l{finest}"
    m = pd.read_parquet(f"{run}.cell_pb.parquet").merge(cells, left_on="cell", right_on="barcode")
    m = m[m[lv] != ""]
    g = m.groupby(lv)
    info = pd.DataFrame(
        {
            "cluster": g.clusters.agg(lambda s: s.value_counts().index[0]),
            "ux": g.umap_x.mean(),
            "uy": g.umap_y.mean(),
        }
    )
    p = pb.join(info, how="inner")
    p["vx"] = p["vy"] = np.nan
    cb_pb = np.nan
    hcols = [c for c in p.columns if c.startswith("h") and c[1:].isdigit()]
    vcols = [c for c in p.columns if c.startswith("v") and c[1:].isdigit()]
    if vcols:
        umap = p[["ux", "uy"]].to_numpy()
        vu = project(p[hcols].to_numpy(), p[vcols].to_numpy(), umap)
        p["vx"], p["vy"] = vu[:, 0], vu[:, 1]
        cb_pb = mean_cbdir(p.cluster.to_numpy(), umap, vu, edges)
    out = pd.DataFrame({"barcode": m.barcode.values})
    for c in ("tau", "vx", "vy"):
        out[c] = m[lv].map(p[c]).values
    return out.rename(columns={"tau": "time"}), cb_pb


def mean_cbdir(clusters, umap, vel, edges):
    return np.nanmean([s for s, _ in cbdir(clusters, umap, vel, edges).values()])


def score(cells, method, edges):
    m = cells.merge(method, on="barcode")
    on = m.ref.notna() & m.time.notna()
    rho = spearmanr(m.time[on], m.ref[on])[0]
    cb = np.nan
    if m.vx.notna().all():
        umap = m[["umap_x", "umap_y"]].to_numpy()
        cb = mean_cbdir(m.clusters.to_numpy(), umap, m[["vx", "vy"]].to_numpy(), edges)
    return rho, cb, len(m)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset")
    ap.add_argument("--senna", action="append", default=[])
    a = ap.parse_args()
    pairs = edges(a.dataset)
    cells = pd.read_csv(os.path.join(ROOT, f"{a.dataset}.cells.tsv.gz"), sep="\t")
    cells["ref"] = reference(cells, a.dataset)

    rows = []
    for path in sorted(glob.glob(os.path.join(ROOT, "runs", f"*_{a.dataset}.cells.parquet"))):
        name = os.path.basename(path).removesuffix(f"_{a.dataset}.cells.parquet")
        rows.append((name, *score(cells, pd.read_parquet(path), pairs), np.nan))
    for run in a.senna:
        method, cb_pb = senna_cells(cells, run, pairs)
        rows.append((f"senna ode {os.path.basename(run)}", *score(cells, method, pairs), cb_pb))
    df = pd.DataFrame(rows, columns=["method", "spearman", "cbdir", "cells", "cbdir_pb"])
    print(df.to_string(index=False, float_format=lambda x: f"{x:+.3f}"))


if __name__ == "__main__":
    main()
