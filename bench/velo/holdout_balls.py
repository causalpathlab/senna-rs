"""Hide the time of whole neighbourhoods, stratified by time bin.

From a previous run's cell embedding, per time bin: draw ball centres among
the bin's cells and hide each centre's K nearest cells (cosine, any bin) until
the bin has about SHARE of its cells hidden. Every bin keeps labelled and
hidden cells, so the test is interpolation, not extrapolation to an unseen bin.

Writes OUT (barcode, TIME_COLUMN) with the hidden cells left out, and prints
the hidden share per bin.

usage: holdout_balls.py DATASET TIME_COLUMN EMBEDDING_RUN OUT [--share 0.3] [--k 30]
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from score_pb import ROOT  # noqa: E402
from score_tde import knn  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset")
    ap.add_argument("time_column")
    ap.add_argument("run")
    ap.add_argument("out")
    ap.add_argument("--share", type=float, default=0.3)
    ap.add_argument("--k", type=int, default=30)
    a = ap.parse_args()
    cells = pd.read_csv(os.path.join(ROOT, f"{a.dataset}.cells.tsv.gz"), sep="\t").set_index("barcode")
    if a.time_column not in cells.columns:
        extra = pd.read_csv(os.path.join(ROOT, f"{a.dataset}.{a.time_column}.tsv"), sep="\t")
        cells = cells.join(extra.set_index("barcode")[a.time_column])
    e = pd.read_parquet(f"{a.run}.cell_embedding.parquet")
    e = e.set_index(e.columns[0]) if e.index.dtype != object else e
    e = e[e.index.isin(cells.index)]
    # Only timed cells are stratified and written; untimed ones stay untimed.
    e = e[cells.loc[e.index, a.time_column].notna().to_numpy()]
    z = e.select_dtypes("number").to_numpy()
    z = z / np.maximum(np.linalg.norm(z, axis=1, keepdims=True), 1e-12)
    raw = cells.loc[e.index, a.time_column]
    # A continuous time is stratified by deciles; bins are kept as they are.
    t = (
        pd.qcut(raw, 10, labels=False, duplicates="drop").astype(str).to_numpy()
        if pd.api.types.is_numeric_dtype(raw) and raw.nunique() > 20
        else raw.astype(str).to_numpy()
    )
    _, ball = knn(z, z, a.k)
    rng = np.random.default_rng(0)
    hidden = np.zeros(len(z), bool)
    for b in np.unique(t):
        in_bin = np.where(t == b)[0]
        target = a.share * len(in_bin)
        for c in rng.permutation(in_bin):
            if hidden[in_bin].sum() >= target:
                break
            if not hidden[c]:
                hidden[ball[c]] = True
    kept = pd.DataFrame({"barcode": e.index[~hidden], a.time_column: raw.to_numpy()[~hidden]})
    kept.to_csv(a.out, sep="\t", index=False)
    share = pd.Series(hidden).groupby(t).mean()
    print("hidden share per bin: " + ", ".join(f"{b} {s:.2f}" for b, s in share.items()))
    print(f"hidden {hidden.sum()} of {len(z)} cells -> {a.out}")


if __name__ == "__main__":
    main()
