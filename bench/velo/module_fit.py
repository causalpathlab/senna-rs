"""Observed against fitted, per module, for a `senna tde-pb --no-embedding` run.

For the finest level's pseudobulks ordered by tau, every module gets a panel:
  * dots: the module's observed share of the pseudobulk's reads (log scale),
    and its observed unspliced share (right axis);
  * lines: the fitted log level log(u_m + s_m)(tau) shifted by the module's
    loading, and the fitted log(u/s)(tau) shifted to the observed mean.
Also prints each module's kinetics with the marker genes it holds.

usage: module_fit.py DATASET RUN_PREFIX OUT.png
"""

import gzip
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.io

ROOT = os.environ.get("VELO_BENCH", os.path.expanduser("~/work/velo-bench"))
MARKERS = {
    "pancreas": ["Sox9", "Hes1", "Spp1", "Neurog3", "Fev", "Pax4", "Ins1", "Ins2", "Gcg", "Sst", "Ghrl", "Pyy", "Chga"],
}
dataset, run, out = sys.argv[1:4]


def curves(k, t):
    """u, s of one module at times t (numpy port of kinetics::curves)."""
    def g2(a, b, d):
        x = (b - a) * d
        mid = np.exp(-0.5 * (a + b) * d)
        small = np.abs(x) < 1e-2
        safe = np.where(small, 1.0, b - a)
        return np.where(small, d * mid * (1 + x * x / 24), (np.exp(-a * d) - np.exp(-b * d)) / safe)

    def g3(a, b, c, d):
        lo, hi = min(a, b, c), max(a, b, c)
        mid = a + b + c - lo - hi
        if (hi - lo) * d.max() < 1e-2:
            return d * d / 2 * np.exp(-(a + b + c) / 3 * d)
        return (g2(lo, mid, d) - g2(mid, hi, d)) / (hi - lo)

    def piece(A, B, u0, s0, lam, be, ga, d):
        z = np.zeros_like(d)
        u = A * g2(z, be, d) + u0 * np.exp(-be * d) + B * g2(lam, be, d)
        s = s0 * np.exp(-ga * d) + A * g2(z, ga, d) + (be * u0 - A) * g2(be, ga, d) + be * B * g3(lam, be, ga, d)
        return u, s

    r, lam, be, ga = k.basal, k["lambda"], k.beta, k.gamma
    dur = k.t_off - k.t_on
    d1 = np.clip(t - k.t_on, 0, dur)
    d2 = np.maximum(t - k.t_off, 0)
    u1, s1 = piece(1.0, r - 1.0, r / be, r / ga, lam, be, ga, d1)
    a_end = 1 - (1 - r) * np.exp(-lam * d1)
    return piece(r, a_end - r, u1, s1, lam, be, ga, d2)


kin = pd.read_parquet(f"{run}.module_kinetics.parquet")
genes = pd.read_parquet(f"{run}.gene_ode.parquet")
pb = pd.read_parquet(f"{run}.pb_ode.parquet")
cell_pb = pd.read_parquet(f"{run}.cell_pb.parquet")
finest = pb.level.max()
p = pb[pb.level == finest].reset_index(drop=True)
lv = f"l{finest}"

# Observed per-pb module reads from the tracks backend's MatrixMarket export.
mtx = os.path.join(ROOT, "mtx", f"{dataset}_tracks")
feat = [l.strip() for l in gzip.open(os.path.join(mtx, "features.tsv.gz"), "rt")]
bc = [l.strip() for l in gzip.open(os.path.join(mtx, "barcodes.tsv.gz"), "rt")]
X = scipy.io.mmread(gzip.open(os.path.join(mtx, "matrix.mtx.gz"))).tocsr()
fi = {f: i for i, f in enumerate(feat)}
pbi = {n: i for i, n in enumerate(p.pb)}
cmap = dict(zip(cell_pb.cell, cell_pb[lv]))
col = np.array([pbi.get(cmap.get(b, ""), -1) for b in bc])
keep = col >= 0
A = scipy.sparse.csr_matrix((np.ones(keep.sum()), (col[keep], np.where(keep)[0])), shape=(len(p), len(bc)))
S = (A @ X[[fi[f"{g}/count/spliced"] for g in genes.gene]].T).toarray()
U = (A @ X[[fi[f"{g}/count/unspliced"] for g in genes.gene]].T).toarray()
mods = genes.module.to_numpy()
M = len(kin)
tot = np.stack([(S + U)[:, mods == m].sum(1) for m in range(M)], 1)
uns = np.stack([U[:, mods == m].sum(1) for m in range(M)], 1)
share = tot / tot.sum(1, keepdims=True)

order = np.argsort(p.tau.to_numpy())
t = p.tau.to_numpy()[order]
grid = np.linspace(0.0, 1.0, 200)
cols = 6
rows = int(np.ceil(M / cols))
fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.2, rows * 2.6), squeeze=False)
for m in range(M):
    ax = axes[m // cols, m % cols]
    k = kin.iloc[m]
    u, s = curves(k, grid)
    obs = np.log(share[order, m] + 1e-9)
    ax.scatter(t, obs, s=3, c="tab:blue", alpha=0.5)
    lvl = np.log(u + s)
    ax.plot(grid, lvl - lvl.mean() + obs.mean(), c="tab:blue", lw=1.5)
    ax2 = ax.twinx()
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio_obs = np.log((uns[order, m] + 0.5) / (tot[order, m] - uns[order, m] + 0.5))
    ax2.scatter(t, ratio_obs, s=3, c="tab:red", alpha=0.4)
    lr = np.log(u / s)
    ax2.plot(grid, lr - lr.mean() + ratio_obs.mean(), c="tab:red", lw=1.5)
    for x in (k.t_on, k.t_off):
        if 0 <= x <= 1:
            ax.axvline(x, c="k", lw=0.6, ls=":")
    marks = [g for g in MARKERS.get(dataset, []) if g in set(genes.gene[mods == m])]
    ax.set_title(f"m{m} ({int(k.genes)}) {' '.join(marks)}", fontsize=8)
    ax.tick_params(labelsize=6)
    ax2.tick_params(labelsize=6)
for m in range(M, rows * cols):
    axes[m // cols, m % cols].axis("off")
fig.suptitle("blue: module share of reads (log) vs fitted level; red: log u/s vs fitted ratio; dotted: switch on/off", fontsize=10)
fig.tight_layout()
fig.savefig(out, dpi=90)

pd.set_option("display.width", 200)
k = kin.copy()
k["markers"] = [" ".join(g for g in MARKERS.get(dataset, []) if g in set(genes.gene[mods == m])) for m in range(M)]
print(k.round(3).to_string())
print(f"switch on in (0,1): {((k.t_on > 0) & (k.t_on < 1)).sum()}, off inside (<0.99): {(k.t_off < 0.99).sum()}, "
      f"rates at cap (>1000): {((k['lambda'] > 1000) | (k.beta > 1000) | (k.gamma > 1000)).sum()}, "
      f"median gamma/beta {np.median(k.gamma / k.beta):.1f}")
print(f"wrote {out}")
