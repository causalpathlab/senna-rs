"""Every gene's splicing beta against degradation gamma (log axes), coloured
by its transcription (alpha) module, marker genes labelled; prints how many
sit at the rate caps.

usage: rates.py RUN_PREFIX OUT.png
"""

import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

MARKERS = ["Sox9", "Hes1", "Spp1", "Neurog3", "Fev", "Pax4", "Ins1", "Ins2", "Gcg", "Sst", "Ghrl", "Pyy", "Chga"]
CAP = np.exp(7.0)

run, out = sys.argv[1:3]
g = pd.read_parquet(f"{run}.gene_ode.parquet")
k = pd.read_parquet(f"{run}.module_kinetics.parquet")

fig, ax = plt.subplots(figsize=(9, 8))
cmap = plt.get_cmap("tab20", max(g.module.max() + 1, 2))
for m in sorted(g.module.unique()):
    q = g.module == m
    ax.scatter(g.beta[q], g.gamma[q], s=7, color=cmap(m), alpha=0.7,
               label=f"m{m} on {k.t_on[m]:.2f} off {k.t_off[m]:.2f}")
for _, r in g[g.gene.isin(MARKERS)].iterrows():
    ax.annotate(r.gene, (r.beta, r.gamma), fontsize=8)
ax.set_xscale("log")
ax.set_yscale("log")
lo, hi = min(g.beta.min(), g.gamma.min()), max(g.beta.max(), g.gamma.max())
ax.plot([lo, hi], [lo, hi], c="grey", lw=0.6, ls=":")
ax.set_xlabel("β (splicing, per unit τ)")
ax.set_ylabel("γ (degradation, per unit τ)")
ax.set_title(f"{len(g)} genes, coloured by α module")
ax.legend(fontsize=6, ncol=2, markerscale=2, loc="lower right")
fig.tight_layout()
fig.savefig(out, dpi=110)

at_cap = ((g.beta > 0.99 * CAP) | (g.gamma > 0.99 * CAP)).sum()
print(f"genes {len(g)}; at a rate cap {at_cap}; median log10 β {np.log10(g.beta).median():.2f}, "
      f"log10 γ {np.log10(g.gamma).median():.2f}, log10 γ/β {np.log10(g.gamma / g.beta).median():.2f}")
spread = g.groupby("module").apply(lambda d: pd.Series({
    "sd_log_beta": np.log(d.beta).std(), "sd_log_gamma": np.log(d.gamma).std()}), include_groups=False)
print("within-module sd of log rates (median over modules):", spread.median().round(2).to_dict())
print(g[g.gene.isin(MARKERS)][["gene", "module", "beta", "gamma"]].round(2).to_string(index=False))
print(f"wrote {out}")
