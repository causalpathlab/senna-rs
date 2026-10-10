"""Score PSEUDOBULK embeddings (phase 1, where the time context acts) of bge vs
tde runs against each pseudobulk's time.

A bge and a tde run at the same seed share their pseudobulks (the collapse
does not see the context), so the tde run's {run}.unit_time.parquet (per pb:
level, share of each time bin) gives the time of both. Per pb, its time is the
mean bin value (bins like "E7.25" read as 7.25). Metrics as in score_tde.py,
per level, with "type" the pb's dominant bin.

With --held-out HO_RUN (a tde run at the same seed given a held-out time
table): the pbs with no labelled cell in HO_RUN's unit_time are held out, and
  * held-out MAE  each held-out pb's time predicted as the mean over its K
                  nearest labelled pbs (same level), against the truth.

usage: score_tde_pb.py TIME_RUN RUN [RUN ...] [--held-out HO_RUN]
  kNN runs on CUDA through torch when available.
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from score_tde import K, knn, score  # noqa: E402


def unit_times(run):
    """The timed pseudobulks of `run`'s unit_time, their mean time, and a
    coarse label: the dominant bin, or the time quintile when continuous."""
    ut = pd.read_parquet(f"{run}.unit_time.parquet").set_index("pb")
    if "tau" in ut.columns:
        t = ut.tau.to_numpy()
        return ut, t, pd.qcut(t, 5, labels=False, duplicates="drop").astype(str)
    bins = [c for c in ut.columns if c != "level"]
    value = np.array([float(b.lstrip("E")) for b in bins])
    share = ut[bins].to_numpy()
    labelled = share.sum(1) > 0
    ut = ut[labelled]
    t = share[labelled] @ value / share[labelled].sum(1)
    return ut, t, np.array(bins)[share[labelled].argmax(1)]


def main():
    argv = sys.argv[1:]
    held_run = None
    if "--held-out" in argv:
        i = argv.index("--held-out")
        held_run = argv[i + 1]
        argv = argv[:i] + argv[i + 2 :]
    time_run, runs = argv[0], argv[1:]
    ut, t_all, dominant = unit_times(time_run)
    hidden = np.zeros(len(ut), bool)
    if held_run:
        ho, _, _ = unit_times(held_run)
        hidden = ~ut.index.isin(ho.index)
    rows = []
    for run in runs:
        e = pd.read_parquet(f"{run}.pb_embedding.parquet").set_index("pb")
        for level in sorted(ut.level.unique()):
            on = (ut.level == level).to_numpy()
            names = ut.index[on]
            z = e.loc[names].to_numpy()
            gap, same, far, near, rho = score(z, t_all[on], dominant[on], np.ones(on.sum(), bool))
            row = [os.path.basename(run), level, gap, same, far, near, rho, int(on.sum())]
            if held_run:
                h, t = hidden[on], t_all[on]
                _, idx = knn(z[~h], z[h], min(K, int((~h).sum())))
                row += [int(h.sum()), float(np.abs(t[~h][idx].mean(1) - t[h]).mean())]
            rows.append(row)
    cols = ["run", "level", "knn |Δt|", "knn bin", "far cos", "near cos", "geodesic ρ", "pbs"]
    if held_run:
        cols += ["held-out pbs", "held-out MAE"]
    df = pd.DataFrame(rows, columns=cols)
    print(df.to_string(index=False, float_format=lambda x: f"{x:+.3f}"))


if __name__ == "__main__":
    main()
