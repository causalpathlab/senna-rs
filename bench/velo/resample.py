"""How often does `senna ode` call the same direction on resampled cells?

Each replicate r keeps a random `--keep` share of the cells (by writing a copy
of the base fit's cell embedding with the other rows dropped; `senna ode`
leaves cells without a base state out of every level) and refits, one
run per GPU at a time; the fit is deterministic given its data, so the
subsample is the only thing that varies. Replicates already on disk are reused.
`--halves` instead splits the cells into two disjoint halves per pair of
replicates: overlapping subsamples share most cells and flatter the agreement,
so the halves are the honest stability check.
`--base-seeds N --total BEANS` also refits the base: `senna bge` on BEANS
with seeds 0..N (cached as OUT_PREFIX.base{s}), and replicate r subsamples
base r mod N. The direction moves with the base fit (its pseudobulks and the
diffusion start), so one base understates the variation.

Per replicate: the level-0 orientation losses (from `{run}.summary.json` if
the run wrote one, else the log line), the finest level's Spearman of tau with
the reference (each pseudobulk's mean reference over its cells: trunk rank,
or the embryonic stage on erythroid), and its mean CBDir when the run writes
velocities. Then the agreement rate, the margin distribution, and how well
replicates agree on the order of the cells they share.

usage: resample.py DATASET TRACKS BASE OUT_PREFIX [--reps 20] [--keep 0.8 | --halves]
                   [--base-seeds N --total BEANS]
                   [--devices 0,1] [--senna senna] [-- extra senna ode args]
"""

import argparse
import itertools
import json
import os
import queue
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from score_pb import ROOT, cbdir, edges, project, reference  # noqa: E402

ORIENTATION = re.compile(
    r"level 0: loss per read ([-+\d.eE]+) \(starting order\) vs ([-+\d.eE]+) \(reversed\)"
)


def subsample_base(base, out, keep, r, halves):
    """Write `{out}.cell_embedding.parquet`: replicate r's rows of `base` —
    a random `keep` share, or under `halves` one half of a random split
    (replicates 2k and 2k+1 are the two disjoint halves of split k)."""
    path = f"{out}.cell_embedding.parquet"
    if os.path.exists(path):
        return
    table = pq.read_table(f"{base}.cell_embedding.parquet")
    n = table.num_rows
    if halves:
        perm = np.random.default_rng(r // 2).permutation(n)
        rows = perm[: n // 2] if r % 2 == 0 else perm[n // 2 :]
    else:
        rows = np.random.default_rng(r).choice(n, round(keep * n), replace=False)
    pq.write_table(table.take(np.sort(rows)), path)


def fit_base(args, s, devices):
    """The base fit with seed `s`, unless it is on disk."""
    out = f"{args.out}.base{s}"
    if os.path.exists(f"{out}.cell_embedding.parquet"):
        return
    dev = devices.get()
    try:
        cmd = [args.senna, "bge", args.total, "-o", out, "--seed", str(s), "--skip-etm",
               "--device", "cuda", "--device-no", dev]
        with open(f"{out}.log", "w") as log:
            subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=True)
    finally:
        devices.put(dev)


def fit(args, extra, r, devices):
    run = f"{args.out}.rs{r}"
    if os.path.exists(f"{run}.pb_ode.parquet") and os.path.exists(f"{run}.log"):
        return
    source = f"{args.out}.base{r % args.base_seeds}" if args.base_seeds else args.base
    base = f"{run}.base"
    subsample_base(source, base, args.keep, r, args.halves)
    dev = devices.get()
    try:
        cmd = [args.senna, "ode", args.tracks, "--base", base, "-o", run, "--device", "cuda", "--device-no", dev, *extra]
        env = {**os.environ, "RUST_LOG": os.environ.get("RUST_LOG", "info")}
        with open(f"{run}.log.tmp", "w") as log:
            subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, env=env, check=True)
        os.replace(f"{run}.log.tmp", f"{run}.log")
    finally:
        devices.put(dev)


def orientation(run):
    """Level-0 losses per read (starting order, reversed), or NaNs."""
    if os.path.exists(f"{run}.summary.json"):
        with open(f"{run}.summary.json") as f:
            return tuple(json.load(f)["orientation_loss"])
    with open(f"{run}.log") as f:
        m = ORIENTATION.search(f.read())
    return (float(m[1]), float(m[2])) if m else (np.nan, np.nan)


def score(dataset, cells, run):
    """Finest level's Spearman with the reference and mean CBDir, and every
    cell's tau (its finest pseudobulk's)."""
    pb = pd.read_parquet(f"{run}.pb_ode.parquet")
    finest = pb.level.max()
    pb = pb[pb.level == finest].set_index("pb")
    cell_pb = pd.read_parquet(f"{run}.cell_pb.parquet")
    m = cell_pb.merge(cells, left_on="cell", right_on="barcode")
    m = m[m[f"l{finest}"] != ""]
    g = m.groupby(f"l{finest}")
    info = pd.DataFrame(
        {
            "ref": g.ref.mean(),
            "cluster": g.clusters.agg(lambda s: s.value_counts().index[0]),
            "ux": g.umap_x.mean(),
            "uy": g.umap_y.mean(),
        }
    )
    p = pb.join(info, how="inner")
    on = p.ref.notna()
    rho = spearmanr(p.tau[on], p.ref[on])[0]
    cb = np.nan
    hcols = [c for c in p.columns if c.startswith("h") and c[1:].isdigit()]
    vcols = [c for c in p.columns if c.startswith("v") and c[1:].isdigit()]
    if vcols:
        umap = p[["ux", "uy"]].to_numpy()
        vu = project(p[hcols].to_numpy(), p[vcols].to_numpy(), umap)
        cb = np.nanmean([s for s, _ in cbdir(p.cluster.to_numpy(), umap, vu, edges(dataset)).values()])
    tau = m.set_index("cell")[f"l{finest}"].map(p.tau)
    return rho, cb, tau


def main():
    argv = sys.argv[1:]
    extra = argv[argv.index("--") + 1 :] if "--" in argv else []
    argv = argv[: argv.index("--")] if "--" in argv else argv
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset")
    ap.add_argument("tracks")
    ap.add_argument("base")
    ap.add_argument("out")
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--keep", type=float, default=0.8)
    ap.add_argument("--devices", default="0,1")
    ap.add_argument("--senna", default="senna")
    ap.add_argument("--halves", action="store_true", help="disjoint 50%% splits instead of --keep")
    ap.add_argument("--base-seeds", type=int, default=0, help="refit the base with this many seeds")
    ap.add_argument("--total", help="the base fit's input (with --base-seeds)")
    args = ap.parse_args(argv)
    if args.base_seeds and not args.total:
        ap.error("--base-seeds needs --total, the base fit's input")
    if not 0 < args.keep < 1:
        ap.error("--keep must be in (0, 1): with every cell the replicates are identical")

    devices = queue.Queue()
    for d in args.devices.split(","):
        devices.put(d)
    with ThreadPoolExecutor(devices.qsize()) as pool:
        for f in [pool.submit(fit_base, args, s, devices) for s in range(args.base_seeds)]:
            f.result()
        for f in [pool.submit(fit, args, extra, r, devices) for r in range(args.reps)]:
            f.result()

    cells = pd.read_csv(os.path.join(ROOT, f"{args.dataset}.cells.tsv.gz"), sep="\t")
    cells["ref"] = reference(cells, args.dataset)
    rows, taus = [], {}
    for r in range(args.reps):
        run = f"{args.out}.rs{r}"
        start, rev = orientation(run)
        rho, cb, taus[r] = score(args.dataset, cells, run)
        rows.append({"rep": r, "loss_start": start, "loss_reversed": rev,
                     "margin": abs(start - rev), "spearman": rho, "cbdir": cb})
    df = pd.DataFrame(rows)
    df.to_csv(f"{args.out}.resample.tsv", sep="\t", index=False)

    pair = []
    for a, b in itertools.combinations(taus, 2):
        shared = taus[a].index.intersection(taus[b].index)
        pair.append(spearmanr(taus[a][shared], taus[b][shared])[0])
    agree = (df.spearman > 0).mean()
    q = df.margin.quantile([0.05, 0.5, 0.95])
    print(df.to_string(index=False, float_format=lambda x: f"{x:+.4f}"))
    print(f"\n{args.dataset}: keep {args.keep}, {args.reps} replicates")
    print(f"  direction right in {agree:.0%} (Spearman median {df.spearman.median():+.3f}, "
          f"min {df.spearman.min():+.3f})")
    print(f"  margin per read: median {q[0.5]:.5f}, 5-95% {q[0.05]:.5f}-{q[0.95]:.5f}")
    if df.spearman.lt(0).any():
        print(f"  margin when wrong: median {df.margin[df.spearman < 0].median():.5f}")
    if pair:
        print(f"  cell tau between replicates: Spearman median {np.median(pair):+.3f}, "
              f"min {np.min(pair):+.3f}")


if __name__ == "__main__":
    main()
