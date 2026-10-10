"""cell2fate on a benchmark dataset, in the same form as baselines.py.

Follows cell2fate's pancreas tutorial: get_training_data (all cells, 20
shared counts, 3000 genes), get_max_modules, train the dynamical model,
export the posterior, total velocity into the "Velocity" layer, then scVelo's
velocity_graph / velocity_embedding on that layer. Time is the posterior's
"Time (hours)".

Written from the tutorial without running it here; check the names against
the installed cell2fate if a call fails.

Writes runs/cell2fate_DATASET.cells.parquet: barcode, time, vx, vy.

usage: (in an env with cell2fate and scvelo) python baselines_c2f.py DATASET
"""

import os
import sys

import cell2fate as c2f
import anndata as ad
import pandas as pd
import scvelo as scv

ROOT = os.environ.get("VELO_BENCH", os.path.expanduser("~/work/velo-bench"))


def run(dataset):
    a = ad.read_h5ad(os.path.join(ROOT, "raw", f"{dataset}.h5ad"))
    a.var_names_make_unique()
    cluster = "clusters" if "clusters" in a.obs else "celltype"
    a = c2f.utils.get_training_data(
        a, cells_per_cluster=10**6, cluster_column=cluster, remove_clusters=[],
        min_shared_counts=20, n_var=3000,
    )
    c2f.Cell2fate_DynamicalModel.setup_anndata(a, spliced_label="spliced", unspliced_label="unspliced")
    mod = c2f.Cell2fate_DynamicalModel(a, n_modules=c2f.utils.get_max_modules(a))
    mod.train()
    a = mod.export_posterior(a)
    mod.compute_and_plot_total_velocity(a, delete=False)
    scv.tl.velocity_graph(a, vkey="Velocity")
    scv.tl.velocity_embedding(a, vkey="Velocity", basis="umap")
    v = a.obsm["Velocity_umap"]
    out = pd.DataFrame(
        {"barcode": a.obs_names, "time": a.obs["Time (hours)"].values, "vx": v[:, 0], "vy": v[:, 1]}
    )
    path = os.path.join(ROOT, "runs", f"cell2fate_{dataset}.cells.parquet")
    out.to_parquet(path, index=False)
    print(f"wrote {path}")


if __name__ == "__main__":
    run(sys.argv[1])
