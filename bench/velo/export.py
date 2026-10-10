"""Export an scVelo h5ad into MatrixMarket sets for data-beans `from-mtx`.

For dataset NAME, writes into mtx/NAME_{set}/ (genes x cells, raw counts):
  tracks   rows {gene}/count/spliced then {gene}/count/unspliced (the ODE input)
  total    rows {gene}, spliced + unspliced (the base fit's input)
  spliced  rows {gene}, spliced only (the base-fit variant)
and NAME.cells.tsv.gz: barcode, clusters (celltype where there are none), UMAP
x/y, and latent time and embryonic stage if present.

Genes with no reads on either track are dropped; names are made unique.

usage: uv run --with anndata,scipy,pandas,numpy python export.py NAME [NAME ...]
"""

import gzip
import os
import sys

import anndata as ad
import numpy as np
import pandas as pd
import scipy.io
import scipy.sparse as sp

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ.get("VELO_BENCH", os.path.expanduser("~/work/velo-bench"))


def counts(a, layer):
    x = sp.csr_matrix(a.layers[layer])
    assert np.all(np.mod(x.data, 1) == 0), f"{layer} is not integer counts"
    return x.astype(np.int64)


def write_set(out, matrix, rows, barcodes):
    os.makedirs(out, exist_ok=True)
    with gzip.open(os.path.join(out, "matrix.mtx.gz"), "wb") as f:
        scipy.io.mmwrite(f, sp.coo_matrix(matrix), field="integer")
    with gzip.open(os.path.join(out, "features.tsv.gz"), "wt") as f:
        f.write("\n".join(rows) + "\n")
    with gzip.open(os.path.join(out, "barcodes.tsv.gz"), "wt") as f:
        f.write("\n".join(barcodes) + "\n")


def export(name):
    a = ad.read_h5ad(os.path.join(ROOT, "raw", f"{name}.h5ad"))
    a.var_names_make_unique()
    s, u = counts(a, "spliced"), counts(a, "unspliced")
    keep = np.asarray((s + u).sum(axis=0)).ravel() > 0
    s, u = s[:, keep], u[:, keep]
    genes = [g.replace("/", "_") for g in a.var_names[keep]]
    barcodes = list(a.obs_names)
    # genes x cells
    st, ut = s.T.tocsr(), u.T.tocsr()
    mtx = os.path.join(ROOT, "mtx")
    write_set(
        os.path.join(mtx, f"{name}_tracks"),
        sp.vstack([st, ut]),
        [f"{g}/count/spliced" for g in genes] + [f"{g}/count/unspliced" for g in genes],
        barcodes,
    )
    write_set(os.path.join(mtx, f"{name}_total"), st + ut, genes, barcodes)
    write_set(os.path.join(mtx, f"{name}_spliced"), st, genes, barcodes)

    cluster = "clusters" if "clusters" in a.obs else "celltype"
    cells = pd.DataFrame({"barcode": barcodes, "clusters": a.obs[cluster].astype(str).values})
    umap = a.obsm["X_umap"]
    cells["umap_x"], cells["umap_y"] = umap[:, 0], umap[:, 1]
    for col in ("latent_time", "velocity_pseudotime", "stage"):
        if col in a.obs:
            cells[col] = a.obs[col].values
    cells.to_csv(os.path.join(ROOT, f"{name}.cells.tsv.gz"), sep="\t", index=False)
    print(
        f"{name}: {len(barcodes)} cells, {len(genes)} genes; "
        f"spliced {int(s.sum())} + unspliced {int(u.sum())} reads"
    )


if __name__ == "__main__":
    for n in sys.argv[1:]:
        export(n)
