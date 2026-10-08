#!/usr/bin/env bash
# Download the velocity benchmarks into raw/ and record their checksums.
set -euo pipefail
mkdir -p "${VELO_BENCH:-$HOME/work/velo-bench}/raw" && cd "${VELO_BENCH:-$HOME/work/velo-bench}/raw"
base=https://github.com/theislab/scvelo_notebooks/raw/master/data
get() { [ -s "$1" ] || curl -fL --retry 3 -A "Mozilla/5.0" -o "$1" "$2"; }
get pancreas.h5ad     "$base/Pancreas/endocrinogenesis_day15.h5ad"
get dentategyrus.h5ad "$base/DentateGyrus/10X43_1.h5ad"
get erythroid.h5ad    https://ndownloader.figshare.com/files/27686871
sha256sum *.h5ad > SHA256SUMS
cat SHA256SUMS
