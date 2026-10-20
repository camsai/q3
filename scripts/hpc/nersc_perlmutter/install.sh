#!/usr/bin/env bash
#
# Install a Q3 variant built by scripts/hpc/nersc_perlmutter/build.sh.
#
#   scripts/hpc/nersc_perlmutter/install.sh <variant> [prefix-root]
#
# Each variant installs to <prefix-root>/<variant>, defaulting to
# /global/cfs/projectdirs/m2651/m3/q3/install. Per-variant subdirectories are
# deliberate: cpu-serial, cpu-mpi and cpu-mpi-cuda all produce binaries called
# q3 and q3_utility, so a shared prefix would silently overwrite one with another
# and leave no way to tell which variant a given q3 is.
#
# `cmake --install` only adds files, it never prunes, so a renamed or removed
# target leaves its old binary behind in the prefix. Check the listing this
# script prints at the end.
#
# RPATH: the installed binaries carry no RPATH beyond darshan, but every Cray
# library they need resolves through /opt/cray/pe/lib64, which is on the default
# loader path -- so they run in a shell with no modules loaded. Export
# CRAY_ADD_RPATH=yes before CONFIGURING (not here) if an install has to survive
# a CPE default change, since those symlinks follow the current default PE.

set -eo pipefail

SOURCE_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd -P)
DEFAULT_ROOT=/global/cfs/projectdirs/m2651/m3/q3/install

usage() { sed -n '2,/^$/p' "${BASH_SOURCE[0]}" | sed 's/^#//;s/^ //'; }

VARIANT=${1:-}
case "$VARIANT" in -h|--help|"") usage; exit 0 ;; esac
PREFIX_ROOT=${2:-$DEFAULT_ROOT}

BUILD_DIR=$SCRATCH/q3/build/$VARIANT
PREFIX=$PREFIX_ROOT/$VARIANT
[ -d "$BUILD_DIR" ] || { echo "no build at $BUILD_DIR -- run build.sh first" >&2; exit 1; }

case "$VARIANT" in
    gpu-cuda|cpu-mpi-cuda) module load gpu PrgEnv-gnu cray-fftw cudatoolkit/13.2 ;;
    *)                     module load cpu PrgEnv-gnu cray-fftw ;;
esac

cd "$SOURCE_DIR"
echo "== installing $VARIANT -> $PREFIX"
cmake --install "$BUILD_DIR" --prefix "$PREFIX"

echo "== contents of $PREFIX/bin"
ls -1 "$PREFIX/bin" | sed 's/^/==   /'
echo "== add to PATH: export PATH=$PREFIX/bin:\$PATH"
