#!/usr/bin/env bash
#
# Validate a Q3 variant built by scripts/hpc/nersc_perlmutter/build.sh.
#
#   scripts/hpc/nersc_perlmutter/validate.sh <variant> [--quick]
#
# cpu-serial / cpu-debug / python run anywhere. cpu-mpi, gpu-cuda and
# cpu-mpi-cuda need an allocation:
#
#   salloc -N 1 -C cpu -q interactive -t 01:00:00 -A m2651 \
#       scripts/hpc/nersc_perlmutter/validate.sh cpu-mpi
#   salloc -N 1 -C gpu -G 1 -q interactive -t 00:45:00 -A m2651_g \
#       scripts/hpc/nersc_perlmutter/validate.sh gpu-cuda
#
# IMPORTANT: under `salloc <command>` the command runs on the LOGIN node and
# only srun steps land on the allocated node. Every real run below therefore
# goes through srun. Checking `hostname` at the top of such a script reports the
# login node, which is a good way to fool yourself into thinking a serial run
# was measured on a compute node.
#
# The reference energies come from
# examples/total_energy--semiconductor--silicon/reference/nc/reference.json.

set -eo pipefail

SOURCE_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd -P)
EXAMPLES=$SOURCE_DIR/examples/total_energy--semiconductor--silicon
REFERENCE_NC=-16.91341767

usage() { sed -n '2,/^$/p' "${BASH_SOURCE[0]}" | sed 's/^#//;s/^ //'; }

VARIANT=${1:-}
case "$VARIANT" in -h|--help|"") usage; exit 0 ;; esac
shift
QUICK=0
[ "${1:-}" = "--quick" ] && QUICK=1

BUILD_DIR=$SCRATCH/q3/build/$VARIANT
WORK_DIR=$SCRATCH/q3/validate/$VARIANT
[ -d "$BUILD_DIR" ] || { echo "no build at $BUILD_DIR -- run build.sh first" >&2; exit 1; }
rm -rf "$WORK_DIR"; mkdir -p "$WORK_DIR"

case "$VARIANT" in
    gpu-cuda|cpu-mpi-cuda) module load gpu PrgEnv-gnu cray-fftw cudatoolkit/13.2 ;;
    *)                     module load cpu PrgEnv-gnu cray-fftw ;;
esac

energy_of() { grep -oE 'total energy *= *-?[0-9.]+' "$1" | tail -1 | grep -oE '\-?[0-9.]+$'; }

report() { printf '%-30s %s\n' "$1" "$2"; }

echo "== validating $VARIANT (build: $BUILD_DIR)"
[ -n "$SLURM_JOB_ID" ] && echo "== allocation $SLURM_JOB_ID; srun targets the compute node"

# ------------------------------------------------------------------ ctest ----
if [ $QUICK -eq 0 ] && [ -x "$BUILD_DIR/q3_tests" ]; then
    cd "$BUILD_DIR"
    OMP_NUM_THREADS=1 ctest -j "${JOBS:-8}" > "$WORK_DIR/ctest.log" 2>&1 || true
    report "ctest" "$(grep -E 'tests passed|tests failed' "$WORK_DIR/ctest.log" | head -1)"
fi

# ------------------------------------------------------- the python module ----
if [ "$VARIANT" = python ]; then
    cd "$SOURCE_DIR"
    if PYTHONPATH=$BUILD_DIR "$SOURCE_DIR/.venv/bin/python3" \
         -c 'import q3_core; assert q3_core.run_scf' 2>"$WORK_DIR/import.log"; then
        report "import q3_core" "OK"
    else
        report "import q3_core" "FAILED (see $WORK_DIR/import.log)"
    fi
    exit 0
fi

# ------------------------------------------------------------ serial check ---
# A tree builds one engine binary, q3, and a parallel tree links its variant
# name to it, so q3 is the serial build only when no such link is there.
if [ -x "$BUILD_DIR/q3" ] && ! ls "$BUILD_DIR"/q3_mpi* "$BUILD_DIR"/q3_gpu* > /dev/null 2>&1; then
    mkdir -p "$WORK_DIR/serial"; cd "$WORK_DIR/serial"
    "$BUILD_DIR/q3" "$EXAMPLES/nc.yml" > run.log 2>&1 || true
    report "q3 nc.yml" "$(energy_of run.log)   (reference $REFERENCE_NC)"
fi

# --------------------------------------------------------------- MPI check ---
if [ -x "$BUILD_DIR/q3_mpi" ] && [ -n "$SLURM_JOB_ID" ]; then
    srun -n 2 "$BUILD_DIR/q3_tests_mpi" > "$WORK_DIR/tests_mpi.log" 2>&1 || true
    report "q3_tests_mpi (2 ranks)" \
        "$(grep -oE '\[  PASSED  \] [0-9]+ tests' "$WORK_DIR/tests_mpi.log" | tail -1)"

    # nc.yml reduces to 2 irreducible k-points, so --pools cannot exceed 2.
    # --pools must also divide the rank count.
    for spec in "2 2" "4 2"; do
        set -- $spec
        n=$1; p=$2
        d=$WORK_DIR/mpi_n${n}_p${p}; mkdir -p "$d"; cd "$d"
        OMP_NUM_THREADS=1 srun -n "$n" "$BUILD_DIR/q3_mpi" --pools "$p" \
            "$EXAMPLES/nc.yml" > run.log 2>&1 || true
        note=""
        grep -q 'Falling back to replicated' run.log && note="  [replicated-G fallback]"
        report "q3_mpi -n $n --pools $p" "$(energy_of run.log)$note"
    done
    # Band groups top out at 4 ranks per pool for this system: the wave FFT grid
    # is 24^3 but holds only ~1564 plane waves per k-point, so a finer slab
    # split leaves some ranks with none and the code falls back to replicated
    # G-vectors, doing the whole solve redundantly on every rank in the pool.
elif [ -x "$BUILD_DIR/q3_mpi" ]; then
    report "q3_mpi" "SKIPPED -- needs an allocation (see --help)"
fi

# --------------------------------------------------------------- GPU check ---
if [ -x "$BUILD_DIR/q3_gpu_cuda" ] && [ -n "$SLURM_JOB_ID" ]; then
    srun -n 1 bash -c 'nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader' \
        > "$WORK_DIR/gpu.txt" 2>&1 || true
    report "GPU" "$(head -1 "$WORK_DIR/gpu.txt")"
    for precision in double mixed; do
        d=$WORK_DIR/gpu_$precision; mkdir -p "$d"; cd "$d"
        OMP_NUM_THREADS=8 srun -n 1 --gpus-per-task=1 "$BUILD_DIR/q3_gpu_cuda" \
            "$EXAMPLES/nc.yml" --precision "$precision" > run.log 2>&1 || true
        report "q3_gpu_cuda --precision $precision" "$(energy_of run.log)"
    done
    # double reproduces the CPU result exactly; mixed (FP32 FFT via cuFFT)
    # differs around the 6th decimal, which is expected rather than a fault.
elif [ -x "$BUILD_DIR/q3_gpu_cuda" ]; then
    report "q3_gpu_cuda" "SKIPPED -- needs a GPU allocation (see --help)"
fi

echo "== logs under $WORK_DIR"
