#!/usr/bin/env bash
#
# Performance measurement for Q3 on Perlmutter.
#
#   salloc -N 1 -C cpu -q interactive -t 02:00:00 -A m2651 \
#       scripts/hpc/nersc_perlmutter/benchmark.sh cpu
#   salloc -N 1 -C gpu -G 1 -q interactive -t 01:00:00 -A m2651_g \
#       scripts/hpc/nersc_perlmutter/benchmark.sh gpu
#
# Two systems, because they exercise different parallelism:
#
#   Cu 3x3x3 supercell, gamma-only, 280 bands (nc_3x3x3_bench.yml)
#       One k-point, so k-point pools are unusable and MPI parallelism is band
#       groups only. This is the large system, and the one PERFORMANCE.md
#       benchmarks; it is also the only one big enough to occupy a GPU.
#   Si 8x8x8, 60 irreducible k-points (nc_kpt_8x8x8.yml)
#       Small per k-point, so it measures pool scaling and nothing else.
#
# Times are the solver's own reported total, not wall clock: srun startup is
# 2-9 s on this machine and swamps the shorter runs. Wall is reported too, so
# the gap is visible. Energies are printed for every run because a speedup that
# changes the answer is not a speedup.
#
# Every real run goes through srun: under `salloc <command>` the script body
# itself executes on the login node.
#
# The threading rows deliberately pass no affinity flags. Binding measured 45%
# SLOWER than leaving the scheduler alone on this workload, concentrated in the
# FFT stage -- see docs/performance_nersc_perlmutter.md. Do not "fix" that by adding
# --cpu-bind=cores without re-measuring.

set -eo pipefail

SOURCE_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd -P)
MODE=${1:-cpu}

CU=$SOURCE_DIR/examples/total_energy--metal--copper/nc_3x3x3_bench.yml
SI=$SOURCE_DIR/examples/total_energy--semiconductor--silicon/nc_kpt_8x8x8.yml
# The Cu benchmark input sets max_iterations: 3 and deliberately does NOT
# converge -- it exists to time a fixed amount of work. So the expected value
# here is its own 3-iteration total energy, NOT the converged
# -10438.57534212 Ry in reference/nc/reference.json. Comparing against the
# converged figure flags every run as a mismatch, which says nothing. What this
# check is for is agreement ACROSS configurations: serial, threaded, MPI and
# GPU-double all have to land on the same number.
CU_REFERENCE=-10438.720620917
SI_REFERENCE=-16.925260460

WORK=$SCRATCH/q3/benchmark/$MODE
rm -rf "$WORK"; mkdir -p "$WORK"

case "$MODE" in
    gpu) module load gpu PrgEnv-gnu cray-fftw cudatoolkit/13.2 ;;
    cpu) module load cpu PrgEnv-gnu cray-fftw ;;
    *)   echo "usage: benchmark.sh [cpu|gpu]" >&2; exit 1 ;;
esac

SERIAL=$SCRATCH/q3/build/cpu-serial
MPI_SL=$SCRATCH/q3/build/cpu-mpi-scalapack
MPI_NS=$SCRATCH/q3/build/cpu-mpi
GPU=$SCRATCH/q3/build/gpu-cuda

header() {
    printf '\n%-40s %-8s %-11s %-11s %-18s %s\n' \
        CASE RANKS SOLVER_MS WALL_S ENERGY_RY NOTE
}

# run <label> <command...> -- records solver ms, wall s, energy.
run() {
    local label=$1 ranks=$2 reference=$3; shift 3
    local d="$WORK/$label"; mkdir -p "$d"; cd "$d" || return
    local t0=$SECONDS
    # `|| rc=$?` rather than a bare call: under `set -e` a failing run would
    # abort the whole sweep before the status could be recorded, so one crash
    # silently truncated every measurement after it. A failed run is a result.
    local rc=0
    "$@" > run.log 2>&1 || rc=$?
    local wall=$((SECONDS - t0)) ms e note
    # Every extraction below has to tolerate absence. A crashed run has no
    # TOTAL line and no energy, and under `set -e` a failing command
    # substitution in an assignment aborts the script -- which truncated two
    # earlier sweeps at the first crash and silently discarded every
    # measurement after it.
    ms=$(grep -oE 'TOTAL: *[0-9.]+ ms' run.log 2>/dev/null | tail -1 |
             grep -oE '[0-9.]+' || true)
    e=$(grep -oE 'total energy *= *-?[0-9.]+' run.log 2>/dev/null | tail -1 |
            grep -oE '\-?[0-9.]+$' || true)
    note=""
    if [ "$rc" -ne 0 ]; then note="EXIT=$rc"; fi
    if grep -q 'Falling back to replicated' run.log 2>/dev/null; then
        note="$note replicated-G"
    fi
    if grep -qi 'treating as a failure' run.log 2>/dev/null; then
        note="$note SHORT-SOLVE"
    fi
    if grep -qE 'Segmentation fault|MPICH ERROR' run.log 2>/dev/null; then
        note="$note MPI-ABORT"
    fi
    if [ -n "$e" ] && [ -n "$reference" ]; then
        python3 - "$e" "$reference" <<'PY' >/dev/null 2>&1 || note="$note ENERGY-MISMATCH"
import sys
got, ref = float(sys.argv[1]), float(sys.argv[2])
sys.exit(0 if abs(got - ref) < 1e-4 else 1)
PY
    fi
    printf '%-40s %-8s %-11s %-11s %-18s %s\n' \
        "$label" "$ranks" "${ms:-?}" "$wall" "${e:-(none)}" "$note"
}

echo "benchmark mode=$MODE  node(s)=$SLURM_JOB_NUM_NODES  job=$SLURM_JOB_ID"
srun -n 1 bash -c 'echo "compute node: $(hostname), $(nproc) cores"'

if [ "$MODE" = cpu ]; then
    echo
    echo "===== Cu 3x3x3 gamma-only, 280 bands: OpenMP threading (1 rank) ====="
    header
    for threads in 1 8 32; do
        OMP_NUM_THREADS=$threads run "cu_serial_omp$threads" "1" "$CU_REFERENCE" \
            srun -n 1 -c "$((threads * 2))" "$SERIAL/q3" "$CU"
    done

    echo
    echo "===== Cu 3x3x3: MPI band groups, ScaLAPACK ON (pools=1, OMP=1) ====="
    header
    for ranks in 2 4 8 16; do
        OMP_NUM_THREADS=1 run "cu_bandgroup_n${ranks}_scalapack" "$ranks" "$CU_REFERENCE" \
            srun -n "$ranks" "$MPI_SL/q3_mpi" --pools 1 "$CU"
    done

    echo
    echo "===== Cu 3x3x3: the same widths with ScaLAPACK OFF, for contrast ====="
    header
    for ranks in 4 16; do
        OMP_NUM_THREADS=1 run "cu_bandgroup_n${ranks}_noscalapack" "$ranks" "$CU_REFERENCE" \
            srun -n "$ranks" "$MPI_NS/q3_mpi" --pools 1 "$CU"
    done

    echo
    echo "===== Si 8x8x8, 60 k-points: pools vs band groups ====="
    header
    OMP_NUM_THREADS=1 run "si_serial" "1" "$SI_REFERENCE" srun -n 1 "$SERIAL/q3" "$SI"
    for ranks in 8 32; do
        OMP_NUM_THREADS=1 run "si_pools_n${ranks}" "$ranks" "$SI_REFERENCE" \
            srun -n "$ranks" "$MPI_SL/q3_mpi" --pools "$ranks" "$SI"
    done
    OMP_NUM_THREADS=1 run "si_n8_p2_scalapack" "8" "$SI_REFERENCE" \
        srun -n 8 "$MPI_SL/q3_mpi" --pools 2 "$SI"
    OMP_NUM_THREADS=1 run "si_n8_p2_noscalapack" "8" "$SI_REFERENCE" \
        srun -n 8 "$MPI_NS/q3_mpi" --pools 2 "$SI"
else
    echo
    echo "===== Cu 3x3x3 gamma-only, 280 bands: GPU vs CPU on the same node ====="
    srun -n 1 bash -c 'nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv,noheader'
    header
    for precision in double mixed; do
        OMP_NUM_THREADS=8 run "cu_gpu_$precision" "1" "$CU_REFERENCE" \
            srun -n 1 --gpus-per-task=1 "$GPU/q3_gpu_cuda" "$CU" --precision "$precision"
    done
    for threads in 1 32; do
        OMP_NUM_THREADS=$threads run "cu_cpu_omp$threads" "1" "$CU_REFERENCE" \
            srun -n 1 -c "$((threads * 2))" "$SERIAL/q3" "$CU"
    done
fi

echo
echo "===== DONE (logs under $WORK) ====="
