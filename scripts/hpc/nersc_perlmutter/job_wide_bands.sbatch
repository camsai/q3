#!/bin/bash
#SBATCH --job-name=q3-wide-bands
#SBATCH --account=m2651
#SBATCH --constraint=cpu
#SBATCH --qos=regular
#SBATCH --nodes=16
#SBATCH --time=03:00:00
#SBATCH --output=%x-%j.out
#
# Width sweep for the 432-atom silicon deck, which is the only Q3 case that can
# use more than one node's worth of ranks.
#
# Why this needs a batch job rather than an interactive one: the interactive QOS
# caps at 4 nodes. And why it needs many nodes at all: every k-point deck in the
# repository saturates inside a single node, because pure pool parallelism
# cannot exceed the irreducible k-point count -- 60 for the silicon k-point
# deck, 76 for graphite, both under one node's 128 cores. nc_6x6x6 is gamma-only,
# so its only axis is --distribute-bands, and it is the deck that axis was
# written for.
#
# Sizing. Four band-distributed ranks hold 38.9 GB each and did not finish three
# SCF iterations in 83 minutes, so this starts at the WIDEST configuration
# rather than the narrowest: if 128 ranks cannot finish, nothing narrower will,
# and the job should find that out in its first twenty minutes instead of its
# last. Ranks per node are held at 8 because memory, not cores, is the binding
# constraint -- 12 ranks at the measured footprint would be 467 GB of a 503 GB
# node, with no headroom for the run to grow into.
#
# Usage:
#   sbatch scripts/hpc/nersc_perlmutter/job_wide_bands.sbatch
#
# Reports solver time and per-rank resident memory for each width. Results feed
# docs/benchmarks/ and docs/performance_nersc_perlmutter.md.

module load cpu PrgEnv-gnu cray-fftw

BINARY=${Q3_MPI_BINARY:-$SCRATCH/q3/build/cpu-mpi-scalapack/q3_mpi}
DECK=${Q3_DECK:-$SCRATCH/q3/wide/si6_capped.yml}
WORK=${Q3_WORK:-$SCRATCH/q3/wide/run-$SLURM_JOB_ID}
mkdir -p "$WORK"

echo "job $SLURM_JOB_ID on $SLURM_JOB_NUM_NODES node(s): $SLURM_JOB_NODELIST"
echo "binary $BINARY"
echo "deck   $DECK"
printf '\n%-8s %-10s %-9s %-12s %-12s %-20s %s\n' \
    RANKS PER_NODE WALL_S SOLVER_MS RSS_GB_RANK ENERGY NOTE

sweep() {
    local ranks=$1 per_node=$2 nodes=$3 limit=$4
    local d="$WORK/n$ranks"; mkdir -p "$d"; cd "$d" || return
    local t0=$SECONDS rc=0

    # Sample memory once the run is established, so the number reflects the
    # solve rather than start-up.
    ( sleep 240; srun --jobid="$SLURM_JOB_ID" --overlap -N 1 -n 1 \
        bash -c 'ps -eo rss,comm --sort=-rss | awk "/q3_mpi/ {print \$1; exit}"' \
        > rss_kb.txt 2>/dev/null ) &
    local sampler=$!

    OMP_NUM_THREADS=1 timeout "$limit" srun -N "$nodes" -n "$ranks" \
        --ntasks-per-node="$per_node" \
        "$BINARY" --pools 1 --distribute-bands "$DECK" > run.log 2>&1 || rc=$?
    wait $sampler 2>/dev/null

    local wall=$((SECONDS - t0)) solver energy rss note=""
    solver=$(grep -oE 'TOTAL: *[0-9.]+ ms' run.log 2>/dev/null | tail -1 | grep -oE '[0-9.]+' || true)
    energy=$(grep -oE 'total energy *= *-?[0-9.]+' run.log 2>/dev/null | tail -1 |
                 grep -oE '\-?[0-9.]+$' || true)
    rss=$(awk '{printf "%.1f", $1/1048576}' rss_kb.txt 2>/dev/null || echo "?")
    [ "$rc" -ne 0 ] && note="EXIT=$rc"
    [ "$rc" = 124 ] && note="TIMEOUT after ${limit}s"
    grep -q 'Falling back to replicated' run.log 2>/dev/null && note="$note replicated-G"
    printf '%-8s %-10s %-9s %-12s %-12s %-20s %s\n' \
        "$ranks" "$per_node" "$wall" "${solver:-?}" "${rss:-?}" "${energy:-(none)}" "$note"
}

# Widest first: it is the one most likely to finish, and a failure there makes
# the narrower widths pointless to attempt.
sweep 128 8 16 2400
sweep  64 8  8 2400
sweep  32 8  4 2400

echo DONE
