#!/bin/bash
# Multi-rank Q3 run on Perlmutter CPU nodes, from the -mpi container.
#
#   DECK=nc_kpt_8x8x8.yml sbatch scripts/hpc/nersc_perlmutter/job_mpi.sbatch
#
# Edit --account below before first use.
#
# CHOOSING --ntasks: Q3 spreads ranks over k-point pools, one rank per pool, and
# refuses a rank count that would leave ranks sharing a pool. So the ceiling is
# the number of irreducible k-points in your input, and a Gamma-only supercell
# is a one-rank job however many you ask for. Get the count first:
#
#     podman-hpc run --rm -v "$PWD/examples:/work" \
#         -w /work/total_energy--semiconductor--silicon \
#         ghcr.io/camsai/q3:latest q3 <deck> --check
#
# Verified on Perlmutter across 1, 2, 4, 8 and 16 ranks: the host-MPI swap under
# --mpi works against the injected Cray MPICH, and the total energy is identical
# at every rank count.
#
# `--mpi` is not the default. Omit it and `srun -n 8` silently gives you eight
# independent single-rank runs computing the same answer.

#SBATCH --account=<your-project>
#SBATCH --qos=debug
#SBATCH --constraint=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=32
#SBATCH --time=00:30:00
#SBATCH --job-name=q3-mpi
#SBATCH --output=%x-%j.out

set -euo pipefail

IMAGE="${IMAGE:-ghcr.io/camsai/q3:latest-mpi}"
BINARY="${BINARY:-q3_mpi}"
DECK="${DECK:?set DECK=<input.yml>}"
MPI_FLAG="${MPI_FLAG:---mpi}"   # --openmpi-pmi2 / --openmpi-pmix for OpenMPI images
REPOSITORY_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
INPUT_DIR="${INPUT_DIR:-$REPOSITORY_ROOT/examples/total_energy--semiconductor--silicon}"
DATA_DIR="${DATA_DIR:-$REPOSITORY_ROOT/examples/data}"

RUN_DIR="${RUN_DIR:-$SCRATCH/${SLURM_JOB_NAME}-$SLURM_JOB_ID}"
mkdir -p "$RUN_DIR/case"
cp -r "$DATA_DIR" "$RUN_DIR/"
cp -r "$INPUT_DIR"/. "$RUN_DIR/case/"

# One thread per rank by default: threads and ranks compete for the same cores.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

echo "=== job $SLURM_JOB_ID  ranks=$SLURM_NTASKS  threads/rank=$OMP_NUM_THREADS ==="
echo "image: $IMAGE   binary: $BINARY   flag: $MPI_FLAG"

echo "=== k-point count, so you can see whether --ntasks fits ==="
srun -n 1 podman-hpc run --rm -v "$RUN_DIR:/work" -w /work/case \
  "$IMAGE" q3 "$DECK" --check

echo "=== running on $SLURM_NTASKS ranks ==="
time srun -n "$SLURM_NTASKS" --cpu-bind=cores \
  podman-hpc run --rm "$MPI_FLAG" -v "$RUN_DIR:/work" -w /work/case \
  -e OMP_NUM_THREADS "$IMAGE" "$BINARY" "$DECK" --log-level info

echo "=== outputs ==="
ls -la "$RUN_DIR/case"
