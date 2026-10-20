#!/bin/bash
# Serial or threaded Q3 run on a Perlmutter CPU node, from the container.
#
#   DECK=nc.yml sbatch scripts/hpc/nersc_perlmutter/job_serial.sbatch
#   DECK=nc.yml OMP_NUM_THREADS=8 sbatch scripts/hpc/nersc_perlmutter/job_serial.sbatch
#
# Edit --account below before first use.
#
# SIZING: `--cpus-per-task=8` is deliberately modest. The serial image links the
# distribution's reference BLAS and LAPACK, which are not threaded, so only the
# FFTs gain from extra threads and small cells lose more to thread overhead than
# they gain -- the 2-atom example spent 144.6 ms in Davidson at one thread and
# 337.5 ms at sixty-four. Scan OMP_NUM_THREADS on your own cell and keep what
# wins. For a genuinely serial workload `--qos=shared` is charged fractionally
# and fits better than a whole node.

#SBATCH --account=<your-project>
#SBATCH --qos=debug
#SBATCH --constraint=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --job-name=q3-serial
#SBATCH --output=%x-%j.out

set -euo pipefail

IMAGE="${IMAGE:-ghcr.io/camsai/q3:latest}"
DECK="${DECK:?set DECK=<input.yml>}"
REPOSITORY_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
# Q3 reaches sideways out of an example directory for its material and
# pseudopotential, so the whole examples tree has to travel with the deck.
INPUT_DIR="${INPUT_DIR:-$REPOSITORY_ROOT/examples/total_energy--semiconductor--silicon}"
DATA_DIR="${DATA_DIR:-$REPOSITORY_ROOT/examples/data}"

# Run on scratch: Q3 writes its output beside the input, and the community
# filesystem is the wrong place for job I/O.
RUN_DIR="${RUN_DIR:-$SCRATCH/${SLURM_JOB_NAME}-$SLURM_JOB_ID}"
mkdir -p "$RUN_DIR/case"
cp -r "$DATA_DIR" "$RUN_DIR/"
cp -r "$INPUT_DIR"/. "$RUN_DIR/case/"

# The container inherits nothing from this environment, and the image sets
# OMP_NUM_THREADS=1, so pass it explicitly. Note that `nproc` inside the
# container reports that value rather than the hardware; `nproc --all` is real.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

echo "=== job $SLURM_JOB_ID on $(hostname) ==="
echo "image:   $IMAGE"
echo "deck:    $DECK"
echo "threads: $OMP_NUM_THREADS"
echo "rundir:  $RUN_DIR"

echo "=== validating the input before spending the allocation ==="
srun -n 1 podman-hpc run --rm -v "$RUN_DIR:/work" -w /work/case \
  -e OMP_NUM_THREADS "$IMAGE" q3 "$DECK" --check

echo "=== running ==="
time srun -n 1 podman-hpc run --rm -v "$RUN_DIR:/work" -w /work/case \
  -e OMP_NUM_THREADS "$IMAGE" q3 "$DECK" --log-level info

echo "=== outputs ==="
ls -la "$RUN_DIR/case"
