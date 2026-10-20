#!/bin/bash
# Single-GPU Q3 run on a Perlmutter GPU node, from the -cuda container.
#
#   DECK=nc_3x3x3.yml sbatch scripts/hpc/nersc_perlmutter/job_gpu.sbatch
#
# Edit --account below before first use; GPU jobs need the _g project.
#
# One rank and one GPU is the whole of what q3_gpu_cuda can use. Raising the
# rank count starts that many independent copies of the same calculation rather
# than splitting one across devices -- see job_gpu_farm.sbatch for the way to
# fill a four-GPU node.
#
# Measured on an A100-SXM4-40GB: the 54-atom Gamma-only silicon supercell took
# 132.5 s against 706.8 s for the serial CPU run, a 5.3x speedup, with the total
# energy identical to every printed digit. Since a Gamma-only cell gets no MPI
# speedup at all, the GPU is the only way to parallelise one.
#
# `--precision mixed` bought no time on that case (133.0 s) and cost 7.2e-6 Ry,
# so `double` is the default here.
#
# Do not smoke-test with nvidia-smi: it is absent from the NGC base the image
# derives from. Look for the backend's own "GPU Device:" line instead. On a
# login node the image prints "WARNING: The NVIDIA Driver was not detected",
# which is expected there.

#SBATCH --account=<your-project>_g
#SBATCH --qos=debug
#SBATCH --constraint=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --time=00:30:00
#SBATCH --job-name=q3-gpu
#SBATCH --output=%x-%j.out

set -euo pipefail

IMAGE="${IMAGE:-ghcr.io/camsai/q3:latest-cuda}"
BINARY="${BINARY:-q3_gpu_cuda}"
DECK="${DECK:?set DECK=<input.yml>}"
PRECISION="${PRECISION:-double}"
REPOSITORY_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
INPUT_DIR="${INPUT_DIR:-$REPOSITORY_ROOT/examples/total_energy--semiconductor--silicon}"
DATA_DIR="${DATA_DIR:-$REPOSITORY_ROOT/examples/data}"

RUN_DIR="${RUN_DIR:-$SCRATCH/${SLURM_JOB_NAME}-$SLURM_JOB_ID}"
mkdir -p "$RUN_DIR/case"
cp -r "$DATA_DIR" "$RUN_DIR/"
cp -r "$INPUT_DIR"/. "$RUN_DIR/case/"

# The device does the work; one CPU thread is enough.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

echo "=== job $SLURM_JOB_ID  gpus=${SLURM_GPUS_ON_NODE:-?}  precision=$PRECISION ==="
echo "image: $IMAGE   binary: $BINARY"

echo "=== running ==="
time srun -n 1 podman-hpc run --rm --gpu -v "$RUN_DIR:/work" -w /work/case \
  -e OMP_NUM_THREADS "$IMAGE" "$BINARY" "$DECK" \
  --precision "$PRECISION" --log-level info

echo "=== outputs ==="
ls -la "$RUN_DIR/case"
