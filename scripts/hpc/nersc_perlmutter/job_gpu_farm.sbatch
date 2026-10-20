#!/bin/bash
# Fill a four-GPU Perlmutter node by running four INDEPENDENT Q3 calculations,
# one per device. q3_gpu_cuda is single-GPU, so this is the only way to use a
# whole GPU node.
#
#   DECKS="a.yml b.yml c.yml d.yml" sbatch scripts/hpc/nersc_perlmutter/job_gpu_farm.sbatch
#
# Edit --account below before first use.
#
# This buys THROUGHPUT, not speed: each calculation still takes as long as it
# would alone. Measured with four copies of the 54-atom silicon supercell on
# four A100s: 160.6 s in total against 132.5 s for one by itself, so 3.3x the
# work in 1.2x the time, every energy matching the single-GPU result. The
# missing 0.7x is CPU and PCIe contention.
#
# ###################### TWO RULES THAT ARE NOT GUESSABLE ######################
#
# 1. Do NOT add --gpus-per-task to the srun steps below. Each backgrounded
#    `srun --overlap` is a separate Slurm step, and each independently takes the
#    first GPU, so every task receives SLURM_STEP_GPUS=0 -- the same physical
#    device. One GPU is then oversubscribed while the rest idle; the observed
#    symptom was a SIGBUS, exit code 135.
#
# 2. podman-hpc's --gpu IGNORES the host's CUDA_VISIBLE_DEVICES and passes every
#    /dev/nvidia* into the container, where the variable arrives unset. Pinning
#    has to happen INSIDE the container, which is what separates the runs onto
#    distinct devices.
#
# The allocation also needs --ntasks at least equal to the number of concurrent
# runs, or the extra steps stall on "Requested nodes are busy".

#SBATCH --account=<your-project>_g
#SBATCH --qos=regular
#SBATCH --constraint=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --time=02:00:00
#SBATCH --job-name=q3-gpu-farm
#SBATCH --output=%x-%j.out

set -euo pipefail

IMAGE="${IMAGE:-ghcr.io/camsai/q3:latest-cuda}"
BINARY="${BINARY:-q3_gpu_cuda}"
DECKS="${DECKS:?set DECKS=\"input1.yml input2.yml ...\" (one per GPU)}"
PRECISION="${PRECISION:-double}"
GPU_COUNT="${SLURM_GPUS_ON_NODE:-4}"
REPOSITORY_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
INPUT_DIR="${INPUT_DIR:-$REPOSITORY_ROOT/examples/total_energy--semiconductor--silicon}"
DATA_DIR="${DATA_DIR:-$REPOSITORY_ROOT/examples/data}"
OUT_BASE="${OUT_BASE:-$SCRATCH/${SLURM_JOB_NAME}-$SLURM_JOB_ID}"

set -- $DECKS
if [ "$#" -gt "$GPU_COUNT" ]; then
  echo "ERROR: $# decks but only $GPU_COUNT GPUs. Submit the rest separately." >&2
  exit 1
fi

echo "=== job $SLURM_JOB_ID  gpus=$GPU_COUNT  concurrent runs=$# ==="

device=0
for deck in "$@"; do
  work="$OUT_BASE/gpu$device"
  mkdir -p "$work/case"
  cp -r "$DATA_DIR" "$work/"
  cp -r "$INPUT_DIR"/. "$work/case/"
  echo "  GPU$device <- $deck   ($work)"
  srun --exact --overlap -n 1 -c "${SLURM_CPUS_PER_TASK:-16}" \
    podman-hpc run --rm --gpu -v "$work:/work" -w /work/case \
      -e CUDA_VISIBLE_DEVICES="$device" \
      -e OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}" \
      "$IMAGE" "$BINARY" "$deck" --precision "$PRECISION" --log-level info \
    > "$work/run.out" 2>&1 &
  device=$((device + 1))
done

wait

echo
echo "=== results ==="
status=0
for directory in "$OUT_BASE"/gpu*; do
  energy=$(grep -E '^\s+total energy\s+=' "$directory/run.out" 2>/dev/null | tail -1 | tr -s ' ')
  if [ -n "$energy" ]; then
    echo "  $(basename "$directory"):$energy"
  else
    echo "  $(basename "$directory"): no energy reported -- see $directory/run.out"
    status=1
  fi
done
echo "=== outputs under $OUT_BASE ==="
exit $status
