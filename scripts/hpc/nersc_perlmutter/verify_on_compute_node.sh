#!/usr/bin/env bash
# Smoke-test the Q3 container from a compute node, before writing a batch script
# around it. Run it inside an allocation:
#
#   salloc -A <your-project> -q interactive -C cpu -N 1 -t 20
#   ./scripts/hpc/nersc_perlmutter/verify_on_compute_node.sh
#
# Stage this on the community filesystem or scratch, not /tmp: /tmp is
# node-local, so `srun /tmp/foo.sh` fails with "execve(): No such file or
# directory" even though the file exists on the login node.

set -uo pipefail

IMAGE="${IMAGE:-ghcr.io/camsai/q3:latest}"
BINARY="${BINARY:-q3}"

echo "### node $(hostname), job ${SLURM_JOB_ID:-<none>}, ${SLURM_CPUS_ON_NODE:-?} cpus"
[ -n "${SLURM_JOB_ID:-}" ] || echo "### WARNING: no allocation detected; run this under salloc/sbatch"
echo

echo "### [1] is the image visible here?"
# Only the squashed (R/O true) row appears on a compute node. That is correct:
# it means the copy on the parallel filesystem was found.
podman-hpc images | awk -v r="${IMAGE%:*}" -v t="${IMAGE##*:}" 'NR==1 || ($1==r && $2==t)'
podman-hpc images | awk -v r="${IMAGE%:*}" -v t="${IMAGE##*:}" '$1==r && $2==t' | grep -q . \
  || { echo "FAIL: $IMAGE not found. Pull it on a login node first."; exit 1; }
echo

echo "### [2] does the binary run?"
srun -n 1 podman-hpc run --rm "$IMAGE" "$BINARY" --version || { echo "FAIL: could not run $BINARY"; exit 1; }
echo

echo "### [3] real cpu count, and the thread default the image ships"
# `nproc` honours OMP_NUM_THREADS, which the image sets to 1, so it reports 1
# even on a 256-core node. That is not a runtime restriction.
srun -n 1 podman-hpc run --rm "$IMAGE" sh -lc \
  'echo "    OMP_NUM_THREADS=${OMP_NUM_THREADS:-<unset>}  nproc=$(nproc)  nproc --all=$(nproc --all)"'
echo

echo "### [4] can the container write to a bind-mounted directory?"
probe_dir="${SCRATCH:-/tmp}/q3-verify-$$"
mkdir -p "$probe_dir"
srun -n 1 podman-hpc run --rm -v "$probe_dir:/work" -w /work "$IMAGE" sh -lc \
  'echo ok > probe.txt && echo "    write inside container: ok"'
if [ -f "$probe_dir/probe.txt" ]; then echo "    host sees the file: ok"; else echo "    FAIL: bind mount not written"; fi
rm -rf "$probe_dir"
echo

echo "### [5] a bad input is refused cleanly rather than crashing"
empty_dir="${SCRATCH:-/tmp}/q3-verify-empty-$$"
mkdir -p "$empty_dir"
srun -n 1 podman-hpc run --rm -v "$empty_dir:/work" -w /work "$IMAGE" q3 --check 2>&1 | head -4 | sed 's/^/    /'
rmdir "$empty_dir" 2>/dev/null
echo
echo "### all compute-node checks done"
