#!/bin/bash
#
# Usage:
#   salloc -N 4 -C gpu --gpus-per-node=4 -q interactive -t 01:30:00 -A m2651_g \
#       scripts/hpc/nersc_perlmutter/gpu_throughput.sh
#
# GPU throughput: N independent calculations on N devices.
#
# NOT one calculation across N GPUs. src/main_gpu_cuda.cpp contains no MPI call
# at all, so q3_gpu_cuda is one process on one device by construction. What is
# measurable today is how many independent calculations a set of devices
# retires at once -- the parameter-sweep case. It says nothing about whether a
# single large calculation would go faster on more devices.
#
# Two things the first attempt got wrong, kept here as the reason for the shape
# of this one:
#   * --gpus-per-task against a -G allocation gave "Invalid generic resource
#     (gres) specification" for every width below the full allocation. The
#     allocation asks for whole nodes' worth of devices and the steps select
#     from them instead.
#   * Nothing pinned a rank to a device, so four ranks on a node would all have
#     driven GPU 0 and the "4 device" number would have been four processes
#     contending for one card. CUDA_VISIBLE_DEVICES per local rank fixes that.
module load gpu PrgEnv-gnu cray-fftw cudatoolkit/13.2

GPU=$SCRATCH/q3/build/gpu-cuda/q3_gpu_cuda
CU=/global/cfs/cdirs/m2651/m3/q3/examples/total_energy--metal--copper/nc_3x3x3_bench.yml
W=$SCRATCH/q3/gpu-throughput; rm -rf $W; mkdir -p $W

echo "allocation: ${SLURM_JOB_NUM_NODES} node(s), $SLURM_JOB_NODELIST"
srun -N 1 -n 1 bash -c 'nvidia-smi --query-gpu=index,name --format=csv,noheader' 2>/dev/null

printf '\n%-9s %-7s %-9s %-14s %-16s %s\n' \
    DEVICES NODES WALL_S PER_DEVICE_S THROUGHPUT_PER_H ENERGY_AGREEMENT

run_throughput() {
    local devices=$1 nodes=$2 per_node=$3
    local d=$W/n$devices; mkdir -p "$d"; cd "$d" || return
    local t0=$SECONDS rc=0
    OMP_NUM_THREADS=8 timeout 2400 srun -N "$nodes" -n "$devices" \
        --ntasks-per-node="$per_node" --gpus-per-node=4 \
        bash -c 'export CUDA_VISIBLE_DEVICES=${SLURM_LOCALID}
                 mkdir -p rank_${SLURM_PROCID} && cd rank_${SLURM_PROCID}
                 '"$GPU $CU"' --precision double' > run.log 2>&1 || rc=$?
    local wall=$((SECONDS - t0))
    [ "$wall" -lt 1 ] && wall=1

    local distinct
    distinct=$(grep -ohE 'total energy *= *-?[0-9.]+' run.log 2>/dev/null |
                   grep -oE '\-?[0-9.]+$' | sort -u | wc -l)
    local agreement
    case "$distinct" in
        0) agreement="no energies (exit $rc)" ;;
        1) agreement="all $devices agree" ;;
        *) agreement="$distinct distinct values" ;;
    esac
    printf '%-9s %-7s %-9s %-14s %-16s %s\n' \
        "$devices" "$nodes" "$wall" \
        "$(awk -v w="$wall" 'BEGIN{printf "%.1f", w}')" \
        "$(awk -v w="$wall" -v n="$devices" 'BEGIN{printf "%.1f", n*3600/w}')" \
        "$agreement"
}

run_throughput  1 1 1
run_throughput  2 1 2
run_throughput  4 1 4
[ "${SLURM_JOB_NUM_NODES:-1}" -ge 2 ] && run_throughput  8 2 4
[ "${SLURM_JOB_NUM_NODES:-1}" -ge 4 ] && run_throughput 16 4 4
echo DONE
