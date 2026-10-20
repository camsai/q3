# Q3 on NERSC Perlmutter

Two ways to run Q3 here, and this directory holds the scripts for both.

**Build it natively** — the faster option, and what the performance numbers are
measured on. `build.sh`, `install.sh`, `validate.sh`, `benchmark.sh`; see
[docs/build_nersc_perlmutter.md](../../../docs/build_nersc_perlmutter.md) and
[docs/performance_nersc_perlmutter.md](../../../docs/performance_nersc_perlmutter.md).

**Run the published container** — nothing to build, and the same image runs on
any centre with `podman-hpc` or Shifter. `pull_images.sh`, the `job_*.sbatch`
files, and `verify_on_compute_node.sh`; see
[docs/podman_hpc_and_shifter.md](../../../docs/podman_hpc_and_shifter.md) for what
`migrate`, `--mpi` and `--gpu` are doing.

Expect the container to be slower than a native build: the published images link
the distribution's reference BLAS and LAPACK, where a Perlmutter build gets Cray
LibSci. Use the container for portability and convenience, and `build.sh` when
the throughput matters.

## Native build

| script                         | what it does                                     |
| ------------------------------ | ------------------------------------------------ |
| [`build.sh`](build.sh)         | Build a variant (`--list` shows them)            |
| [`install.sh`](install.sh)     | Install a built variant under a prefix           |
| [`validate.sh`](validate.sh)   | Validate a built variant, `--quick` for a subset |
| [`benchmark.sh`](benchmark.sh) | Performance measurement, `cpu` or `gpu`          |

## Container

| script                                                   | what it does                                                                                            |
| -------------------------------------------------------- | ------------------------------------------------------------------------------------------------------- |
| [`pull_images.sh`](pull_images.sh)                       | Pull the three images on a login node and check the squashed copy exists, so compute nodes can see them |
| [`verify_on_compute_node.sh`](verify_on_compute_node.sh) | Smoke-test from inside an allocation before writing a batch script                                      |
| [`job_serial.sbatch`](job_serial.sbatch)                 | Serial or threaded `q3` on a CPU node                                                                   |
| [`job_mpi.sbatch`](job_mpi.sbatch)                       | Multi-rank `q3_mpi` on CPU nodes, via `--mpi`                                                           |
| [`job_gpu.sbatch`](job_gpu.sbatch)                       | Single-GPU `q3_gpu_cuda`, via `--gpu`                                                                   |
| [`job_gpu_farm.sbatch`](job_gpu_farm.sbatch)             | Four independent calculations on a four-GPU node                                                        |

Each `.sbatch` file carries `--account=<your-project>` as a placeholder; edit it
before first use, and use the `_g` form of the project for GPU jobs.

```bash
GHCR_USER=<github-user> ./scripts/hpc/nersc_perlmutter/pull_images.sh

salloc -A <your-project> -q interactive -C cpu -N 1 -t 20
./scripts/hpc/nersc_perlmutter/verify_on_compute_node.sh
exit

DECK=nc.yml sbatch scripts/hpc/nersc_perlmutter/job_serial.sbatch
```

The job scripts copy `examples/data` and one example directory onto `$SCRATCH`
and run there, because Q3 writes its output beside its input and because an
example deck reaches sideways into `examples/data` for its material and
pseudopotential. Point `INPUT_DIR` at your own directory to run your own input.

## Four things that cost real debugging time

-   **Name the parallel binary.** Every image also carries the serial `q3`, first
    on `PATH`. Running `q3` out of the `-cuda` image gets you a silent serial CPU
    run; you want `q3_gpu_cuda`.
-   **`nproc` lies inside the container.** The image sets `OMP_NUM_THREADS=1` and
    GNU `nproc` honours it, so it answers `1` on a 256-core node. Nothing is
    restricting the CPUs. Use `nproc --all`.
-   **More threads is not automatically faster.** In the container only the FFTs
    thread, since the reference BLAS does not, and small cells lose more to thread
    overhead than they gain. `benchmark.sh` and
    [docs/performance_nersc_perlmutter.md](../../../docs/performance_nersc_perlmutter.md)
    cover the native build's threading behaviour, which is its own story.
-   **Q3's MPI ceiling is the irreducible k-point count.** One rank per k-point
    pool, and it refuses more; a Gamma-only supercell is a one-rank job. Run
    `q3 <deck> --check` to see the count before choosing `--ntasks`.

## A note on timing

`benchmark.sh` reports the solver's own total rather than wall clock, because
`srun` startup on this machine is a few seconds and swamps short runs. That
applies at least as much to the container: a `podman-hpc run` adds about 1.3 s
of its own on top of the `srun` step, so anything finishing inside a minute is
measuring the launch rather than the calculation.
