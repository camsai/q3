# Running Q3 under podman-hpc or Shifter

Some HPC centres provide [`podman-hpc`](https://github.com/NERSC/podman-hpc) or
[Shifter](https://shifter.readthedocs.io/) rather than Apptainer, usually both,
with `podman-hpc` the one they prefer for new work. Both read the same OCI image
the project publishes, so nothing has to be built or converted to run there. For
the image itself, the tagging scheme, and the Apptainer route used on most other
clusters, see [README §2.1](../README.md#21-run-via-container).

Everything below uses the image name directly:

```
ghcr.io/camsai/q3:latest
```

Some centres run a pull-through mirror of the public registries and prefer you
pull through it, since it is faster and not subject to the source registry's
rate limits. Where one exists, the image name gains the mirror's host as a
prefix, as in `<mirror-host>/ghcr.io/camsai/q3:latest`, and usually needs a
one-time `podman-hpc login <mirror-host>` and an account granted on request.
Check your centre's container documentation. Every command here works either
way, with the name substituted.

While the repository is still private, pulling it needs a credential, and the
kind matters. GitHub's Container Registry accepts a **classic** personal access
token carrying the `read:packages` scope; a fine-grained token authenticates but
is then refused for package reads, which surfaces as a `Login Succeeded!`
followed immediately by `denied` on the pull, and
`permission_denied: The token provided does not match expected scopes` from the
registry API. That reads like a bad password and is not one. Pass the token on
standard input rather than as an argument, since login nodes are shared and
arguments are visible in `ps`:

```bash
printf '%s' "$TOKEN" | podman-hpc login ghcr.io --username <user> --password-stdin
```

## podman-hpc

```bash
podman-hpc pull ghcr.io/camsai/q3:latest
```

```bash
# Make the image readable from the compute nodes
podman-hpc migrate ghcr.io/camsai/q3:latest
```

`migrate` is needed because `podman-hpc` keeps two stores, and only one of them
is shared. The read-write store sits on the login node's local disk, under
`/images/<uid>_hpc/storage`; the squashed copy that `migrate` produces goes to a
parallel filesystem, on Perlmutter `$SCRATCH/storage`, and that is the one
compute nodes read. `podman-hpc pull` writes both, so `migrate` after a pull is
a no-op that costs nothing to leave in a script.

Two consequences follow. `podman-hpc images` lists each image twice, once
`R/O false` and once `R/O true`, and on a compute node only the read-only row
appears — that is the squashed copy being found, not a problem. And because the
read-write store is node-local, logging in to a different login node can make
the image look absent there even though batch jobs continue to run fine. Pull
again if a login-node command cannot see it. On a centre that purges scratch,
the squashed copy is purged with it eventually.

`podman-hpc` isolates the container from the host in two ways Apptainer does
not, and both are worth stating because neither fails loudly:

-   **Nothing is mounted unless you ask.** Bind a directory that holds the
    input _and everything the input refers to_, then set the working directory
    inside it. Mounting only the directory the input file sits in is the easy
    mistake: relative paths that climb out of it, such as the
    `../data/pseudopotentials/...` in the bundled examples, are then outside the
    container and the run stops at `Cannot open JSON file`.
-   **Host environment variables do not cross the boundary.** Exporting
    `OMP_NUM_THREADS` before the run has no effect inside the container, which
    silently leaves it at the image default of one thread. Forward it with `-e`.
-   **`nproc` inside the container reports that default, not the hardware.**
    Because the image sets `OMP_NUM_THREADS=1` and GNU `nproc` honours it,
    `podman-hpc run --rm <image> nproc` answers `1` on a 256-core node, which
    reads as a CPU restriction imposed by the runtime. It is not one: the task's
    affinity mask covers every core, `/proc/cpuinfo` lists them all, and the
    cgroup carries no CPU quota. `nproc --all` reports the real count.

How far to raise `OMP_NUM_THREADS` is worth measuring rather than assuming. The
images link OpenBLAS, built against OpenMP, so the one variable governs the
dense linear algebra and the FFTs together and the two thread pools cannot
oversubscribe each other. That much is a recent change: until 2026-09-11 they
linked the distribution's reference BLAS, which is untuned and single-threaded,
and the Cu 3x3x3 benchmark ran 10x slower in the container than in a native
Perlmutter build at one thread and 42x slower at eight. With OpenBLAS it is
within 2% of native at eight threads (camsai/q3#12).

Threads still have to earn their keep on the problem, not on the node. A small
cell splits into grids too small to cover the cost of splitting them: the 2-atom
silicon example, timed on a Perlmutter CPU node with the older reference-BLAS
image, spent 144.6 ms in Davidson at one thread and 337.5 ms at sixty-four, with
the total energy identical at every count. Those particular numbers predate the
OpenBLAS switch and have not been re-measured; the shape of the answer — measure
your own cell — does not change.

Running one of the bundled examples means mounting `examples/`, not the example
itself, because `nc.yml` reaches sideways into `examples/data/` for its material
and pseudopotential. From the top of a Q3 checkout:

```bash
podman-hpc run --rm -v "$PWD/examples:/work" \
    -w /work/total_energy--semiconductor--silicon -e OMP_NUM_THREADS=8 \
    ghcr.io/camsai/q3:latest q3 nc.yml
```

For your own calculation, mounting the project directory and working from it is
usually enough, since inputs written for it tend to keep their data alongside:

```bash
podman-hpc run --rm -v "$PWD:/work" -w /work -e OMP_NUM_THREADS=8 \
    ghcr.io/camsai/q3:latest q3 input.yml
```

In a batch job the same command goes under `srun`:

```bash
#!/bin/bash
#SBATCH --account <your-project>
#SBATCH --nodes 1
#SBATCH --time 00:30:00

srun -n 1 podman-hpc run --rm -v "$PWD:/work" -w /work -e OMP_NUM_THREADS=32 \
    ghcr.io/camsai/q3:latest q3 input.yml
```

Submit this from the directory holding `input.yml`, and check that anything the
input refers to is under it. How you ask for a CPU node — a partition, a
constraint, a quality-of-service name — differs between centres, so add those
directives from your local queue documentation.

## Shifter

Shifter pulls the same image with a `docker:` prefix, into a system-wide store
rather than a per-user one:

```bash
shifterimg pull docker:ghcr.io/camsai/q3:latest
```

The image is named in the `#SBATCH` directive rather than on the `shifter`
command line. Unlike `podman-hpc`, Shifter bind-mounts your home and scratch
directories and carries your environment into the container. That removes both
of the traps above: an exported variable reaches `q3` as written, and an input
that refers to a sibling directory resolves, so long as the whole tree sits
under a path Shifter already mounts.

```bash
#!/bin/bash
#SBATCH --image docker:ghcr.io/camsai/q3:latest
#SBATCH --account <your-project>
#SBATCH --nodes 1
#SBATCH --time 00:30:00

export OMP_NUM_THREADS=32
srun -n 1 shifter q3 nc.yml
```

Add `--volume="/path/on/host:/path/in/container"` to the directives to mount a
path Shifter does not provide by default; separate several with semicolons.

## Parallel and GPU images

The plain tag is the serial `q3`, scaled only by `OMP_NUM_THREADS` within one
node. Two further images carry the parallel executables, published under the
same repository with a suffix:

| Tag           | `q3` is the  | Also named    | Architectures |
| ------------- | ------------ | ------------- | ------------- |
| `latest`      | serial build | —             | amd64, arm64  |
| `latest-mpi`  | MPI build    | `q3_mpi`      | amd64, arm64  |
| `latest-cuda` | CUDA build   | `q3_gpu_cuda` | amd64         |

Every image carries one engine binary and runs it as `q3`, so a job script
names the same executable whichever image it pulls; the variant name is a
symlink to it. The MPI and CUDA builds run the self-consistent field only, so
the other steps of a workflow still need the plain image. Date-based tags follow the same pattern,
`YYYY.M.D-N-mpi` and `YYYY.M.D-N-cuda`.

### MPI

Both runtimes make MPI work by replacing the container's MPI with the host's at
run time, which needs an image whose MPICH was built from source rather than
installed from a distribution package. `Dockerfile.mpi` does exactly that, so
`--mpi` and `--module=mpich` apply to the `-mpi` tag.

The launcher owns the rank count, not the container. Under `--mpi` each rank is
its own container process started by `srun`, so `mpiexec` does **not** appear
inside the command:

```bash
podman-hpc pull ghcr.io/camsai/q3:latest-mpi
podman-hpc migrate ghcr.io/camsai/q3:latest-mpi
```

```bash
#!/bin/bash
#SBATCH --account <your-project>
#SBATCH --nodes 2
#SBATCH --time 01:00:00

srun -n 8 podman-hpc run --rm --mpi -v "$PWD:/work" -w /work \
    -e OMP_NUM_THREADS=8 ghcr.io/camsai/q3:latest-mpi q3 input.yml
```

Under Shifter the module does the swap and the invocation is shorter:

```bash
#!/bin/bash
#SBATCH --image docker:ghcr.io/camsai/q3:latest-mpi
#SBATCH --module=mpich
#SBATCH --account <your-project>
#SBATCH --nodes 2
#SBATCH --time 01:00:00

export OMP_NUM_THREADS=8
srun -n 8 shifter q3 input.yml
```

Ranks divide into k-point pools by default; `--pools` splits them differently,
and band groups larger than one rank need the MPI-enabled FFTW that this image
carries. See [README §4.2](../README.md#42-parallel--gpu-builds).

One capability is deliberately missing. ScaLAPACK is not in the image, because
the distribution builds link the distribution's MPI — the ABI this image exists
to avoid — and building it from source would pull the Fortran toolchain back in
for a component the build treats as optional. Without it the parallel subspace
diagonalization falls back to a redundant serial solve. That is a supported
configuration, covered by its own leg in the main CI matrix, and it costs time
rather than accuracy.

### GPU

In the `-cuda` image `q3` is the CUDA build (`q3_gpu_cuda`), compiled for the A100 and H100. It is
amd64 only: the GPUs it targets sit in x86_64 nodes, and publishing an untested
aarch64 copy under a shared tag would be worse than publishing none.

```bash
srun -n 1 -G 1 podman-hpc run --rm --gpu -v "$PWD:/work" -w /work \
    ghcr.io/camsai/q3:latest-cuda q3 input.yml
```

One rank and one GPU is not a simplification, it is the whole of what
`q3_gpu_cuda` can use. Raising the rank count does not spread one calculation
over more devices; it starts that many independent copies of it. Measured on a
four-GPU Perlmutter node, `srun -n 4 -G 4` printed four `GPU Device` banners and
four identical total energies, having solved the same ground state four times.
Nothing warns you, and the wall time even goes up.

Filling a multi-GPU node therefore means giving it several _different_
calculations, one per device. Two details make that work, and neither is
guessable:

-   **Do not add `--gpus-per-task`.** Each backgrounded `srun --overlap` is a
    separate step, and each one independently takes the first GPU, so every task
    receives `SLURM_STEP_GPUS=0` — the same physical device. One GPU is then
    oversubscribed while the rest idle; the symptom observed was a SIGBUS, exit
    code 135.
-   **`--gpu` ignores the host's `CUDA_VISIBLE_DEVICES`** and passes every
    `/dev/nvidia*` into the container, where the variable arrives unset. The
    pinning has to happen inside the container instead.

```bash
#!/bin/bash
#SBATCH --account <your-project>
#SBATCH --nodes 1
#SBATCH --ntasks 4          # one task slot per concurrent run, or steps block
#SBATCH --gpus-per-node 4
#SBATCH --time 02:00:00

for gpu in 0 1 2 3; do
    mkdir -p "run$gpu" && cp -r inputs/. "run$gpu"/
    srun --exact --overlap -n 1 -c 16 \
        podman-hpc run --rm --gpu -v "$PWD/run$gpu:/work" -w /work \
        -e CUDA_VISIBLE_DEVICES=$gpu \
        ghcr.io/camsai/q3:latest-cuda q3_gpu_cuda input.yml &
done
wait
```

That buys throughput rather than speed: four 54-atom runs finished in 160.6 s
against 132.5 s for one alone, so 3.3x the work in 1.2x the time, with every
energy matching the single-GPU result. It does not make any one calculation
faster.

The image contains the CUDA runtime and math libraries but not the driver,
which the container runtime injects from the host. That is what `--gpu` is for;
without it the run fails on a CUDA error rather than a missing library.

Because the driver comes from the host, the host decides what the image can
run. This one is built against CUDA 12.8, which needs driver 525 or newer, and
it keeps working on drivers newer than that — an application built against an
older toolkit runs on a newer driver, which is a guarantee CUDA makes. The
reverse is not true, which is why the image is not built against a newer
toolkit: that would refuse to start on every host below driver 580.

Combining GPUs with MPI is not yet possible from a published image: the two
variants are built separately, and one image carrying both is the natural next
step rather than something these tags already do.

## What is verified, and what is not

Every image is built for each architecture it claims and runs the silicon
example on it, asserting the converged total energy against the same reference
the validation suite uses rather than merely that the run converged. The MPI
image additionally runs across one and two ranks, so the ground state is checked
to be independent of the rank topology, and the CUDA image's serial binary is
exercised the same way as the others. Two ranks is the ceiling for that example
rather than a limit of the image: it reduces to two irreducible k-points, and Q3
parallelizes over k-points, so a third rank would have no pool to occupy. Higher
rank counts are covered by the main test suite, against inputs with the k-points
to fill them.

Two gaps are worth stating plainly. No test runs `podman-hpc` or Shifter
themselves, because neither is available on the machines that build these
images; what is verified is that the artefact is a conformant OCI image that
behaves correctly under an OCI runtime, unprivileged included. And the host-MPI
swap that `--mpi` and `--module=mpich` perform cannot be exercised without a
Cray system, so the MPI image is built to the documented requirement rather
than measured against it. The GPU backend's numerical correctness is gated
separately, by the CPU-versus-GPU energy parity test that runs on real
hardware.

### Measured by hand on Perlmutter, 2026-09-10

The second of those gaps has now been closed once, by hand rather than by CI, on
`ghcr.io/camsai/q3:2026.9.10-0` and its `-mpi` and `-cuda` companions. This is a
snapshot, not a regression test: nothing reruns it.

-   **`podman-hpc` itself works**, including `pull`, `migrate`, and the squashed
    store being what compute nodes read.
-   **The host-MPI swap under `--mpi` works.** `q3_mpi` ran across 1, 2, 4, 8
    and 16 ranks against the injected Cray MPICH, reporting the expected
    `N rank(s) -> N k-point pool(s) x 1 rank(s) per pool` topology and an
    identical total energy at every rank count on a 60-k-point input.
-   **The k-point ceiling is enforced, and it is a clean refusal.** A Gamma-only
    54-atom input asked for more than one rank exits with a message naming the
    irreducible k-point count and suggesting `OMP_NUM_THREADS` or `--pools`
    instead — no silent misconfiguration.
-   **`--gpu` works** on an A100-SXM4-40GB, and `q3_gpu_cuda` reproduced the
    serial CPU total energy for the 54-atom cell to every printed digit, at 5.3x
    the speed (132.5 s against 706.8 s). `--precision mixed` bought no time at
    all there and cost 7.2e-6 Ry.
-   **Against Quantum ESPRESSO 7.5 with libxc 7.0.0**, from the NERSC module
    matching the versions the reference data records, the 2-atom silicon example
    agreed to 2.6e-7 Ry.
-   **Ranks sharing a pool aborted**, and that one was a real defect rather than
    a property of the swap: `--pools 1` with more than one rank died in the
    first complex reduction with a null datatype. It reproduced with the same
    image under a plain OCI runtime off a Cray system entirely, and is fixed as
    of 2026-09-11 -- Q3 was passing a Fortran datatype that this image's
    `--disable-fortran` MPICH never initializes. See KNOWN-ISSUES.md; the
    container smoke test now runs that layout.

Two operational notes from the same exercise. Each `podman-hpc run` costs
about 1.3 s of startup, and an `srun` step around it adds roughly 2 s more.
Both are invisible on a real calculation and dominate a small one: a rank scan
on a 2-atom input flattened at about five seconds per point, which was the sum
of the two rather than the calculation. Of the container's 1.3 s, ~0.3 s is the
`podman-hpc` Python wrapper (`podman-hpc --help` alone costs that) and the rest
is podman creating namespaces, mounting the overlay and running the site hooks.
Bypassing the parallel-filesystem squash store with `--additional-stores ''`
made no difference, so that store is not the cost. And `nvidia-smi` is absent
from the `-cuda` image, as it is from the NGC base it derives from, so it is not
a usable smoke test; the backend prints its own `GPU Device:` line instead.
