# Building Q3 on NERSC Perlmutter

A plan for producing the several builds of Q3 that Perlmutter can support — serial,
MPI, CUDA — together with the site-specific facts each one depends on.

[docs/build_reference.md](build_reference.md) is the general reference: every option,
what it defaults to, and a start-to-finish Perlmutter recipe. This page is narrower
and more specific: what is actually installed on this machine, which variants are
therefore reachable, in what order to attempt them, and the handful of places where
the general recipe is not quite enough.

Everything below was checked on a login node on **2026-09-09** and **2026-09-10**, under
**CPE 26.03**. Module versions move; re-check before trusting a version number here.

## Just build it

The recipes on this page are executable, so you do not have to read it to use it:

```bash
scripts/hpc/nersc_perlmutter/build.sh --list                 # cpu-serial cpu-mpi cpu-mpi-scalapack ...
scripts/hpc/nersc_perlmutter/build.sh cpu-serial             # module load + configure + build
scripts/hpc/nersc_perlmutter/validate.sh cpu-serial          # ctest + an example, energy vs reference
scripts/hpc/nersc_perlmutter/install.sh cpu-serial           # -> <prefix>/cpu-serial/bin
scripts/hpc/nersc_perlmutter/benchmark.sh cpu                # timings (inside an allocation)
```

For how these builds actually perform on this machine -- and which parallelism is worth
using -- see [performance_nersc_perlmutter.md](performance_nersc_perlmutter.md). The headline is that
k-point pools scale and band groups do not.

Those three scripts carry the module sets, the CMake flags and every workaround below.
Configure on a login node (it downloads dependencies); `cpu-mpi` and `gpu-cuda`
validation needs an allocation, and `validate.sh --help` gives the `salloc` lines. The
rest of this page is the reasoning behind what the scripts do.

## The environment, as verified

|                         |                                                                 |
| ----------------------- | --------------------------------------------------------------- |
| System                  | Perlmutter (`NERSC_HOST=perlmutter`), login node                |
| Programming environment | `cpe/26.03`, `PrgEnv-gnu/8.7.0`                                 |
| Compiler                | `gcc-native/14` → GCC 14.3.0                                    |
| MPI                     | `cray-mpich/9.1.0`                                              |
| BLAS/LAPACK             | `cray-libsci/26.03.0`, supplied by the `cc`/`CC` wrappers       |
| FFTW                    | `cray-fftw/3.3.10.11`                                           |
| CUDA                    | `cudatoolkit/13.2` (NVHPC 26.5 layout), `craype-accel-nvidia80` |
| GPU                     | NVIDIA A100 → `CMAKE_CUDA_ARCHITECTURES=80`                     |
| CMake                   | `/usr/bin/cmake` 3.28.3; `cmake/3.30.2` module also available   |
| Slurm accounts          | `m2651` (CPU), `m2651_g` (GPU)                                  |

Each of Q3's build inputs, against what the machine provides:

| Requirement                     | Status                                                                                                                                                                                     |
| ------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| GCC 9 or newer (`<filesystem>`) | `gcc-native/14` is 14.3.0 — satisfied                                                                                                                                                      |
| FFTW3, threads, MPI             | `cray-fftw` ships `libfftw3`, `libfftw3_threads` and `libfftw3_mpi` as both `.so` and `.a`, and exports `FFTW_DIR`, `FFTW_INC` and `FFTW_ROOT`, which Q3's discovery reads. No `-D` needed |
| LAPACK                          | `find_package(LAPACK)` resolves through the Cray wrapper to libsci                                                                                                                         |
| Network for `FetchContent`      | Reachable from **login nodes only**. nlohmann_json, yaml-cpp, spdlog, spglib, LibXC 7.0.0, GoogleTest and (for the Python module) nanobind are all fetched at configure time               |
| Schema code generation          | The in-tree `.venv` imports `jinja2`, `jsonref` and `yaml`, and `schemas/esse/dist` is present, so generation is enabled rather than falling back to the committed headers                 |

Do not use `libfftw3_omp`: it is the alternative threading backend, and Q3 expects
`libfftw3_threads`.

## What can be built here

### Tier 1 — the builds worth having

| Variant        | Configure flags beyond the common set                   | Targets produced                         |
| -------------- | ------------------------------------------------------- | ---------------------------------------- |
| `cpu-serial`   | none                                                    | `q3`, `q3_utility`, `q3_tests`           |
| `cpu-mpi`      | `-DQ3_WITH_MPI=ON` (+ `SCALAPACK_LIBRARIES`, see below) | adds `q3_mpi`, `q3_tests_mpi`            |
| `gpu-cuda`     | `-DQ3_WITH_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=80`       | adds `q3_gpu_cuda`                       |
| `cpu-mpi-cuda` | both of the above                                       | all of the above, plus `q3_mpi_gpu_cuda` |

`cpu-mpi-cuda` is no longer only a packaging convenience. `q3_gpu_cuda` is still strictly
single-process — `src/main_gpu_cuda.cpp` never initializes MPI — but this variant also
builds **`q3_mpi_gpu_cuda`**, which gives each MPI rank its own CUDA device and so runs a
single calculation across several GPUs, including across nodes. Measured at 7.88x on eight
devices spanning two nodes; see
[performance_nersc_perlmutter.md](performance_nersc_perlmutter.md).

Launching it as several tasks is refused rather than allowed to look like it worked.
`srun -n 2 ... q3_gpu_cuda` used to run to completion and print the right energy twice,
because each task had set up the whole problem on its own A100 and solved all of it —
2% slower than one task, for twice the allocation. Since 2026-09-11 the binary reads the
launcher's task count from the environment and stops with a message naming the variable
that reported it. `Q3_ALLOW_MULTI_TASK_GPU=1` proceeds anyway, for the one case where
several tasks of the same binary are deliberate: packing independent calculations into
one job step.

### Tier 2 — situational

| Variant        | Flags                                                                                      | For                                                                        |
| -------------- | ------------------------------------------------------------------------------------------ | -------------------------------------------------------------------------- |
| `cpu-debug`    | `-DCMAKE_BUILD_TYPE=Debug`, optionally `-DQ3_COVERAGE=ON`                                  | debugging and coverage; `Q3_COVERAGE` also switches the default build type |
| `python`       | `-DQ3_ENABLE_PYTHON=ON`                                                                    | the `q3_core` nanobind extension                                           |
| compiler sweep | swap `PrgEnv-gnu` for `PrgEnv-cray` (CCE 21), `PrgEnv-llvm/21.1.4` or `PrgEnv-intel/8.7.0` | portability checking, once Tier 1 is green                                 |

### Tier 3 — not reachable on this machine

-   `Q3_WITH_HIP` — Perlmutter has no AMD GPUs.
-   `Q3_WITH_SYCL` — needs `icpx` and `MKL::MKL_SYCL`; the `intel-oneapi` modules here
    carry no NVIDIA SYCL backend.
-   `Q3_WITH_GPU` — the Apple Metal backend, macOS only.
-   `PrgEnv-nvidia` (`nvc++`) — expect trouble with the template-heavy C++17 and with
    the vendored `FetchContent` subprojects. Attempt last, if at all.

## Version axes

To build one variant several ways, these are the axes the machine offers:

| Axis            | Versions                                                            |
| --------------- | ------------------------------------------------------------------- |
| GNU compiler    | `gcc-native/{12, 12.3, 13, 13.2, 14}`                               |
| Other compilers | `cce/{20, 21}`, `intel/2025.3`, `PrgEnv-llvm/21.1.4`, `nvidia/26.5` |
| MPI             | `cray-mpich/{8.1.28, 8.1.30, 8.1.33, 9.0.1, 9.1.0}`                 |
| CUDA            | `cudatoolkit/{12.4, 12.9, 13.0, 13.2}`                              |
| BLAS/LAPACK     | `cray-libsci/{24.07.0, 25.09.0, 26.03.0}`                           |
| FFTW            | `cray-fftw/3.3.10.{3, 6, 8, 11}`                                    |

`cpe/26.03` pins a mutually tested set of these. Vary one axis off that baseline at a
time; a build that moves several at once is not a test of anything in particular.

## Site-specific gotchas

### ScaLAPACK: how to enable it, and the three bugs that were in the way

ScaLAPACK works, but only since 2026-09-10, and enabling it still needs one flag.

**Detection.** Cray libsci embeds ScaLAPACK and BLACS but ships no standalone
`libscalapack` and no `scalapack.pc`, so Q3's search -- `pkg_check_modules(scalapack)`
then `find_library(scalapack)` -- finds nothing and leaves `Q3_WITH_SCALAPACK`
undefined. Point Q3 at libsci:

```bash
cmake ... -DQ3_WITH_MPI=ON -DSCALAPACK_LIBRARIES=$CRAY_LIBSCI_PREFIX/lib/libsci_gnu_mpi_mp.so
```

The `_mp` variant is the OpenMP-threaded one, matching what the wrapper selects for a
build using `-fopenmp`. Configure should then say `ScaLAPACK found:`.

**Then three bugs, in the order they surfaced.** Each hid the next, and none had ever
been seen because the first one stopped any ScaLAPACK build from linking at all:

1.  Q3 declared and called `pzhegv_`, which is not in the ScaLAPACK API -- netlib
    offers `PZHEEV`, `PZHEEVD`, `PZHEEVX` and, for the generalized problem, only the
    expert driver `PZHEGVX`. Every executable failed to link with
    `undefined reference to pzhegv_`. Ported to `pzhegvx_`.
2.  The BLACS grid was built by passing the handle from `Csys2blacs_handle` (the C
    entry point) into `blacs_gridinit_` (the Fortran one). The two read their argument
    in different numbering spaces: the handle came back as `1`, BLACS used it as a
    communicator, and MPICH aborted in `MPI_Comm_group` with
    `Invalid communicator, comm=0x1`. Q3 now stays inside the C API
    (`Cblacs_gridinit`/`Cblacs_gridinfo`/`Cblacs_gridexit`), which also takes the
    context by value rather than by pointer.
3.  The context was additionally fetched via `blacs_get_(handle, 0, &ctxt)`, which asks
    for the _default_ system context -- built on `MPI_COMM_WORLD`, not on the group
    communicator -- so the grid spanned world ranks while `nprow*npcol` was sized from
    the group. Surplus ranks fell outside the grid, reported `{-1,-1}`, and then aborted
    on `blacs_gridexit_` with `Trying to exit non-existent context`.

4.  With the link fixed, a fourth bug surfaced at runtime: after the replicated-G
    fallback, `group_size()` becomes 1 while the group communicator keeps all its ranks,
    so every rank solved its own copy of the problem while still entering the collective
    ScaLAPACK path -- aborting in `MPI_Allreduce` with a segfault on every rank. The
    subspace solve now branches on `group_size() <= 1` and solves locally, which is what
    `MPIExecutionPolicy` documents that callers must do. See
    [performance_nersc_perlmutter.md](performance_nersc_perlmutter.md) for how it was tracked down.

**Verified.** With ScaLAPACK on, `q3_mpi` reproduces the no-ScaLAPACK energies exactly
-- -16.913417932 Ry on `nc.yml` at 4 ranks/2 pools and 8 ranks/2 pools, -16.925260460 Ry
on `nc_kpt_8x8x8.yml` at 8 ranks/2 pools -- with eigenvalues matching to every printed digit.

Note that the ScaLAPACK path only runs with **more than one rank per pool**: with one
rank per pool the subspace eigenproblem is local and never reaches the distributed
solver. A run with `--pools` equal to the rank count tells you nothing about it.

### Configure on a login node, compile on a compute node

Configuration downloads six or seven dependencies. Compute nodes have no route to
github.com or gitlab.com, so `cmake -S ... -B ...` must run on a login node, where it
populates `_deps` once. Only then is `cmake --build` safe to run inside an allocation.

The login node is cgroup-limited to 8 visible cores, so `-j 8` is the ceiling there.
Anything larger belongs in an interactive allocation.

### Build under `$SCRATCH`, not CFS

The source tree lives on CFS, which is the wrong filesystem for build I/O. Keep the
source where it is and put the build tree on scratch:

```bash
cmake -S /global/cfs/cdirs/m2651/m3/q3 -B $SCRATCH/q3/build/<variant>
```

`$SCRATCH` is not backed up and is purged, so the build tree is disposable and the
install prefix is not. Installs go to `/global/cfs/projectdirs/m2651/m3/q3/install`.

Note that this prefix is _inside the source repository_ — the same directory as
`/global/cfs/cdirs/m2651/m3/q3` — and `install/` is not listed in `.gitignore`.

### The source tree has two names, and mixing them breaks code generation

`/global/cfs/cdirs/m2651/m3/q3` and `/global/cfs/projectdirs/m2651/m3/q3` are the same
directory; `cdirs` is the canonical spelling that `readlink -f` returns. CMake bakes
whichever spelling it saw into the generated build rules, and it takes that from the
**current working directory**, not from the absolute path given to `-S`. Configure from
one shell and reconfigure from another sitting at the other spelling, and the schema
generator dies with

```
ValueError: '/global/cfs/projectdirs/.../total_energy.hpp' is not in the subpath of '/global/cfs/cdirs/...'
```

because `scripts/repo/generate_cpp_from_schemas.py` resolved the project root from its own
`__file__` while taking `--output-dir` verbatim, so the two were compared in different
spellings.

**Fixed 2026-09-09**: the generator now canonicalises both, after the relative/absolute
branch rather than before it, so a relative `--output-dir` still means relative to the
project root. Either spelling works. Staying with `cdirs` is still tidier — it is what
`readlink -f` returns, so it is what shows up in build logs and cache entries — but it is
a convention now, not a requirement, and a checkout predating the fix still needs it.

### The Python extension needs `CMAKE_POSITION_INDEPENDENT_CODE`

`-DQ3_ENABLE_PYTHON=ON` alone fails at link:

```
ld: relocation R_X86_64_TPOFF32 against hidden symbol
    `spglib_error_code.lto_priv.0' can not be used when making a shared object
```

`q3_core` is a shared module, but the vendored static dependencies are not built
position-independent, and with interprocedural optimization on -- which it is here,
since the Cray wrapper passes the probe -- spglib's thread-local error code ends up in a
form that cannot be relocated into a `.so`. Add
`-DCMAKE_POSITION_INDEPENDENT_CODE=ON`; `build.sh python` does.

### Band groups top out at four ranks per pool

`--pools` must divide the rank count and must not exceed the irreducible k-point count;
ranks beyond that go into band groups, which distribute G-vectors across each pool. On
the Si 8x8x8 example that stops working above **4 ranks per pool**: the wave FFT grid is
24x24x24 but holds only ~1564 plane waves per k-point, so the G-vector sphere spans a
limited range of z-planes and a finer slab split leaves some ranks with none. The code
then reports

```
Band parallelism: some ranks received 0 plane waves. Falling back to replicated G-vectors.
```

and every rank in the pool redundantly repeats the whole solve -- the Davidson counter
shows 8x the iterations for 8 ranks per pool. The answer stays correct; the work is
wasted. The limit is a property of the system's plane-wave count, not of the node, so it
moves with cutoff and cell size.

### The CUDA math libraries are in a sibling directory

`cudatoolkit/13.2` resolves to the NVHPC 26.5 tree, where `$CUDA_HOME/lib64` holds only
`libcudart`; cuBLAS, cuSOLVER and cuFFT live under `math_libs/13.2/lib64`. CMake's
`FindCUDAToolkit` handles this splayed layout explicitly from 3.24, and 3.28 is what is
installed, so `CUDA::cublas`, `CUDA::cusolver` and `CUDA::cufft` should resolve. Confirm
it in the configure output rather than assuming it — a missing one of these surfaces as
a generation-time error naming the target, not a `find_package` failure.

### Out-of-tree builds and `cmake --install` (both fixed 2026-09-09)

Two defects surfaced during Phase 1. Both are fixed in the tree; they are recorded here
because a checkout predating the fix will hit them.

`tests/integration/test_cli.cpp` derived the binary location as `EXAMPLES_DIR/../build/q3`,
hardcoding an in-tree `build/`. Every out-of-tree build — which this plan requires — failed
`CLISmokeTest.BinaryExistsAndIsExecutable` and `CLISmokeTest.RunSiNCExampleProducesResults`
while the binary itself was fine. CMake now passes `Q3_BINARY_DIR` alongside `EXAMPLES_DIR`
and the test uses it.

`cmake --install` failed outright with `include could not find requested file` on
`_deps/nlohmann_json-build/cmake_install.cmake`. The `CMAKE_SKIP_INSTALL_RULES` guard that
keeps LibXC from scattering its headers and `xc-info` into the prefix stops all five fetched
subprojects from writing an install script, but the top-level script — generated afterwards,
with the variable restored — still `include()`s every one of them. This broke installing on
any machine without a system LibXC, Perlmutter included. The fix writes an empty
`cmake_install.cmake` where each suppressed one would have been.

### The `cpu` and `gpu` meta-modules

`gpu/1.0` sets `craype-accel-nvidia80` and `MPICH_GPU_SUPPORT_ENABLED=1`. Load `cpu` for
the CPU-only variants and `gpu` for the CUDA ones, so the wrapper is not adding GPU link
lines to a build that has no use for them.

### RPATH is not actually needed here

The general advice is to export `CRAY_ADD_RPATH=yes` before configuring, because
`cmake --install` records only the library directories CMake knows about and not those
the compiler wrapper injects. Measured on the `cpu-serial` install, that turns out not
to matter on Perlmutter: the installed `q3` carries no RPATH beyond darshan, yet every
Cray library it needs — `libfftw3`, `libfftw3_threads`, `libsci_gnu_mpi_mp`, `libmpi_gnu`
— resolves through `/opt/cray/pe/lib64`, which is on the default loader path. The binary
runs correctly in a shell with no modules loaded at all, not even `cray-fftw`.

Set `CRAY_ADD_RPATH=yes` anyway if an install must survive a CPE upgrade, since those
`/opt/cray/pe/lib64` symlinks follow the current default PE.

### Compiler and build type are cached

Neither can be changed in an existing build directory. Switching either means
`rm -rf` on the build tree. This is also why each variant gets its own directory
rather than being reconfigured in place.

## Build sequence

### Phase 0 — prepare

Remove the stale `build/` directory in the source tree: it holds a `CMakeCache.txt` and
`CMakeFiles` from a configure that never completed, recording `Q3_WITH_MPI=OFF` and a
compiler choice that cannot now be changed in place. Create `$SCRATCH/q3/build` for build trees and `$SCRATCH/q3/logs` for
configure/build/test logs, which are worth keeping and are not written anywhere by
default. The `.venv` is already in order.

### Phase 1 — `cpu-serial`

The smallest thing that can work, and the one that proves FFTW discovery, libsci, the
schema generator, the interprocedural-optimization probe and every `FetchContent` pull.
Everything after this reuses the dependency cache it populates.

These are the exact commands that produced the installed `cpu-serial` build, verbatim.

```bash
module load cpu PrgEnv-gnu cray-fftw
cd /global/cfs/cdirs/m2651/m3/q3
mkdir -p $SCRATCH/q3/build $SCRATCH/q3/logs

CC=cc CXX=CC cmake -S /global/cfs/cdirs/m2651/m3/q3 -B $SCRATCH/q3/build/cpu-serial \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX=/global/cfs/projectdirs/m2651/m3/q3/install

cmake --build $SCRATCH/q3/build/cpu-serial -j 8

cd $SCRATCH/q3/build/cpu-serial && OMP_NUM_THREADS=1 ctest -j 8 --output-on-failure

cmake --install $SCRATCH/q3/build/cpu-serial \
    --prefix /global/cfs/projectdirs/m2651/m3/q3/install
```

`module load cpu` swaps out `gpu/1.0` and takes `cudatoolkit` and
`craype-accel-nvidia80` with it, leaving `CRAY_ACCEL_TARGET` unset — which is what a
CPU-only build wants. Note also that `module load` must not appear in a shell pipeline:
`module` is a shell function, and a pipeline runs it in a subshell where the environment
changes are discarded.

Read the configure output before building: which Python interpreter it chose, whether
`Found FFTW3 threads` appears, and whether interprocedural optimization was enabled or
skipped.

**Result, 2026-09-09.** Clean at `-j 8` on a login node, about two minutes. FFTW
discovery needed no `-D`, OpenMP 4.5 and `libfftw3_threads` were found, the in-tree
`.venv` was selected for code generation, LibXC was fetched and built, and the
interprocedural-optimization probe _passed_ under the Cray wrapper — so this build has
IPO enabled, which the CMakeLists comments treat as unlikely on Cray toolchains.

Zero errors, 213 warnings, all in Q3's own headers rather than the vendored
dependencies; 131 are `-Wclass-memaccess`, concentrated in `davidson.hpp` (91) and the
Hamiltonian applicators.

`ctest` reports **1105/1105** once the two fixes above are in place. The silicon example
gives a total energy of -16.913417932 Ry against the reference -16.91341767 Ry, a
difference of 2.6e-7 Ry. `q3` and `q3_utility` are installed to
`/global/cfs/projectdirs/m2651/m3/q3/install/bin` and run correctly from a shell with no
modules loaded.

Note that the `q3_utility` _target_ produces a binary named `q3_utility`.

### Phase 2 — `cpu-mpi`

Adds `-DQ3_WITH_MPI=ON`. Pass `SCALAPACK_LIBRARIES` too for the distributed subspace
eigensolver -- see the ScaLAPACK gotcha above; `build.sh` leaves it off, so add it
explicitly:

```bash
module load cpu PrgEnv-gnu cray-fftw
cd /global/cfs/cdirs/m2651/m3/q3
CC=cc CXX=CC cmake -S /global/cfs/cdirs/m2651/m3/q3 -B $SCRATCH/q3/build/cpu-mpi \
    -DCMAKE_BUILD_TYPE=Release -DQ3_WITH_MPI=ON
cmake --build $SCRATCH/q3/build/cpu-mpi -j 8
```

Configure should report `Found FFTW3 MPI:`, and either `ScaLAPACK found:` or the
`ScaLAPACK not found` warning depending on whether the flag was passed. Both build and
both run; without it the subspace eigenproblem is solved redundantly on every rank of a
pool.

Verify with `q3_tests_mpi` and a short `srun -n 4 ./q3_mpi --pools 4` on an example.
`--pools` must divide the rank count and must not exceed the irreducible k-point count,
which the log reports as `IBZ reduction`. Run it in an allocation, not on a login node:
oversubscribed MPI busy-waits in a way indistinguishable from a hang.

**Result, 2026-09-10.** `ctest` 1235/1235. `q3_tests_mpi` on 2 ranks: 14 passed, 1
skipped. Every topology gives the same energy as serial: 2 ranks/2 pools and 4 ranks/2
pools both -16.913417932 Ry on `nc.yml`, and 4 ranks/4 pools gives -16.925260460 Ry on
`nc_kpt_8x8x8.yml`, matching that input's own serial run exactly. Band groups work (the
4-rank/2-pool run reports `G-vector distribution = slab-distributed across the band
group` with no fallback).

Pool scaling on `nc_kpt_8x8x8.yml` (60 irreducible k-points), taking the solver's own
reported total rather than wall clock -- `srun` startup is 2-9 s and swamps runs this
short:

| Ranks | Pools | Ranks/pool | Solver    | Speedup |
| ----- | ----- | ---------- | --------- | ------- |
| 1     | 1     | 1          | 2288.2 ms | --      |
| 4     | 4     | 1          | 570.9 ms  | 4.0x    |
| 8     | 8     | 1          | 300.1 ms  | 7.6x    |
| 16    | 16    | 1          | 164.8 ms  | 13.9x   |
| 32    | 32    | 1          | 81.7 ms   | 28.0x   |
| 64    | 32    | 2          | 93.0 ms   | 24.6x   |
| 128   | 32    | 4          | 110.5 ms  | 20.7x   |

Near-linear to 32 ranks, then it reverses: 60 k-points cap the pool count, so ranks past
32 can only add band groups, and on this problem those cost more than they return. 32
ranks is the optimum, and 128 is 35% slower than 32.

Two reporting bugs turned up here, both since fixed:

-   `number of k-points` and the `Eigenvalues (eV):` list were **pool-local**: with 32
    pools the log said `number of k-points = 2` and printed 2 eigenvalue blocks for a
    60-k-point calculation, labelled `k_0`/`k_1` rather than by global index. On top of
    that the list was capped at 10 with no indication, so even a serial run showed 10 of 60. `results.json` was never affected -- it carried all 60 throughout. The iterator
    now assembles the global grid before reporting, reusing the same pack-and-gather the
    SCF flavor uses to build `results.json`, and the cap is gone: **every k-point prints,
    in every configuration, labelled by its irreducible index.**
-   After the replicated-G fallback the summary contradicted the topology line, printing
    `1 rank(s) per pool` where the topology line said 8. Ranks per pool is now derived
    from `processes / pools`, so the parenthetical always factors the rank count.

### Phase 3 — `gpu-cuda`

```bash
module load gpu PrgEnv-gnu cray-fftw cudatoolkit/13.2
```

with `-DQ3_WITH_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=80`. This is the configure most
likely to need attention; check that the CUDA math-library targets resolved. Validate in
an allocation on `-C gpu` against the `m2651_g` account.

**Result, 2026-09-10.** Configure found the toolkit despite the splayed NVHPC layout,
and `q3_gpu_cuda` built with no errors. Validated on an A100-SXM4-40GB (compute
capability 8.0, confirming `sm_80`): `nc.yml` in double precision gives -16.913417932 Ry
and `nc_kpt_8x8x8.yml` -16.925260460 Ry, both exactly matching CPU. Mixed precision gives
-16.913416823 Ry, differing at the sixth decimal, as expected for FP32 FFTs via cuFFT.

Not a throughput result: Si with 60 k-points is far too small to occupy an A100, and the
8x8x8 case took longer on the GPU than on one CPU core. `PERFORMANCE.md` benchmarks a
large Cu supercell for that reason.

### Phase 4 — `cpu-mpi-cuda`, then Tier 2

Combine phases 2 and 3 in one tree for a single install prefix. Add `cpu-debug` and
`python` as the work calls for them.

**Result, 2026-09-10.** `cpu-mpi-cuda` builds all six binaries (`q3`, `q3_utility`,
`q3_mpi`, `q3_gpu_cuda`, `q3_tests`, `q3_tests_mpi`) with no errors. `cpu-debug` builds
and passes 1235/1235. `python` builds `q3_core.cpython-313-x86_64-linux-gnu.so` once
`CMAKE_POSITION_INDEPENDENT_CODE` is on, and it imports, exposing `run_scf`, `run_nscf`,
`SCFSolver` and `reset_counters`.

### Phase 5 — compiler sweep

Only once Tier 1 is green, one programming environment at a time, each in a fresh build
directory.

**Result, 2026-09-10.** Two of the three alternative programming environments build and
pass the full suite:

| PrgEnv  | Compiler                    | Configure | Build        | ctest     |
| ------- | --------------------------- | --------- | ------------ | --------- |
| `gnu`   | GCC 14.3.0                  | yes       | 205 warnings | 1235/1235 |
| `cray`  | Cray clang 21.0.0           | yes       | 76 warnings  | 1235/1235 |
| `intel` | Intel oneAPI DPC++ 2025.3.1 | yes       | 81 warnings  | 1235/1235 |
| `llvm`  | --                          | **no**    | --           | --        |

`PrgEnv-intel` needs two flags the other environments do not, both added to
`scripts/hpc/nersc_perlmutter/build.sh` on 2026-09-11 and neither visible in a build log or a test
result:

-   **The threaded libsci.** The Cray wrappers link `libsci_<compiler>_mp` when they see
    an OpenMP flag they recognise on the link line, and CMake's FindOpenMP picks
    `-fiopenmp` for oneAPI, which they do not count. The intel build therefore linked the
    serial `libsci_intel.so.6` and got **1.13x out of eight threads, against 4.24x for gnu
    and 4.67x for cray** on the Cu 3x3x3 benchmark — at one thread all three are within 2%
    of each other, so it was entirely the threading (camsai/q3#14). The build script now
    puts `-fopenmp`, oneAPI's alias for the same switch, on the link line. Every variant
    now also reports which libsci it linked at the end of the build, since a serial one
    builds cleanly and passes the whole suite.
-   **An rpath for `libiomp5.so`.** The intel binaries would not start outside the shell
    that built them — `error while loading shared libraries: libiomp5.so` — because
    neither `module load intel` nor `PrgEnv-intel` puts oneAPI's lib directory anywhere a
    non-interactive launch inherits. The link line now carries an rpath to wherever the
    compiler reports the library, so a batch script needs no `LD_LIBRARY_PATH`.

`PrgEnv-llvm/21.1.4` cannot be used through the Cray wrappers on CPE 26.03. The module
does not load `craype`, so `CC` does not exist and `cc` falls through to `/usr/bin/cc`;
loading `craype` alongside makes the wrapper reject the environment outright with
`Error: LLVM is not a recognized PrgEnv.` Building under it would mean bypassing the
wrappers and supplying FFTW, LAPACK and MPI flags by hand, which has not been tried.
`PrgEnv-nvidia` has not been attempted.

### Phase 7 — measure

`scripts/hpc/nersc_perlmutter/benchmark.sh cpu` and `... gpu`, inside the allocations its header
names. Results and interpretation live in
[performance_nersc_perlmutter.md](performance_nersc_perlmutter.md).

### Phase 6 — record what worked

Fold the recipes that succeeded back into this page, or into module files under
`/global/common/software/m2651`, so the next person does not repeat the investigation.

## Open questions

-   **`install/` is untracked but not ignored.** The chosen prefix sits inside the git
    worktree and `.gitignore` has no entry for it, so `git add -A` would stage the
    installed binaries. Adding `install/` to `.gitignore` would settle it.
-   **Allocations.** Phases 1 through 5 each want an interactive allocation to compile
    and validate in. Small variants can be built on the login node at `-j 8` instead.
-   **Scope.** Whether to work through all of Tier 1 or stop after `cpu-serial` and
    `cpu-mpi`.
