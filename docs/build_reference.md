# Build Configuration Reference

Every option the build accepts, what it defaults to, and what to set it to on
the three kinds of machine Q3 is built on: a Homebrew Mac, an Ubuntu box, and a
module-based HPC system (the examples use NERSC Perlmutter).

For the three-command version, see [§2.2 of the README](../README.md#22-build-from-source).
This page is the reference you open when one of those three commands does not
work.

## Before CMake: the Python the build needs

`cmake --build` runs `scripts/repo/generate_cpp_from_schemas.py`, which needs
**jinja2, jsonref and pyyaml**. These are inputs to the build, not runtime
dependencies of the `q3` Python package, so create the environment first:

```bash
python3 -m venv .venv && .venv/bin/pip install -e ".[build]"
```

A `.venv` at the repository root is detected automatically. To use a different
interpreter, name it and it wins over the `.venv`:

```bash
cmake -S . -B build -DPython3_EXECUTABLE=$(which python)
```

Configuration reports which interpreter it chose. If the modules are missing it
warns and falls back to the committed headers in `src/generated`, which is
correct as long as you are not changing `schemas/` — and is also what lets a
clone made without `--recurse-submodules` build. Pass
`-DQ3_REQUIRE_SCHEMA_CODEGEN=ON` to make that a configuration failure instead;
CI does.

## Passing variables

One `-D` per variable, each a single value:

```bash
cmake -S . -B build -DFFTW3_LIB=/path/libfftw3.so -DFFTW3_MPI_LIB=/path/libfftw3_mpi.so
```

Two rules that account for most confusion:

-   **Never put several paths in one variable.** CMake lists are
    semicolon-separated, and a space-separated string becomes one value, so the
    build looks for a single file whose name is the whole string and reports
    `No rule to make target '<path> <path> <path>'`.
-   **The compiler and the build type are cached.** Re-running `cmake` in an
    existing build directory will not change them. To switch compilers, delete
    the directory:

    ```bash
    rm -rf build && CC=cc CXX=CC cmake -S . -B build
    ```

## Toolchain

| Variable               | Default                                 | Homebrew macOS     | Ubuntu | Perlmutter          |
| ---------------------- | --------------------------------------- | ------------------ | ------ | ------------------- |
| `CMAKE_C_COMPILER`     | system                                  | `clang` / `gcc-14` | `gcc`  | `cc`                |
| `CMAKE_CXX_COMPILER`   | system                                  | `clang++`/`g++-14` | `g++`  | `CC`                |
| `CMAKE_BUILD_TYPE`     | `Release` (`Debug` under `Q3_COVERAGE`) | —                  | —      | —                   |
| `CMAKE_INSTALL_PREFIX` | `/usr/local`                            | any writable path  | any    | `$CFS/<project>/q3` |

**GCC 9 is the floor.** Q3 uses `<filesystem>`, which libstdc++ ships from GCC 8
and links without `-lstdc++fs` only from GCC 9. Older compilers fail at
configuration with that explanation rather than partway through the first
compile. On an HPC login node the _default_ compiler is often far older than the
modules the machine offers, so load one and configure in a fresh build
directory.

Interprocedural optimization is enabled when the toolchain supports it and
skipped when it does not — some compiler wrappers cannot link LTO objects. GPU
targets never use it.

## Numerical libraries

| Variable              | Purpose                  | Homebrew macOS                     | Ubuntu                                  | Perlmutter                                     |
| --------------------- | ------------------------ | ---------------------------------- | --------------------------------------- | ---------------------------------------------- |
| `FFTW3_LIB`           | FFTW (required)          | `/opt/homebrew/lib/libfftw3.dylib` | `/usr/lib/x86_64-linux-gnu/libfftw3.so` | `$FFTW_DIR/libfftw3.so`                        |
| `FFTW3_INCLUDE`       | `fftw3.h` (required)     | `/opt/homebrew/include`            | `/usr/include`                          | `$FFTW_INC`                                    |
| `FFTW3_THREADS_LIB`   | threaded transforms      | `…/libfftw3_threads.dylib`         | `…/libfftw3_threads.so`                 | `$FFTW_DIR/libfftw3_threads.so`                |
| `FFTW3_MPI_LIB`       | distributed transforms   | `…/libfftw3_mpi.dylib`             | `…/libfftw3_mpi.so`                     | `$FFTW_DIR/libfftw3_mpi.so`                    |
| `BLA_VENDOR`          | pick a BLAS/LAPACK       | unset (Accelerate)                 | unset (reference)                       | unset — the wrapper supplies libsci            |
| `Libxc_DIR`           | system LibXC 5.2 or 7.0  | unset (fetched)                    | unset if `libxc-dev` is installed       | unset (fetched)                                |
| `SCALAPACK_LIBRARIES` | MPI subspace eigensolver | unset                              | unset if `libscalapack-*-dev` installed | `$CRAY_LIBSCI_PREFIX/lib/libsci_gnu_mpi_mp.so` |

**Usually none of these need setting**, with one exception: Cray libsci embeds ScaLAPACK
but ships no standalone `libscalapack` and no `scalapack.pc`, so Q3's search finds
nothing and `q3_mpi` silently solves the subspace eigenproblem redundantly on every rank
of a pool. Naming the libsci library enables the distributed solver. (Q3's ScaLAPACK path
did not work at all before 2026-09-10; see
[build_nersc_perlmutter.md](build_nersc_perlmutter.md) for what was wrong, and
[performance_nersc_perlmutter.md](performance_nersc_perlmutter.md) for a remaining limitation above
four ranks per pool.)
FFTW discovery reads `FFTW_ROOT`,
`FFTW_DIR`, `FFTW_HOME`, `FFTWDIR`, `FFTW3_ROOT` and `FFTW_INC` from the
environment, which is what `module load cray-fftw` exports, so on Perlmutter
loading the module is enough. Set them explicitly only when discovery fails —
and note the library may be `.a`, and the directory may be `lib64`, so look
before guessing:

```bash
env | grep -i fftw
```

```bash
find "$FFTW_DIR" "$FFTW_ROOT" -name 'libfftw3_mpi*' 2>/dev/null
```

`-DCMAKE_LIBRARY_PATH=<directory>` also works and is less brittle than naming
each file. `-DCMAKE_PREFIX_PATH` is not reliable here, because it searches
`<prefix>/lib` and misses `lib64`.

**A system LibXC is used only from a tested series:** 5.2, which Ubuntu 24.04
ships and CI builds with, and 7.0, the series built from source otherwise. Any
other version found — Homebrew's `libxc` was 7.1.2 at the time of writing — is
named in the configure output and LibXC 7.0.0 is built from source instead,
because LibXC 7.1.0 to 7.1.2 send spin-polarized PBE to NaN in an
antiferromagnet ([KNOWN-ISSUES.md](../KNOWN-ISSUES.md)).
`-DQ3_ALLOW_UNTESTED_SYSTEM_LIBXC=ON` takes whatever LibXC is found; a run still
stops at the first non-finite value LibXC returns rather than converging to NaN.

## Q3 options

| Option                      | Default | Effect                                                             |
| --------------------------- | ------- | ------------------------------------------------------------------ |
| `Q3_BUILD_TESTS`            | `ON`    | Builds `q3_tests` (fetches GoogleTest)                             |
| `Q3_WITH_MPI`               | `OFF`   | `q3` is the MPI build, also named `q3_mpi`                         |
| `Q3_WITH_OPENMP`            | `ON`    | Links an OpenMP runtime                                            |
| `Q3_WITH_FFTW_THREADS`      | `ON`    | Threaded FFTs, driven by `OMP_NUM_THREADS` at run time             |
| `Q3_WITH_GPU`               | `OFF`   | `q3` is the Apple Metal build, also named `q3_gpu`                 |
| `Q3_WITH_CUDA`              | `OFF`   | `q3` is the NVIDIA build, also named `q3_gpu_cuda`                 |
| `Q3_WITH_HIP`               | `OFF`   | `q3` is the AMD build, also named `q3_gpu_hip`                     |
| `Q3_WITH_SYCL`              | `OFF`   | `q3` is the Intel build, also named `q3_gpu_sycl`                  |
| `Q3_ENABLE_PYTHON`          | `OFF`   | Builds the `q3_core` nanobind extension                            |
| `Q3_COVERAGE`               | `OFF`   | Coverage instrumentation; also switches the default build type     |
| `Q3_REQUIRE_SCHEMA_CODEGEN` | `OFF`   | Fail configuration when the generator's Python modules are missing |

A build tree compiles one engine binary, always named `q3`; the options above
choose which entry point it is, and the variant name is a symlink to it. At
most one GPU option may be on, and `Q3_WITH_MPI` combines only with
`Q3_WITH_CUDA`, which gives `q3_mpi_gpu_cuda`. A serial and a parallel binary
come from two build trees.

For CUDA, `CMAKE_CUDA_ARCHITECTURES` defaults to `80` (A100). Set it for other
hardware — `90` for H100. `native` is not usable on a login node, which has no
GPU at configure time.

## Installing to a prefix

The build directory is not an install tree. To place the binaries somewhere
specific, either choose the prefix when configuring:

```bash
cmake -S . -B build -DCMAKE_INSTALL_PREFIX=$CFS/<project>/q3 && cmake --build build -j 16 && cmake --install build
```

or decide afterwards, which overrides it:

```bash
cmake --install build --prefix $CFS/<project>/q3
```

Either way you get `<prefix>/bin/q3` and `<prefix>/bin/q3_utility`, plus the
variant name (`q3_mpi`, `q3_gpu_cuda`, ...) as a relative symlink to `q3` when
the tree is not the serial build. Add `<prefix>/bin` to `PATH`.

Installed binaries record the directories of the libraries they were linked
against, so they run without `LD_LIBRARY_PATH`. Libraries injected implicitly by
a compiler wrapper are invisible to CMake and are not covered: on Cray systems
either export `CRAY_ADD_RPATH=yes` before configuring, or keep the same modules
loaded at run time.

## A build on Perlmutter, start to finish

The steps below are also packaged as scripts, which carry the module sets, the flags and
every site workaround -- see
[build_nersc_perlmutter.md](build_nersc_perlmutter.md) for the reasoning and
[performance_nersc_perlmutter.md](performance_nersc_perlmutter.md) for how the variants perform:

```bash
scripts/hpc/nersc_perlmutter/build.sh cpu-mpi
scripts/hpc/nersc_perlmutter/validate.sh cpu-mpi     # inside an allocation; --help gives the salloc line
scripts/hpc/nersc_perlmutter/install.sh cpu-mpi
```

Doing it by hand:

```bash
module load PrgEnv-gnu cmake cray-fftw cray-libsci    # cray-mpich is already default
```

```bash
git clone --recurse-submodules <repository-url> q3 && cd q3
python3 -m venv .venv && .venv/bin/pip install -e ".[build]"
```

```bash
rm -rf build && CC=cc CXX=CC cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DQ3_WITH_MPI=ON
```

Read the configuration output before building — it reports the interpreter, the
FFTW paths, whether interprocedural optimization is on, and the install prefix.
Then:

```bash
cmake --build build -j 16 && cmake --install build --prefix $CFS/<project>/q3
```

Run on a compute node, not the login node:

```bash
salloc -N 1 -C cpu -q interactive -t 00:30:00 -A <account>
srun -n 4 ./build/q3 --pools 4 examples/total_energy--semiconductor--silicon/nc.yml
```

`--pools` should divide the rank count and not exceed the irreducible k-point
count, which the log reports as `IBZ reduction`. One rank per pool is the
configuration with measured scaling; see
[KNOWN-ISSUES.md](../KNOWN-ISSUES.md) for what band groups cost.

## When it still fails

[TROUBLESHOOTING.md](../TROUBLESHOOTING.md) lists the build and environment
failures seen so far, each with its cause and fix.
