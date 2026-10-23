#!/usr/bin/env bash
#
# Build a Q3 variant on NERSC Perlmutter.
#
#   scripts/hpc/nersc_perlmutter/build.sh <variant> [extra cmake args...]
#   scripts/hpc/nersc_perlmutter/build.sh --list
#
# Variants (each tree builds one engine binary, always named q3; the variant
# name in parentheses is a symlink to it):
#   cpu-serial     q3, q3_utility, q3_tests
#   cpu-mpi        q3 (q3_mpi), q3_utility, q3_tests, q3_tests_mpi
#   cpu-mpi-scalapack  cpu-mpi plus the distributed subspace eigensolver
#   gpu-cuda       q3 (q3_gpu_cuda; NVIDIA A100, sm_80), q3_utility, q3_tests
#   cpu-mpi-cuda   q3 (q3_mpi_gpu_cuda), one CUDA device per MPI rank
#   cpu-debug      Debug build type
#   python         q3_core nanobind extension
#
# Build trees go under $SCRATCH/q3/build/<variant>; logs under $SCRATCH/q3/logs.
# Install with scripts/hpc/nersc_perlmutter/install.sh, validate with validate.sh.
#
# Run the CONFIGURE step on a login node: it downloads six or seven
# FetchContent dependencies and compute nodes have no route to github.com. Only
# once _deps is populated is `cmake --build` safe inside an allocation.
#
# Everything this script encodes that is not obvious is explained in
# docs/build_nersc_perlmutter.md. The short version:
#
#   * ScaLAPACK needs -DSCALAPACK_LIBRARIES pointing at libsci: it embeds
#     ScaLAPACK but ships no standalone libscalapack and no scalapack.pc, so
#     Q3's find_library/pkg-config search comes up empty. The
#     cpu-mpi-scalapack variant passes it. It is a separate variant rather than
#     the default for cpu-mpi because the distributed solver only engages with
#     more than one rank per pool, and until 2026-09-10 it did not work at all
#     (see the ScaLAPACK section of docs/build_nersc_perlmutter.md for the three
#     bugs that were in the way).
#   * The python variant needs -DCMAKE_POSITION_INDEPENDENT_CODE=ON. The
#     vendored static dependencies are not built PIC, and with LTO on (the Cray
#     wrapper passes the IPO probe here) spglib's thread-local error code cannot
#     be relocated into a shared object.
#   * `module load` must not appear in a shell pipeline. `module` is a shell
#     function, and a pipeline runs it in a subshell where the environment
#     changes are thrown away -- the modules appear to load and nothing happens.
#   * FFTW needs no -D at all: cray-fftw exports FFTW_ROOT/FFTW_DIR/FFTW_INC and
#     Q3's discovery reads them.

set -eo pipefail

SOURCE_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd -P)

usage() { sed -n '2,/^$/p' "${BASH_SOURCE[0]}" | sed 's/^#//;s/^ //'; }

VARIANT=${1:-}
case "$VARIANT" in
    -h|--help|"") usage; exit 0 ;;
    --list) echo "cpu-serial cpu-mpi cpu-mpi-scalapack gpu-cuda cpu-mpi-cuda cpu-debug python"
            exit 0 ;;
esac
shift

if [ -z "$SCRATCH" ]; then
    echo "build.sh: \$SCRATCH is unset -- is this Perlmutter?" >&2
    exit 1
fi

# Q3_PRGENV selects the programming environment (gnu, cray, llvm, intel, nvidia)
# and Q3_BUILD_DIR overrides where the tree goes, so a compiler sweep is a loop
# without a second script:
#
#   for pe in cray llvm intel; do
#       Q3_PRGENV=$pe Q3_BUILD_DIR=$SCRATCH/q3/build/cpu-$pe \
#           scripts/hpc/nersc_perlmutter/build.sh cpu-serial
#   done
#
# The compiler is cached, so each programming environment needs its own fresh
# build directory -- reconfiguring an existing one will not switch compilers.
PRGENV=${Q3_PRGENV:-gnu}
BUILD_DIR=${Q3_BUILD_DIR:-$SCRATCH/q3/build/$VARIANT}
LOG_DIR=$SCRATCH/q3/logs
LOG_TAG=$(basename "$BUILD_DIR")
mkdir -p "$BUILD_DIR" "$LOG_DIR"

# ---------------------------------------------------------------- modules ----
# `module load cpu` swaps out gpu/1.0 and takes cudatoolkit and
# craype-accel-nvidia80 with it, leaving CRAY_ACCEL_TARGET unset -- which is
# what a CPU-only build wants, so the wrapper adds no GPU link lines.
case "$VARIANT" in
    gpu-cuda|cpu-mpi-cuda)
        module load gpu "PrgEnv-$PRGENV" cray-fftw cudatoolkit/13.2
        ;;
    cpu-serial|cpu-mpi|cpu-mpi-scalapack|cpu-debug|python)
        module load cpu "PrgEnv-$PRGENV" cray-fftw
        ;;
    *)
        echo "build.sh: unknown variant '$VARIANT' (see --list)" >&2
        exit 1
        ;;
esac

# ------------------------------------------------------------ cmake flags ----
CMAKE_ARGS=(-DCMAKE_BUILD_TYPE=Release)
case "$VARIANT" in
    cpu-serial)   ;;
    cpu-mpi)      CMAKE_ARGS+=(-DQ3_WITH_MPI=ON) ;;
    # CRAY_LIBSCI_PREFIX is exported by the module loaded just above.
    cpu-mpi-scalapack)
                  CMAKE_ARGS+=(-DQ3_WITH_MPI=ON
                               -DSCALAPACK_LIBRARIES="$CRAY_LIBSCI_PREFIX/lib/libsci_gnu_mpi_mp.so") ;;
    gpu-cuda)     CMAKE_ARGS+=(-DQ3_WITH_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=80) ;;
    cpu-mpi-cuda) CMAKE_ARGS+=(-DQ3_WITH_MPI=ON -DQ3_WITH_CUDA=ON
                               -DCMAKE_CUDA_ARCHITECTURES=80) ;;
    cpu-debug)    CMAKE_ARGS=(-DCMAKE_BUILD_TYPE=Debug) ;;
    python)       CMAKE_ARGS+=(-DQ3_ENABLE_PYTHON=ON
                               -DCMAKE_POSITION_INDEPENDENT_CODE=ON) ;;
esac

# --------------------------------------------------- programming env flags ----
# The Cray wrappers link the threaded libsci (libsci_<compiler>_mp) when they
# see an OpenMP flag they recognise on the link line. CMake's FindOpenMP picks
# -fiopenmp for oneAPI, which the wrapper does not count as one, so
# PrgEnv-intel linked the serial libsci and got 1.13x out of eight threads
# where gnu got 4.24x and cray 4.67x on a BLAS-bound case -- with the threaded
# copy sitting in the same directory as the one being linked (camsai/q3#14).
# -fopenmp is oneAPI's alias for the same switch and is the spelling the
# wrapper looks for; putting it on the link line leaves CMake's own choice of
# compile flag alone.
#
# The same line carries an rpath for libiomp5.so, because the intel binaries
# otherwise refuse to start outside the shell that built them: neither `module
# load intel` nor `PrgEnv-intel` puts oneAPI's lib directory anywhere a
# non-interactive launch inherits, so a batch script gets
# `error while loading shared libraries: libiomp5.so`.
if [ "$PRGENV" = intel ]; then
    INTEL_LINK_FLAGS="-fopenmp"
    IOMP_LIBRARY=$(CC -print-file-name=libiomp5.so 2>/dev/null || true)
    if [ -f "$IOMP_LIBRARY" ]; then
        INTEL_LINK_FLAGS="$INTEL_LINK_FLAGS -Wl,-rpath,$(dirname "$IOMP_LIBRARY")"
    else
        echo "== note: libiomp5.so not found through CC -print-file-name, so no rpath was"
        echo "==       added; the binaries may need LD_LIBRARY_PATH pointed at oneAPI's"
        echo "==       lib directory to start outside this shell"
    fi
    CMAKE_ARGS+=(-DCMAKE_EXE_LINKER_FLAGS="$INTEL_LINK_FLAGS")
fi

# A login node is cgroup-limited to 8 visible cores; inside an allocation take
# what the allocation actually has.
JOBS=${JOBS:-$( [ -n "$SLURM_JOB_ID" ] && nproc || echo 8 )}

echo "== variant     : $VARIANT"
echo "== prgenv      : PrgEnv-$PRGENV"
echo "== source      : $SOURCE_DIR"
echo "== build       : $BUILD_DIR"
echo "== cmake flags : ${CMAKE_ARGS[*]} $*"
echo "== parallelism : -j $JOBS"
echo "== compiler    : $(CC --version 2>&1 | head -1)"

cd "$SOURCE_DIR"

# The venv supplies jinja2/jsonref/pyyaml for the schema generator. Without it
# CMake warns and falls back to the committed headers in src/generated, which is
# correct only while schemas/ is unchanged.
if [ ! -x "$SOURCE_DIR/.venv/bin/python3" ]; then
    echo "== note: no .venv -- create one with"
    echo "==   python3 -m venv .venv && .venv/bin/pip install -e \".[build]\""
fi

CONFIGURE_LOG=$LOG_DIR/$LOG_TAG-configure.log
BUILD_LOG=$LOG_DIR/$LOG_TAG-build.log

echo "== configuring (log: $CONFIGURE_LOG)"
CC=cc CXX=CC cmake -S "$SOURCE_DIR" -B "$BUILD_DIR" \
    "${CMAKE_ARGS[@]}" "$@" > "$CONFIGURE_LOG" 2>&1 || {
        echo "== CONFIGURE FAILED; last lines:" >&2
        tail -25 "$CONFIGURE_LOG" >&2
        exit 1
    }

# The lines worth reading back. A cpu-mpi build SHOULD report "ScaLAPACK not
# found" -- see the header.
grep -E 'Found FFTW3|MPI found|ScaLAPACK|CUDA backend|Interprocedural|Python for schema|Q3_ENABLE_PYTHON' \
    "$CONFIGURE_LOG" || true

echo "== building (log: $BUILD_LOG)"
if [ "$VARIANT" = python ]; then
    cmake --build "$BUILD_DIR" -j "$JOBS" --target q3_core > "$BUILD_LOG" 2>&1 || {
        echo "== BUILD FAILED; errors:" >&2
        grep -iE 'error' "$BUILD_LOG" | head -10 >&2
        exit 1
    }
else
    cmake --build "$BUILD_DIR" -j "$JOBS" > "$BUILD_LOG" 2>&1 || {
        echo "== BUILD FAILED; errors:" >&2
        grep -iE 'error:' "$BUILD_LOG" | head -10 >&2
        exit 1
    }
fi

echo "== built OK ($(grep -ciE 'warning:' "$BUILD_LOG") warnings)"
# Executables only. A plain listing also picks up CTest's generated .cmake
# files, and any binary left over from a target that has since been renamed --
# `cmake --build` no more prunes the build tree than `cmake --install` prunes
# the prefix.
for candidate in "$BUILD_DIR"/q3*; do
    if [ -L "$candidate" ]; then
        echo "==   $(basename "$candidate") -> $(readlink "$candidate")"
    elif [ -f "$candidate" ] && [ -x "$candidate" ]; then
        echo "==   $(basename "$candidate")"
    fi
done

# Which libsci the wrapper actually linked, rather than which one it was asked
# for. A serial libsci builds and passes the whole test suite; the only symptom
# is that OMP_NUM_THREADS stops buying anything on BLAS-bound work, which is
# exactly the kind of thing that goes unnoticed for a release or two
# (camsai/q3#14). Reported for every variant, since the flag spelling that
# triggered it is per programming environment and untried ones may spell it
# differently again.
if command -v ldd > /dev/null 2>&1 && [ -x "$BUILD_DIR/q3" ]; then
    LIBSCI_LINE=$(ldd "$BUILD_DIR/q3" 2>/dev/null | grep -o 'libsci_[a-z0-9_]*\.so[^ ]*' | head -1)
    if ldd "$BUILD_DIR/q3" 2>/dev/null | grep -q 'libsci_[a-z0-9_]*_mp\.so'; then
        echo "== libsci      : $LIBSCI_LINE (threaded)"
    elif [ -n "$LIBSCI_LINE" ]; then
        echo "== libsci      : $LIBSCI_LINE (SERIAL)"
        echo "==   BLAS will not thread in this build, so OMP_NUM_THREADS buys only the"
        echo "==   FFTs. The threaded copy, ${LIBSCI_LINE%%.so*}_mp.so, is in the same"
        echo "==   directory; the wrapper links it when it sees an OpenMP flag it"
        echo "==   recognises on the link line. See the programming-environment note"
        echo "==   in this script."
    fi
fi
echo "== next: scripts/hpc/nersc_perlmutter/validate.sh $VARIANT"
