cmake_minimum_required(VERSION 3.16)
set(CMAKE_POLICY_VERSION_MINIMUM 3.5)
project(q3 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_C_STANDARD 11)
# Interprocedural optimization is requested globally, so it propagates into every
# FetchContent subproject; spglib and GoogleTest configure with CMP0069=NEW and
# hard-error on toolchains whose link-time-optimization probe fails (seen with HPC
# compiler wrappers, e.g. Cray cc/CC). Probe first, and build without it where the
# toolchain cannot do it. GPU targets opt out individually below: the probe covers
# only C and CXX, and CUDA device LTO additionally needs separable compilation,
# which Q3 does not use.
include(CheckIPOSupported)
check_ipo_supported(RESULT Q3_INTERPROCEDURAL_OPTIMIZATION_SUPPORTED LANGUAGES C CXX)
if(Q3_INTERPROCEDURAL_OPTIMIZATION_SUPPORTED)
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)
else()
    message(STATUS "Interprocedural optimization unsupported by this toolchain; building without it")
endif()

# Q3 uses <filesystem> in a dozen translation units. libstdc++ ships that header
# only from GCC 8, and links it without an explicit -lstdc++fs (which this project
# does not add) only from GCC 9. Older toolchains fail partway through the first
# compile with "filesystem: No such file or directory", naming a header rather than
# the real cause. Say it at configure time instead: HPC login nodes routinely
# default to a system compiler far older than the modules the machine offers.
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 9)
    message(FATAL_ERROR
        "Q3 requires GCC 9 or newer; found ${CMAKE_CXX_COMPILER_VERSION} at "
        "${CMAKE_CXX_COMPILER}.\n"
        "Load a newer compiler, then configure in a FRESH build directory -- "
        "the compiler is cached and cannot be changed in an existing one:\n"
        "  rm -rf <builddir>\n"
        "  CC=<c-compiler> CXX=<c++-compiler> cmake -S . -B <builddir>")
endif()

# ---------- Options ----------

option(Q3_BUILD_TESTS "Build unit tests (requires internet for GoogleTest)" ON)
option(Q3_COVERAGE "Enable code coverage instrumentation (use with Debug build)" OFF)
# Declared here (not at the MPI section) because Q3_WITH_MPI is consulted by the
# FFTW3-MPI and core-library sections earlier in the file.
option(Q3_WITH_MPI "Build with MPI support" OFF)
# The GPU backends, for the same reason. At most one may be ON: a build tree
# compiles one q3 binary (see the Main executable section).
option(Q3_WITH_GPU "Build q3 with the Metal + Accelerate GPU backend (Apple Silicon)" OFF)
option(Q3_WITH_CUDA "Build q3 with the CUDA GPU backend" OFF)
option(Q3_WITH_HIP "Build q3 with the HIP/ROCm GPU backend" OFF)
option(Q3_WITH_SYCL "Build q3 with the SYCL/oneAPI GPU backend" OFF)

# A bare `cmake -S . -B build` otherwise produces an unoptimized binary with no
# -O flag at all, which is a poor default for a code whose runtime is the point.
# Coverage builds want -O0 -g instead, so leave those alone.
get_property(Q3_GENERATOR_IS_MULTI_CONFIG GLOBAL PROPERTY GENERATOR_IS_MULTI_CONFIG)
if(NOT Q3_GENERATOR_IS_MULTI_CONFIG AND NOT CMAKE_BUILD_TYPE)
    if(Q3_COVERAGE)
        set(CMAKE_BUILD_TYPE Debug CACHE STRING "Build type" FORCE)
    else()
        set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
    endif()
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS Debug Release RelWithDebInfo MinSizeRel)
    message(STATUS "CMAKE_BUILD_TYPE not set -- defaulting to ${CMAKE_BUILD_TYPE}")
endif()

if(Q3_COVERAGE)
    add_compile_options(--coverage -O0 -g)
    add_link_options(--coverage)
endif()

# ---------- FFT backend ----------
# The serial FFTGrid has two implementations. FFTW3 (GPL-2.0-or-later) is the
# default and the measured baseline. PocketFFT (BSD-3-Clause, header-only, the
# library NumPy and SciPy use) builds a binary that links no GPL FFT library.
# With pocketfft selected, nothing below searches for or links FFTW at all --
# a configure that still needed it would defeat the point -- so the paths that
# have only an FFTW implementation are refused here rather than quietly
# relinking it: the MPI build (its stick-distributed grid plans with FFTW) and
# the GPU builds (their CPU fallbacks do).
set(Q3_FFT_BACKEND "fftw" CACHE STRING "Serial FFT library: fftw or pocketfft")
set_property(CACHE Q3_FFT_BACKEND PROPERTY STRINGS fftw pocketfft)
if(NOT Q3_FFT_BACKEND MATCHES "^(fftw|pocketfft)$")
    message(FATAL_ERROR "Q3_FFT_BACKEND must be fftw or pocketfft, not '${Q3_FFT_BACKEND}'")
endif()
if(Q3_FFT_BACKEND STREQUAL "pocketfft")
    foreach(fftw_only_option Q3_WITH_MPI Q3_WITH_GPU Q3_WITH_CUDA Q3_WITH_HIP Q3_WITH_SYCL)
        if(${fftw_only_option})
            message(FATAL_ERROR
                "Q3_FFT_BACKEND=pocketfft does not support ${fftw_only_option}=ON yet: "
                "that build still plans FFTs with FFTW. Use Q3_FFT_BACKEND=fftw for it.")
        endif()
    endforeach()
    message(STATUS "FFT backend: PocketFFT (FFTW is neither searched for nor linked)")
else()
    message(STATUS "FFT backend: FFTW3")
endif()

if(Q3_FFT_BACKEND STREQUAL "fftw")
# ---------- FFTW3 ----------
# Module-based HPC systems install FFTW outside the distribution paths and
# advertise it through a site-chosen environment variable. A plain find_library()
# reads none of them -- <Package>_ROOT is a find_package convention and does not
# apply here -- so collect whatever the site set and search those first. Some
# sites point the variable at the install prefix and others straight at the
# library directory, so try both, and lib64 as well as lib.
set(Q3_FFTW_LIBRARY_HINTS "")
set(Q3_FFTW_INCLUDE_HINTS "")
foreach(fftw_environment_variable FFTW_ROOT FFTW_DIR FFTW_HOME FFTWDIR FFTW3_ROOT)
    if(DEFINED ENV{${fftw_environment_variable}})
        set(fftw_prefix "$ENV{${fftw_environment_variable}}")
        list(APPEND Q3_FFTW_LIBRARY_HINTS
            "${fftw_prefix}" "${fftw_prefix}/lib" "${fftw_prefix}/lib64")
        list(APPEND Q3_FFTW_INCLUDE_HINTS
            "${fftw_prefix}/include" "${fftw_prefix}/../include")
    endif()
endforeach()
if(DEFINED ENV{FFTW_INC})
    list(APPEND Q3_FFTW_INCLUDE_HINTS "$ENV{FFTW_INC}")
endif()
list(APPEND Q3_FFTW_LIBRARY_HINTS
    /opt/homebrew/lib /usr/local/lib /usr/lib
    /usr/lib/x86_64-linux-gnu /usr/lib/aarch64-linux-gnu)
list(APPEND Q3_FFTW_INCLUDE_HINTS
    /opt/homebrew/include /usr/local/include /usr/include)

find_library(FFTW3_LIB fftw3 HINTS ${Q3_FFTW_LIBRARY_HINTS})
find_path(FFTW3_INCLUDE fftw3.h HINTS ${Q3_FFTW_INCLUDE_HINTS})

if(NOT FFTW3_LIB OR NOT FFTW3_INCLUDE)
    message(FATAL_ERROR
        "FFTW3 not found (library: ${FFTW3_LIB}, headers: ${FFTW3_INCLUDE}).\n"
        "  macOS:  brew install fftw\n"
        "  Ubuntu: sudo apt-get install libfftw3-dev\n"
        "On a module-based system, load the FFTW module -- FFTW_ROOT, FFTW_DIR and "
        "FFTW_INC are picked up automatically -- or name the paths directly:\n"
        "  -DFFTW3_LIB=<library file> -DFFTW3_INCLUDE=<include directory>\n"
        "Searched: ${Q3_FFTW_LIBRARY_HINTS}")
endif()

# ---------- FFTW3 Threading ----------
option(Q3_WITH_FFTW_THREADS "Enable threaded FFTW transforms" ON)
set(Q3_FFTW_THREADS_AVAILABLE OFF)
if(Q3_WITH_FFTW_THREADS)
    find_library(FFTW3_THREADS_LIB fftw3_threads HINTS ${Q3_FFTW_LIBRARY_HINTS})
    if(FFTW3_THREADS_LIB)
        message(STATUS "Found FFTW3 threads: ${FFTW3_THREADS_LIB}")
        set(Q3_FFTW_THREADS_AVAILABLE ON)
    else()
        # A normal variable of the same name would shadow the cache entry, leaving
        # `cmake -L` reporting ON while threading is actually off.
        message(WARNING
            "Q3_WITH_FFTW_THREADS=ON but libfftw3_threads was not found -- FFTs will "
            "be single-threaded and OMP_NUM_THREADS will not speed them up. Pass "
            "-DFFTW3_THREADS_LIB=<library file>, or -DQ3_WITH_FFTW_THREADS=OFF.")
        set(Q3_FFTW_THREADS_AVAILABLE OFF)
    endif()
endif()

# ---------- FFTW3 MPI ----------
# Optional, even under Q3_WITH_MPI. Distributed FFTs are needed only for
# band-group parallelism; with one rank per band group create_fft_grid()
# already returns a SerialFFTGrid. When the library is absent, define nothing
# and let the guards in fft_grid_mpi.hpp and the MPI execution policy compile
# it out, so q3_mpi still builds and still parallelizes over k-points.
if(Q3_WITH_MPI)
    find_library(FFTW3_MPI_LIB fftw3_mpi HINTS ${Q3_FFTW_LIBRARY_HINTS})
    if(FFTW3_MPI_LIB)
        message(STATUS "Found FFTW3 MPI: ${FFTW3_MPI_LIB}")
    else()
        message(STATUS
            "FFTW3-MPI not found -- building q3_mpi with serial FFTs. k-point pool "
            "parallelism is unaffected; band groups (more than one rank per group) "
            "are rejected at run time. To enable them, point at an MPI-enabled FFTW "
            "with -DFFTW3_MPI_LIB=<library file>.")
    endif()
endif()
endif() # Q3_FFT_BACKEND STREQUAL "fftw"



# ---------- LAPACK ----------
find_package(LAPACK REQUIRED)



# ---------- LibXC: a tested system package, otherwise the pinned source build ----------
# A system LibXC is used only from a series Q3 is tested against: 5.2, which Ubuntu 24.04
# ships and every CI job and container image builds with, and 7.0, the series FetchContent
# builds below. Any other version falls back to the source build. LibXC 7.1.0, 7.1.1 and
# 7.1.2 return NaN for the gradient derivatives of spin-polarized PBE correlation wherever
# the total density gradient vanishes while the spin gradients do not -- between the
# opposite moments of an antiferromagnet -- and Homebrew's 7.1.2, picked up silently, sent
# the chromium example to NaN at its first SCF step (KNOWN-ISSUES.md). The evaluation
# refuses a non-finite result either way; Q3_ALLOW_UNTESTED_SYSTEM_LIBXC takes any system
# LibXC.
set(Q3_TESTED_LIBXC_SERIES 5.2 7.0)
list(JOIN Q3_TESTED_LIBXC_SERIES ", " Q3_TESTED_LIBXC_SERIES_TEXT)
option(Q3_ALLOW_UNTESTED_SYSTEM_LIBXC
       "Use a system LibXC outside the tested series (${Q3_TESTED_LIBXC_SERIES_TEXT})" OFF)
find_package(Libxc QUIET)
string(REGEX MATCH "^[0-9]+\\.[0-9]+" Q3_SYSTEM_LIBXC_SERIES "${Libxc_VERSION}")
if(Libxc_FOUND AND Q3_SYSTEM_LIBXC_SERIES IN_LIST Q3_TESTED_LIBXC_SERIES)
    message(STATUS "Using system LibXC ${Libxc_VERSION}: ${Libxc_DIR}")
    set(Q3_LIBXC_TARGET Libxc::xc)
elseif(Libxc_FOUND AND Q3_ALLOW_UNTESTED_SYSTEM_LIBXC)
    message(STATUS "Using untested system LibXC ${Libxc_VERSION} "
                   "(Q3_ALLOW_UNTESTED_SYSTEM_LIBXC): ${Libxc_DIR}")
    set(Q3_LIBXC_TARGET Libxc::xc)
else()
    if(Libxc_FOUND)
        message(STATUS "System LibXC ${Libxc_VERSION} at ${Libxc_DIR} is not from a tested "
                       "series (${Q3_TESTED_LIBXC_SERIES_TEXT}) — building LibXC 7.0.0 from source "
                       "via FetchContent (set Q3_ALLOW_UNTESTED_SYSTEM_LIBXC=ON to use it anyway)")
    else()
        message(STATUS "System LibXC not found — building from source via FetchContent")
    endif()
    set(Q3_LIBXC_FETCHCONTENT TRUE)
endif()

# ---------- External libraries (FetchContent) ----------
include(FetchContent)
FetchContent_Declare(
    nlohmann_json
    URL https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz
)
FetchContent_Declare(
    yaml-cpp
    URL https://github.com/jbeder/yaml-cpp/archive/refs/tags/0.8.0.tar.gz
)
FetchContent_Declare(
    spdlog
    URL https://github.com/gabime/spdlog/archive/refs/tags/v1.15.3.tar.gz
)
FetchContent_Declare(
    spglib
    GIT_REPOSITORY https://github.com/spglib/spglib.git
    GIT_TAG        v2.7.0
)

# Library build options
set(YAML_CPP_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(YAML_CPP_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
set(YAML_CPP_BUILD_CONTRIB OFF CACHE BOOL "" FORCE)
set(SPDLOG_FMT_EXTERNAL OFF CACHE BOOL "" FORCE)
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(ENABLE_FORTRAN OFF CACHE BOOL "" FORCE)
set(SPGLIB_WITH_Fortran OFF CACHE BOOL "" FORCE)
set(SPGLIB_WITH_Python OFF CACHE BOOL "" FORCE)
set(SPGLIB_WITH_TESTS OFF CACHE BOOL "" FORCE)
set(DISABLE_TESTING ON CACHE BOOL "Disable LibXC tests" FORCE)
set(BUILD_TESTING OFF CACHE BOOL "Disable LibXC tests" FORCE)

if(Q3_LIBXC_FETCHCONTENT)
    FetchContent_Declare(
        Libxc
        GIT_REPOSITORY https://gitlab.com/libxc/libxc.git
        GIT_TAG        7.0.0
    )
    # LibXC is the one vendored dependency whose install() rules still fire when
    # it is built as a subproject: `cmake --install` would otherwise scatter
    # xc-info, libxc.a, the LibXC headers and its CMake/pkg-config files into the
    # prefix, shadowing a system LibXC in a shared one. Suppressing install rules
    # across this call drops them, while Q3's own rules -- registered later -- are
    # unaffected. The xc target still builds, because q3 links it. Setting
    # EXCLUDE_FROM_ALL on the directory afterwards does not work: the rules are
    # registered during the call. The other vendored dependencies already gate
    # their install rules on being the top-level project.
    set(CMAKE_SKIP_INSTALL_RULES ON)
    FetchContent_MakeAvailable(nlohmann_json yaml-cpp spdlog Libxc spglib)
    set(CMAKE_SKIP_INSTALL_RULES OFF)
    # CMAKE_SKIP_INSTALL_RULES stops each subproject above from writing its own
    # cmake_install.cmake, but the top-level one -- generated afterwards, with
    # the variable restored -- still include()s every subdirectory
    # unconditionally. `cmake --install` then aborts on the first missing file
    # ("include could not find requested file"), which made installing
    # impossible on any machine without a system LibXC. Leave an empty script
    # where each suppressed one would have been, so the include succeeds and
    # still installs nothing.
    foreach(skipped_install_subproject nlohmann_json yaml-cpp spdlog libxc spglib)
        file(WRITE
            "${CMAKE_BINARY_DIR}/_deps/${skipped_install_subproject}-build/cmake_install.cmake"
            "# Install rules intentionally suppressed; see CMAKE_SKIP_INSTALL_RULES above.\n")
    endforeach()
    set(Q3_LIBXC_TARGET xc)
else()
    FetchContent_MakeAvailable(nlohmann_json yaml-cpp spdlog spglib)
endif()

# PocketFFT is one header and fetched in every configuration, not only when it
# is the backend: the FFTW build compiles the PocketFFT grid too, so a unit test
# can run the same transforms through both. Header-only, nothing links. Pinned
# to a commit of the upstream cpp branch, which carries no release tags.
FetchContent_Declare(
    pocketfft
    GIT_REPOSITORY https://github.com/mreineck/pocketfft.git
    GIT_TAG        c90e55b3d529f8efa40ed01a20de22405f45fc65
)
FetchContent_MakeAvailable(pocketfft)

# Restore BUILD_TESTING so Q3's own tests are still enabled
set(BUILD_TESTING ON CACHE BOOL "" FORCE)

# ---------- Build identity ----------
# Stamped into every results.json so a result can name the source that produced
# it. Regenerated on EVERY build rather than captured at configure time: a
# configure-time value is fixed when `cmake` last ran, so every commit made
# afterwards would be reported under its predecessor's SHA, and a provenance
# record that confidently names the wrong source is worse than one that admits
# it does not know. The script also reports nothing for a dirty worktree, which
# is not the commit it sits on. Release and container builds with no .git pass
# -DQ3_GIT_COMMIT=<sha>; empty is the honest default everywhere else.
set(Q3_GIT_COMMIT "" CACHE STRING "Git commit SHA override for builds with no .git")
set(Q3_GENERATED_INCLUDE_DIR "${CMAKE_BINARY_DIR}/generated_identity")
add_custom_target(q3_build_identity ALL
    COMMAND ${CMAKE_COMMAND}
        -DQ3_SOURCE_DIR=${CMAKE_SOURCE_DIR}
        -DQ3_OUTPUT_FILE=${Q3_GENERATED_INCLUDE_DIR}/q3_git_commit.h
        -DQ3_GIT_COMMIT_OVERRIDE=${Q3_GIT_COMMIT}
        -P ${CMAKE_SOURCE_DIR}/cmake/write_git_commit_header.cmake
    BYPRODUCTS ${Q3_GENERATED_INCLUDE_DIR}/q3_git_commit.h
    COMMENT "Recording build identity for results provenance")
# Present before the first build so the header always exists to be included.
execute_process(COMMAND ${CMAKE_COMMAND}
    -DQ3_SOURCE_DIR=${CMAKE_SOURCE_DIR}
    -DQ3_OUTPUT_FILE=${Q3_GENERATED_INCLUDE_DIR}/q3_git_commit.h
    -DQ3_GIT_COMMIT_OVERRIDE=${Q3_GIT_COMMIT}
    -P ${CMAKE_SOURCE_DIR}/cmake/write_git_commit_header.cmake)

# ---------- Core library (shared between q3 executable and tests) ----------
add_library(q3_cpp INTERFACE)
target_include_directories(q3_cpp INTERFACE src ${Q3_GENERATED_INCLUDE_DIR})

# FFTW's include directory is whatever prefix happens to hold fftw3.h, and on a
# package-manager install that is the manager's entire include tree
# (/opt/homebrew/include, /usr/local/include). Added as a plain -I it precedes
# every FetchContent dependency's own -I, so any locally installed copy of one
# of those dependencies wins over the version this project pinned. Homebrew's
# spdlog shadowing the fetched one is the case that bites: it is configured
# with SPDLOG_FMT_EXTERNAL, pulls in Homebrew's fmt, and nothing links either
# library, so the build dies at link on undefined fmt symbols.
#
# SYSTEM makes it -isystem, which the compiler searches after every -I. FFTW's
# header is still found; it just cannot outrank the vendored dependencies. It
# also silences warnings from headers this project does not own.
if(Q3_FFT_BACKEND STREQUAL "fftw")
    target_include_directories(q3_cpp SYSTEM INTERFACE ${FFTW3_INCLUDE})
endif()
# PocketFFT keeps a small per-process cache of twiddle tables keyed by length;
# without it every 3D transform recomputes them for each of its three axes.
target_include_directories(q3_cpp SYSTEM INTERFACE ${pocketfft_SOURCE_DIR})
target_compile_definitions(q3_cpp INTERFACE POCKETFFT_CACHE_SIZE=16)
if(Q3_FFT_BACKEND STREQUAL "pocketfft")
    target_compile_definitions(q3_cpp INTERFACE Q3_FFT_BACKEND_POCKETFFT)
endif()

# Say so when that prefix really does carry a competing copy of a dependency
# this project vendors. The mitigation above is silent when it works, and a
# link failure in fmt or spdlog is a long way from its cause -- naming the
# shadow here turns an afternoon of bisecting into one line of configure log.
foreach(vendored_header spdlog/spdlog.h fmt/format.h nlohmann/json.hpp yaml-cpp/yaml.h)
    if(FFTW3_INCLUDE AND EXISTS "${FFTW3_INCLUDE}/${vendored_header}")
        get_filename_component(shadowed_package "${vendored_header}" DIRECTORY)
        list(APPEND Q3_SHADOWED_PACKAGES "${shadowed_package}")
    endif()
endforeach()
if(Q3_SHADOWED_PACKAGES)
    list(JOIN Q3_SHADOWED_PACKAGES ", " Q3_SHADOWED_PACKAGES_TEXT)
    message(STATUS
        "FFTW headers live in ${FFTW3_INCLUDE}, which also provides: "
        "${Q3_SHADOWED_PACKAGES_TEXT}. Q3 builds against its own pinned copies; "
        "that directory is included as SYSTEM so it cannot outrank them.")
endif()
if(Q3_LIBXC_FETCHCONTENT)
    target_include_directories(q3_cpp INTERFACE
        ${libxc_SOURCE_DIR}/src ${libxc_BINARY_DIR}/src
        ${libxc_BINARY_DIR})
endif()
if(Q3_FFT_BACKEND STREQUAL "fftw")
    target_link_libraries(q3_cpp INTERFACE ${FFTW3_LIB})
endif()
# The BLAS/LAPACK this build links, recorded in every result's provenance: a run's speed
# can hinge on it -- a projector-overlap GEMM measured at single-core speed on one
# cluster -- and nothing in a result said which library had been linked.
string(REPLACE ";" " " Q3_LINEAR_ALGEBRA_LIBRARIES "${LAPACK_LIBRARIES}")
target_compile_definitions(q3_cpp INTERFACE
    "Q3_LINEAR_ALGEBRA_LIBRARIES=\"${Q3_LINEAR_ALGEBRA_LIBRARIES}\"")
target_link_libraries(q3_cpp INTERFACE ${LAPACK_LIBRARIES}
    nlohmann_json::nlohmann_json yaml-cpp::yaml-cpp spdlog::spdlog_header_only
    ${Q3_LIBXC_TARGET} Spglib::symspg)
if(Q3_WITH_MPI AND FFTW3_MPI_LIB)
    target_link_libraries(q3_cpp INTERFACE ${FFTW3_MPI_LIB})
    target_compile_definitions(q3_cpp INTERFACE Q3_WITH_FFTW_MPI)
endif()
if(Q3_FFTW_THREADS_AVAILABLE)
    target_link_libraries(q3_cpp INTERFACE ${FFTW3_THREADS_LIB})
    target_compile_definitions(q3_cpp INTERFACE Q3_USE_FFTW_THREADS)
endif()
# -Wno-nan-infinity-disabled exists only in Clang 18 and later. Compilers accept
# an unknown -Wno-<name> silently but then warn about it as soon as any other
# diagnostic fires, so probe the positive spelling, which they reject outright.
include(CheckCXXCompilerFlag)
check_cxx_compiler_flag(-Wnan-infinity-disabled Q3_HAS_NAN_INFINITY_DISABLED_WARNING)

# Scoped to CXX so none of these reach nvcc, which does not forward unknown host
# flags to the host compiler.
target_compile_options(q3_cpp INTERFACE
    $<$<COMPILE_LANGUAGE:CXX>:-ffp-contract=fast -Wall -Wextra -Wno-unused-parameter -Wno-deprecated-declarations>
    $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<BOOL:${Q3_HAS_NAN_INFINITY_DISABLED_WARNING}>>:-Wno-nan-infinity-disabled>)

# ---------- OpenMP ----------
option(Q3_WITH_OPENMP "Build with OpenMP support" ON)
set(Q3_OPENMP_ENABLED OFF)
if(Q3_WITH_OPENMP)
    find_package(OpenMP)
    if(OpenMP_FOUND)
        target_link_libraries(q3_cpp INTERFACE OpenMP::OpenMP_CXX)
        set(Q3_OPENMP_ENABLED ON)
    else()
        # Fallback for Apple Clang with Homebrew libomp
        find_library(OMP_LIB omp HINTS /opt/homebrew/opt/libomp/lib /opt/homebrew/lib)
        find_path(OMP_INCLUDE omp.h HINTS /opt/homebrew/opt/libomp/include /opt/homebrew/include)
        if(OMP_LIB AND OMP_INCLUDE)
            message(STATUS "Found OpenMP via fallback (Homebrew libomp)")
            target_compile_options(q3_cpp INTERFACE -Xpreprocessor -fopenmp)
            target_include_directories(q3_cpp SYSTEM INTERFACE ${OMP_INCLUDE})
            target_link_libraries(q3_cpp INTERFACE ${OMP_LIB})
            set(Q3_OPENMP_ENABLED ON)
        else()
            message(WARNING
                "Q3_WITH_OPENMP=ON but no OpenMP runtime was found -- building "
                "without OpenMP. macOS: brew install libomp. Pass "
                "-DQ3_WITH_OPENMP=OFF to stop asking.")
        endif()
    endif()
endif()

# The hot loops carry `#pragma omp` directives unconditionally, which is what
# keeps them readable and greppable. A compiler without OpenMP enabled ignores
# them -- correctly, that is the whole point -- but -Wall then reports each one
# under -Wunknown-pragmas, so a build with OpenMP off would drown in warnings
# about directives it is meant to drop. Silence that one warning, and only when
# there is no OpenMP to honour the directives.
if(NOT Q3_OPENMP_ENABLED)
    target_compile_options(q3_cpp INTERFACE
        $<$<COMPILE_LANGUAGE:CXX>:-Wno-unknown-pragmas>)
endif()

# ---------- MPI ----------
if(Q3_WITH_MPI)
    find_package(MPI REQUIRED)
    target_link_libraries(q3_cpp INTERFACE MPI::MPI_CXX)
    target_compile_definitions(q3_cpp INTERFACE Q3_WITH_MPI)
    message(STATUS "MPI found: ${MPI_CXX_COMPILER}")

    # ScaLAPACK search. ON keeps the long-standing behaviour: use it whenever it
    # can be found. OFF exists because finding it is not always a win: measured on
    # copper 8x8x8 at four ranks, the distributed subspace solve took 1,766 ms
    # against 585 ms for the serial one, since redistributing a bands-by-bands
    # matrix onto a BLACS grid costs more than solving it at the sizes Q3 runs.
    # Cluster toolchains that bundle ScaLAPACK (Cray LibSci among them) would
    # otherwise switch it on silently.
    option(Q3_WITH_SCALAPACK "Use ScaLAPACK for the subspace eigenproblem when it can be found" ON)
    set(SCALAPACK_FOUND FALSE)
    if(Q3_WITH_SCALAPACK)
        find_package(PkgConfig QUIET)
        if(PkgConfig_FOUND)
            pkg_check_modules(SCALAPACK scalapack)
        endif()

        if(NOT SCALAPACK_FOUND)
            find_library(SCALAPACK_LIBRARIES NAMES scalapack HINTS /opt/homebrew/lib /usr/local/lib)
            if(SCALAPACK_LIBRARIES)
                set(SCALAPACK_FOUND TRUE)
            endif()
        endif()
    else()
        message(STATUS "ScaLAPACK disabled (Q3_WITH_SCALAPACK=OFF): subspace diagonalization uses the serial solve on every rank")
    endif()

    if(SCALAPACK_FOUND)
        message(STATUS "ScaLAPACK found: ${SCALAPACK_LIBRARIES}")
        target_link_libraries(q3_cpp INTERFACE ${SCALAPACK_LIBRARIES})
        target_compile_definitions(q3_cpp INTERFACE Q3_WITH_SCALAPACK)
    elseif(Q3_WITH_SCALAPACK)
        message(WARNING "ScaLAPACK not found — parallel subspace diagonalization will use redundant serial fallback")
    endif()
endif()

set(Q3_GPU_KERNEL_SOURCES
    ${CMAKE_SOURCE_DIR}/src/layer_4_computation/diagonalizer/planewave/internal/hamiltonian_applicator/gpu_kernels_serial.cpp)

# ---------- Schema-to-C++ code generation ----------
# An in-tree .venv is the convenient default, but only as a hint: an explicit
# -DPython3_EXECUTABLE used to be overridden by it silently, which is a hard
# thing to see when the wrong interpreter is the whole problem.
if(NOT Python3_EXECUTABLE AND EXISTS "${CMAKE_SOURCE_DIR}/.venv/bin/python3")
    set(Python3_EXECUTABLE "${CMAKE_SOURCE_DIR}/.venv/bin/python3")
endif()
find_package(Python3 REQUIRED COMPONENTS Interpreter)
set(Q3_PYTHON "${Python3_EXECUTABLE}")
message(STATUS "Python for schema code generation: ${Q3_PYTHON}")

# The generator needs jinja2, jsonref and pyyaml, and reads ESSE's schemas from
# the mat3ra-esse package (written to the untracked schemas/esse by
# scripts/repo/schemas.py). They are build inputs, not runtime
# dependencies of the q3 package, so a fresh clone rarely has them and the
# failure arrives mid-build as a bare ModuleNotFoundError. Probe here, and say
# what to install. src/generated is committed and a pre-commit hook keeps it
# fresh, so a build without the generator is correct as long as the schemas
# have not changed. Pass -DQ3_REQUIRE_SCHEMA_CODEGEN=ON (as CI does) to refuse
# instead.
option(Q3_REQUIRE_SCHEMA_CODEGEN
       "Fail configuration when the schema generator's Python modules are missing" OFF)
set(Q3_SCHEMA_CODEGEN_INSTALL_HINT
    "${Q3_PYTHON} -m pip install \"${CMAKE_SOURCE_DIR}[build]\"")
execute_process(COMMAND "${Q3_PYTHON}" -c "import jinja2, jsonref, yaml, mat3ra.esse"
                RESULT_VARIABLE Q3_SCHEMA_CODEGEN_PROBE_RESULT
                OUTPUT_QUIET ERROR_QUIET)
if(Q3_SCHEMA_CODEGEN_PROBE_RESULT EQUAL 0)
    # Written here, at configure time, so the glob below sees ESSE's schemas.
    execute_process(COMMAND "${Q3_PYTHON}" ${CMAKE_SOURCE_DIR}/scripts/repo/schemas.py
                    RESULT_VARIABLE Q3_ESSE_SCHEMAS_RESULT)
endif()
if(Q3_SCHEMA_CODEGEN_PROBE_RESULT EQUAL 0 AND Q3_ESSE_SCHEMAS_RESULT EQUAL 0)
    set(Q3_SCHEMA_CODEGEN_ENABLED ON)
elseif(Q3_REQUIRE_SCHEMA_CODEGEN)
    message(FATAL_ERROR
        "Schema code generation is unavailable and Q3_REQUIRE_SCHEMA_CODEGEN=ON.\n"
        "  Python modules (jinja2, jsonref, pyyaml, mat3ra-esse): install with\n"
        "    ${Q3_SCHEMA_CODEGEN_INSTALL_HINT}")
else()
    set(Q3_SCHEMA_CODEGEN_ENABLED OFF)
    message(WARNING
        "Schema code generation skipped -- the committed src/generated headers are "
        "used as-is. This is fine unless you are changing schemas/. To enable it:\n"
        "  ${Q3_SCHEMA_CODEGEN_INSTALL_HINT}")
endif()

file(GLOB_RECURSE Q3_SCHEMA_FILES
    ${CMAKE_SOURCE_DIR}/schemas/q3/*.json
    ${CMAKE_SOURCE_DIR}/schemas/esse/*.json)
# The target exists either way so the add_dependencies() calls below need no
# condition of their own.
if(Q3_SCHEMA_CODEGEN_ENABLED)
    add_custom_target(generate_schemas
        COMMAND ${Q3_PYTHON}
            ${CMAKE_SOURCE_DIR}/scripts/repo/generate_cpp_from_schemas.py
            --manifest ${CMAKE_SOURCE_DIR}/schemas/manifest.yaml
            --output-dir ${CMAKE_SOURCE_DIR}/src/generated
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        DEPENDS
            ${CMAKE_SOURCE_DIR}/schemas/manifest.yaml
            ${CMAKE_SOURCE_DIR}/scripts/repo/generate_cpp_from_schemas.py
            ${CMAKE_SOURCE_DIR}/scripts/repo/templates/struct.hpp.j2
        COMMENT "Generating C++ headers from JSON schemas"
    )
else()
    add_custom_target(generate_schemas
        COMMENT "Schema code generation skipped; using the committed src/generated headers")
endif()

# Installed binaries otherwise carry no RPATH at all, so they only run while the
# same modules are loaded (or with LD_LIBRARY_PATH set by hand). This embeds the
# directories of the libraries actually linked against -- FFTW, LAPACK, MPI.
# Libraries injected implicitly by a compiler wrapper are unknown to CMake and
# still are not covered; on Cray systems export CRAY_ADD_RPATH=yes when
# configuring, or keep the modules loaded at run time.
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH ON)

# ---------- Main executable ----------
# One build tree compiles one engine binary, and it is always `q3` (#316).
# Which entry point that binary is follows from the Q3_WITH_* options, and the
# variant's own name -- q3_mpi, q3_gpu_cuda, q3_mpi_gpu_cuda, ... -- is a
# symlink to it, in the build tree and in the install tree alike. So an image
# or module built for MPI or for a GPU is invoked exactly like the plain one,
# and a downstream workflow names `q3` without knowing which build it landed
# on, while every script that already says `q3_mpi` keeps working.
#
# This replaces a cumulative build in which every variant tree also compiled
# the serial q3 beside the variant binary. A tree that needs both -- the CPU/GPU
# parity check, the MPI checkpoint test -- now configures two trees, which is
# what separating the cases at the directory level means.
#
# The variant binaries run the self-consistent field flavor only (see
# src/main_mpi.cpp and src/io/gpu_entry_point.hpp), and refuse any other input
# with an error naming the serial build. That restriction is the runtime's,
# not the packaging's, and is lifted there.
set(Q3_GPU_BACKENDS_REQUESTED "")
foreach(gpu_backend GPU CUDA HIP SYCL)
    if(Q3_WITH_${gpu_backend})
        list(APPEND Q3_GPU_BACKENDS_REQUESTED Q3_WITH_${gpu_backend})
    endif()
endforeach()
list(LENGTH Q3_GPU_BACKENDS_REQUESTED Q3_GPU_BACKEND_COUNT)
string(REPLACE ";" " and " Q3_GPU_BACKENDS_REQUESTED_TEXT "${Q3_GPU_BACKENDS_REQUESTED}")
if(Q3_GPU_BACKEND_COUNT GREATER 1)
    message(FATAL_ERROR
        "${Q3_GPU_BACKENDS_REQUESTED_TEXT} are ON together, but a build tree compiles one q3 binary. "
        "Configure one tree per GPU backend.")
endif()
if(Q3_WITH_MPI AND (Q3_WITH_GPU OR Q3_WITH_HIP OR Q3_WITH_SYCL))
    message(FATAL_ERROR
        "Q3_WITH_MPI has no combined entry point with ${Q3_GPU_BACKENDS_REQUESTED_TEXT}; only CUDA has "
        "one (q3_mpi_gpu_cuda). Configure the MPI and the GPU build as separate trees.")
endif()

# The name the variant is reached by besides `q3`, and the source of its main().
# Empty for the serial build, which has no other name.
if(Q3_WITH_CUDA AND Q3_WITH_MPI)
    set(Q3_VARIANT_NAME q3_mpi_gpu_cuda)
elseif(Q3_WITH_CUDA)
    set(Q3_VARIANT_NAME q3_gpu_cuda)
elseif(Q3_WITH_HIP)
    set(Q3_VARIANT_NAME q3_gpu_hip)
elseif(Q3_WITH_SYCL)
    set(Q3_VARIANT_NAME q3_gpu_sycl)
elseif(Q3_WITH_GPU)
    set(Q3_VARIANT_NAME q3_gpu)
elseif(Q3_WITH_MPI)
    set(Q3_VARIANT_NAME q3_mpi)
else()
    set(Q3_VARIANT_NAME "")
endif()
if(Q3_VARIANT_NAME)
    string(REGEX REPLACE "^q3_" "src/main_" Q3_MAIN_SOURCE "${Q3_VARIANT_NAME}.cpp")
else()
    set(Q3_MAIN_SOURCE src/main.cpp)
endif()

if(Q3_WITH_GPU)
    # Metal + Accelerate, for Apple Silicon.
    enable_language(OBJCXX)
    set(Q3_BACKEND_SOURCES
        src/layer_0_core/execution/gpu/device/metal.mm
        src/layer_0_core/math/linear_algebra/gpu/blas/metal.mm
        src/layer_0_core/math/linear_algebra/gpu/lapack/metal.mm
        src/layer_3_representation/basis/planewave/internal/fft/gpu/metal.mm
        ${Q3_GPU_KERNEL_SOURCES})
    # Enable ARC for Objective-C++ files
    set_source_files_properties(
        src/layer_0_core/execution/gpu/device/metal.mm
        src/layer_0_core/math/linear_algebra/gpu/blas/metal.mm
        src/layer_0_core/math/linear_algebra/gpu/lapack/metal.mm
        src/layer_3_representation/basis/planewave/internal/fft/gpu/metal.mm
        PROPERTIES COMPILE_FLAGS "-fobjc-arc")
    set(Q3_BACKEND_LIBRARIES
        "-framework Accelerate"
        "-framework Metal"
        "-framework MetalPerformanceShaders"
        "-framework Foundation")
elseif(Q3_WITH_CUDA)
    # Login nodes usually have no GPU at configure time, so `native` is not an
    # option; 80 is the A100 the collaborators build for. CUDAARCHS or an explicit
    # -DCMAKE_CUDA_ARCHITECTURES both win over this.
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES AND NOT DEFINED ENV{CUDAARCHS})
        set(CMAKE_CUDA_ARCHITECTURES 80 CACHE STRING "CUDA architectures (80 = A100, 90 = H100)")
        message(STATUS "CMAKE_CUDA_ARCHITECTURES not set -- defaulting to 80 (NVIDIA A100)")
    endif()
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    # With MPI as well this is q3_mpi_gpu_cuda: one CUDA device per MPI rank, so
    # a single calculation can span more than one GPU. q3_gpu_cuda cannot -- it
    # goes through run_gpu_entry_point(), which builds a serial execution
    # policy -- so `srun -n 8 q3_gpu_cuda` is eight duplicate runs rather than
    # one wider one.
    set(Q3_BACKEND_SOURCES
        src/layer_0_core/execution/gpu/device/cuda.cpp
        src/layer_0_core/math/linear_algebra/gpu/blas/cuda.cpp
        src/layer_0_core/math/linear_algebra/gpu/lapack/cuda.cpp
        src/layer_3_representation/basis/planewave/internal/fft/gpu/cuda.cpp
        src/layer_4_computation/diagonalizer/planewave/internal/hamiltonian_applicator/gpu_kernels_cuda.cu)
    set(Q3_BACKEND_LIBRARIES CUDA::cudart CUDA::cublas CUDA::cusolver CUDA::cufft)
elseif(Q3_WITH_HIP)
    # HIP is compiled with the standard C++ compiler + hip headers
    find_package(hip REQUIRED)
    find_package(hipblas REQUIRED)
    find_package(hipsolver REQUIRED)
    find_package(hipfft REQUIRED)
    set(Q3_BACKEND_SOURCES
        src/layer_0_core/execution/gpu/device/hip.cpp
        src/layer_0_core/math/linear_algebra/gpu/blas/hip.cpp
        src/layer_0_core/math/linear_algebra/gpu/lapack/hip.cpp
        src/layer_3_representation/basis/planewave/internal/fft/gpu/hip.cpp
        src/layer_4_computation/diagonalizer/planewave/internal/hamiltonian_applicator/gpu_kernels_hip.cpp)
    set(Q3_BACKEND_LIBRARIES hip::host roc::hipblas roc::hipsolver hip::hipfft)
elseif(Q3_WITH_SYCL)
    # SYCL compilation requires icpx (Intel oneAPI DPC++ compiler).
    # oneMKL provides BLAS, LAPACK, and DFT via MKL::MKL_SYCL.
    find_package(MKL REQUIRED)
    set(Q3_BACKEND_SOURCES
        src/layer_0_core/execution/gpu/device/sycl.cpp
        src/layer_0_core/math/linear_algebra/gpu/blas/sycl.cpp
        src/layer_0_core/math/linear_algebra/gpu/lapack/sycl.cpp
        src/layer_3_representation/basis/planewave/internal/fft/gpu/sycl.cpp
        src/layer_4_computation/diagonalizer/planewave/internal/hamiltonian_applicator/gpu_kernels_sycl.cpp)
    set(Q3_BACKEND_LIBRARIES MKL::MKL_SYCL)
else()
    set(Q3_BACKEND_SOURCES ${Q3_GPU_KERNEL_SOURCES})
    set(Q3_BACKEND_LIBRARIES "")
endif()

add_executable(q3 ${Q3_MAIN_SOURCE} ${Q3_BACKEND_SOURCES})
add_dependencies(q3 generate_schemas)
target_link_libraries(q3 PRIVATE q3_cpp ${Q3_BACKEND_LIBRARIES})
if(Q3_GPU_BACKEND_COUNT GREATER 0)
    # The IPO probe covers only C and CXX; CUDA under CMP0069=NEW would
    # hard-error here, and device LTO would additionally need separable
    # compilation that Q3 does not use.
    set_property(TARGET q3 PROPERTY INTERPROCEDURAL_OPTIMIZATION FALSE)
endif()
if(Q3_WITH_GPU)
    target_compile_options(q3 PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Wno-unused-parameter>)
elseif(Q3_WITH_SYCL)
    target_compile_options(q3 PRIVATE -fsycl -Wall -Wextra -Wno-unused-parameter)
    target_link_options(q3 PRIVATE -fsycl)
elseif(Q3_GPU_BACKEND_COUNT GREATER 0)
    target_compile_options(q3 PRIVATE -Wall -Wextra -Wno-unused-parameter)
endif()

# The variant name, as a target and as a symlink beside q3. The target is what
# keeps `cmake --build build --target q3_mpi` working; it runs on every build,
# rather than as a POST_BUILD step of q3, so a tree that already holds an
# up-to-date q3 -- or a stale q3_mpi binary from the cumulative layout -- still
# ends up with the link.
if(Q3_VARIANT_NAME)
    add_custom_target(${Q3_VARIANT_NAME} ALL
        COMMAND ${CMAKE_COMMAND} -E create_symlink q3 $<TARGET_FILE_DIR:q3>/${Q3_VARIANT_NAME}
        COMMENT "Linking ${Q3_VARIANT_NAME} -> q3"
        VERBATIM)
    add_dependencies(${Q3_VARIANT_NAME} q3)
    message(STATUS "Building q3 from ${Q3_MAIN_SOURCE}, also reachable as ${Q3_VARIANT_NAME}")
else()
    message(STATUS "Building q3 from ${Q3_MAIN_SOURCE} (serial)")
endif()

# q3_utility: the utility binary -- subcommands that consume a finished run's
# saved records rather than an input manifest. `interpolate` evaluates an
# interpolation record along a k-path and on a DOS mesh; `convert` re-exports a
# saved record (mean-field -> BerkeleyGW, interpolation -> Wannier90) or
# imports Wannier90 files into one. One binary with subcommands, so no
# science- or format-facing suffix ever sits beside the q3_mpi/q3_gpu*
# variant links; serial by construction, since evaluating a saved model is
# dense algebra on a few hundred orbitals. It is a separate tool rather than a
# variant of the engine, so every build tree has it.
add_executable(q3_utility src/main_utility.cpp ${Q3_GPU_KERNEL_SOURCES})
add_dependencies(q3_utility generate_schemas)
target_link_libraries(q3_utility PRIVATE q3_cpp)
target_compile_options(q3_utility PRIVATE -ffp-contract=fast -Wall -Wextra -Wno-unused-parameter -Wno-nan-infinity-disabled)

# ---------- Tests (GoogleTest) ----------
if(Q3_BUILD_TESTS)
    enable_testing()
    include(FetchContent)
    FetchContent_Declare(
        googletest
        URL https://github.com/google/googletest/archive/refs/tags/v1.15.2.tar.gz
    )
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)

    add_executable(q3_tests
        ${Q3_GPU_KERNEL_SOURCES}
        # layer_0_core
        tests/unit/layer_0_core/types/test_vector_3d.cpp
        tests/unit/layer_0_core/types/test_matrix_3x3.cpp
        tests/unit/layer_0_core/types/test_integer_vector_3d.cpp
        tests/unit/layer_0_core/types/test_constants.cpp
        tests/unit/layer_0_core/types/test_matrix.cpp
        tests/unit/layer_0_core/math/test_on_grid_function_mixin.cpp
        tests/unit/layer_0_core/math/integration/test_simpson.cpp
        tests/unit/layer_0_core/math/interpolation/test_uniform_q_grid.cpp
        tests/unit/layer_0_core/math/interpolation/test_uniform_q_grid_table.cpp
        tests/unit/layer_0_core/math/interpolation/test_lagrange.cpp
        tests/unit/layer_0_core/math/linear_algebra/test_blas.cpp
        tests/unit/layer_0_core/math/linear_algebra/test_lapack.cpp
        tests/unit/layer_0_core/math/linear_algebra/test_dense_matrix.cpp
        tests/unit/layer_0_core/math/linear_algebra/test_dense_matrix_product.cpp
        tests/unit/layer_0_core/math/linear_algebra/test_hermitian_eigensolver.cpp
        tests/unit/layer_0_core/math/linear_algebra/test_singular_value_decomposition.cpp
        tests/unit/layer_0_core/math/linear_algebra/test_unitary_exponential.cpp
        tests/unit/layer_0_core/math/test_convergence_window.cpp
        tests/unit/layer_0_core/math/spherical/test_harmonics.cpp
        tests/unit/layer_0_core/math/spherical/test_bessel.cpp
        tests/unit/layer_0_core/math/spherical/test_coefficients.cpp
        tests/unit/layer_0_core/math/differentiation/test_gradient.cpp
        tests/unit/layer_0_core/math/transform/test_spherical_bessel.cpp
        tests/unit/layer_0_core/math/test_phase_factor.cpp
        tests/unit/layer_0_core/math/test_tolerances.cpp
        tests/unit/layer_0_core/execution/test_threading.cpp
        tests/unit/layer_0_core/profiling/test_counters.cpp
        tests/unit/layer_0_core/profiling/test_memory.cpp
        tests/unit/layer_0_core/test_points_grid.cpp
        tests/unit/layer_0_core/test_version.cpp
        tests/unit/layer_4_computation/iterator/kohn_sham/planewave/internal/test_kpoint_distributor.cpp
        tests/unit/layer_4_computation/iterator/kohn_sham/planewave/execution/test_serial.cpp
        # layer_1_structure
        tests/unit/layer_1_structure/crystal/test_crystal.cpp
        tests/unit/layer_1_structure/crystal/test_periodic_table.cpp
        tests/unit/layer_1_structure/crystal/test_lattice.cpp
        tests/unit/layer_1_structure/reciprocal_space/test_kpoints_grid.cpp
        tests/unit/layer_1_structure/reciprocal_space/test_reciprocal_space_vector.cpp
        tests/unit/layer_1_structure/reciprocal_space/test_kpoint.cpp
        tests/unit/layer_1_structure/reciprocal_space/test_reciprocal_space_vectors_grid.cpp
        tests/unit/layer_1_structure/reciprocal_space/test_kpoint_neighbor_shells.cpp
        tests/unit/layer_1_structure/reciprocal_space/test_kpoint_stencil.cpp
        tests/unit/layer_1_structure/reciprocal_space/test_lattice_fourier_transform.cpp
        tests/unit/layer_1_structure/reciprocal_space/test_wigner_seitz_supercell.cpp
        tests/unit/layer_1_structure/symmetry/test_symmetry.cpp
        tests/unit/layer_1_structure/symmetry/test_operation.cpp
        # layer_2_quantum_system
        tests/unit/layer_2_quantum_system/core/test_occupation.cpp
        tests/unit/layer_2_quantum_system/core/test_quantum_state.cpp
        tests/unit/layer_2_quantum_system/core/test_spin_resolved_entity.cpp
        tests/unit/layer_2_quantum_system/core/wavefunction/test_reciprocal_space.cpp
        tests/unit/layer_2_quantum_system/derived/test_charge_density.cpp
        tests/unit/layer_2_quantum_system/derived/test_effective_potential.cpp
        tests/unit/layer_2_quantum_system/test_quantum_system.cpp
        # layer_3_representation
        tests/unit/layer_3_representation/basis/planewave/internal/fft/test_fft_grid.cpp
        tests/unit/layer_3_representation/basis/planewave/internal/fft/test_serial_pocket_fft_grid.cpp
        tests/unit/layer_3_representation/basis/planewave/internal/fft/test_grid_sizer.cpp
        tests/unit/layer_3_representation/basis/planewave/internal/fft/test_stick_partition.cpp
        tests/unit/layer_3_representation/basis/planewave/internal/test_g_vectors_grid.cpp
        tests/unit/layer_3_representation/basis/planewave/internal/test_g_vector.cpp
        tests/unit/layer_3_representation/basis/planewave/internal/test_spherical_harmonics_table.cpp
        tests/unit/layer_3_representation/basis/planewave/internal/test_miller_index_lookup.cpp
        tests/unit/layer_3_representation/basis/planewave/internal/fft/test_pair.cpp
        tests/unit/layer_3_representation/basis/planewave/internal/kpoint/test_plane_wave_selector.cpp
        tests/unit/layer_3_representation/basis/planewave/internal/kpoint/test_plane_wave_table.cpp
        tests/unit/layer_3_representation/mixins/test_ff_transformable.cpp
        tests/unit/layer_3_representation/pseudopotential/internal/test_projector.cpp
        tests/unit/layer_3_representation/pseudopotential/internal/test_wavefunction_atomic.cpp
        tests/unit/layer_3_representation/pseudopotential/test_set.cpp
        tests/unit/layer_3_representation/pseudopotential/paw/test_compensation_charge.cpp
        tests/unit/layer_3_representation/basis/planewave/pseudopotential/internal/test_beta_function_table.cpp
        tests/unit/layer_3_representation/basis/planewave/pseudopotential/test_reciprocal_projectors.cpp
        tests/unit/layer_4_computation/paw/one_center/test_radial_poisson.cpp
        tests/unit/layer_4_computation/paw/one_center/test_angular_grid.cpp
        tests/unit/layer_4_computation/paw/one_center/test_densities.cpp
        tests/unit/layer_4_computation/paw/one_center/test_energy_and_d_matrix.cpp
        tests/unit/layer_4_computation/paw/one_center/test_scf_terms.cpp
        tests/unit/layer_4_computation/paw/one_center/convention/test_registry.cpp
        tests/unit/layer_4_computation/paw/one_center/test_occupation_symmetrization.cpp
        tests/unit/layer_3_representation/basis/planewave/pseudopotential/internal/test_projector_mapping_table.cpp
        tests/unit/layer_3_representation/basis/planewave/pseudopotential/internal/test_core_charge_density_table.cpp
        tests/unit/layer_3_representation/basis/planewave/pseudopotential/internal/test_local_potential_table.cpp
        tests/unit/layer_3_representation/basis/planewave/pseudopotential/internal/test_core_charge_data.cpp
        tests/unit/layer_3_representation/basis/planewave/pseudopotential/test_factory.cpp
        tests/unit/layer_3_representation/basis/planewave/pseudopotential/ultrasoft/test_init.cpp
        tests/unit/layer_3_representation/basis/planewave/internal/transform/test_charge_density_transformer.cpp
        tests/unit/layer_3_representation/basis/planewave/internal/transform/test_wavefunction_transformer.cpp
        # layer_4_computation
        tests/unit/layer_4_computation/diagonalizer/planewave/test_factory.cpp
        tests/unit/layer_4_computation/diagonalizer/planewave/test_direct.cpp
        tests/unit/layer_4_computation/diagonalizer/planewave/internal/hamiltonian_applicator/test_gpu_fft_batch.cpp
        tests/unit/layer_4_computation/diagonalizer/planewave/internal/overlap/test_orbital_metric.cpp
        tests/unit/layer_4_computation/diagonalizer/planewave/subspace/test_subspace_diagonalizer.cpp
        tests/unit/layer_4_computation/effective_potential/internal/dispersion/test_d3bj.cpp
        tests/unit/layer_4_computation/effective_potential/internal/dispersion/test_factory.cpp
        tests/unit/layer_4_computation/effective_potential/internal/electrostatic/test_ewald_summation.cpp
        tests/unit/layer_4_computation/effective_potential/internal/exchange_correlation/test_factory.cpp
        tests/unit/layer_4_computation/effective_potential/internal/exchange_correlation/test_libxc.cpp
        tests/unit/layer_4_computation/effective_potential/internal/hubbard/test_hubbard.cpp
        tests/unit/layer_4_computation/effective_potential/internal/hubbard/test_hubbard_ortho_atomic.cpp
        tests/unit/layer_4_computation/effective_potential/internal/hubbard/test_hubbard_occupation_mixer.cpp
        tests/unit/layer_4_computation/effective_potential/internal/hubbard/test_hubbard_seeding.cpp
        tests/unit/layer_4_computation/effective_potential/planewave/test_result.cpp
        tests/unit/layer_4_computation/effective_potential/planewave/internal/exchange_correlation_potential/test_factory.cpp
        tests/unit/layer_4_computation/flavors/test_non_self_consistent_field.cpp
        tests/unit/layer_4_computation/flavors/test_self_consistent_field.cpp
        tests/unit/layer_4_computation/flavors/test_verification.cpp
        tests/unit/layer_4_computation/iterator/test_config.cpp
        tests/unit/layer_4_computation/iterator/test_energy_history.cpp
        tests/unit/layer_4_computation/iterator/test_init.cpp
        tests/unit/layer_4_computation/iterator/internal/mixing/test_factory.cpp
        tests/unit/layer_4_computation/iterator/internal/mixing/real_space/test_anderson.cpp
        tests/unit/layer_4_computation/iterator/internal/smearing/test_factory.cpp
        tests/unit/layer_4_computation/iterator/internal/smearing/test_fixed.cpp
        tests/unit/layer_4_computation/iterator/internal/smearing/test_gaussian.cpp
        tests/unit/layer_4_computation/iterator/internal/smearing/test_methfessel_paxton.cpp
        tests/unit/layer_4_computation/iterator/internal/smearing/test_fermi_dirac.cpp
        tests/unit/layer_4_computation/iterator/internal/smearing/test_marzari_vanderbilt.cpp
        tests/unit/layer_4_computation/iterator/internal/mixing/reciprocal_space/test_broyden.cpp
        tests/unit/layer_4_computation/iterator/internal/mixing/preconditioner/test_kerker.cpp
        tests/unit/layer_4_computation/iterator/internal/mixing/preconditioner/test_local_thomas_fermi.cpp
        tests/unit/layer_4_computation/iterator/internal/mixing/preconditioner/test_factory.cpp
        tests/unit/layer_4_computation/iterator/kohn_sham/planewave/internal/test_energy_calculator.cpp
        tests/unit/layer_4_computation/iterator/kohn_sham/planewave/internal/test_reporter.cpp
        tests/unit/layer_4_computation/iterator/kohn_sham/planewave/internal/test_memory_estimate.cpp
        tests/unit/layer_4_computation/iterator/kohn_sham/test_factory.cpp
        tests/unit/layer_4_computation/charge_density/planewave/test_init.cpp
        tests/unit/layer_4_computation/charge_density/planewave/internal/test_accumulator.cpp
        tests/unit/layer_4_computation/fermi_energy/planewave/test_fermi_energy.cpp
        tests/unit/layer_3_representation/wavefunction/planewave/norm_conserving/test_initializer.cpp
        # io
        tests/unit/io/test_launcher.cpp
        tests/unit/io/provenance/test_init.cpp
        tests/unit/io/provenance/test_factory.cpp
        tests/unit/io/provenance/planewave/test_init.cpp
        tests/unit/io/provenance/planewave/test_pseudopotential.cpp
        tests/unit/io/interpolation/wannier90/test_round_trip.cpp
        tests/unit/io/interpolation/wannier90/test_importer.cpp
        tests/unit/io/interpolation/test_checkpoint.cpp
        tests/unit/io/interpolation/test_factory.cpp
        tests/unit/io/interpolation/test_record_and_export.cpp
        tests/unit/io/interpolation/test_results_document.cpp
        tests/unit/io/test_json_file.cpp
        tests/unit/io/input/test_data.cpp
        tests/unit/io/input/test_system_builder.cpp
        tests/unit/io/input/test_utils.cpp
        tests/unit/io/input/parser/test_factory.cpp
        tests/unit/io/input/parser/test_input_format_registry.cpp
        tests/unit/io/input/parser/test_qe.cpp
        tests/unit/io/input/parser/test_yaml.cpp
        tests/unit/io/output/material/test_init.cpp
        tests/unit/io/output/test_companion_outputs.cpp
        tests/unit/io/output/test_results.cpp
        tests/unit/io/pseudopotential/test_upf_reader.cpp
        tests/unit/io/pseudopotential/test_paw_upf_reader.cpp
        tests/unit/io/pseudopotential/test_functional_label.cpp
        tests/unit/io/pseudopotential/test_loader.cpp
        tests/unit/io/pseudopotential/test_pseudopotential_format_registry.cpp
        tests/unit/io/mean_field_export/test_factory.cpp
        tests/unit/io/mean_field_export/berkeleygw/test_header.cpp
        tests/unit/io/mean_field_export/berkeleygw/test_cell_symmetry.cpp
        tests/unit/io/mean_field_export/berkeleygw/test_wavefunction.cpp
        tests/unit/io/mean_field_export/berkeleygw/test_charge_density.cpp
        tests/unit/io/mean_field_export/berkeleygw/test_exchange_correlation.cpp
        tests/unit/io/mean_field_export/berkeleygw/test_exchange_correlation_potential.cpp
        tests/unit/io/mean_field_export/berkeleygw/test_self_consistent_potential.cpp
        tests/unit/io/mean_field_export/berkeleygw/test_field.cpp
        tests/unit/io/mean_field_export/berkeleygw/test_kleinman_bylander_projectors.cpp
        tests/unit/io/mean_field_export/berkeleygw/test_init.cpp
        tests/unit/io/mean_field_record/test_checkpoint.cpp
        tests/unit/io/quantum_system/kohn_sham/planewave/checkpoint/test_q3.cpp
        tests/unit/io/checkpoint/test_packed_record_buffer.cpp
        tests/unit/io/wavefunction/kohn_sham/planewave/checkpoint/test_qe.cpp
        tests/unit/io/charge_density/kohn_sham/planewave/checkpoint/test_qe.cpp
        tests/unit/io/charge_density/kohn_sham/planewave/checkpoint/test_qe_reader.cpp
        tests/unit/io/wavefunction/kohn_sham/planewave/checkpoint/test_qe_reader.cpp
        tests/unit/io/volumetric/test_field.cpp
        tests/unit/io/volumetric/test_cube.cpp
        tests/unit/io/volumetric/test_cube_reader.cpp
        tests/unit/io/volumetric/test_xsf.cpp
        tests/unit/io/volumetric/test_factory.cpp
        tests/unit/io/volumetric/test_inverse_transform.cpp
        # utils
        tests/unit/utils/test_sha256.cpp
        tests/unit/utils/test_registry.cpp
        tests/unit/utils/test_time.cpp
        # integration
        tests/integration/test_scf_integration.cpp
        tests/integration/test_non_self_consistent_field_integration.cpp
        tests/integration/test_cli.cpp
        # layer_3 augmentation
        tests/unit/layer_3_representation/pseudopotential/planewave/ultrasoft/internal/test_charge_density_augmentation.cpp
        tests/unit/layer_3_representation/pseudopotential/planewave/internal/test_atomic_orbital_builder.cpp
        tests/unit/layer_3_representation/pseudopotential/planewave/internal/test_lowdin_orthogonalization.cpp
        # layer_5_properties/projections
        tests/unit/layer_5_properties/projections/planewave/test_init.cpp
        # layer_3_representation/local_orbital
        tests/unit/layer_3_representation/local_orbital/test_init.cpp
        tests/unit/layer_3_representation/local_orbital/test_hamiltonian_decay.cpp
        tests/unit/layer_3_representation/basis/planewave/local_orbital/test_hydrogenic_orbital_set.cpp
        tests/unit/layer_3_representation/basis/planewave/local_orbital/test_pseudo_atomic_orbital_set.cpp
        # layer_4_computation/interpolation (umbrella), its wannier method, diagonalizer/local_orbital
        tests/unit/layer_4_computation/interpolator/test_factory.cpp
        tests/unit/layer_4_computation/interpolator/target/quantum_state/test_local_orbital.cpp
        tests/unit/layer_4_computation/interpolator/method/wannier/test_init.cpp
        tests/unit/layer_4_computation/interpolator/method/wannier/test_construction_inputs.cpp
        tests/unit/layer_4_computation/interpolator/method/wannier/test_diagnostic.cpp
        tests/unit/layer_4_computation/interpolator/method/wannier/test_diagnostic_error.cpp
        tests/unit/layer_4_computation/interpolator/method/wannier/report/test_init.cpp
        tests/unit/layer_4_computation/interpolator/method/wannier/settings/test_init.cpp
        tests/unit/layer_4_computation/interpolator/method/wannier/subspace/test_factory.cpp
        tests/unit/layer_4_computation/interpolator/method/wannier/subspace/test_isolated_manifold_selection.cpp
        tests/unit/layer_4_computation/interpolator/method/wannier/subspace/test_projectability_disentanglement_selection.cpp
        tests/unit/layer_4_computation/interpolator/method/wannier/gauge/test_factory.cpp
        tests/unit/layer_4_computation/interpolator/method/wannier/gauge/test_projection_gauge_choice.cpp
        tests/unit/layer_4_computation/interpolator/method/wannier/gauge/test_maximally_localized_gauge_choice.cpp
        tests/unit/layer_4_computation/interpolator/method/wannier/test_inputs.cpp
        tests/unit/layer_4_computation/interpolator/method/wannier/internal/test_projection_gauge.cpp
        tests/unit/layer_4_computation/interpolator/method/wannier/internal/phase_branch/test_principal_phase_branch.cpp
        tests/unit/layer_4_computation/interpolator/method/wannier/internal/phase_branch/test_guided_phase_branch.cpp
        tests/unit/layer_4_computation/interpolator/method/wannier/planewave/test_init.cpp
        tests/unit/layer_4_computation/interpolator/method/wannier/planewave/test_init_augmentation.cpp
        tests/unit/layer_4_computation/interpolator/method/wannier/internal/test_band_classification.cpp
        tests/unit/layer_4_computation/interpolator/method/wannier/internal/test_band_classifier.cpp
        tests/unit/layer_4_computation/interpolator/method/wannier/internal/test_bloch_overlaps.cpp
        tests/unit/layer_4_computation/interpolator/method/wannier/internal/test_subspace_disentangler.cpp
        tests/unit/layer_4_computation/interpolator/method/wannier/internal/test_spread_functional.cpp
        tests/unit/layer_4_computation/interpolator/method/wannier/internal/test_spread_minimizer.cpp
        tests/unit/layer_4_computation/interpolator/method/wannier/internal/test_quality_assessor.cpp
        tests/unit/layer_4_computation/interpolator/method/wannier/planewave/internal/test_projections.cpp
        tests/unit/layer_4_computation/interpolator/method/wannier/planewave/internal/test_overlaps.cpp
        tests/unit/layer_4_computation/interpolator/method/wannier/planewave/internal/test_overlaps_augmentation.cpp
        # layer_5_properties: local-orbital band structure and DOS
        tests/unit/layer_5_properties/band_structure/test_band_edges.cpp
        tests/unit/layer_5_properties/band_structure/interpolated/test_init.cpp
        tests/unit/layer_5_properties/density_of_states/interpolated/test_init.cpp
        # layer_5_properties: Ewald force and stress
        tests/unit/layer_5_properties/forces/planewave/internal/test_ewald.cpp
        tests/unit/layer_5_properties/stress_tensor/planewave/internal/test_ewald.cpp
        # layer_5_properties/isolated_system_correction
        tests/unit/layer_5_properties/stress_tensor/planewave/internal/exchange_correlation_gradient/test_factory.cpp
        tests/unit/layer_5_properties/isolated_system_correction/planewave/test_factory.cpp
        tests/unit/layer_5_properties/isolated_system_correction/planewave/test_makov_payne.cpp
        tests/unit/layer_5_properties/isolated_system_correction/planewave/test_martyna_tuckerman.cpp
        tests/test_main.cpp
    )
    add_dependencies(q3_tests generate_schemas)
    target_link_libraries(q3_tests PRIVATE q3_cpp GTest::gtest)
    target_include_directories(q3_tests PRIVATE
        ${googletest_SOURCE_DIR}/googletest/include
        ${googletest_SOURCE_DIR}/googlemock/include
        ${CMAKE_SOURCE_DIR}/tests/unit)
    # Q3_BINARY_DIR names the directory the executables are actually written to.
    # The CLI smoke tests used to derive it from EXAMPLES_DIR as
    # <source>/build/q3, which is only correct for an in-tree build directory
    # called exactly "build" -- every out-of-tree build (an HPC one on a scratch
    # filesystem, say) failed those two tests while the binary was fine.
    target_compile_definitions(q3_tests PRIVATE
        EXAMPLES_DIR="${CMAKE_SOURCE_DIR}/examples"
        Q3_BINARY_DIR="${CMAKE_BINARY_DIR}")
    target_compile_options(q3_tests PRIVATE -Wall -Wextra -Wno-unused-parameter)

    include(GoogleTest)
    gtest_discover_tests(q3_tests WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/tests)

    # MPI unit tests — separate executable run via mpirun
    if(Q3_WITH_MPI)
        add_executable(q3_tests_mpi
            ${Q3_GPU_KERNEL_SOURCES}
            tests/unit/layer_0_core/execution/test_mpi_execution_policy.cpp
            tests/unit/layer_3_representation/basis/planewave/internal/fft/test_fft_grid_sticks.cpp
            tests/unit/layer_4_computation/diagonalizer/planewave/internal/subspace_manager/test_serial.cpp
            tests/test_main.cpp
        )
        target_link_libraries(q3_tests_mpi PRIVATE q3_cpp GTest::gtest MPI::MPI_CXX)
        target_include_directories(q3_tests_mpi PRIVATE
            ${googletest_SOURCE_DIR}/googletest/include
            ${CMAKE_SOURCE_DIR}/tests/unit)
        target_compile_definitions(q3_tests_mpi PRIVATE Q3_WITH_MPI)
        target_compile_options(q3_tests_mpi PRIVATE -Wall -Wextra -Wno-unused-parameter)
        message(STATUS "MPI tests target: q3_tests_mpi (run with mpirun -np N ./build/q3_tests_mpi)")
    endif()
endif()

# ---------- Build-time add-ons ----------
# Every src/addons/<name>/CMakeLists.txt present in the tree is built in, after
# the test targets exist so an add-on can add its tests to them. The core names
# no add-on; see cmake/Q3Addons.cmake.
include(${CMAKE_SOURCE_DIR}/cmake/Q3Addons.cmake)
q3_build_addons()

# ---------- Install ----------
# q3_utility ships wherever q3 does: the container image and any
# `cmake --install` tree would otherwise lack the converter that turns a
# saved mean-field record into BerkeleyGW (or future) formats. The variant
# name is installed as a relative symlink to q3, so the install tree can be
# copied or relocated -- a container's COPY of bin/ included -- and still
# resolve it.
install(TARGETS q3 q3_utility DESTINATION bin)
if(Q3_VARIANT_NAME)
    install(CODE "
        set(variant_link \"\$ENV{DESTDIR}\${CMAKE_INSTALL_PREFIX}/bin/${Q3_VARIANT_NAME}\")
        message(STATUS \"Installing: \${variant_link} -> q3\")
        file(REMOVE \"\${variant_link}\")
        file(CREATE_LINK q3 \"\${variant_link}\" SYMBOLIC)
    ")
endif()
message(STATUS "Install prefix: ${CMAKE_INSTALL_PREFIX}")

# ---------- Python extension module (optional — nanobind) ----------
#
# Build with:   cmake -DQ3_ENABLE_PYTHON=ON ..
#               cmake --build . --target q3_core
#
# The resulting q3_core.*.so is importable from the build directory:
#   PYTHONPATH=build python -c "import q3_core; print(q3_core.run_scf.__doc__)"
#
option(Q3_ENABLE_PYTHON "Build the q3_core nanobind Python extension module" OFF)

if(Q3_ENABLE_PYTHON)
    # Locate the Python interpreter + Development.Module (nanobind requirement).
    # Hint only, so an explicit -DPython_EXECUTABLE still wins (see the schema
    # code-generation block above for the same rule).
    if(NOT Python_EXECUTABLE AND EXISTS "${CMAKE_SOURCE_DIR}/.venv/bin/python3")
        set(Python_EXECUTABLE "${CMAKE_SOURCE_DIR}/.venv/bin/python3")
    endif()
    find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
    message(STATUS "Python_EXECUTABLE: ${Python_EXECUTABLE}")
    message(STATUS "Python_INCLUDE_DIRS: ${Python_INCLUDE_DIRS}")

    # Fetch nanobind via FetchContent.
    include(FetchContent)
    FetchContent_Declare(
        nanobind
        GIT_REPOSITORY https://github.com/wjakob/nanobind.git
        GIT_TAG        v2.2.0
    )
    FetchContent_MakeAvailable(nanobind)

    # Let nanobind set up its CMake integration.
    nanobind_add_module(
        q3_core
        NB_STATIC
        src/bindings/q3_core_module.cpp
        ${Q3_GPU_KERNEL_SOURCES}
    )
    add_dependencies(q3_core generate_schemas)

    # Link against the same q3_core interface target used by the executable.
    target_link_libraries(q3_core PRIVATE q3_cpp)

    # The q3_core INTERFACE target name conflicts with the nanobind module name.
    # We therefore create an alias.  The INTERFACE target is already named
    # q3_core and the nanobind module is also q3_core — CMake disambiguates
    # them because one is INTERFACE and the other is a MODULE library.
    # If there is a naming conflict in the future, rename the INTERFACE target
    # to q3_cpp_core and update all target_link_libraries calls.

    message(STATUS "Q3_ENABLE_PYTHON=ON: building q3_core nanobind extension")
    message(STATUS "  Python: ${Python3_EXECUTABLE}")
endif()
