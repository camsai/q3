# Q3 — OCI image, and the single source of truth for every container runtime.
#
# Apptainer, podman-hpc and Shifter all consume OCI images; only Apptainer
# consumes SIF. So the image is built here once and the SIF is derived from it
# (q3.def bootstraps from this image), rather than the other way round. The
# reverse direction — `apptainer build --sandbox` + `podman import` — produces a
# single opaque layer with no architecture, entrypoint or labels in its config,
# which is what made the old ORAS-pushed SIF unusable outside Apptainer.
#
# Two stages: the toolchain, Python and sources stay in `builder`, and the
# published image carries the two executables plus the shared libraries they
# actually link. Size is load-bearing on HPC, not cosmetic — `podman-hpc
# migrate` squashes the image onto every compute node, and Shifter unpacks it
# per job.

# ---------- Stage 1: build ----------
FROM ubuntu:24.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive

# git is not optional: spglib and (when the system LibXC is too old to be
# found) LibXC are pulled by FetchContent with GIT_REPOSITORY. git-lfs is
# absent on purpose — the LFS payload is example data, which the build never
# reads, and .dockerignore keeps it out of the context entirely.
#
# OpenBLAS, not the reference netlib BLAS the -dev names without a vendor
# resolve to. Both compute the same answers; the reference implementation is
# simply not tuned, and Q3 is dense-BLAS3-bound outside the FFT. On Perlmutter
# the netlib-linked image ran the Cu 3x3x3 benchmark 10x slower than a native
# build at one thread and 42x slower at eight, because reference BLAS is also
# single-threaded, and swapping in OpenBLAS recovered all of it: 272,009 ms ->
# 28,049 ms at one thread, 265,604 ms -> 6,097 ms at eight, against 26,834 and
# 6,308 ms native (camsai/q3#12). The FFT stage moved 0.1%, which is what
# identifies the BLAS as the cause rather than the container.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        git \
        libfftw3-dev \
        libopenblas-dev \
        libxc-dev \
        libomp-dev \
        python3 \
        python3-venv \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/q3-src
COPY . /opt/q3-src

# Set up the Python environment BEFORE configuring/building: CMake's
# schema-generation step (generate_cpp_from_schemas.py, needs jsonref + jinja2
# + pyyaml) runs during `cmake --build`, not after it. The venv must live at
# exactly ${CMAKE_SOURCE_DIR}/.venv for CMakeLists.txt to auto-detect and use
# it (see the Q3_PYTHON / Python_EXECUTABLE checks); any other path silently
# falls back to the system python3, which lacks these packages.
RUN python3 -m venv /opt/q3-src/.venv \
    && /opt/q3-src/.venv/bin/pip install --no-cache-dir pyyaml jinja2 jsonref

# The commit this image was built from, stamped into q3 --version and every
# results.json. .dockerignore drops .git/ from the build context (see there),
# so the SHA has to be handed in; an empty value is the honest default and
# reports "commit unknown", which is what every published image said before
# the workflow started passing it (camsai/q3#6).
ARG Q3_GIT_COMMIT=""
RUN cmake -S . -B build \
        -DQ3_BUILD_TESTS=OFF \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX=/opt/q3 \
        -DQ3_GIT_COMMIT="${Q3_GIT_COMMIT}" \
    && cmake --build build -j"$(nproc)" \
    && cmake --install build

# ---------- Stage 2: runtime ----------
FROM ubuntu:24.04

LABEL org.opencontainers.image.title="Q3" \
      org.opencontainers.image.description="Q3 — a modern, from-scratch, extensible Density Functional Theory engine" \
      org.opencontainers.image.vendor="Mat3ra" \
      org.opencontainers.image.licenses="Apache-2.0"

ENV DEBIAN_FRONTEND=noninteractive

# Runtime halves of the builder's -dev packages, resolved for Ubuntu 24.04:
# libfftw3-dev -> libfftw3-double3 (CMake looks for `fftw3`, the double
# build, and no other precision), libopenblas-dev -> libopenblas0-openmp,
# libxc-dev -> libxc9. The -openmp build of OpenBLAS rather than -pthread:
# Q3 already sets OMP_NUM_THREADS and passes it to fftw_plan_with_nthreads,
# so one knob governs both thread pools and they cannot oversubscribe each
# other. update-alternatives points libblas.so.3 and liblapack.so.3 at it on
# install, which is what the binary loads. libgomp1 is GCC's OpenMP runtime: the build
# uses g++, so `find_package(OpenMP)` resolves to -fopenmp/libgomp rather than
# to the LLVM libomp that libomp-dev provides. Everything else Q3 depends on
# (spglib, yaml-cpp, spdlog, nlohmann_json) is static — CMakeLists.txt forces
# BUILD_SHARED_LIBS OFF for the FetchContent tree — so it needs nothing here.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libfftw3-double3 \
        libopenblas0-openmp \
        libxc9 \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Whatever cmake installed, under whatever names. Enumerating the binaries
# here couples this file to names that change -- q3-convert is folding into a
# `q3_utility convert` subcommand -- and installing to a private prefix rather
# than /usr/local means only Q3's own binaries are in the directory to copy,
# whatever base image the builder used.
COPY --from=builder /opt/q3/bin/ /usr/local/bin/

# A missing runtime library is a wrong entry in the package list above, and it
# must fail the build rather than the user's first job on a compute node. `ldd`
# prints "not found" and still exits 0, so grep decides the outcome. `q3
# --version` is the execution check: it is answered before any input is
# resolved, and `q3` is the one binary name that is not being renamed.
RUN if ldd /usr/local/bin/* | grep 'not found'; then exit 1; fi \
    && q3 --version > /dev/null

ENV PATH=/usr/local/bin:$PATH \
    OMP_NUM_THREADS=1

# Bare `q3`, so `podman-hpc run <image> q3 input.yml` and `shifter q3
# input.yml` name the executable the same way the docs do. ENTRYPOINT would
# make the image's default command implicit and those two forms wrong.
CMD ["q3"]
