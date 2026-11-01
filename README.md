<p align="center">
  <img src="docs/figures/logo.webp" alt="Q3 logo" width="128"/>
</p>

<h1 align="center">Quantum 3xplorer (Q3)</h1>

<p align="center">
  An AI-native engine for next-generation quantum materials modeling, written
  from scratch by AI coding agents under the direction and supervision of
  domain scientists.
</p>

<p align="center">
  <a href="https://github.com/camsai/q3/actions/workflows/ci.yml"><img src="https://github.com/camsai/q3/actions/workflows/ci.yml/badge.svg" alt="CI Status"/></a>
  <a href="LICENSE.md"><img src="https://img.shields.io/badge/License-Apache_2.0-blue.svg" alt="License"/></a>
  <a href="https://en.cppreference.com/w/cpp/17"><img src="https://img.shields.io/badge/language-C%2B%2B17-blue.svg" alt="Language"/></a>
  <img src="https://img.shields.io/badge/platform-macOS%20%7C%20Linux-lightgrey.svg" alt="Platform"/>
</p>

Q3 is a self-contained, from-scratch quantum simulation engine. Its first
implementation is plane-wave density functional theory (DFT): it computes the
ground-state electronic structure of periodic crystals — total energies, forces,
stress, and band structures — in modern, readable C++ with a Python layer on top.

| If you want to…                       | Go here                                                                                                        |
| ------------------------------------- | -------------------------------------------------------------------------------------------------------------- |
| Run your first calculation            | [Tutorials](docs/tutorials/README.md) — seven runnable lessons, or [§2](#2-getting-started) for the short path |
| Know what Q3 can do today             | [FUNCTIONALITY.md](FUNCTIONALITY.md) — capabilities by area, each with its validation status                   |
| Know where Q3 is still weak           | [KNOWN-ISSUES.md](KNOWN-ISSUES.md) — the honest list; read it before trusting an untested regime               |
| Look up an input field or output file | [docs/io_reference.md](docs/io_reference.md) — every field, flag and format, with units and defaults           |
| Understand or change the code         | [docs/architecture.md](docs/architecture.md) — the layer map and dependency rules                              |
| Configure a build, or fix one         | [docs/build_reference.md](docs/build_reference.md) — every CMake variable, with HPC examples                   |

## Contents

-   [1. Introduction](#1-introduction) — [what Q3 does](#11-what-q3-does), [highlights](#12-key-highlights), [details](#13-in-more-detail)
-   [2. Getting Started](#2-getting-started) — [container](#21-run-via-container), [build](#22-build-from-source), [first run](#23-run-your-first-calculation), [pw.x namelists](#24-run-a-pwx-namelist-input), [Python](#25-install-the-python-package), [release archives](#26-download-a-release-archive)
-   [3. Workflows](#3-workflows) — [band structure](#31-band-structure-cli), [relaxation](#34-geometry-relaxation), [variable cell](#35-variable-cell-relaxation), [phonons](#36-γ-point-phonons)
-   [4. Development](#4-development) — [branches](#41-branches), [MPI & GPU builds](#42-parallel--gpu-builds), [testing](#43-testing), [architecture](#44-architecture--directory-layout), [contributing](#45-contributing)
-   [5. Documentation](#5-documentation)
-   [6. License](#6-license)

## 1. Introduction

### 1.1 What Q3 Does

Density functional theory predicts where the electrons sit in a material, and
most of what you want to know follows from that: the total energy of a
structure, the forces pulling each atom toward equilibrium, the stress on the
unit cell, and the band structure that decides whether the material conducts.

Q3 solves that problem the way most codes for crystalline solids do — the
plane-wave pseudopotential method. Three ideas carry the whole calculation:

-   **Plane waves.** A crystal repeats indefinitely in three directions, so its
    electronic wavefunctions are naturally expressed as sums of periodic waves.
    How many waves you keep is the _cutoff_, and it is the main
    accuracy-versus-cost dial you turn.
-   **Pseudopotentials.** An atom's tightly bound core electrons take enormous
    numbers of plane waves to describe and almost never participate in
    chemistry. A pseudopotential replaces them with a smooth effective
    potential, leaving only the valence electrons to compute. Q3 reads
    norm-conserving, ultrasoft, and PAW datasets.
-   **Self-consistency.** The potential the electrons feel depends on where the
    electrons are, which in turn depends on the potential. Q3 starts from a
    guess and iterates — the self-consistent field, or SCF, loop — until the two
    agree to a tolerance you set. That loop is where the runtime goes.

You supply a crystal structure, an exchange-correlation functional, cutoffs, a
k-point grid, and pseudopotential files. Q3 returns structured JSON — total
energy, forces, stress, eigenvalues — plus binary checkpoints you can restart
from. Inputs come either as a Q3 YAML manifest or as an unmodified Quantum ESPRESSO
(pw.x) namelist input ([§2.4](#24-run-a-pwx-namelist-input)), and you
can drive it from the command line or from Python ([§3](#3-workflows)).

One caveat worth internalizing early: a calculation that converges is not the
same as a calculation that is correct. Cutoffs, k-grids and smearing all have to
be converged deliberately for the system in front of you, which is what
[tutorial 03](docs/tutorials/03-convergence.md) exists to teach.

### 1.2 Key Highlights

-   **C++ Core, Native Python Bindings, Python Workflows** — a from-scratch C++17 engine, stateful `q3_core` nanobind bindings, and a `q3` Python package with an ASE calculator and band-structure / relaxation / phonon workflows.
-   **Differentiable, JAX Support** — analytic forces (−dE/dR) and stress (dE/dε), exposed as a JAX-differentiable energy primitive via `q3.jax`; the full gradient API is charted in [docs/differentiability.md](docs/differentiability.md).
-   **MPI & GPU (Multi-Arch) Support** — k-point-pool and band-group MPI parallelism, GPU backends for NVIDIA (CUDA), Apple Silicon (Metal), AMD (HIP), and Intel (SYCL), and amd64 + arm64 container images.
-   **Modular Layered Architecture, Clear Data Standards** — six dependency-ordered layers from math foundations to property calculations, with structured I/O built on the [ESSE](https://github.com/mat3ra/esse) data standard.
-   **Object-Oriented Design, Compartmentalized Complexity** — RAII, modern memory management, strategy/factory patterns, and no global state.
-   **Drop-In Support for Quantum ESPRESSO Formats** — runs unmodified pw.x namelist inputs and writes Quantum ESPRESSO-compatible binary checkpoints, so existing pipelines keep working.

### 1.3 In More Detail

-   **Explicit Naming (No Abbreviations)**: Completely self-documenting code. Variable names are fully spelled out (e.g., `number_of_plane_waves` instead of `npw`, `pseudopotential_set` instead of `ppset`).
-   **Pseudopotential Support**: Norm-Conserving (NC), Ultrasoft (US), and Projector Augmented-Wave (PAW) pseudopotentials, including spin polarization, forces and stress. UPF paths are validated against a LibXC-enabled Quantum ESPRESSO build.
-   **pw.x-namelist compatible**: runs an unmodified Quantum ESPRESSO (pw.x) namelist input, refusing by name any key it cannot honour rather than ignoring it — walked through in [tutorial 06](docs/tutorials/06-existing-inputs.md).
-   **LibXC Integration**: All exchange-correlation via the standard LibXC library — named functionals (PZ, PBE, PBEsol, BLYP, revPBE, RPBE) plus arbitrary `libxc:X:C` combinations, addressed by LibXC id or by name.
-   **Symmetry-Aware**: Integrates with SPGlib for spacegroup detection and Monkhorst-Pack grid symmetry reduction.
-   **Standardized Data Models**: Built on the ESSE (Exabyte Source of Schemas and Examples) data standard, outputting structured JSON results and writing Q3-native and Quantum ESPRESSO-compatible binary checkpoints.

## 2. Getting Started

Four ways in, in increasing order of setup cost: download a release archive
([§2.6](#26-download-a-release-archive)), pull the container
([§2.1](#21-run-via-container)), build from source
([§2.2](#22-build-from-source)), or install the Python package on top of a build
([§2.5](#25-install-the-python-package)). For a guided path — from a fresh clone
to a converged result you can defend — start with the
[tutorials](docs/tutorials/README.md) instead; if something breaks along the
way, [TROUBLESHOOTING.md](TROUBLESHOOTING.md) collects the failures seen so far
and their root causes.

### 2.1 Run via Container

A pre-built container is available for HPC clusters where installing
dependencies is impractical. Each merge to `main` tags the repository and cuts a
release, which in turn builds and publishes an OCI image to
`ghcr.io/camsai/q3`. Images are built natively (not emulated) for **x86_64
(amd64) and aarch64 (arm64)**, and published under a single multi-architecture
tag. The image contains `q3` and `q3_utility` installed beside it.

One image serves every runtime. Apptainer, podman-hpc and Shifter all read OCI
images, and Apptainer converts to its own SIF format on the way in, so the same
tag works whichever of the three your centre provides:

-   **Apptainer** (formerly Singularity) — most university and national clusters.
-   **`podman-hpc`** — the preferred runtime at centres that provide it.
-   **Shifter** — those same centres, and older Cray systems.

`latest` tracks `main` and is republished on every merge, so the image behind
it changes without the name changing. Pin the date-based tag (`YYYY.M.D-N`) for
anything whose results have to stay comparable — a rerun months later against
`latest` is not the same calculation. See [§4.1 Branches](#41-branches). The
plain tags are OCI image indexes covering both architectures, so a pull
resolves the one matching the machine you are on; each architecture is also
published under its own `-amd64` / `-arm64` tag if you would rather be
explicit.

The plain tag holds the **serial** `q3`, scaled within one node by
`OMP_NUM_THREADS`. Running it under `srun` gives you independent single-rank
processes, not a parallel calculation. Two suffixed tags carry the parallel
executables ([§4.2](#42-parallel--gpu-builds)):

| Tag           | `q3` is the  | Also named    | Architectures |
| ------------- | ------------ | ------------- | ------------- |
| `latest`      | serial build | —             | amd64, arm64  |
| `latest-mpi`  | MPI build    | `q3_mpi`      | amd64, arm64  |
| `latest-cuda` | CUDA build   | `q3_gpu_cuda` | amd64         |

Every image carries one engine binary and runs it as `q3`; the variant name is
a symlink to it, for scripts that already use it. The MPI and CUDA builds run
the self-consistent field only, so the other steps of a workflow (NSCF, band
structure) still need the plain image.

Running the parallel ones needs runtime flags that differ per container
runtime, which [docs/podman_hpc_and_shifter.md](docs/podman_hpc_and_shifter.md)
covers.

#### Apptainer

```bash
# Pull the latest release — the architecture is resolved for you
apptainer pull docker://ghcr.io/camsai/q3:latest
```

```bash
# Or a specific version — tags are date-based, YYYY.M.D-N, one per release
apptainer pull docker://ghcr.io/camsai/q3:2026.9.10-0
```

Releases cut before the images became OCI hold a Singularity image pushed as an
ORAS artifact instead. Those tags predate this section and only Apptainer can
read them, with `apptainer pull oras://…` rather than the `docker://` above.

Override the resolved architecture with `--arch` (`apptainer pull --arch arm64 …`)
when preparing a container on one architecture for a cluster of another.

```bash
# Using the runscript (q3 is the default entrypoint)
apptainer run q3_latest.sif examples/total_energy--semiconductor--silicon/nc.yml
```

```bash
# Or explicitly
apptainer exec q3_latest.sif q3 input.yml
```

Apptainer mounts your home directory and the current working directory into the
container by default, so relative paths in an input file resolve the way they do
outside it.

#### podman-hpc and Shifter

Some centres provide these instead of Apptainer, `podman-hpc` being the one they
prefer for new work. Both read this same image:

```bash
podman-hpc pull ghcr.io/camsai/q3:latest
```

Neither mounts the working directory for you the way Apptainer does, and
`podman-hpc` does not pass your environment through either, so the commands
differ enough to be worth their own page: see
[docs/podman_hpc_and_shifter.md](docs/podman_hpc_and_shifter.md) for both
runtimes, batch scripts, and what the MPI flags on each do not yet do for this
image.

#### Building the image yourself

The [`Dockerfile`](Dockerfile) is the single source of truth for what goes into
the image; [`q3.def`](q3.def) only converts the published result to SIF. Build
either with:

```bash
# OCI, for any of the three runtimes (podman and docker are interchangeable here)
podman build -t q3:local .
```

```bash
# SIF, from the published image (requires Apptainer ≥ 1.2)
apptainer build q3.sif q3.def
```

### 2.2 Build from Source

You need a C++17 compiler (**GCC 9 or newer**, or Clang) and a few system
libraries:

-   **macOS**: `brew install cmake fftw libomp lapack`
-   **Linux (Ubuntu/Debian)**: `sudo apt-get install -y cmake build-essential libfftw3-dev libomp-dev libopenblas-dev`

`libopenblas-dev` rather than `liblapack-dev libblas-dev`: those resolve to the reference
netlib implementation, which is correct but untuned and single-threaded. Q3 spends most of
its time outside the FFT in dense BLAS3, and the difference is a factor of ten serial and
forty threaded on a metal supercell — see the container images, which made the same mistake
until 2026-09-11.

All other dependencies (such as `LibXC`, `spglib`, `yaml-cpp`, `spdlog`, `nlohmann_json`, and `GoogleTest`) are fetched automatically during configuration via CMake `FetchContent`.

Clone, and create the Python environment the build itself needs — generating
C++ from the JSON schemas happens during `cmake --build`, and ESSE's schemas
come from the `mat3ra-esse` package, so these modules must be in place before
you configure:

```bash
git clone <repository-url> q3 && cd q3
python3 -m venv .venv && .venv/bin/pip install -e ".[build]"
```

```bash
cmake -S . -B build -DQ3_BUILD_TESTS=ON -DCMAKE_BUILD_TYPE=Release
```

```bash
cmake --build build -j$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)
```

This generates the main executable `build/q3` and the test suite
`build/q3_tests`. For MPI and GPU (CUDA / Metal / HIP / SYCL) builds, see
[§4.2](#42-parallel--gpu-builds).

To install the binaries somewhere specific:

```bash
cmake --install build --prefix /path/you/want
```

Every configuration variable, with example values for Homebrew, Ubuntu and an
HPC system, is in [docs/build_reference.md](docs/build_reference.md); build
failures and their fixes are in [TROUBLESHOOTING.md](TROUBLESHOOTING.md).

### 2.3 Run Your First Calculation

Run a self-consistent field calculation using one of the pre-configured examples:

```bash
./build/q3 examples/total_energy--semiconductor--silicon/nc.yml
```

Results (energies, eigenvalues, charges) are written to the directory named by
`control.outdir` in the manifest — `output/` for the examples. **That path
resolves against your current working directory, not the manifest's**, so the
command above writes `output/si_nc_results.json` under the repository root.
Everything else in a manifest (`material`, `pseudopotentials`) resolves
relative to the manifest itself.

What the numbers in that file mean, and how to pull them out, is
[tutorial 02](docs/tutorials/02-reading-results.md); the exhaustive field
reference is [docs/io_reference.md](docs/io_reference.md).

### 2.4 Run a pw.x Namelist Input

Q3 reads an unmodified Quantum ESPRESSO (pw.x) namelist input — the `&CONTROL`, `&SYSTEM` and
`&ELECTRONS` namelists and their cards — and resolves it to the same internal
model a YAML manifest does. Run it from the directory it lives in, because its
`pseudo_dir` resolves against your current directory:

```bash
cd examples/total_energy--semiconductor--silicon/reference/nc && ../../../../build/q3 scf_qe.in
```

The namelist reader refuses what it cannot honor: an unknown key, or a
recognized key Q3 does not support, is a hard error naming the key. A drop-in
that silently computes different physics is worse than one that stops.

Worked end to end — what the namelist resolves to, and the path trap above — in
[tutorial 06 — Bringing existing inputs](docs/tutorials/06-existing-inputs.md).

### 2.5 Install the Python Package

Q3 ships a `q3` Python package with high-level workflows and an ASE calculator.
It drives the `build/q3` binary from [§2.2](#22-build-from-source), so build
first. Python 3.12 or newer is required. Install from the repository root (the
root `pyproject.toml` packages `src/py/q3`):

```bash
pip install -e ".[test]"
```

The workflows it provides are shown in [§3](#3-workflows), and
[tutorial 05](docs/tutorials/05-python-and-ase.md) walks through driving Q3
from Python end to end.

### 2.6 Download a Release Archive

Every GitHub release carries native archives for Linux x86_64, Linux arm64 and
macOS on Apple Silicon:

```bash
curl -LO https://github.com/camsai/q3/releases/download/<tag>/q3-<tag>-linux-x86_64.tar.gz
curl -LO https://github.com/camsai/q3/releases/download/<tag>/SHA256SUMS
sha256sum --check --ignore-missing SHA256SUMS
tar -xzf q3-<tag>-linux-x86_64.tar.gz
```

The executables are in `bin/`. They link FFTW and BLAS/LAPACK from your system
rather than bundling them, and `INSTALL.txt` in the archive gives the exact
`apt-get install` line for Ubuntu 24.04 — the release pipeline runs that line in
a clean container and then an SCF with the unpacked binary before anything is
attached to a release. `licenses/` holds the notices of the libraries compiled
in, and `sbom.cdx.json` is a CycloneDX bill of materials listing every
component, statically and dynamically linked, with its version. On the public
repository each archive also carries a signed build-provenance attestation:

```bash
gh attestation verify q3-<tag>-linux-x86_64.tar.gz --repo camsai/q3
```

## 3. Workflows

A workflow chains several engine runs into one result — an SCF followed by a
band-structure pass, or a long series of SCF runs driving a geometry
optimizer. The band-structure workflow can be requested directly from a YAML
manifest; the rest are driven from the `q3` Python package
([§2.5](#25-install-the-python-package)), which wraps the engine in ASE's
optimizers and analysis tools.

### 3.1 Band Structure (CLI)

A band structure takes two passes: an SCF run to converge the charge density,
then a non-self-consistent (NSCF) run that holds that density fixed and
evaluates eigenvalues along a path through high-symmetry points of the
Brillouin zone. Set `calculation: band_structure` and Q3 does both:

```yaml
# examples/band_structure--semiconductor--silicon/nc.yml
material: ../data/materials/silicon.json
model:
    type: dft
    subtype: gga
    functional: pbe
method:
    type: pseudopotential
    subtype: norm-conserving
    cutoffs:
        wavefunction: 48.0
        density: 192.0
    pseudopotentials:
        Si: ../data/pseudopotentials/nc/Si.upf
    scf:
        convergence_threshold: 1.0e-08
        number_of_bands: 12
        mixing:
            mode: broyden
            beta: 0.7
            ndim: 8
        kgrid:
            dimensions: [8, 8, 8]
            shifts: [0, 0, 0]
control:
    calculation: band_structure
    outdir: ./output
    prefix: si_bands_nc
    band_path: ["G", "X", "W", "K", "G", "L"]
    band_path_points_per_segment: 30
```

```bash
./build/q3 examples/band_structure--semiconductor--silicon/nc.yml
```

This automatically:

1. Runs SCF with the specified k-grid to get the self-consistent charge density
2. Generates an NSCF input with explicit k-points along `Γ–X–W–K–Γ–L`
3. Runs NSCF to compute eigenvalues at each k-point
4. Writes results to `output/<prefix>_results.json`

### 3.2 Band Structure (Python)

The same workflow via the `q3` Python package (see [§2.5](#25-install-the-python-package)):

```python
from q3.workflows import run_band_structure_workflow

result = run_band_structure_workflow(
    "examples/band_structure--semiconductor--silicon/nc.yml",
    band_path=["G", "X", "W", "K", "G", "L"],
    lattice_type="FCC",
    points_per_segment=30,
    number_of_bands=16,
    q3_binary="./build/q3",
)

# Access results
eigenvalues = result["band_structure"]["eigenvalues_ry"]   # (n_kpoints, n_bands)
fermi_energy = result["band_structure"]["fermi_energy_ry"]
```

### 3.3 High-Symmetry Labels & Plotting

For FCC lattices: `G` (Γ), `X`, `W`, `K`, `L`, `U`. The k-path is interpolated in crystal-fractional coordinates using the primitive reciprocal lattice matrix.

Generate band structure plots with the included script:

```bash
.venv/bin/python scripts/plot/plot_bands_with_dos.py
```

See [docs/band_structure_comparison.md](docs/band_structure_comparison.md) for
validated Q3 vs Quantum ESPRESSO band structure comparisons (Si and Cu, NC and US).

### 3.4 Geometry Relaxation

Relaxation moves the atoms downhill along the forces until the structure sits
at a local energy minimum. Q3 supplies energies and forces; ASE's optimizers
choose each step:

```python
from q3.workflows import run_relax_workflow

result = run_relax_workflow(
    "examples/relaxation--semiconductor--silicon/nc_3x3x3.yml",
    fmax=0.02,             # force convergence threshold (eV/Å)
    steps=50,              # max steps
    q3_binary="./build/q3",
)

print(f"Converged: {result['converged']}")
print(f"Final energy: {result['final_energy_ry']:.6f} Ry")
```

The same thresholds can be set in the manifest instead:

```yaml
control:
    calculation: relaxation
    outdir: ./output
    prefix: si_nc_3x3x3_relax
    relax_max_steps: 50
    relax_force_threshold: 5.0e-3 # Ry/Bohr
    relax_energy_threshold: 1.0e-5 # Ry
```

**From a Quantum ESPRESSO deck.** A pw.x `relax` or `vc-relax` deck runs as it
is — any `ibrav`, with its `if_pos` flags honoured — on any Q3 binary:

```python
from q3.workflows import run_relax_workflow

result = run_relax_workflow(
    "examples/relaxation--semiconductor--silicon/reference/nc/relax.in",
    q3_binary="./build/q3",          # or q3_mpi, or a GPU binary
    # execution_policy=ExecutionPolicy(
    #     number_of_processes=64, number_of_kpoint_pools=64, launcher="srun -n {number_of_processes}"
    # ),
)
```

`ExecutionPolicy` (from `q3.binary`) describes the engine execution policy
from outside — ranks, k-point pools, band distribution, GPU precision and the
MPI launcher (`mpirun -np N` unless given) — and every `q3.workflows` function
takes it: each engine run the workflow makes is started with it — every relaxation step, the
SCF and the NSCF of a band structure, DOS or PDOS, every strain of an equation
of state or elastic-constant fit, every displacement of a phonon calculation.
`q3_mpi` runs the SCF and the NSCF on one rank per k-point pool, which its
automatic topology chooses when `number_of_kpoint_pools` is left unset; the SCF
needs it to write the checkpoint the NSCF reads. So use at most as many
processes as the SCF has irreducible k-points (and the NSCF lists): beyond that
ranks would share a pool, and `q3_mpi` refuses with a message saying how many
to use.

Each geometry is one SCF of the deck itself, run by that binary in the deck's
own directory, so the deck's relative `pseudo_dir` and `outdir` resolve as they
do for pw.x. The first run is the deck as written; the engine reports the
structure it read (the results' `material`) and Python's optimizer continues
from it, passing each new geometry back with `--material`. Every step starts
from the density the previous one saved in `outdir`. The deck's
`forc_conv_thr` and `nstep` become the defaults for `fmax` and `steps`. A
`vc-relax` deck goes through `run_vc_relax_workflow` the same way; give its
`press` as `target_pressure_gpa=press / 10` and remove it from the deck, which
Q3 refuses so that a target pressure is never silently dropped.

Running the deck directly — `q3 relax.in` — is not a relaxation: it runs one
SCF at the starting geometry and says so in the log.

See [PERFORMANCE.md](PERFORMANCE.md) for relaxation benchmarks (single-core, Q3 is 1.6–2.4× faster
than Quantum ESPRESSO across the Si and Cu supercell relaxation cases).

### 3.5 Variable-Cell Relaxation

Variable-cell relaxation adds the unit cell itself to the optimization, so both
the atomic positions and the lattice vectors relax together — driven by ASE's
`FrechetCellFilter`:

```python
from q3.workflows import run_vc_relax_workflow

result = run_vc_relax_workflow(
    "examples/relaxation--semiconductor--silicon/vc_relax_nc.yml",
    fmax=0.01,              # force/stress convergence threshold (eV/Å)
    steps=30,               # max BFGS steps
    q3_binary="./build/q3",
)

print(f"Converged: {result['converged']}")
print(f"Final energy: {result['final_energy_ry']:.6f} Ry")
print(f"Final cell (Å):", result["final_cell_angstrom"])
```

The cell is NOT held fixed: the filter relaxes it to the zero-external-pressure
equilibrium, so a strained starting cell will expand or contract as needed. This
workflow always runs in subprocess mode (not the nanobind `q3_core` bindings mode),
since the bindings' `SCFSolver` has no API to update the cell mid-run — see
[FUNCTIONALITY.md](FUNCTIONALITY.md)'s Variable-Cell Relaxation note for validation
details (a +2% isotropically strained Si cell recovers the unstrained equilibrium
lattice constant and matches an independent Quantum ESPRESSO variable-cell run
to <0.01%).

### 3.6 Phonons

Phonons are the quantized vibrations of the lattice. Q3 obtains them from
finite differences — displace atoms slightly, read the restoring forces — by
two routes. The Γ-point (q=0) route displaces every atom along every Cartesian
axis in the original cell, via ASE's `Phonons` class:

```python
from q3.workflows import run_gamma_phonons_workflow

result = run_gamma_phonons_workflow(
    "examples/total_energy--semiconductor--silicon/nc.yml",
    displacement_angstrom=0.01,  # finite-difference displacement magnitude
    q3_binary="./build/q3",
)

print(f"Frequencies (THz): {result['frequencies_thz']}")
print(f"Frequencies (cm⁻¹): {result['frequencies_cm1']}")
```

The cell is fixed throughout (only atomic positions are perturbed), so this
workflow can safely use the faster nanobind bindings mode. The
reduced-symmetry force bugs originally found through this workflow (a
`symmetrize_vector` transpose error and an IBZ k-point-reduction error) have
both been fixed and regression-tested — silicon's Γ-optical modes now come out
exactly triply degenerate, as required by diamond-structure symmetry (see
[FUNCTIONALITY.md](FUNCTIONALITY.md)'s Γ-Point Phonons note for the validation
status and the remaining cross-check caveat).

The second route uses [Phonopy](https://phonopy.github.io/phonopy/) as a
library and answers at every q, not only at Γ. Phonopy generates the
symmetry-reduced set of displaced supercells, `Q3ASECalculator` evaluates the
forces on each, and the force sets go back to Phonopy as force constants:

```python
from q3.workflows import run_phonopy_workflow

result = run_phonopy_workflow(
    "examples/total_energy--semiconductor--silicon/nc.yml",
    supercell_matrix=(2, 2, 2),
    band_path=["G", "X", "W", "K", "G", "L"],
    q3_binary="./build/q3",
)

print(result["phonon_dispersions"]["yDataSeries"])  # one series per branch, THz
print(result["thermal_properties"]["heat_capacity"]["yDataSeries"][0])  # J/K/mol
```

Symmetry is what makes this cheap: silicon's 2x2x2 supercell needs one
displacement direction, two SCF runs with the central difference the workflow
insists on, and out of them come the whole dispersion plus the harmonic free
energy, entropy and heat capacity. The dispersion, the phonon density of states
and the zero-point energy come back as ESSE properties. ESSE has no schema for
the temperature-dependent thermal properties, so each comes back as ESSE's
abstract two-dimensional plot against temperature, and none for force
constants, which stay in `phonopy_params.yaml`, named by an ESSE `file_content`
property.
Install it with `pip install q3[phonons]`; without Phonopy the rest of
`q3.workflows` is unaffected.

## 4. Development

Everything for working on Q3 itself: the branch model, the parallel and GPU
build variants, the test suites, the code layout, and how to contribute.

### 4.1 Branches

| Branch   | What it is                                                                                        | Who it is for                                                            |
| -------- | ------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------ |
| `main`   | The integration branch. Every green merge tags a release and publishes a container automatically. | Development, and anyone who wants the newest fixes as soon as they land. |
| `stable` | Conservatively managed and advanced deliberately, well behind `main`.                             | Production and long-running studies, where a moving base is the problem. |

`main` moves on every merge, which is what makes a fix reach users the same day
it lands — and also what makes two checkouts taken a fortnight apart genuinely
different codebases. `stable` exists for the case where that churn is the thing
you are trying to avoid: pin to it when a study needs the code underneath it to
stop moving, and take an update as a deliberate step rather than a default.

Container images are published from `main` for both amd64 and arm64 (see
[§2.1](#21-run-via-container)), so `latest` follows `main` rather than `stable`.
To hold a container fixed, pull a specific date-based tag rather than `latest`.

### 4.2 Parallel & GPU Builds

Each parallel backend is an opt-in CMake flag that selects which entry point a
build tree compiles. A tree builds exactly one engine binary, always `q3`, and
links the variant name to it in the build and install trees alike, so a
downstream workflow names `q3` whatever the build. Configure one tree per
variant; at most one GPU flag may be on, and MPI combines only with CUDA
(`q3_mpi_gpu_cuda`).

| CMake flag     | Also named    | Target hardware                                                            |
| -------------- | ------------- | -------------------------------------------------------------------------- |
| `Q3_WITH_MPI`  | `q3_mpi`      | MPI k-point pools + band groups (needs MPI, FFTW3-MPI; ScaLAPACK optional) |
| `Q3_WITH_GPU`  | `q3_gpu`      | Apple Silicon (Metal / Accelerate)                                         |
| `Q3_WITH_CUDA` | `q3_gpu_cuda` | NVIDIA (cuBLAS / cuSOLVER / cuFFT)                                         |
| `Q3_WITH_HIP`  | `q3_gpu_hip`  | AMD (ROCm)                                                                 |
| `Q3_WITH_SYCL` | `q3_gpu_sycl` | Intel (oneAPI / oneMKL)                                                    |

```bash
# MPI build and run: 4 ranks as 4 k-point pools
cmake -S . -B build_mpi -DQ3_WITH_MPI=ON && cmake --build build_mpi -j
```

```bash
mpirun -np 4 ./build_mpi/q3 --pools 4 examples/total_energy--semiconductor--silicon/nc.yml
```

```bash
# GPU run with mixed FP32/FP64 precision
./build_cuda/q3 --precision mixed examples/total_energy--semiconductor--silicon/nc.yml
```

See [FUNCTIONALITY.md](FUNCTIONALITY.md) §2 for per-backend validation status
and measured speedups, and [KNOWN-ISSUES.md](KNOWN-ISSUES.md) for the current
MPI limitations — notably that the stress tensor is skipped under band
parallelism.

### 4.3 Testing

#### 4.3.1 C++ Tests (GTest / CTest)

To run the full suite of unit and integration tests:

```bash
ctest --output-on-failure --test-dir build
```

#### 4.3.2 Python Validation & Regression Tests

Validation tests cross-check physical convergence against reference results;
regression tests verify binary outputs. Set up a virtual environment:

```bash
python3 -m venv .venv && source .venv/bin/activate && pip install pytest numpy pyyaml
```

Then run the fast tier:

```bash
pytest tests/ -v --q3-bin ./build/q3
```

Or the full SCF supercell convergence tier, which is considerably slower:

```bash
pytest tests/ -v --tier2 --q3-bin ./build/q3
# Tier 3, the campaign-scale runs (hours; the scheduled workflow runs them monthly):
pytest tests/validation/test_wannier_campaign.py tests/validation/test_wannier90_parity.py --tier3 --q3-bin ./build/q3
```

#### 4.3.3 MPI Validation Tests

With an MPI build present (auto-skipped when `mpirun` is unavailable), verify
that the total energy is invariant across rank/pool/band topologies:

```bash
pytest tests/validation/test_mpi_topology_invariance.py tests/validation/test_mpi_band_parallelism.py -v --q3-bin ./build_mpi/q3_mpi
```

### 4.4 Architecture & Directory Layout

Q3 is organized into six numbered layers, each depending only on the ones below
it. Layer 0 knows nothing about physics; layer 1 knows about crystals but not
electrons; the quantum-mechanical concepts enter at layer 2, become concrete
plane-wave representations at layer 3, get operated on by algorithms at layer 4,
and turn into observable properties at layer 5. I/O is deliberately a vertical
slice that may reach into any layer.

```
src/
├── main.cpp                        # Serial entry point
├── main_mpi.cpp                    # MPI entry point (k-point pools + band groups)
├── main_gpu.cpp                    # Apple Metal/Accelerate entry point
├── main_gpu_cuda.cpp               # NVIDIA CUDA entry point
├── main_gpu_hip.cpp                # AMD HIP/ROCm entry point
├── main_gpu_sycl.cpp               # Intel SYCL/oneAPI entry point
├── layer_0_core/                   # Foundation: types, math, execution infrastructure
│   ├── types/                      # Real, Complex, Vector3D, Matrix3x3, constants, FP32 types
│   ├── math/                       # BLAS/LAPACK/ScaLAPACK facades (+ GPU), spherical, interpolation
│   ├── execution/                  # Serial/MPI reduction policies, GPU device backends
│   ├── parallel/                   # K-point distributor (MPI pools)
│   ├── precision_policy/           # FP64 / mixed-precision interface
│   ├── profiling/                  # Performance counters
│   └── logging/                    # Serial logger re-export
├── layer_1_structure/              # Crystal structure (no electronic structure knowledge)
│   ├── crystal/                    # Lattice vectors, atoms, species, periodic table
│   ├── symmetry/                   # spglib wrapper, space group operations
│   └── reciprocal_space/           # Monkhorst-Pack k-point grid generation
├── layer_2_quantum_system/         # Abstract quantum-mechanical concepts
│   ├── core/                       # Wavefunction, occupations, quantum state
│   └── derived/                    # Charge density, effective potential
├── layer_3_representation/         # Concrete plane-wave representations
│   ├── basis/                      # PW basis, dual FFT grids (dense + smooth; serial/MPI/GPU)
│   ├── pseudopotential/            # Pseudopotentials + ultrasoft extensions, G-space tables
│   ├── quantum_system/             # Kohn-Sham quantum system snapshots (NC / US)
│   ├── charge_density/             # Plane-wave charge density with augmentation
│   ├── effective_potential/        # Effective potential representation
│   ├── wavefunction/               # Wavefunction initializers & projection
│   ├── quantum_state/              # Quantum state containers
│   └── mixins/                     # FFTransformable and other representation mixins
├── layer_4_computation/            # Algorithms operating on Layer 3 representations
│   ├── flavors/                    # Calculation flavors (SCF, NSCF)
│   ├── iterator/                   # SCF iterators; mixing (Broyden/Anderson, Kerker); smearing
│   ├── diagonalizer/               # Davidson & direct eigensolvers; H·ψ applicators (serial/GPU)
│   ├── effective_potential/        # V_eff = V_Hartree + V_xc + V_local; Ewald; XC via LibXC
│   ├── charge_density/             # Charge density accumulation from wavefunctions
│   ├── fermi_energy/               # Fermi energy determination
│   └── state/                      # SCF runtime state & result structs
├── layer_5_properties/             # Post-SCF property calculations
│   ├── forces/                     # Hellmann-Feynman forces (all contributions)
│   └── stress_tensor/              # Nielsen-Martin stress tensor
├── io/                             # File I/O (vertical slice, reaches into any layer)
│   ├── input/                      # YAML & pw.x-namelist input parsers
│   ├── pseudopotential/            # UPF v1/v2 readers
│   ├── checkpoint/                 # Binary checkpoints (Q3-native + QE-compatible)
│   ├── quantum_system/             # Quantum system I/O
│   ├── charge_density/             # Charge density I/O
│   ├── wavefunction/               # Wavefunction I/O
│   └── mixins/                     # Serialization mixins (binary)
├── bindings/                       # q3_core nanobind Python module
├── py/                             # q3 Python package (workflows, ASE calculator, CLI)
├── generated/                      # Auto-generated ESSE schema wrappers
└── utils/                          # General utilities
```

See [docs/architecture.md](docs/architecture.md) for the layer-by-layer
component map and the dependency rules that keep the layering honest, and
[docs/io_reference.md](docs/io_reference.md) for the full input/output field
reference.

### 4.5 Contributing

Contributions are welcome, and [tutorial 07](docs/tutorials/07-contributing.md)
walks through making one — finding your way around the layers, adding a test,
opening a PR. [CONTRIBUTING.md](CONTRIBUTING.md) covers the license (Apache-2.0,
inbound = outbound), the required DCO sign-off (`git commit -s`), and the
provenance policy every change must honor; engineering conventions live in
[AGENTS.md](AGENTS.md). Because Q3 is written from scratch rather than ported,
[PROVENANCE.md](PROVENANCE.md) is the one to read before consulting any other
engine's source: it defines what makes code a derivative work and the procedure
that avoids writing one.

## 5. Documentation

**Start here**

-   [docs/tutorials/](docs/tutorials/README.md) — seven runnable lessons from a fresh clone to a converged result you can defend, and on to contributing
-   [SUMMARY.md](SUMMARY.md) — what Q3 is, why it exists, and how it compares to established codes

**Using Q3**

-   [docs/io_reference.md](docs/io_reference.md) — every YAML / pw.x-namelist input field, CLI flag, and output file format, with units and defaults
-   [docs/build_reference.md](docs/build_reference.md) — every CMake configuration variable and option, with example values for Homebrew, Ubuntu and HPC systems
-   [docs/podman_hpc_and_shifter.md](docs/podman_hpc_and_shifter.md) — running the container at centres that provide `podman-hpc` or Shifter instead of Apptainer, with batch scripts
-   [FUNCTIONALITY.md](FUNCTIONALITY.md) — current capabilities by area (Science / Performance / Engineering), each with validation status
-   [KNOWN-ISSUES.md](KNOWN-ISSUES.md) — open limitations and the regimes to distrust
-   [TROUBLESHOOTING.md](TROUBLESHOOTING.md) — symptoms, root causes and fixes for failures seen so far
-   [PERFORMANCE.md](PERFORMANCE.md) — single-core and MPI benchmarks, with head-to-head comparisons

**Working on Q3**

-   [docs/architecture.md](docs/architecture.md) — layer-by-layer component map and dependency rules
-   [docs/differentiability.md](docs/differentiability.md) — the gradient/AD architecture note
-   [CONTRIBUTING.md](CONTRIBUTING.md) — license, DCO sign-off, and review expectations
-   [PROVENANCE.md](PROVENANCE.md) — how contributions stay permissively licensed: derivative-work rules and the procedure that avoids them
-   [TODOs.md](TODOs.md) — planned and in-progress work, by area and horizon
-   [STATISTICS.md](STATISTICS.md) — codebase metrics (auto-generated)

## 6. License

Q3 is licensed under the [Apache License 2.0](LICENSE.md).
Copyright and licensing are declared for every file in [REUSE.toml](REUSE.toml)
rather than in per-file headers, including the vendored pseudopotential tables,
which stay under their upstream licenses.
