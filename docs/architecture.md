# Q3 Layered Architecture

This document describes the physical code organization of Q3. The
codebase is structured as a strict layered architecture where each layer may only
depend on layers below it — never sideways or upward.

The visual diagram is in [architecture.drawio](architecture.drawio).

---

## Design Principles

1. **Strict downward dependencies** — Layer N may `#include` headers from layers
   0..N-1, never from N+1 or above.
2. **IO as a vertical slice** — File parsing and serialization live in `src/io/`,
   separate from the physics layers. IO code may reach into any layer it needs to
   read/write.
3. **Strategy pattern** — Pluggable algorithms (XC functionals, diagonalizers,
   mixers, smearing schemes) are abstract base classes with concrete
   implementations, all within the same layer.
4. **Execution-policy injection** — Physics classes are hardware-agnostic; MPI
   and GPU behavior is injected via execution-policy objects constructed in the
   entry points (see [AGENTS.md](../AGENTS.md) §6). Hardware-specific code lives
   in side-by-side files (`xxx.hpp` serial, `xxx_mpi.hpp`, `gpu/xxx.cpp`).
5. **No abbreviations** — All identifiers use full descriptive names
   (`number_of_kpoints`, not `nkp`).

---

## Entry Points & Executables

Six thin `main*.cpp` entry points select the execution policy; all physics code
is shared. Extracting the common setup+SCF workflow into a `CalculationRunner`
is tracked in [TODOs.md](../TODOs.md) §3.2.

| Executable    | Entry point         | CMake option   | Execution                                 |
| ------------- | ------------------- | -------------- | ----------------------------------------- |
| `q3`          | `main.cpp`          | (default)      | Serial CPU (+ OpenMP threads)             |
| `q3_mpi`      | `main_mpi.cpp`      | `Q3_WITH_MPI`  | K-point pools (`--pools N`) + band groups |
| `q3_gpu`      | `main_gpu.cpp`      | `Q3_WITH_GPU`  | Apple Metal / Accelerate                  |
| `q3_gpu_cuda` | `main_gpu_cuda.cpp` | `Q3_WITH_CUDA` | NVIDIA cuBLAS / cuSOLVER / cuFFT          |
| `q3_gpu_hip`  | `main_gpu_hip.cpp`  | `Q3_WITH_HIP`  | AMD hipBLAS / hipSOLVER / hipFFT          |
| `q3_gpu_sycl` | `main_gpu_sycl.cpp` | `Q3_WITH_SYCL` | Intel oneMKL (SYCL)                       |

GPU executables accept `--precision double|mixed` (see
[FUNCTIONALITY.md](../FUNCTIONALITY.md) §2). Two non-executable trees sit beside
them: `src/bindings/` (the `q3_core` nanobind Python module) and `src/py/` (the
`q3` Python package: workflows, ASE calculator, CLI).

One further binary is not an engine at all: `q3_utility` (`main_utility.cpp`, built
by default) holds the subcommands that consume a finished run's saved records
rather than an input manifest — `interpolate` evaluates an interpolation record
(layers 3–5 below) along a k-path and on a DOS mesh, `convert` re-exports a
saved record to an external format (mean-field → BerkeleyGW, interpolation →
Wannier90) or imports Wannier90 files into one, and `volumetric` writes a saved
density, effective potential or band onto a real-space grid as a Gaussian cube,
an XCrySDen `.xsf`, or a format a build-time add-on registers. Each subcommand is one class
under `src/commands/`. It is one binary with subcommands so that no science- or
format-facing suffix sits beside the `_mpi`/`_gpu` infrastructure suffixes, and
it is serial by construction.

---

## Layer 0: Core

Directory: `src/layer_0_core/`

Foundation types, math, and execution infrastructure with zero physics knowledge.

| Component              | Location                          | Purpose                                                                                                     |
| ---------------------- | --------------------------------- | ----------------------------------------------------------------------------------------------------------- |
| **Types**              | `types/`                          | `Real`, `Complex`, `Vector3D`, `Matrix3x3`, `IntegerVector3D`, `Matrix<T>`, FP32 types, physical constants  |
| **Math**               | `math/`                           | BLAS/LAPACK/ScaLAPACK facades, spherical Bessel & harmonics, Simpson integration, interpolation, tolerances |
| **GPU BLAS**           | `math/linear_algebra/gpu/blas/`   | Abstract `GPUBLAS` (FP64 + FP32); vendor impls: cuBLAS, hipBLAS, oneMKL, Accelerate/MPS                     |
| **GPU Eigensolver**    | `math/linear_algebra/gpu/lapack/` | Abstract `GPUEigensolver`; vendor impls: cuSOLVER, hipSOLVER, oneMKL, Accelerate LAPACK                     |
| **Execution policies** | `execution/`                      | `serial.hpp` / `mpi.hpp` reduction policies; `gpu/` device backends (CUDA, HIP, SYCL `.cpp`, Metal `.mm`)   |
| **Parallel**           | `parallel/`                       | `KPointDistributor` — round-robin assignment of IBZ k-points to MPI pools                                   |
| **Precision policy**   | `precision_policy/`               | FP64 / mixed-precision switching interface (concrete policies live in layer 4)                              |
| **Profiling**          | `profiling/counters.hpp`          | Wall-clock counters, MPI-reduced across ranks                                                               |
| **Logging re-export**  | `logging/init.hpp`                | Re-exports the serial logger so physics code never includes `io/logger_mpi.hpp`                             |

**Dependencies**: None (leaf layer).

---

## Layer 1: Physical System

Directory: `src/layer_1_structure/`

Physical description of the periodic crystal — no knowledge of electronic structure.

| Component                     | Location                                         | Purpose                                                                                                                                                                                    |
| ----------------------------- | ------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **Crystal**                   | `crystal/crystal.hpp`                            | Lattice vectors, atoms, species, periodic table, coordinate transforms                                                                                                                     |
| **Symmetry**                  | `symmetry/symmetry.hpp`                          | spglib wrapper — space-group operations, tensor symmetrization                                                                                                                             |
| **KPointsGrid**               | `reciprocal_space/kpoints_grid.hpp`              | Monkhorst-Pack mesh generation, symmetry-reduced k-points with weights                                                                                                                     |
| **G-vectors**                 | `reciprocal_space/`                              | `ReciprocalSpaceVector(sGrid)` — raw reciprocal-lattice vectors                                                                                                                            |
| **Wigner-Seitz supercell**    | `reciprocal_space/wigner_seitz_supercell.hpp`    | Lattice vectors of the k-mesh supercell's Wigner-Seitz cell, with degeneracy weights — the R set a k → R Fourier transform runs over                                                       |
| **Lattice Fourier transform** | `reciprocal_space/lattice_fourier_transform.hpp` | The k-set ↔ lattice sum pair, `to_real_space()` and `to_kpoint()`, over blocks of any size; the inverse evaluates at arbitrary off-mesh k, which is why it is a direct sum and not an FFT |

**Dependencies**: Layer 0 only.

---

## Layer 2: Quantum System

Directory: `src/layer_2_quantum_system/`

Abstract quantum-mechanical concepts — defines _what_ wavefunctions, densities,
and potentials _are_, without specifying how they are represented.

| Component              | Location                          | Purpose                                                    |
| ---------------------- | --------------------------------- | ---------------------------------------------------------- |
| **Wavefunction**       | `core/wavefunction/`              | Abstract wavefunction container (reciprocal-space storage) |
| **Occupation**         | `core/occupation.hpp`             | Band occupations                                           |
| **QuantumState**       | `core/quantum_state.hpp`          | Eigenvalues + occupations per k-point                      |
| **SpinResolvedEntity** | `core/spin_resolved_entity.hpp`   | Spin-channel container shared by density/potential         |
| **ChargeDensity**      | `derived/charge_density.hpp`      | Abstract charge density (real + reciprocal space)          |
| **EffectivePotential** | `derived/effective_potential.hpp` | Abstract effective potential                               |
| **QuantumSystem**      | `quantum_system.hpp`              | Aggregate of the above                                     |

**Dependencies**: Layers 0–1.

---

## Layer 3: Representation

Directory: `src/layer_3_representation/`

Concrete representations of Layer 2 abstractions — plane waves, and the
local-orbital Hamiltonian. The largest layer.

| Component                         | Location                                                 | Purpose                                                                                                                                                                                                                            |
| --------------------------------- | -------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **PlanewaveBasis**                | `basis/planewave/init.hpp`                               | G-vector selection per cutoff, plane-wave tables per k-point, band-group slab distribution                                                                                                                                         |
| **FFT grids**                     | `basis/planewave/internal/fft/`                          | `FFTGrid` interface; `SerialFFTGrid` (FFTW3), `MPIFFTGrid` (FFTW3-MPI slabs), `gpu/` (cuFFT, hipFFT, vDSP, SYCL)                                                                                                                   |
| **Dual-grid pair**                | `basis/planewave/internal/fft/pair.hpp`                  | Dense density grid (ecutrho) + smooth wavefunction grid (min(2·ecutwfc, ecutrho); min(4·ecutwfc, ecutrho), as pw.x sizes it, for QE-format input and with `control.use_exact_wavefunction_grid`)                                   |
| **Pseudopotential**               | `pseudopotential/init.hpp`, `set.hpp`                    | Base class + `PseudopotentialType` enum; polymorphic `PseudopotentialSet` container                                                                                                                                                |
| **Ultrasoft extension**           | `pseudopotential/ultrasoft/`                             | Augmentation charges (Q functions), US-specific D-matrix machinery                                                                                                                                                                 |
| **G-space tables**                | `basis/planewave/pseudopotential/internal/`              | Tabulated β projectors, local potential, core & atomic charge, projector mapping                                                                                                                                                   |
| **KS quantum system**             | `quantum_system/kohn_sham/planewave/`                    | NC and US Kohn-Sham system snapshots consumed by layer 4                                                                                                                                                                           |
| **Charge density (PW)**           | `charge_density/planewave/`                              | Concrete PW charge density with augmentation support                                                                                                                                                                               |
| **Wavefunction init**             | `wavefunction/planewave/norm_conserving/initializer.hpp` | Initial wavefunctions from atomic orbitals + subspace rotation                                                                                                                                                                     |
| **Mixins**                        | `mixins/ff_transformable.hpp`                            | Forward/inverse FFT-transformable behavior                                                                                                                                                                                         |
| **Local-orbital Hamiltonian**     | `local_orbital/init.hpp`                                 | `LocalOrbitalHamiltonian`: H(R) blocks, optional S(R), weighted Bloch sums — the second representation beside plane waves, the same object whether the orbitals are Wannier functions, atomic orbitals or Slater-Koster parameters |
| **Local orbitals in plane waves** | `basis/planewave/local_orbital/`                         | `PlanewaveLocalOrbitalSet` interface (orbitals rendered on a k-point's G-vectors); pseudo-atomic and hydrogenic implementations; factory resolving the input's choice once                                                         |

**Dependencies**: Layers 0–2. **Never** includes Layer 4 or above.

---

## Layer 4: Computation

Directory: `src/layer_4_computation/`

Algorithms that operate on Layer 3 representations. All pluggable strategies live here.

| Component                        | Location                                                         | Purpose                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |
| -------------------------------- | ---------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Calculation flavors**          | `flavors/`                                                       | `SelfConsistentField`, `NonSelfConsistentField` + factory                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             |
| **SCF Iterator**                 | `iterator/kohn_sham/planewave/init.hpp`                          | `PlanewaveKohnShamSCFIterator`: V_eff → diagonalize → accumulate ρ → mix → converge                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
| **PP / US iterators**            | `iterator/kohn_sham/planewave/pseudopotential/`, `…/ultrasoft/`  | KB projectors + D-matrix updates; US augmented density + S-matrix                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
| **Iterator execution**           | `iterator/kohn_sham/planewave/execution/`                        | Serial / MPI (pool + band reductions) / GPU policies for the SCF loop                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |
| **Precision policies**           | `iterator/kohn_sham/planewave/precision_policy/`                 | `FullDouble` (default) and `Mixed` FP32→FP64 switchover                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |
| **Mixing**                       | `iterator/internal/mixing/`                                      | Broyden + linear (reciprocal space, Kerker-preconditioned residual), Anderson (real space)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            |
| **Smearing**                     | `iterator/internal/smearing/`                                    | Fixed, Gaussian, Methfessel-Paxton, Marzari-Vanderbilt, Fermi-Dirac (+ factory)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       |
| **Diagonalizers**                | `diagonalizer/planewave/davidson.hpp`, `direct.hpp`              | Davidson (iterative) and direct (LAPACK) generalized eigensolvers                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
| **H·ψ applicators**              | `diagonalizer/planewave/internal/hamiltonian_applicator/`        | `serial.hpp` (ZGEMM projectors, dual-grid FFT), `gpu.hpp` (GPU BLAS/FFT, batched)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
| **Subspace**                     | `diagonalizer/planewave/internal/subspace_manager/`, `subspace/` | Subspace build/rotation (serial, GPU); diagonalization (serial LAPACK, ScaLAPACK MPI)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |
| **Effective potential**          | `effective_potential/`                                           | V_eff = V_Hartree + V_xc + V_local; Ewald summation; XC via LibXC (`XC_LibXC` + factory)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              |
| **Dispersion**                   | `effective_potential/internal/dispersion/`                       | `D3BJDispersionCalculator`: DFT-D3(BJ) two-body dispersion energy (Grimme parametrization)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            |
| **Density accumulation**         | `charge_density/planewave/internal/accumulator.hpp`              | Band-summed charge density from wavefunctions                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         |
| **Fermi energy**                 | `fermi_energy/`                                                  | Fermi-level bisection with smeared occupations                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        |
| **SCF state**                    | `state/`                                                         | `PlanewaveKohnShamSCFState` + energy component structs                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                |
| **Interpolator: contract**       | `interpolator/input_data/`, `factory.hpp`, `capture.hpp`         | `InterpolationSpecification` in and `InterpolationRecord` out, the two types that cross the component's boundary; the factory selects a method to evaluate a saved record, and capture selects one to build a record from a live plane-wave run                                                                                                                                                                                                                                                                                                                                                       |
| **Interpolator: target**         | `interpolator/target/quantum_state/`                             | `QuantumStateInterpolator` — the electronic state at any k, eigenvalues today; `LocalOrbitalQuantumStateInterpolator` evaluates any method that yields an H(R) via the layer-1 Bloch sum and the shared `SubspaceDiagonalizer`                                                                                                                                                                                                                                                                                                                                                                        |
| **Interpolator: Wannier method** | `interpolator/method/wannier/`                                   | `WannierHamiltonianBuilder`: a `SubspaceSelection` (`subspace/`: the window as it stands, or projectability disentanglement) then a `GaugeChoice` (`gauge/`: the projection gauge U = A(A†A)^(−1/2) as it stands, or the Marzari-Vanderbilt minimum), each chosen by a factory from `WannierConstructionSettings`, then the layer-1 transform to H(R); `WannierInputs` (E, A, M on the coarse grid, also read and written by `io/`); `internal/` holds the kernels on the layer-0 `DenseMatrix`; `planewave/` the capture kernels for M_mn(k,b) and A_mn(k), the only part that touches wavefunctions |

### Note on the Ultrasoft State Pattern

To minimize duplication, Q3 reuses the norm-conserving planewave SCF state for
ultrasoft calculations. The ultrasoft SCF iterator computes the US-specific
**effective D-matrix** (deeq) and patches the shared quantum-system snapshot
before returning the state, so property calculators (forces, stress) operate on
a unified state interface while dispatching to US-specific logic polymorphically.

**Dependencies**: Layers 0–3.

---

## Layer 5: Properties

Directory: `src/layer_5_properties/`

Post-SCF property calculations. These read converged SCF data and produce
physical observables.

| Component                         | Location                                                                          | Purpose                                                                                                                                                                        |
| --------------------------------- | --------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **Forces**                        | `forces/planewave/init.hpp`                                                       | Hellmann-Feynman forces: local, nonlocal, Ewald, NLCC core correction, US/PAW augmentation                                                                                     |
| **Stress tensor**                 | `stress_tensor/planewave/init.hpp`                                                | Nielsen-Martin stress: kinetic, Hartree, XC, Ewald, local PP, nonlocal PP, NLCC, US/PAW augmentation                                                                           |
| **Contribution structs**          | `forces/planewave/contributions.hpp`, `stress_tensor/planewave/contributions.hpp` | Per-term result breakdowns (ESSE-serialized)                                                                                                                                   |
| **Band structure (interpolated)** | `band_structure/interpolated/init.hpp`                                            | Eigenvalues along a k-path from any `QuantumStateInterpolator` — interpolation when the model came from a coarse mesh, exact evaluation when it came from real-space integrals |
| **DOS (interpolated)**            | `density_of_states/interpolated/init.hpp`                                         | Gaussian-broadened density of states from any `QuantumStateInterpolator` on a dense k-mesh                                                                                     |

Individual contribution kernels live under each property's `internal/` and
`pseudopotential/internal/` subdirectories.

**Dependencies**: Layers 0–4.

---

## IO Vertical Slice

Directory: `src/io/`

File I/O is organized as a vertical slice that can reach into any layer.

| Component                          | Location                                                                                                                                                               | Purpose                                                                                                                                                                                                                                                                                                                                                                                           |
| ---------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Input parsers**                  | `input/parser/` (`yaml.hpp`, `qe.hpp`, `factory.hpp`)                                                                                                                  | YAML manifest / QE-namelist → `InputData`; build-time add-ons register further formats                                                                                                                                                                                                                                                                                                            |
| **Input data**                     | `input/data.hpp`                                                                                                                                                       | Plain struct holding parsed input parameters                                                                                                                                                                                                                                                                                                                                                      |
| **Pseudopotential readers**        | `pseudopotential/` (`upf_reader.hpp`, `upf_reader/v1/`, `v2/`, `loader.hpp`)                                                                                           | UPF v1 (tag) / v2 (XML), into one PAW model; build-time add-ons register further dataset formats                                                                                                                                                                                                                                                                                                  |
| **Checkpoints**                    | `checkpoint/` (`q3.hpp`, `qe.hpp`, `factory.hpp`)                                                                                                                      | Binary save/restore in Q3-native and QE-compatible formats                                                                                                                                                                                                                                                                                                                                        |
| **Per-entity writers**             | `wavefunction/`, `charge_density/`, `quantum_system/`                                                                                                                  | Checkpoint I/O per quantum-system entity                                                                                                                                                                                                                                                                                                                                                          |
| **Serialization mixins**           | `mixins/binary_serializable.hpp`                                                                                                                                       | Shared binary serialization behavior                                                                                                                                                                                                                                                                                                                                                              |
| **Interpolation record + exports** | `interpolation/` (`checkpoint.hpp`, `internal/binary_stream.hpp`, `wannier/payload.hpp`, `init.hpp`, `factory.hpp`, `null.hpp`, `record_and_export.hpp`, `wannier90/`) | `checkpoint.hpp` is the `.q3ip` envelope — magic, version, method, lattice, Fermi level — dispatching to `wannier/payload.hpp`, the one file that knows a method; `internal/binary_stream.hpp` holds the bounded length-prefixed primitives; `InterpolationExporter` + factory + Null; `wannier90/` holds the interchange format's `reader`, `writer`, `exporter` (bohr/Ry → Å/eV) and `importer` |
| **Loggers**                        | `logger.hpp`, `logger_mpi.hpp`                                                                                                                                         | spdlog wrapper (serial); MPI variant silences non-zero ranks                                                                                                                                                                                                                                                                                                                                      |

**Key rule**: IO code depends on domain layers, never the reverse. Physics code
includes only `io/logger.hpp` (via `layer_0_core/logging/init.hpp`) for logging.

---

## Build-Time Add-Ons

Directory: `src/addons/<name>/`

An add-on is a directory under `src/addons/` that core never names: no core file
includes it, links to it or mentions it by name. `cmake/Q3Addons.cmake` adds every
`src/addons/*/CMakeLists.txt` it finds, in sorted order, so an add-on is compiled in
whenever its directory exists. Its `q3_register_addon(...)` call appends its unit tests to
the test targets, and two headers are generated into the build tree:
`q3_addon_identity.hpp` holds the active add-ons' names and versions as data, so any
core file may include it, and `q3_addon_registration.hpp` includes each add-on's
registration header and defines `q3::addons::register_all()`, which only the entry
points include and call, before anything else. With no add-on present both are empty.

An add-on reaches core only through registries, one per plug-in point:

| Plug-in point              | Core's built-ins                               | What an add-on registers                                                         |
| -------------------------- | ---------------------------------------------- | -------------------------------------------------------------------------------- |
| Input formats              | YAML (the fallback), QE namelist (`.in`/`.qe`) | Further formats, each with the companion outputs it writes after a run it parsed |
| Pseudopotential formats    | UPF                                            | Further dataset formats, consulted after UPF                                     |
| PAW one-center conventions | `absolute`                                     | Further conventions, each under its own key                                      |
| Volumetric formats         | cube, `.xsf`                                   | Further formats                                                                  |

`q3 --version` and the results file's `provenance.generator.addons` record which add-ons
a binary carries ([io_reference.md](io_reference.md) §2.1 and §4.1). No core directory
may be named `addons`, because the export drops that path segment at any depth, and an
add-on's tests and examples mirror core's layout under the same `addons/<name>/` segment.

---

## Dependency Matrix

```
             L0   L1   L2   L3   L4   L5   IO
Layer 0       —    ✗    ✗    ✗    ✗    ✗    ✗
Layer 1       ✓    —    ✗    ✗    ✗    ✗    ✗
Layer 2       ✓    ✓    —    ✗    ✗    ✗    ✗
Layer 3       ✓    ✓    ✓    —    ✗    ✗    ✗
Layer 4       ✓    ✓    ✓    ✓    —    ✗    ✗
Layer 5       ✓    ✓    ✓    ✓    ✓    —    ✗
IO            ✓    ✓    ✓    ✓    ✓    ✓    —
```

✓ = allowed dependency, ✗ = forbidden dependency.

**Known exceptions** (beyond the sanctioned `io/logger.hpp` logging include):
physics layers currently contain ~11 non-logger `io/` includes — checkpoint
factories reached from the SCF iterator/flavors, `io/input/data.hpp` used as a
config type, and the pseudopotential loader. Removing these via output adapters
and constructor-injected config is tracked in [TODOs.md](../TODOs.md) §3.2
(layer boundary cleanup).

---

## Units Convention

All internal computation uses atomic (Rydberg) units. Constants live in
[`layer_0_core/types/constants.hpp`](../src/layer_0_core/types/constants.hpp)
with full descriptive names (derived from CODATA SI values, matching QE):

| Quantity           | Unit               | Constant                                   |
| ------------------ | ------------------ | ------------------------------------------ |
| Energy             | Rydberg (Ry)       | SI-derived conversions in `constants.hpp`  |
| Length             | Bohr (a.u.)        | `BOHR_RADIUS_ANGSTROM`, `ANGSTROM_TO_BOHR` |
| Pressure output    | kbar               | `RYDBERG_TO_KILOBAR`                       |
| Lattice vectors    | units of `alat`    | —                                          |
| Reciprocal vectors | units of `2π/alat` | —                                          |
| `e²`               | 2.0 (Ry units)     | `electron_charge_squared_rydberg = 2.0`    |
| 2π, 4π             | —                  | `two_pi`, `four_pi`                        |

---

## External Dependencies

Core libraries are fetched via CMake FetchContent; system-provided packages are
required only for the build options that use them.

| Library                           | Purpose                 | Provided via / CMake option                                            |
| --------------------------------- | ----------------------- | ---------------------------------------------------------------------- |
| LibXC                             | XC functionals          | System or FetchContent                                                 |
| spglib                            | Symmetry                | FetchContent                                                           |
| spdlog                            | Logging                 | FetchContent                                                           |
| yaml-cpp                          | YAML parsing            | FetchContent                                                           |
| nlohmann_json                     | JSON                    | FetchContent                                                           |
| GoogleTest                        | Testing                 | FetchContent                                                           |
| FFTW3                             | FFT                     | System                                                                 |
| LAPACK / BLAS                     | Linear algebra          | System                                                                 |
| OpenMP                            | Threading               | System (`Q3_WITH_OPENMP`)                                              |
| MPI + FFTW3-MPI + ScaLAPACK       | Distributed parallelism | System (`Q3_WITH_MPI`; ScaLAPACK optional — redundant-serial fallback) |
| CUDA (cuBLAS, cuSOLVER, cuFFT)    | NVIDIA GPU              | System (`Q3_WITH_CUDA`)                                                |
| ROCm (hipBLAS, hipSOLVER, hipFFT) | AMD GPU                 | System (`Q3_WITH_HIP`)                                                 |
| oneMKL (SYCL)                     | Intel GPU               | System (`Q3_WITH_SYCL`)                                                |
| Accelerate + Metal + MPS          | Apple Silicon           | macOS frameworks (`Q3_WITH_GPU`)                                       |
