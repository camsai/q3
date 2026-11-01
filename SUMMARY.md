# Q3 — Summary

## What Is Q3?

Q3 is a modern, from-scratch plane-wave density functional theory (DFT) code written in C++17. It solves the Kohn–Sham equations self-consistently to compute the ground-state electronic structure of periodic crystalline systems — the same fundamental problem addressed by established, widely used plane-wave DFT codes.

In ~55,000 lines of header-heavy C++ (plus ~27,000 lines of tests — exact, auto-generated figures in [STATISTICS.md](STATISTICS.md)), Q3 implements the core plane-wave pseudopotential DFT pipeline: crystal structure setup, pseudopotential parsing, plane-wave basis generation, Hamiltonian construction, self-consistent field iteration, and post-SCF forces and stress — with **norm-conserving (NC)**, **ultrasoft (US)** and **projector augmented-wave (PAW)** pseudopotentials wired end-to-end, **MPI** k-point pool + band parallelism, and **GPU** backends for NVIDIA (CUDA), Apple Silicon (Metal), AMD (HIP), and Intel (SYCL). PAW reads UPF v2 datasets, and further input, dataset and output formats plug in as build-time add-ons that core never names.

Q3 is developed as part of the [Mat3ra](https://mat3ra.com) ecosystem and targets interoperability with the [ESSE](https://github.com/Exabyte-io/esse) (Exabyte Source of Schemas and Examples) data standard for structured input and output.

---

## Why Q3?

The landscape of plane-wave DFT codes is dominated by a handful of mature projects:

-   **Open-source Fortran codes** — decades of development, hundreds of thousands of lines of code
-   **Proprietary Fortran codes** — decades of development, restricted source access

These are mature, battle-tested, and widely used, and Q3's own validation rests on one of them, Quantum ESPRESSO. They are largely written in Fortran 90/2003 and were designed as standalone applications, before AI coding agents and today's data platforms existed.

Q3 exists because the materials simulation field needs a **clean, modern, extensible** DFT engine that can serve as the computational core of a larger platform — not as a standalone monolithic application.

### Key Advantages

#### 1. From-Scratch Modern C++17 Implementation

Q3 is written from scratch in C++17. Every design decision reflects modern software engineering:

-   **No abbreviations** — `number_of_plane_waves` not `npw`, `pseudopotential_set` not `ppset`. The codebase is self-documenting.
-   **Strategy/Factory patterns** — XC functionals, mixers, diagonalizers, and smearing schemes are interchangeable via clean interfaces.
-   **Layered architecture** — physics layers 0–5 separate structure, representation, computation, and properties; IO is a vertical slice.
-   **No global state** — Q3 passes data through explicit object state rather than module-level singletons.
-   **RAII and modern memory management** — `std::unique_ptr`, `std::vector`, move semantics throughout.

For comparison: a mature Fortran plane-wave code's core solver alone is ~150,000 lines across ~300 files, the accumulation of decades of capability. Q3 achieves validated core SCF + forces + stress functionality — plus MPI and four GPU backends — in ~55,000 lines with clear module boundaries.

#### 2. ESSE-Oriented Data Model

Q3's input and output are designed around [ESSE](https://github.com/Exabyte-io/esse):

-   **YAML input** — human-readable manifests aligned with ESSE examples, with Quantum ESPRESSO's pw.x namelist format as an alternative.
-   **Structured JSON output** — energies, eigenvalues, forces, and stress written as machine-readable JSON alongside binary checkpoints compatible with established codes.
-   **Schema codegen** — a Python generator (`scripts/repo/generate_cpp_from_schemas.py`) produces C++ types under `src/generated/` from ESSE and Q3 JSON schemas.

Full schema validation and complete ESSE property coverage are still in progress. See [FUNCTIONALITY.md](FUNCTIONALITY.md) §3.1 for the current I/O inventory.

Q3 writes structured data from the start, so downstream tools read results rather than parse text output.

#### 3. Validated Against Established Codes

Q3 is not a toy implementation. Automated validation tests compare Q3 directly against an established plane-wave DFT code (LibXC-enabled build) on identical inputs for **NC and US** pseudopotentials:

-   **Total energy**: agreement to < 1 mRy for Si and Cu (NC and US)
-   **Forces**: agreement to < 10⁻⁵ Ry/Bohr (NC), < 10⁻⁴ Ry/Bohr (US)
-   **Stress tensor**: agreement to < 2 kbar per component
-   **Energy contributions**: individual terms (Hartree, XC, Ewald, kinetic) validated independently

This level of numerical agreement demonstrates that Q3's physics is correct at the same level of theory as established codes. PAW is validated against QE (UPF datasets, ~1 mRy/atom).

#### 4. NC and US Pseudopotentials in a Unified Codebase

Q3 supports norm-conserving and ultrasoft pseudopotentials within a single inheritance-based design:

-   **Generalized eigenvalue problem** for US (S-matrix) handled by the diagonalizer and overlap operator
-   **Augmentation charge** Q(G) and **effective D-matrix** updates for US
-   **UPF v1 and v2** readers, compatible with standard pseudopotential libraries
-   **PAW computed end-to-end** from UPF v2 datasets, including spin polarization, forces and stress; UPF v1 PAW is still rejected at read time

In Q3, NC and US are integrated from the start with parallel class hierarchies (basis, quantum system, SCF iterator, forces, stress).

#### 5. Designed for AI-Assisted Development

Q3's codebase is developed with AI coding agents as first-class contributors. The naming conventions, documentation standards, and architectural patterns are specifically chosen to be unambiguous to both human and AI readers:

-   **No abbreviations anywhere** — eliminates the domain-specific shorthand barrier that makes physics codes opaque to general-purpose developers and AI tools
-   **Comprehensive AGENTS.md** — codifies conventions, hard rules, and calculator decomposition patterns for AI agents
-   **Automated validation pipeline** — changes can be verified against a reference implementation on canonical examples
-   **Structured inputs/outputs** — JSON and YAML that AI tools can reason about

#### 6. Compact and Comprehensible

The entire Q3 source tree is ~55,000 lines of C++ in ~321 header files plus 28 translation units (entry points and GPU backends) — small enough for a single developer (or AI agent) to understand in its entirety. For context:

| Code             | Language | Lines (core solver) | Lines (total) |
| ---------------- | -------- | :-----------------: | :-----------: |
| **Q3**           | C++17    |       ~55,000       |    ~83,000    |
| Reference PW DFT | Fortran  |      ~150,000       |   ~500,000+   |
| ABINIT           | Fortran  |      ~100,000       |   ~800,000+   |
| GPAW             | Python/C |       ~50,000       |   ~100,000+   |

This compactness reflects a focused scope: ground-state SCF, forces, stress, band structure, DOS, and geometry relaxation for NC/US/PAW — not the full feature surface of codes developed over three decades. Advanced functionals (meta-GGA, hybrids) are on the roadmap, not yet shipped.

---

## What Q3 Is Not

-   **Not a full-feature replacement** — established codes carry decades of phonon, GW, TDDFT, and other capabilities. Q3 focuses on the core ground-state DFT engine.
-   **Not feature-complete on ESSE** — structured output exists and schema codegen is in place, but not every ESSE property type is emitted or schema-validated yet.
-   **Not cluster-proven yet** — MPI k-point pools and band groups are implemented and validated on a single node (topology-invariant energies on 1/2/4 ranks); multi-node scaling and MPI+GPU are not yet benchmarked. HIP and SYCL backends compile but have not run on AMD/Intel hardware.
-   **Not a standalone research platform** — Q3 is designed as a **component** of the Mat3ra ecosystem, though it works well as a command-line tool today.

---

## Architecture at a Glance

```
src/
├── main.cpp                        # Serial entry point (main_mpi.cpp, main_gpu*.cpp — MPI/GPU variants)
├── layer_0_core/                   # Types (Real, Vector3D, Matrix), math, execution policies, constants
├── layer_1_structure/              # Crystal, spglib symmetry, k-point grids
├── layer_2_quantum_system/         # Abstract wavefunction, density, potential concepts
├── layer_3_representation/         # Plane-wave basis, pseudopotentials (UPF v1/v2), charge density
├── layer_4_computation/            # SCF iterators, mixing, diagonalization, smearing, XC
├── layer_5_properties/             # Forces, stress tensor, ESSE result serialization
├── io/                             # YAML/namelist input, JSON/binary output, UPF readers, checkpoints
├── bindings/                       # q3_core nanobind Python module
├── py/                             # q3 Python package (workflows, ASE calculator)
├── generated/                      # Auto-generated C++ types from JSON schemas
└── utils/                          # Shared utilities (e.g. timestamps)
```

All FetchContent dependencies (LibXC, spglib, spdlog, yaml-cpp, nlohmann_json, GoogleTest) are resolved automatically at configure time. **FFTW3** and **LAPACK** must be installed on the system; **LibXC** is used from the system when it is from a tested series (5.2 or 7.0), otherwise 7.0.0 is built from source.

---

## Current Status

Q3 is in active development. The core SCF engine is production-quality for semiconductors and metals with **NC, US and PAW** pseudopotentials. Forces, stress, band structure, DOS, geometry relaxation, JSON results, and binary checkpoints are validated against reference implementations on silicon and copper examples. MPI k-point pool + band parallelism and the CUDA/Metal GPU backends are validated on a single node. Spin polarization is cross-code validated for norm-conserving and ultrasoft pseudopotentials — NC on ferromagnetic Fe/Ni and antiferromagnetic Cr, ultrasoft against QE. All five smearing schemes are cross-code validated; HIP/SYCL backends await hardware validation. DFT+U runs for all three families and reproduces QE's total energy to 4.5e-7 Ry when handed QE's own input file.

For the authoritative breakdown, see:

-   [FUNCTIONALITY.md](FUNCTIONALITY.md) — current capabilities by area (Science / Performance / Engineering), with validation status
-   [TODOs.md](TODOs.md) — planned and in-progress work, by area and horizon
-   [STATISTICS.md](STATISTICS.md) — codebase metrics (auto-generated)
