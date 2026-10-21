# AGENTS.md

This document describes specific items for Q3 codebase architecture and conventions for AI coding agents.

NOTE: also include ./AGENTS-generic.md - it describes common conventions for all the projects. This document describes project specific conventions.

## Project Structure

```
src/
├── main.cpp                        # Serial entry point
├── main_mpi.cpp                    # MPI entry point; main_gpu*.cpp — GPU entry points (Metal/CUDA/HIP/SYCL)
├── main_utility.cpp                   # q3_utility: subcommands over a finished run's saved records (interpolate, convert)
├── commands/                       # One class per q3_utility subcommand; entry-point code, exempt from layering like main*.cpp
├── layer_0_core/                   # Foundation types, math, execution policies, logging
├── layer_1_structure/              # Crystal structure, symmetry, k-points
├── layer_2_quantum_system/         # Abstract quantum-mechanical concepts
├── layer_3_representation/         # Concrete plane-wave representations
├── layer_4_computation/            # SCF iterators, diagonalizers, mixers
├── layer_5_properties/             # Post-SCF: forces, stress tensor, results
├── io/                             # File I/O (vertical slice)
├── bindings/                       # q3_core nanobind Python module
├── py/                             # q3 Python package (workflows, ASE calculator)
├── generated/                      # Auto-generated ESSE schema wrappers
├── utils/                          # General utilities
└── addons/                         # Build-time add-ons: each compiled in when its directory exists, never named by core
tests/
├── unit/                           # GTest unit tests
├── integration/                    # GTest integration tests
├── regression/                     # Python regression tests
└── validation/                     # Python validation tests vs a reference implementation
docs/
├── architecture.md                 # Layered architecture description
├── architecture.drawio             # Architecture diagram (draw.io)
├── ontology.md                     # Data model ontology
└── ontology.drawio                 # Ontology diagram (draw.io)
```

An add-on mirrors core's layout under its own `addons/<name>/` path segment in `src/`,
`tests/` and `examples/`, and reaches core only through core's registries (input formats,
pseudopotential formats, PAW one-center conventions, volumetric formats; see
[docs/architecture.md](docs/architecture.md)). Core never names an add-on: no core code,
test or doc includes, imports or links into one, and no core directory may be named
`addons`.

## Conventions

### Layer Separation (strong guideline, not a 100% hard rule)

Each layer's _identifiers_ (class, function, parameter, member names) should
only use concepts from its own layer or below. Includes must never point
upward: `layer_N` code must not `#include` a `layer_(N+k)` header — when a
lower layer needs behavior chosen at a higher layer, define an interface at
the lower layer and implement it above (dependency inversion; see
`FFTGridFactory` in layer 3, implemented by `PlanewaveExecutionPolicy` in
layer 4, and `QuantumSystemExecutionPolicy` in layer 2 giving band meaning
to layer-0 process groups).

Concept ownership by layer:

| Layer | Owns (examples)                                       | Must NOT appear in identifiers        |
| ----- | ----------------------------------------------------- | ------------------------------------- |
| 0     | types, math, process groups, devices, logging         | bands, k-points, plane waves, spin    |
| 1     | crystal, symmetry, k-points, reciprocal space         | bands, wavefunctions, pseudopotential |
| 2     | bands, spin, occupations, wavefunctions (abstract)    | plane waves, G-vectors, FFT grids     |
| 3     | plane waves, G-vectors, FFT, pseudopotential tables   | SCF iteration, diagonalizer choice    |
| 4     | SCF, diagonalizers, mixing, smearing, pools           | —                                     |
| 5     | derived properties (forces, stress, DOS, projections) | —                                     |

Accepted exceptions (the "not 100%" part): the centralized constants
registry `layer_0_core/math/tolerances.hpp` deliberately names
physics-specific thresholds in its dedicated section; comments may mention
higher-layer use cases for context (identifiers may not); `io/` is a
vertical slice and may reference any layer.

### Python Package Layering (`src/py/q3`)

The Python package follows the same downward-only rule as the C++ layers, for
the same reason: an upward import becomes an import cycle, and a cycle gets
"fixed" by hiding the import inside a function, which buries the design defect
instead of resolving it. See AGENTS-generic.md § Imports for the general rule.

A module may import from its own tier or any tier below it, never above:

| Tier | Modules                                               | Owns                                                                |
| ---- | ----------------------------------------------------- | ------------------------------------------------------------------- |
| 3    | `q3.workflows`, `q3.jax`                              | orchestration — multi-step runs, differentiation                    |
| 2    | `q3.calculators`, `q3.flavors`                        | one run of the engine, driven through an API                        |
| 1    | `q3.binary`, `q3.io`, `q3.structure`, `q3.properties` | process, format, and structure adapters; fits of derived properties |
| 0    | `q3.constants`, `q3.errors`                           | leaves — import nothing from `q3`                                   |

`q3/__init__.py` and `q3/__main__.py` are untiered entry points: they sit
above everything and are never imported by package code.

Enforcement is automatic, at two levels:

-   `PLC0415` is enabled in ruff (`pyproject.toml`), so a function-level import
    anywhere — package, tests or scripts — fails the lint job, and the
    `no-function-level-imports` pre-commit hook fails any `noqa` or
    `per-file-ignores` entry that would exempt one. There are no accepted
    exceptions: ASE is a base dependency imported at the top, and the
    optional `q3_core` extension and Phonopy are imported at the top behind a
    module-level `try`/`except ImportError` that sets the name to `None`.
-   `scripts/repo/check_python_import_layering.py` (pre-commit hook
    `python-import-layering`) enforces the tier table itself, catching upward
    imports that are legal at runtime and same-tier cycles. It scans the whole
    package on every run, so a cycle cannot slip through by having its two
    halves land in different commits.

Adding a subpackage under `q3/` means adding it to `TIERS` in that script;
`tests/python/test_check_python_import_layering.py` fails if you forget.

Worked example: `material_json_to_ase_atoms` used to live in
`q3.workflows.relax` (tier 3) while `q3.calculators.ase_calculator` (tier 2)
needed it, so the calculator imported the workflow from inside a method body
and the two formed a cycle. The fix was to move the adapter down to
`q3.structure.ase_atoms` (tier 1), which both sides now import at top level.

### File Organization

-   **HARD RULE**: One class per file. Each `.hpp` file must contain exactly one class definition. Do NOT place multiple classes in the same file.
-   Prefer header-only implementations (define methods inline in the `.hpp`) so that each class is fully self-contained in a single file. Use separate `.cpp` files only when necessary (e.g., very large implementations that significantly impact compile times, GPU vendor backends, entry points).
-   File names must match the class name in snake_case (e.g., `RealSpaceMixer` → `real_space_mixer.hpp`).
    -   **Known tension**: the codebase widely uses `init.hpp` as the package-entry file for a directory's primary class (~79 files, Pythonic `__init__` convention) rather than class-named files. Whether to bless `init.hpp` in this rule or migrate to class-named files is an open decision — see [TODOs.md](TODOs.md) §3.3. Until decided, follow the convention of the directory you are editing.

### Calculator Design

-   Avoid conditional branching (`if`/`else`) to handle different physical approximations (e.g., LDA vs GGA, NC vs US, with/without NLCC) inside a single calculator class. Instead, decompose into separate calculator classes — one per physical context — using inheritance or composition, and assemble them in the orchestrator. Each calculator should handle exactly one concern.
    -   ❌ `if (quantum_system.has_nonlinear_core_correction()) { ... }` inside a stress calculator
    -   ❌ `if (has_core_charge) { ... }` flag-based branching
    -   ✅ Base `ExchangeCorrelationStressTensorCalculator` (no PP dependency) with a virtual `build_total_charge_density_g()` method, and a `PseudopotentialExchangeCorrelationStressTensorCalculator` subclass that overrides it to include NLCC core charge

### Units

-   use atomic units (unitless) system during the runtime to minimize the conversion factors within the code
-   use the same units as Quantum ESRESSO for ease of comparison for standard output
-   use ESSE units for outputting the values in JSON/YAML output files
-   Keep physical constants and conversion factors in a separate file, e.g. `constants.hpp`

### Testing

-   **HARD RULE**: The directory structure under `tests/unit/` MUST mirror the directory structure under `src/`. A test file for `src/layer_X/foo/bar.hpp` must be placed at `tests/unit/layer_X/foo/test_bar.cpp`. Do NOT create ad-hoc directories like `tests/unit/factories/` or flatten the hierarchy.
-   In addition of other tests, add Regression tests: `tests/regression/` (Python, validate binary output)
-   Also add validation tests utilizing a set of example calculations
-   Use python for all the tests (pytest)
-   **HARD RULE**: New or modified benchmark/reference manifests, including
    scratch artifacts retained for review or handoff, MUST use paths relative
    to the manifest for repository data. Do not persist machine-specific
    absolute paths. Locate licensed or external data through an environment
    variable and inject its resolved path only into a process-local temporary
    overlay.
-   **Tests that launch a reference code must self-skip when it is absent.**
    The Quantum ESPRESSO comparisons need the pinned build (QE 7.5 + LibXC
    7.0.0) that sits outside the checkout and is on no CI runner, so each such
    test carries the shared `requires_qe` marker from `tests/helpers.py` --
    never a locally re-derived binary path, which is how one of them lost its
    guard and reddened the scheduled tier-2 job for two cycles. `helpers.py`
    finds the build under any ancestor of the repository root, so it is still
    found from a git worktree, and `$Q3_QE_BIN_DIR` overrides the search.
    `tests/validation/test_qe_reference_binary_guards.py` enforces this per
    pull request rather than per schedule.

### Documentation

The project docs have distinct, non-overlapping roles. Keep facts in exactly one place:

-   `SUMMARY.md` — narrative: what Q3 is and why it exists. No capability tables or task lists.
-   `FUNCTIONALITY.md` — present-state capabilities, organized into three user-facing areas: **Science** (§1), **Performance** (§2), **Engineering** (§3). Only list what exists today.
-   `TODOs.md` — planned/in-progress work, using the **same three areas and numbering** as `FUNCTIONALITY.md`, with `[S]`/`[M]`/`[L]` horizon tags.
-   `STATISTICS.md` — generated by `scripts/repo/generate_statistics.py`, and refreshed on
    `main` by CI. It counts the core only: each build-time add-on has its own
    `STATISTICS.md` in its `src/addons/<name>/` directory, generated and guarded the same
    way, and every export recounts the tree it publishes. **Never edit them, and never
    regenerate them in a pull request** — see "Leave STATISTICS.md alone" below.

Rules:

-   **Single source of truth**: a feature is either in `TODOs.md` (planned) or `FUNCTIONALITY.md` (done) — never both.
-   **Handoff on ship**: any change that adds or validates a capability must, in the same PR, add/flip its row in the matching `FUNCTIONALITY.md` area (Science/Performance/Engineering) and remove the corresponding item from the same area in `TODOs.md`.
-   **Gaps live in TODOs only**: do not restate "not implemented" notes in `FUNCTIONALITY.md`; link to the relevant `TODOs.md` section instead.
-   **Leave `STATISTICS.md` alone**: do not edit it, and do not run the generator
    to "keep it fresh" before committing. It counts lines and test cases across
    the whole tree, so it moved with almost every commit that touched source —
    twenty-two of the last forty before this rule — and always on the same
    adjacent rows of a nine-row table, which git cannot merge. Two open branches
    therefore conflicted on a file neither was allowed to author. CI now
    regenerates it on `main` after every push; a pull request that carries it
    fails `scripts/repo/check_statistics_untouched.py`, at commit time through the
    `statistics-untouched` hook and again on the runner. The one exception is a
    change to the generator itself — a new metric, a renamed row — where both
    files move together and the checker asks that they agree.
-   **Naming reference codes**: name the reference code in accuracy/validation claims —
    "validated against Quantum ESPRESSO" — credibility requires the name, and nominative
    use is fine. A claim against an engine that Q3 supports only through a build-time
    add-on is the exception: it lives in that add-on's own docs, never in core docs,
    comments or tests, so it leaves the published tree together with the add-on.
    Wall-clock/performance comparisons against commercial engines are different: their
    license terms restrict benchmark disclosure, so head-to-head timing results are kept
    out of the published tree until written publication permission exists. QE timing
    comparisons are unrestricted.

### Git Workflow

-   **Branch from the tip of the branch already checked out in this session, not from `main`.** When starting new work mid-session, `main` is very likely stale — this repo routinely stacks several open, unmerged feature PRs on top of each other (check `gh pr list --state open` before assuming `main` is current). Branching from `main` silently drops whatever the session's current branch already has.
-   Only branch from `main` instead when explicitly told to, or when `main` is confirmed to already contain the tip of the current work (e.g. the stacked PR chain has actually merged — verify with `git log --oneline main..<branch>` before assuming, don't infer merge state from CI status alone).

## Licensed Material and Provenance (HARD RULES)

> **Read [PROVENANCE.md](PROVENANCE.md) before implementing anything another
> computational materials science code also does.** It explains what does and does not
> count as a derivative work, and gives the step-by-step procedure. The rules
> below are the summary; that document is the reasoning and the recipe.

Q3 ships under Apache-2.0 and its public tree must contain only content we have
the right to distribute. These rules bind every contributor and every coding
agent, in the raw repository and in the published tree:

-   **Never open, quote, or paraphrase the source code of proprietary reference
    engines.** That source is licensed confidential material. Do not cite its file
    names, routine names, internal variable names, or comments anywhere in this
    repository. Ground parity work against such an engine in: published papers (for PAW, Blöchl, PRB
    50, 17953 (1994); Kresse & Joubert, PRB 59, 1758 (1999)), the engine's public manual
    and wiki (documented input tags and behavior are fine to cite), facts observable in
    the dataset files themselves, the engine's _outputs_, and Q3's own parity
    experiments. A build-time add-on that supports such an engine restates this rule in
    that engine's own terms in its `AGENTS.md`.
-   **Quantum ESPRESSO (GPL) may be read, in a research session, and cited**
    — file/routine citations in comments are encouraged as provenance — but
    never translate or transcribe its code. Formulas, constants, and conventions are fine;
    carried-over code structure or expression is not (GPLv2 is incompatible
    with this repo's Apache-2.0). When in doubt, implement from the paper and
    validate against QE's numbers.
-   **Never commit licensed data**: no licensed pseudopotential dataset files, and no
    content derived from them beyond format-level identity facts (a dataset's title
    line, its valence charge, its SHA-256 digest). Licensed datasets stay outside
    version control and are located at runtime through an environment variable.
-   **Every session is a research session or an implementation session, never
    both.** A research session may read GPL source and writes prose notes only,
    no code; an implementation session never fetches or opens GPL or
    proprietary source, and asks for a research session when it needs one.
    Side-by-side reviews are research sessions, run by someone who did not
    write the code, and are recorded in
    [docs/provenance_register.md](docs/provenance_register.md). Never place
    licensed dataset contents in an AI session's context, and use AI tools for
    engine-adjacent work only under no-training and no-retention terms. The
    definitions are in [PROVENANCE.md](PROVENANCE.md) § 3; state the session
    type and the sources consulted in the pull request.

## Repositories: raw and clean

Q3 is developed in a private repository (the **raw** side) and
published through `camsai/q3` (the **clean** side), which receives automated,
content-gated exports. What that means if you are contributing:

-   **Open pull requests against `camsai/q3`.** They are reviewed and merged
    there, then replayed into the raw repository with `git cherry-pick -x`,
    which preserves your authorship.
-   **Do not push directly to the published tree.** Its history is a series of
    `sync:` commits, each a gated snapshot of the raw repository; a direct
    push would be overwritten by the next export.
-   **The published tree's history is never rewritten after the public
    release.** Every container image's source offer (`THIRD_PARTY_NOTICES.md`)
    names a commit of it, and that offer has to hold for as long as the image
    is distributed. A mistake in published history is corrected by a new
    commit, never by a force-push.
-   **The HARD RULES above apply to every contribution.** An automated gate
    enforces them on the way out, but a violation is far cheaper to avoid than
    to unwind.

## External Dependencies

Core libraries fetched via CMake FetchContent — no manual installation needed except FFTW3 and LAPACK.

| Library       | Purpose        | Link target                    |
| ------------- | -------------- | ------------------------------ |
| LibXC         | XC functionals | `Libxc::xc`                    |
| spglib        | Symmetry       | `Spglib::symspg`               |
| spdlog        | Logging        | `spdlog::spdlog_header_only`   |
| yaml-cpp      | YAML           | `yaml-cpp::yaml-cpp`           |
| nlohmann_json | JSON           | `nlohmann_json::nlohmann_json` |
| GoogleTest    | Testing        | `GTest::gtest_main`            |

Parallel/GPU builds additionally require system packages per opt-in CMake flag
(MPI + FFTW3-MPI + optional ScaLAPACK, CUDA Toolkit, ROCm, oneMKL, or the Apple
Accelerate/Metal frameworks) — see [FUNCTIONALITY.md](FUNCTIONALITY.md) §3.3 for
the full table.
