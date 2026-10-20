# Q3 Tutorials

Eight short, runnable lessons that take you from a fresh clone to a converged
calculation you can defend, and then to contributing code.

Every command here runs against the examples committed in this repository, and
every number shown was produced by actually running it — if your output differs
materially, that is a bug worth reporting, not a rounding difference.

| #                                  | Lesson                   | You will                                                               | Time   |
| ---------------------------------- | ------------------------ | ---------------------------------------------------------------------- | ------ |
| [01](01-first-calculation.md)      | Your first calculation   | Build Q3 and run a silicon SCF; read the log                           | 15 min |
| [02](02-reading-results.md)        | Reading the results      | Pull energies, forces, stress and eigenvalues out of `results.json`    | 15 min |
| [03](03-convergence.md)            | Converging a calculation | Scan cutoff, k-grid and smearing; see why the defaults are not answers | 45 min |
| [04](04-band-structure-and-dos.md) | Band structure and DOS   | Run SCF → NSCF by hand and with the workflow; get silicon's real gap   | 30 min |
| [05](05-python-and-ase.md)         | Python and ASE           | Drive Q3 from Python, relax a structure, compute phonons               | 30 min |
| [06](06-existing-inputs.md)        | Bringing existing inputs | Run legacy input file formats                                          | 20 min |
| [07](07-contributing.md)           | Contributing a change    | Find your way around the layers, add a test, submit a PR               | 45 min |
| [08](08-wannier-interpolation.md)  | Wannier interpolation    | Turn a 64-k-point run into a model that gives bands at any k-point     | 40 min |

## Before you start

For lessons 01–04 you need a C++17 compiler (GCC 9 or newer, or Clang), CMake,
FFTW3 and LAPACK, plus a Python environment for the build itself. Tutorial 01
covers the install. Everything else (LibXC, spglib, yaml-cpp, spdlog,
nlohmann_json, GoogleTest) is fetched automatically at configure time, and
[docs/build_reference.md](../build_reference.md) lists every configuration
variable if the defaults do not suit your machine.

Lessons 04 and 05 also use the `q3` Python package, which requires **Python
3.12 or newer**; [tutorial 05](05-python-and-ase.md) covers that install, and
tutorial 04 points at it where it is first needed.

Lesson 08 additionally needs `./build/q3_utility`, built by the same command.

The lessons assume you are in the repository root and that `./build/q3` exists.
They are ordered — 03 in particular is the one that separates a number from a
result — but 06 stands alone if you arrived with a Quantum ESPRESSO namelist
in hand and want to see it run first.

Scans in lesson 03 run real calculations; the copper series through a 16×16×16
grid takes a couple of minutes of CPU on its own. Nothing in these lessons
needs more than a laptop.

## Where to go next

-   [docs/io_reference.md](../io_reference.md) — every input field and output
    file, with units and defaults. The reference to keep open.
-   [FUNCTIONALITY.md](../../FUNCTIONALITY.md) — what Q3 can do today, and what
    is validated against what.
-   [KNOWN-ISSUES.md](../../KNOWN-ISSUES.md) — the honest list. Read it before
    trusting a number in a regime you have not tested.
-   [docs/architecture.md](../architecture.md) — the layer map, if you are here
    to change the code.
