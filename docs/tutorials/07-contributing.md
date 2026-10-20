# 07 — Contributing a change

**Goal:** find where a change belongs in a layered codebase, write it the way
the project writes things, prove it works, and get it merged.

The conventions here are not stylistic preferences with a rationale attached
afterwards. Each one exists because its absence caused a specific problem, and
most of them are enforced by a hook that will reject your commit rather than
leave the judgment to a reviewer. Reading this first is faster than
rediscovering them one failed hook at a time.

## 1. The layer map

Q3 is six numbered layers plus a vertical I/O slice. **Layer N may include from
layers 0…N−1, never upward and never sideways into a sibling's internals.**

| Layer | Owns                                                    |
| ----- | ------------------------------------------------------- |
| 0     | types, math, process groups, devices, logging           |
| 1     | crystal structure, symmetry, k-points, reciprocal space |
| 2     | bands, spin, occupations, wavefunctions (abstract)      |
| 3     | plane waves, G-vectors, FFT, pseudopotential tables     |
| 4     | SCF iteration, diagonalizers, mixing, smearing, pools   |
| 5     | derived properties — forces, stress, DOS, projections   |

The rule binds _identifiers_, not just includes: a layer-1 class may not have a
parameter named `number_of_bands`, because bands are a layer-2 concept. When a
lower layer needs behavior that only a higher layer can choose, the lower layer
declares an interface and the higher layer implements it — `FFTGridFactory` is
declared in layer 3 and implemented by layer 4's `PlanewaveExecutionPolicy`.

`src/io/` is the deliberate exception: parsing and serialization are a vertical
slice and may reach into any layer.

The Python package under `src/py/q3` follows the same discipline with four
tiers — `q3.constants` and `q3.errors` at the bottom, then adapters (`binary`,
`io`, `structure`),
then `calculators` and `flavors`, then `workflows` and `jax`. See
[tutorial 05](05-python-and-ase.md).

Both rules are checked automatically:

```bash
python3 scripts/repo/check_cpp_include_layering.py
```

```bash
python3 scripts/repo/check_python_import_layering.py
```

They print nothing and exit 0 when the tree is clean.

## 2. Where a change goes: a worked example

Say you want to add a smearing scheme — the `degauss` knob from
[tutorial 03](03-convergence.md). Smearing is SCF machinery, so it is layer 4,
and the existing schemes tell you exactly where:

```
src/layer_4_computation/iterator/internal/smearing/
├── init.hpp                  the SmearingScheme base class
├── factory.hpp               create_smearing(name, width)
├── fixed.hpp                 FixedOccupation
├── gaussian.hpp              GaussianSmearing
├── fermi_dirac.hpp           FermiDiracSmearing
├── marzari_vanderbilt.hpp    MarzariVanderbiltSmearing
└── methfessel_paxton.hpp     MethfesselPaxtonSmearing
```

One class per file, file named after the class in snake_case. That is a **hard
rule**, not a preference. Each new scheme is a new file, and the whole
implementation is a subclass overriding three methods:

```cpp
/// Gaussian smearing.
class GaussianSmearing : public SmearingScheme {
  public:
    explicit GaussianSmearing(Real smearing_width = 0.0) : SmearingScheme(smearing_width) {}

    [[nodiscard]] Real occupation(Real x) const override {
        return 0.5 * std::erfc(x);
    }

    [[nodiscard]] Real entropy_weight(Real x) const override {
        const Real sqrtpi_inv = 1.0 / std::sqrt(constants::pi);
        return -0.5 * sqrtpi_inv * std::exp(-x * x);
    }

    [[nodiscard]] std::string name() const override {
        return "gaussian";
    }
};
```

Header-only, because a self-contained class in one file is the default here;
separate `.cpp` files are for entry points, GPU backends, and implementations
big enough to hurt compile times.

Then register it in `factory.hpp`. Note what the existing entries do with
aliases:

```cpp
if (name == "marzari-vanderbilt" || name == "cold" || name == "m-v" || name == "mv") {
    return std::make_unique<MarzariVanderbiltSmearing>(smearing_width);
}
```

Those aliases are Quantum ESPRESSO's `smearing` namelist values, so a QE input
([tutorial 06](06-existing-inputs.md)) translates without a
name-mapping layer in between. A new scheme should accept whatever the
reference codes call it.

## 3. The test goes in the mirror image

**The directory structure under `tests/unit/` must mirror `src/`.** Not
approximately — exactly:

```
src/layer_4_computation/iterator/internal/smearing/gaussian.hpp
tests/unit/layer_4_computation/iterator/internal/smearing/test_gaussian.cpp
```

No ad-hoc `tests/unit/factories/`, no flattening. Tests are GoogleTest, and
they test one class each:

```cpp
#include <gtest/gtest.h>

#include "layer_4_computation/iterator/internal/smearing/fixed.hpp"

using namespace q3;

TEST(FixedOccupation, BelowFermiIsFullyOccupied) {
    FixedOccupation fixed;
    EXPECT_DOUBLE_EQ(fixed.occupation(-1.0), 1.0);
    EXPECT_DOUBLE_EQ(fixed.occupation(-0.001), 1.0);
    EXPECT_DOUBLE_EQ(fixed.occupation(-100.0), 1.0);
}
```

Unit tests are not the whole story. `tests/regression/` runs the binary and
checks its output; `tests/validation/` compares against reference-implementation
results for the committed examples. A change that alters a number belongs in one
of those too — the unit test proves your function does what you meant, and the
regression test proves the calculation still comes out right.

Build and run the C++ suite:

```bash
cmake -S . -B build -DQ3_BUILD_TESTS=ON && cmake --build build -j
```

```bash
ctest --output-on-failure --test-dir build
```

The Python suite needs the package installed
([tutorial 05](05-python-and-ase.md)) — without it every test that imports `q3`
fails to collect, which looks alarming and means nothing:

```bash
.venv/bin/python -m pytest tests/python -q
```

`tests/python/test_imports.py` is the one to run first when something looks
wrong with an install — it imports every public entry point and nothing else,
so it separates "the package is broken" from "my change is broken":

```bash
.venv/bin/python -m pytest tests/python/test_imports.py -q
```

```
tests/python/test_imports.py .........                                   [100%]

============================== 9 passed in 0.30s ==============================
```

## 4. Conventions that will bounce your commit

Install the hooks once and they run on every commit:

```bash
pre-commit install
```

```bash
pre-commit run --all-files
```

What they enforce, and why each one exists:

-   **`clang-format`, `ruff-format`, `ruff`, `prettier`** — formatting for C++,
    Python and Markdown. Prettier reformats Markdown tables, so run it before
    committing docs or your diff will churn.
-   **`cpp-include-layering`, `python-import-layering`** — the layer rules
    above. The Python one scans the whole package every run, so a cycle cannot
    slip through by having its two halves land in separate commits.
-   **`doc-comment-style`** — `/**` and `*/` each alone on their line;
    multi-line comments use `/** … */` rather than a run of `///` or `//`;
    `///` is for single-line docs only. Trailing `//` after code is always
    fine.
-   **`statistics-untouched`** — `STATISTICS.md` is generated, and a workflow
    refreshes it on `main` after every push. Never edit it and never commit it:
    a commit that stages it fails this hook, and a pull request that carries it
    fails the matching CI check. A local "stale" report from
    `scripts/repo/generate_statistics.py --check` is expected.
-   **`format-yaml-floats`, `check-yaml`, `check-json`, `trailing-whitespace`,
    `end-of-file-fixer`, `check-merge-conflict`** — the ordinary hygiene set.

Two more that no hook can check for you:

-   **No abbreviations in identifiers.** `number_of_kpoints`, not `nkp`.
    `smearing_width`, not `sigma`. This is the single most common review
    comment.
-   **Don't branch on physics inside a calculator.** An
    `if (has_nonlinear_core_correction)` inside a stress calculator is the
    wrong shape; a base class plus a subclass that overrides the one differing
    method is the right one. One class, one physical context.

`clang-tidy` runs in the hook set but needs a compile database
(`compile_commands.json`); if it fails locally, that is expected — CI is the
authority on it.

## 5. Documentation has a handoff rule

The project docs have deliberately non-overlapping jobs, and a feature lives in
exactly one place at a time:

-   `FUNCTIONALITY.md` — what exists **today**, in three areas: Science,
    Performance, Engineering.
-   `TODOs.md` — what is planned, using the **same** areas and numbering.
-   `SUMMARY.md` — the narrative. No capability tables.

**Any change that adds or validates a capability must, in the same pull
request, add or flip its row in `FUNCTIONALITY.md` and remove the matching item
from `TODOs.md`.** A feature listed in both is a bug in the documentation. Gaps
belong in `TODOs.md` only — do not restate "not implemented" notes in
`FUNCTIONALITY.md`; link instead.

When you describe validation, refer to "an established plane-wave DFT code" or
"a reference implementation" in those files.

## 6. Branch and submit

Branch from the tip of what you are working on, **not reflexively from `main`**.
This repository routinely stacks several open, unmerged feature branches, so
`main` is often behind the work you are extending:

```bash
gh pr list --state open
```

```bash
git log --oneline main..HEAD
```

Check both before assuming `main` is current — branching from a stale `main`
silently drops whatever the branch you were on already had.

Then push and open the pull request. On this repository `gh pr create` fails
against the GraphQL Projects-classic deprecation, so use the REST endpoint:

```bash
gh api -X POST repos/camsai/q3/pulls --input pr.json --jq '.html_url'
```

with `pr.json` holding `title`, `head`, `base` and `body`. The same applies to
edits — `gh pr edit` fails, `gh api -X PATCH repos/camsai/q3/pulls/<number>`
works.

A good pull request here states what changed, what it was validated against,
and what it deliberately did not do. The last part matters more than it sounds:
`KNOWN-ISSUES.md` exists because this project would rather document a limit
than let someone discover it in a production run.

## Where to go from here

You have run a calculation, read its output, converged it, taken it off the
grid and onto a k-path, driven it from Python, fed it someone else's input
files, and found your way around the source. The reference material picks up
where these tutorials stop:

-   [docs/io_reference.md](../io_reference.md) — every input field and output
    file, with units and defaults.
-   [docs/architecture.md](../architecture.md) — the layer map in full,
    including the execution-policy and GPU story.
-   [docs/ontology.md](../ontology.md) — the data model.
-   [docs/differentiability.md](../differentiability.md) — the JAX interface.
-   [FUNCTIONALITY.md](../../FUNCTIONALITY.md) and
    [KNOWN-ISSUES.md](../../KNOWN-ISSUES.md) — what works, and what to check
    before trusting it.
