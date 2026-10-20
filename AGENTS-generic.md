# AGENTS.md

This document describes generic architecture and conventions for AI coding agents.

## Conventions

### Design Patterns

-   **Factory**: when multiple implementations are possible - e.g. multiple exchange-correlation functionals, or multiple k-point samplers.
-   **Object-oriented design**: define abstract interfaces for components that have multiple implementations - e.g. Method → PseudopotentialMethod, PlaneWaveMethod, Model → DFTModel, HFModel, etc.

### OOP Guidelines & Antipatterns

**Prefer polymorphism over type-checking chains.** Instead of:

```cpp
// ❌ ANTIPATTERN: long if-chain checking object type
if (functional.is_lda()) {
    compute_lda_energy(...);
} else if (functional.is_gga()) {
    compute_gga_energy(...);
} else if (functional.is_meta_gga()) {
    compute_meta_gga_energy(...);
}
```

Use:

```cpp
// ✅ CORRECT: polymorphic dispatch via virtual method
Real energy = functional.compute_energy(density);
```

**Key principles:**

-   **Single Responsibility**: each class does one thing. If a class has methods for reading, computing, and writing, split it.
-   **Open/Closed**: add new behavior by adding new classes, not by adding `if` branches to existing code.
-   **Interface Segregation**: keep interfaces small. Don't force implementors to provide methods they don't need.
-   **No `is_xxx()` type queries**: if you need `is_ultrasoft()`, `is_paw()`, `is_norm_conserving()`, your design likely needs a virtual method instead.
-   **Favor composition over inheritance** for combining behaviors: use mixins (e.g., `BinarySerializableMixin`) rather than deep inheritance hierarchies.
-   **Use factories** to create the right subclass from runtime configuration (e.g., `create_diagonalizer("davidson")`).

### Logging

-   All output via logger
-   Allow log level to be set via command line argument (critical, error, warn, info, debug, trace). Default is error.
-   Only rank 0 outputs (MPI-aware initialization)
-   No print statement in the log

### Testing

-   Unit tests: `tests/unit/` (GTest)
-   Integration tests: `tests/integration/`

### Comment Style

-   **Multiline docstrings** must have `/**` on its own line followed by the comment body:

```cpp
// ✅ CORRECT
/**
 * Compute the angular phase factor (-i)^l.
 */

// ❌ INCORRECT
/** Compute the angular phase factor (-i)^l.
 */
```

-   Use `///` for single-line doc comments. A run of two or more `///` lines is a
    multiline docstring and must be `/** ... */` instead.
-   Use `//` for **single-line** inline implementation comments. A run of two or
    more full-line `//` lines is a multiline comment and must be `/** ... */`
    instead — the same rule as for `///`. A trailing `//` after code is always
    fine, even on consecutive lines, and comments about differently indented
    code are separate comments, not a run.
-   Never use bare `/* ... */` for documentation; use `/** ... */`. The inline
    parameter-name idiom (`f(int /*count*/)`, `g(/*flag=*/true)`) is not
    documentation and is allowed.

These rules are machine-enforced by `scripts/repo/check_doc_comment_style.py`, which
runs as the `doc-comment-style` pre-commit hook (`--fix` auto-repairs everything
except the bare-block rule, which needs a human to confirm the comment is
documentation and not commented-out code). `tests/python/test_check_doc_comment_style.py`
additionally asserts the whole tree is clean, so the rules hold even for a
developer who has not installed the hooks.

### Imports

**HARD RULE: imports go at the top of the file. No exceptions.** A
function-level import is almost always a symptom, and the symptom is usually an
import cycle. Deferring the import to call time makes the cycle invisible to the
interpreter and to every reader — the modules still depend on each other, the
dependency just is not written down anywhere.

When two modules need each other, **fix the design, not the import**:

-   Extract what they share into a third module that sits below both, and have
    each import it downward. (Preferred — it is almost always the answer.)
-   Or invert the dependency: define the interface in the lower module and
    register/inject the higher module's implementation.
-   Never "solve" it by moving the import inside a function.

**Optional third-party dependencies are no exception.** A package the code
needs on its main paths is a base dependency, imported at the top. A package
only one entry point needs stays in an optional extra and is still imported at
the top, behind a module-level guard that the entry point checks:

```python
try:
    from phonopy import Phonopy
except ImportError:  # the optional `phonons` extra; run_phonopy_workflow says so
    Phonopy = None
```

A test that needs an optional package skips through
`pytest.importorskip(...)` at module level, and a test that must import modules
one by one uses `importlib.import_module`.

The rule is machine-enforced. Enable the linter's outside-top-level-import
check (ruff `PLC0415` / pylint `import-outside-top-level`) and let it fail the
build, and fail the build on anything that would silence it — an inline
suppression naming the rule, a blanket suppression, or a per-file ignore. Do
not leave either as a convention nobody runs.

### Linter

Use linting for autoformatting the codebase. Consider language-specific tools and/or prettier.

C++ formatting is pinned, not best-effort: `clang-format` runs at the version
pinned by the pre-commit hook (`.pre-commit-config.yaml`, currently 19.1.7),
which is the single source of truth for this repo's formatting. A
developer's system `clang-format` is advisory only — different major
versions (e.g. 18 vs. 19) disagree on continuation-line breaking, so
formatting with anything other than the pinned hook version can churn a file
back and forth between "formatted" states depending on who committed last.

### Pre-commit

Use pre-commit to run linters and formatters automatically.

### GitHub Actions

Use GitHub Actions to run tests and linters automatically.

## !!! IMPORTANT !!!: Code Editing & Development HARD RULES

### HARD RULE 1: Never commit without explicit ask from user

NEVER commit changes using `git commit` without the user's explicit ask. Leave files in the working directory for the user to review.

### HARD RULE 2: use `<PROJECT_DIRECTORY>/agents/workdir/` for ALL scratch files.

NEVER create any throwaway files at the top level of the project directory (`<PROJECT_DIRECTORY>`). The top level of `<PROJECT_DIRECTORY>` must remain clean and contain only tracked project files. All throw-away scripts — debug helpers, patch scripts, test snippets, one-off analysis scripts — MUST go in `<PROJECT_DIRECTORY>/agents/workdir/tmp/`. Create that directory if it does not exist. Examples of files that belong in `<PROJECT_DIRECTORY>/agents/workdir/tmp/`: `debug_*.py`, `fix_*.py`, `patch_*.py`, `print_*.py`, `test_*.py` / `test_*.cpp` that are not formal tests in `tests/`, any other ephemeral script written to inspect or patch source code. Any potentially reusable agent artifacts should be either in `<PROJECT_DIRECTORY>/agents/workdir/reusable` (if they're intended to be used in the current project only) or in `<PROJECT_DIRECTORY>/agents/plan` (if they're intended to be used in multiple projects). NO EXCEPTIONS.

### HARD RULE 3: Always setup and use a virtual environment

(`venv`) when working with Python. Do NOT install Python packages globally. Use pyenv to select python version(s). Create venv in the agents workdir directory as explained in the next item

## HARD RULE 4: names with no abbreviations, Snake for Py, Camel for JS/TS, classnames

Variables, functions, methods, field names, class names, type names, file names. Always use full, descriptive names. For example:

-   ❌ `nkp`, `nbnd`, `nspin`, `npw`, `ik`, `ib`, `ig`, `ia`, `et`, `pw`, `ppset`
-   ✅ `number_of_kpoints`, `number_of_bands`, `number_of_spin_components`, `number_of_plane_waves`, `kpoint_index`, `band_index`, `g_index`, `atom_index`, `eigenvalues`, `planewave_basis`, `pseudopotential_set`
-   ❌ `Vec3`, `Mat3`, `IVec3`
-   ✅ `Vector3D`, `Matrix3x3`, `IntegerVector3D`

Also:

-   Use **snake_case** for variables, functions, and file names.
-   Use **PascalCase** for classes and structs.
-   Member variables use trailing underscore: `planewave_basis_`, `number_of_bands_`.
-   General rule: if a name looks abbreviated, spell it out. Exceptions could be made for complex physical and mathematical extpressions where the abbreviation is widely know and used for compacting the representation. However, even in these cases, try to spell it out. Make sure to make it obvious from the context what the abbreviation means.

## Other

### JSON Formatting

-   **HARD RULE**: JSON schemas MUST follow ESSE formatting conventions:
    -   **4-space indentation** (matching ESSE `.prettierrc`)
    -   **100 character print width**
    -   **Double quotes** only (standard JSON)
    -   **Trailing newline** at end of file
    -   **Bracket spacing** enabled (e.g., `{ "key": "value" }`)
    -   Q3 postprocessing/result schemas use ESSE-native names (e.g., `density_of_states`, not `dos_result`)
    -   All Q3 result schemas must `$ref` their corresponding ESSE schema in `schemas/esse/schema/`
