# scripts/

Scripts are grouped by what they are about, not by what they do: a script that
checks a commit's sign-off and a script that checks an SCF landed on the right
energy are both "checks", and have nothing else in common.

| Directory               | What it holds                                           | Who runs it                                |
| ----------------------- | ------------------------------------------------------- | ------------------------------------------ |
| `repo/`                 | The repository's own rules, generated files, and mirror | pre-commit, CI, CMake                      |
| `validation/`           | The physics against a reference                         | a person; the container workflow; `parity` |
| `benchmark/`            | The engine's speed against a recorded baseline          | a person; `benchmark`                      |
| `plot/`                 | Figures for the documentation                           | a person                                   |
| `hpc/nersc_perlmutter/` | Build, validate and benchmark kit for one HPC system    | a person, on that system                   |

## `repo/` — repository mechanics

| Script                            | Purpose                                                                            | Run by                           |
| --------------------------------- | ---------------------------------------------------------------------------------- | -------------------------------- |
| `check_cpp_include_layering.py`   | `src/layer_N_*` includes point downward (AGENTS.md § Layer Separation)             | pre-commit                       |
| `check_python_import_layering.py` | the `q3` package's import tiers (AGENTS.md § Python Package Layering)              | pre-commit                       |
| `check_test_mirror.py`            | `tests/unit/` mirrors `src/`, every test is registered                             | pre-commit                       |
| `check_doc_comment_style.py`      | C++ doc-comment style, with `--fix`                                                | pre-commit                       |
| `check_statistics_untouched.py`   | a change does not carry `STATISTICS.md`                                            | pre-commit, `statistics-guard`   |
| `check_dco_signoff.py`            | commits carry a DCO sign-off                                                       | pre-commit (commit-msg), `dco`   |
| `check_script_paths.py`           | every `scripts/` path named in the wiring and docs exists                          | pre-commit                       |
| `check_cpp_type_queries.py`       | dynamic type queries only fall (AGENTS.md § Calculator Design); baseline beside it | pre-commit                       |
| `check_docstring_coverage.py`     | public C++/Python APIs stay documented; baseline beside it                         | pre-commit                       |
| `check_no_addon_leak.py`          | add-on engine text stays under `addons/` paths; hashed denylist beside it          | pre-commit, `addon-leak-check`   |
| `check_provenance.py`             | no new derivation signals (PROVENANCE.md § 4.2); core in `provenance_check/`       | pre-commit, `ci`                 |
| `check_pull_request_size.py`      | a pull request's reviewable size, generated and vendored paths excluded            | `ci` (advisory)                  |
| `format_yaml_floats.py`           | k-point floats in example YAML to nine decimals                                    | pre-commit                       |
| `generate_cpp_from_schemas.py`    | C++ headers from JSON schemas, with `templates/`                                   | CMake, pre-commit (`--check`)    |
| `generate_statistics.py`          | `STATISTICS.md`                                                                    | `statistics` on `main`           |
| `run_clang_tidy.py`               | clang-tidy over the compiled sources                                               | pre-commit, `ci`                 |
| `pack_public.py`                  | the clean public tree and each add-on's own export, and the gate on both           | `ci` (lint), `export-mirror`     |
| `import_public.py`                | contributions back from the public tree                                            | `import-mirror`, `export-mirror` |
| `package_release.py`              | a built tree as a release archive, with its licenses, checksums and SBOM           | `release`                        |

## `validation/` — the physics against a reference

| Script                               | Purpose                                                          |
| ------------------------------------ | ---------------------------------------------------------------- |
| `check_scf_result.py`                | a container SCF converged _and_ landed on the expected energy    |
| `generate_qe_input_from_material.py` | a Quantum ESPRESSO input from a Q3 material, for parity runs     |
| `run_mpi_reproducibility_matrix.py`  | serial, MPI and GPU runs of the same inputs, compared            |
| `run_parity_matrix.py`               | the parity matrix with one binary, for the A/B gate              |
| `compare_results_json.py`            | two such runs, every number and checkpoint byte, exactly         |
| `run_wannier_campaign.py`            | the automatic Wannier protocol over a fixed roster, as one table |
| `wannier_campaign/`                  | that campaign as a package: settings, roster, manifests, table   |

`tests/validation/` is the automated counterpart: what runs in CI in minutes.
This directory holds what takes hours or a cluster and produces a table or a
figure for the documentation.

The parity pair is the one part of it that runs on every pull request. The
`parity` workflow runs `tests/regression/parity_matrix.yml` with the merge base's
binary and the head's, then compares the two trees with no tolerance unless the
pull request declares a `numerics-change`. The reference is the code before the
change rather than another engine, which is what lets the answer be exact.

`wannier_campaign/` is the one package here: the campaign's fixed choices
(`settings.py`), its roster and how a run selects from it, the files the engine
reads per case, the table, and the resumable runner, one class per file.
`run_wannier_campaign.py` is its command line and nothing else; the tests
import the package directly.

## `benchmark/` — the engine's speed against a recorded baseline

| Script                    | Purpose                                                                                       |
| ------------------------- | --------------------------------------------------------------------------------------------- |
| `run_benchmarks.py`       | times the decks in `tests/regression/benchmark_matrix.yml`, one JSON record per run           |
| `compare_benchmarks.py`   | medians and spreads against a baseline in `docs/benchmarks/`; `--write-baseline` promotes one |
| `generate_performance.py` | the generated section of `PERFORMANCE.md`, from those baselines                               |

A timing means something only on the machine it was taken on, so every baseline
names its runner class and the comparison refuses to cross that line. The
`benchmark` workflow runs the decks tagged `ci` on each GitHub-hosted class it
has a leg for. The Perlmutter kit under `hpc/` keeps its own scripts for that
system.

## `plot/`

| Script                   | Purpose                                             |
| ------------------------ | --------------------------------------------------- |
| `plot_bands_with_dos.py` | Q3 and Quantum ESPRESSO bands and DOS, side by side |

## `hpc/nersc_perlmutter/`

The kit for NERSC Perlmutter; its own README covers use. Another system gets a
sibling directory named the same way, `<site>_<machine>`.

## Adding a script

Put it in the directory for what it is about. A script that locates the
repository from its own path must count the directory level it sits in —
`Path(__file__).resolve().parents[2]` from `repo/`, `validation/`, `benchmark/` or
`plot/`;
`../../..` from `hpc/<system>/` — because a root computed one level short points
at `scripts/`, and a checker scanning that finds nothing and passes.
`check_script_paths.py` catches a path written wrong; it cannot see one computed
wrong.
