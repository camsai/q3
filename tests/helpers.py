"""Shared test utilities for q3 Python test suite.

Provides config overlay, subprocess runners, result loading, reference-code
stdout parsers (energy, energy contributions, forces, stress) plus shared
tolerance constants and LibXC version gating — so that all tests reference
examples/ as the single source of truth.

Q3's own numbers come out of results.json, not out of its console report: the
file is written whatever the report does, and nothing about a run's layout can
silently change what a test reads. The Q3-side force, stress and pressure
parsers that used to live here are gone for that reason. `parse_q3_total_energy`
is the one that remains, still used by the reference-code comparison suites that
read both engines' console output side by side.
"""

import copy
import functools
import importlib.util
import json
import math
import os
import re
import shutil
import subprocess
from pathlib import Path

import jsonschema
import numpy as np
import pytest
import yaml
from mat3ra.esse.models.properties_directory.non_scalar.density_of_states import (
    DensityOfStatesPropertySchema,
)
from q3.io.interpolation.results.classes import InterpolationResults
from q3.io.results.classes import Results, eigenvalue_arrays

# ── Canonical paths ──────────────────────────────────────────────────

Q3_ROOT = Path(__file__).parent.parent  # q3-minimal project root
EXAMPLES_DIRECTORY = Q3_ROOT / "examples"
DATA_DIRECTORY = EXAMPLES_DIRECTORY / "data"
PP_DIR = DATA_DIRECTORY / "pseudopotentials"
MATERIALS_DIR = DATA_DIRECTORY / "materials"

# ── Quantum ESPRESSO reference build ──────────────────────────────────

# These tests do not compare against "some pw.x": they compare against the
# pinned reference build, QE 7.5 linked against LibXC 7.0.0. generate_qe_input()
# below feeds QE `input_dft = 'XC-000I-000I-101L-130L'` (LIBXC_PBE), which a QE
# built without LibXC rejects outright, so a stock distribution package is not
# a substitute. That build lives outside the checkout, in
# `reference/dft/q-e/build_libxc/`, and is never present on a CI runner.
#
# Resolved in this order:
#
#   1. $Q3_QE_BIN_DIR, for a reference build kept somewhere else entirely.
#   2. `reference/dft/q-e/build_libxc/bin` under any ancestor of Q3_ROOT.
#
# Walking every ancestor, rather than reading Q3_ROOT.parent alone, is what
# makes this work from a git worktree. `git worktree add .claude/worktrees/<x>`
# puts the checkout four levels below the directory the reference tree sits
# beside, so Q3_ROOT.parent resolved to `.claude/worktrees/reference/dft/...`,
# which does not exist. Every live-QE test then skipped -- on a machine with
# the build present -- reporting the same clean skip and the same green run as
# CI, with no comparison performed. The walk costs one stat() per ancestor.
#
# An environment variable rather than a `--qe-bin` pytest option because the
# guard below is a skipif *condition*, evaluated when the test module is
# imported. That happens before pytest has parsed its command line, so no CLI
# option can reach it without turning every guard into a fixture.

QE_REFERENCE_BIN_SUFFIX = Path("reference") / "dft" / "q-e" / "build_libxc" / "bin"


def _resolve_qe_bin_directory():
    """Locate the LibXC QE build's `bin/`; see the note above for the order."""
    override = os.environ.get("Q3_QE_BIN_DIR")
    if override:
        return Path(override).expanduser()
    for ancestor in Q3_ROOT.resolve().parents:
        candidate = ancestor / QE_REFERENCE_BIN_SUFFIX
        if (candidate / "pw.x").is_file():
            return candidate
    # Absent: name the conventional location, so the skip reason points at the
    # path a developer would actually populate rather than at nothing.
    return Q3_ROOT.parent / QE_REFERENCE_BIN_SUFFIX


QE_BIN_DIR = _resolve_qe_bin_directory()
QE_PW = str(QE_BIN_DIR / "pw.x")
QE_AVAILABLE = os.path.isfile(QE_PW) and os.access(QE_PW, os.X_OK)

# The single guard every live-QE test wears. Centralised so a new QE test
# inherits it by importing it, instead of by remembering to re-derive the
# condition. Re-deriving is how test_qe_graphite_d3_dispersion.py ended up as
# the one module without a guard, and how it turned the scheduled tier-2 job
# red for two consecutive cycles -- runs 30676195249 (2026-08-01) and
# 31853121200 (2026-08-15) -- with a FileNotFoundError raised 1h37 and 1h47
# into a job nobody watches. test_every_live_qe_test_is_guarded in
# tests/validation/test_qe_reference_binary_guards.py enforces that it stays
# on, per pull request rather than per schedule.
requires_qe = pytest.mark.skipif(
    not QE_AVAILABLE,
    reason=f"QE LibXC reference build not available at {QE_PW} (override: $Q3_QE_BIN_DIR)",
)

# pw2bgw.x, the converter the BerkeleyGW export is compared against, from the
# same reference build. Tests that run it wear requires_qe as well.
QE_PW2BGW = str(QE_BIN_DIR / "pw2bgw.x")
requires_pw2bgw = pytest.mark.skipif(
    not (os.path.isfile(QE_PW2BGW) and os.access(QE_PW2BGW, os.X_OK)),
    reason=f"pw2bgw.x not available at {QE_PW2BGW} (override: $Q3_QE_BIN_DIR)",
)

# ── BerkeleyGW ────────────────────────────────────────────────────────

# BerkeleyGW's own executables, for tests that feed it Q3's mean-field files.
# Never on a CI runner: $Q3_BERKELEYGW_BIN_DIR names the build's `bin/`, and
# without it the executables are looked for on PATH (e.g. after `module load
# berkeleygw` on NERSC Perlmutter).


def _resolve_berkeleygw_executable(name):
    override = os.environ.get("Q3_BERKELEYGW_BIN_DIR")
    if override:
        return Path(override).expanduser() / name
    found = shutil.which(name)
    return Path(found) if found else Path(name)


BERKELEYGW_PARABANDS = _resolve_berkeleygw_executable("parabands.cplx.x")
requires_parabands = pytest.mark.skipif(
    not (BERKELEYGW_PARABANDS.is_file() and os.access(BERKELEYGW_PARABANDS, os.X_OK)),
    reason="BerkeleyGW parabands.cplx.x not found (set $Q3_BERKELEYGW_BIN_DIR or put it on PATH)",
)

# Matching Q3's constants.hpp: BOHR_RADIUS_SI = 0.529177210903e-10
BOHR_RADIUS_ANGSTROM = 0.529177210903

LIBXC_PBE = "XC-000I-000I-101L-130L"


# ── Global Test Tolerances (Rydberg) ──────────────────────────────────

# Reference LibXC version that expected values were generated with.
REFERENCE_LIBXC_VERSION = "7.0.0"

# When running with a different LibXC version, individual energy
# contributions and stresses are skipped entirely. This multiplier
# is only applied to total energy, which is much more stable.
CROSS_VERSION_TOLERANCE_MULTIPLIER = 5.0

# The example tolerances are set to 5e-4 to accommodate small numerical variations
# across different platforms and compilation optimization settings without triggering
# false failure flags on standard system examples.
# Rydberg -> electronvolt conversion (CODATA 2018), the same value Quantum
# ESPRESSO uses for its own rytoev so QE-comparison tests share one value.
RYDBERG_TO_ELECTRONVOLT = 13.605693122994

TOLERANCE_EXAMPLES_RYDBERG = 5e-4
TOLERANCE_QE_COMPARISON_RYDBERG = 1e-5
TOLERANCE_REGRESSION_RYDBERG = 0.001

# ── Global Test Tolerances (Stress, kbar) ────────────────────────────
# Internal regression: Q3 vs its own expected values (deterministic).
TOLERANCE_STRESS_REGRESSION_KBAR = 2.0
# QE comparison: Q3 vs Quantum ESPRESSO reference runs.
# The ~5-7 kbar inter-code variance is caused by different self-consistent
# charge densities from different Broyden mixing / diagonalization algorithms,
# NOT formula errors (every stress contribution is cross-checked term-by-term
# against the corresponding contribution QE prints).
TOLERANCE_STRESS_QE_COMPARISON_KBAR = 10.0


# ── Output Parsers ───────────────────────────────────────────────────


def parse_q3_total_energy(standard_output):
    """The last reported total energy, in Rydberg."""
    matches = re.findall(r"total energy\s*=\s*(-?\d+\.\d+)", standard_output)
    assert matches, "no total energy in Q3 output"
    return float(matches[-1])


# The Q3 force, stress and pressure stdout parsers that used to live here are
# gone: every quantity they recovered -- the total force per atom, each force
# and stress term, and the scalar pressure -- is a key of results.json, which
# the engine writes whether or not the console report was printed or its layout
# changed. `pressure_from_tensor` below is all that remains of them, and it
# derives rather than parses.


def extract_quantum_espresso_energy_contributions(quantum_espresso_output):
    contributions = {}
    for line in quantum_espresso_output.split("\n"):
        if "one-electron contribution" in line:
            contributions["one_electron"] = float(line.split("=")[1].split()[0])
        elif "hartree contribution" in line:
            contributions["hartree"] = float(line.split("=")[1].split()[0])
        elif "xc contribution" in line:
            contributions["exchange_correlation"] = float(line.split("=")[1].split()[0])
        elif "ewald contribution" in line:
            contributions["ewald"] = float(line.split("=")[1].split()[0])
        elif "paw contribution" in line:
            contributions["paw"] = float(line.split("=")[1].split()[0])
    return contributions


def extract_quantum_espresso_forces(quantum_espresso_output):
    forces = []
    in_section = False
    for line in quantum_espresso_output.split("\n"):
        if "Forces acting on atoms (cartesian axes, Ry/au):" in line:
            in_section = True
            continue
        if in_section:
            if "The non-local contrib." in line or "Total force =" in line or not line.strip():
                if len(forces) > 0:
                    break
                continue
            if "force =" in line:
                parts = line.split("force =")
                if len(parts) == 2:
                    coordinates = [float(value) for value in parts[1].split()]
                    forces.append(coordinates)
    return forces


def extract_quantum_espresso_stress(quantum_espresso_output):
    contributions = {}
    lines = quantum_espresso_output.split("\n")
    for index, line in enumerate(lines):
        if "total   stress" in line and "P=" in line:
            try:
                pressure_value = float(line.split("P=")[1].strip())
                contributions["total_pressure"] = pressure_value
            except (IndexError, ValueError):
                pass

        if "stress (kbar)" in line:
            parts = line.split("stress (kbar)")
            name = parts[0].strip()
            try:
                diagonal_element_1 = float(parts[1].split()[0])
                diagonal_element_2 = float(lines[index + 1].split()[1])
                diagonal_element_3 = float(lines[index + 2].split()[2])
                average_value = (diagonal_element_1 + diagonal_element_2 + diagonal_element_3) / 3.0
                name_map = {
                    "kinetic": "kinetic",
                    "local": "local PP",
                    "nonloc.": "nonlocal",
                    "hartree": "Hartree",
                    "exc-cor": "XC",
                    "corecor": "core corr",
                    "ewald": "Ewald",
                    "augment": "augment",
                }
                mapped_name = name_map.get(name, name)
                contributions[mapped_name] = round(average_value, 2)
            except (IndexError, ValueError):
                pass
    return contributions


# ── Config overlay ───────────────────────────────────────────────────


def deep_merge(base, overlay):
    """Deep-merge overlay dict into base dict. Returns a new dict.

    Keys set to None in the overlay are removed from the result.
    """
    result = copy.deepcopy(base)
    for key, value in overlay.items():
        if value is None:
            result.pop(key, None)
        elif key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def resolve_relative_paths(config, base_directory):
    """Resolve relative paths in a q3 YAML config to absolute paths.

    Handles:
      - config["material"]: path to material JSON file.
      - config["method"]["pseudopotentials"]: dict of element -> PP file path.

    Args:
        config: Parsed YAML config dict (modified in-place).
        base_directory: Directory that relative paths are resolved against.

    Returns:
        The modified config dict.
    """
    base_directory = Path(base_directory)

    # material may be an inline definition (dict) instead of a file path.
    if isinstance(config.get("material"), str):
        material_path = Path(config["material"])
        if not material_path.is_absolute():
            config["material"] = str((base_directory / material_path).resolve())

    pseudopotentials = config.get("method", {}).get("pseudopotentials", {})
    if isinstance(pseudopotentials, dict):
        for element, pseudopotential_path in pseudopotentials.items():
            pseudopotential_path = Path(pseudopotential_path)
            if not pseudopotential_path.is_absolute():
                pseudopotentials[element] = str((base_directory / pseudopotential_path).resolve())

    return config


def apply_config_overlay(base_yaml_path, overlay, output_directory):
    """Load a base YAML, deep-merge an overlay, write to output_directory.

    Resolves relative paths to absolute paths.

    Args:
        base_yaml_path: Path to the base YAML file in examples/.
        overlay: Dict of overrides to deep-merge on top.
        output_directory: Directory where the merged YAML will be written.

    Returns:
        Path to the merged YAML file.
    """
    base_yaml_path = Path(base_yaml_path)
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)

    with open(base_yaml_path) as file_handle:
        config = yaml.safe_load(file_handle)

    if overlay:
        config = deep_merge(config, overlay)

    resolve_relative_paths(config, base_yaml_path.parent)

    # Ensure outdir points to the output directory.
    config.setdefault("control", {})["outdir"] = str(output_directory / "output")

    merged_path = output_directory / base_yaml_path.name
    with open(merged_path, "w") as file_handle:
        yaml.dump(config, file_handle, default_flow_style=False)

    return merged_path


# ── Subprocess runner ────────────────────────────────────────────────


def run_q3(q3_command, input_path, working_directory=None, timeout=300):
    """Run the q3 binary with the given input file.

    Args:
        q3_command: List of command tokens (e.g., ["./build/q3"]).
        input_path: Path to the YAML input file.
        working_directory: Working directory for the subprocess.
        timeout: Maximum execution time in seconds.

    Returns:
        subprocess.CompletedProcess result.
    """
    if working_directory is None:
        working_directory = Path(input_path).parent

    return subprocess.run(
        list(q3_command) + [str(input_path)],
        cwd=str(working_directory),
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def run_overlay_and_read_energy(
    command, base_yaml, overlay, prefix, run_directory, extra_args=(), timeout=600
):
    """Run a q3 binary on base_yaml (+overlay) in run_directory; return E in Ry.

    The merged input is written into run_directory (with paths resolved and
    outdir redirected there), so concurrent runs never clobber each other or
    the committed example outputs. extra_args are appended after the input
    file (the GPU binaries expect ``q3_gpu <input_file> [flags]``).
    """
    merged_input = apply_config_overlay(base_yaml, overlay, run_directory)
    result = subprocess.run(
        list(command) + [str(merged_input)] + list(extra_args),
        cwd=str(run_directory),
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    assert result.returncode == 0, (
        f"{' '.join(command)} exited with code {result.returncode}\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    results = load_results_json(Path(run_directory) / "output", prefix)
    return float(results.properties.total_energy.value)


def run_overlay_and_read_stdout(
    command, base_yaml, overlay, run_directory, extra_args=(), timeout=600
):
    """Run a q3 binary on base_yaml (+overlay) in run_directory; return raw stdout.

    Same overlay/isolation semantics as run_overlay_and_read_energy(), for tests
    that need to parse something other than total_energy_ry from stdout (e.g.
    the stress tensor, which is not part of the JSON results file).
    """
    merged_input = apply_config_overlay(base_yaml, overlay, run_directory)
    result = subprocess.run(
        list(command) + [str(merged_input)] + list(extra_args),
        cwd=str(run_directory),
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    assert result.returncode == 0, (
        f"{' '.join(command)} exited with code {result.returncode}\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    return result.stdout


def run_q3_and_load_results(q3_command, input_path, prefix, working_directory=None, timeout=300):
    """Run q3 and load the JSON results file."""
    result = run_q3(q3_command, input_path, working_directory, timeout)
    assert result.returncode == 0, (
        f"q3 exited with code {result.returncode}\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )

    output_directory = Path(working_directory or Path(input_path).parent) / "output"
    if not output_directory.exists():
        output_directory = Path(input_path).parent / "output"

    results = load_results_json(output_directory, prefix)
    return results, output_directory


# ── Result loading ───────────────────────────────────────────────────


@functools.cache
def repository_schemas():
    """``scripts/repo/schemas.py``, or ``None`` when mat3ra-esse, which it reads, is not installed."""
    try:
        if importlib.util.find_spec("mat3ra.esse") is None:
            return None
    except ModuleNotFoundError:  # no mat3ra namespace at all
        return None
    spec = importlib.util.spec_from_file_location(
        "schemas", Q3_ROOT / "scripts" / "repo" / "schemas.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@functools.cache
def results_schema():
    """``schemas/q3/results.json`` with every ``$ref`` inlined, ESSE's included."""
    return repository_schemas().resolved("results.json")


def results_schema_errors(document):
    """Every way *document* breaks the results schema, as ``path: message`` lines.

    Empty when mat3ra-esse, which supplies the ESSE schemas it references, is
    not installed: nothing can be checked then, and the schema tests skip.
    """
    if repository_schemas() is None:
        return []
    validator = jsonschema.Draft7Validator(results_schema())
    return [
        f"/{'/'.join(str(step) for step in error.absolute_path)}: {error.message}"
        for error in validator.iter_errors(document)
    ]


def load_results_json(output_directory, prefix):
    """Load the JSON results file produced by q3, failing if it breaks its schema.

    Every test that reads a run's numbers comes through here, so validating
    here checks each file a test suite produces -- every system, flavor and
    option it exercises -- against ``schemas/q3/results.json``, not only the
    two runs ``test_results_schema.py`` makes for that purpose.

    Args:
        output_directory: The output directory (value of control.outdir).
        prefix: The prefix used in control.prefix.

    Returns:
        The file, as :class:`q3.io.results.classes.Results`.
    """
    path = Path(output_directory) / f"{prefix}_results.json"
    if not path.exists():
        pytest.fail(f"Results file not found: {path}")
    document = json.loads(path.read_text())
    errors = results_schema_errors(document)
    if errors:
        pytest.fail(f"{path} does not match schemas/q3/results.json:\n" + "\n".join(errors[:20]))
    return Results.model_validate(document)


def energy_contribution(results, name):
    """The energy contribution *name* of results.json's ``energy_contributions``, in Ry."""
    return getattr(results.properties.energy_contributions, name).value


def atomic_forces(results):
    """The ESSE ``atomic_forces`` record's vectors in Ry/bohr, one per atom in id order."""
    return _vectors(results.properties.atomic_forces)


def stress_tensor(results):
    """The ESSE ``stress_tensor`` record's 3x3 tensor in kbar."""
    return _tensor(results.properties.stress_tensor)


def _vectors(record):
    return [atom.value for atom in record.values]


def _tensor(record):
    return [row.root for row in record.value]


def stress_contributions(results):
    """results.json's ``stress_contributions`` as ``{name: 3x3 tensor in kbar}``."""
    contributions = results.properties.stress_contributions or {}
    return {name: _tensor(tensor) for name, tensor in contributions.items()}


def force_contributions(results):
    """results.json's ``force_contributions`` as ``{name: per-atom vectors in Ry/bohr}``."""
    contributions = results.properties.force_contributions or {}
    return {name: _vectors(forces) for name, forces in contributions.items()}


def _eigenvalue_record(results):
    """The eigenvalues_at_kpoints record of a run's results, or of an interpolation results file."""
    if isinstance(results, dict):
        interpolated = results.get("calculation") == "interpolation"
        results = (InterpolationResults if interpolated else Results).model_validate(results)
    return results.properties.eigenvalues_at_kpoints


def _eigenvalue_arrays(results):
    """The eigenvalue arrays of a run's results, or of an interpolation results file."""
    return eigenvalue_arrays(_eigenvalue_record(results))


def eigenvalues_ry(results):
    """results.json's eigenvalues in Ry, indexed [k-point state][spin channel][band]."""
    return _eigenvalue_arrays(results)[0]


def occupations(results):
    """results.json's occupations, indexed as eigenvalues_ry(); None when it recorded none."""
    return _eigenvalue_arrays(results)[1]


def kpoint_weights(results):
    """results.json's k-point weights, one per k-point state, summing to 2.0."""
    return _eigenvalue_arrays(results)[2]


def evaluated_kpoints(results):
    """The crystal coordinates of each k-point a results file lists eigenvalues at.

    Both list them in ESSE's ``eigenvalues_at_kpoints`` record: an engine
    results file among its ``properties.records``, an interpolation results
    file under ``properties``.
    """
    return [entry.kpoint for entry in _eigenvalue_record(results).values]


def density_of_states_ry(density):
    """An ESSE density_of_states record's (energies in Ry, density) as numpy arrays."""
    record = DensityOfStatesPropertySchema.model_validate(density)
    (values,) = record.yDataSeries
    return np.array(record.xDataArray), np.array(values)


def pressure_from_tensor(tensor_3x3):
    """The scalar pressure (trace/3) of a 3x3 stress tensor, in whatever units it's in.

    Q3's stdout report prints exactly this quantity per contribution (the
    "(P=...)" part of each stress-tensor line) and for the total ("Pressure =
    ... kbar"); results.json carries the full tensors (the ESSE stress_tensor
    record, stress_contributions[name]), and a scalar only for the total.
    This is the shared derivation for tests migrated off stdout parsing --
    see total_pressure_kbar() for the top-level-stress convenience wrapper.
    """
    return (tensor_3x3[0][0] + tensor_3x3[1][1] + tensor_3x3[2][2]) / 3.0


def total_pressure_kbar(results):
    """The scalar total pressure (trace/3) of the results' ESSE stress_tensor record."""
    return pressure_from_tensor(stress_tensor(results))


def birch_murnaghan_pressure_gpa(volume_angstrom3, equation_of_state):
    """P(V) in GPa, compression positive, of a ``run_equation_of_state_workflow`` fit.

    The textbook third-order form (Birch, Phys. Rev. 71, 809 (1947)), written
    out here rather than borrowed from the library that did the fitting, so a
    test comparing it with Q3's stress tensor checks the fit independently:

        P(V) = (3/2) B0 [x^(7/3) - x^(5/3)] {1 + (3/4)(B0' - 4)[x^(2/3) - 1]},  x = V0/V
    """
    ratio = equation_of_state["volume"]["value"] / volume_angstrom3
    return (
        1.5
        * equation_of_state["bulk_modulus_gpa"]
        * (ratio ** (7.0 / 3.0) - ratio ** (5.0 / 3.0))
        * (
            1.0
            + 0.75
            * (equation_of_state["bulk_modulus_pressure_derivative"] - 4.0)
            * (ratio ** (2.0 / 3.0) - 1.0)
        )
    )


# ── QE input generation (shared by test_qe_energy_comparison.py and ────
# ── test_qe_projwfc_comparison.py) ──────────────────────────────────────

# QE ibrav for each Q3 YAML lattice "type" — must match the bravais_lattice_index
# mapping in src/io/input/parser/yaml.hpp exactly (both codes must resolve the
# same conventional cell from the same primitive-cell JSON).
IBRAV_BY_LATTICE_TYPE = {"SC": 1, "CUB": 1, "FCC": 2, "BCC": 3}

# Q3's yaml.hpp stores "a" as the PRIMITIVE lattice vector length for FCC/BCC
# (converting to the conventional celldm(1) internally); SC/CUB has no primitive
# vs conventional distinction. These are the inverse of that conversion.
_CONVENTIONAL_FROM_PRIMITIVE = {
    "FCC": math.sqrt(2.0),
    "BCC": 2.0 / math.sqrt(3.0),
    "SC": 1.0,
    "CUB": 1.0,
}

# Atomic masses (amu) for QE's ATOMIC_SPECIES card. AFM sublattice labels
# (e.g. Cr1/Cr2) are stripped to their element symbol for the lookup, matching
# Q3's own PeriodicTable::atomic_number_by_element_symbol convention.
ATOMIC_MASSES = {
    "Si": 28.085,
    "Cu": 63.546,
    "Fe": 55.845,
    "Ni": 58.6934,
    "Cr": 51.9961,
    "C": 12.011,
}


def compute_celldm1(material_name):
    """Compute celldm(1) in Bohr from Q3's material JSON."""
    with open(MATERIALS_DIR / f"{material_name}.json") as file_handle:
        material = json.load(file_handle)
    lattice = material["lattice"]
    a_primitive_angstrom = lattice["a"]
    lattice_type = lattice["type"]
    a_conventional_angstrom = a_primitive_angstrom * _CONVENTIONAL_FROM_PRIMITIVE[lattice_type]
    return a_conventional_angstrom / BOHR_RADIUS_ANGSTROM


def compute_ibrav(material_name):
    """Look up the QE ibrav matching Q3's lattice "type" for this material."""
    with open(MATERIALS_DIR / f"{material_name}.json") as file_handle:
        material = json.load(file_handle)
    return IBRAV_BY_LATTICE_TYPE[material["lattice"]["type"]]


def read_atomic_positions(material_name):
    """Read atomic positions from Q3's material JSON."""
    with open(MATERIALS_DIR / f"{material_name}.json") as file_handle:
        material = json.load(file_handle)
    basis = material["basis"]
    elements = [entry["value"] for entry in basis["elements"]]
    coordinates = [entry["value"] for entry in basis["coordinates"]]
    unique_species = []
    for element in elements:
        if element not in unique_species:
            unique_species.append(element)
    return elements, coordinates, unique_species


def _mass_for_species(species_label):
    stripped = species_label.rstrip("0123456789")
    return ATOMIC_MASSES.get(stripped, 1.0)


def generate_qe_input(
    case_name,
    material_name,
    pp_type,
    pp_filename,
    ecutwfc,
    ecutrho,
    kgrid,
    kshift,
    number_of_bands=None,
    smearing=None,
    degauss=0.0,
    dispersion=None,
    number_of_spin_components=1,
    starting_magnetization=None,
    mixing_beta=None,
    mixing_mode=None,
    output_directory="/tmp",
    calculation="scf",
    startingpot=None,
    startingwfc=None,
    tot_charge=0.0,
    coordinate_overrides=None,
    convergence_threshold=1.0e-8,
    diagonalization=None,
):
    """Generate a QE pw.x input file matching Q3's parameters.

    starting_magnetization, if given, is a dict of species label -> value
    (matching Q3's YAML method.scf.starting_magnetization map); QE indices
    are assigned in unique_species order, exactly mirroring how Q3's own
    parser resolves species order from the material JSON.

    calculation: 'scf' (default) or 'nscf' — for 'nscf', pass
    startingpot='file' (written into &ELECTRONS, where QE actually declares
    it, not &CONTROL). startingwfc='atomic' is QE's own default and does not
    need to be set explicitly. outdir/prefix must match the prior 'scf'
    run's so QE finds its charge-density/potential files.

    coordinate_overrides: optional dict mapping a 0-based atom index (in the
    material JSON's own basis.coordinates order) to a replacement
    (x, y, z) fractional-coordinate tuple -- e.g. for finite-difference
    force calculations that displace one atom while keeping everything
    else (cell, other atoms, cutoffs, k-grid) identical to a named,
    already-defined material.

    diagonalization: forwarded to &ELECTRONS verbatim (e.g. 'cg'). Every
    PAW dataset tested aborts QE's default Davidson diagonalizer with
    "S matrix not positive definite" at every cutoff combination tried
    (30/120 through 80/960 Ry) -- callers building a PAW case must pass
    diagonalization='cg', which converges every case. Not defaulted to
    'cg' unconditionally because it would silently mask the same failure
    for NC/US datasets, where Davidson working is itself a signal.

    convergence_threshold: density-residual threshold written as QE's
    ``conv_thr``. Callers must pass the value from the matching Q3 input when it
    differs from the repository-wide 1e-8 default.
    """
    celldm1 = compute_celldm1(material_name)
    ibrav = compute_ibrav(material_name)
    elements, coordinates, unique_species = read_atomic_positions(material_name)
    if coordinate_overrides:
        coordinates = list(coordinates)
        for atom_index, replacement in coordinate_overrides.items():
            coordinates[atom_index] = replacement
    pp_dir = PP_DIR / pp_type
    outdir = os.path.join(output_directory, case_name)
    os.makedirs(outdir, exist_ok=True)

    lines = [
        "&CONTROL",
        f"  calculation = '{calculation}'",
        f"  outdir = '{outdir}/'",
        f"  prefix = '{case_name}'",
        f"  pseudo_dir = '{pp_dir}/'",
        "/",
        "&SYSTEM",
    ]
    lines.extend(
        [
            f"  ibrav = {ibrav}",
            f"  celldm(1) = {celldm1:.15f}",
            f"  nat = {len(elements)}",
            f"  ntyp = {len(unique_species)}",
            f"  ecutwfc = {ecutwfc}",
            f"  ecutrho = {ecutrho}",
            # QE's dft-d3 module resolves damping parameters by matching a plain
            # functional name (e.g. 'pbe') against its own table -- it cannot
            # parse the LibXC notation code used everywhere else in this file.
            # Both forms select the identical PBE functional; only the dft-d3
            # module cares about the difference.
            "  input_dft = 'pbe'" if dispersion == "d3bj" else f"  input_dft = '{LIBXC_PBE}'",
        ]
    )
    if tot_charge != 0.0:
        lines.append(f"  tot_charge = {tot_charge}")
    if number_of_bands is not None:
        lines.append(f"  nbnd = {number_of_bands}")
    if smearing:
        lines.extend(
            [
                "  occupations = 'smearing'",
                f"  smearing = '{smearing}'",
                f"  degauss = {degauss}",
            ]
        )
    if dispersion == "d3bj":
        lines.extend(
            [
                "  vdw_corr = 'dft-d3'",
                "  dftd3_version = 4",
                "  dftd3_threebody = .false.",
            ]
        )
    if number_of_spin_components == 2:
        lines.append("  nspin = 2")
        for species_index, species_label in enumerate(unique_species, start=1):
            value = (starting_magnetization or {}).get(species_label, 0.0)
            lines.append(f"  starting_magnetization({species_index}) = {value}")
    lines.extend(
        [
            "/",
            "&ELECTRONS",
            f"  conv_thr = {convergence_threshold}",
        ]
    )
    if diagonalization is not None:
        lines.append(f"  diagonalization = '{diagonalization}'")
    if mixing_beta is not None:
        lines.append(f"  mixing_beta = {mixing_beta}")
    if mixing_mode is not None:
        lines.append(f"  mixing_mode = '{mixing_mode}'")
    if startingpot is not None:
        lines.append(f"  startingpot = '{startingpot}'")
    if startingwfc is not None:
        lines.append(f"  startingwfc = '{startingwfc}'")
    lines.append("/")
    if calculation in ("relax", "vc-relax"):
        # ion_dynamics/cell_dynamics default to 'bfgs' for these calculation
        # types in pw.x's documented input defaults, so an empty &IONS block
        # is sufficient; only 'press' needs stating for &CELL since it does
        # not otherwise default to the zero-external-pressure target that
        # ASE's cell filters also use.
        lines.extend(["&IONS", "/"])
    if calculation == "vc-relax":
        lines.extend(["&CELL", "  press = 0.0", "/"])
    lines.append("ATOMIC_SPECIES")
    for species in unique_species:
        lines.append(f"  {species} {_mass_for_species(species)} {pp_filename}")
    lines.append("ATOMIC_POSITIONS crystal")
    for element, coord in zip(elements, coordinates):
        lines.append(f"  {element} {coord[0]:.10f} {coord[1]:.10f} {coord[2]:.10f}")
    lines.append("K_POINTS automatic")
    lines.append(f"  {kgrid[0]} {kgrid[1]} {kgrid[2]} {kshift[0]} {kshift[1]} {kshift[2]}")

    input_path = os.path.join(output_directory, f"{case_name}.in")
    with open(input_path, "w") as file_handle:
        file_handle.write("\n".join(lines) + "\n")
    return input_path


def run_qe(input_path):
    """Run QE pw.x and return the converged total energy in Ry."""
    return run_qe_full(input_path)[0]


def run_qe_vc_relax(input_path, timeout=1800):
    """Run QE pw.x for a 'vc-relax' calculation; return the final unit-cell
    volume in bohr^3.

    QE reprints "new unit-cell volume = ... a.u.^3" at every BFGS step and
    once more inside the "Begin/End final coordinates" block; the LAST
    occurrence in stdout is always the converged one, so no special marker
    parsing is needed. Raises if QE did not reach "End final coordinates"
    (relaxation did not converge within its own step limit).
    """
    result = subprocess.run(
        [QE_PW, "-in", input_path],
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if "End final coordinates" not in result.stdout:
        raise RuntimeError(
            f"QE vc-relax did not reach final coordinates.\n"
            f"stdout: {result.stdout[-2000:]}\nstderr: {result.stderr[-500:]}"
        )
    final_volume_bohr3 = None
    for line in result.stdout.split("\n"):
        stripped = line.strip()
        if stripped.startswith("new unit-cell volume"):
            final_volume_bohr3 = float(stripped.split("=")[1].split("a.u.")[0])
    if final_volume_bohr3 is None:
        raise RuntimeError(
            f"QE vc-relax: no unit-cell volume found.\nstdout: {result.stdout[-2000:]}"
        )
    return final_volume_bohr3


def run_qe_full(input_path, timeout=600):
    """Run QE pw.x; return (total_energy_ry, total_magnetization_muB_or_None).

    Magnetization is only present in the output for nspin=2 runs; None is
    returned for unpolarized (nspin=1) runs.
    """
    result = subprocess.run(
        [QE_PW, "-in", input_path],
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    total_energy = None
    total_magnetization = None
    for line in result.stdout.split("\n"):
        stripped = line.strip()
        if stripped.startswith("!") and "total energy" in stripped:
            total_energy = float(stripped.split("=")[1].split("Ry")[0])
        elif stripped.startswith("total magnetization"):
            # Last occurrence wins — QE prints this every iteration, and we
            # want the value from the final (converged) iteration.
            total_magnetization = float(stripped.split("=")[1].split("Bohr")[0])
    if total_energy is None:
        raise RuntimeError(
            f"QE did not converge or output total energy.\n"
            f"stdout: {result.stdout[-500:]}\nstderr: {result.stderr[-500:]}"
        )
    return total_energy, total_magnetization
