"""
No-U stress baseline: low symmetry, full-zone parity, and Gamma normalization.

The triclinic case is deliberately inversion-only and carries large shear
terms. It exposed an Ewald assembly defect: reciprocal off-diagonals were
written to the lower triangle, real-space terms to the upper triangle, then
the upper triangle overwrote the lower one. Q3 consequently matched QE on all
diagonals while each total shear differed by exactly the omitted reciprocal
Ewald shear. The same case now has three independent gates: reduced versus
full-zone invariance, all contribution tensors against QE, and the whole Q3
tensor against central derivatives of Q3's own energy.

The Gamma case compares the optimized half sphere with an explicit full sphere
at the identical single k-point. Before the fix every non-XC stress component
was wrong, by factors or offsets ranging from hundreds to thousands of kbar,
even though the energies agreed. Ewald energy and stress are held to roundoff;
every electronic contribution is checked separately so cancellations cannot
hide a missing +-G partner or a double-counted G=0 term.

Run:
    .venv/bin/pytest tests/validation/test_stress_baseline.py -v --q3-bin ./build/q3
"""

import json
import os
import subprocess
from pathlib import Path

import pytest
from q3.io.results.classes import Results

from helpers import stress_contributions, stress_tensor

REPOSITORY = Path(__file__).resolve().parents[2]
DATA_DIRECTORY = Path(__file__).resolve().parent / "data" / "stress_baseline"
TRICLINIC_MATERIAL = json.loads((DATA_DIRECTORY / "material_triclinic.json").read_text())
SILICON_MATERIAL = json.loads(
    (REPOSITORY / "examples" / "data" / "materials" / "silicon.json").read_text()
)
SILICON_PSEUDOPOTENTIAL = REPOSITORY / "examples" / "data" / "pseudopotentials" / "nc" / "Si.upf"
REFERENCE = json.loads((DATA_DIRECTORY / "reference.json").read_text())["inputs"]
REFERENCE_REDUCED = REFERENCE["scf_qe_reduced.in"]

FULL_ZONE_KPOINTS = (
    (-0.25, -0.25, -0.25, 0.25),
    (-0.25, -0.25, 0.25, 0.25),
    (-0.25, 0.25, -0.25, 0.25),
    (-0.25, 0.25, 0.25, 0.25),
    (0.25, -0.25, -0.25, 0.25),
    (0.25, -0.25, 0.25, 0.25),
    (0.25, 0.25, -0.25, 0.25),
    (0.25, 0.25, 0.25, 0.25),
)

STRAIN_COMPONENTS = ((0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2))
FINITE_STRAIN_STEP = 0.002
ANGSTROM_TO_BOHR = 1.0 / 0.529177249
RYDBERG_PER_BOHR_CUBED_TO_KILOBAR = 147105.076

REFERENCE_ENERGY_TOLERANCE_RYDBERG = 2.0e-6
REFERENCE_STRESS_TOLERANCE_KILOBAR = 0.03
ZONE_INVARIANCE_TOLERANCE_KILOBAR = 1.0e-7
GAMMA_CONTRIBUTION_TOLERANCE_KILOBAR = 0.02
GAMMA_EWAL_ROUNDOFF_TOLERANCE_KILOBAR = 1.0e-7
SELF_GRADIENT_TOLERANCE_KILOBAR = 0.02
EWALD_SELF_GRADIENT_TOLERANCE_KILOBAR = 0.005


def determinant(matrix):
    return (
        matrix[0][0] * (matrix[1][1] * matrix[2][2] - matrix[1][2] * matrix[2][1])
        - matrix[0][1] * (matrix[1][0] * matrix[2][2] - matrix[1][2] * matrix[2][0])
        + matrix[0][2] * (matrix[1][0] * matrix[2][1] - matrix[1][1] * matrix[2][0])
    )


def maximum_tensor_difference(first, second):
    return max(
        abs(first[row_index][column_index] - second[row_index][column_index])
        for row_index in range(3)
        for column_index in range(3)
    )


def deform_material(material, direction_a, direction_b, strain):
    deformed = json.loads(json.dumps(material))
    vectors = [material["lattice"]["vectors"][label] for label in ("a", "b", "c")]
    deformation = [[float(row == column) for column in range(3)] for row in range(3)]
    deformation[direction_a][direction_b] += strain
    for label, vector in zip(("a", "b", "c"), vectors):
        deformed["lattice"]["vectors"][label] = [
            sum(deformation[column][index] * vector[index] for index in range(3))
            for column in range(3)
        ]
    return deformed


def write_input(directory, material, label, sampling, wavefunction_cutoff, density_cutoff):
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "material.json").write_text(json.dumps(material, indent=4) + "\n")
    relative_pseudopotential = os.path.relpath(SILICON_PSEUDOPOTENTIAL, directory)
    if sampling == "reduced":
        sampling_block = """    kgrid:
      dimensions: [2, 2, 2]
      shifts: [1, 1, 1]
"""
    elif sampling == "full":
        kpoint_rows = "\n".join(
            "    - [" + ", ".join(str(value) for value in kpoint) + "]"
            for kpoint in FULL_ZONE_KPOINTS
        )
        sampling_block = f"""control_kpoints:
{kpoint_rows}
"""
    elif sampling == "gamma_half":
        sampling_block = """    kgrid:
      dimensions: [1, 1, 1]
      shifts: [0, 0, 0]
"""
    elif sampling == "gamma_full":
        sampling_block = "control_kpoints:\n    - [0.0, 0.0, 0.0, 2.0]\n"
    else:
        raise ValueError(sampling)

    control_kpoints = ""
    if sampling_block.startswith("control_kpoints:"):
        rows = sampling_block.removeprefix("control_kpoints:\n")
        control_kpoints = f"  kpoints_list:\n{rows}"
        sampling_block = ""

    input_path = directory / "input.yml"
    input_path.write_text(
        f"""material: ./material.json
model:
  type: dft
  subtype: gga
  functional: pbe
method:
  type: pseudopotential
  subtype: norm-conserving
  cutoffs:
    wavefunction: {wavefunction_cutoff}
    density: {density_cutoff}
  pseudopotentials:
    Si: {relative_pseudopotential}
  scf:
    convergence_threshold: 1.0e-12
    max_iterations: 200
    number_of_bands: 8
    mixing:
      mode: broyden
      beta: 0.3
{sampling_block}control:
  calculation: scf
  outdir: ./output
  prefix: {label}
  use_deterministic_fft_planning: true
{control_kpoints}"""
    )
    return input_path


def run_q3(q3_bin, directory, material, label, sampling, wavefunction_cutoff, density_cutoff):
    input_path = write_input(
        directory, material, label, sampling, wavefunction_cutoff, density_cutoff
    )
    completed = subprocess.run(
        [str(q3_bin), input_path.name],
        cwd=directory,
        capture_output=True,
        text=True,
        timeout=3600,
    )
    assert completed.returncode == 0, completed.stdout[-3000:]
    results = Results.read(directory / "output" / f"{label}_results.json")
    assert results.runtime_log.converged, completed.stdout[-3000:]
    assert results.properties.stress_contributions is not None
    print(
        f"\n  completed {label}: E={results.properties.total_energy.value:.12f} Ry",
        flush=True,
    )
    return {"output": completed.stdout, "results": results}


@pytest.fixture(scope="module")
def low_symmetry_runs(q3_bin, tmp_path_factory):
    return {
        sampling: run_q3(
            q3_bin,
            tmp_path_factory.mktemp(f"stress_{sampling}"),
            TRICLINIC_MATERIAL,
            f"triclinic_{sampling}",
            sampling,
            60.0,
            240.0,
        )
        for sampling in ("reduced", "full")
    }


@pytest.fixture(scope="module")
def finite_strain_runs(q3_bin, tmp_path_factory, low_symmetry_runs):
    runs = {}
    for direction_a, direction_b in STRAIN_COMPONENTS:
        runs[(direction_a, direction_b)] = {}
        for sign in (-1, 1):
            label = f"strain_{direction_a}{direction_b}_{sign:+d}"
            material = deform_material(
                TRICLINIC_MATERIAL,
                direction_a,
                direction_b,
                sign * FINITE_STRAIN_STEP,
            )
            runs[(direction_a, direction_b)][sign] = run_q3(
                q3_bin,
                tmp_path_factory.mktemp(label),
                material,
                label,
                "reduced",
                60.0,
                240.0,
            )["results"]
    return {"baseline": low_symmetry_runs["reduced"]["results"], "strains": runs}


@pytest.fixture(scope="module")
def gamma_runs(q3_bin, tmp_path_factory):
    return {
        sampling: run_q3(
            q3_bin,
            tmp_path_factory.mktemp(f"stress_{sampling}"),
            SILICON_MATERIAL,
            f"silicon_{sampling}",
            sampling,
            48.0,
            192.0,
        )
        for sampling in ("gamma_half", "gamma_full")
    }


def test_low_symmetry_reduction_matches_the_explicit_full_zone(low_symmetry_runs):
    reduced = low_symmetry_runs["reduced"]
    full = low_symmetry_runs["full"]
    assert "8 -> 4 k-points" in reduced["output"]
    assert (
        abs(
            reduced["results"].properties.total_energy.value
            - full["results"].properties.total_energy.value
        )
        < 1.0e-10
    )
    assert (
        maximum_tensor_difference(
            stress_tensor(reduced["results"]),
            stress_tensor(full["results"]),
        )
        < ZONE_INVARIANCE_TOLERANCE_KILOBAR
    )
    for contribution, reduced_tensor in stress_contributions(reduced["results"]).items():
        assert (
            maximum_tensor_difference(
                reduced_tensor, stress_contributions(full["results"])[contribution]
            )
            < ZONE_INVARIANCE_TOLERANCE_KILOBAR
        )


def test_low_symmetry_stress_and_every_component_match_quantum_espresso(low_symmetry_runs):
    measured = low_symmetry_runs["reduced"]["results"]
    assert abs(
        measured.properties.total_energy.value - REFERENCE_REDUCED["total_energy_rydberg"]
    ) < (REFERENCE_ENERGY_TOLERANCE_RYDBERG)
    assert (
        maximum_tensor_difference(
            stress_tensor(measured),
            REFERENCE_REDUCED["stress_kilobar"],
        )
        < REFERENCE_STRESS_TOLERANCE_KILOBAR
    )
    for contribution, expected_tensor in REFERENCE_REDUCED["stress_contributions_kilobar"].items():
        assert (
            maximum_tensor_difference(stress_contributions(measured)[contribution], expected_tensor)
            < REFERENCE_STRESS_TOLERANCE_KILOBAR
        )
    assert abs(REFERENCE_REDUCED["stress_contributions_kilobar"]["ewald"][1][2]) > 50.0


def test_low_symmetry_tensor_is_q3s_own_energy_gradient(finite_strain_runs):
    vectors = [TRICLINIC_MATERIAL["lattice"]["vectors"][label] for label in ("a", "b", "c")]
    volume_bohr_cubed = abs(determinant(vectors)) * ANGSTROM_TO_BOHR**3
    baseline = finite_strain_runs["baseline"]
    for direction_a, direction_b in STRAIN_COMPONENTS:
        pair = finite_strain_runs["strains"][(direction_a, direction_b)]
        numerical_stress = (
            -(pair[1].properties.total_energy.value - pair[-1].properties.total_energy.value)
            / (2.0 * FINITE_STRAIN_STEP * volume_bohr_cubed)
            * (RYDBERG_PER_BOHR_CUBED_TO_KILOBAR)
        )
        analytic_stress = stress_tensor(baseline)[direction_a][direction_b]
        assert abs(analytic_stress - numerical_stress) < SELF_GRADIENT_TOLERANCE_KILOBAR


def test_ewald_tensor_is_the_gradient_of_the_ewald_energy(finite_strain_runs):
    vectors = [TRICLINIC_MATERIAL["lattice"]["vectors"][label] for label in ("a", "b", "c")]
    volume_bohr_cubed = abs(determinant(vectors)) * ANGSTROM_TO_BOHR**3
    baseline = finite_strain_runs["baseline"]
    for direction_a, direction_b in STRAIN_COMPONENTS:
        pair = finite_strain_runs["strains"][(direction_a, direction_b)]
        numerical_stress = (
            -(
                pair[1].properties.total_energy_contributions.ewald.value
                - pair[-1].properties.total_energy_contributions.ewald.value
            )
            / (2.0 * FINITE_STRAIN_STEP * volume_bohr_cubed)
            * (RYDBERG_PER_BOHR_CUBED_TO_KILOBAR)
        )
        analytic_stress = stress_contributions(baseline)["ewald"][direction_a][direction_b]
        assert abs(analytic_stress - numerical_stress) < EWALD_SELF_GRADIENT_TOLERANCE_KILOBAR


def test_gamma_half_sphere_matches_the_explicit_full_sphere(gamma_runs):
    half = gamma_runs["gamma_half"]["results"]
    full = gamma_runs["gamma_full"]["results"]
    assert "half-sphere" in gamma_runs["gamma_half"]["output"]
    assert "half-sphere" not in gamma_runs["gamma_full"]["output"]
    assert abs(half.properties.total_energy.value - full.properties.total_energy.value) < 1.0e-8
    assert (
        maximum_tensor_difference(
            stress_tensor(half),
            stress_tensor(full),
        )
        < GAMMA_CONTRIBUTION_TOLERANCE_KILOBAR
    )
    for contribution, half_tensor in stress_contributions(half).items():
        assert (
            maximum_tensor_difference(half_tensor, stress_contributions(full)[contribution])
            < GAMMA_CONTRIBUTION_TOLERANCE_KILOBAR
        )
    assert (
        min(
            abs(stress_contributions(half)[name][0][0])
            for name in (
                "kinetic",
                "hartree",
                "exchange_correlation",
                "ewald",
                "local_pseudopotential",
                "nonlocal_pseudopotential",
                "core_correction",
            )
        )
        > 100.0
    )


def test_gamma_ewald_energy_and_stress_are_storage_invariant(gamma_runs):
    half = gamma_runs["gamma_half"]["results"]
    full = gamma_runs["gamma_full"]["results"]
    assert (
        abs(
            half.properties.total_energy_contributions.ewald.value
            - full.properties.total_energy_contributions.ewald.value
        )
        < 1.0e-10
    )
    assert (
        maximum_tensor_difference(
            stress_contributions(half)["ewald"],
            stress_contributions(full)["ewald"],
        )
        < GAMMA_EWAL_ROUNDOFF_TOLERANCE_KILOBAR
    )
