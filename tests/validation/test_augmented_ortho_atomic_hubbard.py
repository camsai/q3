"""Ultrasoft/PAW ortho-atomic DFT+U energy and derivative validation.

The generalized projector is built from the complete cell-wide
``<phi_i|S|phi_j>`` metric. This module checks the three independent contracts
that definition creates:

* ultrasoft energy, Hubbard energy, and occupations against LibXC QE;
* PAW force and full stress tensor against Q3's own converged energy gradient;
* PAW Gamma half-sphere parity with an explicit full-sphere Gamma list.

The PAW stress tolerance was selected after a two-step study at 0.002 and
0.001 strain. The maximum errors were 0.0313 and 0.0455 kbar respectively.

Run:
    .venv/bin/pytest tests/validation/test_augmented_ortho_atomic_hubbard.py -v \
        --q3-bin ./build/q3
"""

import json
import math
import os
import re
import subprocess
from pathlib import Path

import pytest

from helpers import QE_AVAILABLE, QE_PW

REPOSITORY = Path(__file__).resolve().parents[2]
ULTRASOFT_REFERENCE_INPUT = (
    REPOSITORY
    / "examples"
    / "total_energy--semiconductor--silicon"
    / "reference"
    / "us_hubbard"
    / "scf_qe.in"
)
ULTRASOFT_PSEUDOPOTENTIAL_DIRECTORY = REPOSITORY / "examples" / "data" / "pseudopotentials" / "us"
PAW_PSEUDOPOTENTIAL_DIRECTORY = REPOSITORY / "examples" / "data" / "pseudopotentials" / "paw"

LATTICE_VECTORS_BOHR = (
    (-4.119180301160476, 0.0, 4.119180301160476),
    (0.0, 4.119180301160476, 4.119180301160476),
    (-4.119180301160476, 4.119180301160476, 0.0),
)
ATOMIC_POSITIONS_CRYSTAL = ((0.0, 0.0, 0.0), (0.27, 0.263, 0.257))
INDEPENDENT_STRAIN_COMPONENTS = ((0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2))
FINITE_STRAIN_STEP = 0.002
FINITE_POSITION_STEP = 0.001
RYDBERG_PER_BOHR_CUBED_TO_KILOBAR = 147105.076

REFERENCE_ENERGY_TOLERANCE_RYDBERG = 1.0e-6
REFERENCE_HUBBARD_ENERGY_TOLERANCE_RYDBERG = 1.0e-6
REFERENCE_OCCUPATION_TOLERANCE = 1.0e-4
FORCE_GRADIENT_TOLERANCE_RYDBERG_PER_BOHR = 1.0e-4
STRESS_GRADIENT_TOLERANCE_KILOBAR = 0.1
GAMMA_ENERGY_TOLERANCE_RYDBERG = 1.0e-8
GAMMA_FORCE_TOLERANCE_RYDBERG_PER_BOHR = 2.0e-6
GAMMA_STRESS_TOLERANCE_KILOBAR = 0.05


def determinant(matrix):
    return (
        matrix[0][0] * (matrix[1][1] * matrix[2][2] - matrix[1][2] * matrix[2][1])
        - matrix[0][1] * (matrix[1][0] * matrix[2][2] - matrix[1][2] * matrix[2][0])
        + matrix[0][2] * (matrix[1][0] * matrix[2][1] - matrix[1][1] * matrix[2][0])
    )


def deform_lattice(direction_a, direction_b, strain):
    deformation = [[float(row == column) for column in range(3)] for row in range(3)]
    deformation[direction_a][direction_b] += strain
    return [
        [
            sum(deformation[column][coordinate] * vector[coordinate] for coordinate in range(3))
            for column in range(3)
        ]
        for vector in LATTICE_VECTORS_BOHR
    ]


def write_paw_input(directory, lattice_vectors, atomic_positions, label, sampling="mesh"):
    material = {
        "lattice": {
            "vectors": dict(zip(("a", "b", "c"), lattice_vectors)),
            "units": {"length": "bohr", "angle": "degree"},
        },
        "basis": {
            "units": "crystal",
            "elements": [{"id": 0, "value": "Si"}, {"id": 1, "value": "C"}],
            "coordinates": [
                {"id": index, "value": position} for index, position in enumerate(atomic_positions)
            ],
        },
    }
    (directory / "material.json").write_text(json.dumps(material, indent=4) + "\n")
    relative_silicon_pseudopotential = os.path.relpath(
        PAW_PSEUDOPOTENTIAL_DIRECTORY / "Si.upf", directory
    )
    relative_carbon_pseudopotential = os.path.relpath(
        PAW_PSEUDOPOTENTIAL_DIRECTORY / "C.upf", directory
    )
    if sampling == "mesh":
        sampling_block = "    kgrid: {dimensions: [2, 2, 2], shifts: [0, 0, 0]}\n"
        explicit_kpoint_block = ""
    elif sampling == "gamma_half":
        sampling_block = "    kgrid: {dimensions: [1, 1, 1], shifts: [0, 0, 0]}\n"
        explicit_kpoint_block = ""
    elif sampling == "gamma_full":
        sampling_block = ""
        explicit_kpoint_block = "  kpoints_list:\n    - [0.0, 0.0, 0.0, 2.0]\n"
    else:
        raise ValueError(sampling)

    input_path = directory / "input.yml"
    input_path.write_text(
        f"""material: ./material.json
model: {{type: dft, subtype: gga, functional: pbe}}
method:
  type: pseudopotential
  subtype: paw
  cutoffs: {{wavefunction: 45.0, density: 180.0}}
  pseudopotentials:
    Si: {relative_silicon_pseudopotential}
    C: {relative_carbon_pseudopotential}
  scf:
    convergence_threshold: 1.0e-12
    max_iterations: 300
    mixing: {{mode: broyden, beta: 0.3}}
{sampling_block}    number_of_bands: 8
    hubbard_projectors: ortho-atomic
    hubbard:
      - {{species: Si, orbital_label: p, u_ev: 3.0}}
control:
  disable_symmetry: true
  calculation: scf
  outdir: ./output
  prefix: {label}
  use_deterministic_fft_planning: true
{explicit_kpoint_block}"""
    )
    return input_path


def run_paw_q3(q3_bin, directory, lattice_vectors, atomic_positions, label, sampling="mesh"):
    input_path = write_paw_input(
        directory, lattice_vectors, atomic_positions, label, sampling=sampling
    )
    completed = subprocess.run(
        [str(q3_bin), input_path.name],
        cwd=directory,
        capture_output=True,
        text=True,
        timeout=3600,
    )
    assert completed.returncode == 0, completed.stdout[-4000:]
    result = json.loads((directory / "output" / f"{label}_results.json").read_text())
    assert result["converged"], completed.stdout[-4000:]
    print(f"\n  completed {label}: E={result['total_energy_ry']:.12f} Ry", flush=True)
    return result


@pytest.fixture(scope="module")
def ultrasoft_reference_runs(q3_bin, tmp_path_factory):
    if not QE_AVAILABLE:
        pytest.skip(f"LibXC QE binary not found at {QE_PW}")

    input_text = ULTRASOFT_REFERENCE_INPUT.read_text()
    input_text = input_text.replace(
        "pseudo_dir = '../../../data/pseudopotentials/us/'",
        f"pseudo_dir = '{ULTRASOFT_PSEUDOPOTENTIAL_DIRECTORY}/'",
    ).replace("HUBBARD (atomic)", "HUBBARD (ortho-atomic)")
    outputs = {}
    for engine, command in (
        ("q3", [str(q3_bin), "scf.in"]),
        ("qe", [QE_PW, "-in", "scf.in"]),
    ):
        directory = tmp_path_factory.mktemp(f"augmented_ortho_{engine}")
        (directory / "scf.in").write_text(input_text)
        completed = subprocess.run(
            command,
            cwd=directory,
            capture_output=True,
            text=True,
            timeout=3600,
        )
        assert completed.returncode == 0, completed.stdout[-4000:] + completed.stderr[-2000:]
        outputs[engine] = completed.stdout
    return outputs


@pytest.fixture(scope="module")
def paw_derivative_runs(q3_bin, tmp_path_factory):
    baseline = run_paw_q3(
        q3_bin,
        tmp_path_factory.mktemp("augmented_ortho_baseline"),
        LATTICE_VECTORS_BOHR,
        ATOMIC_POSITIONS_CRYSTAL,
        "baseline",
    )

    position_displacements = {}
    for sign, sign_label in ((-1, "minus"), (1, "plus")):
        positions = [list(position) for position in ATOMIC_POSITIONS_CRYSTAL]
        positions[1][0] += sign * FINITE_POSITION_STEP
        position_displacements[sign] = run_paw_q3(
            q3_bin,
            tmp_path_factory.mktemp(f"augmented_ortho_position_{sign_label}"),
            LATTICE_VECTORS_BOHR,
            positions,
            f"position_{sign_label}",
        )

    strained = {}
    for direction_a, direction_b in INDEPENDENT_STRAIN_COMPONENTS:
        strained[(direction_a, direction_b)] = {}
        for sign, sign_label in ((-1, "minus"), (1, "plus")):
            label = f"strain_{direction_a}{direction_b}_{sign_label}"
            strained[(direction_a, direction_b)][sign] = run_paw_q3(
                q3_bin,
                tmp_path_factory.mktemp(f"augmented_ortho_{label}"),
                deform_lattice(direction_a, direction_b, sign * FINITE_STRAIN_STEP),
                ATOMIC_POSITIONS_CRYSTAL,
                label,
            )
    return {"baseline": baseline, "position": position_displacements, "strain": strained}


@pytest.fixture(scope="module")
def paw_gamma_runs(q3_bin, tmp_path_factory):
    return {
        sampling: run_paw_q3(
            q3_bin,
            tmp_path_factory.mktemp(f"augmented_ortho_{sampling}"),
            LATTICE_VECTORS_BOHR,
            ATOMIC_POSITIONS_CRYSTAL,
            sampling,
            sampling=sampling,
        )
        for sampling in ("gamma_half", "gamma_full")
    }


def test_ultrasoft_energy_and_occupations_match_libxc_qe(ultrasoft_reference_runs):
    q3_output = ultrasoft_reference_runs["q3"]
    qe_output = ultrasoft_reference_runs["qe"]
    q3_energy = float(re.findall(r"total energy\s*=\s*(-?\d+\.\d+) Ry", q3_output)[-1])
    qe_energy = float(re.findall(r"!\s+total energy\s*=\s*(-?\d+\.\d+) Ry", qe_output)[-1])
    q3_hubbard_energy = float(re.findall(r"DFT\+U \(Hubbard\)\s*=\s*(-?\d+\.\d+)", q3_output)[-1])
    qe_hubbard_energy = float(re.findall(r"Hubbard energy\s*=\s*(-?\d+\.\d+) Ry", qe_output)[-1])
    q3_occupations = [
        float(value)
        for value in re.findall(r"DFT\+U: atom \d+ Tr\[n\] = (\d+\.\d+)", q3_output)[-2:]
    ]
    qe_occupations = [
        float(value) for value in re.findall(r"Tr\[ns\(\s*\d+\)\] =\s*(\d+\.\d+)", qe_output)[-2:]
    ]

    assert abs(q3_energy - qe_energy) < REFERENCE_ENERGY_TOLERANCE_RYDBERG
    assert abs(q3_hubbard_energy - qe_hubbard_energy) < REFERENCE_HUBBARD_ENERGY_TOLERANCE_RYDBERG
    assert max(abs(q3 - qe) for q3, qe in zip(q3_occupations, qe_occupations)) < (
        REFERENCE_OCCUPATION_TOLERANCE
    )


def test_paw_ortho_atomic_force_is_q3s_energy_gradient(paw_derivative_runs):
    baseline = paw_derivative_runs["baseline"]
    displaced = paw_derivative_runs["position"]
    displacement_vector = LATTICE_VECTORS_BOHR[0]
    displacement_length = math.sqrt(sum(value * value for value in displacement_vector))
    displacement_unit = [value / displacement_length for value in displacement_vector]
    analytic = sum(
        baseline["forces_ry_per_bohr"][1][direction] * displacement_unit[direction]
        for direction in range(3)
    )
    numerical = -(displaced[1]["total_energy_ry"] - displaced[-1]["total_energy_ry"]) / (
        2.0 * FINITE_POSITION_STEP * displacement_length
    )
    assert abs(analytic - numerical) < FORCE_GRADIENT_TOLERANCE_RYDBERG_PER_BOHR


def test_paw_ortho_atomic_stress_is_q3s_full_energy_gradient(paw_derivative_runs):
    baseline = paw_derivative_runs["baseline"]
    assert not baseline.get("stress_is_incomplete", False)
    assert (
        max(abs(value) for row in baseline["stress_contributions_kbar"]["hubbard"] for value in row)
        > 5.0
    )
    volume_bohr_cubed = abs(determinant(LATTICE_VECTORS_BOHR))
    for direction_a, direction_b in INDEPENDENT_STRAIN_COMPONENTS:
        pair = paw_derivative_runs["strain"][(direction_a, direction_b)]
        numerical = (
            -(pair[1]["total_energy_ry"] - pair[-1]["total_energy_ry"])
            / (2.0 * FINITE_STRAIN_STEP * volume_bohr_cubed)
            * RYDBERG_PER_BOHR_CUBED_TO_KILOBAR
        )
        analytic = baseline["stress_kbar"][direction_a][direction_b]
        assert abs(analytic - numerical) < STRESS_GRADIENT_TOLERANCE_KILOBAR, (
            f"component ({direction_a}, {direction_b}): analytic {analytic:.8f} kbar "
            f"vs numerical {numerical:.8f} kbar"
        )


def test_paw_gamma_half_sphere_matches_explicit_full_sphere(paw_gamma_runs):
    half = paw_gamma_runs["gamma_half"]
    full = paw_gamma_runs["gamma_full"]
    assert abs(half["total_energy_ry"] - full["total_energy_ry"]) < GAMMA_ENERGY_TOLERANCE_RYDBERG
    assert (
        max(
            abs(
                half["forces_ry_per_bohr"][atom][direction]
                - full["forces_ry_per_bohr"][atom][direction]
            )
            for atom in range(len(half["forces_ry_per_bohr"]))
            for direction in range(3)
        )
        < GAMMA_FORCE_TOLERANCE_RYDBERG_PER_BOHR
    )
    assert (
        max(
            abs(half["stress_kbar"][row][column] - full["stress_kbar"][row][column])
            for row in range(3)
            for column in range(3)
        )
        < GAMMA_STRESS_TOLERANCE_KILOBAR
    )
    assert (
        max(
            abs(
                half["stress_contributions_kbar"]["augmentation"][row][column]
                - full["stress_contributions_kbar"]["augmentation"][row][column]
            )
            for row in range(3)
            for column in range(3)
        )
        < GAMMA_STRESS_TOLERANCE_KILOBAR
    )
    assert (
        max(
            abs(
                half["stress_contributions_kbar"]["hubbard"][row][column]
                - full["stress_contributions_kbar"]["hubbard"][row][column]
            )
            for row in range(3)
            for column in range(3)
        )
        < GAMMA_STRESS_TOLERANCE_KILOBAR
    )
