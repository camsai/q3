"""Norm-conserving ortho-atomic DFT+U stress against Q3 energy derivatives.

The ortho-atomic projector is a system-wide Lowdin transform of every atomic
orbital in the cell. Its strain derivative therefore contains both the direct
orbital derivative and d(O^-1/2)/d(strain), where O is the complete orbital
overlap matrix. The off-symmetry SiC discriminator below carries U only on Si,
so the carbon orbitals still affect the Hubbard stress through that transform.

The complete six-component tensor is checked at two central-difference steps
with a 1e-12 SCF threshold. A controlled build with the d(O^-1/2) term removed
misses this gate by 36.3 kbar; the complete implementation misses by at most
0.05 kbar at these two steps. A Gamma-only half-sphere run is also compared
with the identical explicit full-sphere calculation for energy, force, total
stress, and the isolated Hubbard stress tensor.

Run:
    .venv/bin/pytest tests/validation/test_ortho_atomic_hubbard_stress.py -v \
        --q3-bin ./build/q3
"""

import json
import os
import subprocess
from pathlib import Path

import pytest

REPOSITORY = Path(__file__).resolve().parents[2]
PSEUDOPOTENTIAL_DIRECTORY = REPOSITORY / "examples" / "data" / "pseudopotentials" / "nc"
LATTICE_VECTORS_BOHR = (
    (-4.119180301160476, 0.0, 4.119180301160476),
    (0.0, 4.119180301160476, 4.119180301160476),
    (-4.119180301160476, 4.119180301160476, 0.0),
)
ATOMIC_POSITIONS_CRYSTAL = ((0.0, 0.0, 0.0), (0.27, 0.263, 0.257))
INDEPENDENT_STRAIN_COMPONENTS = ((0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2))
FINITE_STRAIN_STEPS = (0.002, 0.001)
RYDBERG_PER_BOHR_CUBED_TO_KILOBAR = 147105.076
STRESS_GRADIENT_TOLERANCE_KILOBAR = 0.1
GAMMA_ENERGY_TOLERANCE_RYDBERG = 1.0e-8
GAMMA_FORCE_TOLERANCE_RYDBERG_PER_BOHR = 1.0e-6
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


def write_input(directory, lattice_vectors, label, sampling="mesh"):
    material = {
        "lattice": {
            "vectors": dict(zip(("a", "b", "c"), lattice_vectors)),
            "units": {"length": "bohr", "angle": "degree"},
        },
        "basis": {
            "units": "crystal",
            "elements": [{"id": 0, "value": "Si"}, {"id": 1, "value": "C"}],
            "coordinates": [
                {"id": index, "value": position}
                for index, position in enumerate(ATOMIC_POSITIONS_CRYSTAL)
            ],
        },
    }
    (directory / "material.json").write_text(json.dumps(material, indent=4) + "\n")
    relative_silicon_pseudopotential = os.path.relpath(
        PSEUDOPOTENTIAL_DIRECTORY / "Si.upf", directory
    )
    relative_carbon_pseudopotential = os.path.relpath(
        PSEUDOPOTENTIAL_DIRECTORY / "C.upf", directory
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
  subtype: norm-conserving
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
{explicit_kpoint_block}
"""
    )
    return input_path


def run_q3(q3_bin, directory, lattice_vectors, label, sampling="mesh"):
    input_path = write_input(directory, lattice_vectors, label, sampling)
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
def ortho_atomic_stress_runs(q3_bin, tmp_path_factory):
    baseline = run_q3(
        q3_bin,
        tmp_path_factory.mktemp("ortho_stress_baseline"),
        LATTICE_VECTORS_BOHR,
        "baseline",
    )
    strained = {}
    for strain_step in FINITE_STRAIN_STEPS:
        strained[strain_step] = {}
        step_label = f"{strain_step:.4f}".replace(".", "_")
        for direction_a, direction_b in INDEPENDENT_STRAIN_COMPONENTS:
            strained[strain_step][(direction_a, direction_b)] = {}
            for sign, sign_label in ((-1, "minus"), (1, "plus")):
                label = f"strain_{step_label}_{direction_a}{direction_b}_{sign_label}"
                strained[strain_step][(direction_a, direction_b)][sign] = run_q3(
                    q3_bin,
                    tmp_path_factory.mktemp(label),
                    deform_lattice(direction_a, direction_b, sign * strain_step),
                    label,
                )
    return {"baseline": baseline, "strained": strained}


@pytest.fixture(scope="module")
def gamma_ortho_atomic_runs(q3_bin, tmp_path_factory):
    return {
        sampling: run_q3(
            q3_bin,
            tmp_path_factory.mktemp(f"ortho_stress_{sampling}"),
            LATTICE_VECTORS_BOHR,
            sampling,
            sampling,
        )
        for sampling in ("gamma_half", "gamma_full")
    }


def test_norm_conserving_ortho_atomic_stress_is_reported_as_complete(
    ortho_atomic_stress_runs,
):
    baseline = ortho_atomic_stress_runs["baseline"]
    assert not baseline.get("stress_is_incomplete", False)
    hubbard_tensor = baseline["stress_contributions_kbar"]["hubbard"]
    assert max(abs(value) for row in hubbard_tensor for value in row) > 10.0


@pytest.mark.parametrize("strain_step", FINITE_STRAIN_STEPS)
def test_whole_ortho_atomic_stress_tensor_is_q3s_energy_gradient(
    ortho_atomic_stress_runs,
    strain_step,
):
    baseline = ortho_atomic_stress_runs["baseline"]
    volume_bohr_cubed = abs(determinant(LATTICE_VECTORS_BOHR))
    for direction_a, direction_b in INDEPENDENT_STRAIN_COMPONENTS:
        pair = ortho_atomic_stress_runs["strained"][strain_step][(direction_a, direction_b)]
        numerical_stress = (
            -(pair[1]["total_energy_ry"] - pair[-1]["total_energy_ry"])
            / (2.0 * strain_step * volume_bohr_cubed)
            * RYDBERG_PER_BOHR_CUBED_TO_KILOBAR
        )
        analytic_stress = baseline["stress_kbar"][direction_a][direction_b]
        assert abs(analytic_stress - numerical_stress) < STRESS_GRADIENT_TOLERANCE_KILOBAR, (
            f"strain {strain_step}, component ({direction_a}, {direction_b}): analytic "
            f"{analytic_stress:.8f} kbar vs numerical {numerical_stress:.8f} kbar"
        )


def test_gamma_half_sphere_matches_explicit_full_sphere_for_ortho_atomic_derivatives(
    gamma_ortho_atomic_runs,
):
    half = gamma_ortho_atomic_runs["gamma_half"]
    full = gamma_ortho_atomic_runs["gamma_full"]
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
                half["stress_contributions_kbar"]["hubbard"][row][column]
                - full["stress_contributions_kbar"]["hubbard"][row][column]
            )
            for row in range(3)
            for column in range(3)
        )
        < GAMMA_STRESS_TOLERANCE_KILOBAR
    )
