"""
CPU ↔ GPU parity of the forces and the stress tensor.

The energy parity test (test_gpu_energy_parity.py) cannot see the forces and the
stress: they are computed after the SCF, from projections of the converged
bands, and a GPU run takes those projections on the device (the nonlocal force
and stress, and the projector occupations that the ultrasoft and PAW
augmentation terms contract). This test runs the same input on the serial CPU
binary and on the GPU binary and compares every contribution to the forces and
to the stress, one by one, so that a wrong term cannot hide behind another.

The second atom is moved off its site. On the ideal diamond lattice every force
vanishes by symmetry and the comparison would be of zeros.

Run:
    .venv/bin/pytest tests/validation/test_gpu_force_stress_parity.py -v \\
        --q3-bin ./build/q3 --q3-gpu-bin ./build/q3_gpu
"""

import json
import subprocess
from pathlib import Path

import pytest

from helpers import EXAMPLES_DIRECTORY, apply_config_overlay

FORCE_PARITY_TOLERANCE_RYDBERG_PER_BOHR = 1e-8
"""
Bound on any Cartesian component of any force contribution, CPU against GPU.

Both runs converge the same SCF to 1e-10 Ry or better in the energy (the FP64
energy parity bound), which leaves the forces, first order in the error of the
bands, agreeing far less tightly than the energy does. Measured on the Metal
build, 2026-11-17: the largest difference over all contributions and cases was
1.5e-10 Ry/bohr.
"""

STRESS_PARITY_TOLERANCE_KBAR = 1e-5
"""
Bound on any component of any stress contribution. The contributions are
hundreds to thousands of kbar each, so this is eight to nine significant
figures. Measured on the Metal build, 2026-11-17: at most 6.2e-7 kbar.
"""

SI_EXAMPLE_DIRECTORY = EXAMPLES_DIRECTORY / "total_energy--semiconductor--silicon"
SI_MATERIAL = EXAMPLES_DIRECTORY / "data" / "materials" / "silicon.json"

# Fixed FFT plans, as in the energy parity test, so that the comparison is of the two backends.
BASE_OVERLAY = {"control": {"use_deterministic_fft_planning": True}}
GAMMA_ONLY_KGRID = {"method": {"scf": {"kgrid": {"dimensions": [1, 1, 1], "shifts": [0, 0, 0]}}}}

# (case_id, base input, k-grid overlay, results prefix)
PARITY_CASES = [
    ("si_nc_2x2x2", SI_EXAMPLE_DIRECTORY / "nc.yml", None, "si_nc"),
    ("si_nc_gamma_only", SI_EXAMPLE_DIRECTORY / "nc.yml", GAMMA_ONLY_KGRID, "si_nc"),
    ("si_us_2x2x2", SI_EXAMPLE_DIRECTORY / "us.yml", None, "si_us"),
    ("si_paw_2x2x2", SI_EXAMPLE_DIRECTORY / "paw.yml", None, "si_paw"),
]
CASE_IDS = [case[0] for case in PARITY_CASES]


def write_silicon_with_an_atom_off_its_site(directory):
    """Write the example's silicon cell with its second atom displaced; return the path."""
    with open(SI_MATERIAL) as file_handle:
        material = json.load(file_handle)
    material["basis"]["coordinates"][1]["value"] = [0.135, 0.120, 0.125]
    path = Path(directory) / "silicon_off_site.json"
    with open(path, "w") as file_handle:
        json.dump(material, file_handle)
    return path


def run_and_read_contributions(command, base_yaml, kgrid_overlay, prefix, run_directory):
    """Run one binary on the displaced cell; return its force and stress contributions."""
    run_directory = Path(run_directory)
    run_directory.mkdir(parents=True, exist_ok=True)
    overlay = dict(BASE_OVERLAY)
    overlay["material"] = str(write_silicon_with_an_atom_off_its_site(run_directory))
    if kgrid_overlay:
        overlay.update(kgrid_overlay)
    merged_input = apply_config_overlay(base_yaml, overlay, run_directory)
    result = subprocess.run(
        [*command, str(merged_input), "--forces", "--stress"],
        cwd=str(run_directory),
        capture_output=True,
        text=True,
        timeout=900,
    )
    assert result.returncode == 0, (
        f"{' '.join(str(part) for part in command)} exited with code {result.returncode}\n"
        f"stdout:\n{result.stdout[-3000:]}\nstderr:\n{result.stderr[-3000:]}"
    )
    with open(run_directory / "output" / f"{prefix}_results.json") as file_handle:
        properties = json.load(file_handle)["properties"]
    forces = {
        name: [atom["value"] for atom in record["values"]]
        for name, record in properties["force_contributions"].items()
    }
    stress = {name: record["value"] for name, record in properties["stress_contributions"].items()}
    for record in properties["records"]:
        if record["name"] == "atomic_forces":
            forces["total"] = [atom["value"] for atom in record["values"]]
        if record["name"] == "stress_tensor":
            stress["total"] = record["value"]
    return forces, stress


def largest_difference(first, second):
    """Largest absolute difference between two equally nested lists of numbers."""
    if isinstance(first, (int, float)):
        return abs(first - second)
    assert len(first) == len(second)
    return max((largest_difference(a, b) for a, b in zip(first, second)), default=0.0)


@pytest.mark.parametrize("case_id,base_yaml,kgrid_overlay,prefix", PARITY_CASES, ids=CASE_IDS)
def test_gpu_force_and_stress_parity(
    q3_command, q3_gpu_binary, tmp_path, case_id, base_yaml, kgrid_overlay, prefix
):
    """Every force and stress contribution of a GPU run matches the serial CPU run."""
    serial_forces, serial_stress = run_and_read_contributions(
        q3_command, base_yaml, kgrid_overlay, prefix, tmp_path / "serial"
    )
    gpu_forces, gpu_stress = run_and_read_contributions(
        [q3_gpu_binary], base_yaml, kgrid_overlay, prefix, tmp_path / "gpu"
    )

    # The comparison is of something: the displaced atom is pushed, and the cell is strained.
    assert largest_difference(serial_forces["total"], [[0.0] * 3] * 2) > 1e-3
    assert largest_difference(serial_stress["total"], [[0.0] * 3] * 3) > 1.0

    assert serial_forces.keys() == gpu_forces.keys()
    assert serial_stress.keys() == gpu_stress.keys()
    force_differences = {
        name: largest_difference(serial_forces[name], gpu_forces[name]) for name in serial_forces
    }
    stress_differences = {
        name: largest_difference(serial_stress[name], gpu_stress[name]) for name in serial_stress
    }
    print(f"{case_id} largest force differences (Ry/bohr): {force_differences}")
    print(f"{case_id} largest stress differences (kbar): {stress_differences}")
    for name, difference in force_differences.items():
        assert difference < FORCE_PARITY_TOLERANCE_RYDBERG_PER_BOHR, (
            f"CPU↔GPU force parity violated for {case_id}, contribution '{name}': "
            f"{difference:.2e} Ry/bohr"
        )
    for name, difference in stress_differences.items():
        assert difference < STRESS_PARITY_TOLERANCE_KBAR, (
            f"CPU↔GPU stress parity violated for {case_id}, contribution '{name}': "
            f"{difference:.2e} kbar"
        )
