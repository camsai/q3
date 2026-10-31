"""A screw axis has to survive the irreducible wedge (self-validated).

Wurtzite's space group, P6_3mc, is polar: no operation takes a G-vector to -G,
so each component along the c axis, which every rotation leaves in place, is a
symmetry shell of its own. The 6_3 screw and the c-glides translate by c/2
along that axis, which forbids every (0, 0, l) with l odd: the density's
average over a plane perpendicular to c has to repeat every c/2.

A k-point of the reduced wedge breaks that whenever its time-reversal partner
lies outside its own star, which is the case off the k_z = 0 and 1/2 planes,
so on any mesh odd along c. The density symmetrizer skipped one-member shells,
so those components came through, the potential lost the screw axis, and the
levels it makes degenerate split -- camsai/q3#24, where aluminium nitride with
the reporter's PAW datasets on a 9x9x5 mesh split its lowest pair at K by
7.94 meV, and by 0.0001 meV after the fix. Nothing about it is specific to a
pseudopotential type; the norm-conserving case below split by 35 meV.

Both halves are Q3 runs, so this needs no reference code. A non-self-
consistent pass from the reduced-wedge SCF reads back the potential that SCF
saved and diagonalizes on the k_z = pi/c face, where the screw axis and time
reversal make every level a pair, so no k-point coordinates are needed to know
which levels must agree. The same mesh without symmetry
(`control.disable_symmetry`) is the energy oracle.

Run:
    .venv/bin/pytest tests/validation/test_screw_axis_wedge_consistency.py -v --q3-bin ./build/q3
"""

import json

import pytest
import yaml

from helpers import EXAMPLES_DIRECTORY, eigenvalues_ry, load_results_json, run_q3

RYDBERG_IN_MILLIELECTRONVOLTS = 13605.693122994
NORM_CONSERVING_DIRECTORY = EXAMPLES_DIRECTORY / "data" / "pseudopotentials" / "nc"
PREFIX = "silicon_carbide_2h"

MESH = [3, 3, 3]
"""Odd along c, so the wedge holds k_z = 1/3, whose partner -1/3 is in no star of its own."""

NUMBER_OF_OCCUPIED_BANDS = 8
"""Four atoms of four valence electrons each; only these are compared."""

FACE_POINTS = {"A": [0.0, 0.0, 0.5], "L": [0.5, 0.0, 0.5], "H": [1.0 / 3.0, 1.0 / 3.0, 0.5]}
"""On the k_z = pi/c face of P6_3mc every level is doubly degenerate."""

PAIR_SPLITTING_TOLERANCE_MILLIELECTRONVOLTS = 1.0e-3
"""
Measured before the fix: the occupied pairs split by up to 32.8 meV at A, 35.3 at
L and 35.5 at H. After it, 2.6e-10 meV at most. The bound is a microelectronvolt:
far above the diagonalizer's residual, far below the defect.
"""

WEDGE_ENERGY_TOLERANCE_RYDBERG = 1.0e-9
"""
Measured before the fix: the wedge 3.0e-5 Ry above the full mesh. After it, 2.7e-13
Ry. The bound leaves room for a different platform's SCF trajectory and is still
four orders below the defect.
"""

WURTZITE_SILICON_CARBIDE = {
    "name": "2H-SiC (wurtzite, P6_3mc)",
    "lattice": {
        "type": "HEX",
        "a": 3.0806,
        "b": 3.0806,
        "c": 5.048,
        "alpha": 90,
        "beta": 90,
        "gamma": 120,
        "units": {"length": "angstrom", "angle": "degree"},
    },
    "basis": {
        "units": "crystal",
        "elements": [
            {"id": 0, "value": "Si"},
            {"id": 1, "value": "Si"},
            {"id": 2, "value": "C"},
            {"id": 3, "value": "C"},
        ],
        "coordinates": [
            {"id": 0, "value": [1.0 / 3.0, 2.0 / 3.0, 0.0]},
            {"id": 1, "value": [2.0 / 3.0, 1.0 / 3.0, 0.5]},
            {"id": 2, "value": [1.0 / 3.0, 2.0 / 3.0, 0.3748]},
            {"id": 3, "value": [2.0 / 3.0, 1.0 / 3.0, 0.8748]},
        ],
    },
}


def write_deck(directory, calculation, disable_symmetry):
    material_path = directory / "material.json"
    material_path.write_text(json.dumps(WURTZITE_SILICON_CARBIDE, indent=4) + "\n")
    control = {
        "calculation": calculation,
        "outdir": str(directory / "output"),
        "prefix": PREFIX,
        "disable_symmetry": disable_symmetry,
        "use_deterministic_fft_planning": True,
        # The default 2 x ecutwfc grid aliases the band density, which puts the wedge and the
        # full mesh 3e-7 Ry apart here on any mesh, even along c or odd, before this fix and
        # after it; on the exact grid they agree to 3e-13, so the energy test can be tight.
        "use_exact_wavefunction_grid": True,
    }
    if calculation == "nscf":
        control["kpoints_list"] = [[*point, 1.0] for point in FACE_POINTS.values()]
    deck = {
        "material": str(material_path),
        "model": {"type": "dft", "subtype": "gga", "functional": "pbe"},
        "method": {
            "type": "pseudopotential",
            "subtype": "norm-conserving",
            "cutoffs": {"wavefunction": 25.0, "density": 100.0},
            "pseudopotentials": {
                "Si": str(NORM_CONSERVING_DIRECTORY / "Si.upf"),
                "C": str(NORM_CONSERVING_DIRECTORY / "C.upf"),
            },
            "scf": {
                "convergence_threshold": 1.0e-12,
                "max_iterations": 200,
                "number_of_bands": 12,
                "mixing": {"mode": "broyden", "beta": 0.3},
                "kgrid": {"dimensions": MESH, "shifts": [0, 0, 0]},
            },
        },
        "control": control,
    }
    path = directory / f"{calculation}.yml"
    path.write_text(yaml.safe_dump(deck))
    return path


def run_and_load(q3_command, directory, calculation, disable_symmetry):
    deck = write_deck(directory, calculation, disable_symmetry)
    completed = run_q3(q3_command, deck, working_directory=directory, timeout=900)
    assert completed.returncode == 0, (
        f"{calculation} in {directory.name} exited with {completed.returncode}\n"
        f"stdout:\n{completed.stdout[-4000:]}\nstderr:\n{completed.stderr}"
    )
    return load_results_json(directory / "output", PREFIX)


@pytest.fixture(scope="module")
def wedge(q3_command, tmp_path_factory):
    """The reduced-wedge SCF, and a non-self-consistent pass on the k_z = pi/c face from it."""
    directory = tmp_path_factory.mktemp("wedge")
    self_consistent = run_and_load(q3_command, directory, "scf", disable_symmetry=False)
    face = run_and_load(q3_command, directory, "nscf", disable_symmetry=False)
    return self_consistent, face


@pytest.fixture(scope="module")
def full_mesh(q3_command, tmp_path_factory):
    directory = tmp_path_factory.mktemp("full-mesh")
    return run_and_load(q3_command, directory, "scf", disable_symmetry=True)


def test_the_wedge_keeps_the_screw_axis_pairs_degenerate(wedge):
    _, face = wedge
    widest = {}
    for name, by_spin in zip(FACE_POINTS, eigenvalues_ry(face)):
        bands = by_spin[0] if isinstance(by_spin[0], list) else by_spin
        widest[name] = max(
            (bands[band + 1] - bands[band]) * RYDBERG_IN_MILLIELECTRONVOLTS
            for band in range(0, NUMBER_OF_OCCUPIED_BANDS, 2)
        )
    assert max(widest.values()) < PAIR_SPLITTING_TOLERANCE_MILLIELECTRONVOLTS, (
        "pairs the 6_3 screw makes degenerate came out split (meV): "
        + ", ".join(f"{name} {value:.4f}" for name, value in widest.items())
    )


def test_the_wedge_reproduces_the_full_meshs_total_energy(wedge, full_mesh):
    self_consistent, _ = wedge
    wedge_energy = float(self_consistent.properties.total_energy.value)
    full_energy = float(full_mesh.properties.total_energy.value)
    assert wedge_energy == pytest.approx(full_energy, abs=WEDGE_ENERGY_TOLERANCE_RYDBERG), (
        f"wedge {wedge_energy:.12f} Ry vs full mesh {full_energy:.12f} Ry "
        f"({wedge_energy - full_energy:+.3e} Ry)"
    )
