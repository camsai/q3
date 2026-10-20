"""
Per-atom starting magnetization: a same-species antiferromagnet.

Before per-atom moments existed, an antiferromagnet whose two sublattices are
the same element had to be written as two species (`Cr1`/`Cr2`) so that each
could carry its own per-species starting magnetization. With
`method.scf.atomic_starting_magnetization` both sites stay plain `Cr` and the
opposed moments are given atom by atom.

The test runs the CsCl-type AFM chromium cell both ways and checks that the
per-atom run (1) reaches the antiferromagnetic state -- zero net moment, a
large absolute moment -- and (2) lands on the same total energy and absolute
moment as the established two-species formulation. The two inputs describe
identical physics: the per-atom moments also keep the two same-species sites
symmetry-inequivalent, exactly as distinct species labels do, so any
disagreement beyond SCF noise would mean the moments or the symmetry partition
were lost on the way from input to the initial spin density.

No reference code is launched: this is an internal-consistency check.

Run:
    pytest tests/validation/test_per_atom_starting_magnetization.py -v --q3-bin ./build/q3
"""

import copy
import json
import re

import pytest
import yaml

from helpers import EXAMPLES_DIRECTORY, load_results_json, resolve_relative_paths, run_q3

CHROMIUM_DIRECTORY = EXAMPLES_DIRECTORY / "total_energy--metal--magnetic--chromium"
CHROMIUM_INPUT = CHROMIUM_DIRECTORY / "nc.yml"
PREFIX = "cr_afm_per_atom"

ENERGY_AGREEMENT_TOLERANCE_RYDBERG = 1e-6
"""
The two formulations differ only in how the moments and the symmetry partition
are spelled, so they agree to SCF noise (the threshold below is 1e-8 Ry).
"""

ABSOLUTE_MOMENT_AGREEMENT_TOLERANCE_BOHR_MAGNETON = 1e-3

MINIMUM_AFM_ABSOLUTE_MOMENT_BOHR_MAGNETON = 0.5
"""
A collapsed (non-magnetic) solution has an absolute moment of ~0; the AFM
state of this cell carries roughly 1 muB per site at this lattice constant.
"""

MAXIMUM_AFM_NET_MOMENT_BOHR_MAGNETON = 1e-2

REDUCED_COST = {
    "cutoffs": {"wavefunction": 60.0, "density": 240.0},
    "kgrid": {"dimensions": [6, 6, 6], "shifts": [1, 1, 1]},
}


def last_float(pattern, standard_output):
    matches = re.findall(pattern, standard_output)
    assert matches, f"pattern {pattern!r} not found in Q3 output"
    return float(matches[-1])


def base_config():
    with open(CHROMIUM_INPUT) as file_handle:
        config = yaml.safe_load(file_handle)
    resolve_relative_paths(config, CHROMIUM_DIRECTORY)
    config["method"]["cutoffs"] = copy.deepcopy(REDUCED_COST["cutoffs"])
    config["method"]["scf"]["kgrid"] = copy.deepcopy(REDUCED_COST["kgrid"])
    config["control"]["prefix"] = PREFIX
    return config


def two_species_config():
    """The established formulation: sublattices labelled Cr1 / Cr2."""
    return base_config()


def per_atom_config():
    """Both sites plain Cr; opposed moments given per atom."""
    config = base_config()
    with open(config["material"]) as file_handle:
        material = json.load(file_handle)
    for element in material["basis"]["elements"]:
        element["value"] = "Cr"
    config["material"] = material

    pseudopotentials = config["method"]["pseudopotentials"]
    config["method"]["pseudopotentials"] = {"Cr": pseudopotentials["Cr1"]}

    scf = config["method"]["scf"]
    del scf["starting_magnetization"]
    scf["atomic_starting_magnetization"] = [0.5, -0.5]
    return config


def run_config(q3_command, config, run_directory):
    run_directory.mkdir(parents=True, exist_ok=True)
    config["control"]["outdir"] = str(run_directory / "output")
    input_path = run_directory / "input.yml"
    with open(input_path, "w") as file_handle:
        yaml.dump(config, file_handle, default_flow_style=False)
    result = run_q3(q3_command, input_path, run_directory, timeout=1800)
    assert result.returncode == 0, (
        f"q3 exited with code {result.returncode}\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    results = load_results_json(run_directory / "output", PREFIX)
    return {
        "energy": float(results["total_energy_ry"]),
        "net_moment": last_float(r"[Tt]otal magnetization\s*=\s*(-?\d+\.\d+)", result.stdout),
        "absolute_moment": last_float(r"absolute magnetization\s*=\s*(-?\d+\.\d+)", result.stdout),
    }


@pytest.fixture(scope="module")
def two_species_run(q3_command, tmp_path_factory):
    return run_config(q3_command, two_species_config(), tmp_path_factory.mktemp("cr_two_species"))


@pytest.fixture(scope="module")
def per_atom_run(q3_command, tmp_path_factory):
    return run_config(q3_command, per_atom_config(), tmp_path_factory.mktemp("cr_per_atom"))


@pytest.mark.slow
def test_same_species_cell_reaches_the_antiferromagnetic_state(per_atom_run):
    assert abs(per_atom_run["net_moment"]) < MAXIMUM_AFM_NET_MOMENT_BOHR_MAGNETON
    assert per_atom_run["absolute_moment"] > MINIMUM_AFM_ABSOLUTE_MOMENT_BOHR_MAGNETON


@pytest.mark.slow
def test_per_atom_moments_match_the_two_species_formulation(per_atom_run, two_species_run):
    assert per_atom_run["energy"] == pytest.approx(
        two_species_run["energy"], abs=ENERGY_AGREEMENT_TOLERANCE_RYDBERG
    )
    assert per_atom_run["absolute_moment"] == pytest.approx(
        two_species_run["absolute_moment"], abs=ABSOLUTE_MOMENT_AGREEMENT_TOLERANCE_BOHR_MAGNETON
    )
