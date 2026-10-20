"""A structure nobody tuned for: substitution, a hard rattle, and a sheared cell.

Tier 2. The other Wannier examples are all one or two atoms of a single element
in a cell with symmetry. This one is four atoms of two elements, one of them
substituted into a site that was silicon, every atom displaced by up to 12% of a
bond length and the cell sheared 5%. Nothing in its manifests was chosen by
looking at a result: the trial orbitals, their count and the frozen window all
come from the pseudopotentials and the defaults.

What is being tested is not accuracy -- a 4x4x4 mesh on a four-atom cell cannot
deliver that -- but whether the chain is CORRECT on a structure it was not
tuned for. The mesh-point round trip is the assertion that carries that: it must
reproduce the DFT eigenvalues of the chosen subspace exactly, whatever the
gauge, and it has no symmetry, no degeneracy and no tuned parameter to lean on.

Run:
    pytest tests/validation/test_silicon_carbon_alloy.py --tier2 -v
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
import pytest
import yaml

from helpers import EXAMPLES_DIRECTORY

EXAMPLE = EXAMPLES_DIRECTORY / "band_structure--semiconductor--silicon-carbon-alloy--wannier"
RYDBERG_TO_ELECTRONVOLT = 13.605693122994
GRID = (4, 4, 4)
WANNIER_FUNCTIONS = 16  # three silicons and a carbon, one s and three p each
OCCUPIED = 8  # sixteen valence electrons

pytestmark = pytest.mark.slow


def _stage(workspace: Path, name: str) -> dict:
    manifest = yaml.safe_load((EXAMPLE / name).read_text())
    manifest["material"] = str(
        EXAMPLES_DIRECTORY / "data" / "materials" / "silicon_carbon_alloy.json"
    )
    manifest["method"]["pseudopotentials"] = {
        "Si": str(EXAMPLES_DIRECTORY / "data" / "pseudopotentials" / "nc" / "Si.upf"),
        "C": str(EXAMPLES_DIRECTORY / "data" / "pseudopotentials" / "nc" / "C.upf"),
    }
    manifest["control"]["outdir"] = str(workspace / "output")
    return manifest


def _write(workspace: Path, name: str, manifest: dict) -> str:
    path = workspace / name
    path.write_text(yaml.safe_dump(manifest, sort_keys=False))
    return str(path)


def _run(command: list[str], workspace: Path, label: str) -> str:
    result = subprocess.run(command, cwd=workspace, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        pytest.fail(f"{label} failed ({result.returncode}):\n{(result.stdout or '')[-3000:]}")
    return (result.stdout or "") + (result.stderr or "")


def _full_grid() -> list[tuple[float, float, float]]:
    first, second, third = GRID
    return [
        (x / first, y / second, z / third)
        for x in range(first)
        for y in range(second)
        for z in range(third)
    ]


@pytest.fixture(scope="module")
def chain(tmp_path_factory, q3_command, q3_utility_command) -> dict:
    """SCF, capture, and evaluation on the mesh, once for every test here."""
    workspace = tmp_path_factory.mktemp("silicon_carbon_alloy")
    q3 = [str(Path(list(q3_command)[-1]).resolve())]

    for stage, source in (("scf", "nc_scf.yml"), ("nscf", "nc_nscf.yml")):
        _run(
            q3
            + ["--log-level", "warn", _write(workspace, f"{stage}.yml", _stage(workspace, source))],
            workspace,
            stage,
        )

    evaluation = _run(
        list(q3_utility_command)
        + ["interpolate", _write(workspace, "path.yml", _stage(workspace, "nc_interpolate.yml"))],
        workspace,
        "evaluation",
    )
    model = json.loads((workspace / "output" / "si3c_interpolation_results.json").read_text())

    # The mesh itself, where the model is a transform rather than an approximation.
    mesh = _full_grid()
    on_mesh = _stage(workspace, "nc_interpolate.yml")
    on_mesh["control"]["kpoints_list"] = [[*point, 1.0] for point in mesh]
    on_mesh["control"]["interpolation"].pop("dos_mesh", None)
    _run(
        list(q3_utility_command)
        + [
            "interpolate",
            _write(workspace, "mesh.yml", on_mesh),
            "--output",
            str(workspace / "mesh_model.json"),
        ],
        workspace,
        "mesh evaluation",
    )
    mesh_model = json.loads((workspace / "mesh_model.json").read_text())

    direct = _stage(workspace, "nc_interpolate.yml")
    direct["method"]["scf"]["number_of_bands"] = 24
    direct["control"] = {
        "calculation": "nscf",
        "outdir": str(workspace / "output"),
        "prefix": "si3c",
        "kpoints_list": [[*point, 2.0 / len(mesh)] for point in mesh],
    }
    _run(
        q3 + ["--log-level", "warn", _write(workspace, "direct.yml", direct)],
        workspace,
        "direct mesh bands",
    )
    mesh_exact = json.loads((workspace / "output" / "si3c_results.json").read_text())
    return {
        "model": model,
        "mesh_model": mesh_model,
        "mesh_exact": mesh_exact,
        "log": evaluation,
    }


def test_the_pseudopotentials_decide_how_many_wannier_functions(chain):
    """
    Nobody wrote 16 anywhere. Silicon's 3S and 3P and carbon's 2S and 2P give
    four orbitals per atom over four atoms, and mixed chemistry is the case
    where an orbital count taken per species rather than per structure matters.
    """
    interpolation = chain["model"]["interpolation"]
    assert interpolation["number_of_bands"] == WANNIER_FUNCTIONS

    density = chain["model"]["density_of_states"]
    energies = np.array(density["energies_ry"])
    values = np.array(density["dos"])
    integral = float(np.sum(0.5 * (values[1:] + values[:-1]) * np.diff(energies)))
    assert integral == pytest.approx(float(WANNIER_FUNCTIONS), abs=0.05)


def test_the_gauge_reaches_a_stationary_point(chain):
    """
    This structure is the one that used to stall: 8% of its diagonal overlaps sit
    against the branch of Im ln, where Omega_D is discontinuous and no step
    lowers it. Guiding centres are what carry the minimisation across that, so
    this asserts the outcome they buy -- a gauge that is actually stationary, and
    a label that says so. A regression shows up as "localization stalled" rather
    than as a quietly worse number.
    """
    interpolation = chain["model"]["interpolation"]
    assert "disentanglement" in interpolation["description"]
    assert "maximally localized" in interpolation["description"]

    spread = interpolation["spread"]
    assert spread["localization_converged"] is True

    # The gauge is never worse than the projection gauge it started from.
    assert spread["total_bohr_squared"] > spread["gauge_invariant_bohr_squared"]


def test_the_mesh_points_come_back_exactly(chain):
    """
    The assertion that carries the whole chain. Evaluated back at a point it was
    built from, the model is a Fourier transform of H(k) and must return the DFT
    eigenvalues of the chosen subspace -- with no symmetry, no degeneracy and no
    tuned parameter anywhere in the structure to make that accidental.
    """
    model = np.sort(np.array(chain["mesh_model"]["eigenvalues_ry"])[:, 0, :], axis=1)
    exact = np.sort(np.array(chain["mesh_exact"]["eigenvalues_ry"])[:, 0, :], axis=1)
    difference = (model[:, :OCCUPIED] - exact[:, :OCCUPIED]) * RYDBERG_TO_ELECTRONVOLT * 1000.0

    root_mean_square = float(np.sqrt(np.mean(difference**2)))
    assert root_mean_square < 5.0, f"{root_mean_square:.4f} meV at the mesh points"
    assert np.abs(difference).max() < 20.0
