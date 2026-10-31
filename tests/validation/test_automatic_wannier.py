"""Wannier interpolation with nothing chosen by a human, on a cell with no symmetry.

Tier 2. The three other Wannier examples each carry a number somebody worked
out from a band structure -- bond-centred orbitals, a band range, a frozen
window. This one carries none: the trial orbitals and their count come from
``Si.upf``'s own 3S and 3P shells, the window is disentangled at the protocol's
default thresholds, and the structure is silicon with its cell sheared and both
atoms displaced so that no symmetry operation maps it to itself.

The gate that matters is mesh-point exactness. A Wannier model is a Fourier
transform of H(k) over its mesh, so evaluated back at a mesh point it must
return the DFT eigenvalues of the chosen subspace whatever the gauge and
whatever the localisation -- and with no degeneracies to make a gauge choice
ambiguous for it. Between mesh points the model is only as good as its mesh
allows -- 147 meV rms on this path at 6x6x6 with the spread minimised, against
202 meV without -- so that error is recorded with a loose bound rather than
asserted tightly here; the band distance the automatic workflow judges by is
gated in tests/validation/test_wannier_protocol.py.

Run:
    pytest tests/validation/test_automatic_wannier.py --tier2 -v
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
import pytest
import yaml
from q3.constants import RYDBERG_TO_ELECTRONVOLT

from helpers import EXAMPLES_DIRECTORY, density_of_states_ry, eigenvalues_ry, evaluated_kpoints

EXAMPLE = EXAMPLES_DIRECTORY / "band_structure--semiconductor--silicon-perturbed--wannier"
GRID = (6, 6, 6)
OCCUPIED_BANDS = 4
EXPECTED_WANNIER_FUNCTIONS = 8

pytestmark = pytest.mark.slow


def _stage(workspace: Path, name: str) -> dict:
    manifest = yaml.safe_load((EXAMPLE / name).read_text())
    manifest["material"] = str(EXAMPLES_DIRECTORY / "data" / "materials" / "silicon_perturbed.json")
    manifest["method"]["pseudopotentials"] = {
        "Si": str(EXAMPLES_DIRECTORY / "data" / "pseudopotentials" / "nc" / "Si.upf")
    }
    manifest["control"]["outdir"] = str(workspace / "output")
    return manifest


def _write(workspace: Path, name: str, manifest: dict) -> Path:
    path = workspace / name
    path.write_text(yaml.safe_dump(manifest, sort_keys=False))
    return path


def _run(command: list[str], workspace: Path, label: str) -> None:
    result = subprocess.run(command, cwd=workspace, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        pytest.fail(f"{label} failed ({result.returncode}):\n{(result.stdout or '')[-3000:]}")


def _mesh_kpoints() -> list[list[float]]:
    n1, n2, n3 = GRID
    return [[x / n1, y / n2, z / n3] for x in range(n1) for y in range(n2) for z in range(n3)]


def _sorted_bands(path: Path) -> np.ndarray:
    return np.sort(np.array(eigenvalues_ry(json.loads(path.read_text())))[:, 0, :], axis=1)


@pytest.fixture(scope="module")
def chain(tmp_path_factory, q3_command, q3_utility_command) -> dict:
    """The unattended chain, then the same states computed directly, twice over."""
    workspace = tmp_path_factory.mktemp("automatic_wannier")
    q3 = [str(Path(list(q3_command)[-1]).resolve())]
    output = workspace / "output"

    for stage, source in (("scf", "nc_scf.yml"), ("nscf", "nc_nscf.yml")):
        _run(
            q3
            + [
                "--log-level",
                "warn",
                str(_write(workspace, f"{stage}.yml", _stage(workspace, source))),
            ],
            workspace,
            stage,
        )

    evaluation = _stage(workspace, "nc_interpolate.yml")
    _run(
        list(q3_utility_command)
        + ["interpolate", str(_write(workspace, "evaluate.yml", evaluation))],
        workspace,
        "evaluation",
    )
    on_path = json.loads((output / "si_perturbed_interpolation_results.json").read_text())

    # The reference needs room above the window for the states the model holds.
    direct = _stage(workspace, "nc_interpolate.yml")
    direct["method"]["scf"]["number_of_bands"] = 16
    direct["control"] = {
        "calculation": "nscf",
        "outdir": str(output),
        "prefix": "si_perturbed",
        "kpoints_list": [
            [*point, 2.0 / len(evaluated_kpoints(on_path))] for point in evaluated_kpoints(on_path)
        ],
    }
    _run(
        q3 + ["--log-level", "warn", str(_write(workspace, "direct.yml", direct))],
        workspace,
        "direct bands",
    )
    path_reference = _sorted_bands(output / "si_perturbed_results.json")

    # And again on the mesh the model was built from, where it is not an approximation.
    mesh = _mesh_kpoints()
    on_mesh_request = _stage(workspace, "nc_interpolate.yml")
    on_mesh_request["control"]["kpoints_list"] = [[*point, 1.0] for point in mesh]
    on_mesh_request["control"]["interpolation"].pop("dos_mesh", None)
    _run(
        list(q3_utility_command)
        + [
            "interpolate",
            str(_write(workspace, "mesh.yml", on_mesh_request)),
            "--output",
            str(workspace / "mesh_model.json"),
        ],
        workspace,
        "mesh evaluation",
    )
    direct["control"]["kpoints_list"] = [[*point, 2.0 / len(mesh)] for point in mesh]
    _run(
        q3 + ["--log-level", "warn", str(_write(workspace, "direct_mesh.yml", direct))],
        workspace,
        "direct mesh bands",
    )

    return {
        "on_path": on_path,
        "path_model": np.sort(np.array(eigenvalues_ry(on_path))[:, 0, :], axis=1),
        "path_reference": path_reference,
        "mesh_model": _sorted_bands(workspace / "mesh_model.json"),
        "mesh_reference": _sorted_bands(output / "si_perturbed_results.json"),
    }


def test_the_pseudopotential_decides_how_many_wannier_functions(chain):
    """
    Nothing in the example says eight. Si.upf carries a 3S and a 3P shell, so
    an atom contributes one s and three p orbitals and a two-atom cell holds
    eight -- which is also what makes the 12-band window need disentangling.
    """
    interpolation = chain["on_path"]["construction"]
    assert interpolation["number_of_bands"] == EXPECTED_WANNIER_FUNCTIONS
    assert "disentanglement" in interpolation["description"]

    density = chain["on_path"]["properties"]["density_of_states"]
    energies, values = density_of_states_ry(density)
    integral = float(np.sum(0.5 * (values[1:] + values[:-1]) * np.diff(energies)))
    assert integral == pytest.approx(float(EXPECTED_WANNIER_FUNCTIONS), abs=0.05)


def test_on_its_own_mesh_the_model_is_not_an_approximation(chain):
    """
    The gate. H(R) is a Fourier transform of H(k) over the mesh, so back at a
    mesh point the model must return the DFT eigenvalues of the subspace it
    chose -- with no symmetry to make that easy, and no degeneracy to hide a
    gauge error in. Measured at 0.08 meV rms over 216 points.
    """
    model = chain["mesh_model"][:, :OCCUPIED_BANDS]
    exact = chain["mesh_reference"][:, :OCCUPIED_BANDS]
    difference = (model - exact) * RYDBERG_TO_ELECTRONVOLT * 1000.0

    root_mean_square = float(np.sqrt(np.mean(difference**2)))
    assert root_mean_square < 1.0, f"{root_mean_square:.4f} meV at the mesh points themselves"
    assert float(np.abs(difference).max()) < 5.0


def test_between_mesh_points_it_is_as_good_as_the_localisation(chain):
    """
    147 meV rms on this path with the spread minimised (202 meV before the
    minimiser existed), against 68 meV for the hand-tuned silicon example whose
    four orbitals sit on the bonds. The mesh, not the gauge, is what is left:
    the same recipe moves from 826 to 12.6 meV between 4x4x4 and 8x8x8 on
    undistorted silicon. So the bound here is loose on purpose: it catches a
    model that has stopped working, not one on a coarse mesh.
    """
    model = chain["path_model"][:, :OCCUPIED_BANDS]
    exact = chain["path_reference"][:, :OCCUPIED_BANDS]
    difference = (model - exact) * RYDBERG_TO_ELECTRONVOLT * 1000.0

    root_mean_square = float(np.sqrt(np.mean(difference**2)))
    assert root_mean_square < 500.0, f"{root_mean_square:.1f} meV along the path"
