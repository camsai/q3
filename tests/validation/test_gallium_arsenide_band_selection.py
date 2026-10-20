"""GaAs: an isolated manifold that does not begin at band 1.

Tier 2. The point of `control.interpolation.bands` is that the model must track
the bands it names rather than the lowest ones, and for GaAs those are 19 to 22
because both PseudoDojo datasets put their 3d shells in valence. Getting this
wrong is not subtle: bands 1 to 4 are semicore, roughly 170 eV lower, so a model
built from the wrong window misses by more than the entire valence bandwidth.

Run:
    pytest tests/validation/test_gallium_arsenide_band_selection.py --tier2 -v
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
import pytest
import yaml
from q3.constants import RYDBERG_TO_ELECTRONVOLT

from helpers import EXAMPLES_DIRECTORY

EXAMPLE = EXAMPLES_DIRECTORY / "band_structure--semiconductor--gallium-arsenide--wannier"
FIRST_BAND, LAST_BAND = 19, 22

pytestmark = pytest.mark.slow


def _stage(workspace: Path, name: str) -> dict:
    manifest = yaml.safe_load((EXAMPLE / name).read_text())
    manifest["material"] = str(EXAMPLES_DIRECTORY / "data" / "materials" / "gallium_arsenide.json")
    manifest["method"]["pseudopotentials"] = {
        "Ga": str(EXAMPLES_DIRECTORY / "data" / "pseudopotentials" / "nc" / "Ga.upf"),
        "As": str(EXAMPLES_DIRECTORY / "data" / "pseudopotentials" / "nc" / "As.upf"),
    }
    manifest["control"]["outdir"] = str(workspace / "output")
    return manifest


def _write(workspace: Path, name: str, manifest: dict) -> Path:
    path = workspace / name
    path.write_text(yaml.safe_dump(manifest, sort_keys=False))
    return path


def _run(command: list[str], workspace: Path, label: str) -> str:
    result = subprocess.run(command, cwd=workspace, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        pytest.fail(f"{label} failed ({result.returncode}):\n{(result.stdout or '')[-3000:]}")
    return (result.stdout or "") + (result.stderr or "")


@pytest.fixture(scope="module")
def chain(tmp_path_factory, q3_command, q3_utility_command) -> dict:
    """SCF, the offset capture, and the evaluation, once for every test here."""
    workspace = tmp_path_factory.mktemp("gallium_arsenide")
    q3 = [str(Path(list(q3_command)[-1]).resolve())]

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
    _run(
        list(q3_utility_command)
        + [
            "interpolate",
            str(_write(workspace, "interpolate.yml", _stage(workspace, "nc_interpolate.yml"))),
        ],
        workspace,
        "evaluation",
    )
    interpolated = json.loads(
        (workspace / "output" / "gaas_interpolation_results.json").read_text()
    )

    # The same path, computed directly, with enough bands to hold the window.
    direct_manifest = _stage(workspace, "nc_interpolate.yml")
    kpoints = interpolated["kpoints"]
    direct_manifest["control"] = {
        "calculation": "nscf",
        "outdir": str(workspace / "output"),
        "prefix": "gaas",
        "kpoints_list": [[*point, 2.0 / len(kpoints)] for point in kpoints],
    }
    direct_manifest["method"]["scf"]["number_of_bands"] = LAST_BAND + 4
    _run(
        q3 + ["--log-level", "warn", str(_write(workspace, "direct.yml", direct_manifest))],
        workspace,
        "direct bands",
    )
    direct = json.loads((workspace / "output" / "gaas_results.json").read_text())
    return {"interpolated": interpolated, "direct": direct}


def test_the_model_holds_four_wannier_functions(chain):
    interpolation = chain["interpolated"]["interpolation"]
    assert interpolation["number_of_bands"] == 4
    assert interpolation["total_bloch_sum_weight"] == pytest.approx(64.0, abs=1e-6)

    density = chain["interpolated"]["density_of_states"]
    energies = np.array(density["energies_ry"])
    values = np.array(density["dos"])
    integral = float(np.sum(0.5 * (values[1:] + values[:-1]) * np.diff(energies)))
    assert integral == pytest.approx(4.0, abs=0.05)


def test_it_tracks_the_selected_window_and_not_the_lowest_bands(chain):
    model = np.sort(np.array(chain["interpolated"]["eigenvalues_ry"])[:, 0, :], axis=1)
    reference = np.sort(np.array(chain["direct"]["eigenvalues_ry"])[:, 0, :], axis=1)

    selected = reference[:, FIRST_BAND - 1 : LAST_BAND]
    deviation = np.abs(model - selected).max() * RYDBERG_TO_ELECTRONVOLT
    assert deviation < 1.0, f"the model sits {deviation:.3f} eV from bands {FIRST_BAND}-{LAST_BAND}"

    # The failure this guards against: silently modelling the semicore instead.
    lowest = reference[:, 0:4]
    separation = np.abs(model - lowest).min() * RYDBERG_TO_ELECTRONVOLT
    assert separation > 50.0, (
        "the model is indistinguishable from the lowest four bands, so the band range was ignored"
    )


def test_the_manifold_is_reproduced_to_the_grid_s_accuracy(chain):
    """
    124 meV on this 4x4x4 grid, against silicon's 68 on the same grid: a polar
    bond makes a symmetric s orbital at the midpoint a cruder starting gauge,
    and without a spread minimiser the starting gauge is the final one.
    """
    model = np.sort(np.array(chain["interpolated"]["eigenvalues_ry"])[:, 0, :], axis=1)
    reference = np.sort(np.array(chain["direct"]["eigenvalues_ry"])[:, 0, :], axis=1)
    difference = (model - reference[:, FIRST_BAND - 1 : LAST_BAND]) * RYDBERG_TO_ELECTRONVOLT

    root_mean_square = float(np.sqrt(np.mean(difference**2))) * 1000.0
    assert root_mean_square < 200.0, f"{root_mean_square:.1f} meV over the manifold"

    # The upper two bands are what a transport or optical model reads.
    upper = float(np.sqrt(np.mean(difference[:, 2:] ** 2))) * 1000.0
    assert upper < 100.0, f"the upper valence bands sit {upper:.1f} meV out"
