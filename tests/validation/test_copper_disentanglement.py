"""Copper: the entangled manifold that the projection gauge alone cannot model.

Tier 2. Runs the shipped example end to end -- SCF, a fourteen-band capture on
the full 4x4x4 grid, then a ten-state model chosen by projectability
disentanglement -- and checks the three things that would each fail differently
if the disentangler were wrong:

  * without disentanglement the same record is REFUSED, which is the behaviour
    this whole feature exists to change;
  * with it and the window resolved automatically, the run detects the metal,
    reports the band this 4p-less dataset cannot represent and the stall it
    causes -- the report a driver acts on -- and with the occupied-manifold
    window the ladder falls back to, the subspace iteration converges and the
    model holds ten Wannier functions whose Bloch-sum weights come back to the
    k-point count;
  * the 3d bands, which are what a copper tight-binding model is for, track a
    direct calculation on the same density.

Run:
    pytest tests/validation/test_copper_disentanglement.py --tier2 -v
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
import yaml
from q3.constants import RYDBERG_TO_ELECTRONVOLT

from helpers import EXAMPLES_DIRECTORY

EXAMPLE = EXAMPLES_DIRECTORY / "band_structure--metal--copper--wannier"

pytestmark = pytest.mark.slow


def _stage(workspace: Path, name: str) -> dict:
    """The example's manifest with absolute data paths and a local outdir."""
    manifest = yaml.safe_load((EXAMPLE / name).read_text())
    manifest["material"] = str(EXAMPLES_DIRECTORY / "data" / "materials" / "copper.json")
    manifest["method"]["pseudopotentials"]["Cu"] = str(
        EXAMPLES_DIRECTORY / "data" / "pseudopotentials" / "nc" / "Cu.upf"
    )
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
def captured(tmp_path_factory, q3_command, q3_utility_command) -> dict:
    """SCF and the fourteen-band capture, shared by every test below."""
    workspace = tmp_path_factory.mktemp("copper_disentanglement")
    q3 = [str(Path(list(q3_command)[-1]).resolve())]

    _run(
        q3
        + [
            "--log-level",
            "warn",
            str(_write(workspace, "scf.yml", _stage(workspace, "nc_scf.yml"))),
        ],
        workspace,
        "SCF",
    )
    _run(
        q3
        + [
            "--log-level",
            "warn",
            str(_write(workspace, "nscf.yml", _stage(workspace, "nc_nscf.yml"))),
        ],
        workspace,
        "capture",
    )

    record = workspace / "output" / "cu.save" / "interpolation.q3ip"
    assert record.is_file(), "the capture wrote no interpolation record"
    shutil.copytree(workspace / "output", workspace / "captured", dirs_exist_ok=True)
    return {"workspace": workspace, "q3": q3, "q3_utility": list(q3_utility_command)}


def test_the_projection_gauge_alone_refuses_this_record(captured):
    """A fourteen-band window cannot fill ten Wannier functions on its own."""
    workspace = captured["workspace"]
    manifest = _stage(workspace, "nc_interpolate.yml")
    manifest["control"]["interpolation"]["disentanglement"] = "none"
    path = _write(workspace, "no_disentanglement.yml", manifest)

    result = subprocess.run(
        captured["q3_utility"] + ["interpolate", str(path)],
        cwd=workspace,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0, "without disentanglement this record must be refused"
    message = (result.stdout or "") + (result.stderr or "")
    assert "disentanglement" in message, message[-500:]


def _evaluate(captured: dict, name: str, frozen_window_ev: float | None) -> tuple[str, dict]:
    """The example's evaluation, with the window left to the rule or set."""
    workspace = captured["workspace"]
    manifest = _stage(workspace, "nc_interpolate.yml")
    if frozen_window_ev is not None:
        manifest["control"]["interpolation"]["frozen_window_above_fermi_ev"] = frozen_window_ev
    path = _write(workspace, name, manifest)
    output = _run(captured["q3_utility"] + ["interpolate", str(path)], workspace, name)
    results = json.loads((workspace / "output" / "cu_interpolation_results.json").read_text())
    return output, results


def test_the_automatic_window_reports_the_band_this_dataset_cannot_represent(captured):
    """
    Nothing in the manifest says what copper is. The run finds one band crossing
    the Fermi level, calls it a metal and applies E_F + 2 eV; band 10 has no
    projectability at seven k-points inside that window and is kept out of the
    freeze there rather than frozen, which is the singular-gauge rule; and
    because the same band is frozen wherever it is barely reachable, the tenth
    Wannier function does not localise and the run says so. Every one of those
    facts is in the results file, which is what makes the failure actionable.
    """
    output, results = _evaluate(captured, "interpolate_auto.yml", None)
    assert "a metal" in output, output[-1500:]
    interpolation = results["interpolation"]
    verdict = interpolation["disentanglement"]
    assert verdict["metallic"] is True
    assert verdict["bands_crossing_fermi"] == 1
    assert verdict["frozen_window_automatic"] is True
    assert verdict["frozen_window_above_fermi_ev"] == pytest.approx(2.0)
    # How many k-points depends on projectabilities within a hair of 0.01, so
    # the count is not pinned; the band is.
    excluded = verdict["bands_excluded_from_freeze"]
    assert [entry["band"] for entry in excluded] == [10], excluded
    assert excluded[0]["kpoints"] > 0
    assert "localization stalled" in interpolation["description"]
    assert interpolation["spread"]["localization_converged"] is False
    assert interpolation["quality"]["spread"]["outliers"] == [9], (
        "the Wannier function built on the unrepresentable state is the outlier"
    )


def test_the_occupied_manifold_window_converges_and_builds_a_ten_state_model(captured):
    """
    The window the automatic workflow's ladder falls back to for such a metal:
    freeze the occupied manifold and let projectability decide the rest. This
    is the model the example shipped with a hand-set -5 eV window, to every
    digit of Omega_I, because everything below the Fermi level that the energy
    rule freezes is already frozen by its projectability.
    """
    output, results = _evaluate(captured, "interpolate_occupied.yml", 0.0)
    assert "converged" in output, output[-800:]
    assert "stationary" in output, output[-800:]

    interpolation = results["interpolation"]
    assert interpolation["number_of_bands"] == 10
    assert "disentanglement" in interpolation["description"]
    assert "maximally localized" in interpolation["description"]
    assert interpolation["spread"]["localization_converged"] is True
    assert interpolation["spread"]["gauge_invariant_bohr_squared"] == pytest.approx(
        12.4358, abs=1e-3
    )
    assert interpolation["disentanglement"]["frozen_window_automatic"] is False
    # Band 10 dips below the Fermi level at a few k-points with no projectability
    # there either, so the rule still keeps it out of the freeze at those.
    assert [
        entry["band"] for entry in interpolation["disentanglement"]["bands_excluded_from_freeze"]
    ] in ([], [10])
    # The Wigner-Seitz weights must sum to the number of k-points transformed.
    assert interpolation["total_bloch_sum_weight"] == pytest.approx(64.0, abs=1e-6)
    # Every centre sits on the atom, which is what atomic projections should give.
    assert max(interpolation["quality"]["centers"]["nearest_atom_distance_bohr"]) < 0.5

    density = results["density_of_states"]
    energies = np.array(density["energies_ry"])
    values = np.array(density["dos"])
    integral = float(np.sum(0.5 * (values[1:] + values[:-1]) * np.diff(energies)))
    assert integral == pytest.approx(10.0, abs=0.05), (
        f"the DOS integrates to {integral:.4f}, not the ten Wannier functions in the model"
    )


def test_the_d_bands_track_a_direct_calculation(captured):
    """
    The 3d manifold is what a copper tight-binding model is for, so that is what
    is gated. The band nearest the entanglement is deliberately not: its error is
    the physics this coarse a grid can carry, not a defect.
    """
    workspace = captured["workspace"]
    _, interpolated = _evaluate(captured, "interpolate_occupied.yml", 0.0)

    # The same path, computed directly, on the same density the capture used.
    direct_manifest = _stage(workspace, "nc_interpolate.yml")
    kpoints = interpolated["kpoints"]
    direct_manifest["control"] = {
        "calculation": "nscf",
        "outdir": str(workspace / "output"),
        "prefix": "cu",
        "kpoints_list": [[*point, 2.0 / len(kpoints)] for point in kpoints],
    }
    direct_path = _write(workspace, "direct.yml", direct_manifest)
    _run(captured["q3"] + ["--log-level", "warn", str(direct_path)], workspace, "direct bands")
    direct = json.loads((workspace / "output" / "cu_results.json").read_text())

    model = np.sort(np.array(interpolated["eigenvalues_ry"])[:, 0, :], axis=1)
    reference = np.sort(np.array(direct["eigenvalues_ry"])[:, 0, :], axis=1)

    # Bands 6 to 9 of the model are the 3d manifold; index 5 through 8.
    for band in range(5, 9):
        deviation = np.abs(reference - model[:, [band]]).min(axis=1).mean()
        deviation_mev = deviation * RYDBERG_TO_ELECTRONVOLT * 1000.0
        assert deviation_mev < 150.0, (
            f"model band {band + 1} sits {deviation_mev:.1f} meV from the direct calculation"
        )
