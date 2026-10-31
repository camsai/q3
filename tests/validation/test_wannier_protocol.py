"""The protocol around the Wannier mechanics, on real records.

Tier 2. `tests/python/test_wannier_workflow.py` checks that the ladder does
what the plan says on scripted outcomes; this file checks that the quantities
it acts on mean what they are supposed to mean when the engine produces them:

  * the H(R) decay ratio orders silicon's 4x4x4 and 8x8x8 models the way
    their off-mesh error is ordered, and the hand-tuned bond-centred model
    has the lower spread variance and centres on the bond midpoints;
  * the band distance the workflow reports is the published one, recomputed
    here from the two results files with the weights written out;
  * copper, with nothing in its manifest about the window, is detected as a
    metal, has the band its dataset cannot represent reported, stalls, and is
    carried by the ladder's window rung to the converged model the example
    used to reach with a hand-set number.

Run:
    pytest tests/validation/test_wannier_protocol.py --tier2 -v
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml
from q3.constants import RYDBERG_TO_ELECTRONVOLT
from q3.workflows.interpolation import run_interpolation_workflow
from q3.workflows.wannier import run_automatic_wannier_workflow

from helpers import EXAMPLES_DIRECTORY, eigenvalues_ry

SILICON = EXAMPLES_DIRECTORY / "band_structure--semiconductor--silicon--wannier"
PERTURBED = EXAMPLES_DIRECTORY / "band_structure--semiconductor--silicon-perturbed--wannier"
COPPER = EXAMPLES_DIRECTORY / "band_structure--metal--copper--wannier"

pytestmark = pytest.mark.slow


def _stage(workspace: Path, example: Path, pseudopotentials: dict[str, str]) -> Path:
    """The example's SCF manifest with absolute data paths and a local outdir."""
    manifest = yaml.safe_load((example / "nc_scf.yml").read_text())
    material = Path(manifest["material"]).name
    manifest["material"] = str(EXAMPLES_DIRECTORY / "data" / "materials" / material)
    manifest["method"]["pseudopotentials"] = {
        symbol: str(EXAMPLES_DIRECTORY / "data" / "pseudopotentials" / "nc" / file)
        for symbol, file in pseudopotentials.items()
    }
    manifest["control"]["outdir"] = str(workspace / "output")
    path = workspace / "scf.yml"
    path.write_text(yaml.safe_dump(manifest, sort_keys=False))
    return path


def _binaries(q3_command, q3_utility_command) -> dict:
    return {
        "q3_binary": Path(list(q3_command)[-1]).resolve(),
        "q3_utility_binary": Path(list(q3_utility_command)[-1]).resolve(),
    }


@pytest.fixture(scope="module")
def silicon_models(tmp_path_factory, q3_command, q3_utility_command) -> dict:
    """Silicon three ways: the automatic recipe on 4x4x4 and 8x8x8, and bond-centred on 4x4x4."""
    binaries = _binaries(q3_command, q3_utility_command)
    models = {}
    for label, grid, options in (
        ("automatic_4", (4, 4, 4), {"number_of_bands": 12, "disentanglement": "projectability"}),
        ("automatic_8", (8, 8, 8), {"number_of_bands": 12, "disentanglement": "projectability"}),
        (
            "bond_centred_4",
            (4, 4, 4),
            {
                "projection_centers": [(0.5, 0, 0), (0, 0.5, 0), (0, 0, 0.5), (0, 0, 0)],
                "projection_decay_bohr": 1.0,
            },
        ),
    ):
        workspace = tmp_path_factory.mktemp(f"silicon_{label}")
        scf = _stage(workspace, SILICON, {"Si": "Si.upf"})
        result = run_interpolation_workflow(
            scf, grid=grid, kpoints=[(0.0, 0.0, 0.0)], **binaries, **options
        )
        models[label] = result["interpolation_results"]
    return models


def test_the_hamiltonian_decay_ratio_orders_the_meshes_like_their_error(silicon_models):
    """
    826 meV at 4x4x4 against 12.6 at 8x8x8 for the automatic recipe, measured
    before any of this existed. The outermost Wigner-Seitz term against the
    largest hopping separates the two by more than a decade on the same axis
    (measured: 2.0e-3 and 6.8e-5), and the bond-centred model, whose error is
    a tenth of the automatic one's on the same mesh, sits below it too.
    """
    coarse = silicon_models["automatic_4"]["quality"]["hamiltonian_decay"]
    fine = silicon_models["automatic_8"]["quality"]["hamiltonian_decay"]
    tuned = silicon_models["bond_centred_4"]["quality"]["hamiltonian_decay"]
    assert coarse["ratio"] > 1.0e-3, coarse
    assert fine["ratio"] < 1.0e-4, fine
    assert tuned["ratio"] < coarse["ratio"], (tuned, coarse)
    assert coarse["outermost_shell_size"] == 6 and fine["outermost_shell_size"] == 6, (
        "the outermost shell of a cubic mesh's Wigner-Seitz cell is its six far vertices"
    )


def test_the_hand_tuned_model_has_the_lower_spread_variance_and_sits_on_the_bonds(
    silicon_models,
):
    tuned = silicon_models["bond_centred_4"]
    automatic = silicon_models["automatic_4"]
    assert (
        tuned["quality"]["spread"]["variance_bohr_squared"]
        <= (automatic["quality"]["spread"]["variance_bohr_squared"])
    )
    assert max(tuned["quality"]["centers"]["nearest_bond_midpoint_distance_bohr"]) < 1.0e-2
    assert min(automatic["quality"]["centers"]["nearest_atom_distance_bohr"]) > 1.0, (
        "atomic s+p projections localise onto sp3-like functions pulled off the atoms"
    )
    assert tuned["quality"]["spread"]["outliers"] == []
    assert automatic["quality"]["spread"]["outliers"] == []
    assert len(automatic["construction"]["spread"]["spreads_bohr_squared"]) == 8
    assert sum(automatic["construction"]["spread"]["spreads_bohr_squared"]) == pytest.approx(
        automatic["construction"]["spread"]["total_bohr_squared"], rel=1e-8
    )


def _band_distance_by_hand(reference: dict, model: dict, fermi_energy_ry: float) -> float:
    """eta with the weights written out, independently of q3.workflows.band_distance."""
    mu = fermi_energy_ry * RYDBERG_TO_ELECTRONVOLT + 2.0
    sigma = 0.1
    ref = np.sort(np.array(eigenvalues_ry(reference))[:, 0, :], axis=1) * RYDBERG_TO_ELECTRONVOLT
    mod = np.sort(np.array(eigenvalues_ry(model))[:, 0, :], axis=1) * RYDBERG_TO_ELECTRONVOLT
    bands = min(ref.shape[1], mod.shape[1])
    ref, mod = ref[:, :bands], mod[:, :bands]
    weight = np.sqrt(
        1.0 / (1.0 + np.exp((ref - mu) / sigma)) * 1.0 / (1.0 + np.exp((mod - mu) / sigma))
    )
    return float(np.sqrt((weight * (ref - mod) ** 2).sum() / weight.sum()))


def test_the_workflow_judges_the_perturbed_silicon_by_the_published_band_distance(
    tmp_path_factory, q3_command, q3_utility_command
):
    """
    The symmetry-free silicon, run with nothing chosen by hand: the dataset
    decides on eight Wannier functions and a sixteen-band window, the mesh is
    given to keep the run short, and the reference is the half-shifted
    sub-mesh. Whatever the ladder decides, the eta it reports must be the
    published quantity computed on the two results files it hands back --
    which is what a driver reading `quality.band_distance` relies on.
    """
    workspace = tmp_path_factory.mktemp("perturbed_silicon_protocol")
    scf = _stage(workspace, PERTURBED, {"Si": "Si.upf"})
    result = run_automatic_wannier_workflow(
        scf,
        grid=(4, 4, 4),
        maximum_mesh_refinements=0,
        **_binaries(q3_command, q3_utility_command),
    )
    assert result["number_of_wannier_functions"] == 8
    assert result["number_of_bands"] == 16
    assert result["grid"] == [4, 4, 4]
    assert result["status"] in ("accept", "reject")
    assert result["attempts"][0]["rung"] == "default"
    assert result["reference_results"] is not None
    assert len(result["reference_kpoints"]) == 64

    reported = result["band_distance"]["eta_ev"]
    by_hand = _band_distance_by_hand(
        result["reference_results"], result["interpolation_results"], result["fermi_energy_ry"]
    )
    assert reported == pytest.approx(by_hand, abs=1.0e-4), (reported, by_hand)

    quality = result["interpolation_results"]["quality"]
    assert quality["status"] == result["status"]
    assert quality["band_distance"]["eta_ev"] == pytest.approx(reported)
    assert quality["band_distance"]["reference"] == "half-shifted mesh"
    assert len(quality["ladder"]) == len(result["attempts"])
    verdict = result["interpolation_results"]["construction"]["disentanglement"]
    assert verdict["metallic"] is False
    # The published window, or wider if the ladder's window rung widened it.
    assert verdict["frozen_window_above_fermi_ev"] >= 2.0 - 1e-9


def test_copper_runs_unattended_and_the_ladder_lands_on_the_occupied_manifold(
    tmp_path_factory, q3_command, q3_utility_command
):
    """
    Nothing in copper's manifest says what it is or what to freeze. The
    default rung detects the metal, applies 2 eV, reports band 10 as
    unrepresentable at seven k-points and stalls; the demand rung stalls; the
    window rung freezes the occupied manifold alone and converges to the model
    the example used to reach with a hand-set -5 eV (Omega_I = 12.4358 bohr^2).
    The accept gate is opened wide so the test is about the ladder, not about
    what a 4x4x4 mesh can deliver.
    """
    workspace = tmp_path_factory.mktemp("copper_protocol")
    scf = _stage(workspace, COPPER, {"Cu": "Cu.upf"})
    result = run_automatic_wannier_workflow(
        scf,
        grid=(4, 4, 4),
        number_of_bands=14,
        accept_eta_ev=1.0,
        accept_eta_maximum_ev=10.0,
        maximum_mesh_refinements=0,
        **_binaries(q3_command, q3_utility_command),
    )
    rungs = [attempt["rung"] for attempt in result["attempts"]]
    assert rungs == ["default", "stalled", "window"], result["attempts"]
    assert result["attempts"][0]["localization_converged"] is False
    assert result["attempts"][2]["localization_converged"] is True
    assert result["status"] == "accept"

    interpolation = result["interpolation_results"]["construction"]
    assert "maximally localized" in interpolation["description"]
    assert interpolation["spread"]["gauge_invariant_bohr_squared"] == pytest.approx(
        12.4358, abs=1e-3
    )
    assert interpolation["disentanglement"]["frozen_window_above_fermi_ev"] == 0.0
    assert interpolation["disentanglement"]["metallic"] is True
    ladder = result["interpolation_results"]["quality"]["ladder"]
    assert ladder[0]["settings"].get("frozen_window_above_fermi_ev") is None
    assert ladder[2]["settings"]["frozen_window_above_fermi_ev"] == 0.0
