"""tests/python/test_wannier_campaign_roster.py

The campaign's roster and its bookkeeping, without running anything: the
roster is one fixed set, every case names datasets that exist, the variants
are what they say they are, and the table is built from rows the driver
would produce. Running the roster is tests/validation/test_wannier_campaign.py.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import yaml

ase = pytest.importorskip("ase")

from ase import Atoms  # noqa: E402
from ase.geometry import find_mic  # noqa: E402

# The campaign is a package beside its command line under scripts/validation,
# not part of q3: put that directory on the path and import it as itself.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "validation"))

import wannier_campaign as campaign  # noqa: E402

DATASET_FILES = campaign.DATASET_FILES
RATTLE_FRACTION = campaign.RATTLE_FRACTION
campaign_roster = campaign.campaign_roster
campaign_table = campaign.campaign_table
run_wannier_campaign = campaign.run_wannier_campaign
write_case = campaign.write_case


def _written(directory):
    """The campaign report as it stands on disk, which is the record a rerun reads."""
    return json.loads((Path(directory) / "campaign.json").read_text())


def _atoms(case):
    vectors = case.material["lattice"]["vectors"]
    return Atoms(
        symbols=[element["value"] for element in case.material["basis"]["elements"]],
        scaled_positions=[entry["value"] for entry in case.material["basis"]["coordinates"]],
        cell=[vectors["a"], vectors["b"], vectors["c"]],
        pbc=True,
    )


def test_the_roster_is_one_fixed_set_from_the_shipped_datasets():
    roster = campaign_roster()
    again = campaign_roster()
    assert [case.name for case in roster] == [case.name for case in again]
    assert json.dumps([case.as_dict() for case in roster]) == json.dumps(
        [case.as_dict() for case in again]
    ), "seeded, so the same roster every time"
    assert 30 <= len(roster) <= 50, len(roster)
    assert len({case.name for case in roster}) == len(roster)
    for case in roster:
        for symbol, path in case.pseudopotentials.items():
            assert path.endswith(DATASET_FILES[case.family][symbol]), case.name
        assert case.number_of_atoms == len(case.material["basis"]["elements"])
    families = {case.family for case in roster}
    assert families == {"nc", "paw", "us"}
    assert not any(
        "Fe" in case.formula or "Ni" in case.formula or "Cr" in case.formula for case in roster
    ), "magnetic elements wait for a confirmed spin-polarized capture"


def test_the_variants_are_what_they_say():
    roster = {case.name: case for case in campaign_roster()}
    pristine = _atoms(roster["silicon"])
    rattled = _atoms(roster["silicon--rattled"])
    bond = pristine.get_all_distances(mic=True)
    bond = bond[bond > 1e-6].min()
    # The material JSON wraps fractional coordinates into the cell, so a
    # displacement across a boundary is read back modulo a lattice vector.
    vectors, lengths = find_mic(rattled.positions - pristine.positions, pristine.cell, pbc=True)
    assert lengths.max() <= RATTLE_FRACTION * bond + 1e-9
    assert lengths.min() > 0.0

    plus = _atoms(roster["silicon--strained-plus"])
    minus = _atoms(roster["silicon--strained-minus"])
    assert plus.get_volume() / pristine.get_volume() == pytest.approx(1.03**3)
    assert minus.get_volume() / pristine.get_volume() == pytest.approx(0.97**3)

    doubled = _atoms(roster["silicon--doubled"])
    assert len(doubled) == 2 * len(pristine)
    assert doubled.get_volume() == pytest.approx(2 * pristine.get_volume())

    assert roster["copper"].smeared and not roster["silicon"].smeared
    assert roster["gallium"].number_of_atoms == 4
    assert roster["arsenic"].number_of_atoms == 2
    assert roster["silicon_carbon_alloy"].number_of_atoms == 4
    assert roster["copper--paw"].family == "paw" and roster["copper--us"].family == "us"
    assert "known to be wrong" in roster["copper--paw"].note


def test_a_case_is_written_as_a_material_and_a_manifest_the_engine_reads(tmp_path):
    case = {entry.name: entry for entry in campaign_roster()}["gallium_arsenide"]
    scf = write_case(case, tmp_path / case.name)
    manifest = yaml.safe_load(scf.read_text())
    material = json.loads((tmp_path / case.name / "material.json").read_text())
    assert manifest["material"] == "material.json"
    assert manifest["model"]["functional"] == "pbe", "named, not left to the datasets"
    assert manifest["method"]["subtype"] == "norm-conserving"
    assert manifest["method"]["cutoffs"] == {"wavefunction": 90.0, "density": 360.0}
    assert set(manifest["method"]["pseudopotentials"]) == {"Ga", "As"}
    assert manifest["method"]["scf"]["smearing"] == {
        "type": "gaussian",
        "degauss": 0.02,
    }, "gallium counts as smeared"
    assert manifest["method"]["scf"]["max_iterations"] == 200
    assert all(
        count % 2 == 0 and count >= 4 for count in manifest["method"]["scf"]["kgrid"]["dimensions"]
    )
    assert material["lattice"]["vectors"]["units"] == "angstrom"
    assert len(material["basis"]["coordinates"]) == 2
    assert material["name"] == case.name

    silicon = {entry.name: entry for entry in campaign_roster()}["silicon"]
    unsmeared = yaml.safe_load(write_case(silicon, tmp_path / silicon.name).read_text())
    assert unsmeared["method"]["scf"]["smearing"]["type"] == "none"


def test_the_table_has_one_row_per_case_and_names_every_failure(tmp_path, monkeypatch):
    """The driver against a stand-in workflow: one accepts, one is rejected, one raises."""
    outcomes = iter(
        [
            {
                "status": "accept",
                "reason": "eta = 3.1 meV",
                "number_of_wannier_functions": 8,
                "number_of_bands": 16,
                "grid": [10, 10, 10],
                "attempts": [{"rung": "default", "status": "accept", "eta_ev": 0.0031}],
                "band_distance": {"eta_ev": 0.0031, "eta_maximum_ev": 0.02},
                "quality": {
                    "spread": {"variance_bohr_squared": 0.001},
                    "hamiltonian_decay": {"ratio": 1e-4},
                },
            },
            {
                "status": "reject",
                "reason": "eta = 40 meV is above the 10 meV gate",
                "number_of_wannier_functions": 26,
                "number_of_bands": 43,
                "grid": [10, 10, 10],
                "attempts": [
                    {"rung": "default", "status": "retry", "eta_ev": 0.055},
                    {"rung": "thresholds", "status": "retry", "eta_ev": 0.040},
                ],
                "band_distance": {"eta_ev": 0.040, "eta_maximum_ev": 0.3},
                "quality": {
                    "spread": {"variance_bohr_squared": 2.0},
                    "hamiltonian_decay": {"ratio": 1e-2},
                },
            },
            RuntimeError("the engine fell over"),
        ]
    )

    def fake_workflow(scf, **_):
        outcome = next(outcomes)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    monkeypatch.setattr(campaign.runner, "run_automatic_wannier_workflow", fake_workflow)
    cases = campaign_roster()[:3]
    report = run_wannier_campaign(cases, tmp_path)
    assert [row["status"] for row in report["rows"]] == ["accept", "reject", "error"]
    assert report["accepted"] == 1 and report["total"] == 3
    assert report["rows"][2]["reason"].startswith("RuntimeError")
    assert report["rows"][1]["rung"] == "thresholds"
    assert report["rows"][0]["eta_mev"] == pytest.approx(3.1)
    assert report["rows"][1]["eta_first_mev"] == pytest.approx(55.0)
    assert report["rows"][1]["eta_best_mev"] == pytest.approx(40.0)
    assert report["rows"][2]["rung"] == "" and "eta_first_mev" not in report["rows"][2]

    table = (tmp_path / "campaign.md").read_text()
    assert table == report["table"]
    assert table.count("\n| ") == 3 + 2, "three rows plus the header and the rule"
    assert "Accepted 1 of 3 (1 without a retry), rejected 1, errors 1" in table
    assert "1 of 3 within 20 meV on the first rung" in table
    assert "capped at 6 pairs" in table
    assert "| 55 | 40 |" in table, "eta on the first rung, then the last attempt's"
    assert "eight elements" in table.lower()
    for case in cases:
        assert f"| {case.name} |" in table
    assert "the engine fell over" in table
    written = json.loads((tmp_path / "campaign.json").read_text())
    assert len(written["rows"]) == 3 and len(written["cases"]) == 3


def test_the_table_survives_an_empty_campaign():
    assert "Accepted 0 of 0" in campaign_table([], 0.2)


def test_a_campaign_resumes_from_its_table_and_drops_each_checkpoint(tmp_path, monkeypatch):
    """
    Rows that reached a verdict are kept on a rerun and only the error is
    redone; each finished case's checkpoint directory is removed, since forty
    of them filled a disk once.
    """
    calls: list[str] = []

    def fake_workflow(scf, **_):
        name = Path(scf).parent.name
        calls.append(name)
        checkpoint = Path(scf).parent / "output" / "campaign.save"
        checkpoint.mkdir(parents=True, exist_ok=True)
        (checkpoint / "interpolation.q3ip").write_text("record")
        if name.endswith("--rattled"):
            raise RuntimeError("no space left on device")
        return {
            "status": "accept",
            "reason": "eta = 3 meV",
            "number_of_wannier_functions": 8,
            "number_of_bands": 16,
            "grid": [10, 10, 10],
            "attempts": [{"rung": "default", "status": "accept", "eta_ev": 0.003}],
            "band_distance": {"eta_ev": 0.003, "eta_maximum_ev": 0.02},
            "quality": {
                "spread": {"variance_bohr_squared": 0.0},
                "hamiltonian_decay": {"ratio": 1e-4},
            },
        }

    monkeypatch.setattr(campaign.runner, "run_automatic_wannier_workflow", fake_workflow)
    cases = campaign_roster()[:2]  # silicon, silicon--rattled
    first = run_wannier_campaign(cases, tmp_path)
    assert [row["status"] for row in first["rows"]] == ["accept", "error"]
    assert "no space left" in first["rows"][1]["reason"]
    assert not (tmp_path / "silicon" / "output" / "campaign.save").exists()
    assert not (tmp_path / "silicon--rattled" / "output" / "campaign.save").exists()

    second = run_wannier_campaign(cases, tmp_path)
    assert calls == ["silicon", "silicon--rattled", "silicon--rattled"], "only the error is rerun"
    assert second["rows"][0] == first["rows"][0]

    third = run_wannier_campaign(cases, tmp_path, resume=False, keep_outputs=True)
    assert calls[-2:] == ["silicon", "silicon--rattled"]
    assert (tmp_path / "silicon" / "output" / "campaign.save" / "interpolation.q3ip").exists()
    assert third["rows"][0]["status"] == "accept"


def test_cutoffs_come_from_the_metadata_and_still_match_what_the_examples_ran():
    """
    The cutoffs are read from `examples/data/pseudopotentials/metadata.yml`
    rather than restated in this module, with deviations declared in
    CUTOFF_OVERRIDES_RY. These are the values the campaign ran its 40-row table
    at, transcribed from the table it replaced: centralising the source must not
    quietly move a cutoff, which would silently invalidate every row.
    """
    expected = {
        "silicon": (48.0, 192.0),
        "diamond": (100.0, 400.0),
        "copper": (120.0, 480.0),
        "gallium_arsenide": (90.0, 360.0),
        "arsenic": (90.0, 360.0),
        "silicon--paw": (40.0, 320.0),
        "diamond--paw": (60.0, 480.0),
        "copper--paw": (40.0, 320.0),
        "silicon--us": (40.0, 200.0),
        "copper--us": (40.0, 200.0),
    }
    actual = {case.name: case.cutoffs for case in campaign_roster()}
    for name, cutoffs in expected.items():
        assert actual[name] == cutoffs, f"{name} moved from {cutoffs} to {actual[name]}"

    # Every deviation from the dataset's own hint is declared, with a reason.
    for (family, element), (value, reason) in campaign.CUTOFF_OVERRIDES_RY.items():
        assert element in campaign.DATASET_FILES[family]
        assert value > 0.0 and len(reason) > 20, (family, element)


def test_only_selects_one_structure_or_one_family_and_never_a_neighbour():
    """
    Roster names are ``<family>--<variant>`` and one family's name sits inside
    another's, so a plain substring test on ``gallium`` also caught every
    ``gallium_arsenide`` case -- a different chemistry -- and ``copper`` also
    caught its PAW and ultrasoft entries. An exact name now wins outright, and
    a family stops at the ``--`` boundary.
    """
    roster = campaign_roster()

    gallium = [case.name for case in campaign.select_cases(roster, "gallium")]
    assert gallium == ["gallium", "gallium--rattled"], gallium
    assert not any("arsenide" in name for name in gallium), (
        "the boundary keeps one family out of another whose name contains it"
    )

    arsenide = [case.name for case in campaign.select_cases(roster, "gallium_arsenide")]
    assert len(arsenide) == 5 and all(name.startswith("gallium_arsenide") for name in arsenide)

    alone = [case.name for case in campaign.select_cases(roster, "gallium", exact=True)]
    assert alone == ["gallium"], "--exact isolates the pristine case named after its family"

    copper = [case.name for case in campaign.select_cases(roster, "copper")]
    assert "copper--paw" in copper and "copper--us" in copper, "a family takes its own variants"

    # Given ultrasoft first, returned PAW first: the roster's order, not the caller's.
    several = [case.name for case in campaign.select_cases(roster, "silicon--us,diamond--paw")]
    assert several == ["diamond--paw", "silicon--us"], several

    with pytest.raises(ValueError, match="matched no case"):
        campaign.select_cases(roster, "nickel")


def test_an_error_row_carries_the_engine_reason_not_its_timing_footer(tmp_path, monkeypatch):
    """
    The engine prints its refusal and THEN a timing footer, so the tail of a
    failure is FFT milliseconds. A row that says only "exited with code 1" is
    useless, which is what a PAW case produced before this.
    """
    footer = "\n".join(f"    stage_{index}: {index * 11.3} ms" for index in range(40))

    def fake_workflow(scf, **_):
        raise RuntimeError(
            "q3 exited with code 1: /build/q3 nscf.yml\n"
            "  WARNING: something benign\n"
            "Error: control.interpolation: the pseudopotentials carry augmentation "
            "charges but the quantum system is not the ultrasoft one\n" + footer
        )

    monkeypatch.setattr(campaign.runner, "run_automatic_wannier_workflow", fake_workflow)
    report = run_wannier_campaign(campaign_roster()[:1], tmp_path)
    reason = report["rows"][0]["reason"]
    assert report["rows"][0]["status"] == "error"
    assert "carry augmentation charges" in reason, "the engine's own words survive"
    assert "stage_39" not in reason, "the timing footer does not crowd them out"
    assert reason.startswith("RuntimeError: q3 exited with code 1"), "the command still leads"


def test_a_narrowed_rerun_keeps_the_rows_it_is_not_running(tmp_path, monkeypatch):
    """
    ``--only`` and ``--limit`` choose the work, not the scope of the record.
    Continuing an interrupted campaign one family at a time must not delete the
    verdicts already in ``campaign.json``, and the table on disk must show them.
    """

    def fake_workflow(scf, **_):
        return {
            "status": "accept",
            "reason": "eta = 3 meV",
            "number_of_wannier_functions": 8,
            "number_of_bands": 16,
            "grid": [10, 10, 10],
            "attempts": [{"rung": "default", "status": "accept", "eta_ev": 0.003}],
            "band_distance": {"eta_ev": 0.003, "eta_maximum_ev": 0.02},
            "quality": {
                "spread": {"variance_bohr_squared": 0.0},
                "hamiltonian_decay": {"ratio": 1e-4},
            },
        }

    monkeypatch.setattr(campaign.runner, "run_automatic_wannier_workflow", fake_workflow)
    roster = campaign_roster()[:3]
    run_wannier_campaign(roster, tmp_path)
    assert [row["name"] for row in _written(tmp_path)["rows"]] == [case.name for case in roster]

    narrowed = [roster[1]]
    report = run_wannier_campaign(narrowed, tmp_path, resume=False, roster=roster)
    assert [row["name"] for row in report["rows"]] == [roster[1].name], (
        "the run is the filtered set"
    )

    written = _written(tmp_path)
    assert [row["name"] for row in written["rows"]] == [case.name for case in roster], (
        "the file keeps every row and stays in roster order"
    )
    assert [entry["name"] for entry in written["cases"]] == [case.name for case in roster]
    table = (tmp_path / "campaign.md").read_text()
    for case in roster:
        assert f"| {case.name} |" in table


def test_a_narrowed_run_on_a_fresh_directory_writes_only_what_it_ran(tmp_path, monkeypatch):
    """The union is with what is already there, and an empty directory has nothing."""

    monkeypatch.setattr(
        campaign.runner,
        "run_automatic_wannier_workflow",
        lambda scf, **_: {
            "status": "accept",
            "reason": "eta = 3 meV",
            "number_of_wannier_functions": 8,
            "number_of_bands": 16,
            "grid": [10, 10, 10],
            "attempts": [{"rung": "default", "status": "accept", "eta_ev": 0.003}],
            "band_distance": {"eta_ev": 0.003, "eta_maximum_ev": 0.02},
            "quality": {
                "spread": {"variance_bohr_squared": 0.0},
                "hamiltonian_decay": {"ratio": 1e-4},
            },
        },
    )
    roster = campaign_roster()[:3]
    run_wannier_campaign([roster[2]], tmp_path, roster=roster)
    assert [row["name"] for row in _written(tmp_path)["rows"]] == [roster[2].name]
