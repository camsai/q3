"""The acceptance campaign: a roster of structures through the automatic protocol.

Two tiers. Tier 2 runs the cheapest row of the roster -- pristine silicon, a
thousand k-points at the published spacing, with the threshold walk capped at
two pairs -- which is what a pull request's ninety minutes can afford beside
the other chains. Tier 3 runs the whole roster, which is hours, once a month
from the scheduled workflow, and keeps ``campaign.md`` and ``campaign.json``:
set ``Q3_CAMPAIGN_DIRECTORY`` to say where, so a job can upload them, since a
run that does not keep the table has produced nothing.

Nothing here asserts that a structure accepts: the campaign's finding is the
table, and the assertion is that every structure reaches a status unattended
and the table says which.

Run:
    pytest tests/validation/test_wannier_campaign.py --tier2 -v
    Q3_CAMPAIGN_DIRECTORY=campaign-output pytest tests/validation/test_wannier_campaign.py --tier3 -v -s
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

ase = pytest.importorskip("ase")

# The campaign is a package beside its command line under scripts/validation,
# not part of q3: put that directory on the path and import it as itself.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "validation"))

import wannier_campaign as campaign  # noqa: E402

pytestmark = pytest.mark.slow

SMOKE = ("silicon",)


def _run(selected, directory, maximum_threshold_attempts, q3_command, q3_utility_command) -> dict:
    """The roster's *selected* cases through the campaign, into *directory*; the table printed."""
    report = campaign.run_wannier_campaign(
        selected,
        directory,
        maximum_threshold_attempts=maximum_threshold_attempts,
        q3_binary=Path(list(q3_command)[-1]).resolve(),
        q3_utility_binary=Path(list(q3_utility_command)[-1]).resolve(),
    )
    print("\n" + report["table"])
    assert len(report["rows"]) == len(selected)
    for row in report["rows"]:
        assert row["status"] in ("accept", "reject"), row
        assert row["attempts"] >= 1
        assert row["eta_mev"] is not None
        assert row["rung"]
    assert (Path(directory) / "campaign.md").read_text() == report["table"]
    return report


def test_the_campaign_runs_unattended_and_produces_one_table(
    tmp_path, q3_command, q3_utility_command
):
    roster = campaign.campaign_roster()
    selected = [case for case in roster if case.name in SMOKE]
    assert selected, "the roster lost its smoke cases"

    # Two threshold pairs after the default for the smoke row: every rung is
    # a three-minute evaluation on a thousand k-points, and this test is about
    # the machinery reaching a status, not about how far the walk goes.
    report = _run(selected, tmp_path, 2, q3_command, q3_utility_command)

    silicon = next(row for row in report["rows"] if row["name"] == "silicon")
    assert silicon["grid"] == "10x10x10", "the spacing rule's mesh for silicon"
    assert silicon["wannier_functions"] == 8 and silicon["bands"] == 16


@pytest.mark.monthly
def test_the_full_roster_reaches_a_status_unattended(tmp_path, q3_command, q3_utility_command):
    """Every structure of the roster, with the threshold walk uncapped: the campaign's own table."""
    directory = Path(os.environ.get("Q3_CAMPAIGN_DIRECTORY") or tmp_path)
    directory.mkdir(parents=True, exist_ok=True)
    report = _run(campaign.campaign_roster(), directory, None, q3_command, q3_utility_command)
    assert (directory / "campaign.json").exists()
    assert report["total"] == len(campaign.campaign_roster())
