"""Every case through the workflow, the table rewritten after each, resumable."""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

from q3.errors import first_error_line
from q3.workflows.interpolation import DEFAULT_KPOINT_SPACING_INVERSE_ANGSTROM
from q3.workflows.wannier import run_automatic_wannier_workflow

from .case import CampaignCase
from .manifest import write_case
from .row import CampaignRow
from .settings import (
    DEFAULT_THRESHOLD_ATTEMPTS,
    FAILURE_TAIL_CHARACTERS,
    OUTPUT_DIRECTORY,
    RUN_PREFIX,
)
from .table import campaign_table


def failure_reason(failure: Exception) -> str:
    """The engine's own ``Error:`` line, led by the command; a failure's tail is a timing footer."""
    message = str(failure).strip()
    lines = message.splitlines()
    spoken = first_error_line(message)
    detail = spoken if spoken is not None else message[-FAILURE_TAIL_CHARACTERS:]
    return f"{type(failure).__name__}: {lines[0] if lines else ''} ... {detail}"


def run_wannier_campaign(
    cases: list[CampaignCase],
    directory: str | Path,
    *,
    kpoint_spacing_inverse_angstrom: float = DEFAULT_KPOINT_SPACING_INVERSE_ANGSTROM,
    maximum_mesh_refinements: int = 0,
    maximum_threshold_attempts: int | None = DEFAULT_THRESHOLD_ATTEMPTS,
    q3_binary: str | Path | None = None,
    q3_utility_binary: str | Path | None = None,
    capture_output: bool = True,
    resume: bool = True,
    keep_outputs: bool = False,
    roster: list[CampaignCase] | None = None,
) -> dict:
    """Every case through the workflow; the JSON and Markdown rewritten after each.

    A raise is a row with ``status: error``, never the end of the campaign; with
    ``resume`` rows with a verdict are kept and errors rerun; checkpoints are deleted
    unless ``keep_outputs``; ``roster`` orders the files, so a narrowed run keeps
    every row it is not rerunning.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    report_path = directory / "campaign.json"
    previous = json.loads(report_path.read_text()) if report_path.exists() else {}
    rows_by_name = {row["name"]: row for row in previous.get("rows", [])}
    cases_by_name = {entry["name"]: entry for entry in previous.get("cases", []) if "name" in entry}
    cases_by_name.update({case.name: case.as_dict() for case in cases})
    settled = {
        name
        for name, row in rows_by_name.items()
        if resume and row["status"] in ("accept", "reject")
    }
    ordering = [case.name for case in (cases if roster is None else roster)]

    rows: list[dict] = []
    for case in cases:
        if case.name in settled:
            rows.append(rows_by_name[case.name])
            continue
        started, result, error = time.time(), None, None
        try:
            result = run_automatic_wannier_workflow(
                write_case(case, directory / case.name),
                kpoint_spacing_inverse_angstrom=kpoint_spacing_inverse_angstrom,
                maximum_mesh_refinements=maximum_mesh_refinements,
                maximum_threshold_attempts=maximum_threshold_attempts,
                q3_binary=q3_binary,
                q3_utility_binary=q3_utility_binary,
                capture_output=capture_output,
            )
        except Exception as failure:  # noqa: BLE001 -- one bad structure must not end the campaign
            error = failure_reason(failure)
        elapsed = time.time() - started
        row = (
            CampaignRow.from_error(case, elapsed, error)
            if result is None
            else CampaignRow.from_result(case, result, elapsed)
        )
        rows.append(row.as_dict())
        if not keep_outputs:
            checkpoint = directory / case.name / OUTPUT_DIRECTORY / f"{RUN_PREFIX}.save"
            shutil.rmtree(checkpoint, ignore_errors=True)
        rows_by_name[case.name] = rows[-1]
        names = [name for name in ordering if name in rows_by_name]
        names += [name for name in rows_by_name if name not in set(ordering)]
        report = {
            "kpoint_spacing_inverse_angstrom": kpoint_spacing_inverse_angstrom,
            "maximum_mesh_refinements": maximum_mesh_refinements,
            "maximum_threshold_attempts": maximum_threshold_attempts,
            "cases": [cases_by_name[name] for name in names if name in cases_by_name],
            "rows": [rows_by_name[name] for name in names],
        }
        report_path.write_text(json.dumps(report, indent=2))
        (directory / "campaign.md").write_text(
            campaign_table(
                report["rows"], kpoint_spacing_inverse_angstrom, maximum_threshold_attempts
            )
        )
    return {
        "rows": rows,
        "table": campaign_table(rows, kpoint_spacing_inverse_angstrom, maximum_threshold_attempts),
        "accepted": sum(row["status"] == "accept" for row in rows),
        "total": len(rows),
    }
