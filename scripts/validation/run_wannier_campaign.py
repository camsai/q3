#!/usr/bin/env python3
"""Run the acceptance campaign of the automatic Wannier protocol.

    python scripts/validation/run_wannier_campaign.py --directory campaign --q3-bin build/q3

writes ``campaign.json`` (every row with its ladder history) and ``campaign.md``
after every structure, so an interrupted campaign still reports what it did, and
resumes from them: rows with a verdict are kept, errors are rerun. The campaign
itself is the ``wannier_campaign`` package beside this file; this is its command
line.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# The package beside this script, and the q3 of this checkout ahead of any
# installed one: the campaign measures the tree it sits in.
SCRIPT_DIRECTORY = Path(__file__).resolve().parent
sys.path[:0] = [str(SCRIPT_DIRECTORY), str(SCRIPT_DIRECTORY.parents[1] / "src" / "py")]

from q3.workflows.interpolation import (  # noqa: E402 -- needs the sys.path entries above
    DEFAULT_KPOINT_SPACING_INVERSE_ANGSTROM,
)
from wannier_campaign import (  # noqa: E402
    DEFAULT_DATA_DIRECTORY,
    DEFAULT_THRESHOLD_ATTEMPTS,
    FAMILIES,
    campaign_roster,
    run_wannier_campaign,
    select_cases,
)


def main(arguments: list[str] | None = None) -> int:
    """The command line; resumes from the directory by default."""
    parser = argparse.ArgumentParser(description="Run the Wannier acceptance campaign.")
    add = parser.add_argument
    add("--directory", default="wannier_campaign", help="runs and table go here")
    add("--data-directory", default=str(DEFAULT_DATA_DIRECTORY))
    add("--families", default=",".join(FAMILIES), help="comma-separated: " + ", ".join(FAMILIES))
    add("--only", default="", help="a family or a full case name; comma-separated")
    add("--exact", action="store_true", help="--only names one case, not a family")
    add("--limit", type=int, default=0, help="run only the first N cases")
    add("--spacing", type=float, default=DEFAULT_KPOINT_SPACING_INVERSE_ANGSTROM)
    add("--refinements", type=int, default=0, help="mesh rungs allowed per case")
    add(
        "--threshold-attempts",
        type=int,
        default=DEFAULT_THRESHOLD_ATTEMPTS,
        help="threshold pairs per case; 0 for all",
    )
    add("--q3-bin", default=None)
    add("--q3_utility-bin", default=None)
    add("--list", action="store_true", help="print the roster and exit")
    add("--rerun", action="store_true", help="ignore rows already in the directory")
    add("--keep-outputs", action="store_true", help="keep checkpoints and records")
    options = parser.parse_args(arguments)

    roster = campaign_roster(options.data_directory, tuple(options.families.split(",")))
    cases = select_cases(roster, options.only, exact=options.exact) if options.only else roster
    cases = cases[: options.limit] if options.limit else cases
    if options.list:
        for case in cases:
            print(
                f"{case.name:40s} {case.family:4s} {case.formula:5s} {case.number_of_atoms:3d} atoms  {case.variant}"
            )
        return 0
    report = run_wannier_campaign(
        cases,
        options.directory,
        kpoint_spacing_inverse_angstrom=options.spacing,
        maximum_mesh_refinements=options.refinements,
        maximum_threshold_attempts=options.threshold_attempts or None,
        q3_binary=options.q3_bin,
        q3_utility_binary=options.q3_utility_bin,
        resume=not options.rerun,
        keep_outputs=options.keep_outputs,
        roster=roster,
    )
    print(report["table"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
