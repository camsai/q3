#!/usr/bin/env python3
"""Summarise a benchmark matrix run, and compare it against a baseline.

    scripts/hpc/nersc_perlmutter/compare_benchmark.py summary.json
    scripts/hpc/nersc_perlmutter/compare_benchmark.py summary.json --baseline docs/benchmarks/baseline-....json
    scripts/hpc/nersc_perlmutter/compare_benchmark.py summary.json --write-baseline out.json

Reads the ``summary.json`` written by ``scripts/validation/run_mpi_reproducibility_matrix.py``
and reduces each (case, topology) to a median and a spread over repetitions.

Three things this does deliberately, each because the alternative produced a
wrong answer at some point:

**The first repetition is discarded.** It is not like the others. Measured on
silicon, the first serial run took 182 ms against a 137 ms median, and the
first two-rank run took 341 ms against 122 ms -- page cache, FFTW planning and
allocation warm-up, none of which the second run pays.

**The solver's own total is the figure, not wall clock.** ``srun`` startup
measured 2-9 s on this machine, which is an order of magnitude more than a
small deck spends calculating: the same runs that take 137 ms of solver take
1020 ms of wall. Wall is still reported, because the gap between the two is
worth seeing.

**A difference is only reported when it clears the noise, and the noise depends
on how long the run is.** Repeated runs of an unchanged binary measured:

    silicon, 137 ms serial      relative sd  1.5%   (worst topology 10.7%)
    copper,  39 s   serial      relative sd  5.0%   (both MPI topologies 0.4%)

So a single fixed threshold is wrong in one direction or the other: 25% hides a
real regression on the copper deck, and 2% would flag the machine as a
regression on the silicon one. The threshold is therefore per entry, taken from
the spread that entry actually showed -- three relative standard deviations,
with a floor for the case where a run was too quiet to have measured its own
noise. The floor is what --threshold sets.
"""

import argparse
import json
import statistics
import sys
from pathlib import Path

# Floor for the per-entry threshold, used when an entry's own spread is too
# small to trust as an estimate -- a couple of repetitions that happened to
# agree should not make the check hair-trigger.
DEFAULT_REGRESSION_THRESHOLD = 0.05
# How many relative standard deviations a change must clear. Three is chosen so
# that a single comparison of medians is unlikely to flag the machine.
SIGMA_MULTIPLIER = 3.0


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary", type=Path, help="summary.json from the matrix runner")
    parser.add_argument("--baseline", type=Path, help="baseline JSON to compare against")
    parser.add_argument("--write-baseline", type=Path, help="write this run out as a baseline")
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_REGRESSION_THRESHOLD,
        help=(
            "Relative slowdown counted as a regression. Default "
            f"{DEFAULT_REGRESSION_THRESHOLD:.0%}, set from measured run-to-run spread."
        ),
    )
    return parser.parse_args()


def reduce_records(summary):
    """Median solver time per (case, topology), first repetition dropped."""
    grouped = {}
    for record in summary["records"]:
        timings = record.get("timings_ms") or {}
        if "total" not in timings:
            # No timings block means the run did not get far enough to write
            # one. Recording it as absent beats recording it as zero.
            continue
        key = f"{record['case']}/{record['topology']}"
        grouped.setdefault(key, []).append(
            (record["repetition"], float(timings["total"]), float(record["wall_seconds"]) * 1000.0)
        )

    reduced = {}
    for key, rows in grouped.items():
        rows.sort()
        kept = rows[1:] if len(rows) > 1 else rows
        solver = [value for _, value, _ in kept]
        wall = [value for _, _, value in kept]
        reduced[key] = {
            "repetitions_kept": len(kept),
            "repetitions_total": len(rows),
            "solver_ms_median": statistics.median(solver),
            "solver_ms_min": min(solver),
            "solver_ms_max": max(solver),
            "solver_ms_relative_sd": (
                statistics.stdev(solver) / statistics.median(solver) if len(solver) > 1 else 0.0
            ),
            "wall_ms_median": statistics.median(wall),
        }
    return reduced


def report(reduced):
    print(f"{'case/topology':<52} {'median':>9} {'min':>9} {'max':>9} {'rel.sd':>7} {'wall':>9}  n")
    for key in sorted(reduced):
        entry = reduced[key]
        print(
            f"{key:<52} {entry['solver_ms_median']:>9.1f} {entry['solver_ms_min']:>9.1f} "
            f"{entry['solver_ms_max']:>9.1f} {entry['solver_ms_relative_sd']:>6.1%} "
            f"{entry['wall_ms_median']:>9.1f}  {entry['repetitions_kept']}"
        )


def entry_threshold(now_entry, baseline_entry, floor):
    """Per-entry threshold: three sigma of whichever side is noisier, or the floor.

    Using the entry's own measured spread is what makes one tool work across a
    137 ms deck and a 39 s one. The first has ten times the relative noise of
    the second, so a shared constant would either miss regressions on the slow
    deck or invent them on the fast one.
    """
    spreads = [entry.get("solver_ms_relative_sd", 0.0) for entry in (now_entry, baseline_entry)]
    return max(floor, SIGMA_MULTIPLIER * max(spreads))


def compare(reduced, baseline, threshold):
    """Report regressions, improvements, and entries present in only one side."""
    regressions = []
    print(f"\nAgainst baseline (floor {threshold:.0%}, or 3 sigma where that is wider):")
    for key in sorted(set(reduced) | set(baseline)):
        if key not in baseline:
            print(f"  {key:<52} new, no baseline")
            continue
        if key not in reduced:
            print(f"  {key:<52} MISSING from this run")
            continue
        now = reduced[key]["solver_ms_median"]
        was = baseline[key]["solver_ms_median"]
        limit = entry_threshold(reduced[key], baseline[key], threshold)
        ratio = now / was if was else float("inf")
        if ratio > 1.0 + limit:
            print(f"  {key:<52} {was:8.1f} -> {now:8.1f} ms  SLOWER {ratio:.2f}x  (>{limit:.0%})")
            regressions.append(key)
        elif ratio < 1.0 - limit:
            print(f"  {key:<52} {was:8.1f} -> {now:8.1f} ms  faster {1 / ratio:.2f}x")
        else:
            print(f"  {key:<52} {was:8.1f} -> {now:8.1f} ms  within noise (+-{limit:.0%})")
    return regressions


def main():
    arguments = parse_arguments()
    with arguments.summary.open(encoding="utf-8") as handle:
        summary = json.load(handle)
    reduced = reduce_records(summary)
    if not reduced:
        print("No runs carried a timings_ms block -- is the binary older than that field?")
        return 1
    report(reduced)

    if arguments.write_baseline:
        # Carry the provenance, and refuse to record a baseline that cannot say
        # what produced it. A baseline is only useful as the other half of a
        # comparison, and "slower than some earlier build" is not a finding.
        metadata = summary.get("metadata", {})
        commit = metadata.get("git_commit")
        if not commit:
            print("Refusing to write a baseline with no git_commit in its metadata.")
            return 1
        if metadata.get("git_worktree_dirty"):
            print(
                f"Note: baseline taken from a dirty worktree at {commit[:12]}; "
                "it records uncommitted state and cannot be reproduced from the commit alone."
            )
        payload = {"metadata": metadata, "entries": reduced}
        with arguments.write_baseline.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        print(f"\nBaseline written: {arguments.write_baseline}")

    if arguments.baseline:
        with arguments.baseline.open(encoding="utf-8") as handle:
            baseline_payload = json.load(handle)
        regressions = compare(reduced, baseline_payload.get("entries", {}), arguments.threshold)
        if regressions:
            print(f"\n{len(regressions)} regression(s) beyond the threshold.")
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
