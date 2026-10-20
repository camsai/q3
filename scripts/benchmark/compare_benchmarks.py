#!/usr/bin/env python3
"""Reduce a benchmark run to per-deck figures, and compare it against a baseline.

    scripts/benchmark/compare_benchmarks.py records.json
    scripts/benchmark/compare_benchmarks.py records.json --baseline docs/benchmarks/baseline-<machine>.json
    scripts/benchmark/compare_benchmarks.py records.json --write-baseline docs/benchmarks/baseline-<m>.json

Reads the records ``scripts/benchmark/run_benchmarks.py`` writes, reduces each deck to a
median and a spread over its repetitions, and -- given a baseline -- exits 1 if
any tracked stage got slower by more than the threshold.

Four decisions here, each because the naive alternative gives a wrong answer:

**The first repetition is dropped when there is more than one.** It is the run
that pays the cold page cache for the pseudopotential and material files, the
allocator's first growth, and the first touch of every lazily mapped page. How
much that costs depends entirely on the machine, which is the argument for
dropping it rather than for trusting it: on Perlmutter the first serial run of
a silicon deck took 182 ms against a 137 ms median (measured for
``scripts/hpc/nersc_perlmutter/compare_benchmark.py``), while three
repetitions on a warm x86 development box showed no first-run penalty at all
(262.3 ms, then 263.2 and 261.8). One repetition is cheap; a warm-up charged to
the median is not visible afterwards.

**The threshold widens to the noise the decks actually showed.** ``--threshold``
(default 10%) is a floor, not the whole rule: a deck whose own repetitions
spread by more than that is compared at three relative standard deviations
instead. A single number cannot serve both a one-second deck and a
several-minute one -- the short deck's relative spread is several times the
long one's -- and a gate that fires on the machine's mood teaches people to
ignore it.

**Small stages are reported but not gated.** A stage that takes a couple of
milliseconds can double because the scheduler looked away; gating it buys
nothing and costs a false failure. On the tracked decks the solver total's
repetitions spread by 0.4% to 1.5%, while ``ritz_vectors`` -- two hundredths of
a millisecond -- spread by 33% on the same runs. Anything below
``MINIMUM_GATED_MILLISECONDS`` in the baseline is printed with the rest and
marked ``(ungated)``.

**Work counts and the total energy are compared too.** They are not timings and
they are not gated, but a deck whose FFT count or converged energy moved is not
the same calculation as the baseline, so its timing comparison is not like for
like. That is reported on the deck's line rather than left for someone to
notice.

Baselines are per machine. Two machines' numbers are not comparable, so a
baseline carries the machine label it was taken on and this script refuses to
compare across labels unless ``--ignore-machine-mismatch`` is passed.
"""

import argparse
import json
import statistics
import sys
from pathlib import Path

# Stages worth gating. The Hamiltonian sub-component timers (hamiltonian_fft,
# _kinetic, _nonlocal, _allreduce) are deliberately absent: counters.hpp
# documents that they are not a partition of hamiltonian_application -- they
# are accumulated in the applicator, reached from paths the parent timer does
# not wrap -- so gating both would charge one regression twice, against
# numbers that do not add up. They are still recorded and printed.
TRACKED_TIMING_KEYS = (
    "total",
    "hamiltonian_application",
    "orthogonalize",
    "subspace_update",
    "diagonalize",
    "collapse",
    "seed",
)

# Below this, a stage's own timer granularity and the scheduler dominate any
# real change, so the stage is reported but never fails the run.
MINIMUM_GATED_MILLISECONDS = 25.0

DEFAULT_REGRESSION_THRESHOLD = 0.10

# How many relative standard deviations a change must clear where the measured
# spread is wider than the threshold. Three keeps a single comparison of
# medians from flagging the machine.
SIGMA_MULTIPLIER = 3.0

# Two runs whose total energies differ by more than this are not the same
# calculation, whatever their timings say.
ENERGY_DIFFERENCE_RYDBERG = 1.0e-6


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "records", type=Path, help="records.json from scripts/benchmark/run_benchmarks.py"
    )
    parser.add_argument("--baseline", type=Path, help="baseline JSON to compare against")
    parser.add_argument("--write-baseline", type=Path, help="write this run out as a baseline")
    parser.add_argument(
        "--note",
        help="free-text note stored in a written baseline (how and where it was taken)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_REGRESSION_THRESHOLD,
        help=(
            "floor for the relative slowdown counted as a regression. Default "
            f"{DEFAULT_REGRESSION_THRESHOLD:.0%}; widened per deck to three "
            "relative standard deviations where the measured spread is larger."
        ),
    )
    parser.add_argument(
        "--ignore-machine-mismatch",
        action="store_true",
        help="compare against a baseline taken on a different machine label anyway",
    )
    return parser.parse_args()


def load_payload(path, expected_schema):
    with path.open(encoding="utf-8") as file_handle:
        payload = json.load(file_handle)
    schema = payload.get("schema")
    if schema != expected_schema:
        raise ValueError(f"{path} carries schema {schema!r}, expected {expected_schema!r}")
    return payload


def spread(values):
    """Relative standard deviation, or zero where one value cannot show one."""
    if len(values) < 2:
        return 0.0
    middle = statistics.median(values)
    return statistics.stdev(values) / middle if middle else 0.0


def reduce_records(payload):
    """One entry per deck: medians over the repetitions, first one dropped."""
    grouped = {}
    for record in payload.get("records", []):
        grouped.setdefault(record["deck"], []).append(record)

    reduced = {}
    for deck, records in grouped.items():
        records.sort(key=lambda record: record["repetition"])
        kept = records[1:] if len(records) > 1 else records
        stage_names = sorted({name for record in kept for name in record["timings_ms"]})
        stages = {}
        for name in stage_names:
            values = [record["timings_ms"][name] for record in kept if name in record["timings_ms"]]
            # Rounded to the microsecond: a baseline is a committed file that
            # people read and diff, and the digits below this are the timer's
            # own noise rather than anything about the code.
            stages[name] = {
                "median_ms": round(statistics.median(values), 3),
                "minimum_ms": round(min(values), 3),
                "maximum_ms": round(max(values), 3),
                "relative_spread": round(spread(values), 5),
            }
        wall_values = [record["wall_seconds"] for record in kept]
        reduced[deck] = {
            "repetitions_kept": len(kept),
            "repetitions_total": len(records),
            "stages": stages,
            "wall_seconds_median": round(statistics.median(wall_values), 3),
            "wall_seconds_relative_spread": round(spread(wall_values), 5),
            "counts": kept[-1]["counts"],
            "converged": kept[-1]["converged"],
            "self_consistent_field_iterations": kept[-1]["self_consistent_field_iterations"],
            "total_energy_ry": kept[-1]["total_energy_ry"],
        }
    return reduced


def report(reduced):
    print(f"{'deck / stage':<46} {'median ms':>11} {'min':>10} {'max':>10} {'spread':>8}")
    for deck in sorted(reduced):
        entry = reduced[deck]
        print(
            f"{deck}  ({entry['repetitions_kept']} of {entry['repetitions_total']} runs kept, "
            f"wall {entry['wall_seconds_median']:.2f} s, "
            f"{'converged' if entry['converged'] else 'not converged'}, "
            f"E = {entry['total_energy_ry']:.8f} Ry)"
        )
        for name in sorted(entry["stages"]):
            stage = entry["stages"][name]
            print(
                f"    {name:<42} {stage['median_ms']:>11.1f} {stage['minimum_ms']:>10.1f} "
                f"{stage['maximum_ms']:>10.1f} {stage['relative_spread']:>7.1%}"
            )
        counts = ", ".join(f"{key}={value}" for key, value in sorted(entry["counts"].items()))
        print(f"    counts: {counts}")


def stage_threshold(now_stage, baseline_stage, floor):
    """The floor, or three sigma of whichever side is noisier."""
    spreads = [stage.get("relative_spread", 0.0) for stage in (now_stage, baseline_stage)]
    return max(floor, SIGMA_MULTIPLIER * max(spreads))


def describe_deck_differences(now_entry, baseline_entry):
    """Ways a deck may have stopped being the same calculation as the baseline."""
    notes = []
    if now_entry["counts"] != baseline_entry.get("counts"):
        notes.append("work counts differ from the baseline")
    now_energy = now_entry.get("total_energy_ry")
    baseline_energy = baseline_entry.get("total_energy_ry")
    if now_energy is not None and baseline_energy is not None:
        if abs(now_energy - baseline_energy) > ENERGY_DIFFERENCE_RYDBERG:
            notes.append(
                f"total energy moved by {abs(now_energy - baseline_energy):.2e} Ry "
                f"(> {ENERGY_DIFFERENCE_RYDBERG:.0e})"
            )
    if now_entry["converged"] != baseline_entry.get("converged"):
        notes.append("convergence outcome differs from the baseline")
    return notes


def compare_deck(deck, now_entry, baseline_entry, floor):
    """Print one deck's comparison; return the stages that regressed and are gated."""
    regressions = []
    for note in describe_deck_differences(now_entry, baseline_entry):
        print(f"    NOTE: {note} -- the timings below are not a like-for-like comparison")
    for name in TRACKED_TIMING_KEYS:
        now_stage = now_entry["stages"].get(name)
        baseline_stage = (baseline_entry.get("stages") or {}).get(name)
        if now_stage is None or baseline_stage is None:
            continue
        was = baseline_stage["median_ms"]
        now = now_stage["median_ms"]
        limit = stage_threshold(now_stage, baseline_stage, floor)
        gated = was >= MINIMUM_GATED_MILLISECONDS
        ratio = now / was if was else float("inf")
        marker = "" if gated else "  (ungated)"
        if ratio > 1.0 + limit:
            verdict = f"SLOWER {ratio:.2f}x (> {limit:.0%})"
            if gated:
                regressions.append(f"{deck}/{name}")
        elif ratio < 1.0 - limit:
            verdict = f"faster {1 / ratio:.2f}x"
        else:
            verdict = f"within +-{limit:.0%}"
        print(f"    {name:<28} {was:>10.1f} -> {now:>10.1f} ms  {verdict}{marker}")
    return regressions


def compare(reduced, baseline_entries, floor):
    regressions = []
    shared = sorted(set(reduced) & set(baseline_entries))
    for deck in sorted(set(baseline_entries) - set(reduced)):
        print(f"  {deck}: in the baseline, not in this run")
    for deck in sorted(set(reduced) - set(baseline_entries)):
        print(f"  {deck}: new, no baseline entry")
    if not shared:
        raise ValueError("No deck is present in both the run and the baseline")
    for deck in shared:
        print(f"  {deck}")
        regressions.extend(compare_deck(deck, reduced[deck], baseline_entries[deck], floor))
    return regressions


def machine_label(payload):
    return (payload.get("metadata", {}).get("machine") or {}).get("label", "")


def check_machine_labels(records_payload, baseline_payload, ignore_mismatch):
    now_label = machine_label(records_payload)
    baseline_label = machine_label(baseline_payload)
    if now_label == baseline_label:
        return
    message = (
        f"This run is labelled '{now_label}' and the baseline '{baseline_label}'. "
        "Timings from different machines are not comparable."
    )
    if not ignore_mismatch:
        raise ValueError(message + " Pass --ignore-machine-mismatch to compare anyway.")
    print(f"WARNING: {message}\n")


def write_baseline(path, records_payload, reduced, note):
    """Store the reduced run as a baseline, provenance included.

    A baseline that cannot say which commit and which machine produced it is
    not the other half of a comparison -- "slower than some earlier build" is
    not a finding -- so this refuses to write one.
    """
    metadata = dict(records_payload.get("metadata", {}))
    if not metadata.get("git_commit"):
        raise ValueError("Refusing to write a baseline whose records carry no git_commit")
    if not machine_label(records_payload):
        raise ValueError("Refusing to write a baseline whose records carry no machine label")
    if metadata.get("git_worktree_dirty"):
        print(
            f"Note: baseline taken from a dirty worktree at {metadata['git_commit'][:12]}; "
            "it records uncommitted state and cannot be reproduced from the commit alone."
        )
    if note:
        metadata["note"] = note
    payload = {"schema": "q3-benchmark-baseline-1", "metadata": metadata, "entries": reduced}
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file_handle:
        json.dump(payload, file_handle, indent=4, sort_keys=True)
        file_handle.write("\n")
    print(f"\nBaseline written: {path}")


def main():
    arguments = parse_arguments()
    records_payload = load_payload(arguments.records, "q3-benchmark-records-1")
    reduced = reduce_records(records_payload)
    if not reduced:
        raise ValueError(f"{arguments.records} holds no records")
    report(reduced)

    if arguments.write_baseline:
        write_baseline(arguments.write_baseline, records_payload, reduced, arguments.note)

    if arguments.baseline:
        baseline_payload = load_payload(arguments.baseline, "q3-benchmark-baseline-1")
        check_machine_labels(records_payload, baseline_payload, arguments.ignore_machine_mismatch)
        print(
            f"\nAgainst {arguments.baseline} "
            f"(floor {arguments.threshold:.0%}, or three sigma where that is wider):"
        )
        regressions = compare(reduced, baseline_payload.get("entries", {}), arguments.threshold)
        if regressions:
            print(f"\n{len(regressions)} regression(s) beyond the threshold:")
            for regression in regressions:
                print(f"  {regression}")
            return 1
        print("\nNo tracked stage regressed beyond its threshold.")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except ValueError as error:
        print(str(error), file=sys.stderr)
        sys.exit(1)
