"""tests/python/test_compare_benchmarks.py — Unit tests for the benchmark comparison.

Covers ``scripts/benchmark/compare_benchmarks.py``: the reduction of repeated runs to a
median and a spread (including the dropped first repetition), the threshold
rule that widens to the noise the decks actually showed, the floor below which
a stage is reported but not gated, the ways a deck can stop being the same
calculation as its baseline, and the refusal to compare or record a baseline
that cannot say where it came from.

A regression is asserted through the exit code as well as the return value:
the CI job reads the exit code, so a comparison that printed a regression and
exited 0 would be green and wrong.
"""

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent.parent
SCRIPT_PATH = REPOSITORY_ROOT / "scripts" / "benchmark" / "compare_benchmarks.py"


def _load_comparator():
    """Import the comparison script as a module (it is a script, not a package)."""
    spec = importlib.util.spec_from_file_location("compare_benchmarks", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


comparator = _load_comparator()


def make_record(repetition, total_ms, *, deck="silicon", counts=None, energy=-16.913417):
    return {
        "deck": deck,
        "repetition": repetition,
        "machine": "test-runner",
        "commit": "0" * 40,
        "wall_seconds": total_ms / 1000.0 + 1.0,
        "timings_ms": {
            "total": total_ms,
            "hamiltonian_application": total_ms / 2.0,
            "diagonalize": 2.0,
        },
        "counts": counts or {"fft_calls": 929, "davidson_iterations": 53},
        "converged": True,
        "self_consistent_field_iterations": 7,
        "total_energy_ry": energy,
    }


def make_records_payload(records, machine="test-runner"):
    return {
        "schema": "q3-benchmark-records-1",
        "metadata": {
            "recorded_at_utc": "2026-09-21T00:00:00Z",
            "git_commit": "a" * 40,
            "git_worktree_dirty": False,
            "machine": {"label": machine, "processor": "stub", "logical_cores": 4},
        },
        "records": records,
    }


# ── Reducing repeated runs ────────────────────────────────────────────────────


def test_the_first_repetition_is_dropped():
    """It pays the cold page cache and the allocator's first growth; the others do not."""
    payload = make_records_payload(
        [make_record(1, 1000.0), make_record(2, 100.0), make_record(3, 102.0)]
    )
    entry = comparator.reduce_records(payload)["silicon"]

    assert entry["repetitions_total"] == 3
    assert entry["repetitions_kept"] == 2
    assert entry["stages"]["total"]["median_ms"] == 101.0


def test_a_single_repetition_is_kept_rather_than_discarded():
    """Dropping the only run would leave nothing to compare."""
    payload = make_records_payload([make_record(1, 100.0)])
    entry = comparator.reduce_records(payload)["silicon"]

    assert entry["repetitions_kept"] == 1
    assert entry["stages"]["total"]["median_ms"] == 100.0
    assert entry["stages"]["total"]["relative_spread"] == 0.0


def test_reduction_records_the_spread_it_measured():
    payload = make_records_payload(
        [make_record(1, 500.0), make_record(2, 100.0), make_record(3, 120.0)]
    )
    entry = comparator.reduce_records(payload)["silicon"]

    assert entry["stages"]["total"]["minimum_ms"] == 100.0
    assert entry["stages"]["total"]["maximum_ms"] == 120.0
    assert entry["stages"]["total"]["relative_spread"] == pytest.approx(0.1286, abs=1e-3)


def test_each_deck_is_reduced_separately():
    payload = make_records_payload(
        [make_record(1, 100.0), make_record(1, 900.0, deck="copper")],
    )
    reduced = comparator.reduce_records(payload)

    assert sorted(reduced) == ["copper", "silicon"]


# ── The threshold ─────────────────────────────────────────────────────────────


def test_the_threshold_floor_applies_to_a_quiet_deck():
    quiet = {"relative_spread": 0.001}
    assert comparator.stage_threshold(quiet, quiet, 0.10) == 0.10


def test_the_threshold_widens_to_a_noisy_deck_s_own_spread():
    """A deck that spread by 8% cannot be gated at 10% without flagging the machine."""
    noisy = {"relative_spread": 0.08}
    quiet = {"relative_spread": 0.001}
    assert comparator.stage_threshold(noisy, quiet, 0.10) == pytest.approx(0.24)


# ── Comparing against a baseline ──────────────────────────────────────────────


def reduce(records):
    return comparator.reduce_records(make_records_payload(records))


def test_a_slowdown_beyond_the_threshold_is_a_regression(capsys):
    baseline = reduce([make_record(1, 100.0), make_record(2, 100.0), make_record(3, 100.0)])
    now = reduce([make_record(1, 150.0), make_record(2, 150.0), make_record(3, 150.0)])

    regressions = comparator.compare(now, baseline, 0.10)

    assert regressions == ["silicon/total", "silicon/hamiltonian_application"]
    assert "SLOWER 1.50x" in capsys.readouterr().out


def test_a_slowdown_within_the_threshold_is_not_a_regression():
    baseline = reduce([make_record(1, 100.0), make_record(2, 100.0), make_record(3, 100.0)])
    now = reduce([make_record(1, 105.0), make_record(2, 105.0), make_record(3, 105.0)])

    assert comparator.compare(now, baseline, 0.10) == []


def test_a_small_stage_is_reported_but_not_gated(capsys):
    """diagonalize is 2 ms here: a doubling is the scheduler, not a regression."""
    baseline = reduce([make_record(1, 100.0), make_record(2, 100.0)])
    now = reduce([make_record(1, 100.0), make_record(2, 100.0)])
    now["silicon"]["stages"]["diagonalize"]["median_ms"] = 6.0

    regressions = comparator.compare(now, baseline, 0.10)

    assert regressions == []
    assert "(ungated)" in capsys.readouterr().out


def test_a_deck_missing_from_the_run_is_reported_not_compared(capsys):
    baseline = reduce([make_record(1, 100.0), make_record(1, 900.0, deck="copper")])
    now = reduce([make_record(1, 100.0)])

    assert comparator.compare(now, baseline, 0.10) == []
    assert "in the baseline, not in this run" in capsys.readouterr().out


def test_no_overlap_at_all_is_an_error():
    baseline = reduce([make_record(1, 100.0, deck="copper")])
    now = reduce([make_record(1, 100.0)])

    with pytest.raises(ValueError, match="present in both"):
        comparator.compare(now, baseline, 0.10)


# ── Is it even the same calculation? ──────────────────────────────────────────


def test_changed_work_counts_are_reported():
    now = reduce([make_record(1, 100.0, counts={"fft_calls": 1200})])["silicon"]
    baseline = reduce([make_record(1, 100.0, counts={"fft_calls": 929})])["silicon"]

    assert comparator.describe_deck_differences(now, baseline) == [
        "work counts differ from the baseline"
    ]


def test_a_moved_total_energy_is_reported():
    now = reduce([make_record(1, 100.0, energy=-16.9)])["silicon"]
    baseline = reduce([make_record(1, 100.0, energy=-16.913417)])["silicon"]

    assert "total energy moved" in comparator.describe_deck_differences(now, baseline)[0]


def test_an_unchanged_deck_reports_no_differences():
    now = reduce([make_record(1, 100.0)])["silicon"]
    baseline = reduce([make_record(1, 110.0)])["silicon"]

    assert comparator.describe_deck_differences(now, baseline) == []


# ── Machines ──────────────────────────────────────────────────────────────────


def test_comparing_across_machines_is_refused():
    with pytest.raises(ValueError, match="not comparable"):
        comparator.check_machine_labels(
            make_records_payload([], machine="laptop"),
            {"metadata": {"machine": {"label": "ci-runner"}}},
            ignore_mismatch=False,
        )


def test_comparing_across_machines_can_be_forced(capsys):
    comparator.check_machine_labels(
        make_records_payload([], machine="laptop"),
        {"metadata": {"machine": {"label": "ci-runner"}}},
        ignore_mismatch=True,
    )
    assert "WARNING" in capsys.readouterr().out


# ── Writing a baseline ────────────────────────────────────────────────────────


def test_a_written_baseline_carries_its_provenance(tmp_path):
    payload = make_records_payload([make_record(1, 100.0)])
    path = tmp_path / "baseline.json"

    comparator.write_baseline(path, payload, comparator.reduce_records(payload), "on a quiet box")

    stored = json.loads(path.read_text())
    assert stored["schema"] == "q3-benchmark-baseline-1"
    assert stored["metadata"]["git_commit"] == "a" * 40
    assert stored["metadata"]["machine"]["label"] == "test-runner"
    assert stored["metadata"]["note"] == "on a quiet box"
    assert stored["entries"]["silicon"]["stages"]["total"]["median_ms"] == 100.0


def test_a_baseline_without_a_commit_is_refused(tmp_path):
    """ "Slower than some earlier build" is not a finding."""
    payload = make_records_payload([make_record(1, 100.0)])
    payload["metadata"]["git_commit"] = ""

    with pytest.raises(ValueError, match="git_commit"):
        comparator.write_baseline(
            tmp_path / "baseline.json", payload, comparator.reduce_records(payload), None
        )


def test_a_baseline_without_a_machine_is_refused(tmp_path):
    payload = make_records_payload([make_record(1, 100.0)])
    payload["metadata"]["machine"] = {}

    with pytest.raises(ValueError, match="machine label"):
        comparator.write_baseline(
            tmp_path / "baseline.json", payload, comparator.reduce_records(payload), None
        )


# ── The command line ──────────────────────────────────────────────────────────


def test_a_file_of_the_wrong_schema_is_refused(tmp_path):
    path = tmp_path / "records.json"
    path.write_text(json.dumps({"schema": "something-else"}))

    with pytest.raises(ValueError, match="expected"):
        comparator.load_payload(path, "q3-benchmark-records-1")


def write_json(path, payload):
    path.write_text(json.dumps(payload))
    return path


def run_script(*arguments):
    return subprocess.run(
        [sys.executable, str(SCRIPT_PATH), *arguments], capture_output=True, text=True, check=False
    )


def test_a_regression_exits_non_zero(tmp_path):
    """The exit code is what the CI job reads."""
    baseline_records = make_records_payload([make_record(1, 100.0), make_record(2, 100.0)])
    baseline_path = tmp_path / "baseline.json"
    comparator.write_baseline(
        baseline_path, baseline_records, comparator.reduce_records(baseline_records), None
    )
    records_path = write_json(
        tmp_path / "records.json",
        make_records_payload([make_record(1, 400.0), make_record(2, 400.0)]),
    )

    completed = run_script(str(records_path), "--baseline", str(baseline_path))

    assert completed.returncode == 1
    assert "SLOWER" in completed.stdout


def test_an_unchanged_run_exits_zero(tmp_path):
    records = make_records_payload([make_record(1, 100.0), make_record(2, 100.0)])
    baseline_path = tmp_path / "baseline.json"
    comparator.write_baseline(baseline_path, records, comparator.reduce_records(records), None)
    records_path = write_json(tmp_path / "records.json", records)

    completed = run_script(str(records_path), "--baseline", str(baseline_path))

    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "No tracked stage regressed" in completed.stdout
