"""tests/python/test_check_pull_request_size.py — Unit tests for the size report.

Covers ``scripts/repo/check_pull_request_size.py``: which paths are exempt from the
count, that deletions count the same as additions, the over-ceiling wording,
and that the check stays advisory unless asked to gate.
"""

import importlib.util
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent.parent
SCRIPT_PATH = REPOSITORY_ROOT / "scripts" / "repo" / "check_pull_request_size.py"


def _load_checker():
    """Import the checker script as a module (it is a script, not a package)."""
    spec = importlib.util.spec_from_file_location("check_pull_request_size", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


checker = _load_checker()


# ── What is exempt ────────────────────────────────────────────────────────────


def test_generated_headers_are_not_counted():
    """Reviewing the schema is reviewing the header it generates."""
    assert checker.is_excluded("src/generated/q3/input.hpp")
    assert not checker.is_excluded("schemas/q3/input.json")


def test_the_regenerated_statistics_file_is_not_counted():
    assert checker.is_excluded("STATISTICS.md")
    assert not checker.is_excluded("FUNCTIONALITY.md")


def test_an_addons_regenerated_statistics_file_is_not_counted():
    assert checker.is_excluded("src/addons/alpha/STATISTICS.md")
    assert not checker.is_excluded("src/addons/alpha/FUNCTIONALITY.md")


def test_vendored_and_lock_files_are_not_counted():
    assert checker.is_excluded("schemas/esse/definitions/thing.json")
    assert checker.is_excluded("package-lock.json")
    assert checker.is_excluded("web/package-lock.json")


def test_committed_datasets_are_not_counted():
    """They are measured in bytes, not lines."""
    assert checker.is_excluded("examples/data/pseudopotentials/nc/Si.upf")
    assert not checker.is_excluded("examples/total_energy--semiconductor--silicon/nc.yml")


def test_ordinary_source_is_counted():
    assert not checker.is_excluded("src/layer_4_computation/flavors/result.hpp")
    assert not checker.is_excluded("tests/unit/io/test_launcher.cpp")


# ── The measurement ───────────────────────────────────────────────────────────


def test_deletions_count_the_same_as_additions():
    """ "Was any of that load-bearing?" is asked of every deleted line too."""
    measurement = checker.measure([(0, 500, "src/a.hpp"), (500, 0, "src/b.hpp")])
    assert measurement["counted_lines"] == 1000
    assert measurement["counted_files"] == 2


def test_exempt_files_are_reported_separately_not_silently_dropped():
    measurement = checker.measure([(10, 0, "src/a.hpp"), (9000, 0, "src/generated/q3/input.hpp")])
    assert measurement["counted_lines"] == 10
    assert measurement["excluded_lines"] == 9000
    assert measurement["excluded_files"] == 1


def test_a_binary_file_contributes_no_lines_but_still_appears():
    """git reports '-' for both counts on a binary file."""
    measurement = checker.measure([(0, 0, "docs/diagram.png")])
    assert measurement["counted_lines"] == 0
    assert measurement["counted_files"] == 1


def test_the_largest_files_are_listed_biggest_first():
    measurement = checker.measure(
        [(10, 0, "src/small.hpp"), (300, 0, "src/big.hpp"), (50, 0, "src/middle.hpp")]
    )
    assert [path for _, path in measurement["largest"]] == [
        "src/big.hpp",
        "src/middle.hpp",
        "src/small.hpp",
    ]


# ── The report and the exit status ────────────────────────────────────────────


def test_a_change_under_the_ceiling_says_so_without_advice():
    report = checker.format_report(checker.measure([(100, 0, "src/a.hpp")]), 3000)
    assert "100 reviewable lines" in report
    assert "over" not in report


def test_a_change_over_the_ceiling_says_by_how_much():
    report = checker.format_report(checker.measure([(5000, 0, "src/a.hpp")]), 3000)
    assert "2000 over" in report
    assert "cannot be split" in report


def test_the_ceiling_is_advisory_by_default(monkeypatch, capsys):
    monkeypatch.setattr(checker, "numstat", lambda base, head: [(9999, 0, "src/a.hpp")])
    assert checker.main(["--base", "x", "--head", "y"]) == 0
    assert "over" in capsys.readouterr().out


def test_the_ceiling_can_be_asked_to_gate(monkeypatch):
    monkeypatch.setattr(checker, "numstat", lambda base, head: [(9999, 0, "src/a.hpp")])
    assert checker.main(["--base", "x", "--head", "y", "--fail-over-ceiling"]) == 1


def test_an_unreadable_range_is_an_error_not_a_zero(monkeypatch, capsys):
    assert checker.main(["--base", "no-such-ref", "--head", "HEAD"]) == 2
    assert "Could not diff" in capsys.readouterr().err
