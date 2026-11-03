"""tests/python/test_generate_performance.py — Unit tests for the PERFORMANCE.md generator.

Covers ``scripts/benchmark/generate_performance.py``: the Markdown table padding (which
has to match the Prettier version this repository pins, or every regeneration
leaves a tree the formatter then rewrites), the splice that replaces only what
is between the generated markers, the per-machine grouping, and the property
the whole design rests on -- that the hand-written body of PERFORMANCE.md
survives a regeneration untouched.

The last test asserts the committed PERFORMANCE.md is up to date, the same way
``test_check_doc_comment_style.py`` asserts the tree is clean: the rule holds
for a developer who has not installed the hooks.
"""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent.parent
SCRIPT_PATH = REPOSITORY_ROOT / "scripts" / "benchmark" / "generate_performance.py"


def _load_generator():
    """Import the generator script as a module (it is a script, not a package)."""
    spec = importlib.util.spec_from_file_location("generate_performance", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


generator = _load_generator()


def make_baseline(label="test-runner", note=None, deck="silicon"):
    metadata = {
        "recorded_at_utc": "2026-09-21T00:00:00Z",
        "git_commit": "abcdef0123456789" + "0" * 24,
        "git_worktree_dirty": False,
        "machine": {
            "label": label,
            "processor": "Stub CPU",
            "logical_cores": 4,
            "system": "Linux",
            "release": "6.18",
        },
    }
    if note:
        metadata["note"] = note
    return {
        "schema": "q3-benchmark-baseline-1",
        "metadata": metadata,
        "entries": {
            deck: {
                "repetitions_kept": 2,
                "repetitions_total": 3,
                "wall_seconds_median": 1.25,
                "wall_seconds_relative_spread": 0.01,
                "counts": {
                    "fft_calls": 929,
                    "davidson_iterations": 53,
                    "hamiltonian_applications": 320,
                },
                "converged": True,
                "self_consistent_field_iterations": 7,
                "total_energy_ry": -16.913417,
                "stages": {
                    "total": {
                        "median_ms": 290.0,
                        "minimum_ms": 285.0,
                        "maximum_ms": 295.0,
                        "relative_spread": 0.017,
                    },
                    "seed": {"median_ms": 86.7, "relative_spread": 0.01},
                    "hamiltonian_application": {"median_ms": 125.6, "relative_spread": 0.01},
                },
            }
        },
    }


@pytest.fixture
def baseline_directory(tmp_path, monkeypatch):
    monkeypatch.setattr(generator, "BASELINE_DIRECTORY", tmp_path)
    return tmp_path


def write_baseline(directory, name, payload):
    path = directory / name
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


# ── Table formatting ──────────────────────────────────────────────────────────


def test_columns_are_padded_to_their_widest_cell():
    lines = generator.render_table(
        ["Deck", "Wall (s)"], ["left", "right"], [["`silicon_nc`", "1.25"]]
    )
    assert lines[0] == "| Deck         | Wall (s) |"
    assert lines[1] == "| ------------ | -------: |"
    assert lines[2] == "| `silicon_nc` |     1.25 |"


def test_a_column_narrower_than_three_characters_is_widened_to_three():
    """Prettier's minimum column width; a narrower delimiter is not a table."""
    lines = generator.render_table(["a"], ["left"], [["b"]])
    assert lines == ["| a   |", "| --- |", "| b   |"]


def test_every_row_has_the_same_width():
    lines = generator.render_table(
        ["Deck", "Solver (ms)"], ["left", "right"], [["a", "1.0"], ["bbbbbbbb", "20000.0"]]
    )
    assert len({len(line) for line in lines}) == 1


# ── The splice ────────────────────────────────────────────────────────────────


DOCUMENT = (
    "# Performance\n\nHand written above.\n\n"
    f"{generator.BEGIN_MARKER}\n\nold content\n\n{generator.END_MARKER}\n\n"
    "Hand written below.\n"
)


def test_only_the_region_between_the_markers_is_replaced():
    spliced = generator.splice(DOCUMENT, ["new content"])
    assert "Hand written above." in spliced
    assert "Hand written below." in spliced
    assert "old content" not in spliced
    assert "new content" in spliced


def test_splicing_is_idempotent():
    once = generator.splice(DOCUMENT, ["new content"])
    assert generator.splice(once, ["new content"]) == once


def test_a_document_without_markers_is_an_error():
    with pytest.raises(ValueError, match="markers"):
        generator.splice("# Performance\n\nNo markers here.\n", ["anything"])


# ── Rendering baselines ───────────────────────────────────────────────────────


def test_a_baseline_is_rendered_under_its_machine_label(baseline_directory):
    write_baseline(baseline_directory, "baseline-test-runner.json", make_baseline())
    rendered = "\n".join(generator.render_section())

    assert "### test-runner" in rendered
    assert "Stub CPU" in rendered
    assert "`abcdef012345`" in rendered
    assert "`silicon`" in rendered
    assert "290.0" in rendered


def test_each_machine_gets_its_own_section(baseline_directory):
    """Timings from two machines are not comparable, so they are never one table."""
    write_baseline(baseline_directory, "baseline-one.json", make_baseline(label="one"))
    write_baseline(baseline_directory, "baseline-two.json", make_baseline(label="two"))
    rendered = "\n".join(generator.render_section())

    assert "### one" in rendered
    assert "### two" in rendered


def test_a_baseline_note_is_carried_into_the_document(baseline_directory):
    """A caveat about how a baseline was taken has to travel with its numbers, as a paragraph."""
    write_baseline(
        baseline_directory, "baseline-test.json", make_baseline(note="taken on a busy box")
    )
    assert "taken on a busy box" in generator.render_section()


def test_foreign_baseline_schemas_are_skipped(baseline_directory):
    """docs/benchmarks/ also holds the MPI matrix runner's baselines."""
    write_baseline(baseline_directory, "baseline-perlmutter.json", {"entries": {"case/topo": {}}})
    assert "No benchmark baselines are committed yet." in "\n".join(generator.render_section())


def test_no_baselines_renders_a_statement_not_an_empty_table(baseline_directory):
    assert "No benchmark baselines are committed yet." in "\n".join(generator.render_section())


# ── The committed document ────────────────────────────────────────────────────


def test_committed_performance_document_is_up_to_date():
    """Regenerating changes nothing, so no one has hand-edited the generated section."""
    document = generator.OUTPUT_PATH.read_text(encoding="utf-8")
    assert generator.splice(document, generator.render_section()) == document


def test_regenerating_preserves_the_hand_written_body():
    """The part of PERFORMANCE.md with the reasoning in it is never rewritten."""
    document = generator.OUTPUT_PATH.read_text(encoding="utf-8")
    regenerated = generator.splice(document, generator.render_section())
    head = document.split(generator.BEGIN_MARKER)[0]
    tail = document.split(generator.END_MARKER)[-1]

    assert regenerated.startswith(head)
    assert regenerated.endswith(tail)
    assert "Resolved Performance Investigations" in tail
