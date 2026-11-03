#!/usr/bin/env python3
"""Render the committed benchmark baselines into PERFORMANCE.md.

    # Rewrite the generated section in place:
    python scripts/benchmark/generate_performance.py

    # Verify it is up to date (exit 1 if stale), for CI:
    python scripts/benchmark/generate_performance.py --check

Unlike STATISTICS.md, PERFORMANCE.md is not generated end to end. Most of it is
a written record -- what was measured against Quantum ESPRESSO, what a fix was
worth, why a number is what it is -- and regenerating that from data would
throw away the part with the reasoning in it. So only one section is generated,
the tracked benchmark decks that ``scripts/benchmark/run_benchmarks.py`` re-measures on
demand, and it is delimited in the document by:

    <!-- generated:benchmarks:begin -->
    <!-- generated:benchmarks:end -->

Everything outside those markers is written by hand and is never touched here.
One authored file, one diff, no second copy of the same content.

The input is every ``docs/benchmarks/baseline-*.json`` carrying the baseline
schema ``scripts/benchmark/compare_benchmarks.py`` writes; the Perlmutter baselines from
the MPI matrix runner use a different schema and are skipped. Each baseline is
rendered under its machine label, because timings from two machines are not
comparable and a table that mixed them would invite exactly that comparison.
"""

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_PATH = PROJECT_ROOT / "PERFORMANCE.md"
BASELINE_DIRECTORY = PROJECT_ROOT / "docs" / "benchmarks"
BASELINE_SCHEMA = "q3-benchmark-baseline-1"

BEGIN_MARKER = "<!-- generated:benchmarks:begin -->"
END_MARKER = "<!-- generated:benchmarks:end -->"

# Stage columns, in the order the solver reaches them. Kept short and ASCII:
# the table is formatted to match Prettier's own column padding, and that
# calculation is only reliable for characters one column wide.
STAGE_COLUMNS = (
    ("seed", "seed"),
    ("orthogonalize", "orthog."),
    ("hamiltonian_application", "H apply"),
    ("subspace_update", "subspace"),
    ("diagonalize", "diag."),
    ("collapse", "collapse"),
)


def render_table(headers, alignments, rows):
    """A Markdown table padded the way Prettier 3.1.0 pads one.

    Written out rather than left ragged because the repository formats
    Markdown with Prettier at a pinned version, and a generator whose output
    the formatter then rewrites makes every regeneration a two-step dance with
    a dirty tree in between.
    """
    columns = len(headers)
    widths = []
    for index in range(columns):
        cells = [headers[index], *(row[index] for row in rows)]
        widths.append(max(3, *(len(cell) for cell in cells)))

    def pad(text, width, alignment):
        if alignment == "right":
            return text.rjust(width)
        return text.ljust(width)

    def delimiter(width, alignment):
        return "-" * (width - 1) + ":" if alignment == "right" else "-" * width

    lines = [
        "| " + " | ".join(pad(headers[i], widths[i], alignments[i]) for i in range(columns)) + " |",
        "| " + " | ".join(delimiter(widths[i], alignments[i]) for i in range(columns)) + " |",
    ]
    for row in rows:
        lines.append(
            "| " + " | ".join(pad(row[i], widths[i], alignments[i]) for i in range(columns)) + " |"
        )
    return lines


def load_baselines():
    """Every baseline file in docs/benchmarks/, oldest machine label first."""
    baselines = []
    for path in sorted(BASELINE_DIRECTORY.glob("baseline-*.json")):
        with path.open(encoding="utf-8") as file_handle:
            payload = json.load(file_handle)
        if payload.get("schema") != BASELINE_SCHEMA:
            continue
        baselines.append((path, payload))
    return baselines


def format_milliseconds(value):
    return f"{value:,.1f}" if value is not None else "-"


def format_count(value):
    return f"{value:,}" if value is not None else "-"


def describe_machine(metadata):
    machine = metadata.get("machine") or {}
    parts = [
        f"**{machine.get('label', 'unknown machine')}**",
        f"{machine.get('processor', 'unknown CPU')},",
        f"{machine.get('logical_cores', 0)} logical cores,",
        f"{machine.get('system', '')} {machine.get('release', '')}".strip() + ".",
    ]
    return " ".join(parts)


def describe_source(path):
    """The baseline's path as the repository sees it, or its name if it is elsewhere."""
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return path.name


def render_baseline(path, payload):
    metadata = payload.get("metadata", {})
    entries = payload.get("entries", {})
    lines = [f"### {metadata.get('machine', {}).get('label', path.stem)}", ""]
    lines.append(describe_machine(metadata))
    commit = metadata.get("git_commit", "")
    lines.append(
        f"Recorded {metadata.get('recorded_at_utc', 'at an unrecorded time')} from commit "
        f"`{commit[:12] or 'unknown'}`"
        + (", worktree dirty" if metadata.get("git_worktree_dirty") else "")
        + f". Source: `{describe_source(path)}`."
    )
    # A note is a paragraph, not a block quote (AGENTS.md: notes are paragraphs).
    if metadata.get("note"):
        lines.extend(["", metadata["note"]])
    lines.append("")

    summary_rows = []
    stage_rows = []
    for deck in sorted(entries):
        entry = entries[deck]
        stages = entry.get("stages", {})
        counts = entry.get("counts", {})
        total = stages.get("total", {})
        summary_rows.append(
            [
                f"`{deck}`",
                f"{entry.get('wall_seconds_median', 0.0):,.2f}",
                format_milliseconds(total.get("median_ms")),
                f"{total.get('relative_spread', 0.0):.1%}",
                str(entry.get("self_consistent_field_iterations") or "-"),
                format_count(counts.get("davidson_iterations")),
                format_count(counts.get("fft_calls")),
                f"{entry.get('total_energy_ry', float('nan')):.6f}",
            ]
        )
        stage_rows.append(
            [f"`{deck}`"]
            + [
                format_milliseconds((stages.get(key) or {}).get("median_ms"))
                for key, _ in STAGE_COLUMNS
            ]
        )

    lines.extend(
        render_table(
            ["Deck", "Wall (s)", "Solver (ms)", "Spread", "SCF", "Davidson", "FFT", "Energy (Ry)"],
            ["left", "right", "right", "right", "right", "right", "right", "right"],
            summary_rows,
        )
    )
    lines.extend(["", "Solver stage medians, milliseconds:", ""])
    lines.extend(
        render_table(
            ["Deck", *(label for _, label in STAGE_COLUMNS)],
            ["left", *("right" for _ in STAGE_COLUMNS)],
            stage_rows,
        )
    )
    return lines


def render_section():
    baselines = load_baselines()
    lines = [
        "## Tracked Benchmark Decks",
        "",
        "Generated by `scripts/benchmark/generate_performance.py` from the baselines in",
        "`docs/benchmarks/`. Do not edit this section by hand; edit the baseline it came",
        "from, or take a new one. Everything outside the generated markers in this file is",
        "written by hand.",
        "",
        "The decks are listed in `tests/regression/benchmark_matrix.yml` and re-measured by",
        "`scripts/benchmark/run_benchmarks.py`, which reads the per-phase figures out of the",
        "`timings_ms` block the engine already writes into its results file. Runs are single",
        "threaded with deterministic FFT planning; `Solver (ms)` is the Davidson total, which",
        "excludes process start-up, input parsing and the property run that `Wall (s)` covers.",
        "`Spread` is the relative standard deviation of the repetitions kept, and is the",
        "figure that says how large a change has to be before it means anything.",
        "",
        "Numbers from two machines are not comparable, so each machine is listed separately",
        "and `scripts/benchmark/compare_benchmarks.py` refuses to compare across them.",
        "",
    ]
    if not baselines:
        lines.append("No benchmark baselines are committed yet.")
        return lines
    for index, (path, payload) in enumerate(baselines):
        if index:
            lines.append("")
        lines.extend(render_baseline(path, payload))
    return lines


def splice(document, section_lines):
    if BEGIN_MARKER not in document or END_MARKER not in document:
        raise ValueError(
            f"{OUTPUT_PATH.name} carries no generated-section markers. Add "
            f"{BEGIN_MARKER} and {END_MARKER} around the section this generator owns."
        )
    head, _, remainder = document.partition(BEGIN_MARKER)
    _, _, tail = remainder.partition(END_MARKER)
    body = "\n".join(section_lines).rstrip()
    return f"{head}{BEGIN_MARKER}\n\n{body}\n\n{END_MARKER}{tail}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="verify the generated section is up-to-date (exit 1 if stale)",
    )
    arguments = parser.parse_args()

    document = OUTPUT_PATH.read_text(encoding="utf-8")
    updated = splice(document, render_section())

    if arguments.check:
        if document != updated:
            print(
                "PERFORMANCE.md's generated section is stale. Run "
                "'python scripts/benchmark/generate_performance.py'.",
                file=sys.stderr,
            )
            return 1
        print("PERFORMANCE.md's generated section is up-to-date.")
        return 0

    OUTPUT_PATH.write_text(updated, encoding="utf-8")
    print(f"Wrote {OUTPUT_PATH.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except ValueError as error:
        print(str(error), file=sys.stderr)
        sys.exit(1)
