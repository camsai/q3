#!/usr/bin/env python3
"""Compare two Q3 result trees leaf by leaf, exactly by default.

This is the comparison half of the A/B parity gate (the codebase-improvement
program's item A2, re-baselined as M1 in the 2026-09-05 master plan). Its
premise is that a refactor which claims to move code without changing physics
has to *prove* it: the same inputs, run by the binary before and after, must
leave byte-for-byte the same numbers behind. Anything less turns "pure move"
into an unfalsifiable claim, and Q3 merged six ``src/layer_*`` refactors on
that claim alone.

What it compares
----------------

Given two paths, either two ``*_results.json`` files or two directories that a
matrix run filled:

* **Every numeric leaf** of the JSON, addressed by its JSON pointer, compared
  with ``==`` on the parsed ``float``. Exactness is the default because the
  gate's job is to detect *any* drift, and a refactor that genuinely changes
  no arithmetic reproduces its own output bit for bit.
* **Every non-numeric leaf** (strings, booleans, nulls), compared with ``==``.
* **The structure itself**: a key present on one side only, or an array whose
  length moved, is a difference even when no leaf disagrees.
* **Every other file** in directory mode, byte for byte. That is what puts the
  ``.save/`` checkpoint trees under the gate alongside the JSON.

What it ignores, and why
------------------------

``VOLATILE_POINTERS`` lists the leaves that differ between two runs of the
*same* binary, so comparing them would make the gate fail on noise and teach
everyone to skip it. Each entry is a measurement of the run rather than of the
physics: wall-clock timings, peak resident memory, and the engine's own
version and commit -- which differ between merge-base and head by
construction, since that is exactly what the two builds are.

Nothing else is ignorable. There is no committed allowlist and no "expected
difference" file, because a gate whose exceptions grow by editing a list is a
gate that reports whatever its last editor found convenient. The one row-level
exemption, below, belongs to a single pull request and is stated in it.

The ``numerics-change`` escape
------------------------------

A change that is *meant* to move numbers -- a corrected formula, a better
default -- cannot pass an exact gate, and should not have to. Passing
``--tolerance`` switches to a relative comparison, and ``--reason`` is then
mandatory: the escape must be spelled out in the run's own log, not merely
implied by a label. The CI job supplies both from the ``numerics-change``
pull-request label and the label's stated reason.

A tolerance cannot pass a corrected formula, though. The rows it is meant to
move move by more than any tolerance that still catches drift, and their
``.save/`` checkpoints are compared byte for byte whatever the tolerance. So the
escape can also name the matrix rows a change moves, with ``--moving-rows``
(comma-separated row names; ``*`` and ``?`` match). Those rows are reported,
each with its count of differences and the first of them, instead of failing
the comparison; every other row is still held to the tolerance. A pattern that
names no row is refused, so a typo cannot quietly widen the exemption, and a
row declared to move that did not is reported as well. The CI job reads the
rows from a ``numerics-change-rows:`` line in the pull request's body, next to
its reason, so the exemption lives in that one pull request and is printed in
its log.

Usage::

    python scripts/validation/compare_results_json.py BASE HEAD
    python scripts/validation/compare_results_json.py BASE HEAD --tolerance 1e-12 \
        --reason "PDOS moves to the S metric for ultrasoft datasets"
    python scripts/validation/compare_results_json.py BASE HEAD --tolerance 1e-12 \
        --reason "a smeared metal stays neutral" --moving-rows "copper-*,iron-*"

Exit status is 0 when the two sides agree, 1 when they differ, and 2 when the
comparison could not be performed at all.
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import math
import sys
from pathlib import Path

# Leaves that two runs of the same binary disagree on. See the module
# docstring: this list is the whole of the gate's tolerance for noise, and it
# names measurements of the run rather than of the physics.
VOLATILE_POINTERS = (
    "/runtime_log/timings_ms",
    "/runtime_log/execution/peak_resident_memory_bytes",
    "/provenance/generator/engine_version",
    "/provenance/generator/engine_commit",
    # Which build-time add-ons a binary compiled in is a fact about the build,
    # not the physics: a core-only build and one with add-ons compare equal on
    # an input neither add-on touches.
    "/provenance/generator/addons",
)

# How many differing leaves to print before summarizing the rest. A refactor
# that moves one number prints one line; a refactor that moves everything
# should not bury the run log in a hundred thousand of them.
DEFAULT_MAXIMUM_REPORTED = 20


class ComparisonError(RuntimeError):
    """The comparison could not be performed (missing or unreadable input)."""


def is_volatile(pointer: str) -> bool:
    """Answer whether this JSON pointer is at or below a volatile subtree."""
    return any(
        pointer == volatile or pointer.startswith(volatile + "/") for volatile in VOLATILE_POINTERS
    )


def escape_pointer_token(token: str) -> str:
    """Escape one JSON-pointer token (RFC 6901: ``~`` then ``/``)."""
    return token.replace("~", "~0").replace("/", "~1")


def flatten(document, pointer: str = "") -> dict[str, object]:
    """Flatten a parsed JSON document into {JSON pointer: leaf value}.

    Containers contribute a leaf of their own -- an empty object or array has
    no children to compare, so without one it would vanish from the flattened
    view and an emptied array would read as no difference at all.
    """
    if isinstance(document, dict):
        if not document:
            return {pointer: {}}
        flattened: dict[str, object] = {}
        for key, value in document.items():
            flattened.update(flatten(value, f"{pointer}/{escape_pointer_token(str(key))}"))
        return flattened
    if isinstance(document, list):
        if not document:
            return {pointer: []}
        flattened = {}
        for index, value in enumerate(document):
            flattened.update(flatten(value, f"{pointer}/{index}"))
        return flattened
    return {pointer: document}


def values_agree(base_value, head_value, tolerance: float) -> bool:
    """Answer whether two leaf values agree under the active comparison mode.

    Numbers are compared exactly when ``tolerance`` is zero. Booleans are not
    numbers here even though Python says they are: ``True == 1.0`` would let a
    boolean flag flip into an integer without the gate noticing.
    """
    base_is_number = isinstance(base_value, (int, float)) and not isinstance(base_value, bool)
    head_is_number = isinstance(head_value, (int, float)) and not isinstance(head_value, bool)
    if base_is_number != head_is_number:
        return False
    if not base_is_number:
        return type(base_value) is type(head_value) and base_value == head_value

    base_number = float(base_value)
    head_number = float(head_value)
    if math.isnan(base_number) or math.isnan(head_number):
        # NaN never equals itself, but a NaN that stayed a NaN is not drift.
        return math.isnan(base_number) and math.isnan(head_number)
    if tolerance == 0.0:
        return base_number == head_number
    difference = abs(base_number - head_number)
    scale = max(abs(base_number), abs(head_number))
    return difference <= tolerance * scale if scale > 0.0 else difference == 0.0


def compare_documents(base_document, head_document, tolerance: float) -> list[str]:
    """Compare two parsed JSON documents, returning one message per difference."""
    base_leaves = flatten(base_document)
    head_leaves = flatten(head_document)
    differences: list[str] = []

    for pointer in sorted(set(base_leaves) - set(head_leaves)):
        if not is_volatile(pointer):
            differences.append(f"{pointer or '/'}: present in base, missing from head")
    for pointer in sorted(set(head_leaves) - set(base_leaves)):
        if not is_volatile(pointer):
            differences.append(f"{pointer or '/'}: missing from base, present in head")

    for pointer in sorted(set(base_leaves) & set(head_leaves)):
        if is_volatile(pointer):
            continue
        base_value = base_leaves[pointer]
        head_value = head_leaves[pointer]
        if values_agree(base_value, head_value, tolerance):
            continue
        differences.append(f"{pointer or '/'}: base {base_value!r} vs head {head_value!r}")

    return differences


def read_json(path: Path):
    """Parse one JSON file, reporting the path when it cannot be read."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except OSError as error:
        raise ComparisonError(f"cannot read {path}: {error}") from error
    except json.JSONDecodeError as error:
        raise ComparisonError(f"{path} is not valid JSON: {error}") from error


def relative_paths_under(root: Path) -> set[Path]:
    """Every file under ``root``, as paths relative to it."""
    return {path.relative_to(root) for path in root.rglob("*") if path.is_file()}


def compare_directories(base_root: Path, head_root: Path, tolerance: float) -> list[str]:
    """Compare two run trees: JSON files leaf by leaf, everything else byte for byte."""
    base_files = relative_paths_under(base_root)
    head_files = relative_paths_under(head_root)
    differences: list[str] = []

    for relative in sorted(base_files - head_files):
        differences.append(f"{relative}: produced by base, not by head")
    for relative in sorted(head_files - base_files):
        differences.append(f"{relative}: produced by head, not by base")

    for relative in sorted(base_files & head_files):
        base_path = base_root / relative
        head_path = head_root / relative
        if relative.suffix == ".json":
            for message in compare_documents(read_json(base_path), read_json(head_path), tolerance):
                differences.append(f"{relative}{message}")
            continue
        if base_path.read_bytes() != head_path.read_bytes():
            differences.append(f"{relative}: bytes differ")

    return differences


def rows_in(root: Path) -> set[str]:
    """The matrix rows a run tree holds: one top-level directory per row."""
    return {path.name for path in root.iterdir() if path.is_dir()}


def row_of(difference: str) -> str:
    """The row a directory-mode difference belongs to: the first component of its path."""
    return difference.split("/", 1)[0]


def compare(base: Path, head: Path, tolerance: float) -> list[str]:
    """Compare two results files, or two directories of them."""
    if base.is_dir() != head.is_dir():
        raise ComparisonError(f"{base} and {head} are not the same kind of path")
    if base.is_dir():
        return compare_directories(base, head, tolerance)
    if not base.exists():
        raise ComparisonError(f"{base} does not exist")
    if not head.exists():
        raise ComparisonError(f"{head} does not exist")
    return compare_documents(read_json(base), read_json(head), tolerance)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("base", type=Path, help="results file or run directory from the merge base")
    parser.add_argument("head", type=Path, help="results file or run directory from the head")
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.0,
        help=(
            "relative tolerance for numeric leaves; 0 (the default) compares exactly. "
            "A non-zero value is the numerics-change escape and requires --reason."
        ),
    )
    parser.add_argument(
        "--reason",
        default="",
        help="why this comparison is allowed to be inexact; mandatory with --tolerance",
    )
    parser.add_argument(
        "--moving-rows",
        default="",
        help=(
            "comma-separated matrix rows the change is meant to move (* and ? match); "
            "reported rather than failed. Part of the numerics-change escape: needs --reason "
            "and two run directories."
        ),
    )
    parser.add_argument(
        "--max-report",
        type=int,
        default=DEFAULT_MAXIMUM_REPORTED,
        help=f"how many differences to print in full (default {DEFAULT_MAXIMUM_REPORTED})",
    )
    return parser


def report_moving_rows(moving_rows: list[str], differences: list[str]) -> None:
    """Print each declared row with its count of differences and the first of them."""
    print(f"Rows declared to move ({len(moving_rows)}):")
    for row in moving_rows:
        in_row = [difference for difference in differences if row_of(difference) == row]
        if in_row:
            print(f"  {row}: {len(in_row)} difference(s), declared to move; first: {in_row[0]}")
        else:
            print(f"  {row}: declared to move, but identical")


def main(argv: list[str] | None = None) -> int:
    arguments = build_argument_parser().parse_args(argv)

    if arguments.tolerance < 0.0:
        print("--tolerance cannot be negative.", file=sys.stderr)
        return 2
    if arguments.tolerance > 0.0 and not arguments.reason.strip():
        print(
            "--tolerance is the numerics-change escape and needs --reason: state what "
            "moves the numbers, so the run log carries it.",
            file=sys.stderr,
        )
        return 2

    moving_patterns = [
        pattern.strip() for pattern in arguments.moving_rows.split(",") if pattern.strip()
    ]
    if moving_patterns and not arguments.reason.strip():
        print(
            "--moving-rows is part of the numerics-change escape and needs --reason: state "
            "what moves the numbers, so the run log carries it.",
            file=sys.stderr,
        )
        return 2
    if moving_patterns and not (arguments.base.is_dir() and arguments.head.is_dir()):
        print("--moving-rows names matrix rows, so it needs two run directories.", file=sys.stderr)
        return 2

    try:
        differences = compare(arguments.base, arguments.head, arguments.tolerance)
    except ComparisonError as error:
        print(f"Parity comparison could not run: {error}", file=sys.stderr)
        return 2

    moving_rows: list[str] = []
    if moving_patterns:
        rows = sorted(rows_in(arguments.base) | rows_in(arguments.head))
        unmatched = [
            pattern
            for pattern in moving_patterns
            if not any(fnmatch.fnmatchcase(row, pattern) for row in rows)
        ]
        if unmatched:
            print(
                f"--moving-rows pattern(s) {', '.join(unmatched)} match no row in the run; "
                "a declaration may only name rows the matrix has.",
                file=sys.stderr,
            )
            return 2
        moving_rows = [
            row
            for row in rows
            if any(fnmatch.fnmatchcase(row, pattern) for pattern in moving_patterns)
        ]
        report_moving_rows(moving_rows, differences)
        differences = [
            difference for difference in differences if row_of(difference) not in moving_rows
        ]

    mode = (
        "exactly"
        if arguments.tolerance == 0.0
        else f"to {arguments.tolerance:g} relative ({arguments.reason.strip()})"
    )
    if moving_rows:
        mode += f" outside the {len(moving_rows)} row(s) declared to move"
    if not differences:
        print(f"Parity holds {mode}: {arguments.base} == {arguments.head}")
        return 0

    print(f"Parity broken: {len(differences)} difference(s) comparing {mode}.", file=sys.stderr)
    for message in differences[: max(arguments.max_report, 0)]:
        print(f"  {message}", file=sys.stderr)
    if len(differences) > arguments.max_report:
        print(f"  ... and {len(differences) - arguments.max_report} more.", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
