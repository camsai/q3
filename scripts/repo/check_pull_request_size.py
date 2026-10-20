#!/usr/bin/env python3
"""Report a pull request's reviewable size against a soft ceiling.

A 17,700-line pull request across 115 files got eighteen reviews and merged
anyway, because there is no size at which a reviewer can say "this is too
large" and point at a rule. This is that rule, and it is deliberately a soft
one: it annotates, it does not fail. A ceiling that blocks gets argued with or
routed around; a ceiling that shows up as a number in every run gets designed
for.

What counts, and what does not
------------------------------

Only lines a human has to read count. ``EXCLUDED_PATTERNS`` drops the files
that change in bulk without being reviewed line by line:

* ``src/generated/`` -- produced by ``generate_cpp_from_schemas.py`` from
  ``schemas/``, which *is* counted. Reviewing the schema is reviewing the
  header.
* ``STATISTICS.md`` -- regenerated on ``main`` after every push.
* Vendored and lock files, which are reviewed as a version bump.
* Data under ``examples/data/``, which is measured in bytes rather than lines.

Deletions count the same as additions. Deleting a thousand lines is not free
to review -- the question "was any of that load-bearing?" is asked of every
one of them -- and counting only additions would reward splitting a rename
into a delete and an add.

Usage::

    python scripts/repo/check_pull_request_size.py --base origin/main
    python scripts/repo/check_pull_request_size.py --base origin/main --ceiling 3000

Exit status is 0 whatever the size, unless ``--fail-over-ceiling`` is passed
or the diff cannot be read at all.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys

DEFAULT_CEILING = 3000

# Paths whose lines are not read one by one. Each entry is a prefix or a
# filename, matched against the repository-relative path.
EXCLUDED_PATTERNS = (
    re.compile(r"^src/generated/"),
    re.compile(r"^STATISTICS\.md$"),
    re.compile(r"^schemas/esse/"),
    re.compile(r"^examples/data/"),
    re.compile(r"(^|/)(package-lock\.json|poetry\.lock|uv\.lock|Cargo\.lock)$"),
)


def is_excluded(path: str) -> bool:
    """Answer whether this path's lines are exempt from the count."""
    return any(pattern.search(path) for pattern in EXCLUDED_PATTERNS)


def numstat(base: str, head: str) -> list[tuple[int, int, str]]:
    """Read ``git diff --numstat`` for a range, as (added, deleted, path) rows.

    A binary file reports ``-`` for both counts; it contributes no lines and
    is returned as zeros so that it still appears in the file count.
    """
    completed = subprocess.run(
        ["git", "diff", "--numstat", f"{base}...{head}"],
        capture_output=True,
        text=True,
        check=True,
    )
    rows = []
    for line in completed.stdout.splitlines():
        fields = line.split("\t")
        if len(fields) != 3:
            continue
        added_text, deleted_text, path = fields
        added = 0 if added_text == "-" else int(added_text)
        deleted = 0 if deleted_text == "-" else int(deleted_text)
        rows.append((added, deleted, path))
    return rows


def measure(rows) -> dict:
    """Split a numstat into what counts and what is exempt."""
    counted_lines = 0
    counted_files = 0
    excluded_lines = 0
    excluded_files = 0
    largest: list[tuple[int, str]] = []

    for added, deleted, path in rows:
        lines = added + deleted
        if is_excluded(path):
            excluded_lines += lines
            excluded_files += 1
            continue
        counted_lines += lines
        counted_files += 1
        largest.append((lines, path))

    largest.sort(reverse=True)
    return {
        "counted_lines": counted_lines,
        "counted_files": counted_files,
        "excluded_lines": excluded_lines,
        "excluded_files": excluded_files,
        "largest": largest[:10],
    }


def format_report(measurement: dict, ceiling: int) -> str:
    """Render the measurement as the lines a run log should carry."""
    counted = measurement["counted_lines"]
    verdict = (
        f"{counted} reviewable lines across {measurement['counted_files']} file(s); "
        f"the soft ceiling is {ceiling}."
    )
    if counted > ceiling:
        verdict += (
            f"\nThis is {counted - ceiling} over. Splitting by subsystem, so each part "
            f"can be reviewed and reverted on its own, is what the ceiling is asking for. "
            f"If the change genuinely cannot be split, say so in the pull request."
        )
    lines = [verdict]
    if measurement["excluded_files"]:
        lines.append(
            f"Not counted: {measurement['excluded_lines']} line(s) across "
            f"{measurement['excluded_files']} generated or vendored file(s)."
        )
    if measurement["largest"]:
        lines.append("Largest counted files:")
        lines.extend(f"  {size:>6}  {path}" for size, path in measurement["largest"])
    return "\n".join(lines)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--base", default="origin/main", help="the branch this merges into")
    parser.add_argument("--head", default="HEAD", help="the branch being measured")
    parser.add_argument("--ceiling", type=int, default=DEFAULT_CEILING)
    parser.add_argument(
        "--fail-over-ceiling",
        action="store_true",
        help="exit non-zero when over the ceiling; off by default, because the ceiling is soft",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = build_argument_parser().parse_args(argv)
    try:
        rows = numstat(arguments.base, arguments.head)
    except subprocess.CalledProcessError as error:
        print(
            f"Could not diff {arguments.base}...{arguments.head}: {error.stderr.strip() or error}",
            file=sys.stderr,
        )
        return 2

    measurement = measure(rows)
    print(format_report(measurement, arguments.ceiling))
    if arguments.fail_over_ceiling and measurement["counted_lines"] > arguments.ceiling:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
