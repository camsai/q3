#!/usr/bin/env python3
"""Check that a change does not carry STATISTICS.md, or an add-on's own copy.

STATISTICS.md counts lines and test cases across the whole tree, so it moves
with almost every commit that touches source. Carrying it in a pull request
means every pair of open branches conflicts on it -- on adjacent lines of a
nine-row table, which git cannot merge -- for a file no person is allowed to
edit. Of the last forty non-merge commits before this check existed, twenty-two
touched it.

So it is no longer maintained by contributors at all. A workflow in the
development repository regenerates it on `main` after every push and commits the
result; the published mirror receives that file with the next export snapshot.
This check is what keeps that workflow the only writer.

Two modes, the same pair as scripts/repo/check_dco_signoff.py:

- ``--staged``: the files staged for the commit being made. Used by the
  ``statistics-untouched`` pre-commit hook, so the rule lands at commit time
  rather than in CI. Nothing staged means nothing to say, which is also what
  makes ``pre-commit run --all-files`` quiet.
- ``--range BASE...HEAD``: the files a pull request changes. Used by
  .github/workflows/statistics-guard.yml, because a pre-commit hook only binds
  developers who installed it.

Each build-time add-on has its own STATISTICS.md, in src/addons/<name>/, counting
only its own paths, and the same holds for it.

The one legitimate reason for STATISTICS.md to appear in a change is a change
to the generator itself -- a new metric, a renamed row. Then both files move
together, and this asks instead that the committed file match what the new
generator produces.
"""

import argparse
import re
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
STATISTICS_PATH = "STATISTICS.md"
#: Each build-time add-on's own counts, beside its code.
ADDON_STATISTICS_PATH = re.compile(r"^src/addons/[^/]+/STATISTICS\.md$")
GENERATOR_PATH = "scripts/repo/generate_statistics.py"

REFRESHED_AUTOMATICALLY = """\
Generated statistics, refreshed on `main` automatically since 2026-09-11 rather
than by contributors: {paths}. Drop them from this change:

    git restore --staged --worktree {paths}

The numbers on `main` will be correct one push after this merges. If you meant
to change what is measured, stage {generator} along with them."""


def is_generated(path):
    """Whether a path is one of the generated statistics files."""
    return path == STATISTICS_PATH or bool(ADDON_STATISTICS_PATH.match(path))


def changed_files(staged, range_specifier):
    if staged:
        command = ["git", "diff", "--cached", "--name-only"]
    else:
        command = ["git", "diff", "--name-only", range_specifier]
    completed = subprocess.run(
        command, cwd=PROJECT_ROOT, capture_output=True, text=True, check=True
    )
    return {line for line in completed.stdout.splitlines() if line}


def generated_file_is_current():
    """Whether every committed statistics file matches what the generator now writes."""
    completed = subprocess.run(
        [sys.executable, GENERATOR_PATH, "--check"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return completed.returncode == 0


def check(staged, range_specifier):
    """Exit status for one mode: 0 if the change may stand, 1 if it may not."""
    changed = changed_files(staged, range_specifier)
    generated = sorted(path for path in changed if is_generated(path))
    if not generated:
        return 0

    paths = " ".join(generated)
    if GENERATOR_PATH not in changed:
        print(
            REFRESHED_AUTOMATICALLY.format(paths=paths, generator=GENERATOR_PATH), file=sys.stderr
        )
        return 1

    if not generated_file_is_current():
        print(
            f"{GENERATOR_PATH} changed, so {paths} may move with it -- but the\n"
            f"committed files are not what the new generator writes. Run:\n\n"
            f"    python3 {GENERATOR_PATH}\n",
            file=sys.stderr,
        )
        return 1

    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--staged", action="store_true", help="check the staged files")
    mode.add_argument("--range", metavar="BASE...HEAD", help="check the files a range changes")
    arguments = parser.parse_args()
    return check(arguments.staged, arguments.range)


if __name__ == "__main__":
    sys.exit(main())
