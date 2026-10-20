#!/usr/bin/env python3
"""Check that every scripts/ path the repository's wiring and documentation names exists.

Most of what runs a script names it by path: the pre-commit hooks, the workflows,
CMake's schema generator, the export's ignore list, the procedures in AGENTS.md
and the docs. Moving or renaming a script breaks each of those silently, and some
of them only run on `main` -- the mirror export and import, the statistics
refresh, the container build -- so a pull request that breaks one passes its own
checks and fails after merge. This check reads every such file and fails on a
path that does not exist, at commit time.

It checks what a path is written as, not what a script computes at run time: a
script that finds the repository from its own location must be moved with that
computation adjusted, which nothing here can see.

Only tokens that look like a file (an extension) or a directory (a trailing
slash) are checked, so prose that happens to contain "scripts/" is not a path.
Private records are not read: `plan/` and `vault/` describe past trees and
scripts that do not exist yet, and STATISTICS.md is regenerated on `main` one
push after a change.
"""

import re
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]

#: The files whose script paths are live: wiring, documentation, tests, scripts.
SCANNED = (
    ".pre-commit-config.yaml",
    ".releaseignore",
    "CMakeLists.txt",
    "pyproject.toml",
    "*.md",
    ".github/workflows/*.yml",
    ".github/actions/**/*",
    ".claude/skills/**/*.md",
    "cmake/**/*",
    "docs/**/*.md",
    "examples/**/*.md",
    "schemas/README.md",
    "scripts/**/*",
    "tests/**/*.py",
    "tests/**/*.md",
)
#: STATISTICS.md is regenerated on main one push after a change; this checker's
#: own tests name paths that deliberately do not exist.
NOT_SCANNED = ("STATISTICS.md", "test_check_script_paths.py")

#: A path starts at a boundary: not inside a longer name ("myscripts/"), but
#: after a string's "\n" escape it does -- the "n" is a word character, and
#: treating it as part of a name hid every path written as a line of a literal.
PATH_TOKEN = re.compile(r"(?:(?<=\\n)|(?<![\w.-]))(?:\.github/)?scripts/[\w][\w./-]*")
FILE_OR_DIRECTORY = re.compile(r"(\.[A-Za-z0-9]+|/)$")


def scanned_files(root: Path) -> list[Path]:
    """The files matched by SCANNED under *root*, each once, in a stable order."""
    found: set[Path] = set()
    for pattern in SCANNED:
        found.update(path for path in root.glob(pattern) if path.is_file())
    return sorted(
        path for path in found if path.name not in NOT_SCANNED and "__pycache__" not in path.parts
    )


def missing_paths(root: Path) -> list[str]:
    """``file:line: path`` for every named script path under *root* that does not exist."""
    problems = []
    for path in scanned_files(root):
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except UnicodeDecodeError:
            continue
        for number, line in enumerate(lines, start=1):
            for match in PATH_TOKEN.finditer(line):
                token = match.group(0).rstrip(".,:;")
                if "*" in token or not FILE_OR_DIRECTORY.search(token):
                    continue
                if not (root / token).exists():
                    problems.append(f"{path.relative_to(root)}:{number}: {token}")
    return problems


def main() -> int:
    problems = missing_paths(REPOSITORY_ROOT)
    for problem in problems:
        print(f"{problem} does not exist", file=sys.stderr)
    if problems:
        print(
            "\nA script was moved or renamed without updating what names it. "
            "scripts/README.md lists where each kind of script lives.",
            file=sys.stderr,
        )
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
