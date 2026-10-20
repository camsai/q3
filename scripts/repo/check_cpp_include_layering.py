#!/usr/bin/env python3
"""Enforce the C++ ``src/layer_N_*`` include layering (AGENTS.md § Layer Separation).

AGENTS.md states the rule in prose: "Includes must never point upward:
``layer_N`` code must not ``#include`` a ``layer_(N+k)`` header". The Python
package got an automated checker for its own analogous rule in
``scripts/repo/check_python_import_layering.py`` (see AGENTS.md § Python Package
Layering); this script is the C++ counterpart, so it mirrors that script's
shape closely.

The rule itself is simpler than the Python one: a C++ layer's number is
encoded directly in its directory name (``layer_4_computation`` is layer 4),
so there is no separate tier-assignment table to maintain — a file's own
layer and an include's target layer are both read straight off the path.
There is also no same-layer "cycle" concept to detect here (unlike the
Python tiers, which name whole subpackages that could import each other):
sibling headers within one layer including each other is normal and is not
checked.

Exempt paths (they sit above or outside the layer stack and are never
checked for upward includes, each also documented in AGENTS.md):

  * ``src/io/`` — a vertical slice (AGENTS.md): explicitly allowed to
    reference any layer.
  * ``src/bindings/`` — the ``q3_core`` nanobind Python module boundary; by
    definition it must expose every layer to Python.
  * ``src/main*.cpp`` (serial, MPI, and GPU-flavored entry points) — they
    assemble every layer into a runnable binary, so they sit above the
    stack, not inside it.
  * ``src/utils/`` — general utilities, outside the layer stack.
  * ``src/generated/`` — auto-generated ESSE schema wrappers, outside the
    layer stack.
  * ``src/py/`` — the ``q3`` Python package; already governed by
    ``check_python_import_layering.py``, not this script.
  * ``src/addons/`` — build-time add-ons (cmake/Q3Addons.cmake). Each is a
    vertical slice like ``io/``: it plugs into the core's registries and may
    include any layer.

One rule runs the other way, for every file under ``src/`` including the
exempt ones: **nothing outside ``src/addons/`` may include an ``addons/...``
header.** The core names no add-on and must build with ``src/addons/``
absent, so an add-on is reached only through the registries it fills, never
by an include. The generated registration header that does include add-on
headers lives in the build directory, not in ``src/``.

Known, pre-existing upward includes that are not fixed by this checker are
allowlisted explicitly in ``KNOWN_UPWARD_INCLUDES``, keyed by
``(path-or-directory relative to src/, included layer)`` with a comment
pointing at the TODOs.md item that tracks the eventual fix. Any upward
include *not* covered by an allowlist entry fails the check — so the
allowlist can only ever hide a specific, already-measured include, never a
newly introduced one.

Usage::

    python scripts/repo/check_cpp_include_layering.py [PATH ...]

With no arguments the whole ``src`` tree is scanned. Exits non-zero and
prints one line per violation when the layering is broken.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = REPOSITORY_ROOT / "src"

# File extensions this checker treats as C++ source: headers, translation
# units, CUDA (.cu), and the Objective-C++ Metal GPU backend (.mm).
SOURCE_EXTENSIONS = {".hpp", ".cpp", ".h", ".cc", ".cu", ".mm"}

# Matches a layer directory's name, e.g. "layer_4_computation" -> group(1) = "4".
LAYER_DIRECTORY_RE = re.compile(r"^layer_(\d+)_")

# A real preprocessor directive's '#' is the first non-blank character on its
# line. Anchoring here (rather than searching anywhere in the line) is what
# keeps this from misreading the #include examples quoted inside `/** */`
# usage docstrings (e.g. src/layer_0_core/types/init.hpp, src/io/logger.hpp)
# as real includes: those lines start with " * ", not "#".
INCLUDE_RE = re.compile(r'^\s*#\s*include\s*"([^"]+)"')

# Any include, quoted or angle-bracketed, whose target is an add-on header.
ADDON_INCLUDE_RE = re.compile(r'^\s*#\s*include\s*[<"](addons/[^">]+)[">]')

# The top-level src/ directory that holds build-time add-ons.
ADDONS_TOP_LEVEL_DIRECTORY = "addons"

# The six layers AGENTS.md defines today. Not consulted by the numeric M > N
# check itself (the layer number is read directly off the directory name),
# only by the guard test that every top-level src/ entry is accounted for.
LAYER_DIRECTORY_NAMES = {
    "layer_0_core",
    "layer_1_structure",
    "layer_2_quantum_system",
    "layer_3_representation",
    "layer_4_computation",
    "layer_5_properties",
}

# Top-level src/ directories that sit above or outside the layer stack and
# are never checked for upward includes — see the module docstring for why
# each one is exempt.
EXEMPT_TOP_LEVEL_DIRECTORIES = {
    "io",
    "bindings",
    "utils",
    "generated",
    "py",
    # q3_utility's subcommand classes: entry-point code split out of main_utility.cpp
    # one class per file, so it composes any layer the way main*.cpp does.
    "commands",
    # Build-time add-ons: vertical slices like io/, reached only through the core's
    # registries. The reverse rule (nothing else may include them) still applies.
    ADDONS_TOP_LEVEL_DIRECTORY,
}

# Top-level src/ files matching this pattern are entry points (serial, MPI,
# and GPU-flavored mains) that assemble every layer into a binary.
EXEMPT_ENTRY_POINT_RE = re.compile(r"^main.*\.cpp$")

# Known, pre-existing upward includes that this checker does not fix. Each
# entry is (path-or-directory-prefix relative to src/, included layer) -> a
# comment pointing at the TODOs.md item that tracks the eventual fix. A path
# key matches exactly one file; a directory key (no file extension) matches
# every file below it. Any upward include *not* covered by one of these
# entries fails the check.
KNOWN_UPWARD_INCLUDES: dict[tuple[str, int], str] = {
    # flavors/ orchestrates post-SCF properties (forces, stress tensor,
    # projections, isolated-system correction), which today live one layer
    # up in layer_5_properties/. TODOs.md § 3.2 "Layer boundary cleanup"
    # tracks the decision to either move flavors/ or bless it as a vertical
    # slice like io/; until then this is a documented, contained exception,
    # not a silent violation. Measured 2026-07-29: exactly these two files,
    # four includes, all targeting layer_5 -- nothing else in the tree.
    ("layer_4_computation/flavors", 5): (
        "TODOs.md § 3.2 Layer boundary cleanup -- flavors/ orchestrates "
        "post-SCF properties (forces, stress, projections, isolated-system "
        "correction) that currently live in layer_5_properties/"
    ),
}


class Violation:
    """One layering violation, ready to be printed as a compiler-style diagnostic."""

    def __init__(self, path: Path, line: int, message: str) -> None:
        self.path = path
        self.line = line
        self.message = message

    def __str__(self) -> str:
        return f"{self.path}:{self.line}: {self.message}"


def layer_of(relative_path: Path) -> int | None:
    """Return the layer number for *relative_path* (relative to src/), or None.

    None means *relative_path* is not inside a ``layer_N_*`` directory at all
    (its top-level component is something else, e.g. ``io`` or ``main.cpp``).
    """
    match = LAYER_DIRECTORY_RE.match(relative_path.parts[0])
    return int(match.group(1)) if match else None


def is_exempt(relative_path: Path) -> bool:
    """Return whether *relative_path* sits above/outside the layer stack.

    Covers ``io/``, ``bindings/``, ``utils/``, ``generated/``, ``py/``, and
    the top-level ``main*.cpp`` entry points -- see the module docstring.
    """
    top = relative_path.parts[0]
    if top in EXEMPT_TOP_LEVEL_DIRECTORIES:
        return True
    return len(relative_path.parts) == 1 and bool(EXEMPT_ENTRY_POINT_RE.match(top))


def is_known_upward_include(relative_path: Path, included_layer: int) -> bool:
    """Return whether (*relative_path*, *included_layer*) is on the allowlist."""
    posix_path = relative_path.as_posix()
    for allowed_path, allowed_layer in KNOWN_UPWARD_INCLUDES:
        if allowed_layer != included_layer:
            continue
        if posix_path == allowed_path or posix_path.startswith(allowed_path + "/"):
            return True
    return False


def included_layer_headers(path: Path) -> list[tuple[str, int]]:
    """Return every (#include target, line number) in *path* naming a layer_N header."""
    found: list[tuple[str, int]] = []
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        match = INCLUDE_RE.match(line)
        if match and LAYER_DIRECTORY_RE.match(match.group(1)):
            found.append((match.group(1), line_number))
    return found


def included_addon_headers(path: Path) -> list[tuple[str, int]]:
    """Return every (#include target, line number) in *path* naming an add-on header."""
    found: list[tuple[str, int]] = []
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        match = ADDON_INCLUDE_RE.match(line)
        if match:
            found.append((match.group(1), line_number))
    return found


def check_addon_includes(path: Path, relative_path: Path) -> list[Violation]:
    """Return a violation for every add-on header a file outside src/addons/ includes."""
    if relative_path.parts[0] == ADDONS_TOP_LEVEL_DIRECTORY:
        return []
    return [
        Violation(
            path,
            line_number,
            f'core file {relative_path.as_posix()} includes the add-on header "{included}". '
            "The core names no add-on and must build without src/addons/: reach the add-on "
            "through the registry it fills (see cmake/Q3Addons.cmake), not by an include.",
        )
        for included, line_number in included_addon_headers(path)
    ]


def check_file(path: Path) -> list[Violation]:
    """Return the layering violations in a single source file."""
    try:
        relative_path = path.resolve().relative_to(SOURCE_ROOT.resolve())
    except ValueError:
        return []  # outside src/ entirely -- not this checker's business
    addon_violations = check_addon_includes(path, relative_path)
    if is_exempt(relative_path):
        return addon_violations
    own_layer = layer_of(relative_path)
    if own_layer is None:
        # Unrecognized top-level entry; see test_every_top_level_source_entry_is_classified.
        return addon_violations

    violations: list[Violation] = list(addon_violations)
    for included, line_number in included_layer_headers(path):
        included_layer = int(LAYER_DIRECTORY_RE.match(included).group(1))
        if included_layer <= own_layer:
            continue
        if is_known_upward_include(relative_path, included_layer):
            continue
        violations.append(
            Violation(
                path,
                line_number,
                f"upward include: {relative_path.as_posix()} (layer {own_layer}) includes "
                f'"{included}" (layer {included_layer}). Includes must never point upward '
                "-- define an interface at the lower layer and implement it above "
                "(dependency inversion), or, for a pre-existing tracked exception, add a "
                "KNOWN_UPWARD_INCLUDES entry pointing at a TODOs.md item "
                "(see AGENTS.md § Layer Separation).",
            )
        )
    return violations


def collect_source_files(arguments: list[str]) -> list[Path]:
    """Expand command-line arguments into the list of files to check."""
    if not arguments:
        return sorted(
            p for p in SOURCE_ROOT.rglob("*") if p.is_file() and p.suffix in SOURCE_EXTENSIONS
        )
    paths: list[Path] = []
    for argument in arguments:
        path = Path(argument)
        if path.is_dir():
            paths.extend(
                sorted(p for p in path.rglob("*") if p.is_file() and p.suffix in SOURCE_EXTENSIONS)
            )
        elif path.suffix in SOURCE_EXTENSIONS:
            paths.append(path)
    return paths


def main() -> int:
    """Check the layering and report violations; return the process exit code."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("paths", nargs="*", help="files or directories to check")
    arguments = parser.parse_args()

    # Whole-tree scanning happens regardless of arguments: the hook invokes
    # this with pass_filenames: false (see .pre-commit-config.yaml), so a
    # violation introduced by only some of a change's files being staged
    # cannot slip through, exactly as check_python_import_layering.py does
    # for the Python side.
    files_to_check = collect_source_files(arguments.paths)

    violations: list[Violation] = []
    for path in files_to_check:
        violations.extend(check_file(path))

    for violation in sorted(violations, key=lambda v: (str(v.path), v.line)):
        print(violation, file=sys.stderr)

    if violations:
        print(
            f"\n{len(violations)} include-layering violation(s). Allowlist and rule are "
            "defined in scripts/repo/check_cpp_include_layering.py.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
