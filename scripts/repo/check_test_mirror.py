#!/usr/bin/env python3
"""Enforce the ``tests/unit/`` mirror and CMake test registration (AGENTS.md § Testing).

AGENTS.md states the rule in prose: "The directory structure under
``tests/unit/`` MUST mirror the directory structure under ``src/``. A test
file for ``src/layer_X/foo/bar.hpp`` must be placed at
``tests/unit/layer_X/foo/test_bar.cpp``." This script is the automated
checker for that rule, in the same shape as ``check_cpp_include_layering.py``
and ``check_python_import_layering.py``.

It checks two related but distinct things, both defect classes found by hand
while auditing the tree (see the codebase-improvement-program plan, item A5):

1. **Placement.** For every ``tests/unit/**/test_X.cpp``, if it has a local
   ``#include`` whose basename matches ``X`` (i.e. this test's own name says
   which header it tests), that header's directory relative to ``src/`` must
   equal the test file's own directory relative to ``tests/unit/``. This is
   deliberately grounded in the file's *actual* declared dependency rather
   than a name-guessing heuristic over the whole tree: it is exactly how the
   three real violations fixed alongside this checker were found, and it does
   not fire on a test file that legitimately doesn't follow the
   one-name-matches-one-header convention (e.g. a test covering several
   related classes in one directory) -- those are silently skipped rather
   than guessed at, so the check only ever fires with the same confidence a
   human reviewer would have.
2. **Registration.** Every ``.cpp`` under ``tests/unit/`` or
   ``tests/integration/`` must be listed in one of the explicit test-source
   lists in ``CMakeLists.txt`` (``q3_tests``, ``q3_tests_mpi``) -- a file
   that exists but isn't registered silently never runs, which is a stronger
   version of the "green PR ran zero tests" hole the ctest-on-PRs fix closed
   for the suite as a whole. The reverse direction is checked too: a listed
   path that doesn't exist on disk means a rename or deletion left a stale
   entry behind.

**Build-time add-ons** (``src/addons/<name>/``, cmake/Q3Addons.cmake) follow
the same two rules with two allowances. An add-on lists its tests in the
``UNIT_TESTS`` of the ``q3_register_addon(...)`` call in its own
``src/addons/<name>/CMakeLists.txt``, which counts as registration. And an
add-on test that exercises a *core* header with the add-on's data mirrors
that header's core path under the add-on's prefix: a test of
``src/<path>/bar.hpp`` may sit at ``tests/unit/addons/<name>/<path>/test_bar.cpp``.
A tree without ``src/addons/`` (the public export) has no add-on tests
either, and both checks pass on it unchanged.

Neither check requires *every* ``src/`` header to have a test -- that is a
coverage-completeness question (see TODOs.md's mirror-fraction ratchet),
answered separately and more permissively than the hard placement/
registration rules here.

Known, pre-existing exceptions are allowlisted explicitly in
``KNOWN_MIRROR_EXCEPTIONS``, the same shape as the other checkers' allowlists:
any violation not covered by an entry fails the check, so the allowlist can
only ever hide a specific, already-measured exception, never a new one.

Usage::

    python scripts/repo/check_test_mirror.py [PATH ...]

With no arguments the whole ``tests/`` tree and ``CMakeLists.txt`` are
scanned (this script's checks are both tree-wide by nature -- a partial file
list cannot express "is this file registered anywhere?" -- so, like its two
counterparts, it always rescans everything regardless of arguments; PATH
exists only to narrow which placement violations are *printed*, for local
iteration).
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
TESTS_UNIT_ROOT = REPOSITORY_ROOT / "tests" / "unit"
TESTS_INTEGRATION_ROOT = REPOSITORY_ROOT / "tests" / "integration"
CMAKE_LISTS_PATH = REPOSITORY_ROOT / "CMakeLists.txt"

# Same include-matching rule as check_cpp_include_layering.py: a real
# preprocessor directive's '#' is the first non-blank character on its line,
# which is what keeps this from misreading an #include quoted inside a
# /** */ usage docstring as a real one.
INCLUDE_RE = re.compile(r'^\s*#\s*include\s*"([^"]+)"')

# A registered CMake test source: a bare, unquoted, repo-relative path under
# tests/, one per line inside an add_executable(...) block. Deliberately
# does not match CMake variables like ${Q3_GPU_KERNEL_SOURCES} (no '$') or
# tests/test_main.cpp (excluded on purpose -- see collect_test_source_files).
CMAKE_TEST_SOURCE_RE = re.compile(r"^\s*(tests/(?:unit|integration)/\S+\.cpp)\s*$", re.MULTILINE)

# CMake targets whose source list this checker treats as "registered". Each
# is an add_executable(<name> ...) block; every bare tests/... path between
# the matching open and close parens counts.
CMAKE_TEST_TARGETS = ("q3_tests", "q3_tests_mpi")

# The CMake call an add-on lists its tests in (cmake/Q3Addons.cmake).
ADDON_REGISTRATION_CALL = "q3_register_addon"

# The tests/unit/ (and src/) directory that holds build-time add-ons.
ADDONS_DIRECTORY = "addons"

# Known, pre-existing exceptions to either check, keyed by
# (repo-relative path, "placement" | "registration") -> a reason. A
# placement entry means: this test's name matches a header by convention,
# but is deliberately not colocated with it (e.g. it covers several headers
# in the directory and is filed under the directory itself). A registration
# entry means: this file exists under tests/unit or tests/integration on
# purpose without being built by CMake (e.g. a fixture/helper module, not a
# GTest translation unit itself).
KNOWN_MIRROR_EXCEPTIONS: dict[tuple[str, str], str] = {}


class Violation:
    """One test-mirror violation, ready to be printed as a compiler-style diagnostic."""

    def __init__(self, path: Path, message: str) -> None:
        self.path = path
        self.message = message

    def __str__(self) -> str:
        return f"{self.path}: {self.message}"


def _is_known_exception(relative_path: Path, kind: str) -> bool:
    return (relative_path.as_posix(), kind) in KNOWN_MIRROR_EXCEPTIONS


def local_includes(path: Path) -> list[str]:
    """Return every quoted #include target in *path*, in file order."""
    return [
        match.group(1)
        for line in path.read_text().splitlines()
        if (match := INCLUDE_RE.match(line))
    ]


def check_placement(test_path: Path) -> Violation | None:
    """Return a placement violation for *test_path*, or None if it mirrors correctly.

    *test_path* must be a ``tests/unit/**/test_*.cpp`` file. Looks for every
    local include whose basename (without ``.hpp``) equals *test_path*'s own
    name (without the ``test_`` prefix and ``.cpp`` suffix) -- there can be
    more than one, since ``init.hpp`` alone names ~79 files (the blessed
    package-entry convention), so a test can easily include one *other*
    directory's ``init.hpp`` as a dependency alongside its own. The test
    passes if *any* matching include's directory (relative to ``src/``)
    equals the test's own directory (relative to ``tests/unit/``); it is only
    a violation if there is at least one matching include and *none* of them
    do -- reported against the first one found, deterministically. A test
    whose name matches no include at all (e.g. named after the class it
    produces rather than the header it tests) is silently not checked -- see
    the module docstring for why that trade-off was made deliberately.
    """
    relative_test_path = test_path.resolve().relative_to(TESTS_UNIT_ROOT.resolve())
    if _is_known_exception(Path("tests/unit") / relative_test_path, "placement"):
        return None

    tested_name = test_path.stem.removeprefix("test_")
    own_directory = relative_test_path.parent.as_posix()
    core_directory_under_addon = _core_directory_under_addon(own_directory)

    candidates = [
        include
        for include in local_includes(test_path)
        if Path(include).suffix == ".hpp" and Path(include).stem == tested_name
    ]
    if not candidates:
        return None  # no include matched this test's name; not this checker's business
    for include in candidates:
        include_directory = Path(include).parent.as_posix()
        if include_directory == own_directory:
            return None  # at least one matching include confirms correct placement
        if (
            core_directory_under_addon is not None
            and not include.startswith(ADDONS_DIRECTORY + "/")
            and include_directory == core_directory_under_addon
        ):
            return None  # an add-on's test of a core header, mirrored under the add-on prefix

    include = candidates[0]
    expected_directory = Path(include).parent.as_posix()
    expected_path = (Path("tests/unit") / expected_directory / test_path.name).as_posix()
    return Violation(
        Path("tests/unit") / relative_test_path,
        f'includes "{include}", so it tests a header at src/{expected_directory}/ '
        f"but is not placed at {expected_path}. The directory structure under "
        "tests/unit/ must mirror src/ (AGENTS.md § Testing) -- move the file, or "
        "if this placement is intentional, add a KNOWN_MIRROR_EXCEPTIONS entry "
        "(scripts/repo/check_test_mirror.py) explaining why.",
    )


def _core_directory_under_addon(test_directory: str) -> str | None:
    """Return the core path an add-on test directory mirrors, or None outside add-ons.

    ``addons/<name>/<path>`` mirrors the core ``<path>``; the add-on's root
    directory itself mirrors the core root, ``""``.
    """
    parts = test_directory.split("/")
    if len(parts) < 2 or parts[0] != ADDONS_DIRECTORY:
        return None
    return "/".join(parts[2:])


def collect_unit_test_files(arguments: list[str]) -> list[Path]:
    """Expand command-line arguments into the tests/unit/**/test_*.cpp files to check."""
    if not arguments:
        return sorted(TESTS_UNIT_ROOT.rglob("test_*.cpp"))
    paths: list[Path] = []
    for argument in arguments:
        path = Path(argument)
        if path.is_dir():
            paths.extend(sorted(path.rglob("test_*.cpp")))
        elif path.name.startswith("test_") and path.suffix == ".cpp":
            paths.append(path)
    return paths


def registered_cmake_test_sources() -> set[str]:
    """Return every bare tests/... path listed inside a known CMake test target.

    Scans each add_executable(<target> ...) block by paren-depth, starting
    from the line containing the target name (mirrors how a human reads the
    file: "everything between this add_executable( and its matching )").
    """
    text = CMAKE_LISTS_PATH.read_text()
    registered: set[str] = set()
    for target in CMAKE_TEST_TARGETS:
        match = re.search(rf"add_executable\(\s*{re.escape(target)}\b", text)
        if not match:
            continue
        depth = 0
        block_start = match.start()
        cursor = block_start
        for cursor in range(block_start, len(text)):
            if text[cursor] == "(":
                depth += 1
            elif text[cursor] == ")":
                depth -= 1
                if depth == 0:
                    break
        block = text[block_start:cursor]
        registered.update(m.group(1) for m in CMAKE_TEST_SOURCE_RE.finditer(block))
    for addon_cmake_lists in sorted(SOURCE_ROOT.glob(f"{ADDONS_DIRECTORY}/*/CMakeLists.txt")):
        registered.update(_addon_registered_test_sources(addon_cmake_lists.read_text()))
    return registered


def _addon_registered_test_sources(text: str) -> set[str]:
    """Return every bare tests/... path inside an add-on's q3_register_addon(...) call."""
    registered: set[str] = set()
    for match in re.finditer(rf"{ADDON_REGISTRATION_CALL}\(", text):
        depth = 0
        cursor = match.start()
        for cursor in range(match.start(), len(text)):
            if text[cursor] == "(":
                depth += 1
            elif text[cursor] == ")":
                depth -= 1
                if depth == 0:
                    break
        block = text[match.start() : cursor]
        registered.update(m.group(1) for m in CMAKE_TEST_SOURCE_RE.finditer(block))
    return registered


def on_disk_test_source_files() -> set[str]:
    """Return every real .cpp under tests/unit/ or tests/integration/, repo-relative.

    tests/test_main.cpp (the shared GTest main()) is deliberately excluded:
    it is not a mirror of any src/ header and is registered by name, not
    convention -- it isn't the subject of either check this script runs.
    """
    files: set[str] = set()
    for root in (TESTS_UNIT_ROOT, TESTS_INTEGRATION_ROOT):
        for path in root.rglob("*.cpp"):
            files.add(path.resolve().relative_to(REPOSITORY_ROOT.resolve()).as_posix())
    return files


def check_registration() -> list[Violation]:
    """Return every registration mismatch between disk and CMakeLists.txt."""
    on_disk = on_disk_test_source_files()
    registered = registered_cmake_test_sources()
    violations: list[Violation] = []

    for missing in sorted(on_disk - registered):
        if _is_known_exception(Path(missing), "registration"):
            continue
        violations.append(
            Violation(
                Path(missing),
                "exists under tests/unit/ or tests/integration/ but is not listed in "
                "any add_executable(q3_tests[...]) source list in CMakeLists.txt, nor "
                "in an add-on's q3_register_addon(... UNIT_TESTS ...) -- it will never "
                "be built or run. Add it to the list, or if it is intentionally not a "
                "GTest translation unit, add a KNOWN_MIRROR_EXCEPTIONS entry "
                "(scripts/repo/check_test_mirror.py).",
            )
        )
    for stale in sorted(registered - on_disk):
        violations.append(
            Violation(
                Path(stale),
                "is listed in CMakeLists.txt's test source list but does not exist on "
                "disk -- likely a stale entry left behind by a rename or deletion.",
            )
        )
    return violations


def main() -> int:
    """Check the test mirror and CMake registration; return the process exit code."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "paths", nargs="*", help="tests/unit files or directories to narrow placement output to"
    )
    arguments = parser.parse_args()

    # Both checks always scan the whole tree, regardless of arguments -- see
    # the module docstring's Usage section for why (matches
    # check_cpp_include_layering.py and check_python_import_layering.py,
    # which pass pass_filenames: false for the same reason).
    violations: list[Violation] = []
    for test_path in collect_unit_test_files(arguments.paths):
        violation = check_placement(test_path)
        if violation is not None:
            violations.append(violation)
    violations.extend(check_registration())

    for violation in sorted(violations, key=lambda v: str(v.path)):
        print(violation, file=sys.stderr)

    if violations:
        print(
            f"\n{len(violations)} test-mirror violation(s). Allowlist and rules are "
            "defined in scripts/repo/check_test_mirror.py.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
