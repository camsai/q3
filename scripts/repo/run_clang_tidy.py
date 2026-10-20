#!/usr/bin/env python3
"""Run clang-tidy over every source file the current CMake build actually compiled.

Backs the ``clang-tidy`` pre-commit hook and the CI step described in
TODOs.md § 3.3. It exists because naively globbing every non-``src/bindings/``
``.cpp`` file (what the hook did before) and handing each straight to
``clang-tidy -p build`` breaks as soon as the tree contains files a *particular*
CMake configuration does not compile:

  * The GPU vendor backends (``cuda.cpp``/``hip.cpp``/``sycl.cpp`` under
    ``layer_0_core``, ``layer_3_representation``, ``layer_4_computation``) and
    their entry points (``main_gpu*.cpp``) are gated behind ``Q3_WITH_CUDA``
    / ``Q3_WITH_HIP`` / ``Q3_WITH_SYCL``, all OFF by default.
  * ``main_mpi.cpp`` and ``tests/unit/layer_0_core/execution/test_mpi_execution_policy.cpp``
    are gated behind ``Q3_WITH_MPI``, also OFF by default (MPI gets its own
    build and CI job, ``build_mpi`` / the ``mpi-validation`` workflow job).
  * Locally and in CI's ``build-and-test`` job, none of those flags are set,
    so none of those files are ever in ``build/compile_commands.json``.

A file with no compile-database entry isn't skipped by clang-tidy -- it falls
back to a guessed set of compiler flags and reports whatever that guess
breaks on (e.g. ``'mpi.h' file not found``), which is a build-configuration
artifact, not a real finding. Driving the file list from the compile
database itself -- exactly what this configuration actually built -- avoids
that false signal without a hand-maintained (and easily stale) exclude list
of vendor filenames. Test file coverage also legitimately varies between
checkouts -- ``Q3_BUILD_TESTS`` is a CMake option, and a developer who
configured with it OFF has no ``tests/`` entries to check -- which a static
file list could not track either. (CI's ``build-and-test`` job builds with it
ON for every event, so the test sources are always checked there.)

Usage::

    python scripts/repo/run_clang_tidy.py [--list] [-j JOBS] [substring ...]

With no positional arguments every compiled, non-``src/bindings/`` ``.cpp``
file is checked. Optional ``substring`` arguments narrow that down to files
whose repo-relative path contains any of them (handy for checking one
subtree while iterating). ``--list`` prints the selected files instead of
running clang-tidy on them.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path
from subprocess import CompletedProcess, SubprocessError, run

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
BUILD_DIRECTORY = REPOSITORY_ROOT / "build"
COMPILE_COMMANDS = BUILD_DIRECTORY / "compile_commands.json"

# Matches the pre-commit hook's previous exclusion: bindings/ is the nanobind
# Python module boundary (FetchContent-only headers, not in the compile
# database in any configuration this repo builds today).
EXCLUDED_RELATIVE_PREFIXES = ("src/bindings/",)

# clang-tidy diagnostic lines end in " [check-name]", or " [check-name,-warnings-as-errors]"
# when WarningsAsErrors promoted it (as .clang-tidy's WarningsAsErrors: "*" always
# does here) -- used for the per-check finding breakdown printed at the end of a run.
CHECK_NAME_RE = re.compile(r"\[([a-zA-Z0-9_.,-]+)\]\s*$")

DEFAULT_TIMEOUT_SECONDS = 180


def compiled_source_files() -> list[Path]:
    """Return every repo ``.cpp`` file this build actually compiled.

    Restricted to files that resolve under ``REPOSITORY_ROOT`` but NOT under
    ``BUILD_DIRECTORY``. The build-directory exclusion is what keeps
    third-party FetchContent sources out: FetchContent's default base dir is
    ``${CMAKE_BINARY_DIR}/_deps`` -- INSIDE the repository when the build dir
    is ``./build`` -- so a repo-root containment check alone happily accepts
    vendored yaml-cpp/googletest sources (CI proved it: 6
    google-readability-todo findings, all in ``build/_deps/yaml-cpp-src/``,
    none in Q3 code). We lint our code, not our dependencies. Also excludes
    ``EXCLUDED_RELATIVE_PREFIXES``. Deduplicated: a file built into more than
    one target (e.g. both ``q3`` and ``q3_tests`` objects) appears once.
    """
    entries = json.loads(COMPILE_COMMANDS.read_text())
    files: list[Path] = []
    seen: set[Path] = set()
    for entry in entries:
        path = Path(entry["file"]).resolve()
        if path.suffix != ".cpp" or path in seen:
            continue
        try:
            relative_posix = path.relative_to(REPOSITORY_ROOT).as_posix()
        except ValueError:
            continue  # outside the repository entirely -- not ours
        if path.is_relative_to(BUILD_DIRECTORY):
            continue  # vendored FetchContent source under build/_deps -- not ours
        if any(relative_posix.startswith(prefix) for prefix in EXCLUDED_RELATIVE_PREFIXES):
            continue
        seen.add(path)
        files.append(path)
    return sorted(files)


def clang_tidy_missing_message() -> str | None:
    """Return why clang-tidy cannot be run, or ``None`` when it is available.

    Without this the first worker thread dies on ``FileNotFoundError`` and the
    pool re-raises it as a traceback pointing at ``concurrent.futures``, which
    reads like a bug in this script rather than a missing tool.
    """
    if shutil.which("clang-tidy") is not None:
        return None
    if sys.platform == "darwin":
        return (
            "error: clang-tidy not found on PATH. Xcode does not ship it, so on macOS it\n"
            "comes from Homebrew's LLVM and that keg is not linked into PATH by default:\n"
            "  brew install llvm\n"
            '  export PATH="/opt/homebrew/opt/llvm/bin:$PATH"\n'
            "(see .pre-commit-config.yaml's clang-tidy hook comment)."
        )
    return (
        "error: clang-tidy not found on PATH. Install it, e.g.:\n  sudo apt-get install clang-tidy"
    )


def platform_extra_arguments() -> list[str]:
    """Return ``--extra-arg`` flags clang-tidy needs beyond the compile database.

    On macOS the compile database CMake writes records ``/usr/bin/c++`` with no
    ``-isysroot``: Apple's clang knows where the SDK is implicitly, so CMake
    never has to say. Homebrew's clang-tidy does not share that assumption and
    fails every file with ``'cctype' file not found`` -- a toolchain
    disagreement reported as 117 identical "findings". Naming the active SDK
    explicitly closes the gap.

    Empty everywhere else, so the Linux path (all of CI) is untouched.
    """
    if sys.platform != "darwin":
        return []
    try:
        result = run(["xcrun", "--show-sdk-path"], capture_output=True, text=True, timeout=30)
    except (OSError, SubprocessError):
        return []  # no xcrun: leave clang-tidy to its own defaults
    sdk_path = result.stdout.strip()
    if result.returncode != 0 or not sdk_path or not Path(sdk_path).is_dir():
        return []
    return ["--extra-arg=-isysroot", f"--extra-arg={sdk_path}"]


def run_clang_tidy_on(path: Path, extra_arguments: list[str]) -> CompletedProcess:
    """Run ``clang-tidy -p build`` on a single file, capturing its output."""
    return run(
        ["clang-tidy", "-p", str(BUILD_DIRECTORY), *extra_arguments, str(path)],
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        text=True,
        timeout=DEFAULT_TIMEOUT_SECONDS,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "substrings",
        nargs="*",
        help="only check compiled files whose repo-relative path contains one of these",
    )
    parser.add_argument("-j", "--jobs", type=int, default=4, help="parallel clang-tidy instances")
    parser.add_argument(
        "--list", action="store_true", help="print the selected files and exit, without running"
    )
    arguments = parser.parse_args()

    if not COMPILE_COMMANDS.exists():
        print(
            f"error: {COMPILE_COMMANDS} does not exist. Configure it first, e.g.:\n"
            "  cmake -S . -B build -DCMAKE_EXPORT_COMPILE_COMMANDS=ON\n"
            "(see TODOs.md § 3.3 and AGENTS-generic.md § Pre-commit).",
            file=sys.stderr,
        )
        return 1

    files = compiled_source_files()
    if arguments.substrings:
        files = [
            path
            for path in files
            if any(
                substring in path.relative_to(REPOSITORY_ROOT).as_posix()
                for substring in arguments.substrings
            )
        ]

    if arguments.list:
        for path in files:
            print(path.relative_to(REPOSITORY_ROOT).as_posix())
        return 0

    missing_message = clang_tidy_missing_message()
    if missing_message is not None:
        print(missing_message, file=sys.stderr)
        return 1

    extra_arguments = platform_extra_arguments()
    print(f"Running clang-tidy on {len(files)} compiled source file(s)...", file=sys.stderr)

    failed_files: list[Path] = []
    check_name_counts: Counter[str] = Counter()
    with ThreadPoolExecutor(max_workers=max(1, arguments.jobs)) as executor:
        results = executor.map(partial(run_clang_tidy_on, extra_arguments=extra_arguments), files)
        for path, result in zip(files, results):
            relative = path.relative_to(REPOSITORY_ROOT).as_posix()
            if result.returncode != 0:
                failed_files.append(path)
                print(f"==> {relative}", file=sys.stderr)
                print(result.stdout, file=sys.stderr)
                if result.stderr:
                    print(result.stderr, file=sys.stderr)
                for line in result.stdout.splitlines():
                    match = CHECK_NAME_RE.search(line)
                    if match:
                        # "-warnings-as-errors" rides along in the same bracket
                        # when WarningsAsErrors promoted the diagnostic; strip it
                        # so the breakdown is keyed by the real check name.
                        check_name = match.group(1).split(",")[0]
                        check_name_counts[check_name] += 1

    print(
        f"\n{len(files)} file(s) checked, {len(failed_files)} with findings or errors.",
        file=sys.stderr,
    )
    if check_name_counts:
        print("\nFindings by check:", file=sys.stderr)
        for check_name, count in check_name_counts.most_common():
            print(f"  {count:5d}  {check_name}", file=sys.stderr)

    return 1 if failed_files else 0


if __name__ == "__main__":
    raise SystemExit(main())
