"""q3.binary — locate the ``q3`` binary and run calculations via subprocess.

This module provides the low-level interface between the Python layer and the
compiled C++ binary.  Higher-level code (flavors, workflows) should call the
helpers here rather than spawning subprocesses directly.

Public API
----------
find_binary(hint)
    Locate the ``q3`` binary.  Accepts an explicit path, falls back to
    auto-detection in common build directories.

run_q3(input_path, *, q3_binary, output_directory, capture_output)
    Run ``q3 <input_path>`` and return ``results.json`` as a dict.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import yaml

from q3.errors import Q3RuntimeError, diagnostic_code
from q3.io.output_writer import raise_if_not_converged

#: How many of a failed run's last lines its error carries: enough for the
#: engine's closing report and the diagnostic lines that follow it.
ERROR_TAIL_LINES = 30

# --------------------------------------------------------------------------- #
# Binary location                                                              #
# --------------------------------------------------------------------------- #


def find_binary(hint: str | Path | None = None) -> Path:
    """Locate the ``q3`` executable.

    Resolution order:
    1. *hint* — an explicit path supplied by the caller.
    2. ``Q3_BINARY`` environment variable.
    3. Auto-detection in common build directories relative to the repository
       root (``build/q3``, ``build_mpi/q3``, ``cmake-build-release/q3``).

    Raises:
        FileNotFoundError: When the binary cannot be found.
    """
    candidates: list[Path] = []

    if hint is not None:
        candidates.append(Path(hint).expanduser().resolve())

    env_binary = os.environ.get("Q3_BINARY")
    if env_binary:
        candidates.append(Path(env_binary).expanduser().resolve())

    # Auto-detect relative to the repository root (two levels up from this file:
    # src/py/q3/binary.py → src/py/q3 → src/py → src → project_root)
    project_root = Path(__file__).parent.parent.parent.parent
    for relative_path in ("build/q3", "build_mpi/q3", "cmake-build-release/q3"):
        candidates.append(project_root / relative_path)

    for candidate in candidates:
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return candidate

    search_description = ", ".join(str(c) for c in candidates)
    raise FileNotFoundError(
        f"q3 binary not found.  Searched: {search_description}\n"
        "Set Q3_BINARY environment variable or pass q3_binary= explicitly."
    )


# --------------------------------------------------------------------------- #
# Subprocess runner                                                            #
# --------------------------------------------------------------------------- #


def run_q3(
    input_path: str | Path,
    *,
    q3_binary: str | Path | None = None,
    extra_args: list[str] | None = None,
    capture_output: bool = True,
) -> dict:
    """Run ``q3 <input_path>`` and return the parsed ``results.json``.

    The binary writes ``results.json`` into the ``outdir`` specified inside the
    YAML input file.  This helper reads that file and returns its contents as a
    Python dict.

    Parameters
    ----------
    input_path:
        Path to the YAML input file.
    q3_binary:
        Explicit path to the ``q3`` binary.  If ``None``, ``find_binary()``
        is called with no hint.
    extra_args:
        Additional command-line arguments appended after ``input_path``.
    capture_output:
        When ``True`` (default) stdout/stderr are captured.  When ``False``
        they are shown as they arrive and still kept, so a failure names its
        cause either way.

    Returns
    -------
    dict
        Parsed contents of the ``results.json`` file written by the binary.

    Raises
    ------
    Q3RuntimeError
        If the binary exits with a non-zero status.
    SCFNotConvergedError
        If the binary exits successfully but reports non-convergence in
        ``results.json``.
    FileNotFoundError
        If ``results.json`` is not found after a successful run.
    """
    input_path = Path(input_path).expanduser().resolve()
    binary = find_binary(q3_binary)

    command = [str(binary), str(input_path)] + (extra_args or [])
    _completed("q3", command, capture_output=capture_output)

    # Locate results.json — we need to parse the input YAML to find outdir.
    results_json_path = _find_results_json(input_path)

    with open(results_json_path) as file_handle:
        results = json.load(file_handle)

    # Check SCF convergence — the binary exits 0 even on non-convergence.
    raise_if_not_converged(results, returncode=0)

    return results


def _find_results_json(input_path: Path) -> Path:
    """Find the ``results.json`` path written by a q3 run for *input_path*.

    Reads the ``outdir`` and ``prefix`` from the YAML to construct the path.
    Falls back to searching the directory containing the input file.
    """
    with open(input_path) as file_handle:
        config = yaml.safe_load(file_handle)

    control = config.get("control", {})
    outdir = control.get("outdir", ".")
    prefix = control.get("prefix", "q3")

    # outdir may be relative to the input file's directory.
    base_directory = input_path.parent
    output_directory = (base_directory / outdir).resolve()

    # All current CLI and binding entry points write the prefixed canonical name.
    results_json = output_directory / f"{prefix}_results.json"
    if results_json.exists():
        return results_json

    # Bindings released before the canonical contract wrote an unprefixed name.
    legacy_path = output_directory / "results.json"
    if legacy_path.exists():
        return legacy_path

    raise FileNotFoundError(
        f"results.json not found after q3 run.\n"
        f"Searched: {results_json}\n"
        f"Also tried: {legacy_path}\n"
        f"Check that 'outdir' in {input_path} is correct."
    )


# --------------------------------------------------------------------------- #
# The utility binary                                                           #
# --------------------------------------------------------------------------- #


def find_utility_binary(hint: str | Path | None = None) -> Path:
    """Locate the ``q3_utility`` executable (subcommands over a run's saved records).

    An explicit *hint* is authoritative (a wrong path raises rather than
    falling back, unlike :func:`find_binary`); otherwise the
    ``Q3_UTILITY_BINARY`` environment variable, then the build directories
    beside the repository root. It is built by the same CMake invocation as
    ``q3`` (target ``q3_utility``).
    """
    candidates: list[Path] = []
    if hint is not None:
        # An explicit path is authoritative: silently falling back to a build
        # directory would run a different binary than the one the caller named.
        explicit = Path(hint).expanduser().resolve()
        if not (explicit.is_file() and os.access(explicit, os.X_OK)):
            raise FileNotFoundError(f"q3_utility not found at {explicit}")
        return explicit
    env_binary = os.environ.get("Q3_UTILITY_BINARY")
    if env_binary:
        candidates.append(Path(env_binary).expanduser().resolve())
    project_root = Path(__file__).parent.parent.parent.parent
    for relative_path in (
        "build/q3_utility",
        "build_mpi/q3_utility",
        "cmake-build-release/q3_utility",
    ):
        candidates.append(project_root / relative_path)
    for candidate in candidates:
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return candidate
    search_description = ", ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(
        f"q3_utility binary not found.  Searched: {search_description}\n"
        "Set Q3_UTILITY_BINARY or pass q3_utility_binary= explicitly (CMake target q3_utility)."
    )


def run_q3_utility(
    arguments: list[str],
    *,
    q3_utility_binary: str | Path | None = None,
    capture_output: bool = True,
    cwd: str | Path | None = None,
) -> subprocess.CompletedProcess:
    """Run ``q3_utility <arguments>`` (in ``cwd`` if given) and return the completed process.

    Unlike :func:`run_q3` there is no single results file to read back: each
    subcommand writes its own artefact (``interpolate`` a results JSON,
    ``convert`` the exported files or the imported record), and the caller
    knows which. A non-zero exit raises :class:`Q3RuntimeError` with the
    binary's stderr, so a failure names its cause rather than surfacing later
    as a missing file.

    ``capture_output=False`` shows the output as it arrives and still keeps it,
    so the error carries the binary's own words either way: the Wannier ladder
    chooses its next rung from the diagnostic a failed evaluation prints, and
    output sent only to the terminal reached it as a bare exit code, which it
    can only reject on.
    """
    binary = find_utility_binary(q3_utility_binary)
    return _completed(
        "q3_utility", [str(binary), *arguments], capture_output=capture_output, cwd=cwd
    )


# --------------------------------------------------------------------------- #
# Running either binary                                                        #
# --------------------------------------------------------------------------- #


def _completed(
    program: str,
    command: list[str],
    *,
    capture_output: bool,
    cwd: str | Path | None = None,
) -> subprocess.CompletedProcess:
    """Run *command* to the end; a non-zero exit raises with the binary's own words.

    An empty stderr next to a non-zero exit is the normal shape of a Q3
    failure, not a silent one: the engine logs through stdout, so the error
    carries the end of stdout as well. Without it a driver that captured the
    output reports "exited with code 1" and nothing else, which is what the
    Wannier campaign's first gallium arsenide row said. Output that was shown
    rather than captured arrives as one merged stream, and its end stands in
    for both.
    """
    if capture_output:
        result = subprocess.run(command, capture_output=True, text=True, cwd=cwd)
        stderr, stdout_tail = result.stderr, _tail(result.stdout or "")
    else:
        result = _run_showing_output(command, cwd)
        stderr, stdout_tail = _tail(result.stdout), ""
    if result.returncode != 0:
        raise Q3RuntimeError(
            f"{program} exited with code {result.returncode}.\n"
            f"Command: {' '.join(command)}\n"
            f"Stderr:\n{stderr}\n"
            f"Last lines of stdout:\n{stdout_tail}",
            returncode=result.returncode,
            stderr=stderr,
            stdout=result.stdout or "",
            command=command,
            diagnostic=diagnostic_code((result.stdout or "") + "\n" + stderr),
        )
    return result


def _tail(text: str, lines: int = ERROR_TAIL_LINES) -> str:
    """The last *lines* lines of *text*, which is where the binaries put their error text."""
    return "\n".join(text.splitlines()[-lines:])


def _run_showing_output(command: list[str], cwd: str | Path | None) -> subprocess.CompletedProcess:
    """Run *command*, echoing both streams line by line as they arrive, and keep them.

    Merged into one stream in the order they were written, which is what a
    reader of the terminal saw; the completed process carries it as stdout.
    """
    kept: list[str] = []
    with subprocess.Popen(
        command, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    ) as process:
        assert process.stdout is not None
        for line in process.stdout:
            sys.stdout.write(line)
            kept.append(line)
    return subprocess.CompletedProcess(command, process.returncode, "".join(kept), "")
