"""q3.binary — locate the ``q3`` binary and run calculations via subprocess.

This module provides the low-level interface between the Python layer and the
compiled C++ binary.  Higher-level code (flavors, workflows) should call the
helpers here rather than spawning subprocesses directly.

Public API
----------
find_binary(hint)
    Locate the ``q3`` binary.  Accepts an explicit path, falls back to
    auto-detection in common build directories.

ExecutionPolicy(number_of_processes, number_of_kpoint_pools, ...)
    The engine execution policy every run of a calculation is started with:
    ranks, k-point pools, band distribution, GPU precision and the MPI
    launcher.  A workflow hands one to each of its units, so the SCF, the
    NSCF and every relaxation step run alike.

run_q3(input_path, *, q3_binary, execution_policy, extra_args, capture_output)
    Run ``q3 <input_path>`` and return ``results.json`` as a dict.
"""

from __future__ import annotations

import os
import shlex
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

from q3.errors import Q3RuntimeError, diagnostic_code
from q3.io.output_writer import raise_if_not_converged
from q3.io.results.classes import Results

#: How many of a failed run's last lines its error carries: enough for the
#: engine's closing report and the diagnostic lines that follow it.
ERROR_TAIL_LINES = 30

#: How :class:`ExecutionPolicy` starts more than one process unless told otherwise.
DEFAULT_MPI_LAUNCHER = ("mpirun", "-np", "{number_of_processes}")

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
# Execution policy                                                             #
# --------------------------------------------------------------------------- #


class ExecutionPolicy(BaseModel):
    """The execution policy every engine run of a calculation is started with.

    The engine's own execution policy (``PlanewaveSerialExecutionPolicy``,
    ``PlanewaveMPIExecutionPolicy``, the GPU ones) is built inside the binary
    from the ranks it was launched on and its command-line options; Python
    cannot hand it the object.  This describes that policy from outside, in
    the engine's own terms -- the fields are the ``execution`` block of
    ``results.json`` -- and starts the binary so that it builds it.

    One value, handed to a workflow, reaches each of its units: the SCF, the
    NSCF of a band structure or a DOS, every step of a relaxation, every
    strain of an equation of state.

    It names no binary.  Which one runs -- ``q3``, ``q3_mpi``, a GPU binary --
    is the caller's ``q3_binary``; this says only how it is started.  The
    fields are those of ``schemas/q3/execution_policy.json``.

    Parameters
    ----------
    number_of_processes:
        MPI ranks.  One runs the binary directly, without the launcher.
    number_of_kpoint_pools:
        k-point pools (``--pools``).  ``None`` leaves the choice to the input's
        ``control.number_of_kpoint_pools``, or to the engine.
    distribute_bands:
        Split each pool's bands over its ranks (``--distribute-bands``).
    precision:
        Arithmetic of the GPU binaries (``--precision``); ``None`` is theirs.
    launcher:
        The MPI launcher, one word per entry, with ``{number_of_processes}``
        where the rank count goes.  A string is split as a shell would.

    Example::

        execution_policy = ExecutionPolicy(number_of_processes=8, number_of_kpoint_pools=8)
        run_band_structure_workflow(
            "scf.yml", q3_binary="q3_mpi", execution_policy=execution_policy
        )
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    number_of_processes: int = Field(1, ge=1)
    number_of_kpoint_pools: int | None = Field(None, ge=1)
    distribute_bands: bool = False
    precision: Literal["double", "mixed"] | None = None
    launcher: tuple[str, ...] = DEFAULT_MPI_LAUNCHER

    @field_validator("launcher", mode="before")
    @classmethod
    def _split_as_a_shell_would(cls, value: Any) -> Any:
        if isinstance(value, str):
            return tuple(shlex.split(value))
        if isinstance(value, Sequence):
            return tuple(str(word) for word in value)
        return value

    @property
    def is_default(self) -> bool:
        """True for one process on the binary's defaults: the command is the binary alone.

        Only then may a calculation run in-process through the ``q3_core``
        bindings, which can honour neither a launcher nor a binary option.
        """
        return not self.launch_words and not self.binary_options

    def check_results(self, results: Results) -> None:
        """Refuse results that ran on a different number of processes than this policy asked for.

        Launching the serial ``q3`` under ``mpirun -np 4`` runs four separate
        serial calculations over the same output files, and every one exits 0;
        the ``execution`` block the engine writes is what tells the two apart.

        Raises:
            Q3RuntimeError: When the results report another process count.
        """
        execution = results.runtime_log.execution if results.runtime_log else None
        reported = execution.number_of_processes if execution else None
        if reported is not None and reported != self.number_of_processes:
            raise Q3RuntimeError(
                f"The execution policy asked for {self.number_of_processes} processes, but "
                f"the run reports {reported}. Under a launcher, give q3_binary an MPI build "
                "(q3_mpi or q3_mpi_gpu_cuda): the serial binary runs one copy per process."
            )

    @property
    def launch_words(self) -> list[str]:
        """What goes before the binary: the launcher for more than one process, else nothing."""
        if self.number_of_processes == 1:
            return []
        return [word.format(number_of_processes=self.number_of_processes) for word in self.launcher]

    @property
    def binary_options(self) -> list[str]:
        """The binary's own options that build this policy."""
        options: list[str] = []
        if self.number_of_kpoint_pools is not None:
            options += ["--pools", str(self.number_of_kpoint_pools)]
        if self.distribute_bands:
            options.append("--distribute-bands")
        if self.precision is not None:
            options += ["--precision", self.precision]
        return options

    def command(self, binary: str | Path, *arguments: str) -> list[str]:
        """The command that runs *binary* with *arguments* under this policy."""
        return [*self.launch_words, str(binary), *arguments, *self.binary_options]


def execution_policy_or_serial(execution_policy: ExecutionPolicy | None) -> ExecutionPolicy:
    """*execution_policy*, or the serial one (one process, the binary's defaults) for ``None``."""
    return ExecutionPolicy() if execution_policy is None else execution_policy


# --------------------------------------------------------------------------- #
# Subprocess runner                                                            #
# --------------------------------------------------------------------------- #


def run_q3(
    input_path: str | Path,
    *,
    q3_binary: str | Path | None = None,
    execution_policy: ExecutionPolicy | None = None,
    extra_args: list[str] | None = None,
    capture_output: bool = True,
) -> Results:
    """Run ``q3 <input_path>`` and return the parsed ``results.json``.

    The binary writes ``results.json`` into the ``outdir`` specified inside the
    YAML input file.  This helper reads that file and returns it as
    :class:`q3.io.results.classes.Results`.

    Parameters
    ----------
    input_path:
        Path to the YAML input file.
    q3_binary:
        Explicit path to the ``q3`` binary.  If ``None``, ``find_binary()``
        is called with no hint.
    execution_policy:
        The execution policy the binary is started with (ranks, pools, the
        launcher).  ``None`` runs one process with the binary's defaults.
    extra_args:
        Additional command-line arguments appended after ``input_path``.
    capture_output:
        When ``True`` (default) stdout/stderr are captured.  When ``False``
        they are shown as they arrive and still kept, so a failure names its
        cause either way.

    Returns
    -------
    Results
        The ``results.json`` file written by the binary.

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

    command = execution_policy_or_serial(execution_policy).command(
        binary, str(input_path), *(extra_args or [])
    )
    _completed("q3", command, capture_output=capture_output)

    # Locate results.json — we need to parse the input YAML to find outdir.
    results_json_path = _find_results_json(input_path)

    results = Results.read(results_json_path)

    # Check SCF convergence — the binary exits 0 even on non-convergence.
    raise_if_not_converged(results, returncode=0)
    execution_policy_or_serial(execution_policy).check_results(results)

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
