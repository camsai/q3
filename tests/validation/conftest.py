"""Shared fixtures for validation tests."""

import os
import subprocess
from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def q3_bin(pytestconfig):
    """Resolve the q3 MPI binary from --q3-bin (default: ./build_mpi/q3_mpi).

    The --q3-bin option itself is registered in the root tests/conftest.py.

    The default names `q3_mpi`, the link an MPI build tree puts beside its one
    `q3` (#316), so it fails to resolve in a serial tree rather than quietly
    naming the serial binary. Whether the binary actually is the MPI entry
    point is `require_mpi_binary`'s question, not this fixture's -- see its
    docstring for why the skips live there.
    """
    binary_option = pytestconfig.getoption("--q3-bin") or "./build_mpi/q3_mpi"
    binary = Path(binary_option).resolve()
    if not binary.is_file():
        pytest.skip(f"q3 MPI binary not found at {binary}")
    return str(binary)


@pytest.fixture(scope="session")
def q3_utility_command(q3_command):
    """Resolve `q3_utility`, which CMake builds beside `q3` in the same tree.

    Derived from the already-resolved `q3_command` rather than from a new CLI
    option so that whichever build directory the suite was pointed at is the
    one whose utility gets tested -- pairing a converter from one build with a
    `q3` from another would make a byte-identity comparison meaningless.
    `q3_command` may be a full command (e.g. `mpirun -np 2 ./build/q3`), so the
    binary is its last token; the utility is serial and is invoked directly.

    Skips when the binary is absent: the GPU CI workflows build a named target
    subset (`--target q3 q3_gpu_*`) that does not include `q3_utility`.
    """
    utility = Path(list(q3_command)[-1]).resolve().parent / "q3_utility"
    if not utility.is_file() or not os.access(utility, os.X_OK):
        pytest.skip(f"q3_utility not found at {utility} (build the q3_utility target)")
    return [str(utility)]


@pytest.fixture(scope="session")
def q3_convert_command(q3_utility_command):
    """`q3_utility convert`, the converter subcommand, as a command prefix."""
    return [*q3_utility_command, "convert"]


# ── MPI binary guard ──────────────────────────────────────────────────────────


def _is_mpi_binary(binary_path):
    """Whether `binary_path` is `q3_mpi` rather than the serial `q3`.

    `mpirun -np 4 <serial q3>` does not fail: it starts four *independent*
    single-rank calculations that all write the same results file, and they
    race. The file usually survives -- the last writer truncates and wins --
    but sometimes interleaves, and the test then dies in `json.loads` with
    "Extra data" a long way from the cause. That looks exactly like a physics
    regression and is not one; it cost two false alarms before being pinned
    down, and a third afterwards.

    The third is why this asks the binary what it is rather than what it links
    against. An MPI build tree used to hold a serial `q3` and `q3_mpi` side by
    side, both built from a library compiled with Q3_WITH_MPI, so both linked
    MPI -- but only `q3_mpi` was `src/main_mpi.cpp`. Inspecting the linkage
    passed the serial one straight through. A tree now builds one binary, and
    in an MPI tree `q3` *is* the MPI entry point (#316), but a serial binary
    can still be handed in by path. `--version` prints the entry point's own
    name, whatever the file is called, and settles it.

    Falls back to running the test when the binary will not answer, rather than
    skipping silently: a missed check is better than a missed test.
    """
    try:
        identity = subprocess.run(
            [str(binary_path), "--version"], capture_output=True, text=True, timeout=60
        )
    except (OSError, subprocess.SubprocessError):
        return True
    reported = (identity.stdout + identity.stderr).strip()
    if not reported:
        return True
    return reported.startswith("q3_mpi ")


@pytest.fixture(autouse=True)
def require_mpi_binary(request):
    """Refuse to run any MPI test against the serial binary — see `_is_mpi_binary`.

    Applies to every `test_mpi_*` module rather than one of them: the guard was
    first written into `test_mpi_band_parallelism.py` alone, and the other two
    modules went on failing intermittently for exactly the reason it exists.

    Skipping, not failing: `--q3-bin` is shared with the serial suites, and CI's
    tier-1 job runs this whole directory with `--q3-bin ./build/q3`. Asking for
    the serial tests is not a claim that the MPI ones should run.

    `q3_bin` is resolved inside the body, after the filename check, and never as
    a fixture parameter. Pytest resolves parameters before the body runs, so
    declaring it would apply `q3_bin`'s own "MPI binary not found" skip to every
    test in this directory -- including the zero-prerequisite guards whose
    docstrings promise they run everywhere, and the `q3_command` tests that want
    the serial build. A serial-only checkout then reports the whole validation
    suite as skipped rather than failed, so nothing notices it stopped running.
    The MPI CI leg builds `build_mpi` and masks this; only serial environments
    see it.
    """
    if not request.node.nodeid.split("/")[-1].startswith("test_mpi_"):
        return
    q3_bin = request.getfixturevalue("q3_bin")
    if not _is_mpi_binary(q3_bin):
        pytest.skip(
            f"{q3_bin} is not the MPI binary; these tests need q3_mpi (e.g. "
            f"--q3-bin {Path(q3_bin).parent / 'q3_mpi'}). Running them against the serial binary "
            "makes four independent processes race on one results file, which fails "
            "intermittently and looks like a physics regression."
        )
