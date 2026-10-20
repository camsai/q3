"""Regression tests for the `tests/validation/` autouse guards.

These pin a failure mode that is invisible in CI: a guard that skips *more* than
it means to. Skips are not failures, so a validation suite that stops running
reports the same green tick as one that runs and passes. The MPI CI leg builds
`build_mpi` and would mask a repeat of this, so the pin lives here, in the
serial Python suite, where the prerequisite is genuinely absent.
"""

import importlib.util
import inspect
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
VALIDATION_CONFTEST_PATH = PROJECT_ROOT / "tests" / "validation" / "conftest.py"
MODULE_SPECIFICATION = importlib.util.spec_from_file_location(
    "validation_conftest", VALIDATION_CONFTEST_PATH
)
validation_conftest = importlib.util.module_from_spec(MODULE_SPECIFICATION)
MODULE_SPECIFICATION.loader.exec_module(validation_conftest)

# A validation module that needs no MPI. Any non-`test_mpi_*` module proves the
# point; this one is picked because it collects and runs without the q3 binary
# at all, so no skip can come from anywhere but the guard under test.
NON_MPI_VALIDATION_MODULE = "tests/validation/test_qe_reference_binary_guards.py"

MISSING_MPI_BINARY_SKIP_REASON = "q3 MPI binary not found"


def test_require_mpi_binary_does_not_declare_q3_bin_as_a_parameter():
    """The autouse guard must resolve `q3_bin` in its body, not in its signature.

    Pytest resolves fixture parameters before the body runs, so declaring
    `q3_bin` applies its "MPI binary not found" skip to every test in
    `tests/validation/` -- not just the `test_mpi_*` modules the guard is for.
    The filename check in the body never gets a chance to return first.

    Asserted on the signature because that is where the bug lives: the body has
    always been correct, and re-adding the parameter is the specific edit that
    silently disables the suite again.
    """
    parameters = inspect.signature(validation_conftest.require_mpi_binary).parameters

    assert "q3_bin" not in parameters, (
        "require_mpi_binary must not take q3_bin as a fixture parameter; "
        "resolve it inside the body with request.getfixturevalue('q3_bin') "
        "after the test_mpi_ filename check, or every validation test skips "
        "when ./build_mpi/q3 is absent."
    )


def test_non_mpi_validation_tests_run_without_an_mpi_binary():
    """A non-MPI validation module must not skip for a missing MPI binary.

    The behavioural half of the test above: runs a real collection with
    `--q3-bin` pointed at a path that cannot exist, and fails if the MPI guard's
    skip reason reaches a module it does not apply to. Tests in the module may
    still skip for their own prerequisites (licensed POTCARs), which is why the
    assertion is on the *reason* rather than on a pass count.
    """
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            NON_MPI_VALIDATION_MODULE,
            "--q3-bin",
            "/nonexistent/build_mpi/q3",
            "-q",
            "-rs",
            "-p",
            "no:cacheprovider",
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        timeout=300,
    )

    assert MISSING_MPI_BINARY_SKIP_REASON not in completed.stdout, (
        f"{NON_MPI_VALIDATION_MODULE} skipped for a missing MPI binary. The "
        "autouse require_mpi_binary guard is firing outside the test_mpi_ "
        f"modules it is scoped to.\n\n{completed.stdout}"
    )
