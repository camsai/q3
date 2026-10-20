"""tests/python/conftest.py — fixtures shared by the Python workflow tests."""

import os
from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def q3_binary(request):
    """Resolve the q3 binary path for Python tests."""
    # Honour the same --q3-bin CLI option defined in the root conftest.
    binary = request.config.getoption("--q3-bin", default=None)
    if binary:
        return Path(binary).resolve()

    q3_root = Path(__file__).parent.parent.parent
    for candidate in ("build/q3", "build_mpi/q3", "cmake-build-release/q3"):
        path = q3_root / candidate
        if path.is_file() and os.access(path, os.X_OK):
            return path

    pytest.skip("q3 binary not found. Use --q3-bin.")


@pytest.fixture(scope="session")
def examples_dir():
    """Return the path to the examples directory."""
    return Path(__file__).parent.parent.parent / "examples"


@pytest.fixture(scope="session")
def si_scf_input(examples_dir):
    """Path to the Si NC SCF example input."""
    return examples_dir / "total_energy--semiconductor--silicon" / "nc.yml"
