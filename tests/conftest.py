"""Root conftest — shared pytest fixtures and hooks for q3 tests."""

import os
import sys
from pathlib import Path

import pytest

# Make modules in tests/utils/qe/ importable (binary_readers, qe_runner).
sys.path.insert(0, str(Path(__file__).parent / "utils" / "qe"))

# Make the build directory importable so we can load the C++ bindings module (q3_core)
sys.path.insert(0, str(Path(__file__).parent.parent / "build"))


# ── CLI options ──────────────────────────────────────────────────────


def pytest_addoption(parser):
    """Add command-line options for specifying the q3 binary."""
    parser.addoption(
        "--q3-bin",
        default=None,
        help="Path to the q3 binary (default: auto-detect in build/)",
    )
    parser.addoption(
        "--q3-cmd",
        default=None,
        help="Full command to run q3, e.g. 'mpirun -np 2 ./build/q3'",
    )
    parser.addoption(
        "--q3-gpu-bin",
        default=None,
        help="Path to the q3 GPU binary (default: auto-detect in build*/)",
    )
    parser.addoption(
        "--tier2",
        action="store_true",
        default=False,
        help="Run tier-2 (slow) tests: full-SCF supercell convergence tests",
    )
    parser.addoption(
        "--tier3",
        action="store_true",
        default=False,
        help="Run tier-3 (monthly) tests as well: campaign-scale runs that take hours; implies --tier2",
    )


# ── Tier-based test collection ───────────────────────────────────────


def pytest_collection_modifyitems(config, items):
    """Skip @pytest.mark.slow tests unless --tier2 is passed, and @pytest.mark.monthly unless --tier3.

    Three tiers, each including the ones below it: the unit tests always, the
    slow chains on --tier2, and the campaign-scale runs -- the whole Wannier
    roster, the Wannier90 parity cases -- on --tier3, which the scheduled
    workflow runs once a month rather than on every merge.
    """
    tier3 = config.getoption("--tier3")
    tier2 = tier3 or config.getoption("--tier2")
    skip_slow = pytest.mark.skip(reason="tier-2 test: pass --tier2 to run")
    skip_monthly = pytest.mark.skip(reason="tier-3 test: pass --tier3 to run")
    for item in items:
        if "monthly" in item.keywords and not tier3:
            item.add_marker(skip_monthly)
        elif "slow" in item.keywords and not tier2:
            item.add_marker(skip_slow)


# ── Fixtures ─────────────────────────────────────────────────────────


@pytest.fixture(scope="session")
def q3_command(request):
    """Resolve the q3 command to use for tests.

    Checks (in order):
      1. --q3-cmd CLI option (full command string, split on whitespace)
      2. --q3-bin CLI option (path to binary)
      3. Auto-detection in common build directories
    """
    cmd = request.config.getoption("--q3-cmd")
    if cmd:
        return cmd.split()

    binary = request.config.getoption("--q3-bin")
    if binary:
        return [str(Path(binary).resolve())]

    # Look in current q-3-minimal root and build folders
    q3_root = Path(__file__).parent.parent
    for candidate in ["build/q3", "build_mpi/q3", "cmake-build-release/q3"]:
        path = q3_root / candidate
        if path.is_file() and os.access(path, os.X_OK):
            return [str(path.resolve())]

    pytest.skip("q3 binary not found. Use --q3-bin or --q3-cmd.")


@pytest.fixture(scope="session")
def q3_gpu_binary(request):
    """Resolve the q3 GPU binary to use for GPU parity tests.

    Checks (in order):
      1. --q3-gpu-bin CLI option
      2. Auto-detection in common build directories (Metal on macOS,
         CUDA / HIP / SYCL builds on Linux)
    """
    binary = request.config.getoption("--q3-gpu-bin")
    if binary:
        path = Path(binary).resolve()
        if not path.is_file():
            pytest.skip(f"q3 GPU binary not found at {path}")
        return str(path)

    q3_root = Path(__file__).parent.parent
    candidates = [
        "build/q3_gpu",
        "build/q3_gpu_cuda",
        "build_cuda/q3_gpu_cuda",
        "build_hip/q3_gpu_hip",
        "build_sycl/q3_gpu_sycl",
    ]
    for candidate in candidates:
        path = q3_root / candidate
        if path.is_file() and os.access(path, os.X_OK):
            return str(path.resolve())

    pytest.skip("q3 GPU binary not found. Use --q3-gpu-bin.")
