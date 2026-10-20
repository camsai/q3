"""
Session fixtures the regression tests share: `q3_utility`, one finished
silicon SCF, and that run's density exported as a cube.
"""

from pathlib import Path

import pytest
from volumetric_helpers import SHIFTED_OVERLAY, export, run_scf


@pytest.fixture(scope="session")
def q3_utility_binary(q3_command):
    """`q3_utility`, which CMake builds beside `q3` in the same tree."""
    utility = Path(list(q3_command)[-1]).resolve().parent / "q3_utility"
    if not utility.is_file():
        pytest.skip(f"q3_utility not found at {utility} (build the q3_utility target)")
    return str(utility)


@pytest.fixture(scope="session")
def shifted_run(q3_command, tmp_path_factory):
    """A 2x2x2 shifted silicon SCF: the ordinary, whole-G-sphere case."""
    return run_scf(q3_command, tmp_path_factory, SHIFTED_OVERLAY, "volumetric_shifted")


@pytest.fixture(scope="session")
def density_cube(q3_utility_binary, shifted_run):
    export(q3_utility_binary, shifted_run, "--field", "density", "--format", "cube")
    return shifted_run["output_directory"] / f"{shifted_run['prefix']}_density.cube"
