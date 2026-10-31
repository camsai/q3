"""
A non-self-consistent run on a GPU binary, after a self-consistent one on the same binary.

A band structure or a DOS is an SCF and then a non-self-consistent pass, and a
workflow started on a GPU binary runs both there. Until the GPU entry points
handed every flavor their diagonalizer, they ran an SCF whatever the input
asked for, so the NSCF of such a workflow never reached the device; the energy
parity of test_gpu_energy_parity.py covers the SCF alone.

The pin: an SCF and an NSCF both on the GPU binary give the eigenvalues and
occupations a serial SCF and NSCF give at the same listed k-points, in FP64
and in mixed precision, and report themselves as the NSCF.

Run:
    pytest tests/validation/test_gpu_non_self_consistent_field_parity.py -v \\
        --q3-bin ./build_cuda/q3 --q3-gpu-bin ./build_cuda/q3_gpu_cuda
"""

import subprocess

import numpy as np
import pytest
from q3.io.results.classes import Results

from helpers import EXAMPLES_DIRECTORY, apply_config_overlay, eigenvalues_ry, occupations

SILICON_INPUT = EXAMPLES_DIRECTORY / "total_energy--semiconductor--silicon" / "nc.yml"
PREFIX = "si_nc"

SCF_OVERLAY = {"control": {"use_deterministic_fft_planning": True}}
"""Fixed FFT plans on both sides; see DETERMINISTIC_FFT_PLANNING_OVERLAY in test_gpu_energy_parity.py."""

NSCF_KPOINTS = [
    [0.0, 0.0, 0.0, 1.0],
    [0.5, 0.0, 0.5, 1.0],
    [0.5, 0.5, 0.5, 1.0],
    [0.1, 0.2, 0.3, 1.0],
]
"""Gamma, X, L and a point of no symmetry, none on the SCF grid."""

NUMBER_OF_OCCUPIED_BANDS = 4

FP64_OCCUPIED_TOLERANCE_RY = 1.0e-8
"""
How far the GPU's occupied FP64 eigenvalues may sit from the serial ones (Ry).

The SCF energies agree to 1e-10 Ry (test_gpu_energy_parity.py); an eigenvalue
is first order where the energy is variational, so it is pinned less tightly.
A k-point's eigenvalues in another's slot would differ by tenths of a Rydberg.
"""

MIXED_PRECISION_OCCUPIED_TOLERANCE_RY = 1.0e-5
"""The mixed-precision energy bound of test_gpu_energy_parity.py, applied per eigenvalue."""

EMPTY_EIGENVALUE_TOLERANCE_RY = 1.0e-4
"""
The empty bands, which the Davidson solver refines only until a step moves them
by less than math::tolerances::empty_band_tolerance_floor (1e-5 Ry): where it
stops depends on round-off, which differs between the backends. Ten times the
floor, as test_mpi_non_self_consistent_field.py sets for k-point pools.
"""


def _run_ok(command, working_directory, description):
    result = subprocess.run(
        command, cwd=str(working_directory), capture_output=True, text=True, timeout=1200
    )
    assert result.returncode == 0, (
        f"{description} failed (exit {result.returncode})\n"
        f"stdout: {result.stdout[-2000:]}\nstderr: {result.stderr[-2000:]}"
    )


def _scf_then_nscf(binary, working_directory, description, extra_arguments=()):
    """Run the SCF and then the NSCF on `binary`, and return the NSCF's results."""
    scf_input = apply_config_overlay(SILICON_INPUT, SCF_OVERLAY, working_directory)
    _run_ok([binary, str(scf_input), *extra_arguments], working_directory, f"{description} SCF")
    nscf_overlay = {
        "control": {
            **SCF_OVERLAY["control"],
            "calculation": "nscf",
            "kpoints_list": NSCF_KPOINTS,
        }
    }
    nscf_input = apply_config_overlay(SILICON_INPUT, nscf_overlay, working_directory)
    _run_ok([binary, str(nscf_input), *extra_arguments], working_directory, f"{description} NSCF")
    return Results.read(working_directory / "output" / f"{PREFIX}_results.json")


@pytest.fixture(scope="module")
def serial_results(q3_bin, tmp_path_factory):
    return _scf_then_nscf(q3_bin, tmp_path_factory.mktemp("serial"), "serial")


@pytest.mark.parametrize(
    "extra_arguments,occupied_tolerance",
    [
        ((), FP64_OCCUPIED_TOLERANCE_RY),
        (("--precision", "mixed"), MIXED_PRECISION_OCCUPIED_TOLERANCE_RY),
    ],
    ids=["fp64", "mixed_precision"],
)
def test_gpu_nscf_matches_serial(
    q3_gpu_binary, serial_results, tmp_path, extra_arguments, occupied_tolerance
):
    gpu_results = _scf_then_nscf(q3_gpu_binary, tmp_path, "GPU", extra_arguments)

    # Until the GPU entry points handed every flavor their diagonalizer, this
    # input ran an SCF and reported it under that name.
    assert gpu_results.runtime_log.flavor == "non_self_consistent_field"

    serial = np.array(eigenvalues_ry(serial_results))
    gpu = np.array(eigenvalues_ry(gpu_results))
    assert gpu.shape == serial.shape
    difference = np.abs(gpu - serial)
    occupied = np.max(difference[..., :NUMBER_OF_OCCUPIED_BANDS])
    empty = np.max(difference[..., NUMBER_OF_OCCUPIED_BANDS:])
    assert occupied < occupied_tolerance, (
        f"GPU NSCF occupied eigenvalues differ from serial by {occupied:.3e} Ry"
    )
    assert empty < EMPTY_EIGENVALUE_TOLERANCE_RY, (
        f"GPU NSCF empty eigenvalues differ from serial by {empty:.3e} Ry"
    )
    np.testing.assert_allclose(
        occupations(gpu_results), occupations(serial_results), rtol=0, atol=1e-6
    )
