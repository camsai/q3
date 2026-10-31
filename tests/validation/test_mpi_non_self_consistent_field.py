"""
A non-self-consistent run on `q3_mpi`, after a self-consistent one on the same ranks.

A workflow chains engine runs -- a band structure is an SCF and then a
non-self-consistent pass along the path, a DOS the same over a dense mesh --
and the execution policy it was launched with has to reach every one of them.
Until this, `q3_mpi` refused anything but the SCF, so the NSCF of every such
workflow ran on one process however many the SCF had.

The pin: an SCF and an NSCF both on four single-rank pools give the
eigenvalues, occupations, weights, k-points and orbital projections a serial
SCF and NSCF give, in the order the input listed the k-points. Five k-points
over four pools deal out 2, 1, 1, 1, so a pool holds non-adjacent entries of
the list and a missing gather, or one into the wrong slots, shows.

Ranks sharing a pool split the density grid, which the whole potential the
NSCF loads cannot be laid onto, so that topology refuses -- with a message
saying how to run it instead.
"""

import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
from q3.io.results.classes import Results

from helpers import (
    EXAMPLES_DIRECTORY,
    apply_config_overlay,
    eigenvalues_ry,
    kpoint_weights,
    occupations,
)

pytestmark = pytest.mark.skipif(
    shutil.which("mpirun") is None,
    reason="mpirun not found — MPI tests require an MPI installation",
)

SILICON_INPUT = EXAMPLES_DIRECTORY / "total_energy--semiconductor--silicon" / "nc.yml"
PREFIX = "si_nc"

NUMBER_OF_RANKS = 4

SCF_OVERLAY = {
    "method": {"scf": {"kgrid": {"dimensions": [4, 4, 4], "shifts": [1, 1, 1]}}},
    "control": {"use_deterministic_fft_planning": True},
}

NSCF_KPOINTS = [
    [0.0, 0.0, 0.0, 1.0],
    [0.5, 0.0, 0.5, 1.0],
    [0.5, 0.5, 0.5, 1.0],
    [0.1, 0.2, 0.3, 1.0],
    [0.25, 0.0, 0.25, 1.0],
]
"""Gamma, X, L, a point of no symmetry and a point halfway to X, none on the SCF grid."""

NUMBER_OF_OCCUPIED_BANDS = 4

OCCUPIED_EIGENVALUE_TOLERANCE_RY = 1.0e-7
"""
How far the pooled NSCF's occupied eigenvalues may sit from the serial one's (Ry).

The same bound test_mpi_checkpoint.py sets for an NSCF restored from a pooled
checkpoint, for the same reason: the two SCFs add the same k-point
contributions in a different order, which can move where they stop by a
fraction of an iteration. Measured: 6e-9. A k-point's eigenvalues landing in
another's slot would move them by tenths of a Rydberg.
"""

EMPTY_EIGENVALUE_TOLERANCE_RY = 1.0e-4
"""
The same for the empty bands, which the Davidson solver refines only until a
step moves them by less than math::tolerances::empty_band_tolerance_floor
(1e-5 Ry). Where it stops depends on round-off in the Hamiltonian, which the
pools' reductions change at the 1e-15 level, so an empty band is pinned only
to a few times that floor -- measured 2e-5 Ry at X -- however the k-points are
distributed. Ten times the floor still catches a misplaced k-point.
"""

PROJECTION_TOLERANCE = 1.0e-4
"""
How far a pooled occupied-band projection |<phi|psi>|^2 may sit from the serial one.

Measured: 5e-6. A projection is first order in the eigenvector, which the
solver pins less tightly than the eigenvalue it converges on; another
k-point's projections in this one's slot would differ by tenths.

Projections are squared moduli, so the unitary freedom of degenerate bands at
Gamma, X and L leaves the per-band values free to redistribute within a
degenerate set. The comparison is over the sum of each degenerate set, which
that freedom leaves fixed, and over the occupied bands, which the solver
converges tightly (see EMPTY_EIGENVALUE_TOLERANCE_RY for the empty ones).
"""

DEGENERACY_TOLERANCE_RY = 1.0e-5


@pytest.fixture(scope="module")
def serial_q3_bin(pytestconfig, q3_bin):
    """
    A serial `q3`, from a build tree of its own, for the reference runs.

    A build tree compiles one q3, and in an MPI tree that `q3` is q3_mpi under
    its plain name (#316), so the serial one comes from --q3-serial-bin or the
    default serial tree, and is asked what it is, as test_mpi_checkpoint.py does.
    """
    serial_binary = Path(pytestconfig.getoption("--q3-serial-bin") or "./build/q3").resolve()
    if not serial_binary.is_file():
        pytest.skip(
            f"serial q3 not found at {serial_binary} (configure a tree without Q3_WITH_MPI, "
            "build its q3 target, and pass --q3-serial-bin)"
        )
    identity = subprocess.run(
        [str(serial_binary), "--version"], capture_output=True, text=True, timeout=60
    )
    if serial_binary == Path(q3_bin).resolve() or identity.stdout.startswith("q3_mpi"):
        pytest.skip(f"{serial_binary} is the MPI binary, not a serial q3; pass --q3-serial-bin")
    return str(serial_binary)


def _run(command, working_directory):
    return subprocess.run(
        command, cwd=str(working_directory), capture_output=True, text=True, timeout=600
    )


def _run_ok(command, working_directory, description):
    result = _run(command, working_directory)
    assert result.returncode == 0, (
        f"{description} failed (exit {result.returncode})\n"
        f"stdout: {result.stdout[-2000:]}\nstderr: {result.stderr[-2000:]}"
    )
    return result


def _nscf_input(working_directory):
    overlay = {
        **SCF_OVERLAY,
        "control": {
            **SCF_OVERLAY["control"],
            "calculation": "nscf",
            "kpoints_list": NSCF_KPOINTS,
            "compute_projections": True,
        },
    }
    return apply_config_overlay(SILICON_INPUT, overlay, working_directory)


def _scf_then_nscf(command, working_directory, description):
    """Run the SCF and then the NSCF with the same `command`, and return the NSCF's results."""
    scf_input = apply_config_overlay(SILICON_INPUT, SCF_OVERLAY, working_directory)
    _run_ok([*command, str(scf_input)], working_directory, f"{description} SCF")
    nscf_input = _nscf_input(working_directory)
    _run_ok([*command, str(nscf_input)], working_directory, f"{description} NSCF")
    results_path = working_directory / "output" / f"{PREFIX}_results.json"
    return Results.read(results_path)


@pytest.fixture(scope="module")
def serial_results(serial_q3_bin, tmp_path_factory):
    return _scf_then_nscf([serial_q3_bin], tmp_path_factory.mktemp("serial"), "serial")


@pytest.fixture(scope="module")
def pooled_results(q3_bin, tmp_path_factory):
    command = ["mpirun", "-np", str(NUMBER_OF_RANKS), "--oversubscribe", q3_bin]
    return _scf_then_nscf(
        [*command, "--pools", str(NUMBER_OF_RANKS)], tmp_path_factory.mktemp("pooled"), "pooled"
    )


def test_pooled_nscf_reports_every_listed_kpoint(serial_results, pooled_results):
    # Until the MPI binary handed every flavor its policy, it refused this input:
    # before that refusal it ran an SCF and reported it under the NSCF's name.
    assert pooled_results.runtime_log.flavor == "non_self_consistent_field"
    assert pooled_results.runtime_log.execution.number_of_kpoint_pools == NUMBER_OF_RANKS
    assert len(eigenvalues_ry(pooled_results)) == len(NSCF_KPOINTS)
    np.testing.assert_allclose(
        kpoint_weights(pooled_results), kpoint_weights(serial_results), rtol=0, atol=1e-12
    )


def test_pooled_nscf_eigenvalues_match_serial(serial_results, pooled_results):
    serial = np.array(eigenvalues_ry(serial_results))
    pooled = np.array(eigenvalues_ry(pooled_results))
    assert pooled.shape == serial.shape
    difference = np.abs(pooled - serial)
    occupied = np.max(difference[..., :NUMBER_OF_OCCUPIED_BANDS])
    empty = np.max(difference[..., NUMBER_OF_OCCUPIED_BANDS:])
    assert occupied < OCCUPIED_EIGENVALUE_TOLERANCE_RY, (
        f"pooled NSCF occupied eigenvalues differ from serial by {occupied:.3e} Ry"
    )
    assert empty < EMPTY_EIGENVALUE_TOLERANCE_RY, (
        f"pooled NSCF empty eigenvalues differ from serial by {empty:.3e} Ry"
    )
    np.testing.assert_allclose(
        occupations(pooled_results), occupations(serial_results), rtol=0, atol=1e-6
    )


def _degenerate_set_sums(projections, eigenvalues):
    """Per k-point, the projections summed over each set of degenerate bands."""
    sums = []
    for kpoint_projections, kpoint_eigenvalues in zip(projections, eigenvalues, strict=True):
        kpoint_projections = np.asarray(kpoint_projections)
        start = 0
        for band in range(1, len(kpoint_eigenvalues) + 1):
            if (
                band == len(kpoint_eigenvalues)
                or kpoint_eigenvalues[band] - kpoint_eigenvalues[band - 1] > DEGENERACY_TOLERANCE_RY
            ):
                sums.append(kpoint_projections[start:band].sum(axis=0))
                start = band
    return np.array(sums)


def test_pooled_nscf_projections_match_serial(serial_results, pooled_results):
    assert (
        pooled_results.properties.orbital_projections.labels
        == serial_results.properties.orbital_projections.labels
    )
    occupied = slice(0, NUMBER_OF_OCCUPIED_BANDS)
    serial_eigenvalues = np.array(eigenvalues_ry(serial_results))[:, 0, occupied]
    serial = _degenerate_set_sums(
        [kpoint[occupied] for kpoint in serial_results.properties.orbital_projections.values],
        serial_eigenvalues,
    )
    pooled = _degenerate_set_sums(
        [kpoint[occupied] for kpoint in pooled_results.properties.orbital_projections.values],
        serial_eigenvalues,
    )
    assert pooled.shape == serial.shape
    difference = np.max(np.abs(pooled - serial))
    assert difference < PROJECTION_TOLERANCE, (
        f"pooled NSCF projections differ from serial by {difference:.3e}"
    )


def test_nscf_on_ranks_sharing_a_pool_refuses_and_says_how_to_run_it(q3_bin, tmp_path):
    """Two ranks in one pool split the grid; the run must say so, not return wrong bands."""
    serial_scf_input = apply_config_overlay(SILICON_INPUT, SCF_OVERLAY, tmp_path)
    _run_ok(
        ["mpirun", "-np", "2", "--oversubscribe", q3_bin, "--pools", "2", str(serial_scf_input)],
        tmp_path,
        "SCF on two pools",
    )
    nscf_input = _nscf_input(tmp_path)
    result = _run(
        ["mpirun", "-np", "2", "--oversubscribe", q3_bin, "--pools", "1", str(nscf_input)],
        tmp_path,
    )
    assert result.returncode != 0
    output = result.stdout + result.stderr
    assert "one rank per pool" in output, output[-2000:]
