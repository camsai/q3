"""
Checkpoint output from a multi-rank `q3_mpi` run.

When every rank owns its own k-point pool -- the layout the automatic MPI
topology chooses -- the run writes the same `<prefix>.save/` a single-process
run does. Each pool writes the `wfc<N>.dat` files of its own k-points under
their global index. The root writes `charge-density.dat` and
`effective-potential.q3cp`: the density and the potential are reduced across
pools, so the root holds them whole, and the eigenvalues are gathered into
global order first. Before this every multi-rank run refused, so the SCF, the
one step that could use many ranks, had to run on one for a serial
non-self-consistent run to restore it.

The pin is that use: a serial NSCF restoring the pooled checkpoint must give
the eigenvalues it gives from a serial SCF's checkpoint. NSCF reads only
`effective-potential.q3cp`, so the file-level checks cover the rest, and also
say which part broke when the NSCF comparison fails.

Ranks sharing a pool split each k-point's coefficients and the density grid,
and nothing gathers them, so that layout still refuses. That refusal is pinned
in test_mpi_mean_field_export.py.
"""

import shutil
import struct
import subprocess
from pathlib import Path

import numpy as np
import pytest
from binary_readers import read_charge_density, read_wavefunction
from q3.io.results.classes import Results

from helpers import EXAMPLES_DIRECTORY, apply_config_overlay, eigenvalues_ry

pytestmark = pytest.mark.skipif(
    shutil.which("mpirun") is None,
    reason="mpirun not found — MPI tests require an MPI installation",
)

SILICON_INPUT = EXAMPLES_DIRECTORY / "total_energy--semiconductor--silicon" / "nc.yml"
PREFIX = "si_nc"

NUMBER_OF_RANKS = 4
"""
Four single-rank pools over ten k-points deal them out 3, 3, 2, 2, so a pool
writes non-adjacent global indices and the pools do not hold equal shares.
"""

EXPECTED_NUMBER_OF_KPOINTS = 10
"""The shifted 4x4x4 grid reduces to ten irreducible k-points in silicon."""

NUMBER_OF_OCCUPIED_BANDS = 4

SCF_OVERLAY = {
    "method": {"scf": {"kgrid": {"dimensions": [4, 4, 4], "shifts": [1, 1, 1]}}},
    "control": {"use_deterministic_fft_planning": True},
}

NSCF_KPOINTS = [
    [0.0, 0.0, 0.0, 1.0],
    [0.5, 0.0, 0.5, 1.0],
    [0.5, 0.5, 0.5, 1.0],
    [0.1, 0.2, 0.3, 1.0],
]
"""Gamma, X, L and a point of no symmetry, none of them on the SCF grid."""

EIGENVALUE_TOLERANCE_RY = 1.0e-7
"""
How far a pooled run's eigenvalues may sit from a serial run's (Ry).

The two SCFs add the same k-point contributions in a different order, which
can move where they stop by a fraction of an iteration at the 1e-8 Ry
threshold. Measured on macOS: the stored eigenvalues agree to 9e-15 Ry and the
NSCF eigenvalues to 1.4e-10 Ry. The bound leaves room for another platform's
summation order. A root writing its own pool's density or potential instead of
the reduced one would move them by tenths of a Rydberg.
"""

DENSITY_TOLERANCE = 1.0e-8
"""
Largest difference allowed between the two runs' rho(G), relative to rho(G = 0).
Measured: 1e-16.
"""

OCCUPIED_SUBSPACE_TOLERANCE = 1.0e-6
"""
How far the overlap of the two runs' occupied subspaces at one k-point may fall
short of complete (measured: 1e-14). The coefficients are fixed only up to a
unitary rotation of the occupied bands, which this measure ignores. A file
holding a different k-point's bands already fails the Miller-index check, since
no two k-points here share a plane-wave set.
"""


@pytest.fixture(scope="module")
def serial_q3_bin(pytestconfig, q3_bin):
    """
    A serial `q3`, from a build tree of its own. Every NSCF here runs on it, so
    the comparison isolates the checkpoint from the topology of the NSCF.

    `q3_bin` is q3_mpi here: conftest's `require_mpi_binary` skips this module
    otherwise, since four serial processes under mpirun would each write the
    whole checkpoint and pass every check below without exercising a pool.

    The serial binary no longer sits beside it: a build tree compiles one q3,
    and in an MPI tree that `q3` is q3_mpi under its plain name (#316). So it
    comes from --q3-serial-bin, or the default serial tree, and is asked what
    it is for the same reason `require_mpi_binary` asks.
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


def _run(command, working_directory, description):
    result = subprocess.run(
        command, cwd=str(working_directory), capture_output=True, text=True, timeout=600
    )
    assert result.returncode == 0, (
        f"{description} failed (exit {result.returncode})\n"
        f"stdout: {result.stdout[-2000:]}\nstderr: {result.stderr[-2000:]}"
    )
    return result


def _scf_then_nscf(launcher, serial_q3_bin, working_directory, description):
    """
    Run the SCF with `launcher`, then a serial NSCF from its checkpoint.

    Returns the save directory, the SCF results and the NSCF eigenvalues. The
    checkpoint is copied aside before the NSCF runs, since NSCF writes into the
    same output directory.
    """
    scf_input = apply_config_overlay(SILICON_INPUT, SCF_OVERLAY, working_directory)
    scf_output = _run(launcher + [str(scf_input)], working_directory, f"{description} SCF")
    output_directory = working_directory / "output"
    scf_results = Results.read(output_directory / f"{PREFIX}_results.json")

    save_directory = output_directory / f"{PREFIX}.save"
    assert save_directory.is_dir(), (
        f"{description}: no checkpoint directory was written\n{scf_output.stdout[-2000:]}"
    )
    saved_checkpoint = working_directory / "checkpoint"
    shutil.copytree(save_directory, saved_checkpoint)

    nscf_overlay = {
        **SCF_OVERLAY,
        "control": {
            **SCF_OVERLAY["control"],
            "calculation": "nscf",
            "kpoints_list": NSCF_KPOINTS,
        },
    }
    nscf_input = apply_config_overlay(SILICON_INPUT, nscf_overlay, working_directory)
    _run([serial_q3_bin, str(nscf_input)], working_directory, f"NSCF from the {description} SCF")
    nscf_results = Results.read(output_directory / f"{PREFIX}_results.json")
    nscf_eigenvalues = np.array(eigenvalues_ry(nscf_results))
    return saved_checkpoint, scf_results, nscf_eigenvalues


def _read_checkpoint_eigenvalues(path):
    """
    The header and eigenvalues of `effective-potential.q3cp`.

    The layout is the one `Q3PlanewaveKohnShamQuantumSystemCheckpoint` writes:
    magic, version, four ints (iteration, spin components, bands, k-point
    states), a double, two flag bytes, the density section, then one row of
    eigenvalues per k-point state, each prefixed by its length.
    """
    data = Path(path).read_bytes()
    magic, _version = struct.unpack_from("<II", data, 0)
    assert magic == 0x51334350, f"{path} is not a Q3 checkpoint"
    _iteration, number_of_spin_components, number_of_bands, number_of_kpoint_states = (
        struct.unpack_from("<4i", data, 8)
    )
    offset = 8 + 4 * 4 + 8 + 1 + 1
    (number_of_density_components,) = struct.unpack_from("<i", data, offset)
    offset += 4
    for _ in range(number_of_density_components):
        (number_of_grid_points,) = struct.unpack_from("<i", data, offset)
        offset += 4 + 8 * number_of_grid_points
    rows = []
    for _ in range(number_of_kpoint_states):
        (row_length,) = struct.unpack_from("<i", data, offset)
        offset += 4
        rows.append(np.frombuffer(data, dtype=np.float64, count=row_length, offset=offset))
        offset += 8 * row_length
    return {
        "number_of_spin_components": number_of_spin_components,
        "number_of_bands": number_of_bands,
        "number_of_kpoint_states": number_of_kpoint_states,
        "eigenvalues": np.array(rows),
    }


@pytest.fixture(scope="module")
def serial_run(serial_q3_bin, tmp_path_factory):
    working_directory = tmp_path_factory.mktemp("serial")
    return _scf_then_nscf([serial_q3_bin], serial_q3_bin, working_directory, "serial")


@pytest.fixture(scope="module")
def pooled_run(q3_bin, serial_q3_bin, tmp_path_factory):
    working_directory = tmp_path_factory.mktemp("pooled")
    launcher = ["mpirun", "-np", str(NUMBER_OF_RANKS), q3_bin]
    return _scf_then_nscf(launcher, serial_q3_bin, working_directory, "pooled")


def test_pooled_run_uses_one_rank_per_pool(pooled_run):
    """The layout under test: the automatic topology, every rank its own pool."""
    _, scf_results, _ = pooled_run
    execution = scf_results.runtime_log.execution
    assert execution.number_of_processes == NUMBER_OF_RANKS
    assert execution.number_of_kpoint_pools == NUMBER_OF_RANKS
    assert execution.band_group_size == 1


def test_nscf_from_pooled_checkpoint_matches_nscf_from_serial_checkpoint(serial_run, pooled_run):
    """The acceptance test: a serial NSCF cannot tell the two checkpoints apart."""
    _, _, serial_eigenvalues = serial_run
    _, _, pooled_eigenvalues = pooled_run
    assert pooled_eigenvalues.shape == serial_eigenvalues.shape
    difference = np.max(np.abs(pooled_eigenvalues - serial_eigenvalues))
    assert difference < EIGENVALUE_TOLERANCE_RY, (
        f"NSCF eigenvalues differ by {difference:.3e} Ry between checkpoints"
    )


def test_every_kpoint_has_its_wavefunction_file(serial_run, pooled_run):
    """All ten files, from four pools, under the names a serial run uses."""
    serial_checkpoint, _, _ = serial_run
    pooled_checkpoint, _, _ = pooled_run
    expected = {f"wfc{index}.dat" for index in range(1, EXPECTED_NUMBER_OF_KPOINTS + 1)}
    assert {path.name for path in serial_checkpoint.glob("wfc*.dat")} == expected
    assert {path.name for path in pooled_checkpoint.glob("wfc*.dat")} == expected


@pytest.mark.parametrize("kpoint_number", range(1, EXPECTED_NUMBER_OF_KPOINTS + 1))
def test_wavefunction_file_holds_its_global_kpoint(serial_run, pooled_run, kpoint_number):
    """
    wfc<N>.dat from a pool is the serial run's wfc<N>.dat: same k-point, same
    plane waves, same occupied subspace.
    """
    serial_checkpoint, _, _ = serial_run
    pooled_checkpoint, _, _ = pooled_run
    serial = read_wavefunction(serial_checkpoint / f"wfc{kpoint_number}.dat")
    pooled = read_wavefunction(pooled_checkpoint / f"wfc{kpoint_number}.dat")

    assert pooled["kpoint_index"] == kpoint_number
    np.testing.assert_allclose(
        pooled["kpoint_coordinates"], serial["kpoint_coordinates"], rtol=0, atol=1e-12
    )
    assert pooled["number_of_bands"] == serial["number_of_bands"]
    np.testing.assert_array_equal(pooled["miller_indices"], serial["miller_indices"])

    serial_occupied = serial["coefficients"][:NUMBER_OF_OCCUPIED_BANDS]
    pooled_occupied = pooled["coefficients"][:NUMBER_OF_OCCUPIED_BANDS]
    overlap = serial_occupied.conj() @ pooled_occupied.T
    shortfall = NUMBER_OF_OCCUPIED_BANDS - np.sum(np.abs(overlap) ** 2)
    assert abs(shortfall) < OCCUPIED_SUBSPACE_TOLERANCE, (
        f"wfc{kpoint_number}.dat: occupied subspaces overlap {shortfall:.3e} short of complete"
    )


def test_charge_density_is_the_reduced_one(serial_run, pooled_run):
    """The root writes the density summed over every pool, not its own share."""
    serial_checkpoint, _, _ = serial_run
    pooled_checkpoint, _, _ = pooled_run
    serial = read_charge_density(serial_checkpoint / "charge-density.dat")
    pooled = read_charge_density(pooled_checkpoint / "charge-density.dat")
    np.testing.assert_array_equal(pooled["miller_indices"], serial["miller_indices"])
    scale = abs(serial["density_g_space"][0])
    difference = np.max(np.abs(pooled["density_g_space"] - serial["density_g_space"])) / scale
    assert difference < DENSITY_TOLERANCE, f"rho(G) differs by {difference:.3e} of rho(0)"


def test_checkpoint_eigenvalues_cover_every_kpoint_in_global_order(serial_run, pooled_run):
    """effective-potential.q3cp carries all ten k-points, ordered as serially."""
    serial_checkpoint, _, _ = serial_run
    pooled_checkpoint, pooled_results, _ = pooled_run
    serial = _read_checkpoint_eigenvalues(serial_checkpoint / "effective-potential.q3cp")
    pooled = _read_checkpoint_eigenvalues(pooled_checkpoint / "effective-potential.q3cp")

    assert pooled["number_of_kpoint_states"] == EXPECTED_NUMBER_OF_KPOINTS
    assert pooled["eigenvalues"].shape == serial["eigenvalues"].shape
    difference = np.max(np.abs(pooled["eigenvalues"] - serial["eigenvalues"]))
    assert difference < EIGENVALUE_TOLERANCE_RY, f"stored eigenvalues differ by {difference:.3e} Ry"
    # And they are the ones the same run reported in results.json.
    reported = np.array(eigenvalues_ry(pooled_results))[:, 0, :]
    np.testing.assert_allclose(pooled["eigenvalues"], reported, rtol=0, atol=1e-12)
