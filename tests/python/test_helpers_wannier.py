"""tests/python/test_helpers_wannier.py

Unit tests for the independent Wannier90-format reader and metrics
(tests/helpers_wannier.py), against synthetic files built by small,
self-contained writers defined in this module.

Independence within this test file matters: the writers below emit plain
formatted text and never import the reader's dataclasses, parsing helpers, or
private functions. If a fixture were produced by reusing the reader's own
notion of the layout, a bug shared between "how the fixture is written" and
"how the reader parses it" would cancel out and this suite would pass
regardless. These tests exercise the READER and the METRICS only; they say
nothing about Q3's future C++ writers for these formats.

The fault-injection cases are the point of the file: the reader's job in the
parity harness is to fail loudly on a malformed file, because a silently
mis-parsed reference would turn an oracle comparison into a comparison of two
different bugs.
"""

from __future__ import annotations

import numpy as np
import pytest
from helpers_wannier import (
    WannierFormatError,
    band_distance,
    compute_b_vectors,
    gauge_invariant_spread,
    projectabilities,
    read_eigenvalue_file,
    read_neighbor_list_file,
    read_overlap_file,
    read_projection_file,
    solve_finite_difference_weights,
)

# ── Independent synthetic writers ─────────────────────────────────────


def write_overlap_file(path, overlaps, neighbor_kpoint, periodic_image_shift):
    """Emit a .mmn from ``overlaps[k][n]`` matrices, first index m fastest."""
    number_of_kpoints = len(overlaps)
    number_of_neighbors = len(overlaps[0])
    number_of_bands = len(overlaps[0][0])
    out = [
        "written by the test suite",
        f"{number_of_bands} {number_of_kpoints} {number_of_neighbors}",
    ]
    for kpoint_index in range(number_of_kpoints):
        for neighbor_index in range(number_of_neighbors):
            shift = periodic_image_shift[kpoint_index][neighbor_index]
            out.append(
                f"{kpoint_index + 1} {neighbor_kpoint[kpoint_index][neighbor_index] + 1} "
                f"{shift[0]} {shift[1]} {shift[2]}"
            )
            block = overlaps[kpoint_index][neighbor_index]
            # m fastest: inner loop over the ROW index.
            for column in range(number_of_bands):
                for row in range(number_of_bands):
                    value = block[row][column]
                    out.append(f"{value.real:.12f} {value.imag:.12f}")
    path.write_text("\n".join(out) + "\n")
    return path


def write_projection_file(path, projections):
    """Emit a .amn from ``projections[k][m][n]``."""
    number_of_kpoints = len(projections)
    number_of_bands = len(projections[0])
    number_of_wannier = len(projections[0][0])
    out = [
        "written by the test suite",
        f"{number_of_bands} {number_of_kpoints} {number_of_wannier}",
    ]
    for kpoint_index in range(number_of_kpoints):
        for wannier_index in range(number_of_wannier):
            for band_index in range(number_of_bands):
                value = projections[kpoint_index][band_index][wannier_index]
                out.append(
                    f"{band_index + 1} {wannier_index + 1} {kpoint_index + 1} "
                    f"{value.real:.12f} {value.imag:.12f}"
                )
    path.write_text("\n".join(out) + "\n")
    return path


def write_eigenvalue_file(path, eigenvalues):
    """Emit a .eig from ``eigenvalues[k][band]``."""
    out = []
    for kpoint_index, row in enumerate(eigenvalues):
        for band_index, value in enumerate(row):
            out.append(f"{band_index + 1} {kpoint_index + 1} {value:.12f}")
    path.write_text("\n".join(out) + "\n")
    return path


def write_neighbor_list_file(
    path, real_lattice, reciprocal_lattice, kpoints, neighbor_kpoint, periodic_image_shift
):
    """Emit the blocks of a .nnkp that the harness reads."""
    number_of_kpoints = len(kpoints)
    number_of_neighbors = len(neighbor_kpoint[0])
    # 12 decimals, not the 8 a real wannier90 .nnkp carries: this fixture
    # feeds an analytic weight check at rtol 1e-10, and 8 decimals would put
    # ~2e-9 of formatting error into |b| -- which would make that test measure
    # the writer's precision rather than the solver's correctness.
    out = ["File written by the test suite", "", "calc_only_A  :  F", ""]
    out.append("begin real_lattice")
    for row in real_lattice:
        out.append(f"  {row[0]:.12f} {row[1]:.12f} {row[2]:.12f}")
    out.append("end real_lattice")
    out.append("begin recip_lattice")
    for row in reciprocal_lattice:
        out.append(f"  {row[0]:.12f} {row[1]:.12f} {row[2]:.12f}")
    out.append("end recip_lattice")
    out.append("begin kpoints")
    out.append(f"  {number_of_kpoints}")
    for kpoint in kpoints:
        out.append(f"  {kpoint[0]:.12f} {kpoint[1]:.12f} {kpoint[2]:.12f}")
    out.append("end kpoints")
    out.append("begin nnkpts")
    out.append(f"  {number_of_neighbors}")
    for kpoint_index in range(number_of_kpoints):
        for neighbor_index in range(number_of_neighbors):
            shift = periodic_image_shift[kpoint_index][neighbor_index]
            out.append(
                f"  {kpoint_index + 1} {neighbor_kpoint[kpoint_index][neighbor_index] + 1} "
                f"{shift[0]} {shift[1]} {shift[2]}"
            )
    out.append("end nnkpts")
    path.write_text("\n".join(out) + "\n")
    return path


# ── Fixtures ──────────────────────────────────────────────────────────


def simple_cubic_neighbor_data(mesh=2, lattice_parameter=4.0):
    """A 2x2x2 simple-cubic mesh with its 6 nearest-neighbour b-vectors.

    Returns everything a .nnkp needs. The reciprocal lattice is diagonal with
    entries 2*pi/a, so the b-vectors are the six +-(2*pi/(a*mesh)) axis
    vectors and the completeness weight is analytic (see the weight test).
    """
    reciprocal = np.eye(3) * (2.0 * np.pi / lattice_parameter)
    real = np.eye(3) * lattice_parameter
    coordinates = [i / mesh for i in range(mesh)]
    kpoints = [[x, y, z] for x in coordinates for y in coordinates for z in coordinates]
    number_of_kpoints = len(kpoints)

    steps = [
        (1, 0, 0),
        (-1, 0, 0),
        (0, 1, 0),
        (0, -1, 0),
        (0, 0, 1),
        (0, 0, -1),
    ]

    def index_of(triple):
        return (triple[0] * mesh + triple[1]) * mesh + triple[2]

    neighbor_kpoint = []
    periodic_image_shift = []
    for kpoint_index in range(number_of_kpoints):
        grid = (
            kpoint_index // (mesh * mesh),
            (kpoint_index // mesh) % mesh,
            kpoint_index % mesh,
        )
        neighbors = []
        shifts = []
        for step in steps:
            raw = [grid[axis] + step[axis] for axis in range(3)]
            wrapped = [value % mesh for value in raw]
            neighbors.append(index_of(tuple(wrapped)))
            shifts.append([(raw[axis] - wrapped[axis]) // mesh for axis in range(3)])
        neighbor_kpoint.append(neighbors)
        periodic_image_shift.append(shifts)
    return real, reciprocal, kpoints, neighbor_kpoint, periodic_image_shift


# ── seedname.mmn ──────────────────────────────────────────────────────


def test_overlap_round_trip_preserves_m_fastest_ordering(tmp_path):
    """A deliberately asymmetric block: C-order would transpose it silently.

    This is the single most important ordering check in the module -- an
    overlap matrix is close to Hermitian for real data, so a transposition
    survives eyeballing. The fixture below is chosen so M[0,1] != M[1,0] by a
    wide margin.
    """
    block = np.array([[1.0 + 0.5j, 2.0 - 0.25j], [30.0 + 0.0j, 4.0 + 0.125j]])
    path = write_overlap_file(tmp_path / "seed.mmn", [[block]], [[0]], [[[0, 0, 0]]])
    parsed = read_overlap_file(path)

    assert parsed.number_of_bands == 2
    assert parsed.number_of_kpoints == 1
    assert parsed.number_of_neighbors == 1
    np.testing.assert_allclose(parsed.overlaps[0, 0], block, atol=1e-12)
    assert parsed.overlaps[0, 0][0, 1] == pytest.approx(2.0 - 0.25j)
    assert parsed.overlaps[0, 0][1, 0] == pytest.approx(30.0 + 0.0j)


def test_overlap_records_neighbour_and_shift(tmp_path):
    block = np.eye(2, dtype=complex)
    path = write_overlap_file(
        tmp_path / "seed.mmn",
        [[block, block]],
        [[0, 0]],
        [[[1, 0, 0], [0, -1, 2]]],
    )
    parsed = read_overlap_file(path)
    np.testing.assert_array_equal(parsed.periodic_image_shift[0, 0], [1, 0, 0])
    np.testing.assert_array_equal(parsed.periodic_image_shift[0, 1], [0, -1, 2])


def test_overlap_truncated_file_is_an_error(tmp_path):
    block = np.eye(2, dtype=complex)
    path = write_overlap_file(tmp_path / "seed.mmn", [[block]], [[0]], [[[0, 0, 0]]])
    lines = path.read_text().splitlines()
    path.write_text("\n".join(lines[:-1]) + "\n")
    with pytest.raises(WannierFormatError, match="significant lines"):
        read_overlap_file(path)


def test_overlap_header_count_mismatch_is_an_error(tmp_path):
    block = np.eye(2, dtype=complex)
    path = write_overlap_file(tmp_path / "seed.mmn", [[block]], [[0]], [[[0, 0, 0]]])
    lines = path.read_text().splitlines()
    lines[1] = "2 2 1"  # claims 2 k-points; only one block present
    path.write_text("\n".join(lines) + "\n")
    with pytest.raises(WannierFormatError, match="significant lines"):
        read_overlap_file(path)


def test_overlap_blocks_out_of_order_are_an_error(tmp_path):
    """Blocks that do not sweep k in order would mis-associate every matrix."""
    block = np.eye(2, dtype=complex)
    path = write_overlap_file(
        tmp_path / "seed.mmn", [[block], [block]], [[0], [1]], [[[0, 0, 0]], [[0, 0, 0]]]
    )
    lines = path.read_text().splitlines()
    lines[2] = "2 1 0 0 0"  # first block claims to be k-point 2
    path.write_text("\n".join(lines) + "\n")
    with pytest.raises(WannierFormatError, match="sweep k-points in order"):
        read_overlap_file(path)


def test_overlap_neighbour_index_out_of_range_is_an_error(tmp_path):
    block = np.eye(2, dtype=complex)
    path = write_overlap_file(tmp_path / "seed.mmn", [[block]], [[0]], [[[0, 0, 0]]])
    lines = path.read_text().splitlines()
    lines[2] = "1 7 0 0 0"  # only 1 k-point exists
    path.write_text("\n".join(lines) + "\n")
    with pytest.raises(WannierFormatError, match="neighbour k-point index 7"):
        read_overlap_file(path)


def test_overlap_non_numeric_element_is_an_error(tmp_path):
    block = np.eye(2, dtype=complex)
    path = write_overlap_file(tmp_path / "seed.mmn", [[block]], [[0]], [[[0, 0, 0]]])
    lines = path.read_text().splitlines()
    lines[3] = "nan_but_not_a_float 0.0"
    path.write_text("\n".join(lines) + "\n")
    with pytest.raises(WannierFormatError, match="two real numbers"):
        read_overlap_file(path)


# ── seedname.amn ──────────────────────────────────────────────────────


def test_projection_round_trip(tmp_path):
    projections = np.array(
        [
            [[1.0 + 0.0j, 0.5 - 0.5j], [0.25 + 0.75j, 2.0 + 0.0j], [0.0 + 1.0j, 3.0 + 0.0j]],
            [[4.0 + 0.0j, 0.0 + 0.0j], [0.0 + 0.0j, 5.0 + 0.0j], [6.0 + 0.0j, 0.0 + 0.0j]],
        ]
    )
    path = write_projection_file(tmp_path / "seed.amn", projections)
    parsed = read_projection_file(path)
    assert parsed.number_of_bands == 3
    assert parsed.number_of_kpoints == 2
    assert parsed.number_of_wannier_functions == 2
    np.testing.assert_allclose(parsed.projections, projections, atol=1e-12)


def test_projection_duplicate_entry_is_an_error(tmp_path):
    projections = np.ones((1, 2, 2), dtype=complex)
    path = write_projection_file(tmp_path / "seed.amn", projections)
    lines = path.read_text().splitlines()
    lines[-1] = lines[-2]  # repeat a triple, leaving another slot unwritten
    path.write_text("\n".join(lines) + "\n")
    with pytest.raises(WannierFormatError, match="duplicate entry"):
        read_projection_file(path)


def test_projection_count_mismatch_is_an_error(tmp_path):
    projections = np.ones((1, 2, 2), dtype=complex)
    path = write_projection_file(tmp_path / "seed.amn", projections)
    lines = path.read_text().splitlines()
    path.write_text("\n".join(lines[:-1]) + "\n")
    with pytest.raises(WannierFormatError, match="element lines"):
        read_projection_file(path)


def test_projection_band_index_out_of_range_is_an_error(tmp_path):
    projections = np.ones((1, 2, 2), dtype=complex)
    path = write_projection_file(tmp_path / "seed.amn", projections)
    lines = path.read_text().splitlines()
    lines[2] = "9 1 1 1.0 0.0"
    path.write_text("\n".join(lines) + "\n")
    with pytest.raises(WannierFormatError, match="band index 9"):
        read_projection_file(path)


# ── seedname.eig ──────────────────────────────────────────────────────


def test_eigenvalue_round_trip(tmp_path):
    eigenvalues = np.array([[-6.4, 1.5, 19.4], [-5.0, 2.5, 18.0]])
    path = write_eigenvalue_file(tmp_path / "seed.eig", eigenvalues)
    parsed = read_eigenvalue_file(path)
    assert parsed.number_of_bands == 3
    assert parsed.number_of_kpoints == 2
    np.testing.assert_allclose(parsed.eigenvalues, eigenvalues, atol=1e-12)


def test_eigenvalue_missing_entry_is_an_error(tmp_path):
    """A hole must be an error, not a silent zero -- see the module docstring."""
    eigenvalues = np.array([[-6.4, 1.5, 19.4], [-5.0, 2.5, 18.0]])
    path = write_eigenvalue_file(tmp_path / "seed.eig", eigenvalues)
    lines = path.read_text().splitlines()
    del lines[1]  # drop band 2 of k-point 1; max indices still imply 2x3
    path.write_text("\n".join(lines) + "\n")
    with pytest.raises(WannierFormatError, match="are absent"):
        read_eigenvalue_file(path)


def test_eigenvalue_duplicate_entry_is_an_error(tmp_path):
    eigenvalues = np.array([[-6.4, 1.5], [-5.0, 2.5]])
    path = write_eigenvalue_file(tmp_path / "seed.eig", eigenvalues)
    lines = path.read_text().splitlines()
    lines[1] = lines[0]
    path.write_text("\n".join(lines) + "\n")
    with pytest.raises(WannierFormatError, match="duplicate entry"):
        read_eigenvalue_file(path)


def test_eigenvalue_zero_index_is_an_error(tmp_path):
    """Indices are 1-based; a 0 means a writer forgot the Fortran origin."""
    eigenvalues = np.array([[-6.4, 1.5]])
    path = write_eigenvalue_file(tmp_path / "seed.eig", eigenvalues)
    lines = path.read_text().splitlines()
    lines[0] = "0 1 -6.4"
    path.write_text("\n".join(lines) + "\n")
    with pytest.raises(WannierFormatError, match="1-based"):
        read_eigenvalue_file(path)


# ── seedname.nnkp ─────────────────────────────────────────────────────


def test_neighbor_list_round_trip(tmp_path):
    real, reciprocal, kpoints, neighbors, shifts = simple_cubic_neighbor_data()
    path = write_neighbor_list_file(
        tmp_path / "seed.nnkp", real, reciprocal, kpoints, neighbors, shifts
    )
    parsed = read_neighbor_list_file(path)
    assert parsed.number_of_neighbors == 6
    assert parsed.kpoints.shape == (8, 3)
    np.testing.assert_allclose(parsed.reciprocal_lattice, reciprocal, atol=1e-8)
    np.testing.assert_array_equal(parsed.neighbor_kpoint, np.array(neighbors))
    np.testing.assert_array_equal(parsed.periodic_image_shift, np.array(shifts))


def test_neighbor_list_unterminated_block_is_an_error(tmp_path):
    real, reciprocal, kpoints, neighbors, shifts = simple_cubic_neighbor_data()
    path = write_neighbor_list_file(
        tmp_path / "seed.nnkp", real, reciprocal, kpoints, neighbors, shifts
    )
    text = path.read_text().replace("end nnkpts", "")
    path.write_text(text)
    with pytest.raises(WannierFormatError, match="never closed"):
        read_neighbor_list_file(path)


def test_neighbor_list_row_count_mismatch_is_an_error(tmp_path):
    real, reciprocal, kpoints, neighbors, shifts = simple_cubic_neighbor_data()
    path = write_neighbor_list_file(
        tmp_path / "seed.nnkp", real, reciprocal, kpoints, neighbors, shifts
    )
    lines = path.read_text().splitlines()
    end_index = lines.index("end nnkpts")
    del lines[end_index - 1]
    path.write_text("\n".join(lines) + "\n")
    with pytest.raises(WannierFormatError, match="nnkpts.*but holds"):
        read_neighbor_list_file(path)


def test_neighbor_list_missing_block_is_an_error(tmp_path):
    real, reciprocal, kpoints, neighbors, shifts = simple_cubic_neighbor_data()
    path = write_neighbor_list_file(
        tmp_path / "seed.nnkp", real, reciprocal, kpoints, neighbors, shifts
    )
    text = path.read_text().replace("begin recip_lattice", "begin something_else")
    text = text.replace("end recip_lattice", "end something_else")
    path.write_text(text)
    with pytest.raises(WannierFormatError, match="no 'begin recip_lattice' block"):
        read_neighbor_list_file(path)


# ── b-vectors and weights ─────────────────────────────────────────────


def test_simple_cubic_weights_match_the_hand_derived_value(tmp_path):
    """Simple cubic, 6 neighbours: 2 w b^2 = 1 per axis, so w = 1/(2 b^2).

    Hand-derivable and therefore an honest check of the least-squares solve
    rather than a self-consistency one.
    """
    mesh, lattice_parameter = 2, 4.0
    real, reciprocal, kpoints, neighbors, shifts = simple_cubic_neighbor_data(
        mesh, lattice_parameter
    )
    path = write_neighbor_list_file(
        tmp_path / "seed.nnkp", real, reciprocal, kpoints, neighbors, shifts
    )
    parsed = read_neighbor_list_file(path)
    b_vectors = compute_b_vectors(parsed)

    magnitude = 2.0 * np.pi / (lattice_parameter * mesh)
    np.testing.assert_allclose(np.linalg.norm(b_vectors[0], axis=1), magnitude, rtol=1e-10)

    weights = solve_finite_difference_weights(b_vectors)
    np.testing.assert_allclose(weights, 1.0 / (2.0 * magnitude**2), rtol=1e-10)

    # And the completeness condition it was solved for actually holds.
    tensor = np.einsum("b,bi,bj->ij", weights, b_vectors[0], b_vectors[0])
    np.testing.assert_allclose(tensor, np.eye(3), atol=1e-10)


def test_incomplete_shell_is_rejected(tmp_path):
    """Two collinear b-vectors cannot span the tensor; that must be loud."""
    b_vectors = np.array([[[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]]])
    with pytest.raises(WannierFormatError, match="B1 completeness"):
        solve_finite_difference_weights(b_vectors)


# ── Metrics ───────────────────────────────────────────────────────────


def test_gauge_invariant_spread_vanishes_for_identity_overlaps():
    """If every M is the identity, the manifold is perfectly reproduced."""
    number_of_kpoints, number_of_neighbors, number_of_bands = 4, 6, 3
    overlaps = np.tile(
        np.eye(number_of_bands, dtype=complex),
        (number_of_kpoints, number_of_neighbors, 1, 1),
    )
    weights = np.full(number_of_neighbors, 0.25)
    assert gauge_invariant_spread(overlaps, weights, number_of_bands) == pytest.approx(0.0)


def test_gauge_invariant_spread_is_invariant_under_per_k_unitaries():
    """The property the whole cross-code comparison strategy rests on.

    Two codes producing different gauges must still agree on Omega_I. This
    applies an independent random unitary at every k-point -- which changes
    every M element -- and requires the spread to be unchanged.
    """
    rng = np.random.default_rng(20260825)
    number_of_kpoints, number_of_neighbors, number_of_bands = 3, 4, 3
    weights = rng.uniform(0.2, 0.8, size=number_of_neighbors)

    # Neighbour n of k maps to k itself here; the identity of the partner does
    # not matter for the invariance algebra, only that the same U is applied
    # on both sides consistently.
    base = rng.normal(
        size=(number_of_kpoints, number_of_neighbors, number_of_bands, number_of_bands)
    )
    base = base + 1j * rng.normal(size=base.shape)
    base *= 0.1  # keep singular values below 1, as physical overlaps are

    unitaries = []
    for _ in range(number_of_kpoints):
        matrix = rng.normal(size=(number_of_bands, number_of_bands)) + 1j * rng.normal(
            size=(number_of_bands, number_of_bands)
        )
        unitary, _ = np.linalg.qr(matrix)
        unitaries.append(unitary)

    rotated = np.empty_like(base)
    for kpoint_index in range(number_of_kpoints):
        for neighbor_index in range(number_of_neighbors):
            rotated[kpoint_index, neighbor_index] = (
                unitaries[kpoint_index].conj().T
                @ base[kpoint_index, neighbor_index]
                @ unitaries[kpoint_index]
            )

    original = gauge_invariant_spread(base, weights, number_of_bands)
    transformed = gauge_invariant_spread(rotated, weights, number_of_bands)
    assert transformed == pytest.approx(original, rel=1e-10)


def test_projectabilities_of_an_orthonormal_set_are_one():
    number_of_kpoints, number_of_bands = 2, 4
    projections = np.zeros((number_of_kpoints, number_of_bands, number_of_bands), dtype=complex)
    for kpoint_index in range(number_of_kpoints):
        projections[kpoint_index] = np.eye(number_of_bands, dtype=complex)
    np.testing.assert_allclose(projectabilities(projections), 1.0, atol=1e-12)


def test_projectabilities_drop_when_a_band_leaves_the_manifold():
    projections = np.zeros((1, 2, 2), dtype=complex)
    projections[0, 0] = [1.0, 0.0]
    projections[0, 1] = [0.0, 0.6]  # partially outside the trial span
    values = projectabilities(projections)
    assert values[0, 0] == pytest.approx(1.0)
    assert values[0, 1] == pytest.approx(0.36)


def test_band_distance_is_zero_for_identical_bands():
    eigenvalues = np.array([[-5.0, -3.0, 1.0], [-4.5, -2.5, 1.5]])
    result = band_distance(eigenvalues, eigenvalues, fermi_energy=0.0)
    assert result.eta == pytest.approx(0.0)
    assert result.eta_maximum == pytest.approx(0.0)


def test_band_distance_recovers_a_uniform_shift_well_below_the_window():
    """All states far below mu carry weight ~1, so eta is just the shift."""
    reference = np.array([[-20.0, -19.0], [-18.0, -17.5]])
    shift = 0.05
    result = band_distance(reference, reference + shift, fermi_energy=0.0)
    assert result.eta == pytest.approx(shift, rel=1e-6)
    assert result.eta_maximum == pytest.approx(shift, rel=1e-6)


def test_band_distance_suppresses_deviations_far_above_the_window():
    """A large error at high energy must not dominate: that is the point of f~."""
    reference = np.array([[-20.0, 40.0]])
    comparison = np.array([[-20.0, 45.0]])  # 5 eV error, 40 eV above mu
    result = band_distance(comparison, reference, fermi_energy=0.0, energy_offset=2.0)
    assert result.eta < 1e-3


def test_band_distance_sorts_before_comparing():
    """Two chains need not label bands identically, only span the manifold."""
    ascending = np.array([[-5.0, -3.0, 1.0]])
    shuffled = np.array([[1.0, -5.0, -3.0]])
    result = band_distance(ascending, shuffled, fermi_energy=0.0)
    assert result.eta == pytest.approx(0.0)


def test_band_distance_compares_the_lowest_common_bands():
    reference = np.array([[-5.0, -3.0, 1.0, 8.0]])
    comparison = np.array([[-5.0, -3.0]])
    result = band_distance(reference, comparison, fermi_energy=0.0)
    assert result.number_of_states == 2
    assert result.eta == pytest.approx(0.0)


def test_band_distance_rejects_mismatched_kpoint_counts():
    with pytest.raises(ValueError, match="k-point counts differ"):
        band_distance(np.zeros((2, 3)), np.zeros((3, 3)), fermi_energy=0.0)


def test_band_distance_rejects_an_all_zero_weight_set():
    """Guards the likely unit slip: Rydberg energies against an eV window."""
    eigenvalues = np.array([[500.0, 600.0]])
    with pytest.raises(ValueError, match="zero Fermi-Dirac weight"):
        band_distance(eigenvalues, eigenvalues, fermi_energy=0.0)
