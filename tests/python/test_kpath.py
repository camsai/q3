"""tests/python/test_kpath.py — Unit tests for q3.workflows.kpath."""

import itertools
import json
import math

import numpy as np
import pytest
from mat3ra.made.reciprocal.lattice_reciprocal import ReciprocalLattice
from q3.structure.lattice import reciprocal_lattice_vectors_inverse_angstrom
from q3.workflows.kpath import (
    _FRACTIONAL_FROM_AFLOW,
    DEFAULT_POINTS_PER_SEGMENT,
    band_path_from_input,
    high_symmetry_points,
    interpolate_kpath,
    kpoints_list_with_uniform_weight,
)

# ── high_symmetry_points ─────────────────────────────────────────────────────


def test_fcc_gamma_at_origin():
    points = high_symmetry_points("FCC")
    gamma = points.get("G") or points.get("Γ")
    assert gamma == (0.0, 0.0, 0.0)


def test_fcc_x_point():
    """X in Q3's ibrav=2 basis, which is AFLOW's (0.5, 0, 0.5) re-expressed.

    Both triples are X -- the star has six members and a signed permutation of
    the primitive vectors moves between them, which is why the wrong-basis
    table passed every FCC test it had. The value is pinned so that a change of
    member is a deliberate one; that it IS an X point is checked against the
    zone in test_cubic_points_sit_at_their_textbook_distance_from_gamma.
    """
    points = high_symmetry_points("FCC")
    assert points["X"] == pytest.approx((0.0, 0.5, 0.5))


def test_bcc_high_symmetry_points():
    points = high_symmetry_points("BCC")
    assert "H" in points
    assert "P" in points
    assert "N" in points


def test_unknown_lattice_raises():
    with pytest.raises(ValueError, match="Unknown Bravais"):
        high_symmetry_points("UNKNOWN")


def test_case_insensitive_alias():
    points_fcc = high_symmetry_points("fcc")
    assert "X" in points_fcc


# ── interpolate_kpath ─────────────────────────────────────────────────────────


def test_single_segment_point_count():
    """A single G→X segment with n points_per_segment produces n+1 total points."""
    kpoints, distances = interpolate_kpath(["G", "X"], "FCC", points_per_segment=10)
    assert len(kpoints) == 11  # 10 + 1 (endpoint included on last segment)


def test_two_segments_no_duplicate_junction():
    """G→X→L: junction point X is not duplicated."""
    kpoints, distances = interpolate_kpath(["G", "X", "L"], "FCC", points_per_segment=5)
    # Segment 1: 5 (G...just before X) + Segment 2: 6 (X...L inclusive) = 11
    assert len(kpoints) == 11


def test_gamma_coordinate():
    """First k-point in a path starting at G is at the origin."""
    kpoints, _ = interpolate_kpath(["G", "X"], "FCC", points_per_segment=5)
    assert kpoints[0] == pytest.approx((0.0, 0.0, 0.0))


def test_endpoint_coordinate():
    """Last k-point in a path ending at X is at the X point."""
    kpoints, _ = interpolate_kpath(["G", "X"], "FCC", points_per_segment=5)
    assert kpoints[-1] == pytest.approx(high_symmetry_points("FCC")["X"])


def test_distances_monotonically_increasing():
    kpoints, distances = interpolate_kpath(["G", "X", "W", "L"], "FCC", points_per_segment=10)
    for i in range(1, len(distances)):
        assert distances[i] >= distances[i - 1]


def test_distances_start_at_zero():
    _, distances = interpolate_kpath(["G", "X"], "FCC", points_per_segment=5)
    assert distances[0] == 0.0


# ── kpoints_list_with_uniform_weight ─────────────────────────────────────────


def test_uniform_weight_assigned():
    kpoints = [(0.0, 0.0, 0.0), (0.5, 0.0, 0.0)]
    result = kpoints_list_with_uniform_weight(kpoints)
    assert len(result) == 2
    assert all(entry[3] == 1.0 for entry in result)
    assert result[0][:3] == (0.0, 0.0, 0.0)


@pytest.mark.parametrize(
    ("lattice_type", "description"),
    [
        ("MCL", "monoclinic"),
        ("MCLC", "C-centred monoclinic"),
        ("TRI", "triclinic"),
        ("RHL", "rhombohedral"),
        ("ORCC", "C-centred orthorhombic"),
        ("ORCF", "face-centred orthorhombic"),
        ("ORCI", "body-centred orthorhombic"),
    ],
)
def test_a_lattice_without_a_table_is_refused_not_approximated(lattice_type, description):
    """
    These seven used to resolve to another lattice's table: monoclinic and
    triclinic to the cubic points, rhombohedral to the hexagonal ones, and the
    three centred orthorhombic variants to simple orthorhombic.

    That is not an approximation. A monoclinic cell has no point where cubic X
    sits, so a band structure requested along "G-X-M" was evaluated at
    coordinates from a different Brillouin zone and then labelled with that
    zone's names -- a plot that looks entirely normal and is not the path it
    claims to be. Refusing is the only honest answer, and the error names the
    way through: explicit coordinates per vertex.
    """
    with pytest.raises(ValueError) as failure:
        high_symmetry_points(lattice_type)

    message = str(failure.value)
    assert description in message, message
    assert "explicit fractional coordinates" in message, message


def test_the_supported_lattices_still_resolve():
    """The six that have real tables are untouched by the refusal above."""
    for lattice_type in ("FCC", "BCC", "CUB", "SC", "HEX", "TET", "TETR", "ORC"):
        points = high_symmetry_points(lattice_type)
        assert "G" in points, f"{lattice_type} lost its gamma point"
        assert points["G"] == (0.0, 0.0, 0.0)


# ── The table against the zone it claims to describe ─────────────────────────
#
# Every test above this line checks the table against itself: ask for FCC X,
# get the triple the table holds. None of them can see the failure this file
# was extended for -- a triple that is a perfectly good number and not the
# point it is named after. These check the table against the Brillouin zone
# Q3's own lattice construction produces, which is the only thing that makes a
# fractional triple mean anything.

#: Representative cells per lattice type, in the ESSE form materials use --
#: including the length unit, because an ESSE lattice without one means bohr
#: (src/io/input/parser/yaml.hpp's read_lattice_length_units) and these numbers
#: are angstrom, as every material under examples/data/materials says of its
#: own. The axial ratios are varied deliberately: a labelled point is a
#: symmetry point for every ratio, so a table that is right only for c = a is
#: not right.
LATTICES_BY_TYPE = {
    "FCC": [{"type": "FCC", "a": 3.867, "units": "angstrom"}],
    "BCC": [{"type": "BCC", "a": 2.482, "units": "angstrom"}],
    "CUB": [{"type": "CUB", "a": 4.0, "units": "angstrom"}],
    "SC": [{"type": "SC", "a": 4.0, "units": "angstrom"}],
    "HEX": [
        {"type": "HEX", "a": 2.464, "c": 6.711, "units": "angstrom"},
        {"type": "HEX", "a": 3.2, "c": 2.1, "units": "angstrom"},
    ],
    "TET": [
        {"type": "TET", "a": 3.0, "c": 5.0, "units": "angstrom"},
        {"type": "TET", "a": 5.0, "c": 3.0, "units": "angstrom"},
    ],
    "ORC": [
        {"type": "ORC", "a": 3.0, "b": 4.0, "c": 5.0, "units": "angstrom"},
        {"type": "ORC", "a": 5.0, "b": 3.1, "c": 4.2, "units": "angstrom"},
    ],
}


def _zone_geometry(fractional, reciprocal_vectors):
    """Return (|k|, shortest image of k, how many nonzero G tie with |k|).

    A point of the first Brillouin zone's *boundary* is one no reciprocal
    lattice vector brings closer to the origin, and that at least one brings
    exactly as close -- it lies on that Bragg plane. A point strictly inside
    ties with nothing; a point outside has a shorter image.
    """
    wavevector = np.asarray(fractional, dtype=float) @ reciprocal_vectors
    norm = float(np.linalg.norm(wavevector))
    tolerance = 1e-9 * max(1.0, norm)

    shortest = norm
    ties = 0
    for translation in itertools.product((-2, -1, 0, 1, 2), repeat=3):
        if translation == (0, 0, 0):
            continue
        image = float(np.linalg.norm(wavevector - np.asarray(translation) @ reciprocal_vectors))
        shortest = min(shortest, image)
        if abs(image - norm) < tolerance:
            ties += 1
    return norm, shortest, ties


@pytest.mark.parametrize("lattice_type", sorted(LATTICES_BY_TYPE))
def test_every_labelled_point_lies_on_the_zone_boundary(lattice_type):
    """A high-symmetry point sits on a Bragg plane. Gamma is the one exception.

    This is the check that catches a triple copied out of a paper whose
    primitive vectors are not Q3's. Two were: BCC H read 1.73 x 2pi/a, well
    outside a zone whose furthest boundary point is 0.87, and BCC P read 0.5,
    strictly inside a boundary at 0.87. Hexagonal K and H read 1.02 with a
    shortest image of 0.385 -- a generic interior point wearing the name of the
    zone corner. Each is an ordinary-looking number that no self-consistent
    table test can tell from the truth.
    """
    points = high_symmetry_points(lattice_type)
    for lattice in LATTICES_BY_TYPE[lattice_type]:
        reciprocal_vectors = reciprocal_lattice_vectors_inverse_angstrom(lattice)
        for label, fractional in sorted(points.items()):
            if label in ("G", "Γ"):
                continue
            norm, shortest, ties = _zone_geometry(fractional, reciprocal_vectors)
            assert shortest == pytest.approx(norm, rel=1e-9), (
                f"{lattice_type} {label} = {fractional} is outside the first zone of "
                f"{lattice}: |k| = {norm:.6f} 1/angstrom but its shortest image is "
                f"{shortest:.6f}"
            )
            assert ties >= 1, (
                f"{lattice_type} {label} = {fractional} is strictly inside the first zone "
                f"of {lattice} at |k| = {norm:.6f} 1/angstrom -- it lies on no Bragg plane, "
                "so it is not the high-symmetry point it is named after"
            )


#: |Gamma to point| for the cubic families, in units of 2*pi/a_conventional,
#: read off the zone's shape: FCC X is a square-face centre at (1,0,0), L a
#: hexagonal-face centre at (1/2,1/2,1/2), W a vertex at (1,1/2,0), K and U
#: the two remaining vertex classes; BCC H is a face centre at (0,0,1), N an
#: edge midpoint at (1/2,1/2,0), P a vertex at (1/2,1/2,1/2).
CUBIC_DISTANCES_IN_UNITS_OF_TWO_PI_OVER_A = {
    "FCC": {
        "X": 1.0,
        "L": 0.8660254037844386,
        "W": 1.1180339887498949,
        "K": 1.0606601717798212,
        "U": 1.0606601717798212,
    },
    "BCC": {"H": 1.0, "N": 0.7071067811865476, "P": 0.8660254037844386},
    "CUB": {"X": 0.5, "M": 0.7071067811865476, "R": 0.8660254037844386},
}

#: The conventional cubic lattice constant per type, given the ESSE "a" the
#: material file carries. Q3's parser makes the same conversion (its "a" for
#: FCC and BCC is the primitive vector length, not the cube edge).
CONVENTIONAL_FROM_ESSE_A = {"FCC": 2.0**0.5, "BCC": 2.0 / 3.0**0.5, "CUB": 1.0}


@pytest.mark.parametrize("lattice_type", sorted(CUBIC_DISTANCES_IN_UNITS_OF_TWO_PI_OVER_A))
def test_cubic_points_sit_at_their_textbook_distance_from_gamma(lattice_type):
    """The zone-boundary test says a point is on *a* Bragg plane; this says which."""
    lattice = LATTICES_BY_TYPE[lattice_type][0]
    reciprocal_vectors = reciprocal_lattice_vectors_inverse_angstrom(lattice)
    conventional_a = lattice["a"] * CONVENTIONAL_FROM_ESSE_A[lattice_type]
    unit = 2.0 * math.pi / conventional_a

    points = high_symmetry_points(lattice_type)
    for label, expected in CUBIC_DISTANCES_IN_UNITS_OF_TWO_PI_OVER_A[lattice_type].items():
        wavevector = np.asarray(points[label], dtype=float) @ reciprocal_vectors
        assert float(np.linalg.norm(wavevector)) / unit == pytest.approx(expected, rel=1e-12), (
            f"{lattice_type} {label} = {points[label]} is not at {expected:.6f} x 2pi/a"
        )


def test_hexagonal_points_sit_where_the_prism_says():
    """K at 2/3 x 2pi/a and M at 1/sqrt(3), for any c/a; H and L above them.

    The hexagonal zone is a prism: K is its vertical edge (the hexagon's
    corner), M the centre of a side face, and H and L sit directly above them
    on the top face, displaced along c* alone. So the in-plane distances are
    fixed by a and nothing else, which is what makes them worth pinning -- and
    what the old K = (2/3, 1/3, 0) failed, landing at 1.018 instead of 0.667.
    """
    points = high_symmetry_points("HEX")
    for lattice in LATTICES_BY_TYPE["HEX"]:
        reciprocal_vectors = reciprocal_lattice_vectors_inverse_angstrom(lattice)
        unit = 2.0 * math.pi / lattice["a"]

        def in_plane(label, vectors=reciprocal_vectors, scale=unit):
            wavevector = np.asarray(points[label], dtype=float) @ vectors
            return float(np.linalg.norm(wavevector[:2])) / scale

        assert in_plane("K") == pytest.approx(2.0 / 3.0, rel=1e-12), (
            f"K = {points['K']} is not the zone corner of {lattice}"
        )
        assert in_plane("M") == pytest.approx(1.0 / math.sqrt(3.0), rel=1e-12), (
            f"M = {points['M']} is not a side-face centre of {lattice}"
        )
        assert in_plane("H") == pytest.approx(in_plane("K"), rel=1e-12), "H is not above K"
        assert in_plane("L") == pytest.approx(in_plane("M"), rel=1e-12), "L is not above M"


# ── Path length is a distance in reciprocal space ────────────────────────────


def test_segment_length_is_the_wavevector_distance():
    """Gamma-X on FCC silicon is 2pi/a_conventional, to the digit."""
    lattice = {"type": "FCC", "a": 3.867, "units": "angstrom"}
    conventional_a = lattice["a"] * 2.0**0.5
    _, distances = interpolate_kpath(["G", "X"], "FCC", points_per_segment=1, lattice=lattice)
    assert distances[-1] == pytest.approx(2.0 * math.pi / conventional_a, rel=1e-12)


def test_fcc_segments_come_out_in_proportion():
    """Gamma-X : Gamma-L is 2 : sqrt(3), whatever basis the coordinates are in.

    This is what a fractional-triple norm gets wrong while looking plausible:
    |(0.5, 0, 0.5)| = 0.707 against |(0.5, 0.5, 0.5)| = 0.866 gives 0.816,
    where the zone says 1.155. Gamma-L happens to come out exact, so the error
    shows up only as one segment drawn too short against another -- on an axis
    with no units printed on it.
    """
    lattice = {"type": "FCC", "a": 3.867, "units": "angstrom"}
    _, to_x = interpolate_kpath(["G", "X"], "FCC", points_per_segment=1, lattice=lattice)
    _, to_l = interpolate_kpath(["G", "L"], "FCC", points_per_segment=1, lattice=lattice)
    assert to_x[-1] / to_l[-1] == pytest.approx(2.0 / math.sqrt(3.0), rel=1e-12)


def test_cubic_proportions_need_no_lattice():
    """For the cubic types b = c = a is not an assumption, so the axis is exact."""
    with_lattice = interpolate_kpath(
        ["G", "X", "L"],
        "FCC",
        points_per_segment=4,
        lattice={"type": "FCC", "a": 3.867, "units": "angstrom"},
    )[1]
    without = interpolate_kpath(["G", "X", "L"], "FCC", points_per_segment=4)[1]
    scale = with_lattice[-1] / without[-1]
    for measured, unscaled in zip(with_lattice, without, strict=True):
        assert measured == pytest.approx(unscaled * scale, rel=1e-12)


def test_hexagonal_axis_follows_c_over_a():
    """Gamma-A is pi/c, so two cells with the same a and different c differ here.

    Exactly the case the fractional-triple norm cannot see: (0, 0, 1/2) has the
    same fractional length whatever c is, so a graphite path and a beryllium
    path would have drawn the same axis.
    """
    for c_angstrom in (6.711, 3.2):
        lattice = {"type": "HEX", "a": 2.464, "c": c_angstrom, "units": "angstrom"}
        _, distances = interpolate_kpath(["G", "A"], "HEX", points_per_segment=1, lattice=lattice)
        assert distances[-1] == pytest.approx(math.pi / c_angstrom, rel=1e-12)


def test_explicit_coordinates_are_measured_in_the_same_metric():
    """A vertex given by hand is not a special case for the axis."""
    lattice = {"type": "HEX", "a": 2.464, "c": 6.711, "units": "angstrom"}
    _, named = interpolate_kpath(["G", "M"], "HEX", points_per_segment=1, lattice=lattice)
    _, by_hand = interpolate_kpath(
        ["G", "Q"],
        "HEX",
        points_per_segment=1,
        explicit_coordinates={"Q": (0.5, 0.0, 0.0)},
        lattice=lattice,
    )
    assert by_hand[-1] == pytest.approx(named[-1], rel=1e-12)


def test_the_supported_tables_do_not_depend_on_the_cell():
    """kpath asks made about a unit cell, which is only sound if this holds."""
    for lattice_type in ("CUB", "FCC", "BCC", "TET", "ORC", "HEX"):
        tables = {
            tuple(
                (point["point"], tuple(round(value, 12) for value in point["coordinates"]))
                for point in ReciprocalLattice(type=lattice_type, a=a, b=b, c=c).symmetry_points
            )
            for a, b, c in ((1.0, 1.0, 1.0), (3.7, 5.1, 9.3), (9.3, 1.2, 0.4))
        }
        assert len(tables) == 1, f"made's {lattice_type} points move with the cell parameters"


def test_the_basis_changes_are_relabellings():
    """Integer and unimodular: the same lattice with its primitive vectors relabelled.

    Derived from q3.structure.lattice, so a change there that altered Q3's basis
    surfaces here as a non-integer matrix rather than as points moving off the zone.
    """
    for lattice_type, matrix in _FRACTIONAL_FROM_AFLOW.items():
        assert np.allclose(matrix, np.round(matrix), atol=1e-12), lattice_type
        assert abs(abs(np.linalg.det(matrix)) - 1.0) < 1e-12, lattice_type


# ── band_path_from_input ─────────────────────────────────────────────────────

SILICON_LATTICE = {"type": "FCC", "a": 3.867, "units": {"length": "angstrom"}}


def test_band_path_defaults_come_from_the_input_and_its_material(tmp_path):
    """One input file describes one path: control.band_path, its point count,
    and the type of the lattice the material names."""
    (tmp_path / "material.json").write_text(json.dumps({"lattice": SILICON_LATTICE}))
    config = {
        "material": "material.json",
        "control": {"band_path": ["G", "X", "L"], "band_path_points_per_segment": 12},
    }
    band_path, lattice_type, points_per_segment, lattice = band_path_from_input(
        tmp_path / "nc.yml", config
    )
    assert (band_path, lattice_type, points_per_segment) == (["G", "X", "L"], "FCC", 12)
    assert lattice == SILICON_LATTICE


def test_band_path_arguments_win_over_the_input(tmp_path):
    config = {"material": "missing.json", "control": {"band_path": ["G", "X"]}}
    assert band_path_from_input(tmp_path / "nc.yml", config, ["G", "L"], "BCC", 5) == (
        ["G", "L"],
        "BCC",
        5,
        None,
    )


def test_band_path_point_count_has_a_default(tmp_path):
    config = {"control": {"band_path": ["G", "X"]}}
    assert band_path_from_input(tmp_path / "nc.yml", config, lattice_type="FCC")[2] == (
        DEFAULT_POINTS_PER_SEGMENT
    )


def test_a_missing_band_path_names_where_to_put_one(tmp_path):
    with pytest.raises(ValueError, match="control.band_path"):
        band_path_from_input(tmp_path / "nc.yml", {"control": {}}, lattice_type="FCC")
