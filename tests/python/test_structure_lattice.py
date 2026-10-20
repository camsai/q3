"""tests/python/test_structure_lattice.py — q3.structure.lattice against the parser.

This module's whole job is to agree with ``src/io/input/parser/yaml.hpp``: it
builds, in Python, the cell Q3 builds in C++ from the same material JSON. So
the tests here are written the other way round from a usual unit test -- the
expected value is the parser's own arithmetic, restated, rather than a number
this module happens to produce.

The one that matters is the length unit. The parser's
``read_lattice_length_units`` defaults to **bohr** when the lattice node
carries no ``units``, and applies it to a, b and c alike. This adapter read
``a`` as angstrom regardless, so the two built cells differed by 1.8897x on any
file without the annotation -- latent, because every material shipped under
``examples/data/materials/`` annotates angstrom, and consequential on the path
that exists to keep the frames consistent: ASE would have relaxed a crystal
1.89x larger than the one Q3 solved.

Needs no ASE, unlike ``test_ase_atoms_roundtrip.py``, which is why the unit
handling is tested here rather than there.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from q3.constants import BOHR_TO_ANGSTROM
from q3.structure.lattice import (
    primitive_lattice_vectors_angstrom,
    reciprocal_lattice_vectors_inverse_angstrom,
)

#: src/layer_0_core/types/constants.hpp:90, BOHR_RADIUS_SI * 1e10 (CODATA 2018).
#: This is the value the parser converts with, and it is not quite the one this
#: module carries -- see test_the_two_bohr_constants_have_not_drifted.
PARSER_BOHR_IN_ANGSTROM = 0.529177210903

#: Loose enough to admit that constant difference (2.3e-11 relative) and
#: nothing else: a wrong unit is a factor 1.89, a wrong convention a factor of
#: order 1.
AGREEMENT = 1e-9


def parser_cell_angstrom(lattice: dict) -> np.ndarray:
    """The cell ``parse_material`` would build, restated in angstrom.

    Mirrors src/io/input/parser/yaml.hpp: resolve the length unit (bohr unless
    annotated), scale a/b/c by it, then either the ibrav construction for a
    cubic type or ``get_lattice_vectors_from_cell_parameters``.
    """
    units = lattice.get("units")
    unit_name = units if isinstance(units, str) else (units or {}).get("length", "bohr")
    to_angstrom = PARSER_BOHR_IN_ANGSTROM if str(unit_name).lower() == "bohr" else 1.0

    lattice_type = str(lattice.get("type", "")).upper()
    a = float(lattice["a"]) * to_angstrom
    b = float(lattice["b"]) * to_angstrom if "b" in lattice else a
    c = float(lattice["c"]) * to_angstrom if "c" in lattice else a

    if lattice_type == "FCC":
        conventional = a * math.sqrt(2.0)
        return np.array([[-0.5, 0.0, 0.5], [0.0, 0.5, 0.5], [-0.5, 0.5, 0.0]]) * conventional
    if lattice_type == "BCC":
        conventional = a * 2.0 / math.sqrt(3.0)
        return np.array([[0.5, 0.5, 0.5], [-0.5, 0.5, 0.5], [-0.5, -0.5, 0.5]]) * conventional
    if lattice_type in ("SC", "CUB"):
        return np.eye(3) * a

    gamma = math.radians(float(lattice.get("gamma", 120.0 if lattice_type == "HEX" else 90.0)))
    alpha = math.radians(float(lattice.get("alpha", 90.0)))
    beta = math.radians(float(lattice.get("beta", 90.0)))
    c_direction_y = (math.cos(alpha) - math.cos(beta) * math.cos(gamma)) / math.sin(gamma)
    return np.array(
        [
            [a, 0.0, 0.0],
            [b * math.cos(gamma), b * math.sin(gamma), 0.0],
            [
                c * math.cos(beta),
                c * c_direction_y,
                c * math.sqrt(max(0.0, 1.0 - math.cos(beta) ** 2 - c_direction_y**2)),
            ],
        ]
    )


# ── The length unit ──────────────────────────────────────────────────────────

#: Every spelling of the annotation the parser accepts, and what it means.
UNIT_ANNOTATIONS = [
    pytest.param(None, "bohr", id="absent-means-bohr"),
    pytest.param("bohr", "bohr", id="string-bohr"),
    pytest.param("angstrom", "angstrom", id="string-angstrom"),
    pytest.param({"length": "bohr"}, "bohr", id="object-bohr"),
    pytest.param({"length": "angstrom", "angle": "degree"}, "angstrom", id="object-angstrom"),
    pytest.param({"angle": "degree"}, "bohr", id="object-without-length-means-bohr"),
]


#: Cell parameters per type. HEX names b as well: the parser reads it into the
#: second vector, and so does this module since HEX took the parser's generic
#: construction rather than a special case of its own.
CELLS_BY_TYPE = {
    "FCC": {"a": 5.13},
    "BCC": {"a": 5.13},
    "CUB": {"a": 5.13},
    "SC": {"a": 5.13},
    "HEX": {"a": 5.13, "b": 5.4, "c": 7.4},
    "TET": {"a": 5.13, "c": 7.4},
    "ORC": {"a": 5.13, "b": 6.2, "c": 7.4},
}


@pytest.mark.parametrize(("annotation", "expected_unit"), UNIT_ANNOTATIONS)
@pytest.mark.parametrize("lattice_type", sorted(CELLS_BY_TYPE))
def test_lengths_are_read_in_the_unit_the_parser_reads_them_in(
    annotation, expected_unit, lattice_type
):
    """The cell this module builds is the cell the parser builds, unit included."""
    lattice: dict = {"type": lattice_type, **CELLS_BY_TYPE[lattice_type]}
    if annotation is not None:
        lattice["units"] = annotation

    produced = primitive_lattice_vectors_angstrom(lattice)
    assert produced == pytest.approx(parser_cell_angstrom(lattice), rel=AGREEMENT), (
        f"{lattice_type} with units={annotation!r} disagrees with the parser"
    )

    # And the factor is the one the unit names, not the other one.
    in_angstrom = primitive_lattice_vectors_angstrom({**lattice, "units": "angstrom"})
    ratio = float(np.linalg.norm(produced)) / float(np.linalg.norm(in_angstrom))
    expected_ratio = PARSER_BOHR_IN_ANGSTROM if expected_unit == "bohr" else 1.0
    assert ratio == pytest.approx(expected_ratio, rel=AGREEMENT)


def test_an_unannotated_lattice_constant_is_bohr_not_angstrom():
    """The defect this file was added for, stated as one number.

    5.13 bohr is 2.7146 angstrom. Reading it as angstrom gave a cell 1.8897x
    too large -- and silently, because a cubic cell of either size is a
    perfectly plausible crystal.
    """
    produced = primitive_lattice_vectors_angstrom({"type": "CUB", "a": 5.13})
    assert produced[0][0] == pytest.approx(5.13 * PARSER_BOHR_IN_ANGSTROM, rel=AGREEMENT)
    assert produced[0][0] == pytest.approx(2.71468, rel=1e-5)
    assert produced[0][0] != pytest.approx(5.13, rel=1e-3)


def test_the_cubic_conventional_conversion_still_applies_after_the_unit():
    """ESSE's `a` is the primitive length for FCC/BCC; the unit scales it first.

    Two multipliers on one number, and applying only one of them is the shape
    of the original bug -- so pin the product: 5.13 bohr x sqrt(2) for FCC.
    """
    fcc = primitive_lattice_vectors_angstrom({"type": "FCC", "a": 5.13})
    conventional = 5.13 * PARSER_BOHR_IN_ANGSTROM * math.sqrt(2.0)
    assert fcc[0] == pytest.approx([-0.5 * conventional, 0.0, 0.5 * conventional], rel=AGREEMENT)

    bcc = primitive_lattice_vectors_angstrom({"type": "BCC", "a": 5.13})
    conventional = 5.13 * PARSER_BOHR_IN_ANGSTROM * 2.0 / math.sqrt(3.0)
    assert bcc[0] == pytest.approx([0.5 * conventional] * 3, rel=AGREEMENT)


@pytest.mark.parametrize("lattice_type", ["HEX", "TET", "ORC"])
def test_b_and_c_carry_the_unit_too(lattice_type):
    """The parser scales a, b and c by the same factor; scaling only a would
    change the cell's shape, not just its size -- a worse failure than the
    original, since it survives any comparison of ratios."""
    bohr = {"type": lattice_type, **CELLS_BY_TYPE[lattice_type]}
    angstrom = {**bohr, "units": "angstrom"}

    scaled = primitive_lattice_vectors_angstrom(bohr) / PARSER_BOHR_IN_ANGSTROM
    assert scaled == pytest.approx(primitive_lattice_vectors_angstrom(angstrom), rel=AGREEMENT)


def test_a_missing_b_or_c_still_defaults_to_a():
    """In its own unit, so the default is a length and not a mixed one."""
    produced = primitive_lattice_vectors_angstrom({"type": "ORC", "a": 5.13})
    edge = 5.13 * PARSER_BOHR_IN_ANGSTROM
    assert np.diag(produced) == pytest.approx([edge, edge, edge], rel=AGREEMENT)


# ── The vectors branch, which was already right, stays right ─────────────────


def test_explicit_vectors_keep_their_own_precedence():
    """A unit on the vectors block wins over one on the lattice, as in the parser."""
    lattice = {
        "units": {"length": "bohr"},
        "vectors": {
            "a": [4.0, 0.0, 0.0],
            "b": [0.0, 4.0, 0.0],
            "c": [0.0, 0.0, 4.0],
            "units": "angstrom",
        },
    }
    assert primitive_lattice_vectors_angstrom(lattice)[0][0] == pytest.approx(4.0)

    del lattice["vectors"]["units"]
    assert primitive_lattice_vectors_angstrom(lattice)[0][0] == pytest.approx(
        4.0 * PARSER_BOHR_IN_ANGSTROM, rel=AGREEMENT
    )


def test_alat_multiplies_the_rows():
    lattice = {
        "vectors": {
            "a": [1.0, 0.0, 0.0],
            "b": [0.0, 1.0, 0.0],
            "c": [0.0, 0.0, 1.0],
            "alat": 7.3,
            "units": "angstrom",
        }
    }
    assert primitive_lattice_vectors_angstrom(lattice)[0][0] == pytest.approx(7.3)


@pytest.mark.parametrize(
    ("lattice", "description"),
    [
        ({"type": "CUB", "a": 4.0, "units": "furlong"}, "lattice"),
        (
            {
                "vectors": {
                    "a": [1.0, 0.0, 0.0],
                    "b": [0.0, 1.0, 0.0],
                    "c": [0.0, 0.0, 1.0],
                    "units": "furlong",
                }
            },
            "lattice.vectors",
        ),
    ],
)
def test_unknown_length_units_are_refused_and_the_node_is_named(lattice, description):
    """Guessing would be a silently wrong cell, and the message says where to look."""
    with pytest.raises(ValueError, match="unknown length units") as failure:
        primitive_lattice_vectors_angstrom(lattice)
    assert description in str(failure.value), failure.value


# ── The constant itself ──────────────────────────────────────────────────────


def test_the_bohr_radius_is_the_packages_own():
    """A cell given in bohr is converted by `q3.constants`, the package's one table.

    Not by a literal of this module's: a copied constant is one that can
    drift, and the tree had three. The package's value agrees with the
    parser's (`BOHR_RADIUS_ANGSTROM` in constants.hpp) to 7e-10, inside this
    file's AGREEMENT, which is why every comparison above can be made against
    the parser's number.
    """
    ours = primitive_lattice_vectors_angstrom({"type": "CUB", "a": 1.0})[0][0]
    assert ours == BOHR_TO_ANGSTROM
    assert ours == pytest.approx(PARSER_BOHR_IN_ANGSTROM, rel=AGREEMENT)


# ── The reciprocal vectors follow the unit ───────────────────────────────────


def test_reciprocal_vectors_follow_from_the_same_cell():
    """b_i . a_j = 2 pi delta_ij, whatever unit the lengths were given in."""
    for lattice in ({"type": "FCC", "a": 5.13}, {"type": "FCC", "a": 5.13, "units": "angstrom"}):
        cell = primitive_lattice_vectors_angstrom(lattice)
        reciprocal = reciprocal_lattice_vectors_inverse_angstrom(lattice)
        assert cell @ reciprocal.T == pytest.approx(2.0 * math.pi * np.eye(3), abs=1e-12)


# ── The two shapes a material JSON takes ─────────────────────────────────────


def test_the_cubic_primitive_vectors_meet_at_sixty_degrees():
    """An ESSE cubic `a` is the primitive vector length, so the rows are it.

    Moved here from test_interpolation_workflow.py with the construction it
    tests. Silicon's ESSE a = 3.867 angstrom is |a1|, not the cube edge, and
    the angle between primitive FCC vectors is 60 degrees whichever of the two
    references chose them.
    """
    rows = primitive_lattice_vectors_angstrom({"type": "FCC", "a": 3.866976, "units": "angstrom"})
    for row in rows:
        assert float(np.linalg.norm(row)) == pytest.approx(3.866976, abs=1e-6)
    cosine = float(np.dot(rows[0], rows[1])) / float(
        np.linalg.norm(rows[0]) * np.linalg.norm(rows[1])
    )
    assert cosine == pytest.approx(0.5, abs=1e-9)


def test_a_hexagonal_cell_given_as_parameters_gets_gamma_120():
    """The parser's default for HEX alone, and c is independent of a."""
    rows = primitive_lattice_vectors_angstrom(
        {"type": "HEX", "a": 2.46, "c": 6.7, "units": "angstrom"}
    )
    assert float(np.linalg.norm(rows[1])) == pytest.approx(2.46)
    assert rows[1][0] == pytest.approx(2.46 * -0.5)
    assert float(np.linalg.norm(rows[2])) == pytest.approx(6.7)


def test_a_lattice_that_is_neither_shape_says_so():
    """Not a KeyError naming 'a', which says nothing about what was expected."""
    with pytest.raises(ValueError, match="either 'vectors' or 'type' and 'a'"):
        primitive_lattice_vectors_angstrom({"units": "angstrom"})


@pytest.mark.parametrize(
    ("angles", "expected"),
    [
        ({"alpha": 90.0, "beta": 90.0, "gamma": 180.0}, "degenerate"),
        ({"alpha": 20.0, "beta": 20.0, "gamma": 170.0}, "three-dimensional"),
    ],
)
def test_angles_that_describe_no_cell_are_refused(angles, expected):
    """The parser throws on both; clamping would hand back a cell of zero height
    and the first matrix inversion would fail as a singular matrix, naming
    neither the angle nor the file."""
    with pytest.raises(ValueError, match=expected):
        primitive_lattice_vectors_angstrom({"type": "TRI", "a": 3.0, "units": "angstrom", **angles})
