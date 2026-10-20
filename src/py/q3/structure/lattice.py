"""q3.structure.lattice — the primitive lattice vectors a Q3 run actually uses.

Mirrors ``src/io/input/parser/yaml.hpp``: FCC, BCC and SC/CUB from QE's ibrav
1-3 (the ESSE ``a`` being the primitive length for FCC and BCC), every other
type from a, b, c and the angles as ``get_lattice_vectors_from_cell_parameters``
builds them. The orientation matters -- a force Q3 reports, or a fractional
k-point, is only meaningful in Q3's own frame -- and it is the one thing
``mat3ra-made`` does not model: its ``Lattice`` keeps only a, b, c and the
angles and rebuilds vectors in a frame of its own. Everything that needs only
lengths or the metric comes from made, through ``reciprocal_lattice``.

Lengths carry the ESSE ``units`` annotation, defaulting to **bohr** as the
parser does.
"""

from __future__ import annotations

import math

import numpy as np
from mat3ra.made.reciprocal.lattice_reciprocal import ReciprocalLattice

from q3.constants import BOHR_TO_ANGSTROM

_LENGTH_UNIT_IN_ANGSTROM = {"bohr": BOHR_TO_ANGSTROM, "angstrom": 1.0}

#: ESSE's FCC/BCC ``a`` is the primitive vector length; ibrav takes the cube edge.
CONVENTIONAL_FROM_PRIMITIVE = {"FCC": math.sqrt(2.0), "BCC": 2.0 / math.sqrt(3.0)}

#: lattice.hpp's ibrav 2 and 3, in units of the conventional constant.
_IBRAV_ROWS = {
    "FCC": [[-0.5, 0.0, 0.5], [0.0, 0.5, 0.5], [-0.5, 0.5, 0.0]],
    "BCC": [[0.5, 0.5, 0.5], [-0.5, 0.5, 0.5], [-0.5, -0.5, 0.5]],
}

#: Q3 input spellings made knows by another name.
MADE_TYPE_ALIASES = {"SC": "CUB", "TETR": "TET"}


def _length_unit(node: dict) -> str | None:
    """*node*'s length unit -- ``units`` is a string or ``{length: ...}`` -- or None."""
    units = node.get("units")
    name = units.get("length") if isinstance(units, dict) else units
    return str(name).lower() if name else None


def _length_to_angstrom(node: dict, description: str) -> float:
    name = _length_unit(node) or "bohr"
    if name not in _LENGTH_UNIT_IN_ANGSTROM:
        raise ValueError(
            f"{description} carries unknown length units {name!r}; "
            f"expected one of {sorted(_LENGTH_UNIT_IN_ANGSTROM)}"
        )
    return _LENGTH_UNIT_IN_ANGSTROM[name]


def primitive_lattice_vectors_angstrom(lattice: dict):
    """Rows a1, a2, a3 in angstrom for an ESSE ``lattice`` node, in Q3's frame."""
    vectors = lattice.get("vectors")
    if vectors is not None:
        missing = [key for key in ("a", "b", "c") if key not in vectors]
        if missing:
            raise KeyError(f"lattice.vectors must contain 'a', 'b' and 'c' rows; missing {missing}")
        # A unit on the vectors block is the more specific one and wins.
        node, description = (
            (vectors, "lattice.vectors") if _length_unit(vectors) else (lattice, "lattice")
        )
        scale = float(vectors.get("alat", 1.0)) * _length_to_angstrom(node, description)
        return np.array([vectors["a"], vectors["b"], vectors["c"]], dtype=float) * scale

    if "a" not in lattice:
        raise ValueError(
            f"the lattice must give either 'vectors' or 'type' and 'a'; got keys {sorted(lattice)}"
        )
    lattice_type = str(lattice.get("type", "")).upper()
    to_angstrom = _length_to_angstrom(lattice, "lattice")
    a = float(lattice["a"]) * to_angstrom

    if lattice_type in _IBRAV_ROWS:
        return np.array(_IBRAV_ROWS[lattice_type]) * (a * CONVENTIONAL_FROM_PRIMITIVE[lattice_type])
    if lattice_type in ("SC", "CUB"):
        return np.eye(3) * a

    b = float(lattice["b"]) * to_angstrom if "b" in lattice else a
    c = float(lattice["c"]) * to_angstrom if "c" in lattice else a
    alpha, beta, gamma = (
        math.radians(float(lattice.get(name, default)))
        for name, default in (
            ("alpha", 90.0),
            ("beta", 90.0),
            ("gamma", 120.0 if lattice_type == "HEX" else 90.0),
        )
    )
    # Both refusals are the parser's: clamping would return a cell of zero
    # height, and the first inversion would fail naming neither angle nor file.
    if abs(math.sin(gamma)) < 1.0e-12:
        raise ValueError(f"gamma = {math.degrees(gamma)} degrees gives a degenerate cell")
    c_y = (math.cos(alpha) - math.cos(beta) * math.cos(gamma)) / math.sin(gamma)
    c_z_squared = 1.0 - math.cos(beta) * math.cos(beta) - c_y * c_y
    if c_z_squared <= 0.0:
        raise ValueError(
            f"cell angles ({math.degrees(alpha)}, {math.degrees(beta)}, "
            f"{math.degrees(gamma)}) degrees do not describe a three-dimensional cell"
        )
    return np.array(
        [
            [a, 0.0, 0.0],
            [b * math.cos(gamma), b * math.sin(gamma), 0.0],
            [c * math.cos(beta), c * c_y, c * math.sqrt(c_z_squared)],
        ]
    )


def reciprocal_lattice_vectors_inverse_angstrom(lattice: dict):
    """Rows b1, b2, b3 in inverse angstrom with the 2 pi: the exact dual, in Q3's frame."""
    return 2.0 * math.pi * np.linalg.inv(primitive_lattice_vectors_angstrom(lattice)).T


def reciprocal_lattice(lattice: dict) -> ReciprocalLattice:
    """made's ``ReciprocalLattice`` for the cell Q3 builds -- for lengths and the metric.

    Its orientation is made's, not Q3's, and its ``symmetry_points`` are in
    AFLOW's basis; ``q3.workflows.kpath`` owns the conversion to Q3's.
    """
    declared = str(lattice.get("type") or "").upper()
    vectors = primitive_lattice_vectors_angstrom(lattice).tolist()
    try:
        return ReciprocalLattice.from_vectors_array(
            vectors, type=MADE_TYPE_ALIASES.get(declared, declared)
        )
    except ValueError:
        return ReciprocalLattice.from_vectors_array(vectors)
