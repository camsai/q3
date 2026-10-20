"""q3.workflows.kpath — high-symmetry points and the k-paths through them.

The points are ``mat3ra-made``'s: AFLOW's tables (Setyawan & Curtarolo 2010,
Comp. Mat. Sci. 49, 299-312), from ``ReciprocalLattice.symmetry_points``.

What this module adds is the basis. A fractional triple means nothing without
the primitive vectors that read it, and AFLOW's are not the ones Q3 builds
(QE's ibrav 2/3/4, ``src/layer_1_structure/crystal/lattice.hpp``). The two are
the same lattice with its primitive vectors relabelled, so each triple is
re-expressed on the way in: identity for CUB, TET and ORC; an integer,
determinant-one matrix for FCC, BCC and HEX. Read without it, AFLOW's numbers
put BCC H outside the Brillouin zone and hexagonal K at (2/3, 1/3, 0) instead
of (2/3, -1/3, 0). ``tests/python/test_kpath.py`` holds
every point to the zone of the cell ``q3.structure.lattice`` builds.

Supported: FCC, BCC, CUB (also SC), HEX, TET (also TETR) and ORC -- the types
Q3's parser builds a primitive cell for. The centred and low-symmetry types are
refused: made has their points, but in the centred primitive cell, while Q3
builds the conventional one, so they would be read against a different lattice.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from mat3ra.made.cell.primitive_cell import get_primitive_lattice_vectors_from_config
from mat3ra.made.reciprocal.lattice_reciprocal import ReciprocalLattice

from q3.io.input_reader import load_material
from q3.structure.lattice import (
    CONVENTIONAL_FROM_PRIMITIVE,
    MADE_TYPE_ALIASES,
    primitive_lattice_vectors_angstrom,
    reciprocal_lattice_vectors_inverse_angstrom,
)

#: How many k-points a segment of a band path is sampled with.
DEFAULT_POINTS_PER_SEGMENT = 30

#: The types Q3's parser builds a primitive cell for.
_SUPPORTED_TYPES = ("CUB", "FCC", "BCC", "HEX", "TET", "ORC")

#: AFLOW's tables for the supported types do not depend on the cell's
#: proportions (test_the_supported_tables_do_not_depend_on_the_cell).
_UNIT_CELL = {"a": 1.0, "b": 1.0, "c": 1.0, "alpha": 90.0, "beta": 90.0, "gamma": 90.0}


def _fractional_basis_change(lattice_type: str) -> np.ndarray:
    """f_q3 = f_aflow . (A_q3 . inv(A_aflow))^T, both sides built at a conventional constant of one.

    AFLOW's side is made's own primitive cell (Setyawan & Curtarolo Tables
    1-6), the basis its symmetry points are expressed in.
    """
    q3 = primitive_lattice_vectors_angstrom(
        {
            "type": lattice_type,
            "a": 1.0 / CONVENTIONAL_FROM_PRIMITIVE.get(lattice_type, 1.0),
            "c": 1.0,
            "units": "angstrom",
        }
    )
    aflow = np.asarray(
        get_primitive_lattice_vectors_from_config(
            ReciprocalLattice(type=lattice_type, **_UNIT_CELL)
        ),
        dtype=float,
    )
    return (q3 @ np.linalg.inv(aflow)).T


_FRACTIONAL_FROM_AFLOW = {name: _fractional_basis_change(name) for name in _SUPPORTED_TYPES}

_REFUSED = {
    "ORCC": "C-centred orthorhombic",
    "ORCF": "face-centred orthorhombic",
    "ORCI": "body-centred orthorhombic",
    "BCT": "body-centred tetragonal",
    "MCL": "monoclinic",
    "MCLC": "C-centred monoclinic",
    "TRI": "triclinic",
    "RHL": "rhombohedral",
}

_GAMMA = (0.0, 0.0, 0.0)
_GAMMA_LABELS = ("G", "Γ", "GAMMA", "GM")


def high_symmetry_points(lattice_type: str) -> dict[str, tuple[float, float, float]]:
    """Label → fractional coordinates in Q3's basis, Gamma under ``"G"`` and ``"Γ"``.

    *lattice_type* is case-insensitive. The cell itself is not asked for: the
    supported tables do not depend on it.

    Raises ValueError for a type Q3 cannot place points for, naming the way
    through (explicit coordinates per vertex).
    """
    normalized = str(lattice_type).upper()
    key = MADE_TYPE_ALIASES.get(normalized, normalized)
    if key not in _FRACTIONAL_FROM_AFLOW:
        known = ", ".join(sorted([*_FRACTIONAL_FROM_AFLOW, *MADE_TYPE_ALIASES]))
        description = _REFUSED.get(key)
        if description is None:
            raise ValueError(f"Unknown Bravais lattice type '{lattice_type}'. Recognised: {known}")
        raise ValueError(
            f"No high-symmetry points for '{lattice_type}' ({description}): made tabulates "
            "them in that centring's primitive cell, and Q3 builds the conventional cell, so "
            f"they would be read against a different lattice. Q3 places points for {known}. "
            "Give the path vertices explicit fractional coordinates instead -- in YAML "
            "`band_path: [{label: X, coordinates: [.., .., ..]}, ...]`, in Python "
            "interpolate_kpath's `explicit_coordinates`."
        )

    points = {"G": _GAMMA, "Γ": _GAMMA}
    for entry in ReciprocalLattice(type=key, **_UNIT_CELL).symmetry_points:
        coordinates = np.asarray(entry["coordinates"], dtype=float) @ _FRACTIONAL_FROM_AFLOW[key]
        if entry["point"] not in _GAMMA_LABELS and not np.allclose(coordinates, 0.0):
            points[str(entry["point"])] = tuple(round(float(value), 12) for value in coordinates)
    return points


def interpolate_kpath(
    path_labels: list[str],
    lattice_type: str,
    points_per_segment: int = DEFAULT_POINTS_PER_SEGMENT,
    explicit_coordinates: dict[str, tuple[float, float, float]] | None = None,
    lattice: dict | None = None,
) -> tuple[list[tuple[float, float, float]], list[float]]:
    """K-points along *path_labels*, and the cumulative distance to each.

    Each segment gets *points_per_segment* points; only the last includes its
    endpoint, so junctions are not duplicated. *explicit_coordinates* overrides
    the table per label.

    Distances are wavevector lengths through the reciprocal metric of
    *lattice* (the material's ESSE node) -- in inverse angstrom with the 2 pi,
    and never the norm of a fractional difference, which drew FCC's Gamma-X 29%
    short. Without *lattice* the metric is a unit cell of the type: exact for
    the cubic types, where b = c = a, and only proportional otherwise.
    """
    known = high_symmetry_points(lattice_type)
    if explicit_coordinates:
        known.update(explicit_coordinates)

    cell = {**lattice} if lattice else {"type": lattice_type, "a": 1.0, "units": "angstrom"}
    cell.setdefault("type", lattice_type)
    reciprocal = reciprocal_lattice_vectors_inverse_angstrom(cell)
    metric = reciprocal @ reciprocal.T

    def vertex(label: str) -> np.ndarray:
        if label in known:
            return np.asarray(known[label], dtype=float)
        if label in _GAMMA_LABELS:
            return np.zeros(3)
        raise ValueError(
            f"High-symmetry point '{label}' not found for lattice type "
            f"'{lattice_type}'.  Available: {sorted(known)}"
        )

    vertices = [vertex(label) for label in path_labels]
    kpoints: list[tuple[float, float, float]] = []
    distances: list[float] = []
    travelled = 0.0
    for index, (start, end) in enumerate(zip(vertices, vertices[1:], strict=False)):
        delta = end - start
        length = float(math.sqrt(delta @ metric @ delta))
        fractions = (
            np.arange(points_per_segment + (index == len(vertices) - 2)) / points_per_segment
        )
        kpoints.extend(
            tuple(float(value) for value in row) for row in start + fractions[:, None] * delta
        )
        distances.extend((travelled + fractions * length).tolist())
        travelled += length
    return kpoints, distances


def kpoints_list_with_uniform_weight(
    kpoints: list[tuple[float, float, float]],
) -> list[tuple[float, float, float, float]]:
    """``(kx, ky, kz, 1.0)`` per k-point, the ``kpoints_list`` shape a manifest carries.

    Plain floats, whatever the coordinates arrived as, so the list can be written as YAML.
    """
    return [(float(kx), float(ky), float(kz), 1.0) for kx, ky, kz in kpoints]


# TODO: workflow plumbing, not k-path geometry -- move to mat3ra-wode, beside the
#   workflow context it resolves defaults from, once wode can read a Q3 input.
def band_path_from_input(
    scf_input_path: str | Path,
    config: dict,
    band_path: list[str] | None = None,
    lattice_type: str | None = None,
    points_per_segment: int | None = None,
) -> tuple[list[str], str, int, dict | None]:
    """The band path a workflow follows: each argument given wins, else the input file's own.

    ``control.band_path`` and ``control.band_path_points_per_segment`` from
    *config* (the manifest read from *scf_input_path*), and ``lattice.type`` from
    the material it names -- so one input file describes one path, for
    electrons and phonons alike.  The fourth value is that material's lattice
    node when it was read for the type, which also measures the path's length;
    otherwise None.
    """
    control = config.get("control", {})
    if band_path is None:
        band_path = control.get("band_path")
        if band_path is None:
            raise ValueError(
                f"band_path was not passed and {scf_input_path} has no control.band_path -- "
                "pass band_path explicitly, or add it to the input file "
                "(see examples/band_structure--*/*.yml)."
            )
    if points_per_segment is None:
        points_per_segment = control.get("band_path_points_per_segment", DEFAULT_POINTS_PER_SEGMENT)
    lattice = None
    if lattice_type is None:
        lattice = load_material(config, Path(scf_input_path).parent)["lattice"]
        lattice_type = lattice["type"]
    return list(band_path), str(lattice_type), int(points_per_segment), lattice
