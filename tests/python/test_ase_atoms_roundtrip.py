"""Structure adapter round trip: ASE Atoms → Q3 material JSON → ASE Atoms.

The two directions used to be written by different hands -- the reader in
``q3.structure.ase_atoms``, the writer inlined in the ASE calculator -- and they
did not agree: the writer emitted ``lattice.vectors`` and the reader understood
only ``lattice.type`` plus a lattice constant, so reading back what the
calculator had just written raised ``KeyError``.  Nothing exercised the pair
together, which is why the disagreement survived.

These tests exercise the pair together.  The tolerance is 1e-12 angstrom rather
than a physical tolerance because nothing here is a measurement: a conversion to
fractional coordinates and back is exact arithmetic, and anything above
floating-point noise is a bug.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
from ase import Atoms
from ase.constraints import FixCartesian
from q3.structure.ase_atoms import (
    ase_atoms_to_material_json,
    material_json_to_ase_atoms,
)

EXACT_TOLERANCE_ANGSTROM = 1e-12


def _write_and_read_back(atoms, tmp_path, template_material=None):
    """Round trip *atoms* through a material JSON file on disk."""
    material = ase_atoms_to_material_json(atoms, template_material)
    material_path = tmp_path / "material.json"
    material_path.write_text(json.dumps(material))
    return material_json_to_ase_atoms(material_path)


def _silicon_diamond(cell):
    """Two-atom diamond-structure silicon in the given cell."""
    return Atoms(
        symbols=["Si", "Si"],
        scaled_positions=[[0.0, 0.0, 0.0], [0.25, 0.25, 0.25]],
        cell=cell,
        pbc=True,
    )


FCC_CELL = np.array([[-2.715, 0.0, 2.715], [0.0, 2.715, 2.715], [-2.715, 2.715, 0.0]])
BCC_CELL = np.array([[1.435, 1.435, 1.435], [-1.435, 1.435, 1.435], [-1.435, -1.435, 1.435]])
SIMPLE_CUBIC_CELL = np.diag([4.05, 4.05, 4.05])
HEXAGONAL_CELL = np.array([[3.21, 0.0, 0.0], [-1.605, 2.7799, 0.0], [0.0, 0.0, 5.21]])
# A cell with no right angles and no equal edges: the case a Bravais
# reparametrization cannot represent and raw vectors can.
SHEARED_CELL = np.array([[4.1, 0.0, 0.0], [0.83, 3.92, 0.0], [0.41, 0.57, 4.33]])


@pytest.mark.parametrize(
    "cell",
    [FCC_CELL, BCC_CELL, SIMPLE_CUBIC_CELL, HEXAGONAL_CELL, SHEARED_CELL],
    ids=["fcc", "bcc", "simple_cubic", "hexagonal", "sheared"],
)
def test_cell_and_positions_survive_the_round_trip(cell, tmp_path):
    """Every lattice shape comes back as the same cell and the same atoms."""
    original = _silicon_diamond(cell)
    recovered = _write_and_read_back(original, tmp_path)

    assert np.max(np.abs(recovered.get_cell()[:] - original.get_cell()[:])) < (
        EXACT_TOLERANCE_ANGSTROM
    )
    assert np.max(np.abs(recovered.get_positions() - original.get_positions())) < (
        EXACT_TOLERANCE_ANGSTROM
    )
    assert recovered.get_chemical_symbols() == original.get_chemical_symbols()


def test_written_material_uses_explicit_vectors_in_angstrom():
    """The writer emits raw Cartesian rows, which is what Q3's parser reads."""
    material = ase_atoms_to_material_json(_silicon_diamond(SHEARED_CELL))

    lattice = material["lattice"]
    assert "vectors" in lattice
    assert lattice["vectors"]["units"] == "angstrom"
    assert np.allclose(np.array([lattice["vectors"][row] for row in ("a", "b", "c")]), SHEARED_CELL)
    assert material["basis"]["units"] == "crystal"


def test_displaced_atoms_are_carried_through(tmp_path):
    """A rattled structure round trips: this is the sampling path for training data."""
    original = _silicon_diamond(FCC_CELL)
    original.positions[1] += np.array([0.07, -0.03, 0.05])

    recovered = _write_and_read_back(original, tmp_path)

    assert np.max(np.abs(recovered.get_positions() - original.get_positions())) < (
        EXACT_TOLERANCE_ANGSTROM
    )


def test_template_site_identifiers_are_preserved():
    """Site ids from a template material survive, so a written material keeps its own numbering."""
    template = {
        "name": "Silicon FCC",
        "basis": {
            "elements": [{"id": 7, "value": "Si"}, {"id": 9, "value": "Si"}],
            "coordinates": [
                {"id": 7, "value": [0.0, 0.0, 0.0]},
                {"id": 9, "value": [0.25, 0.25, 0.25]},
            ],
            "units": "crystal",
        },
        "lattice": {"type": "FCC", "a": 3.84},
    }

    material = ase_atoms_to_material_json(_silicon_diamond(FCC_CELL), template)

    assert [entry["id"] for entry in material["basis"]["coordinates"]] == [7, 9]
    assert [entry["id"] for entry in material["basis"]["elements"]] == [7, 9]
    assert material["name"] == "Silicon FCC"


def test_template_is_not_mutated():
    """The caller's template must not be edited underneath it."""
    template = {
        "basis": {
            "elements": [{"id": 1, "value": "Si"}, {"id": 2, "value": "Si"}],
            "coordinates": [
                {"id": 1, "value": [0.0, 0.0, 0.0]},
                {"id": 2, "value": [0.25, 0.25, 0.25]},
            ],
            "units": "crystal",
        },
        "lattice": {"type": "FCC", "a": 3.84},
    }

    displaced = _silicon_diamond(FCC_CELL)
    displaced.positions[1] += np.array([0.1, 0.0, 0.0])
    ase_atoms_to_material_json(displaced, template)

    assert template["lattice"] == {"type": "FCC", "a": 3.84}
    assert template["basis"]["coordinates"][1]["value"] == [0.25, 0.25, 0.25]


def test_site_count_change_renumbers_rather_than_mismatching():
    """A vacancy structure gets fresh ids instead of silently reusing the template's."""
    template = {
        "basis": {
            "elements": [{"id": 1, "value": "Si"}, {"id": 2, "value": "Si"}],
            "coordinates": [
                {"id": 1, "value": [0.0, 0.0, 0.0]},
                {"id": 2, "value": [0.25, 0.25, 0.25]},
            ],
            "units": "crystal",
        },
        "lattice": {"type": "FCC", "a": 3.84},
    }

    single_atom = Atoms(symbols=["Si"], scaled_positions=[[0.0, 0.0, 0.0]], cell=FCC_CELL, pbc=True)
    material = ase_atoms_to_material_json(single_atom, template)

    assert [entry["id"] for entry in material["basis"]["coordinates"]] == [1]
    assert len(material["basis"]["elements"]) == 1


# ── Per-sublattice species labels ─────────────────────────────────────────────
# ASE knows only chemical symbols, but a Q3 material may label sites with
# per-sublattice species -- an antiferromagnet's Cr1/Cr2, which carry opposite
# starting magnetizations and their own method.pseudopotentials entries.


CHROMIUM_TEMPLATE = {
    "basis": {
        "elements": [{"id": 0, "value": "Cr1"}, {"id": 1, "value": "Cr2"}],
        "coordinates": [
            {"id": 0, "value": [0.0, 0.0, 0.0]},
            {"id": 1, "value": [0.5, 0.5, 0.5]},
        ],
        "units": "crystal",
    },
    "lattice": {"type": "CUB", "a": 2.8845},
}


def _antiferromagnetic_chromium():
    return Atoms(
        symbols=["Cr", "Cr"],
        scaled_positions=[[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
        cell=np.eye(3) * 2.8845,
        pbc=True,
    )


def test_sublattice_species_labels_survive():
    """Collapsing Cr1/Cr2 to Cr would break the pseudopotential lookup and the magnetism."""
    material = ase_atoms_to_material_json(_antiferromagnetic_chromium(), CHROMIUM_TEMPLATE)

    assert [entry["value"] for entry in material["basis"]["elements"]] == ["Cr1", "Cr2"]


def test_labels_are_dropped_when_they_no_longer_describe_the_structure():
    """A substituted basis is a different structure; ASE's symbols are then the truth."""
    substituted = Atoms(
        symbols=["Fe", "Fe"],
        scaled_positions=[[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
        cell=np.eye(3) * 2.8845,
        pbc=True,
    )

    material = ase_atoms_to_material_json(substituted, CHROMIUM_TEMPLATE)

    assert [entry["value"] for entry in material["basis"]["elements"]] == ["Fe", "Fe"]


def test_labels_are_dropped_when_the_site_count_changes():
    """A vacancy structure cannot inherit a two-site template's labels."""
    single_site = Atoms(symbols=["Cr"], scaled_positions=[[0.0, 0.0, 0.0]], cell=np.eye(3) * 2.8845)

    material = ase_atoms_to_material_json(single_site, CHROMIUM_TEMPLATE)

    assert [entry["value"] for entry in material["basis"]["elements"]] == ["Cr"]


def test_plain_symbols_are_used_without_a_template():
    """Nothing to inherit means ASE's symbols."""
    material = ase_atoms_to_material_json(_antiferromagnetic_chromium())

    assert [entry["value"] for entry in material["basis"]["elements"]] == ["Cr", "Cr"]


def test_single_letter_symbol_does_not_inherit_a_longer_label():
    """A prefix test would write carbon out as chromium: "Cr1" starts with "C"."""
    carbon = Atoms(
        symbols=["C", "C"],
        scaled_positions=[[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
        cell=np.eye(3) * 3.567,
        pbc=True,
    )

    material = ase_atoms_to_material_json(carbon, CHROMIUM_TEMPLATE)

    assert [entry["value"] for entry in material["basis"]["elements"]] == ["C", "C"]


# ── Explicit lattice vectors: alat and units ──────────────────────────────────
# Q3's own YAML parser honours lattice.vectors.alat and defaults the units to
# BOHR. A reader that assumed angstrom would hand ASE a cell 1.89x too small
# while Q3 solved the right one, and nothing downstream compares the two.

BOHR_TO_ANGSTROM = 0.52917721092


def _material_with_vectors(rows, **extra):
    vectors = {"a": rows[0], "b": rows[1], "c": rows[2], **extra}
    return {
        "lattice": {"vectors": vectors},
        "basis": {
            "elements": [{"id": 1, "value": "Si"}],
            "coordinates": [{"id": 1, "value": [0.0, 0.0, 0.0]}],
            "units": "crystal",
        },
    }


def _read_material(material, tmp_path):
    path = tmp_path / "material.json"
    path.write_text(json.dumps(material))
    return material_json_to_ase_atoms(path)


def test_unannotated_vectors_are_read_as_bohr(tmp_path):
    """The parser's default, and the case that would silently shrink every cell."""
    atoms = _read_material(
        _material_with_vectors([[10.26, 0, 0], [0, 10.26, 0], [0, 0, 10.26]]), tmp_path
    )

    assert atoms.get_cell()[0][0] == pytest.approx(10.26 * BOHR_TO_ANGSTROM)


def test_angstrom_annotated_vectors_are_taken_at_face_value(tmp_path):
    """What the writer emits, so the round trip stays exact."""
    atoms = _read_material(
        _material_with_vectors([[5.43, 0, 0], [0, 5.43, 0], [0, 0, 5.43]], units="angstrom"),
        tmp_path,
    )

    assert atoms.get_cell()[0][0] == pytest.approx(5.43)


def test_alat_scales_the_rows(tmp_path):
    """Vectors given in units of a lattice constant come out at their true size."""
    atoms = _read_material(
        _material_with_vectors([[1, 0, 0], [0, 1, 0], [0, 0, 1]], alat=5.43, units="angstrom"),
        tmp_path,
    )

    assert atoms.get_cell()[0][0] == pytest.approx(5.43)


def test_vector_units_override_the_lattice_level_annotation(tmp_path):
    """The more specific annotation wins, matching the engine's own precedence."""
    material = _material_with_vectors([[5.43, 0, 0], [0, 5.43, 0], [0, 0, 5.43]], units="angstrom")
    material["lattice"]["units"] = "bohr"

    assert _read_material(material, tmp_path).get_cell()[0][0] == pytest.approx(5.43)


def test_unknown_length_units_are_refused(tmp_path):
    """Guessing would be a silently wrong cell."""
    material = _material_with_vectors([[1, 0, 0], [0, 1, 0], [0, 0, 1]], units="furlong")

    with pytest.raises(ValueError, match="unknown length units"):
        _read_material(material, tmp_path)


def test_esse_constraints_become_fix_cartesian_and_labels_their_element():
    """ESSE ``true`` lets a component move (pw.x if_pos 1); a label's index is dropped."""
    material = {
        "lattice": {
            "vectors": {
                "a": [-2.7, 0.0, 2.7],
                "b": [0.0, 2.7, 2.7],
                "c": [-2.7, 2.7, 0.0],
                "units": "angstrom",
            }
        },
        "basis": {
            "units": "crystal",
            "elements": [{"id": 1, "value": "Si1"}, {"id": 2, "value": "Si2"}],
            "coordinates": [{"id": 1, "value": [0, 0, 0]}, {"id": 2, "value": [0.26, 0.25, 0.25]}],
            "constraints": [
                {"id": 1, "value": [False, False, False]},
                {"id": 2, "value": [True, False, True]},
            ],
        },
    }

    atoms = material_json_to_ase_atoms(material)

    assert atoms.get_chemical_symbols() == ["Si", "Si"]
    assert all(isinstance(constraint, FixCartesian) for constraint in atoms.constraints)
    displaced_positions = atoms.positions + 0.1
    for constraint in atoms.constraints:
        constraint.adjust_positions(atoms, displaced_positions)
    assert np.allclose(displaced_positions - atoms.positions, [[0, 0, 0], [0.1, 0, 0.1]])
