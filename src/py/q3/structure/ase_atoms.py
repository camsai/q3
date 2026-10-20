"""q3.structure.ase_atoms — Convert Q3 material JSON into an ASE ``Atoms`` object.

This adapter is the single place that knows how to turn the ESSE-format
material JSON used by Q3 examples into an ASE structure, and it deliberately
sits *below* both ``q3.calculators`` and ``q3.workflows`` in the package
layering (see AGENTS.md § Python Package Layering).

It used to live in ``q3.workflows.relax``, which forced ``q3.calculators``
(tier 2) to import a workflow (tier 3) and ``q3.workflows.relax`` to import
``q3.calculators`` — a genuine import cycle that was being worked around with
function-level imports at four call sites.  Owning the adapter here breaks the
cycle by construction rather than by deferring it to call time.

Public functions
----------------
material_json_to_ase_atoms(material)
    Build an ASE ``Atoms`` object from a Q3 material JSON file or document.

ase_atoms_to_material_json(atoms, template_material=None)
    Build a Q3 material JSON document from an ASE ``Atoms`` object.
"""

from __future__ import annotations

import copy
import json
import re
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.constraints import FixCartesian

from q3.structure.lattice import primitive_lattice_vectors_angstrom


def material_json_to_ase_atoms(material: str | Path | dict):
    """Build an ASE Atoms object from a Q3 material JSON file or document.

    Handles the ESSE-format material JSON used by Q3 examples.  A site's
    ``basis.constraints`` entry becomes a ``FixCartesian`` on the components
    it holds fixed (ESSE ``true`` lets a component move, as pw.x ``if_pos 1``).

    IMPORTANT: The cell vectors are constructed using the same Bravais-lattice
    conventions as Q3 (see src/layer_1_structure/crystal/lattice.hpp).  This
    ensures that atomic forces returned by Q3 in its Cartesian frame are
    consistent with the Cartesian frame ASE uses for the atoms — a prerequisite
    for BFGS to move atoms in the correct direction.
    """
    if not isinstance(material, dict):
        with open(material) as file_handle:
            material = json.load(file_handle)

    lattice = material["lattice"]
    basis = material["basis"]

    # The cell is Q3's own -- QE's ibrav orientations for the cubic types,
    # cell parameters otherwise, explicit vectors when the material carries
    # them. q3.structure.lattice owns that construction, so this module and
    # the k-path read the same cell.
    cell = primitive_lattice_vectors_angstrom(lattice)

    atoms = _atoms_from_basis(Atoms, basis, cell)
    site_ids = sorted(entry["id"] for entry in basis["coordinates"])
    site_index = {site_id: index for index, site_id in enumerate(site_ids)}
    atoms.set_constraint(
        [
            FixCartesian(site_index[entry["id"]], mask=~np.asarray(entry["value"], dtype=bool))
            for entry in basis.get("constraints", [])
            if not all(entry["value"])
        ]
    )
    return atoms


def _atoms_from_basis(atoms_class, basis: dict, cell):
    """Assemble an ASE Atoms object from an ESSE basis and a Cartesian cell.

    Shared by every lattice branch of ``material_json_to_ase_atoms`` so the
    site ordering and the crystal/cartesian units rule are stated once.
    """
    elements = {item["id"]: item["value"] for item in basis["elements"]}
    coords = {item["id"]: item["value"] for item in basis["coordinates"]}

    # A label may carry a sublattice index (Cr1, Cr2); ASE takes the element.
    symbols = [_symbol_of_species_label(elements[i]) for i in sorted(elements)]
    positions = np.array([coords[i] for i in sorted(coords)])

    units = basis.get("units", "crystal")
    if units == "crystal":
        return atoms_class(symbols=symbols, scaled_positions=positions, cell=cell, pbc=True)
    return atoms_class(symbols=symbols, positions=positions, cell=cell, pbc=True)


def _species_labels(template_basis: dict, symbols: list[str]) -> list[str]:
    """Choose per-site species labels, keeping a template's where they still apply.

    ASE only knows chemical symbols, but a Q3 material may label sites with
    per-sublattice species — an antiferromagnet's ``Cr1``/``Cr2``, which carry
    opposite starting magnetizations and have their own ``method.pseudopotentials``
    entries.  Writing ASE's symbols over those would collapse both sublattices
    to ``Cr``, leaving the pseudopotential lookup with no matching key and the
    magnetic configuration destroyed — a relaxation that used to run
    antiferromagnetically would silently stop being one.

    So a template label is kept when it still describes the same site, meaning
    the counts match and the label is that site's own symbol with an optional
    sublattice index — ``Cr1`` for a ``Cr``.  The comparison strips the index
    and matches the symbol exactly rather than testing a prefix: ``"Cr1"``
    starts with ``"C"``, so a prefix test would write a carbon atom out as
    chromium, and every single-letter symbol (C, S, N, O, B, P, H, K, V, W, Y,
    F, I, U) collides that way.

    A label that fails the test is not describing this structure any more — a
    substituted or reordered basis — and ASE's symbols are the trustworthy
    answer instead.
    """
    template_labels = [entry["value"] for entry in template_basis.get("elements", [])]
    if len(template_labels) != len(symbols):
        return symbols
    if all(
        _symbol_of_species_label(label) == symbol for label, symbol in zip(template_labels, symbols)
    ):
        return template_labels
    return symbols


def _symbol_of_species_label(label: str) -> str:
    """Strip a sublattice index from a species label: ``Cr1`` -> ``Cr``.

    Returns the label unchanged when it does not have that shape, so anything
    unrecognized fails the equality test above rather than being coerced.
    """
    match = re.fullmatch(r"([A-Za-z]{1,3})\d*", label)
    return match.group(1) if match else label


def ase_atoms_to_material_json(atoms, template_material: dict | None = None) -> dict:
    """Build a Q3 material JSON document from an ASE Atoms object.

    The inverse of :func:`material_json_to_ase_atoms`, and the single place the
    ASE-to-Q3 direction is defined.  The cell is written as explicit Cartesian
    ``lattice.vectors`` in angstrom rather than being fitted back to a Bravais
    ``type`` plus lattice constant: the raw rows are exact for any cell,
    including one that has relaxed away from the symmetry it started in, and
    they are what Q3's own YAML parser reads for a free lattice.

    Parameters
    ----------
    atoms:
        The ASE ``Atoms`` object to convert.  Positions are written as crystal
        (fractional) coordinates, matching the ESSE convention Q3 examples use.
    template_material:
        An existing material document to copy non-structural fields from (name,
        metadata, per-site ids, species labels).  When given, its site ids and
        species labels are preserved so a material written back carries the
        identity it arrived with; when omitted, sites are numbered from 1 and
        labelled with ASE's chemical symbols.

    Returns
    -------
    dict
        An ESSE-format material document with ``lattice`` and ``basis`` keys.
    """
    material = copy.deepcopy(template_material) if template_material else {}

    scaled_positions = atoms.get_scaled_positions().tolist()
    symbols = list(atoms.get_chemical_symbols())

    existing_coordinates = material.get("basis", {}).get("coordinates", [])
    site_ids = [entry["id"] for entry in existing_coordinates]
    if len(site_ids) != len(scaled_positions):
        site_ids = list(range(1, len(scaled_positions) + 1))

    basis = material.setdefault("basis", {})
    basis["elements"] = [
        {"id": site_id, "value": label}
        for site_id, label in zip(site_ids, _species_labels(basis, symbols))
    ]
    basis["coordinates"] = [
        {"id": site_id, "value": position} for site_id, position in zip(site_ids, scaled_positions)
    ]
    basis["units"] = "crystal"

    cell_angstrom = atoms.get_cell()[:].tolist()
    material["lattice"] = {
        "vectors": {
            "a": cell_angstrom[0],
            "b": cell_angstrom[1],
            "c": cell_angstrom[2],
            "units": "angstrom",
        }
    }
    return material
