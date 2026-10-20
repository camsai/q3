"""The roster's structures: the base cells, and the variants each one gets."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from ase import Atoms
from ase.build import bulk
from ase.spacegroup import crystal
from q3.structure.ase_atoms import material_json_to_ase_atoms

from .base_structure import BaseStructure
from .settings import (
    ALLOY_MATERIAL,
    DOUBLED,
    GALLIUM_CELL_PARAMETERS,
    GALLIUM_SITE,
    GALLIUM_SPACE_GROUP,
    LATTICE_CONSTANTS_ANGSTROM,
    PRISTINE,
    RATTLE_FRACTION,
    RATTLED,
    STRAIN,
    STRAINED_MINUS,
    STRAINED_PLUS,
)
from .utils import shortest_bond


def rattled(atoms: Atoms, seed: int) -> Atoms:
    """Every atom moved in a seeded random direction, by between half of RATTLE_FRACTION of the
    shortest bond and all of it."""
    rng = np.random.default_rng(seed)
    directions = rng.normal(size=(len(atoms), 3))
    directions /= np.linalg.norm(directions, axis=1)[:, None]
    lengths = rng.uniform(0.5 * RATTLE_FRACTION, RATTLE_FRACTION, size=len(atoms))
    moved = atoms.copy()
    moved.positions = atoms.positions + directions * (lengths * shortest_bond(atoms))[:, None]
    return moved


def strained(atoms: Atoms, strain: float) -> Atoms:
    """The cell scaled isotropically, atoms carried with it: a volume change and nothing else."""
    scaled = atoms.copy()
    scaled.set_cell(atoms.cell * (1.0 + strain), scale_atoms=True)
    return scaled


def variants(atoms: Atoms, seed: int) -> dict[str, Atoms]:
    """Every variant a base structure can have, by the suffix that names its case."""
    return {
        PRISTINE: atoms,
        RATTLED: rattled(atoms, seed),
        STRAINED_PLUS: strained(atoms, STRAIN),
        STRAINED_MINUS: strained(atoms, -STRAIN),
        DOUBLED: atoms.repeat((2, 1, 1)),
    }


def base_structures(data_directory: Path) -> list[BaseStructure]:
    """The cells the roster is built from, cheapest first.

    The seed offsets are written out rather than counted from the order: they
    decide every random variant, and moving a row must not move a structure.
    """
    constants = LATTICE_CONSTANTS_ANGSTROM
    gallium = crystal(
        "Ga",
        basis=[GALLIUM_SITE],
        spacegroup=GALLIUM_SPACE_GROUP,
        cellpar=list(GALLIUM_CELL_PARAMETERS),
        primitive_cell=True,
    )
    return [
        BaseStructure(
            formula="Si",
            key="silicon",
            atoms=bulk("Si", "diamond", a=constants["Si"]),
            seed_offset=1,
        ),
        BaseStructure(
            formula="GaAs",
            key="gallium_arsenide",
            atoms=bulk("GaAs", "zincblende", a=constants["GaAs"]),
            seed_offset=2,
        ),
        BaseStructure(
            formula="SiC",
            key="silicon_carbide",
            atoms=bulk("SiC", "zincblende", a=constants["SiC"]),
            seed_offset=3,
        ),
        BaseStructure(
            formula="Si3C",
            key="silicon_carbon_alloy",
            atoms=material_json_to_ase_atoms(data_directory / ALLOY_MATERIAL),
            seed_offset=4,
        ),
        BaseStructure(formula="As", key="arsenic", atoms=bulk("As"), seed_offset=5),
        BaseStructure(
            formula="C",
            key="diamond",
            atoms=bulk("C", "diamond", a=constants["C"]),
            seed_offset=6,
        ),
        BaseStructure(
            formula="Cu", key="copper", atoms=bulk("Cu", "fcc", a=constants["Cu"]), seed_offset=7
        ),
        BaseStructure(formula="Ga", key="gallium", atoms=gallium, seed_offset=8),
    ]
