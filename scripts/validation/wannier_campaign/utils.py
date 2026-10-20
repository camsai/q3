"""Helpers with no campaign state: a bond length, an energy for the table, a table cell."""

from __future__ import annotations

from ase import Atoms
from ase.neighborlist import neighbor_list
from q3.workflows.wannier.settings import MILLIELECTRONVOLT_PER_ELECTRONVOLT

#: How far past the shortest cell vector the neighbour search reaches. An atom's
#: own periodic image sits at that distance, so the search always finds a bond,
#: a one-atom cell's included.
NEIGHBOR_SEARCH_MARGIN = 1.01


def shortest_bond(atoms: Atoms) -> float:
    """The shortest interatomic distance, periodic images included."""
    cutoff = NEIGHBOR_SEARCH_MARGIN * float(min(atoms.cell.lengths()))
    return float(neighbor_list("d", atoms, cutoff).min())


def millielectronvolts(value_ev: float | None, digits: int) -> float | None:
    """An energy in eV as a rounded number of meV, or None where there is none."""
    if value_ev is None:
        return None
    return round(MILLIELECTRONVOLT_PER_ELECTRONVOLT * value_ev, digits)


def table_cell(value, digits: int = 3) -> str:
    """A table cell: an em dash where a failed row produced nothing."""
    if value is None:
        return "—"
    return f"{value:.{digits}g}" if isinstance(value, float) else str(value)
