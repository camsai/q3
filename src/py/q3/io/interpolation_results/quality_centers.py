"""q3.io.interpolation_results.quality_centers — :class:`QualityCenters`, one block of the results file of ``q3_utility interpolate``."""

from __future__ import annotations

from q3.io.interpolation_results.results_block import ResultsBlock


class QualityCenters(ResultsBlock):
    """``interpolation.quality.centers``: the Wannier centres against the atoms."""

    nearest_atom_distance_bohr: list[float] | None = None
    nearest_bond_midpoint_distance_bohr: list[float] | None = None
    shortest_bond_bohr: float | None = None
