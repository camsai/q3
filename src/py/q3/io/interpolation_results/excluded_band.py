"""q3.io.interpolation_results.excluded_band — :class:`ExcludedBand`, one block of the results file of ``q3_utility interpolate``."""

from __future__ import annotations

from q3.io.interpolation_results.results_block import ResultsBlock


class ExcludedBand(ResultsBlock):
    """One band the singular-gauge rule kept out of the freeze, one-based, at ``kpoints`` k-points."""

    band: int | None = None
    kpoints: int | None = None
