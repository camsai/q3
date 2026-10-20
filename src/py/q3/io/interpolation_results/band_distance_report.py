"""q3.io.interpolation_results.band_distance_report — :class:`BandDistanceReport`, one block of the results file of ``q3_utility interpolate``."""

from __future__ import annotations

from q3.io.interpolation_results.results_block import ResultsBlock


class BandDistanceReport(ResultsBlock):
    """``interpolation.quality.band_distance``: the band distance the workflow measured."""

    eta_ev: float | None = None
    eta_maximum_ev: float | None = None
    number_of_states: int | None = None
    reference: str | None = None
    number_of_kpoints: int | None = None
