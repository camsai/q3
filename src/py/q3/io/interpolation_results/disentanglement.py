"""q3.io.interpolation_results.disentanglement — :class:`Disentanglement`, one block of the results file of ``q3_utility interpolate``."""

from __future__ import annotations

from q3.io.interpolation_results.excluded_band import ExcludedBand
from q3.io.interpolation_results.results_block import ResultsBlock


class Disentanglement(ResultsBlock):
    """``interpolation.disentanglement``: the classifier's verdict, present when the model was disentangled."""

    metallic: bool | None = None
    bands_crossing_fermi: int | None = None
    frozen_window_above_fermi_ev: float | None = None
    frozen_window_automatic: bool | None = None
    bands_excluded_from_freeze: list[ExcludedBand] | None = None
