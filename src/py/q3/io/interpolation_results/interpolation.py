"""q3.io.interpolation_results.interpolation — :class:`Interpolation`, one block of the results file of ``q3_utility interpolate``."""

from __future__ import annotations

from q3.io.interpolation_results.disentanglement import Disentanglement
from q3.io.interpolation_results.quality import Quality
from q3.io.interpolation_results.results_block import ResultsBlock
from q3.io.interpolation_results.spread import Spread


class Interpolation(ResultsBlock):
    """``interpolation``: the model that was evaluated -- how it was built, its size, its quality."""

    method: str | None = None
    description: str | None = None
    record: str | None = None
    number_of_bands: int | None = None
    total_bloch_sum_weight: float | None = None
    diagnostics: dict[str, float] | None = None
    spread: Spread | None = None
    disentanglement: Disentanglement | None = None
    quality: Quality | None = None
