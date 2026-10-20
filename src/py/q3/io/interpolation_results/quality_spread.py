"""q3.io.interpolation_results.quality_spread — :class:`QualitySpread`, one block of the results file of ``q3_utility interpolate``."""

from __future__ import annotations

from q3.io.interpolation_results.results_block import ResultsBlock


class QualitySpread(ResultsBlock):
    """``interpolation.quality.spread``: statistics of the per-function spreads."""

    median_bohr_squared: float | None = None
    variance_bohr_squared: float | None = None
    maximum_bohr_squared: float | None = None
    outliers: list[int] | None = None
    above_ten_angstrom_squared: list[int] | None = None
