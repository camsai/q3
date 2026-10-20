"""q3.io.interpolation_results.quality — :class:`Quality`, one block of the results file of ``q3_utility interpolate``."""

from __future__ import annotations

from q3.io.interpolation_results.band_distance_report import BandDistanceReport
from q3.io.interpolation_results.hamiltonian_decay import HamiltonianDecay
from q3.io.interpolation_results.ladder_attempt import LadderAttempt
from q3.io.interpolation_results.quality_centers import QualityCenters
from q3.io.interpolation_results.quality_spread import QualitySpread
from q3.io.interpolation_results.results_block import ResultsBlock


class Quality(ResultsBlock):
    """``interpolation.quality``: what the model says about itself, and what the workflow judged."""

    spread: QualitySpread | None = None
    centers: QualityCenters | None = None
    hamiltonian_decay: HamiltonianDecay | None = None
    localization_converged: bool | None = None
    status: str | None = None
    reason: str | None = None
    ladder: list[LadderAttempt] | None = None
    band_distance: BandDistanceReport | None = None
