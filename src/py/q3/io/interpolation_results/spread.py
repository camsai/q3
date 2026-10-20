"""q3.io.interpolation_results.spread — :class:`Spread`, one block of the results file of ``q3_utility interpolate``."""

from __future__ import annotations

from q3.io.interpolation_results.results_block import ResultsBlock


class Spread(ResultsBlock):
    """``interpolation.spread``: the Marzari-Vanderbilt spread at the gauge the model has.

    Omega_I and ``localized`` are always written; the rest whenever the record
    carried the neighbour overlaps that define them and the mesh had no
    two-point axis, localised or not.
    """

    gauge_invariant_bohr_squared: float | None = None
    localized: bool | None = None
    total_bohr_squared: float | None = None
    guiding_centers: str | None = None
    trial_orbital_centers_bohr: list[list[float]] | None = None
    initial_minimizable_bohr_squared: float | None = None
    centers_bohr: list[list[float]] | None = None
    spreads_bohr_squared: list[float] | None = None
    localization_iterations: int | None = None
    localization_converged: bool | None = None
    localization_gradient_norm: float | None = None
