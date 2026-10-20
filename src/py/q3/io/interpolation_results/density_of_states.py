"""q3.io.interpolation_results.density_of_states — :class:`DensityOfStates`, one block of the results file of ``q3_utility interpolate``."""

from __future__ import annotations

from q3.io.interpolation_results.results_block import ResultsBlock


class DensityOfStates(ResultsBlock):
    """``density_of_states``: the interpolated DOS on the dense mesh, when one was asked for."""

    mesh: list[int] | None = None
    broadening_ry: float | None = None
    energies_ry: list[float] | None = None
    dos: list[float] | None = None
