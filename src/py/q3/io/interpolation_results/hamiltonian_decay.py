"""q3.io.interpolation_results.hamiltonian_decay — :class:`HamiltonianDecay`, one block of the results file of ``q3_utility interpolate``."""

from __future__ import annotations

from q3.io.interpolation_results.results_block import ResultsBlock


class HamiltonianDecay(ResultsBlock):
    """``interpolation.quality.hamiltonian_decay``: how far H(R) decayed by the edge of its domain."""

    outermost_shell_distance_bohr: float | None = None
    outermost_shell_size: int | None = None
    outermost_shell_maximum_ry: float | None = None
    largest_hopping_ry: float | None = None
    ratio: float | None = None
