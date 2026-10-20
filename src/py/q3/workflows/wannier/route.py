"""q3.workflows.wannier.route — :class:`Route`, what the ladder does about a failed evaluation."""

from __future__ import annotations

from enum import StrEnum


class Route(StrEnum):
    """What the ladder tries next; the ``error:<route>`` statuses of its attempts."""

    #: The disentanglement ran out of iterations; a looser tolerance is what to try.
    UNCONVERGED = "unconverged"
    #: The record itself cannot support what was asked -- no overlaps, no
    #: projectabilities, a two-point axis -- and no threshold fixes that.
    LOCALIZATION = "localization"
    #: The subspace could not be filled, or the gauge went singular; vary the thresholds.
    INFEASIBLE = "infeasible"
    #: Nothing the ladder knows what to do about; it rejects.
    UNKNOWN = "unknown"
