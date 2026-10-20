"""q3.workflows.wannier.attempt_status — :class:`AttemptStatus`, what became of one attempt of the ladder."""

from __future__ import annotations

from enum import StrEnum

from q3.workflows.wannier.diagnostic import Diagnostic
from q3.workflows.wannier.route import Route


class AttemptStatus(StrEnum):
    """The ``status`` of a ladder attempt, as the results file and the returned history spell it.

    An evaluation that produced a model is judged ``accept`` or ``retry``; one
    that raised is ``error:<route>``, the route being what the ladder does
    about that failure; and the mesh rung, which evaluates nothing, records
    that it sent the run back to capture.
    """

    ACCEPT = "accept"
    RETRY = "retry"
    RECAPTURE = "recapture"
    ERROR_UNCONVERGED = "error:" + Route.UNCONVERGED
    ERROR_LOCALIZATION = "error:" + Route.LOCALIZATION
    ERROR_INFEASIBLE = "error:" + Route.INFEASIBLE
    ERROR_UNKNOWN = "error:" + Route.UNKNOWN

    @classmethod
    def of_failure(cls, diagnostic: Diagnostic | None) -> AttemptStatus:
        """The status of an evaluation that raised: its code's route, or unknown without a code."""
        route = diagnostic.route if diagnostic is not None else Route.UNKNOWN
        return cls("error:" + route)
