"""q3.workflows.wannier.rungs.stalled_rung — :class:`StalledRung`: demand the localisation."""

from __future__ import annotations

from q3.workflows.wannier.attempt_status import AttemptStatus
from q3.workflows.wannier.ladder_state import LadderState
from q3.workflows.wannier.rung_name import RungName
from q3.workflows.wannier.rungs.rung import Rung
from q3.workflows.wannier.settings import STALLED_LOCALIZATION_ITERATIONS


class StalledRung(Rung):
    """A converged model whose gauge did not localise: demand it, with twice the budget.

    A demand the localiser refuses -- a record without overlaps, a two-point
    axis -- ends the whole ladder, since no threshold changes the record.
    """

    name = RungName.STALLED

    def applies(self, state: LadderState) -> bool:
        return state.stalled

    def attempt(self, state: LadderState) -> LadderState:
        settings = state.settings.updated(
            localization="maximal",
            localization_maximum_iterations=STALLED_LOCALIZATION_ITERATIONS,
        )
        state = self.evaluation.attempt(state, self.name, settings)
        if state.status is AttemptStatus.ERROR_LOCALIZATION:
            return state.stopping(state.record.reason)
        return state.adopting(settings)
