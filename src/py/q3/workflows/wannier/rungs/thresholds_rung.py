"""q3.workflows.wannier.rungs.thresholds_rung — :class:`ThresholdsRung`: walk the published grid."""

from __future__ import annotations

from q3.workflows.wannier.attempt_status import AttemptStatus
from q3.workflows.wannier.ladder_state import LadderState
from q3.workflows.wannier.rung_name import RungName
from q3.workflows.wannier.rungs.rung import Rung
from q3.workflows.wannier.utils import threshold_walk


class ThresholdsRung(Rung):
    """The published projectability grid, until one pair accepts or the cap is spent.

    The pairs are tried against the settings the earlier rungs promoted, and
    are not promoted themselves: a pair that did not accept is not one to
    build on.
    """

    name = RungName.THRESHOLDS

    def applies(self, state: LadderState) -> bool:
        return not state.accepted and (
            state.evaluated or state.status is AttemptStatus.ERROR_INFEASIBLE
        )

    def attempt(self, state: LadderState) -> LadderState:
        walk = threshold_walk(state.tried)
        cap = self.evaluation.maximum_threshold_attempts
        if cap is not None:
            walk = walk[: max(0, int(cap))]
        for minimum, maximum in walk:
            state = state.model_copy(update={"tried": state.tried | {(minimum, maximum)}})
            settings = state.settings.updated(
                projectability_minimum=minimum, projectability_maximum=maximum
            )
            state = self.evaluation.attempt(state, self.name, settings)
            if state.accepted:
                return state
            # A pair the classifier refuses is one to walk past; any other failure ends the walk.
            if not state.evaluated and state.status is not AttemptStatus.ERROR_INFEASIBLE:
                return state
        return state
