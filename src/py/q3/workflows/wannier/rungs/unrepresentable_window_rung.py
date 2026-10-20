"""q3.workflows.wannier.rungs.unrepresentable_window_rung — :class:`UnrepresentableWindowRung`."""

from __future__ import annotations

from q3.workflows.wannier.ladder_state import LadderState
from q3.workflows.wannier.rung_name import RungName
from q3.workflows.wannier.rungs.rung import Rung
from q3.workflows.wannier.settings import OCCUPIED_MANIFOLD_WINDOW_EV
from q3.workflows.wannier.utils import window_holds_unrepresentable_states


class UnrepresentableWindowRung(Rung):
    """A metal whose window holds states its dataset cannot reach: freeze the occupied manifold.

    The classifier already keeps a state out of the freeze where its
    projectability is below the minimum, but one it can barely reach is frozen
    where it is above it, and a Wannier function built on such a state does not
    localise. Copper's sp band on a dataset without a 4p shell stalls exactly
    so, which is what this rung is for.
    """

    name = RungName.WINDOW

    def applies(self, state: LadderState) -> bool:
        return state.stalled and window_holds_unrepresentable_states(state.results)

    def attempt(self, state: LadderState) -> LadderState:
        settings = state.settings.updated(frozen_window_above_fermi_ev=OCCUPIED_MANIFOLD_WINDOW_EV)
        state = self.evaluation.attempt(state, self.name, settings)
        if state.evaluated and state.record.localization_converged:
            return state.adopting(settings)
        return state
