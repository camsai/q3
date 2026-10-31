"""q3.workflows.wannier.rungs.window_rung — :class:`WindowRung`: lower or widen the frozen window."""

from __future__ import annotations

from q3.workflows.wannier.attempt_status import AttemptStatus
from q3.workflows.wannier.diagnostic import Diagnostic
from q3.workflows.wannier.ladder_state import LadderState
from q3.workflows.wannier.rung_name import RungName
from q3.workflows.wannier.rungs.rung import Rung
from q3.workflows.wannier.settings import (
    DEFAULT_FROZEN_WINDOW_EV,
    LOWERED_WINDOWS_EV,
    RAISED_WINDOW_STEPS_EV,
)
from q3.workflows.wannier.utils import window_holds_unrepresentable_states


class WindowRung(Rung):
    """The window itself, in whichever direction the failure asks for.

    Too many frozen states is the classifier's own refusal and lowering is the
    answer it names. A converged gauge that still misses the gate wants the
    opposite: the band distance weighs states up to E_F + 2 eV, and a state at
    the window's edge is reproduced only when the states just above it are
    frozen with it -- silicon at 12x12x12 goes 220 -> 14 -> 5 meV with the
    window at E_F + 0, 2 and 3 eV, while the threshold walk moves it 221 -> 219.
    Widening is skipped where the window already holds states the dataset
    cannot represent, since wider means more of them. The two directions are
    one rung because a pass lowers or widens, never both.
    """

    name = RungName.WINDOW

    def applies(self, state: LadderState) -> bool:
        return self._overfull(state) or self._short_of_the_gate(state)

    def attempt(self, state: LadderState) -> LadderState:
        if self._overfull(state):
            return self._lower(state)
        return self._widen(state)

    @staticmethod
    def _overfull(state: LadderState) -> bool:
        """The classifier's refusal of a window that freezes more bands than there are functions."""
        return (
            state.status is AttemptStatus.ERROR_INFEASIBLE
            and state.diagnostic is Diagnostic.FROZEN_WINDOW_OVERFULL
        )

    @staticmethod
    def _short_of_the_gate(state: LadderState) -> bool:
        return state.short_of_the_gate and not window_holds_unrepresentable_states(state.results)

    def _lower(self, state: LadderState) -> LadderState:
        for lowered_ev in LOWERED_WINDOWS_EV:
            settings = state.settings.updated(frozen_window_above_fermi_ev=lowered_ev)
            state = self.evaluation.attempt(state, self.name, settings)
            if state.evaluated:
                return state.adopting(settings)
        return state

    def _widen(self, state: LadderState) -> LadderState:
        verdict = state.results.construction.disentanglement
        applied = float(
            verdict.frozen_window_above_fermi_ev
            if verdict is not None and verdict.frozen_window_above_fermi_ev is not None
            else DEFAULT_FROZEN_WINDOW_EV
        )
        best_eta = state.record.eta_ev
        for step in RAISED_WINDOW_STEPS_EV:
            settings = state.settings.updated(frozen_window_above_fermi_ev=applied + step)
            wider = self.evaluation.attempt(state, self.name, settings)
            improved = (
                wider.evaluated
                and wider.record.localization_converged
                and wider.record.eta_ev is not None
                and (best_eta is None or wider.record.eta_ev < best_eta)
            )
            if not wider.accepted and not improved:
                # The attempt is history, but the model before it stays the one
                # judged: only its record of what was tried is kept.
                return state.model_copy(
                    update={
                        "attempts": wider.attempts,
                        "disentanglement_tolerance": wider.disentanglement_tolerance,
                    }
                )
            state = wider.adopting(settings)
            best_eta = state.record.eta_ev
            if state.accepted:
                return state
        return state
