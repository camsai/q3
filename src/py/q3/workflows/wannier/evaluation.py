"""q3.workflows.wannier.evaluation — :class:`Evaluation`, the one way a saved record is evaluated.

The captured record and what it is judged against, fixed for one mesh pass;
every rung evaluates through it.
"""

from __future__ import annotations

import numpy as np
from pydantic import BaseModel, ConfigDict

from q3.errors import Q3Error, Q3RuntimeError
from q3.io.interpolation_results import InterpolationResults, LadderAttempt
from q3.io.interpolation_settings import EvaluationSettings
from q3.workflows.interpolation import EngineChain, Grid, KPoint
from q3.workflows.wannier.accept_gate import AcceptGate
from q3.workflows.wannier.attempt_status import AttemptStatus
from q3.workflows.wannier.diagnostic import Diagnostic
from q3.workflows.wannier.ladder_state import LadderState
from q3.workflows.wannier.rung_name import RungName
from q3.workflows.wannier.settings import (
    DEFAULT_DISENTANGLEMENT_TOLERANCE,
    DEFAULT_PROJECTABILITY_MAXIMUM,
    DEFAULT_PROJECTABILITY_MINIMUM,
    REFERENCE_OFF_THE_MESH,
    REFERENCE_ON_A_PATH,
)
from q3.workflows.wannier.utils import loosened_disentanglement_tolerance


class Evaluation(BaseModel):
    """The captured record and what it is judged against, fixed for one mesh pass.

    Evaluating an attempt needs more than the ladder's state: the chain that
    runs the utility, the reference bands, the Fermi level the window is
    measured from, the gate, and the cap on the threshold walk. All of it is
    fixed once the mesh is captured, so it lives here rather than in the state
    the rungs pass on -- and every rung reads it through :meth:`attempt`, the
    one way a record is evaluated.
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    chain: EngineChain
    mesh: Grid
    #: What the trial-orbital rule counted; a model of another size is refused.
    wannier_functions: int
    reference_set: list[KPoint]
    #: The reference bands at `reference_set`, (k-point, band), in electronvolts.
    reference_ev: np.ndarray
    #: The capture's Fermi level: the one the classifier measures the frozen window from.
    fermi_energy_ev: float
    gate: AcceptGate
    #: Whether the reference is a path the caller asked for rather than the off-mesh set.
    fixed_reference: bool
    dos_mesh: Grid | None = None
    dos_broadening_ry: float | None = None
    #: A cap on the threshold walk, for a campaign that would rather report than spend an hour.
    maximum_threshold_attempts: int | None = None

    def initial_state(self, attempts: tuple[LadderAttempt, ...]) -> LadderState:
        """The state before the first attempt on this mesh: the defaults, and the history so far."""
        return LadderState(
            settings=EvaluationSettings(
                localization="auto",
                projectability_minimum=DEFAULT_PROJECTABILITY_MINIMUM,
                projectability_maximum=DEFAULT_PROJECTABILITY_MAXIMUM,
            ),
            tried=frozenset({(DEFAULT_PROJECTABILITY_MINIMUM, DEFAULT_PROJECTABILITY_MAXIMUM)}),
            attempts=attempts,
        )

    def attempt(
        self, state: LadderState, rung: RungName, settings: EvaluationSettings
    ) -> LadderState:
        """One rung's evaluation under *settings*, judged and recorded.

        The state returned carries the record always and the results only when
        the evaluation produced a model, so a rung can tell "ran and was not
        good enough" from "could not run at all" -- which is the distinction
        the next rung is chosen on. Every attempt joins the history, whether it
        ran or raised.

        An evaluation whose disentanglement ran out of iterations is retried
        once, straight away, as rung ``unconverged``, under the tolerance its
        own spreads call for (:func:`loosened_disentanglement_tolerance`),
        whichever rung it came from. The tolerance is kept for every later
        attempt on the record, which disentangles the same overlaps.
        """
        state, failure = self._evaluate(state, rung, settings)
        if state.status is not AttemptStatus.ERROR_UNCONVERGED or failure is None:
            return state
        loosened = loosened_disentanglement_tolerance(failure.text, state.disentanglement_tolerance)
        if loosened is None:
            return state
        loosened_state = state.model_copy(update={"disentanglement_tolerance": loosened})
        return self._evaluate(loosened_state, RungName.UNCONVERGED, settings)[0]

    def _evaluate(
        self, state: LadderState, rung: RungName, settings: EvaluationSettings
    ) -> tuple[LadderState, Q3RuntimeError | None]:
        """The state after one evaluation, and the failure when it raised instead of judging."""
        evaluated = settings
        if state.disentanglement_tolerance != DEFAULT_DISENTANGLEMENT_TOLERANCE:
            evaluated = settings.updated(
                disentanglement_convergence_tolerance=state.disentanglement_tolerance
            )
        try:
            results = self.chain.evaluate(
                self.reference_set,
                evaluated.updated(
                    dos_mesh=list(self.dos_mesh) if self.dos_mesh is not None else None,
                    disentanglement="projectability",
                ),
                dos_broadening_ry=self.dos_broadening_ry,
            )
        except Q3RuntimeError as error:
            diagnostic = Diagnostic.of(error)
            record = LadderAttempt(
                rung=rung,
                grid=list(self.mesh),
                settings=evaluated.to_block(),
                status=AttemptStatus.of_failure(diagnostic),
                reason=error.error_line,
            )
            return state.with_attempt(record, None, diagnostic), error
        self._require_the_counted_functions(results)
        verdict = self.gate.judge(results, self.reference_ev, self.fermi_energy_ev)
        record = LadderAttempt(
            rung=rung,
            grid=list(self.mesh),
            settings=evaluated.to_block(),
            status=verdict.status,
            reason=verdict.reason,
            eta_ev=verdict.distance.eta,
            eta_maximum_ev=verdict.distance.eta_maximum,
            localized=verdict.localized,
            localization_converged=verdict.localization_converged,
        )
        results.quality_block().band_distance = verdict.distance.to_report(
            reference=REFERENCE_ON_A_PATH if self.fixed_reference else REFERENCE_OFF_THE_MESH,
            number_of_kpoints=len(self.reference_set),
        )
        return state.with_attempt(record, results, None, verdict.distance), None

    def _require_the_counted_functions(self, results: InterpolationResults) -> None:
        """Refuse a model with a different number of functions than the rule counted.

        The band window, the shells the capture was told to use and the count
        in the report all come from the rule; the engine builds from the shells
        it was told. If the two ever read a dataset differently, every number
        downstream describes a model other than the one judged.
        """
        built = results.interpolation.number_of_bands if results.interpolation else None
        if built is not None and int(built) != self.wannier_functions:
            raise Q3Error(
                f"the engine built {int(built)} Wannier functions where the trial-orbital rule "
                f"counted {self.wannier_functions}; the capture and q3.workflows.wannier read the "
                "datasets' PP_CHI shells differently"
            )
