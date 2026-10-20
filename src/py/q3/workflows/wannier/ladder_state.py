"""q3.workflows.wannier.ladder_state — :class:`LadderState`, what the ladder carries from rung to rung.

The rungs are not independent: each judges the attempt the last one left, and
a rung that helps promotes its settings so later rungs inherit them. That
shared state is one frozen value, which a rung returns a changed copy of
rather than mutating in place -- so what a rung changed is exactly what its
``model_copy`` names, and the attempt history cannot alias across mesh passes.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from q3.io.interpolation_results import InterpolationResults, LadderAttempt
from q3.io.interpolation_settings import EvaluationSettings
from q3.workflows.band_distance import BandDistance
from q3.workflows.wannier.attempt_status import AttemptStatus
from q3.workflows.wannier.diagnostic import Diagnostic
from q3.workflows.wannier.settings import DEFAULT_DISENTANGLEMENT_TOLERANCE


class LadderState(BaseModel):
    """One mesh's worth of ladder: the last attempt, its results, and what the next rung starts from.

    A value, not a place: every change is a new state from ``model_copy``, and
    ``attempts`` is a tuple so no rung can append to a list another mesh pass
    still holds. What the rungs share is exactly the fields here.
    """

    model_config = ConfigDict(frozen=True)

    #: The most recent attempt, which is what every rung's condition tests; None before the first.
    record: LadderAttempt | None = None
    #: Its results file, or None when the evaluation raised rather than judged.
    results: InterpolationResults | None = None
    #: The band distance the gate measured on those results; None with them.
    distance: BandDistance | None = None
    #: The engine's code for the last attempt, when it raised and printed one.
    diagnostic: Diagnostic | None = None
    #: The settings the next rung starts from; a rung that helps promotes its own.
    settings: EvaluationSettings
    #: Threshold pairs already spent, so the walk never repeats one.
    tried: frozenset[tuple[float, float]]
    #: The disentanglement tolerance every attempt on this record runs under: the
    #: engine's own until one runs out of iterations, then the one that attempt's
    #: spreads called for. It belongs to the record, not to any rung's choice, so
    #: it is not in `settings`.
    disentanglement_tolerance: float = DEFAULT_DISENTANGLEMENT_TOLERANCE
    #: Set by a rung that ends the whole ladder rather than passing it on.
    stop_reason: str | None = None
    #: Every attempt of every mesh, in order: the history a rejection reports.
    attempts: tuple[LadderAttempt, ...] = ()

    @property
    def status(self) -> AttemptStatus | None:
        """What became of the most recent attempt; None before the first."""
        return AttemptStatus(self.record.status) if self.record is not None else None

    @property
    def accepted(self) -> bool:
        return self.status is AttemptStatus.ACCEPT

    @property
    def evaluated(self) -> bool:
        """Whether the last attempt produced a model at all, good enough or not."""
        return self.results is not None

    @property
    def stalled(self) -> bool:
        """A model the gate sent back whose gauge did not reach a stationary point."""
        return (
            self.evaluated
            and self.status is AttemptStatus.RETRY
            and not self.record.localization_converged
        )

    @property
    def short_of_the_gate(self) -> bool:
        """A model the gate sent back although its gauge is stationary: the distance is the matter."""
        return (
            self.evaluated
            and self.status is AttemptStatus.RETRY
            and bool(self.record.localization_converged)
        )

    def with_attempt(
        self,
        record: LadderAttempt,
        results: InterpolationResults | None,
        diagnostic: Diagnostic | None,
        distance: BandDistance | None = None,
    ) -> LadderState:
        """The state after one more evaluation: its record, its results, the history one longer."""
        return self.model_copy(
            update={
                "record": record,
                "results": results,
                "distance": distance,
                "diagnostic": diagnostic,
                "attempts": self.attempts + (record,),
            }
        )

    def adopting(self, settings: EvaluationSettings) -> LadderState:
        """The same state with *settings* promoted: what later rungs start from."""
        return self.model_copy(update={"settings": settings})

    def stopping(self, reason: str) -> LadderState:
        """The same state, ending the ladder: no later rung runs."""
        return self.model_copy(update={"stop_reason": reason})
