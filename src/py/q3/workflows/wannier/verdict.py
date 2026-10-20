"""q3.workflows.wannier.verdict — :class:`Verdict`, what one evaluation earned at the gate."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from q3.workflows.band_distance import BandDistance
from q3.workflows.wannier.attempt_status import AttemptStatus


class Verdict(BaseModel):
    """What one evaluation earned: the distance, the status and why, and the gauge's state."""

    model_config = ConfigDict(frozen=True)

    distance: BandDistance
    #: ``accept``, or ``retry`` for a model the ladder should try to improve.
    status: AttemptStatus
    reason: str
    localized: bool | None
    localization_converged: bool | None
