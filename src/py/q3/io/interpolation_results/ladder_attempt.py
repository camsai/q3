"""q3.io.interpolation_results.ladder_attempt — :class:`LadderAttempt`, one block of the results file of ``q3_utility interpolate``."""

from __future__ import annotations

from typing import Any

from pydantic import ConfigDict, model_serializer

from q3.io.interpolation_results.results_block import ResultsBlock


class LadderAttempt(ResultsBlock):
    """``interpolation.quality.ladder[]``: one evaluation of the saved record, and what was made of it.

    Written by the workflow, so its fields are the writer's contract: the five
    that describe the attempt are required, and the four that judge it are
    null for an evaluation that raised. Built once, after the evaluation, and
    frozen: an attempt is what happened, not a form to fill in.
    """

    model_config = ConfigDict(extra="allow", frozen=True)

    rung: str
    grid: list[int]
    settings: dict[str, Any]
    status: str
    reason: str
    eta_ev: float | None = None
    eta_maximum_ev: float | None = None
    localized: bool | None = None
    localization_converged: bool | None = None

    @model_serializer(mode="wrap")
    def _with_its_nulls(self, handler) -> dict[str, Any]:
        """All nine keys, nulls included, however the document around it is dumped.

        A raised attempt has no band distance, and the file says so with a
        null a reader can tell from "not recorded". Dumping the results file
        drops absent fields (``exclude_none``), which would drop these too;
        what the attempt declares is put back, so no caller has to patch the
        ladder in by hand.
        """
        document = handler(self)
        return {name: getattr(self, name) for name in type(self).model_fields} | {
            name: value for name, value in document.items() if name not in type(self).model_fields
        }
