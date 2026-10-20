"""q3.io.interpolation_results.results_block — :class:`ResultsBlock`, one block of the results file of ``q3_utility interpolate``."""

from __future__ import annotations

from mat3ra.code.entity import InMemoryEntityPydantic
from pydantic import ConfigDict


class ResultsBlock(InMemoryEntityPydantic):
    """A block of the results file: unknown keys kept, so a newer file round-trips.

    The base class is the org's, for ``to_dict``/``to_json``/``clone``. Its
    ``to_dict`` strips ``None`` recursively, so a block whose nulls are data
    is dumped with ``model_dump`` instead (see :class:`LadderAttempt`).
    """

    model_config = ConfigDict(extra="allow")
