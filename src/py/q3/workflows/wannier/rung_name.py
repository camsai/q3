"""q3.workflows.wannier.rung_name — :class:`RungName`, how the attempts of the ladder are labelled."""

from __future__ import annotations

from enum import StrEnum


class RungName(StrEnum):
    """The ``rung`` of a ladder attempt, as the results file and the returned history spell it.

    Two rungs share ``window``: freezing the occupied manifold alone and
    lowering or widening the frozen window are one kind of answer, and a
    reader of the history wants them under one name.
    """

    #: The first evaluation on a mesh, under the engine's own defaults.
    DEFAULT = "default"
    #: The immediate retry of a disentanglement that ran out of iterations, under a looser tolerance.
    UNCONVERGED = "unconverged"
    STALLED = "stalled"
    WINDOW = "window"
    THRESHOLDS = "thresholds"
    #: The one rung that recaptures: the mesh spacing halved.
    MESH = "mesh"
