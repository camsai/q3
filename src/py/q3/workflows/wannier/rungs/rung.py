"""q3.workflows.wannier.rungs.rung — :class:`Rung`, the interface every rung of the ladder has."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar

from q3.workflows.wannier.evaluation import Evaluation
from q3.workflows.wannier.ladder_state import LadderState
from q3.workflows.wannier.rung_name import RungName


class Rung(ABC):
    """One rung: a condition on the state it is handed, and the attempts it makes if so."""

    #: How the attempts this rung makes are labelled in the history.
    name: ClassVar[RungName]

    def __init__(self, evaluation: Evaluation):
        self.evaluation = evaluation

    @abstractmethod
    def applies(self, state: LadderState) -> bool:
        """Whether the attempt the last rung left is one this rung has an answer for."""

    @abstractmethod
    def attempt(self, state: LadderState) -> LadderState:
        """Try the answer: the state after this rung's attempts, with what helped promoted."""
