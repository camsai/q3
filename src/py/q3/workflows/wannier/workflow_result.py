"""q3.workflows.wannier.workflow_result — :class:`WorkflowResult`, what the protocol returns."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

from q3.io.interpolation.results.classes import InterpolationResults, LadderAttempt
from q3.io.results.classes import Results
from q3.workflows.band_distance import BandDistance


class WorkflowResult(BaseModel):
    """What the protocol returns: the verdict, the model, and everything it was judged against.

    The engine's own results files are carried as their models, and handed on
    by :meth:`as_dict` as the documents they are. :meth:`as_dict` is the dict the workflow has
    always returned, its fifteen keys in the documented order.
    """

    model_config = ConfigDict(frozen=True)

    status: Literal["accept", "reject"]
    reason: str
    #: The mesh the accepted or last model was captured on.
    grid: tuple[int, int, int]
    number_of_wannier_functions: int
    #: Per species, the ``PP_CHI`` shells the capture was told to use.
    trial_shells: dict[str, list[int]]
    number_of_bands: int
    #: One entry per rung fired, with its settings and outcome.
    attempts: tuple[LadderAttempt, ...]
    #: The accepted or last attempt's eta, or None when nothing was judged.
    band_distance: BandDistance | None
    #: The results file, with the band distance, the verdict and the ladder written into it.
    interpolation_results: InterpolationResults | None
    reference_results: Results | None
    reference_kpoints: list[tuple[float, float, float]]
    scf_results: Results | None
    nscf_results: Results | None
    fermi_energy_ry: float

    def as_dict(self) -> dict[str, Any]:
        """The result as every caller reads it; ``quality`` is the results file's own block."""
        document = (
            self.interpolation_results.to_document()
            if self.interpolation_results is not None
            else None
        )
        return {
            "status": self.status,
            "reason": self.reason,
            "grid": list(self.grid),
            "number_of_wannier_functions": self.number_of_wannier_functions,
            "trial_shells": self.trial_shells,
            "number_of_bands": self.number_of_bands,
            "attempts": [attempt.model_dump() for attempt in self.attempts],
            "band_distance": self.band_distance.as_dict() if self.band_distance else None,
            "quality": document["quality"] if document is not None else None,
            "interpolation_results": document,
            "reference_results": _document(self.reference_results),
            "reference_kpoints": [list(point) for point in self.reference_kpoints],
            "scf_results": _document(self.scf_results),
            "nscf_results": _document(self.nscf_results),
            "fermi_energy_ry": self.fermi_energy_ry,
        }


def _document(results: Results | None) -> dict[str, Any] | None:
    """A run's results file as its JSON, as ``as_dict`` hands every block on."""
    return results.to_document() if results is not None else None
