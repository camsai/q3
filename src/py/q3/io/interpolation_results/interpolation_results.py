"""q3.io.interpolation_results.interpolation_results — :class:`InterpolationResults`, one block of the results file of ``q3_utility interpolate``."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from q3.io.interpolation_results.density_of_states import DensityOfStates
from q3.io.interpolation_results.interpolation import Interpolation
from q3.io.interpolation_results.quality import Quality
from q3.io.interpolation_results.results_block import ResultsBlock


class InterpolationResults(ResultsBlock):
    """The whole file: ``<outdir>/<prefix>_interpolation_results.json``."""

    calculation: str | None = None
    provenance: dict[str, Any] | None = None
    fermi_energy_ry: float | None = None
    kpoints: list[list[float]] | None = None
    eigenvalues_ry: list[list[list[float]]] | None = None
    density_of_states: DensityOfStates | None = None
    interpolation: Interpolation | None = None

    @classmethod
    def read(cls, path: str | Path) -> InterpolationResults:
        """The file at ``path``, whatever subset of the blocks it carries."""
        return cls.model_validate(json.loads(Path(path).read_text()))

    def quality_block(self) -> Quality:
        """``interpolation.quality``, the block a driver annotates; made, with its parent, where absent."""
        if self.interpolation is None:
            self.interpolation = Interpolation()
        if self.interpolation.quality is None:
            self.interpolation.quality = Quality()
        return self.interpolation.quality

    def to_document(self) -> dict:
        """The file's JSON: absent fields absent; a ladder attempt keeps its own nulls."""
        return self.model_dump(exclude_none=True)

    def write(self, path: str | Path) -> None:
        """Write the file as the engine formats it: four-space indents, a trailing newline."""
        Path(path).write_text(json.dumps(self.to_document(), indent=4) + "\n")
