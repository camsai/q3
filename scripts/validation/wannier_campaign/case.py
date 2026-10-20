"""One structure of the roster."""

from __future__ import annotations

from pydantic import BaseModel


class CampaignCase(BaseModel):
    """One structure of the roster, with everything its manifest needs.

    ``cutoffs`` is (wavefunction, density) in Rydberg, the largest any of the
    case's species asks for.
    """

    name: str
    family: str
    formula: str
    variant: str
    number_of_atoms: int
    material: dict
    pseudopotentials: dict[str, str]
    cutoffs: tuple[float, float]
    smeared: bool
    note: str = ""

    @property
    def wavefunction_cutoff_ry(self) -> float:
        return self.cutoffs[0]

    @property
    def density_cutoff_ry(self) -> float:
        return self.cutoffs[1]

    def as_dict(self) -> dict:
        return self.model_dump()
