"""One base cell of the roster, before its variants."""

from __future__ import annotations

from ase import Atoms
from pydantic import BaseModel, ConfigDict


class BaseStructure(BaseModel):
    """A cell the roster builds variants of.

    ``key`` names the family of cases it becomes (``silicon``, then
    ``silicon--rattled``, ...); ``seed_offset`` separates its random variants
    from every other structure's under the one roster seed.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    formula: str
    key: str
    atoms: Atoms
    seed_offset: int
