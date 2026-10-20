"""q3.io.interpolation_settings.mesh — a k-point mesh as the block carries it: three positive integers."""

from __future__ import annotations

from typing import Annotated, Any

from pydantic import BeforeValidator, ValidationInfo

Grid = tuple[int, int, int]


def validated_grid(grid: Any, name: str = "grid") -> Grid:
    """Three positive integers, refused here rather than minutes later by the engine.

    The mesh is what a caller most often gets wrong, because it must be the
    full unreduced grid the k-point list enumerates rather than the
    symmetry-reduced one an ordinary calculation would use.
    """
    values = tuple(grid)
    if len(values) != 3 or any(int(value) != value or value <= 0 for value in values):
        raise ValueError(f"{name} must be three positive integers, got {grid!r}")
    return tuple(int(value) for value in values)  # type: ignore[return-value]


def _validated_field(value: Any, info: ValidationInfo) -> list[int]:
    return list(validated_grid(value, info.field_name or "mesh"))


#: A mesh as a field of the block: a model that holds one holds a valid one.
Mesh = Annotated[list[int], BeforeValidator(_validated_field)]
