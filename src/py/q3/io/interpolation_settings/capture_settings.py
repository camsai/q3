"""q3.io.interpolation_settings.capture_settings — :class:`CaptureSettings`, one model of the ``control.interpolation`` block."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field, field_validator

from q3.io.interpolation_settings.interpolation_settings import InterpolationSettings
from q3.io.interpolation_settings.mesh import Mesh


class CaptureSettings(InterpolationSettings):
    """The build request an NSCF run carries: the grid, and how the trial orbitals are made."""

    #: The full k-point grid the NSCF's ``kpoints_list`` enumerates, [n1, n2, n3].
    grid: Mesh
    #: The 1-based inclusive band range the model is built from; absent means the lowest bands.
    bands: (
        Annotated[list[Annotated[int, Field(ge=1)]], Field(min_length=2, max_length=2)] | None
    ) = None
    #: Centres of the s-type trial orbitals in crystal coordinates; absent selects the
    #: pseudopotentials' own atom-centred orbitals.
    projection_centers: list[Annotated[list[float], Field(min_length=3, max_length=3)]] | None = (
        None
    )
    #: Decay length of those orbitals in bohr.
    projection_decay_bohr: Annotated[float, Field(ge=0.0)] | None = None
    #: Per species, the 0-based pseudo-atomic shells to build trial orbitals from; a species
    #: left out keeps every shell its dataset does not mark unusable.
    trial_shells: (
        dict[str, Annotated[list[Annotated[int, Field(ge=0)]], Field(min_length=1)]] | None
    ) = None
    #: Interoperability formats to write beside the native record.
    export_formats: list[Literal["wannier90"]] | None = None

    @field_validator("trial_shells")
    @classmethod
    def _names_each_shell_once(
        cls, trial_shells: dict[str, list[int]] | None
    ) -> dict[str, list[int]] | None:
        for species, shells in (trial_shells or {}).items():
            if len(set(shells)) != len(shells):
                raise ValueError(f"trial_shells names a shell of {species} twice: {shells}")
        return trial_shells
