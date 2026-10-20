"""q3.workflows.wannier.protocol_options — :class:`ProtocolOptions`, what a caller may choose."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, ValidationInfo, field_validator

from q3.workflows.band_distance import DEFAULT_ENERGY_OFFSET_EV, DEFAULT_SMEARING_EV
from q3.workflows.interpolation import (
    DEFAULT_KPOINT_SPACING_INVERSE_ANGSTROM,
    Grid,
    KPoint,
    validated_grid,
)
from q3.workflows.kpath import DEFAULT_POINTS_PER_SEGMENT
from q3.workflows.wannier.accept_gate import AcceptGate
from q3.workflows.wannier.settings import (
    ACCEPT_ETA_EV,
    ACCEPT_ETA_MAXIMUM_EV,
    DEFAULT_MAXIMUM_MESH_REFINEMENTS,
)


class ProtocolOptions(BaseModel):
    """Every choice the automated protocol leaves to its caller, each with the protocol's own default.

    Nothing here says anything about Wannier functions a caller has to know:
    the mesh, the trial orbitals, the band window and the thresholds are all
    chosen by the protocol, and these only steer it. What the engine would
    refuse -- a mesh that is not three positive integers, an export format it
    does not write -- is refused here, before any calculation has run.
    """

    model_config = ConfigDict(frozen=True)

    #: The mesh rule's spacing (2 pi included); the mesh rung halves it and recaptures.
    kpoint_spacing_inverse_angstrom: float = DEFAULT_KPOINT_SPACING_INVERSE_ANGSTROM
    #: An explicit grid, overriding the rule; the mesh rung doubles it on every axis.
    grid: Grid | None = None
    #: Overrides the window rule ceil(1.5 N) + 4 for N trial orbitals.
    number_of_bands: int | None = None
    #: Where the model is compared with a direct calculation: a path of high-symmetry
    #: labels, or explicit k-points; with neither, points half a mesh step off every mesh point.
    band_path: list[str] | None = None
    lattice_type: str | None = None
    points_per_segment: int = DEFAULT_POINTS_PER_SEGMENT
    reference_kpoints: list[KPoint] | None = None
    #: The gate: 10 and 100 meV, measured up to 2 eV above the Fermi level with 0.1 eV smearing.
    accept_eta_ev: float = ACCEPT_ETA_EV
    accept_eta_maximum_ev: float = ACCEPT_ETA_MAXIMUM_EV
    energy_offset_ev: float = DEFAULT_ENERGY_OFFSET_EV
    smearing_ev: float = DEFAULT_SMEARING_EV
    #: How many times the mesh rung may fire; 0 disables recapture.
    maximum_mesh_refinements: int = DEFAULT_MAXIMUM_MESH_REFINEMENTS
    #: A cap on the threshold walk (thirty pairs in full), for a campaign that would
    #: rather report than spend an hour on one structure.
    maximum_threshold_attempts: int | None = None
    #: Passed through to the evaluation and the capture.
    dos_mesh: Grid | None = None
    dos_broadening_ry: float | None = None
    export_formats: tuple[Literal["wannier90"], ...] = ()
    #: Reuse an existing SCF checkpoint in the outdir.
    skip_scf: bool = False

    @field_validator("grid", "dos_mesh", mode="before")
    @classmethod
    def _three_positive_integers(cls, mesh: Any, info: ValidationInfo) -> Any:
        return mesh if mesh is None else validated_grid(mesh, info.field_name)

    @property
    def gate(self) -> AcceptGate:
        return AcceptGate(
            eta_ev=self.accept_eta_ev,
            eta_maximum_ev=self.accept_eta_maximum_ev,
            energy_offset_ev=self.energy_offset_ev,
            smearing_ev=self.smearing_ev,
        )
