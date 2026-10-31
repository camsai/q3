"""q3.workflows.wannier.driver — :func:`run_automatic_wannier_workflow`, the entry point the package exports.

The function is the protocol's public face and nothing more: it gathers the
caller's choices into :class:`~q3.workflows.wannier.protocol_options.ProtocolOptions`,
opens the chain over the SCF manifest, runs
:class:`~q3.workflows.wannier.protocol.WannierProtocol`, and returns its
:class:`~q3.workflows.wannier.workflow_result.WorkflowResult` as the dict every
caller reads.
"""

from __future__ import annotations

from pathlib import Path

from q3.binary import ExecutionPolicy
from q3.workflows.band_distance import DEFAULT_ENERGY_OFFSET_EV, DEFAULT_SMEARING_EV
from q3.workflows.interpolation import (
    DEFAULT_KPOINT_SPACING_INVERSE_ANGSTROM,
    EngineChain,
    Grid,
    KPoint,
)
from q3.workflows.kpath import DEFAULT_POINTS_PER_SEGMENT
from q3.workflows.wannier.protocol import WannierProtocol
from q3.workflows.wannier.protocol_options import ProtocolOptions
from q3.workflows.wannier.settings import (
    ACCEPT_ETA_EV,
    ACCEPT_ETA_MAXIMUM_EV,
    DEFAULT_MAXIMUM_MESH_REFINEMENTS,
)


def run_automatic_wannier_workflow(
    scf_input_path: str | Path,
    *,
    kpoint_spacing_inverse_angstrom: float = DEFAULT_KPOINT_SPACING_INVERSE_ANGSTROM,
    grid: Grid | None = None,
    number_of_bands: int | None = None,
    band_path: list[str] | None = None,
    lattice_type: str | None = None,
    points_per_segment: int = DEFAULT_POINTS_PER_SEGMENT,
    reference_kpoints: list[KPoint] | None = None,
    accept_eta_ev: float = ACCEPT_ETA_EV,
    accept_eta_maximum_ev: float = ACCEPT_ETA_MAXIMUM_EV,
    energy_offset_ev: float = DEFAULT_ENERGY_OFFSET_EV,
    smearing_ev: float = DEFAULT_SMEARING_EV,
    maximum_mesh_refinements: int = DEFAULT_MAXIMUM_MESH_REFINEMENTS,
    maximum_threshold_attempts: int | None = None,
    dos_mesh: Grid | None = None,
    dos_broadening_ry: float | None = None,
    export_formats: list[str] | tuple[str, ...] = (),
    q3_binary: str | Path | None = None,
    q3_utility_binary: str | Path | None = None,
    execution_policy: ExecutionPolicy | None = None,
    skip_scf: bool = False,
    capture_output: bool = True,
) -> dict:
    """SCF, capture on the chosen mesh, a reference calculation, and the retry ladder.

    Parameters
    ----------
    scf_input_path:
        A Q3 SCF manifest. Nothing in it needs to say anything about Wannier
        functions: the mesh, the trial orbitals (:func:`trial_shells`), the band
        window and the thresholds are all chosen here. ``control.outdir``/``prefix`` are
        reused so the record and the checkpoint are found.
    kpoint_spacing_inverse_angstrom, grid:
        The mesh rule's spacing (2 pi included), or an explicit grid that
        overrides it. The mesh rung halves the spacing and recaptures.
    number_of_bands:
        Overrides the window rule ceil(1.5 N) + 4 for N trial orbitals.
    band_path, lattice_type, points_per_segment, reference_kpoints:
        Where the model is compared with a direct calculation: explicit
        k-points, or a path of high-symmetry labels through
        ``q3.workflows.kpath``, or -- when neither is given -- points half a
        mesh step off every mesh point (:func:`off_mesh_reference_kpoints`).
    accept_eta_ev, accept_eta_maximum_ev, energy_offset_ev, smearing_ev:
        The gate and the band-distance window (10 and 100 meV; 2 eV above the
        Fermi level with 0.1 eV smearing).
    maximum_mesh_refinements:
        How many times the mesh rung may fire; 0 disables recapture. An
        explicit *grid* is doubled on every axis when it does.
    maximum_threshold_attempts:
        A cap on the threshold walk (thirty pairs in full), for a campaign
        that would rather report than spend an hour on one structure.
    dos_mesh, dos_broadening_ry, export_formats:
        Passed through to the evaluation and the capture.
    execution_policy:
        The execution policy every engine run starts the binary with; see
        :func:`q3.workflows.interpolation.run_interpolation_workflow`.
    skip_scf:
        Reuse an existing SCF checkpoint in the outdir.

    Returns
    -------
    dict
        :class:`WorkflowResult` as a dict: ``status`` (``accept`` or
        ``reject``), ``reason``, ``grid``, ``number_of_wannier_functions``,
        ``trial_shells`` (per species, the ``PP_CHI`` shells the capture was
        told to use), ``number_of_bands``, ``attempts`` (one dict per rung
        fired), ``band_distance`` (the accepted or last attempt's eta),
        ``quality`` (the results file's block, with the band distance and the
        status written into it), ``interpolation_results``,
        ``reference_results``, ``reference_kpoints``, ``scf_results``,
        ``nscf_results`` and ``fermi_energy_ry``.
    """
    options = ProtocolOptions(
        kpoint_spacing_inverse_angstrom=kpoint_spacing_inverse_angstrom,
        grid=grid,
        number_of_bands=number_of_bands,
        band_path=band_path,
        lattice_type=lattice_type,
        points_per_segment=points_per_segment,
        reference_kpoints=reference_kpoints,
        accept_eta_ev=accept_eta_ev,
        accept_eta_maximum_ev=accept_eta_maximum_ev,
        energy_offset_ev=energy_offset_ev,
        smearing_ev=smearing_ev,
        maximum_mesh_refinements=maximum_mesh_refinements,
        maximum_threshold_attempts=maximum_threshold_attempts,
        dos_mesh=dos_mesh,
        dos_broadening_ry=dos_broadening_ry,
        export_formats=tuple(export_formats),
        skip_scf=skip_scf,
    )
    with EngineChain(
        Path(scf_input_path).expanduser().resolve(),
        q3_binary=q3_binary,
        q3_utility_binary=q3_utility_binary,
        capture_output=capture_output,
        execution_policy=execution_policy,
    ) as chain:
        return WannierProtocol(chain, options).run().as_dict()
