"""The files the engine reads for one case: its material and its SCF manifest."""

from __future__ import annotations

from pathlib import Path

from q3.io.input_writer import write_material_json, write_scf_input
from q3.workflows.interpolation import mesh_for_spacing

from .case import CampaignCase
from .settings import (
    EXCHANGE_CORRELATION_FUNCTIONAL,
    METHOD_SUBTYPE,
    OUTPUT_DIRECTORY,
    RUN_PREFIX,
    SCF_CONVERGENCE_THRESHOLD_RY,
    SCF_KPOINT_SPACING_INVERSE_ANGSTROM,
    SCF_MAXIMUM_ITERATIONS,
    SCF_MIXING_BETA,
    SMEARING_RYDBERG,
    SMEARING_TYPE,
)

MATERIAL_FILE = "material.json"
MANIFEST_FILE = "scf.yml"


def write_case(case: CampaignCase, directory: str | Path) -> Path:
    """The case's ``material.json`` and ``scf.yml``; returns the manifest's path.

    The manifest names the material by its file name alone, so a case's
    directory can be moved or archived whole.
    """
    directory = Path(directory)
    write_material_json(case.material, directory / MATERIAL_FILE)
    return write_scf_input(
        directory / MANIFEST_FILE,
        material_path=MATERIAL_FILE,
        pseudopotentials=dict(case.pseudopotentials),
        pseudopotential_subtype=METHOD_SUBTYPE[case.family],
        wavefunction_cutoff=case.wavefunction_cutoff_ry,
        density_cutoff=case.density_cutoff_ry,
        kgrid_dimensions=mesh_for_spacing(
            case.material["lattice"], SCF_KPOINT_SPACING_INVERSE_ANGSTROM
        ),
        convergence_threshold=SCF_CONVERGENCE_THRESHOLD_RY,
        maximum_iterations=SCF_MAXIMUM_ITERATIONS,
        mixing_beta=SCF_MIXING_BETA,
        exchange_correlation_functional=EXCHANGE_CORRELATION_FUNCTIONAL,
        outdir=f"./{OUTPUT_DIRECTORY}",
        prefix=RUN_PREFIX,
        smearing=SMEARING_TYPE if case.smeared else "none",
        smearing_width=SMEARING_RYDBERG if case.smeared else 0.0,
    )
