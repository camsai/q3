"""tests/python/test_imports.py — Smoke tests for q3 Python package imports.

These tests are fast (no binary execution) and verify that all public modules
can be imported and that the package is correctly installed from src/py/.

Each module is imported by its own test, through ``importlib``, so a broken
module fails the test that names it rather than the whole file's collection.
"""

import importlib

import pytest
from q3.errors import (
    Q3Error,
    Q3InputError,
    Q3RuntimeError,
    Q3VersionMismatchError,
    SCFNotConvergedError,
)
from q3.workflows.kpath import high_symmetry_points, interpolate_kpath

PUBLIC_NAMES = {
    "q3": ["Q3Error", "SCFNotConvergedError"],
    "q3.errors": [
        "Q3Error",
        "Q3InputError",
        "Q3RuntimeError",
        "Q3VersionMismatchError",
        "SCFNotConvergedError",
    ],
    "q3.binary": ["find_binary", "run_q3"],
    "q3.flavors": ["run_nscf", "run_scf"],
    "q3.properties": [
        "cubic_voigt_strain_set",
        "fit_birch_murnaghan",
        "fit_cubic_elastic_constants",
        "hydrostatic_pressure_gpa",
    ],
    "q3.io": [
        "ensure_calculation_type",
        "extract_band_structure",
        "extract_dos",
        "extract_lowdin_charges",
        "extract_pdos",
        "extract_total_energy",
        "write_nscf_input",
        "write_scf_input",
        "write_supercell_scf_input",
    ],
    "q3.workflows": [
        "run_band_structure_workflow",
        "run_cubic_elastic_constants_workflow",
        "run_dos_workflow",
        "run_equation_of_state_workflow",
        "run_pdos_workflow",
        "run_relax_workflow",
        "run_vc_relax_workflow",
    ],
    "q3.calculators": ["Q3ASECalculator"],
    "q3.structure": ["material_json_to_ase_atoms"],
}


@pytest.mark.parametrize("module_name", PUBLIC_NAMES)
def test_module_exposes_its_public_names(module_name):
    """Every public module imports cleanly and exposes the names its callers use."""
    module = importlib.import_module(module_name)

    for name in PUBLIC_NAMES[module_name]:
        assert getattr(module, name, None) is not None, f"{module_name}.{name}"


def test_error_hierarchy():
    """q3.errors must expose the full exception hierarchy."""
    assert issubclass(Q3InputError, Q3Error)
    assert issubclass(Q3RuntimeError, Q3Error)
    assert issubclass(Q3VersionMismatchError, Q3Error)
    assert issubclass(SCFNotConvergedError, Q3RuntimeError)


def test_kpath_module():
    """q3.workflows.kpath must expose interpolation utilities."""
    # Basic sanity: Γ is at the origin for FCC.
    points = high_symmetry_points("FCC")
    assert "G" in points or "Γ" in points
    gamma = points.get("G") or points.get("Γ")
    assert gamma == (0.0, 0.0, 0.0)

    # Two-segment path produces the right number of k-points.
    kpoints, distances = interpolate_kpath(["G", "X", "L"], "FCC", points_per_segment=5)
    # First segment: 5 points (G inclusive, X exclusive)
    # Second segment: 5+1=6 points (X inclusive, L inclusive)
    assert len(kpoints) == 11
    assert len(distances) == 11
    assert distances[0] == 0.0
    assert distances[-1] > 0.0
