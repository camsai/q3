"""Tests for q3.structure.strain: homogeneous strain in the cell's own frame."""

from pathlib import Path

import numpy as np
import pytest
from q3.structure.ase_atoms import material_json_to_ase_atoms
from q3.structure.strain import isotropic_strain_matrix, strain_atoms, strain_matrix_from_voigt

SILICON_MATERIAL = (
    Path(__file__).resolve().parents[2] / "examples" / "data" / "materials" / "silicon.json"
)


def test_isotropic_strain_matrix_scales_every_vector():
    assert np.array_equal(isotropic_strain_matrix(1.02), 1.02 * np.eye(3))


@pytest.mark.parametrize("bad_factor", [0.0, -1.0])
def test_isotropic_strain_matrix_refuses_a_non_positive_factor(bad_factor):
    with pytest.raises(ValueError, match="positive"):
        isotropic_strain_matrix(bad_factor)


def test_voigt_shear_is_an_engineering_strain():
    """eps_4 = 2 eps_yz: a Voigt shear of 0.01 puts 0.005 in each off-diagonal yz slot."""
    pytest.importorskip("ase")
    matrix = strain_matrix_from_voigt([0.0, 0.0, 0.0, 0.01, 0.0, 0.0])
    expected = np.eye(3)
    expected[1, 2] = expected[2, 1] = 0.005
    assert np.allclose(matrix, expected, rtol=0.0, atol=1.0e-15)


def test_isotropic_strain_is_bitwise_the_scaled_cell():
    """The volume scan's cells are exactly ``reference * scale``, to the last bit.

    Pinned because the equation-of-state numbers in FUNCTIONALITY.md were
    measured on cells built that way, before the scan went through this module.
    """
    pytest.importorskip("ase")
    atoms = material_json_to_ase_atoms(SILICON_MATERIAL)
    reference_cell = np.array(atoms.get_cell())
    for scale in (0.96, 0.99, 1.0, 1.03):
        strain_atoms(atoms, reference_cell, isotropic_strain_matrix(scale))
        assert np.array_equal(np.array(atoms.get_cell()), reference_cell * scale)


def test_strain_keeps_fractional_coordinates_and_the_frame():
    pytest.importorskip("ase")
    atoms = material_json_to_ase_atoms(SILICON_MATERIAL)
    reference_cell = np.array(atoms.get_cell())
    reference_fractional = atoms.get_scaled_positions()
    reference_volume = atoms.get_volume()

    strain_matrix = strain_matrix_from_voigt([0.01, 0.0, 0.0, 0.0, 0.0, 0.0])
    strain_atoms(atoms, reference_cell, strain_matrix)

    assert np.allclose(atoms.get_scaled_positions(), reference_fractional, atol=1.0e-12)
    # Uniaxial x strain in Q3's frame stretches the x components only.
    assert np.allclose(np.array(atoms.get_cell())[:, 0], reference_cell[:, 0] * 1.01)
    assert np.allclose(np.array(atoms.get_cell())[:, 1:], reference_cell[:, 1:])
    assert atoms.get_volume() == pytest.approx(reference_volume * 1.01, rel=1.0e-12)
