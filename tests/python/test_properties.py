"""Tests for q3.properties: the fits behind the equation-of-state and elastic workflows.

Synthetic inputs only -- the physics against the real binary is
tests/validation/test_equation_of_state.py and test_elastic_constants.py.
"""

import numpy as np
import pytest
from mat3ra.code.constants import Coefficients
from q3.properties import (
    DEFAULT_LINEAR_SCALE_FACTORS,
    DEFAULT_STRAIN_MAGNITUDES,
    cubic_voigt_strain_set,
    fit_birch_murnaghan,
    fit_cubic_elastic_constants,
    hydrostatic_pressure_gpa,
    pressure_gpa_to_ev_per_angstrom3,
    voigt_stress_gpa,
)

GPA = Coefficients.EV_PER_ANGSTROM3_TO_GPA


# ── stress ──────────────────────────────────────────────────────────────


def test_compressive_stress_is_a_positive_pressure():
    """ASE's stress is tensile positive; a compressed cell reads negative, a pressure positive."""
    stress = np.array([-0.01, -0.01, -0.01, 0.0, 0.0, 0.0])
    assert hydrostatic_pressure_gpa(stress) == pytest.approx(0.01 * GPA, rel=1.0e-14)


def test_voigt_stress_conversion_keeps_sign_and_order():
    stress = np.array([0.1, -0.2, 0.3, -0.4, 0.5, -0.6])
    assert np.allclose(voigt_stress_gpa(stress), stress * GPA, rtol=1.0e-15, atol=0.0)


def test_pressure_conversion_round_trips():
    assert pressure_gpa_to_ev_per_angstrom3(10.0) * GPA == pytest.approx(10.0, rel=1.0e-14)


# ── equation of state ───────────────────────────────────────────────────


def _birch_murnaghan_energy(volumes, e0, v0, b0, b0_prime):
    """Third-order Birch-Murnaghan E(V), Birch, Phys. Rev. 71, 809 (1947)."""
    eta = (v0 / volumes) ** (2.0 / 3.0)
    return e0 + 9.0 * v0 * b0 / 16.0 * (
        (eta - 1.0) ** 3 * b0_prime + (eta - 1.0) ** 2 * (6.0 - 4.0 * eta)
    )


def test_birch_murnaghan_fit_recovers_its_own_parameters():
    pytest.importorskip("ase")
    v0, e0, b0_gpa, b0_prime = 40.89, -215.0, 88.0, 4.3
    volumes = v0 * np.asarray(DEFAULT_LINEAR_SCALE_FACTORS) ** 3
    energies = _birch_murnaghan_energy(volumes, e0, v0, b0_gpa / GPA, b0_prime)

    fit = fit_birch_murnaghan(volumes.tolist(), energies.tolist())

    assert fit["volume"] == {"name": "volume", "units": "angstrom^3", "value": pytest.approx(v0)}
    assert fit["total_energy"]["name"] == "total_energy"
    assert fit["total_energy"]["units"] == "eV"
    assert fit["total_energy"]["value"] == pytest.approx(e0, abs=1.0e-8)
    assert fit["bulk_modulus_gpa"] == pytest.approx(b0_gpa, rel=1.0e-5)
    assert fit["bulk_modulus_pressure_derivative"] == pytest.approx(b0_prime, rel=1.0e-4)


# ── cubic elastic constants ─────────────────────────────────────────────


def test_cubic_strain_set_is_reference_first_and_sign_symmetric():
    strain_set = cubic_voigt_strain_set((0.005, 0.01))
    assert len(strain_set) == 4 * 2 + 1
    assert strain_set[0][0] == "reference"
    assert np.array_equal(strain_set[0][1], np.zeros(6))
    for pattern, voigt_index in (("uniaxial_x", 0), ("shear_yz", 3)):
        signed = [strain[voigt_index] for name, strain in strain_set if name == pattern]
        assert signed == [-0.01, -0.005, 0.005, 0.01]
        others = [np.delete(strain, voigt_index) for name, strain in strain_set if name == pattern]
        assert not np.any(others)


def test_default_strain_set_has_two_magnitudes():
    assert len(cubic_voigt_strain_set()) == 4 * len(DEFAULT_STRAIN_MAGNITUDES) + 1


def _cubic_stress(voigt_strain, c11, c12, c44):
    stiffness = np.zeros((6, 6))
    stiffness[:3, :3] = c12
    np.fill_diagonal(stiffness[:3, :3], c11)
    stiffness[3, 3] = stiffness[4, 4] = stiffness[5, 5] = c44
    return stiffness @ voigt_strain


def test_cubic_fit_recovers_constants_and_ignores_every_even_term():
    """The sign-symmetric set is the method: an even term of any size never reaches the slope.

    The stress carries a pre-stress (the reference cell's own), a constant term
    on every deformed cell (the k-point-set discontinuity) and a quadratic one
    (third-order elasticity) -- the fit must return the linear constants exactly.
    """
    c11, c12, c44 = 152.6, 56.2, 99.8
    pre_stress = np.array([0.3, 0.3, 0.3, 0.0, 0.0, 0.0])
    points = []
    for pattern, voigt_strain in cubic_voigt_strain_set((0.005, 0.01)):
        deformed = 0.0 if pattern == "reference" else 1.0
        even = deformed * (0.985 * np.ones(6) + 250.0 * float(np.sum(voigt_strain)) ** 2)
        stress = _cubic_stress(voigt_strain, c11, c12, c44) + pre_stress + even
        points.append(
            {
                "pattern": pattern,
                "voigt_strain": voigt_strain.tolist(),
                "voigt_stress_gpa": stress.tolist(),
            }
        )

    constants = fit_cubic_elastic_constants(points)

    assert constants["c11_gpa"] == pytest.approx(c11, rel=1.0e-12)
    assert constants["c12_gpa"] == pytest.approx(c12, rel=1.0e-12)
    assert constants["c44_gpa"] == pytest.approx(c44, rel=1.0e-12)
