"""Validation of the cubic elastic-constant workflow on silicon.

Runs ``q3.workflows.run_cubic_elastic_constants_workflow`` against the real
binary and checks C11, C12 and C44 against published silicon values, plus the
internal consistency the fit itself cannot fake: the bulk modulus the constants
imply has to match the one the equation-of-state workflow gets from an
independent E(V) fit, and the constants have to satisfy the Born stability
criteria of a crystal that actually exists.

Published comparisons (GPa):

| constant | experiment | PBE, this kind of setup                    |
| -------- | ---------- | ------------------------------------------ |
| C11      | 165.7      | ~153                                        |
| C12      | 63.9       | ~57                                         |
| C44      | 79.6       | ~100 **clamped ion**, ~75 relaxed           |
| B        | 97.9       | ~88                                         |

The C44 row is the one to read carefully.  This workflow holds fractional
coordinates fixed, so it measures the clamped-ion C44: under a C44 shear the two
sublattices of the diamond structure slide relative to each other, and that
relaxation -- not included here -- is what brings ~100 GPa down to the ~75 GPa
that compares with the experimental 79.6.  The number below is therefore
compared against the clamped-ion value, which is what the workflow computes;
see ``q3.workflows.elastic``'s module docstring and TODOs.md §4.

Run::

    python -m pytest tests/validation/test_elastic_constants.py -v --q3-bin ./build/q3
"""

import numpy as np
import pytest
from q3.workflows import run_cubic_elastic_constants_workflow

from helpers import EXAMPLES_DIRECTORY, apply_config_overlay

SILICON_SCF_INPUT = EXAMPLES_DIRECTORY / "total_energy--semiconductor--silicon" / "nc.yml"

# Elastic constants need a denser k-grid than the total energy the example ships
# for: deforming the cell changes the symmetry-reduced k-point set, and although
# the sign-symmetric strain set cancels the even part of that artefact exactly,
# the odd part still carries a few percent at 2x2x2 (C44 comes out 95 GPa there
# against 99.8 at both 4x4x4 and 6x6x6).
CONVERGED_KGRID_OVERLAY = {
    "method": {"scf": {"kgrid": {"dimensions": [4, 4, 4], "shifts": [1, 1, 1]}}}
}

# The bulk modulus tests/validation/test_equation_of_state.py gates from a
# completely separate E(V) Birch-Murnaghan fit. Stated here as a number rather
# than measured again so this module does not pay for a second nine-point scan.
SILICON_PBE_BULK_MODULUS_GPA = 88.5


@pytest.fixture(scope="module")
def silicon_elastic_constants(q3_command, tmp_path_factory):
    """One strain scan (reference + both signs of two magnitudes, two patterns)."""
    directory = tmp_path_factory.mktemp("elastic_constants")
    scf_input = apply_config_overlay(
        SILICON_SCF_INPUT, CONVERGED_KGRID_OVERLAY, directory / "input"
    )
    return run_cubic_elastic_constants_workflow(
        scf_input,
        workdir=directory / "scan",
        # q3_command may be a full command (e.g. "mpirun -np 2 ./build/q3"); the
        # binary is its last token.
        q3_binary=q3_command[-1],
    )


def test_reference_cell_is_at_equilibrium_before_any_strain(silicon_elastic_constants):
    """The constants are those of the reference cell, pre-stress and all.

    A large residual pressure here would not make the fit wrong, but it would
    make the reported numbers the constants of a compressed crystal rather than
    of silicon, so it is worth knowing which one was measured.
    """
    reference = silicon_elastic_constants["points"][0]
    assert reference["pattern"] == "reference"
    stress = np.asarray(reference["voigt_stress_gpa"])
    # Tensile-positive stress, so the pressure is minus the mean normal component.
    pressure = -float(np.mean(stress[:3]))
    print(f"\n  reference pressure: {pressure:+.4f} GPa")
    assert abs(pressure) < 0.5
    assert np.max(np.abs(stress[3:])) < 1.0e-3


def test_cubic_constants_match_published_pbe_silicon(silicon_elastic_constants):
    c11 = silicon_elastic_constants["c11_gpa"]
    c12 = silicon_elastic_constants["c12_gpa"]
    c44 = silicon_elastic_constants["c44_gpa"]
    print(f"\n  C11 = {c11:.2f} GPa, C12 = {c12:.2f} GPa, C44 = {c44:.2f} GPa (clamped ion)")
    assert c11 == pytest.approx(153.0, rel=0.06)
    assert c12 == pytest.approx(57.0, rel=0.12)
    # Clamped ion, not the experimental 79.6 -- see the module docstring.
    assert c44 == pytest.approx(100.0, rel=0.12)


def test_constants_satisfy_the_born_stability_criteria(silicon_elastic_constants):
    """C11 > |C12|, C11 + 2 C12 > 0, C44 > 0 -- a cubic crystal that holds together.

    Cheap, but it is the check that fails loudly if the Voigt bookkeeping or the
    tensile-positive sign flip is ever inverted: a sign error turns silicon into
    a crystal that spontaneously collapses.
    """
    c11 = silicon_elastic_constants["c11_gpa"]
    c12 = silicon_elastic_constants["c12_gpa"]
    c44 = silicon_elastic_constants["c44_gpa"]
    assert c11 > abs(c12)
    assert c11 + 2.0 * c12 > 0.0
    assert c44 > 0.0


def test_bulk_modulus_agrees_with_the_equation_of_state_route(silicon_elastic_constants):
    """(C11 + 2 C12)/3 against the Birch-Murnaghan B0 from an E(V) scan.

    Two entirely separate measurements: this one differentiates the stress with
    respect to strain at fixed volume-changing direction, the other fits total
    energies across volume.  They share the engine and nothing else, so
    agreement here is a real cross-check rather than a restatement.
    """
    from_constants = (
        silicon_elastic_constants["c11_gpa"] + 2.0 * silicon_elastic_constants["c12_gpa"]
    ) / 3.0
    print(
        f"\n  B from C_ij = {from_constants:.2f} GPa, "
        f"B0 from the E(V) fit = {SILICON_PBE_BULK_MODULUS_GPA:.2f} GPa"
    )
    assert from_constants == pytest.approx(SILICON_PBE_BULK_MODULUS_GPA, rel=0.05)


def test_anisotropy_ratio_is_the_known_silicon_value(silicon_elastic_constants):
    """Zener A = 2 C44/(C11 - C12) ~ 2 for silicon (clamped ion; ~1.56 relaxed)."""
    zener_ratio = (
        2.0
        * silicon_elastic_constants["c44_gpa"]
        / (silicon_elastic_constants["c11_gpa"] - silicon_elastic_constants["c12_gpa"])
    )
    assert zener_ratio == pytest.approx(2.0, rel=0.15)


def test_the_stress_response_is_linear_in_the_strain(silicon_elastic_constants):
    """Doubling the strain must double the odd part of the stress response.

    The odd part is the whole of what the elastic constants are fitted from, so
    this is the linearity assumption stated as a measurement: at 0.5 % and 1 %
    silicon answers 0.500 and 0.998 GPa in shear, a ratio of 1.996.

    The even part is printed beside it because it is what the sign-symmetric
    strain set exists to cancel, and the two patterns show the two different
    things it can be. Under uniaxial strain it is genuine third-order
    elasticity and scales as the square of the magnitude (0.0070 -> 0.0265 GPa,
    a factor of 3.8). Under shear it is the k-point-set discontinuity -- the
    deformed cell has a different symmetry-reduced k-point set than the
    undeformed reference -- and is CONSTANT in the magnitude (0.98468 ->
    0.98446 GPa), because it depends on the deformation existing at all and not
    on its size. At 4x4x4 it is twice the physical shear response being
    measured, which is why a one-sided difference would be worthless here and
    the symmetric set is not a refinement but the method.
    """
    smaller, larger = 0.005, 0.01
    points = silicon_elastic_constants["points"]
    reference = np.asarray(points[0]["voigt_stress_gpa"])
    for pattern, voigt_index in (("uniaxial_x", 0), ("shear_yz", 3)):
        parts = {}
        for magnitude in (smaller, larger):
            stress = {
                np.sign(point["voigt_strain"][voigt_index]): np.asarray(point["voigt_stress_gpa"])
                for point in points
                if point["pattern"] == pattern
                and abs(point["voigt_strain"][voigt_index]) == pytest.approx(magnitude)
            }
            assert set(stress) == {1.0, -1.0}, f"{pattern} at {magnitude} is not a +/- pair"
            # The reference stress cancels from the odd part and is removed from the even one.
            parts[magnitude] = (
                float(np.abs(0.5 * (stress[1.0] - stress[-1.0])).max()),
                float(np.abs(0.5 * (stress[1.0] + stress[-1.0]) - reference).max()),
            )
        odd_ratio = parts[larger][0] / parts[smaller][0]
        print(
            f"\n  {pattern}: odd {parts[smaller][0]:.5f} -> {parts[larger][0]:.5f} GPa "
            f"(x{odd_ratio:.3f}), even {parts[smaller][1]:.5f} -> {parts[larger][1]:.5f} GPa"
        )
        assert odd_ratio == pytest.approx(larger / smaller, rel=0.03)
        # The even part may be constant (the k-point artefact) or grow as the
        # square of the strain (third-order elasticity); anything faster means
        # the strains have left the range the linear fit describes.
        assert parts[larger][1] <= 4.5 * max(parts[smaller][1], 1.0e-4)
