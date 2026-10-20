"""q3.properties.elastic_constants — cubic C11, C12 and C44 from a strain-stress scan.

The strain set and the fit belong together, because the fit is only right for
the set: one uniaxial and one shear pattern, each applied in **both signs** of
every magnitude, and each constant taken as a zero-intercept least-squares
slope of the stress against the strain.

**Every strain in both signs is the method, not a refinement.**  Deforming a
cell changes a symmetry-reduced k-point set, which adds a stress term that is
EVEN in the strain: in silicon's shear component ~66 kbar at a 2x2x2 k-grid --
six times the physical response -- ~9.8 kbar at 4x4x4 and ~1.6 kbar at 6x6x6,
constant in the strain magnitude.  A zero-intercept slope over a sign-symmetric
set removes every even term exactly (the reference cell's own stress included),
which is why C44 is k-converged at 4x4x4 (99.82 GPa against 99.78 at 6x6x6)
while a one-sided difference at 2x2x2 overestimates it sevenfold.

**Cubic symmetry only**, with the cubic axes along x, y and z: on a
lower-symmetry cell the two patterns return weighted averages of several real
constants.  Strains are engineering strains in Voigt order (eps_4 = 2 eps_yz);
stresses are tensile positive, in GPa.

**Upstream candidate (mat3ra-prode)**, with the rest of ``q3.properties``.
"""

# TODO: move this module to mat3ra-prode, the stack's home for derived properties,
#   and import it from there (TODOs.md §4, item 5).

from __future__ import annotations

from typing import Any

import numpy as np

#: Positive strain magnitudes, each applied in both signs.  Two rather than one
#: let the points show the response is linear: its odd part doubles when the
#: strain does.
DEFAULT_STRAIN_MAGNITUDES = (0.005, 0.01)

#: (pattern name, the Voigt index it strains) -- the two that determine a cubic crystal.
CUBIC_STRAIN_PATTERNS = (("uniaxial_x", 0), ("shear_yz", 3))


def cubic_voigt_strain_set(
    strain_magnitudes=DEFAULT_STRAIN_MAGNITUDES,
) -> list[tuple[str, np.ndarray]]:
    """The strains to run, as ``(pattern, voigt_strain)`` pairs, the unstrained cell first.

    ``4 * len(strain_magnitudes) + 1`` of them: the ``"reference"`` cell, then
    both signs of every magnitude for ``"uniaxial_x"`` and ``"shear_yz"``,
    each pattern in ascending order of signed strain.
    """
    strain_set: list[tuple[str, np.ndarray]] = []
    for pattern, voigt_index, magnitudes in (
        ("reference", 0, (0.0,)),
        *((name, index, strain_magnitudes) for name, index in CUBIC_STRAIN_PATTERNS),
    ):
        for signed_magnitude in sorted(
            {sign * float(magnitude) for magnitude in magnitudes for sign in (1, -1)}
        ):
            voigt_strain = np.zeros(6)
            voigt_strain[voigt_index] = signed_magnitude
            strain_set.append((pattern, voigt_strain))
    return strain_set


def _zero_intercept_slope(
    points: list[dict[str, Any]], pattern: str, voigt_index: int, stress_components: list[int]
) -> float:
    """Least-squares slope through the origin of the mean of *stress_components* against one strain.

    Over a sign-symmetric set no even term in the strain -- nor the reference
    cell's own stress -- reaches it.
    """
    selected = [point for point in points if point["pattern"] == pattern]
    strain = np.array([point["voigt_strain"][voigt_index] for point in selected])
    stress = np.array([point["voigt_stress_gpa"] for point in selected])
    return float(strain @ stress[:, stress_components].mean(axis=1) / (strain @ strain))


def fit_cubic_elastic_constants(points: list[dict[str, Any]]) -> dict[str, float]:
    """C11, C12 and C44 in GPa from the stresses of a ``cubic_voigt_strain_set`` scan.

    Parameters
    ----------
    points:
        One entry per strained cell, each with its ``"pattern"``, its
        ``"voigt_strain"`` and the ``"voigt_stress_gpa"`` computed for it
        (tensile positive).  The ``"reference"`` point is not used by the fit.

    Returns
    -------
    dict
        ``"c11_gpa"``, ``"c12_gpa"`` and ``"c44_gpa"`` (ESSE has no schema for
        elastic constants).
    """
    return {
        "c11_gpa": _zero_intercept_slope(points, "uniaxial_x", 0, [0]),
        "c12_gpa": _zero_intercept_slope(points, "uniaxial_x", 0, [1, 2]),
        "c44_gpa": _zero_intercept_slope(points, "shear_yz", 3, [3]),
    }
