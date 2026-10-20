"""q3.properties.stress — stress and pressure in the units a report is read in.

The stress arrives in ASE's convention -- Voigt order, eV/A^3, tensile positive
-- because that is what ``Q3ASECalculator.get_stress()`` returns; a pressure
leaves compression positive, the sign Quantum ESPRESSO and Q3's own output use.

**Upstream candidate (mat3ra-prode)**, with the rest of ``q3.properties``.
"""

# TODO: move this module to mat3ra-prode, the stack's home for derived properties,
#   and import it from there (TODOs.md §4, item 5).

from __future__ import annotations

import numpy as np
from mat3ra.code.constants import Coefficients as _Coefficients


def voigt_stress_gpa(voigt_stress_ev_per_angstrom3) -> np.ndarray:
    """A Voigt stress in eV/A^3 converted to GPa, sign and order unchanged."""
    return np.asarray(voigt_stress_ev_per_angstrom3) * _Coefficients.EV_PER_ANGSTROM3_TO_GPA


def hydrostatic_pressure_gpa(voigt_stress_ev_per_angstrom3) -> float:
    """Pressure in GPa, compression positive, of a tensile-positive Voigt stress in eV/A^3."""
    stress = np.asarray(voigt_stress_ev_per_angstrom3)
    return -float(np.mean(stress[:3])) * _Coefficients.EV_PER_ANGSTROM3_TO_GPA


def pressure_gpa_to_ev_per_angstrom3(pressure_gpa: float) -> float:
    """A pressure in GPa in ASE's eV/A^3 -- the unit its cell filters take ``scalar_pressure`` in."""
    return pressure_gpa / _Coefficients.EV_PER_ANGSTROM3_TO_GPA
