"""q3.properties — material properties derived from the quantities a run computes.

Mirrors the C++ ``layer_5_properties`` in scope: given energies and stresses
that some run already produced, it fits or differentiates them into a property
-- a Birch-Murnaghan equation of state, cubic elastic constants -- or gives
what a library computed (Phonopy's phonon frequencies and thermodynamics) the
shape of an ESSE property.  It runs nothing itself; the ``q3.workflows``
that produce the inputs import it, downward.

**Upstream candidate (mat3ra-prode).**  None of this is specific to Q3: it takes
plain arrays and returns plain numbers or ESSE properties, and the stack's
property package, mat3ra-prode, is its natural home.  It is kept small and
dependency-light so that it can move there as it is; TODOs.md §4 lists the
signatures.
"""

# TODO: move this module to mat3ra-prode, the stack's home for derived properties,
#   and import it from there (TODOs.md §4, items 5 and 6).

from q3.properties.elastic_constants import (
    DEFAULT_STRAIN_MAGNITUDES,
    cubic_voigt_strain_set,
    fit_cubic_elastic_constants,
)
from q3.properties.equation_of_state import (
    DEFAULT_LINEAR_SCALE_FACTORS,
    fit_birch_murnaghan,
)
from q3.properties.phonons import (
    force_constants_file,
    harmonic_thermal_properties,
    phonon_density_of_states,
    phonon_dispersions,
    phonon_frequencies_ev_to_thz_and_cm1,
    zero_point_energy,
)
from q3.properties.stress import (
    hydrostatic_pressure_gpa,
    pressure_gpa_to_ev_per_angstrom3,
    voigt_stress_gpa,
)

__all__ = [
    "DEFAULT_LINEAR_SCALE_FACTORS",
    "DEFAULT_STRAIN_MAGNITUDES",
    "cubic_voigt_strain_set",
    "fit_birch_murnaghan",
    "fit_cubic_elastic_constants",
    "force_constants_file",
    "harmonic_thermal_properties",
    "hydrostatic_pressure_gpa",
    "phonon_density_of_states",
    "phonon_dispersions",
    "phonon_frequencies_ev_to_thz_and_cm1",
    "pressure_gpa_to_ev_per_angstrom3",
    "voigt_stress_gpa",
    "zero_point_energy",
]
