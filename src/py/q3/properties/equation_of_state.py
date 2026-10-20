"""q3.properties.equation_of_state — the third-order Birch-Murnaghan fit of an E(V) scan.

The fit itself is ASE's ``EquationOfState(..., eos="birchmurnaghan")``
(Birch, Phys. Rev. 71, 809 (1947)); this module states the default scan and
returns the result as ESSE properties where a schema exists.  Requires ASE.

**Upstream candidate (mat3ra-prode)**, with the rest of ``q3.properties``.
"""

# TODO: move this module to mat3ra-prode, the stack's home for derived properties,
#   and import it from there (TODOs.md §4, item 5).

from __future__ import annotations

from typing import Any

from ase.eos import EquationOfState
from mat3ra.code.constants import Coefficients as _Coefficients
from mat3ra.esse.models.properties_directory.scalar.total_energy import TotalEnergyPropertySchema
from mat3ra.esse.models.properties_directory.structural.volume import VolumeSchema

#: Multipliers on the lattice vectors (volume scales as the cube): +-4 % in
#: length, +-12 % in volume -- wide enough to resolve the cubic term B0' needs,
#: narrow enough to stay where a third-order expansion holds.
DEFAULT_LINEAR_SCALE_FACTORS = (0.96, 0.97, 0.98, 0.99, 1.00, 1.01, 1.02, 1.03, 1.04)


def fit_birch_murnaghan(volumes_angstrom3, energies_ev) -> dict[str, Any]:
    """Fit a third-order Birch-Murnaghan equation of state to E(V).

    Returns
    -------
    dict
        ``"volume"`` (V0) and ``"total_energy"`` (E0) as ESSE properties, and
        ``"bulk_modulus_gpa"`` and ``"bulk_modulus_pressure_derivative"`` (B0
        and B0', for which ESSE has no schema) as plain numbers.
    """
    equation_of_state = EquationOfState(volumes_angstrom3, energies_ev, eos="birchmurnaghan")
    equilibrium_volume, equilibrium_energy, bulk_modulus = equation_of_state.fit()
    return {
        "volume": VolumeSchema(
            name="volume", units="angstrom^3", value=equilibrium_volume
        ).model_dump(mode="json"),
        "total_energy": TotalEnergyPropertySchema(
            name="total_energy", units="eV", value=equilibrium_energy
        ).model_dump(mode="json"),
        "bulk_modulus_gpa": float(bulk_modulus) * _Coefficients.EV_PER_ANGSTROM3_TO_GPA,
        "bulk_modulus_pressure_derivative": float(equation_of_state.eos_parameters[2]),
    }
