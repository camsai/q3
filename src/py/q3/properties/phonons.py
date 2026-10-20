"""q3.properties.phonons — phonon frequencies and harmonic thermodynamics as ESSE properties.

The physics is the libraries': ASE's ``Phonons`` and Phonopy compute the
frequencies, the density of states and the thermal properties, and the
workflows in ``q3.workflows.phonons`` drive them.  This module only gives their
arrays the shape the stack reads, and it uses neither library beyond ASE's unit
constants -- it takes plain arrays, so it can move upstream as it is.

ESSE coverage, checked against mat3ra-esse 2026.10.8:

* ``phonon_dispersions``, ``phonon_dos`` and ``zero_point_energy`` exist and
  are used as they are.
* **Temperature-dependent thermal properties have no schema.**  ESSE's
  ``heat_capacity_cv`` and ``total_entropy`` are single scalars in cal/mol-K
  that do not record the temperature they hold at (they serve molecular
  thermochemistry), and there is no free-energy property at all.  A harmonic
  phonon calculation gives each of the three as a function of temperature, so
  ``harmonic_thermal_properties`` returns them as ESSE's abstract two-dimensional
  plot (``core/abstract/2d_plot``) -- the base that ``phonon_dos`` and
  ``phonon_dispersions`` themselves extend -- with temperature on the x axis.
  That is the shape a future ESSE property would take, without a property
  ``name`` ESSE does not define.
* **Force constants have no schema.**  They stay in the file Phonopy writes,
  ``phonopy_params.yaml``, which ``force_constants_file`` reports as ESSE's
  ``file_content`` property: the stack's way of naming a result held in a file.

**Upstream candidate (mat3ra-prode)**, with the rest of ``q3.properties``.
"""

# TODO: move this module to mat3ra-prode, the stack's home for derived properties,
#   and import it from there (TODOs.md §4, item 6).  Once ESSE gains schemas for
#   temperature-dependent thermal properties and for force constants, return those.

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from ase import units as ase_units
from mat3ra.esse.models.core.abstract._2d_plot import Field2DimensionPlotSchema
from mat3ra.esse.models.properties_directory.non_scalar.file_content import (
    FileContentPropertySchema,
)
from mat3ra.esse.models.properties_directory.non_scalar.phonon_dispersions import (
    PhononBandStructurePropertySchema,
)
from mat3ra.esse.models.properties_directory.non_scalar.phonon_dos import (
    PhononDensityOfStatesPropertySchema,
)
from mat3ra.esse.models.properties_directory.scalar.zero_point_energy import (
    ZeroPointEnergyPropertySchema,
)


def phonon_frequencies_ev_to_thz_and_cm1(frequencies_ev: Any) -> tuple[Any, Any]:
    """Convert phonon frequencies from eV (ASE's ``Phonons`` convention) to THz and cm^-1.

    eV -> THz via E = h*f (h from ``ase.units``, SI Planck constant divided
    by the elementary charge to convert the eV numerator to Joules).
    eV -> cm^-1 via ``ase.units.invcm``, which is already "the energy of a
    1 cm^-1 photon" expressed in ASE's own eV-based unit system -- dividing
    by it directly gives cm^-1.
    """
    ev_to_hz = ase_units._e / ase_units._hplanck
    frequencies_thz = frequencies_ev * ev_to_hz / 1e12
    frequencies_cm1 = frequencies_ev / ase_units.invcm
    return frequencies_thz, frequencies_cm1


def phonon_dispersions(qpoints, frequencies, units: str = "THz") -> dict[str, Any]:
    """ESSE ``phonon_dispersions``: frequencies along a list of q-points.

    *qpoints* are in crystal coordinates; *frequencies* are indexed
    ``[q-point][branch]``, as Phonopy and ASE give them.  ESSE plots one series
    per branch, so ``yDataSeries`` is the transpose.  *units* is ``"THz"`` or
    ``"cm-1"``.
    """
    return PhononBandStructurePropertySchema(
        name="phonon_dispersions",
        xAxis={"label": "qpoints", "units": "crystal"},
        yAxis={"label": "frequency", "units": units},
        xDataArray=[[float(component) for component in qpoint] for qpoint in qpoints],
        yDataSeries=np.transpose(np.asarray(frequencies, dtype=float)).tolist(),
    ).model_dump(mode="json")


def phonon_density_of_states(frequency_points_thz, states_per_thz) -> dict[str, Any]:
    """ESSE ``phonon_dos``: the total density of states (states/THz per unit cell)."""
    return PhononDensityOfStatesPropertySchema(
        name="phonon_dos",
        xAxis={"label": "frequency", "units": "THz"},
        yAxis={"label": "Phonon DOS", "units": "states/THz"},
        xDataArray=np.asarray(frequency_points_thz, dtype=float).tolist(),
        yDataSeries=[np.asarray(states_per_thz, dtype=float).tolist()],
    ).model_dump(mode="json")


def zero_point_energy(value: float, units: str = "kJ/mol") -> dict[str, Any]:
    """ESSE ``zero_point_energy``; Phonopy's unit, kJ/mol of unit cells, by default."""
    return ZeroPointEnergyPropertySchema(
        name="zero_point_energy", units=units, value=float(value)
    ).model_dump(mode="json")


def harmonic_thermal_properties(
    temperatures_kelvin,
    free_energy_kj_per_mol,
    entropy_j_per_mol_per_kelvin,
    heat_capacity_j_per_mol_per_kelvin,
) -> dict[str, dict[str, Any]]:
    """Harmonic free energy, entropy and heat capacity against temperature.

    ESSE has no property for any of the three as a function of temperature
    (see the module docstring), so each is ESSE's abstract two-dimensional plot
    -- ``xAxis`` temperature in K, one series in ``yDataSeries`` -- keyed
    ``"free_energy"`` (kJ/mol), ``"entropy"`` and ``"heat_capacity"`` (J/K/mol):
    Phonopy's units, per mole of unit cells.
    """
    temperatures = np.asarray(temperatures_kelvin, dtype=float).tolist()

    def plot(label: str, units: str, values) -> dict[str, Any]:
        return Field2DimensionPlotSchema(
            xAxis={"label": "temperature", "units": "K"},
            yAxis={"label": label, "units": units},
            xDataArray=temperatures,
            yDataSeries=[np.asarray(values, dtype=float).tolist()],
        ).model_dump(mode="json")

    return {
        "free_energy": plot("free_energy", "kJ/mol", free_energy_kj_per_mol),
        "entropy": plot("entropy", "J/K/mol", entropy_j_per_mol_per_kelvin),
        "heat_capacity": plot("heat_capacity", "J/K/mol", heat_capacity_j_per_mol_per_kelvin),
    }


def force_constants_file(path: str | Path) -> dict[str, Any]:
    """ESSE ``file_content`` naming the file that holds the force constants.

    ESSE has no force-constants schema; the file -- Phonopy's
    ``phonopy_params.yaml``, read back with ``phonopy.load`` -- is the result.
    ``pathname`` is the directory, absolute because a local run has no job
    root to be relative to; ``objectData`` is empty because nothing was
    uploaded to object storage.
    """
    path = Path(path).expanduser().resolve()
    return FileContentPropertySchema(
        name="file_content",
        filetype="text",
        objectData={},
        pathname=str(path.parent),
        basename=path.name,
    ).model_dump(mode="json", exclude_none=True)
