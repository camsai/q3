"""tests/python/test_properties_phonons.py — Unit tests for q3.properties.phonons.

Tests the frequency-unit conversion and the ESSE shapes the phonon workflows
return, in isolation, without invoking Q3, ASE's Phonons or Phonopy (which are
well-tested external libraries; this module's own work is the eV -> THz /
cm^-1 conversion and the layout of each property).
"""

from pathlib import Path

import numpy as np
import pytest
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
from q3.properties.phonons import (
    force_constants_file,
    harmonic_thermal_properties,
    phonon_density_of_states,
    phonon_dispersions,
    phonon_frequencies_ev_to_thz_and_cm1,
    zero_point_energy,
)

# Well-known CODATA-derived conversion constants (NIST "Hz-eV relationship"
# and "eV-1/cm relationship"), independent of ase.units' own internal
# constant set -- this is what actually pins the conversion, not a
# tautological re-check of the same numbers phonons.py itself uses.
EV_TO_THZ_REFERENCE = 241.79893
EV_TO_CM1_REFERENCE = 8065.544


def test_one_ev_converts_to_known_thz_and_cm1():
    thz, cm1 = phonon_frequencies_ev_to_thz_and_cm1(np.array([1.0]))
    assert thz[0] == pytest.approx(EV_TO_THZ_REFERENCE, rel=1e-4)
    assert cm1[0] == pytest.approx(EV_TO_CM1_REFERENCE, rel=1e-4)


def test_conversion_is_linear_across_an_array():
    thz, cm1 = phonon_frequencies_ev_to_thz_and_cm1(np.array([0.0, 1.0, 2.0]))
    assert thz.tolist() == pytest.approx(
        [0.0, EV_TO_THZ_REFERENCE, 2 * EV_TO_THZ_REFERENCE], rel=1e-4
    )
    assert cm1.tolist() == pytest.approx(
        [0.0, EV_TO_CM1_REFERENCE, 2 * EV_TO_CM1_REFERENCE], rel=1e-4
    )


def test_negative_ev_stays_negative():
    """A negative frequency_ev (ASE's convention for an imaginary/unstable
    mode) must convert to a negative THz/cm^-1, not e.g. an absolute value.
    """
    thz, cm1 = phonon_frequencies_ev_to_thz_and_cm1(np.array([-0.001]))
    assert thz[0] < 0.0
    assert cm1[0] < 0.0


def test_matches_known_silicon_optical_phonon_order_of_magnitude():
    """Sanity anchor (not a Q3 validation): silicon's Gamma-optical phonon
    is experimentally ~15.53 THz (~518 cm^-1) -- confirms the conversion
    lands in the physically expected range for a typical semiconductor
    optical mode, independent of any Q3-computed value.
    """
    silicon_optical_ev = 0.0642  # ~15.5 THz, close to the experimental value
    thz, cm1 = phonon_frequencies_ev_to_thz_and_cm1(np.array([silicon_optical_ev]))
    assert 10.0 < thz[0] < 20.0
    assert 300.0 < cm1[0] < 700.0


# ── ESSE shapes ───────────────────────────────────────────────────────────────


def test_phonon_dispersions_is_one_series_per_branch():
    """Phonopy and ASE index [q-point][branch]; ESSE plots one series per branch."""
    qpoints = [[0.0, 0.0, 0.0], [0.5, 0.0, 0.5]]
    frequencies = [[0.0, 0.0, 15.0], [4.0, 4.0, 12.0]]
    dispersions = phonon_dispersions(qpoints, frequencies)
    PhononBandStructurePropertySchema.model_validate(dispersions)
    assert dispersions["name"] == "phonon_dispersions"
    assert dispersions["yAxis"]["units"] == "THz"
    assert dispersions["xDataArray"] == qpoints
    assert dispersions["yDataSeries"] == [[0.0, 4.0], [0.0, 4.0], [15.0, 12.0]]


def test_phonon_dispersions_at_gamma_in_wavenumbers():
    dispersions = phonon_dispersions([[0.0, 0.0, 0.0]], [np.array([0.0, 480.0])], units="cm-1")
    assert dispersions["yAxis"]["units"] == "cm-1"
    assert dispersions["yDataSeries"] == [[0.0], [480.0]]


def test_phonon_density_of_states_is_esse_phonon_dos():
    dos = phonon_density_of_states(np.array([0.0, 1.0, 2.0]), np.array([0.0, 0.5, 0.25]))
    PhononDensityOfStatesPropertySchema.model_validate(dos)
    assert dos["name"] == "phonon_dos"
    assert dos["xDataArray"] == [0.0, 1.0, 2.0]
    assert dos["yDataSeries"] == [[0.0, 0.5, 0.25]]


def test_zero_point_energy_is_esse_scalar_in_phonopy_units():
    energy = zero_point_energy(np.float64(11.2))
    ZeroPointEnergyPropertySchema.model_validate(energy)
    assert energy == {"name": "zero_point_energy", "units": "kJ/mol", "value": 11.2}


def test_harmonic_thermal_properties_are_esse_plots_against_temperature():
    """ESSE has no temperature-dependent property, so each quantity is its 2D plot."""
    thermal = harmonic_thermal_properties(
        np.array([0.0, 300.0]), [11.2, 6.0], [0.0, 30.0], [0.0, 40.5]
    )
    assert set(thermal) == {"free_energy", "entropy", "heat_capacity"}
    for plot in thermal.values():
        Field2DimensionPlotSchema.model_validate(plot)
        assert plot["xAxis"] == {"label": "temperature", "units": "K"}
        assert plot["xDataArray"] == [0.0, 300.0]
    assert thermal["free_energy"]["yAxis"] == {"label": "free_energy", "units": "kJ/mol"}
    assert thermal["heat_capacity"]["yAxis"]["units"] == "J/K/mol"
    assert thermal["heat_capacity"]["yDataSeries"] == [[0.0, 40.5]]


def test_force_constants_file_names_the_file_as_esse_file_content(tmp_path):
    path = tmp_path / "phonopy_params.yaml"
    file_content = force_constants_file(path)
    FileContentPropertySchema.model_validate(file_content)
    assert file_content["name"] == "file_content"
    assert file_content["filetype"] == "text"
    assert Path(file_content["pathname"]) / file_content["basename"] == path.resolve()
