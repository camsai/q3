"""tests/python/test_berkeleygw_format_reader.py

Unit tests for the independent BerkeleyGW mean-field file reader
(tests/helpers_berkeleygw.py), against synthetic files built by a small,
self-contained writer defined in this module.

Independence within this test file matters: the synthetic writer below is
built from plain `struct.pack`/`numpy.ndarray.tobytes()` primitives, and
never imports the reader's own private dtypes or helper functions. If the
test fixture were built by reusing the reader's internal record dtypes, a
bug shared between "how the fixture is written" and "how the reader parses
it" would cancel out and this suite would pass regardless. This tests the
READER only -- it says nothing about Q3's real C++ writer (see
tests/validation/test_berkeleygw_export.py for that end-to-end check, which
this file's docstring in tests/helpers_berkeleygw.py explains further).
"""

from __future__ import annotations

import io
import struct

import numpy as np
import pytest

from helpers_berkeleygw import (
    BerkeleyGWFormatError,
    read_berkeleygw_charge_density_file,
    read_berkeleygw_exchange_correlation_file,
    read_berkeleygw_wavefunction_file,
)

# ── Local, independent Fortran-record writer (test-only) ──────────────


def _write_record(file_handle, payload):
    """Writes one Fortran unformatted-sequential record: [len][payload][len]."""
    length_bytes = struct.pack("<i", len(payload))
    file_handle.write(length_bytes)
    file_handle.write(payload)
    file_handle.write(length_bytes)


def _pack_title_field(text, width=32):
    if len(text) > width:
        raise ValueError(f"{text!r} does not fit in {width} bytes")
    return text.ljust(width).encode("ascii")


def _fortran_bytes(array, dtype):
    """Flattens an array to bytes in Fortran (column-major) order."""
    return np.asarray(array, dtype=dtype).tobytes(order="F")


# ── Synthetic system: a small, deliberately asymmetric fixture ────────


def _make_synthetic_wfn_system():
    """A small, self-consistent-in-shape (not physically meaningful) WFN
    fixture. Every matrix uses distinct entries (no accidental symmetry) so
    a transpose or interleaving bug cannot hide behind coincidentally equal
    values.
    """
    return {
        "title": "WFN-Complex",
        "creation_date": "01-Jan-2026",
        "creation_time": "12:00:00",
        "is_complex_flavor": True,
        "number_of_spin_components": 1,
        "number_of_g_vectors": 5,
        "number_of_symmetry_operations": 2,
        "cell_symmetry_class": 0,
        "number_of_atoms": 2,
        "charge_density_cutoff_rydberg": 80.0,
        "number_of_kpoints": 2,
        "number_of_bands": 2,
        "maximum_g_vectors_per_kpoint": 4,
        "wavefunction_cutoff_rydberg": 20.0,
        "fft_grid_dimensions": np.array([8, 8, 8], dtype=np.int32),
        "kpoint_grid_dimensions": np.array([2, 1, 1], dtype=np.int32),
        "kpoint_grid_shifts": np.array([0.5, 0.0, 0.0], dtype=np.float64),
        "cell_volume_bohr3": 123.456,
        "lattice_parameter_bohr": 10.2,
        "lattice_vectors_alat_units": np.array(
            [[1.0, 4.0, 7.0], [2.0, 5.0, 8.0], [3.0, 6.0, 9.5]], dtype=np.float64
        ),
        "real_space_metric_tensor_bohr2": np.array(
            [[1.1, 2.2, 3.3], [4.4, 5.5, 6.6], [7.7, 8.8, 9.9]], dtype=np.float64
        ),
        "reciprocal_cell_volume_bohr3": 0.654321,
        "reciprocal_lattice_parameter_inverse_bohr": 0.6161,
        "reciprocal_lattice_vectors_tpiba_units": np.array(
            [[9.0, 8.0, 7.0], [6.0, 5.0, 4.0], [3.0, 2.0, 1.5]], dtype=np.float64
        ),
        "reciprocal_space_metric_tensor_inverse_bohr2": np.array(
            [[1.0, 0.1, 0.2], [0.3, 2.0, 0.4], [0.5, 0.6, 3.0]], dtype=np.float64
        ),
        "symmetry_rotation_matrices": np.stack(
            [
                np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.int32),
                np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=np.int32),
            ],
            axis=-1,
        ),
        "symmetry_fractional_translations_two_pi": np.array(
            [[0.0, 1.1], [0.0, 2.2], [0.0, 3.3]], dtype=np.float64
        ),
        "atom_positions_alat_units": np.array(
            [[0.1, 0.4], [0.2, 0.55], [0.3, 0.6]], dtype=np.float64
        ),
        "atomic_numbers": np.array([14, 6], dtype=np.int32),
        "number_of_gvectors_per_kpoint": np.array([3, 4], dtype=np.int32),
        "kpoint_weights": np.array([0.5, 0.5], dtype=np.float64),
        "kpoints_crystal_coordinates": np.array(
            [[0.0, 0.5], [0.0, 0.0], [0.0, 0.0]], dtype=np.float64
        ),
        "lowest_occupied_band_index": np.array([1, 1], dtype=np.int32),
        "highest_occupied_band_index": np.array([2, 2], dtype=np.int32),
        "band_energies_rydberg": np.array([[-1.0, -0.9], [0.5, 0.6]], dtype=np.float64),
        "band_occupations": np.array([[1.0, 1.0], [0.0, 0.0]], dtype=np.float64),
        "charge_density_gvectors": np.array(
            [[0, 1, -1, 0, 1], [0, 0, 0, 1, 1], [0, 0, 0, 0, 0]], dtype=np.int32
        ),
        "kpoint_gvectors": [
            np.array([[0, 1, -1], [0, 0, 0], [0, 0, 0]], dtype=np.int32),
            np.array([[0, 1, 0, 1], [0, 0, 1, 1], [0, 0, 0, 0]], dtype=np.int32),
        ],
        "wavefunction_coefficients": [
            np.array(
                [
                    [1 + 0j, 0.1 - 0.2j, 0.05 + 0.01j],
                    [0.2 + 0.1j, 1 - 0j, -0.3 + 0.4j],
                ],
                dtype=np.complex128,
            ),
            np.array(
                [
                    [0.9 + 0j, -0.1 + 0.05j, 0.02 - 0.03j, 0.01 + 0.01j],
                    [0.15 - 0.05j, 0.6 + 0.2j, -0.4 + 0.1j, 0.2 + 0.2j],
                ],
                dtype=np.complex128,
            ),
        ],
    }


def _write_synthetic_wfn(file_handle, system):
    """Writes ``system`` as a complete BerkeleyGW WFN file to ``file_handle``."""
    # Record 1.
    _write_record(
        file_handle,
        _pack_title_field(system["title"])
        + _pack_title_field(system["creation_date"])
        + _pack_title_field(system["creation_time"]),
    )
    # Record 2 (mixed int32/float64 -- built with plain struct.pack, not the
    # reader's structured dtype).
    _write_record(
        file_handle,
        struct.pack(
            "<5id3id",
            system["number_of_spin_components"],
            system["number_of_g_vectors"],
            system["number_of_symmetry_operations"],
            system["cell_symmetry_class"],
            system["number_of_atoms"],
            system["charge_density_cutoff_rydberg"],
            system["number_of_kpoints"],
            system["number_of_bands"],
            system["maximum_g_vectors_per_kpoint"],
            system["wavefunction_cutoff_rydberg"],
        ),
    )
    # Record 3.
    _write_record(
        file_handle,
        struct.pack("<6i", *system["fft_grid_dimensions"], *system["kpoint_grid_dimensions"])
        + struct.pack("<3d", *system["kpoint_grid_shifts"]),
    )
    _write_lattice_and_symmetry_records(file_handle, system)
    # Record 9.
    _write_record(file_handle, _fortran_bytes(system["number_of_gvectors_per_kpoint"], np.int32))
    # Record 10.
    _write_record(file_handle, _fortran_bytes(system["kpoint_weights"], np.float64))
    # Record 11.
    _write_record(file_handle, _fortran_bytes(system["kpoints_crystal_coordinates"], np.float64))
    # Records 12-13.
    _write_record(file_handle, _fortran_bytes(system["lowest_occupied_band_index"], np.int32))
    _write_record(file_handle, _fortran_bytes(system["highest_occupied_band_index"], np.int32))
    # Records 14-15.
    _write_record(file_handle, _fortran_bytes(system["band_energies_rydberg"], np.float64))
    _write_record(file_handle, _fortran_bytes(system["band_occupations"], np.float64))
    # Records 16-18.
    _write_record(file_handle, struct.pack("<i", 1))
    _write_record(file_handle, struct.pack("<i", system["number_of_g_vectors"]))
    _write_record(file_handle, _fortran_bytes(system["charge_density_gvectors"], np.int32))

    # Body.
    coefficient_dtype = np.complex128 if system["is_complex_flavor"] else np.float64
    for kpoint_index in range(system["number_of_kpoints"]):
        ngk = int(system["number_of_gvectors_per_kpoint"][kpoint_index])
        _write_record(file_handle, struct.pack("<i", 1))
        _write_record(file_handle, struct.pack("<i", ngk))
        _write_record(
            file_handle, _fortran_bytes(system["kpoint_gvectors"][kpoint_index], np.int32)
        )
        for band_index in range(system["number_of_bands"]):
            _write_record(file_handle, struct.pack("<i", 1))
            _write_record(file_handle, struct.pack("<i", ngk))
            coefficients = system["wavefunction_coefficients"][kpoint_index][band_index, :]
            _write_record(file_handle, np.asarray(coefficients, dtype=coefficient_dtype).tobytes())


def _write_lattice_and_symmetry_records(file_handle, system):
    """Writes records 4-8 (shared layout between WFN and RHO fixtures)."""
    # Record 4.
    _write_record(
        file_handle,
        struct.pack("<2d", system["cell_volume_bohr3"], system["lattice_parameter_bohr"])
        + _fortran_bytes(system["lattice_vectors_alat_units"], np.float64)
        + _fortran_bytes(system["real_space_metric_tensor_bohr2"], np.float64),
    )
    # Record 5.
    _write_record(
        file_handle,
        struct.pack(
            "<2d",
            system["reciprocal_cell_volume_bohr3"],
            system["reciprocal_lattice_parameter_inverse_bohr"],
        )
        + _fortran_bytes(system["reciprocal_lattice_vectors_tpiba_units"], np.float64)
        + _fortran_bytes(system["reciprocal_space_metric_tensor_inverse_bohr2"], np.float64),
    )
    # Record 6.
    _write_record(file_handle, _fortran_bytes(system["symmetry_rotation_matrices"], np.int32))
    # Record 7.
    _write_record(
        file_handle, _fortran_bytes(system["symmetry_fractional_translations_two_pi"], np.float64)
    )
    # Record 8: interleaved (position, atomic_number) per atom -- built atom-by-atom
    # with struct.pack, independent of the reader's structured dtype.
    positions = system["atom_positions_alat_units"]
    atomic_numbers = system["atomic_numbers"]
    interleaved = b"".join(
        struct.pack("<3di", *positions[:, atom_index], int(atomic_numbers[atom_index]))
        for atom_index in range(system["number_of_atoms"])
    )
    _write_record(file_handle, interleaved)


def _build_wfn_bytes(system):
    buffer = io.BytesIO()
    _write_synthetic_wfn(buffer, system)
    return buffer.getvalue()


# ── WFN: header + body round-trip ──────────────────────────────────────


def test_wfn_header_round_trips_every_field(tmp_path):
    system = _make_synthetic_wfn_system()
    path = tmp_path / "WFN"
    with open(path, "wb") as file_handle:
        _write_synthetic_wfn(file_handle, system)

    result = read_berkeleygw_wavefunction_file(path)

    scalar_fields = [
        "title",
        "creation_date",
        "creation_time",
        "is_complex_flavor",
        "number_of_spin_components",
        "number_of_g_vectors",
        "number_of_symmetry_operations",
        "cell_symmetry_class",
        "number_of_atoms",
        "charge_density_cutoff_rydberg",
        "number_of_kpoints",
        "number_of_bands",
        "maximum_g_vectors_per_kpoint",
        "wavefunction_cutoff_rydberg",
        "cell_volume_bohr3",
        "lattice_parameter_bohr",
        "reciprocal_cell_volume_bohr3",
        "reciprocal_lattice_parameter_inverse_bohr",
    ]
    for field_name in scalar_fields:
        assert result[field_name] == system[field_name], f"field {field_name!r} did not round-trip"

    array_fields = [
        "fft_grid_dimensions",
        "kpoint_grid_dimensions",
        "kpoint_grid_shifts",
        "lattice_vectors_alat_units",
        "real_space_metric_tensor_bohr2",
        "reciprocal_lattice_vectors_tpiba_units",
        "reciprocal_space_metric_tensor_inverse_bohr2",
        "symmetry_rotation_matrices",
        "symmetry_fractional_translations_two_pi",
        "atom_positions_alat_units",
        "atomic_numbers",
        "number_of_gvectors_per_kpoint",
        "kpoint_weights",
        "kpoints_crystal_coordinates",
        "lowest_occupied_band_index",
        "highest_occupied_band_index",
        "band_energies_rydberg",
        "band_occupations",
        "charge_density_gvectors",
    ]
    for field_name in array_fields:
        assert np.array_equal(result[field_name], system[field_name]), (
            f"array field {field_name!r} did not round-trip:\n"
            f"expected:\n{system[field_name]}\ngot:\n{result[field_name]}"
        )

    # Exact dtypes, not just equal values (HARD RULE: explicit numpy dtypes).
    assert result["symmetry_rotation_matrices"].dtype == np.int32
    assert result["atomic_numbers"].dtype == np.int32
    assert result["lattice_vectors_alat_units"].dtype == np.float64
    assert result["source_path"] == path


def test_wfn_body_round_trips_gvectors_and_coefficients(tmp_path):
    system = _make_synthetic_wfn_system()
    path = tmp_path / "WFN"
    with open(path, "wb") as file_handle:
        _write_synthetic_wfn(file_handle, system)

    result = read_berkeleygw_wavefunction_file(path)

    assert len(result["kpoint_gvectors"]) == system["number_of_kpoints"]
    assert len(result["wavefunction_coefficients"]) == system["number_of_kpoints"]
    for kpoint_index in range(system["number_of_kpoints"]):
        assert np.array_equal(
            result["kpoint_gvectors"][kpoint_index], system["kpoint_gvectors"][kpoint_index]
        )
        expected_coefficients = system["wavefunction_coefficients"][kpoint_index]
        actual_coefficients = result["wavefunction_coefficients"][kpoint_index]
        assert actual_coefficients.dtype == np.complex128
        assert np.array_equal(actual_coefficients, expected_coefficients)


# ── RHO: header + body round-trip ──────────────────────────────────────


def _make_synthetic_rho_system():
    wfn_system = _make_synthetic_wfn_system()
    return {
        "title": "RHO-Complex",
        "creation_date": wfn_system["creation_date"],
        "creation_time": wfn_system["creation_time"],
        "is_complex_flavor": True,
        "number_of_spin_components": 1,
        "number_of_g_vectors": wfn_system["number_of_g_vectors"],
        "number_of_symmetry_operations": wfn_system["number_of_symmetry_operations"],
        "cell_symmetry_class": wfn_system["cell_symmetry_class"],
        "number_of_atoms": wfn_system["number_of_atoms"],
        "charge_density_cutoff_rydberg": wfn_system["charge_density_cutoff_rydberg"],
        "fft_grid_dimensions": wfn_system["fft_grid_dimensions"],
        "cell_volume_bohr3": wfn_system["cell_volume_bohr3"],
        "lattice_parameter_bohr": wfn_system["lattice_parameter_bohr"],
        "lattice_vectors_alat_units": wfn_system["lattice_vectors_alat_units"],
        "real_space_metric_tensor_bohr2": wfn_system["real_space_metric_tensor_bohr2"],
        "reciprocal_cell_volume_bohr3": wfn_system["reciprocal_cell_volume_bohr3"],
        "reciprocal_lattice_parameter_inverse_bohr": wfn_system[
            "reciprocal_lattice_parameter_inverse_bohr"
        ],
        "reciprocal_lattice_vectors_tpiba_units": wfn_system[
            "reciprocal_lattice_vectors_tpiba_units"
        ],
        "reciprocal_space_metric_tensor_inverse_bohr2": wfn_system[
            "reciprocal_space_metric_tensor_inverse_bohr2"
        ],
        "symmetry_rotation_matrices": wfn_system["symmetry_rotation_matrices"],
        "symmetry_fractional_translations_two_pi": wfn_system[
            "symmetry_fractional_translations_two_pi"
        ],
        "atom_positions_alat_units": wfn_system["atom_positions_alat_units"],
        "atomic_numbers": wfn_system["atomic_numbers"],
        "charge_density_gvectors": wfn_system["charge_density_gvectors"],
        "charge_density_g_times_cell_volume": np.array(
            [[10.0 + 0j], [-0.5 + 0.1j], [-0.5 - 0.1j], [0.2 + 0j], [0.2 + 0j]],
            dtype=np.complex128,
        ),
    }


def _write_synthetic_rho(file_handle, system):
    _write_record(
        file_handle,
        _pack_title_field(system["title"])
        + _pack_title_field(system["creation_date"])
        + _pack_title_field(system["creation_time"]),
    )
    _write_record(
        file_handle,
        struct.pack(
            "<5id",
            system["number_of_spin_components"],
            system["number_of_g_vectors"],
            system["number_of_symmetry_operations"],
            system["cell_symmetry_class"],
            system["number_of_atoms"],
            system["charge_density_cutoff_rydberg"],
        ),
    )
    _write_record(file_handle, struct.pack("<3i", *system["fft_grid_dimensions"]))
    _write_lattice_and_symmetry_records(file_handle, system)

    number_of_g_vectors = system["number_of_g_vectors"]
    _write_record(file_handle, struct.pack("<i", 1))
    _write_record(file_handle, struct.pack("<i", number_of_g_vectors))
    _write_record(file_handle, _fortran_bytes(system["charge_density_gvectors"], np.int32))

    coefficient_dtype = np.complex128 if system["is_complex_flavor"] else np.float64
    _write_record(file_handle, struct.pack("<i", 1))
    _write_record(file_handle, struct.pack("<i", number_of_g_vectors))
    _write_record(
        file_handle,
        _fortran_bytes(system["charge_density_g_times_cell_volume"], coefficient_dtype),
    )


def test_rho_header_and_body_round_trip(tmp_path):
    system = _make_synthetic_rho_system()
    path = tmp_path / "RHO"
    with open(path, "wb") as file_handle:
        _write_synthetic_rho(file_handle, system)

    result = read_berkeleygw_charge_density_file(path)

    for field_name in [
        "title",
        "number_of_spin_components",
        "number_of_g_vectors",
        "number_of_symmetry_operations",
        "cell_symmetry_class",
        "number_of_atoms",
        "charge_density_cutoff_rydberg",
        "cell_volume_bohr3",
        "lattice_parameter_bohr",
        "reciprocal_cell_volume_bohr3",
        "reciprocal_lattice_parameter_inverse_bohr",
    ]:
        assert result[field_name] == system[field_name], f"field {field_name!r} did not round-trip"

    for field_name in [
        "fft_grid_dimensions",
        "lattice_vectors_alat_units",
        "real_space_metric_tensor_bohr2",
        "reciprocal_lattice_vectors_tpiba_units",
        "reciprocal_space_metric_tensor_inverse_bohr2",
        "symmetry_rotation_matrices",
        "symmetry_fractional_translations_two_pi",
        "atom_positions_alat_units",
        "atomic_numbers",
        "charge_density_gvectors",
        "charge_density_g_times_cell_volume",
    ]:
        assert np.array_equal(result[field_name], system[field_name]), (
            f"array field {field_name!r} did not round-trip"
        )
    assert result["charge_density_g_times_cell_volume"].dtype == np.complex128


# ── Deliberately malformed files ────────────────────────────────────────


def test_corrupted_trailing_record_length_raises(tmp_path):
    """Record 1 is exactly [4-byte len][96-byte payload][4-byte len] -- flip
    the trailing marker only, leaving everything else (including the
    payload) untouched, and check the error names the record.
    """
    system = _make_synthetic_wfn_system()
    raw_bytes = bytearray(_build_wfn_bytes(system))
    trailing_length_offset = 4 + 96  # record 1's trailing length marker.
    raw_bytes[trailing_length_offset : trailing_length_offset + 4] = struct.pack("<i", 999)
    path = tmp_path / "WFN"
    path.write_bytes(bytes(raw_bytes))

    with pytest.raises(BerkeleyGWFormatError, match=r"record #1"):
        read_berkeleygw_wavefunction_file(path)


def test_truncated_file_raises(tmp_path):
    system = _make_synthetic_wfn_system()
    raw_bytes = _build_wfn_bytes(system)
    truncated_bytes = raw_bytes[: len(raw_bytes) // 2]
    path = tmp_path / "WFN"
    path.write_bytes(truncated_bytes)

    with pytest.raises(BerkeleyGWFormatError):
        read_berkeleygw_wavefunction_file(path)


def test_truncated_file_raises_even_at_a_record_boundary(tmp_path):
    """Truncating exactly between two records (not mid-payload) must still
    raise -- the next record's leading length marker is simply absent.
    """
    system = _make_synthetic_wfn_system()
    raw_bytes = _build_wfn_bytes(system)
    # Record 1 ends at byte 104 (4 + 96 + 4); cut exactly there.
    path = tmp_path / "WFN"
    path.write_bytes(raw_bytes[:104])

    with pytest.raises(BerkeleyGWFormatError, match=r"end of file"):
        read_berkeleygw_wavefunction_file(path)


def test_unrecognized_stitle_raises(tmp_path):
    system = _make_synthetic_wfn_system()
    system["title"] = "GARBAGE-Flavor"
    path = tmp_path / "WFN"
    with open(path, "wb") as file_handle:
        _write_synthetic_wfn(file_handle, system)

    with pytest.raises(BerkeleyGWFormatError, match=r"stitle"):
        read_berkeleygw_wavefunction_file(path)


def test_spin_polarized_body_raises_explicit_error(tmp_path):
    """ns > 1 uses a structurally different (spin-interleaved) body layout
    that this reader intentionally does not attempt to parse (see
    tests/helpers_berkeleygw.py's docstring); it must fail loudly, not
    silently misparse.
    """
    system = _make_synthetic_wfn_system()
    system["number_of_spin_components"] = 2
    system["number_of_kpoints"] = 1
    system["number_of_bands"] = 1
    system["number_of_gvectors_per_kpoint"] = np.array([3], dtype=np.int32)
    system["maximum_g_vectors_per_kpoint"] = 3
    system["kpoint_weights"] = np.array([1.0], dtype=np.float64)
    system["kpoints_crystal_coordinates"] = np.array([[0.0], [0.0], [0.0]], dtype=np.float64)
    system["lowest_occupied_band_index"] = np.array([1, 1], dtype=np.int32)
    system["highest_occupied_band_index"] = np.array([1, 1], dtype=np.int32)
    system["band_energies_rydberg"] = np.array([[-1.0, -0.9]], dtype=np.float64)
    system["band_occupations"] = np.array([[1.0, 1.0]], dtype=np.float64)
    # No body needed: the reader must raise before consuming any body bytes.
    system["kpoint_gvectors"] = []
    system["wavefunction_coefficients"] = []
    path = tmp_path / "WFN"

    # Only the 18 header records are written below -- ns=2 makes any body
    # layout this fixture could produce structurally wrong anyway (see
    # read_berkeleygw_wavefunction_file's docstring), so this test does not
    # attempt one; the reader must raise before consuming any body bytes.
    with open(path, "wb") as file_handle:
        _write_record(
            file_handle,
            _pack_title_field(system["title"])
            + _pack_title_field(system["creation_date"])
            + _pack_title_field(system["creation_time"]),
        )
        _write_record(
            file_handle,
            struct.pack(
                "<5id3id",
                system["number_of_spin_components"],
                system["number_of_g_vectors"],
                system["number_of_symmetry_operations"],
                system["cell_symmetry_class"],
                system["number_of_atoms"],
                system["charge_density_cutoff_rydberg"],
                system["number_of_kpoints"],
                system["number_of_bands"],
                system["maximum_g_vectors_per_kpoint"],
                system["wavefunction_cutoff_rydberg"],
            ),
        )
        _write_record(
            file_handle,
            struct.pack("<6i", *system["fft_grid_dimensions"], *system["kpoint_grid_dimensions"])
            + struct.pack("<3d", *system["kpoint_grid_shifts"]),
        )
        _write_lattice_and_symmetry_records(file_handle, system)
        _write_record(
            file_handle, _fortran_bytes(system["number_of_gvectors_per_kpoint"], np.int32)
        )
        _write_record(file_handle, _fortran_bytes(system["kpoint_weights"], np.float64))
        _write_record(
            file_handle, _fortran_bytes(system["kpoints_crystal_coordinates"], np.float64)
        )
        _write_record(file_handle, _fortran_bytes(system["lowest_occupied_band_index"], np.int32))
        _write_record(file_handle, _fortran_bytes(system["highest_occupied_band_index"], np.int32))
        _write_record(file_handle, _fortran_bytes(system["band_energies_rydberg"], np.float64))
        _write_record(file_handle, _fortran_bytes(system["band_occupations"], np.float64))
        _write_record(file_handle, struct.pack("<i", 1))
        _write_record(file_handle, struct.pack("<i", system["number_of_g_vectors"]))
        _write_record(file_handle, _fortran_bytes(system["charge_density_gvectors"], np.int32))

    with pytest.raises(BerkeleyGWFormatError, match=r"number_of_spin_components"):
        read_berkeleygw_wavefunction_file(path)


# ── Mixed-type record 2 and interleaved atom record 8, in isolation ────


def test_mixed_type_header_record_parsed_correctly(tmp_path):
    """Record 2 packs int32 x5, float64, int32 x3, float64 into ONE record.
    Uses distinct, easily-confused-if-misaligned values (e.g. an int32 that
    would look like a plausible float64 fragment) to make a byte-offset bug
    show up as a wrong VALUE, not just a crash.
    """
    system = _make_synthetic_wfn_system()
    system["number_of_spin_components"] = 1
    system["number_of_g_vectors"] = 5
    system["number_of_symmetry_operations"] = 2
    system["cell_symmetry_class"] = 1
    system["number_of_atoms"] = 2
    system["charge_density_cutoff_rydberg"] = 271.50
    system["number_of_kpoints"] = 2
    system["number_of_bands"] = 2
    system["maximum_g_vectors_per_kpoint"] = 4
    system["wavefunction_cutoff_rydberg"] = 33.25
    path = tmp_path / "WFN"
    with open(path, "wb") as file_handle:
        _write_synthetic_wfn(file_handle, system)

    result = read_berkeleygw_wavefunction_file(path)

    assert result["number_of_spin_components"] == 1
    assert result["number_of_g_vectors"] == 5
    assert result["number_of_symmetry_operations"] == 2
    assert result["cell_symmetry_class"] == 1
    assert result["number_of_atoms"] == 2
    assert result["charge_density_cutoff_rydberg"] == 271.5
    assert isinstance(result["charge_density_cutoff_rydberg"], float)
    assert result["number_of_kpoints"] == 2
    assert result["number_of_bands"] == 2
    assert result["maximum_g_vectors_per_kpoint"] == 4
    assert result["wavefunction_cutoff_rydberg"] == 33.25
    assert isinstance(result["wavefunction_cutoff_rydberg"], float)


def test_interleaved_atom_record_parsed_correctly(tmp_path):
    """Record 8 interleaves (3 float64, 1 int32) per atom. Distinct atoms
    with distinct positions AND distinct atomic numbers catch an
    off-by-one-atom (rather than off-by-one-field) bug too.
    """
    system = _make_synthetic_wfn_system()
    system["number_of_atoms"] = 3
    system["atom_positions_alat_units"] = np.array(
        [[0.11, 0.22, 0.33], [0.44, 0.55, 0.66], [0.77, 0.88, 0.99]], dtype=np.float64
    )
    system["atomic_numbers"] = np.array([1, 14, 79], dtype=np.int32)  # H, Si, Au
    path = tmp_path / "WFN"
    with open(path, "wb") as file_handle:
        _write_synthetic_wfn(file_handle, system)

    result = read_berkeleygw_wavefunction_file(path)

    assert np.array_equal(result["atomic_numbers"], np.array([1, 14, 79], dtype=np.int32))
    assert np.array_equal(
        result["atom_positions_alat_units"],
        np.array([[0.11, 0.22, 0.33], [0.44, 0.55, 0.66], [0.77, 0.88, 0.99]], dtype=np.float64),
    )
    # Cross-check per atom explicitly (position column i belongs to atom i).
    for atom_index, expected_z in enumerate([1, 14, 79]):
        assert int(result["atomic_numbers"][atom_index]) == expected_z


# ── Column-major flattening, isolated and explicit ─────────────────────


def test_column_major_flattening_recovers_asymmetric_matrix_exactly(tmp_path):
    """All nine entries of ``avec`` are distinct, including off-diagonal
    pairs (e.g. avec[0,1] != avec[1,0]). Reshaping with order="C" instead of
    "F" would silently return the transpose rather than erroring, so exact
    equality -- and the explicit anti-assertion against the transpose --
    is the right check, not just "some matrix came back".
    """
    system = _make_synthetic_wfn_system()
    asymmetric_matrix = np.array(
        [[11.0, 12.0, 13.0], [21.0, 22.0, 23.0], [31.0, 32.0, 33.5]], dtype=np.float64
    )
    assert not np.array_equal(
        asymmetric_matrix, asymmetric_matrix.T
    )  # sanity on the fixture itself
    system["lattice_vectors_alat_units"] = asymmetric_matrix
    path = tmp_path / "WFN"
    with open(path, "wb") as file_handle:
        _write_synthetic_wfn(file_handle, system)

    result = read_berkeleygw_wavefunction_file(path)

    assert np.array_equal(result["lattice_vectors_alat_units"], asymmetric_matrix)
    assert not np.array_equal(result["lattice_vectors_alat_units"], asymmetric_matrix.T)


def test_column_major_flattening_of_3d_symmetry_matrix_stack(tmp_path):
    """Same check as above, one dimension up: mtrx has shape (3,3,ntran).
    Uses two different, individually-asymmetric 3x3 blocks so a bug that
    mixes up the ntran axis with either in-matrix axis is caught.
    """
    system = _make_synthetic_wfn_system()
    first_matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.int32)
    second_matrix = np.array([[9, 8, 7], [6, 5, 4], [3, 2, 1]], dtype=np.int32)
    system["symmetry_rotation_matrices"] = np.stack([first_matrix, second_matrix], axis=-1)
    path = tmp_path / "WFN"
    with open(path, "wb") as file_handle:
        _write_synthetic_wfn(file_handle, system)

    result = read_berkeleygw_wavefunction_file(path)

    assert np.array_equal(result["symmetry_rotation_matrices"][:, :, 0], first_matrix)
    assert np.array_equal(result["symmetry_rotation_matrices"][:, :, 1], second_matrix)


# ── vxc.dat (ASCII) ──────────────────────────────────────────────────────


def _format_fixed_width(value, width, decimals=None):
    text = str(int(value)) if decimals is None else f"{value:.{decimals}f}"
    if len(text) > width:
        raise ValueError(f"{text!r} does not fit in a {width}-character field")
    return text.rjust(width)


def _write_vxc_header_line(file_handle, kx, ky, kz, ndiag, noffdiag):
    line = (
        _format_fixed_width(kx, 13, 9)
        + _format_fixed_width(ky, 13, 9)
        + _format_fixed_width(kz, 13, 9)
        + _format_fixed_width(ndiag, 8)
        + _format_fixed_width(noffdiag, 8)
    )
    file_handle.write(line + "\n")


def _write_vxc_diagonal_line(file_handle, spin, band, real_part, imaginary_part):
    line = (
        _format_fixed_width(spin, 8)
        + _format_fixed_width(band, 8)
        + _format_fixed_width(real_part, 15, 9)
        + _format_fixed_width(imaginary_part, 15, 9)
    )
    file_handle.write(line + "\n")


def _write_vxc_offdiagonal_line(file_handle, spin, band_bra, band_ket, real_part, imaginary_part):
    line = (
        _format_fixed_width(spin, 8)
        + _format_fixed_width(band_bra, 8)
        + _format_fixed_width(band_ket, 8)
        + _format_fixed_width(real_part, 15, 9)
        + _format_fixed_width(imaginary_part, 15, 9)
    )
    file_handle.write(line + "\n")


def test_vxc_dat_round_trips_diagonal_and_offdiagonal_entries(tmp_path):
    path = tmp_path / "vxc.dat"
    with open(path, "w", encoding="ascii") as file_handle:
        _write_vxc_header_line(file_handle, 0.0, 0.0, 0.0, 2, 0)
        _write_vxc_diagonal_line(file_handle, 1, 1, -5.123456789, 0.0)
        _write_vxc_diagonal_line(file_handle, 1, 2, 3.987654321, 0.0)
        _write_vxc_header_line(file_handle, 0.5, 0.0, 0.0, 2, 1)
        _write_vxc_diagonal_line(file_handle, 1, 1, -4.5, 0.0)
        _write_vxc_diagonal_line(file_handle, 1, 2, 2.5, 0.0)
        _write_vxc_offdiagonal_line(file_handle, 1, 1, 2, 0.123456789, -0.987654321)

    result = read_berkeleygw_exchange_correlation_file(path)

    assert result["number_of_kpoints"] == 2
    first_kpoint = result["kpoints"][0]
    assert first_kpoint["kpoint_crystal_coordinates"] == pytest.approx([0.0, 0.0, 0.0])
    assert first_kpoint["diagonal_entry_count"] == 2
    assert first_kpoint["offdiagonal_entry_count"] == 0
    assert first_kpoint["diagonal_entries"][0]["spin_index"] == 1
    assert first_kpoint["diagonal_entries"][0]["band_index"] == 1
    assert first_kpoint["diagonal_entries"][0]["vxc_ev"] == pytest.approx(-5.123456789 + 0.0j)
    assert first_kpoint["diagonal_entries"][1]["vxc_ev"] == pytest.approx(3.987654321 + 0.0j)

    second_kpoint = result["kpoints"][1]
    assert second_kpoint["kpoint_crystal_coordinates"] == pytest.approx([0.5, 0.0, 0.0])
    assert second_kpoint["offdiagonal_entry_count"] == 1
    offdiagonal_entry = second_kpoint["offdiagonal_entries"][0]
    assert offdiagonal_entry["spin_index"] == 1
    assert offdiagonal_entry["band_index_bra"] == 1
    assert offdiagonal_entry["band_index_ket"] == 2
    assert offdiagonal_entry["vxc_ev"] == pytest.approx(0.123456789 - 0.987654321j)


def test_vxc_dat_truncated_line_raises(tmp_path):
    path = tmp_path / "vxc.dat"
    with open(path, "w", encoding="ascii") as file_handle:
        _write_vxc_header_line(file_handle, 0.0, 0.0, 0.0, 1, 0)
        file_handle.write("short line\n")

    with pytest.raises(BerkeleyGWFormatError):
        read_berkeleygw_exchange_correlation_file(path)
