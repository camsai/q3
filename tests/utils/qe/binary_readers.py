"""Fortran unformatted binary readers for QE-format output files.

Provides readers for charge-density.dat and wfcN.dat files produced
by Quantum ESPRESSO and q3 in Fortran unformatted sequential format.
"""

import struct

import numpy as np

# QE's write_wfc header ``ik, xk, ispin, gamma_only, scalef``: 4 + 24 + 4 + 4 + 8.
KPOINT_HEADER_BYTES = 44


def _interleaved_miller_indices(record, number_of_vectors):
    """De-flatten a Fortran ``mill(3, n)`` record into an (n, 3) array."""
    miller_flat = np.frombuffer(record, dtype=np.int32)
    assert len(miller_flat) == 3 * number_of_vectors, (
        f"Expected {3 * number_of_vectors} miller indices, got {len(miller_flat)}"
    )
    return miller_flat.reshape(number_of_vectors, 3)


# ── Fortran unformatted record reader ────────────────────────────────


def read_record(file_handle):
    """Read a single Fortran unformatted record.

    Format: [4-byte length] [data] [4-byte length]
    Returns the raw bytes of the data payload.
    """
    header = file_handle.read(4)
    if len(header) < 4:
        raise EOFError("Unexpected end of file reading record header")
    number_of_bytes = struct.unpack("<i", header)[0]
    data = file_handle.read(number_of_bytes)
    if len(data) != number_of_bytes:
        raise EOFError(f"Expected {number_of_bytes} bytes, got {len(data)}")
    trailer = file_handle.read(4)
    number_of_bytes_check = struct.unpack("<i", trailer)[0]
    assert number_of_bytes == number_of_bytes_check, (
        f"Record length mismatch: header={number_of_bytes}, trailer={number_of_bytes_check}"
    )
    return data


# ── Charge density reader ───────────────────────────────────────────


def read_charge_density(path):
    """Read a QE-format charge-density.dat file.

    Returns:
        dict with keys: gamma_only, number_of_g_vectors,
            number_of_spin_components, reciprocal_vectors (1/Bohr),
            miller_indices (n, 3), density_g_space (first spin record),
            density_g_space_per_spin
    """
    with open(path, "rb") as file_handle:
        # Record 1: gamma_only, ngm_g, nspin
        record_1 = read_record(file_handle)
        gamma_only, number_of_g_vectors, number_of_spin_components = struct.unpack("<3i", record_1)

        # Record 2: b1, b2, b3 in 1/Bohr
        record_2 = read_record(file_handle)
        reciprocal_vectors = np.frombuffer(record_2, dtype=np.float64).reshape(3, 3)

        # Record 3: Miller indices mill_g(3, ngm_g) -- Fortran column-major,
        # component index fastest, so the flat array is h0 k0 l0 h1 k1 l1 ...
        record_3 = read_record(file_handle)
        miller_indices = _interleaved_miller_indices(record_3, number_of_g_vectors)

        # Records 4..: rho(G) per spin component — complex128 arrays
        density_records = []
        for _spin_index in range(number_of_spin_components):
            record = np.frombuffer(read_record(file_handle), dtype=np.complex128)
            assert len(record) == number_of_g_vectors
            density_records.append(record)
        density_g_space = density_records[0]

    return {
        "gamma_only": gamma_only,
        "number_of_g_vectors": number_of_g_vectors,
        "number_of_spin_components": number_of_spin_components,
        "reciprocal_vectors": reciprocal_vectors,
        "miller_indices": miller_indices,
        "density_g_space": density_g_space,
        "density_g_space_per_spin": density_records,
    }


# ── Wavefunction reader ─────────────────────────────────────────────


def read_wavefunction(path):
    """Read a QE-format wfcN.dat file.

    Returns:
        dict with keys: kpoint_index, kpoint_coordinates, spin_index,
            gamma_only, scale_factor, number_of_plane_waves,
            max_plane_waves, number_of_polarizations,
            number_of_bands, reciprocal_vectors,
            miller_indices, coefficients
    """
    with open(path, "rb") as file_handle:
        # Record 1: ik, xk(3), ispin, gamma_only, scalef -- a packed Fortran
        # list: int32 @0, 3 x float64 @4, int32 @28, LOGICAL(4) @32, float64 @36.
        record_1 = read_record(file_handle)
        assert len(record_1) == KPOINT_HEADER_BYTES, (
            f"Expected a {KPOINT_HEADER_BYTES}-byte packed k-point header, got {len(record_1)}"
        )
        (kpoint_index, kx, ky, kz, spin_index, gamma_only, scale_factor) = struct.unpack(
            "<i3diid", record_1
        )
        kpoint_coordinates = (kx, ky, kz)

        # Record 2: ngw, igwx, npol, nbnd
        record_2 = read_record(file_handle)
        (number_of_plane_waves, max_plane_waves, number_of_polarizations, number_of_bands) = (
            struct.unpack("<4i", record_2)
        )

        # Record 3: b1(3), b2(3), b3(3)
        record_3 = read_record(file_handle)
        reciprocal_vectors = np.frombuffer(record_3, dtype=np.float64).reshape(3, 3)

        # Record 4: Miller indices mill(3, igwx), interleaved per G-vector
        record_4 = read_record(file_handle)
        miller_indices = _interleaved_miller_indices(record_4, max_plane_waves)

        # Records 5..5+nbnd-1: band coefficients
        coefficients = []
        for _band_index in range(number_of_bands):
            record = read_record(file_handle)
            band_coefficients = np.frombuffer(record, dtype=np.complex128)
            assert len(band_coefficients) == number_of_polarizations * max_plane_waves
            coefficients.append(band_coefficients)
        coefficients = np.array(coefficients)

    return {
        "kpoint_index": kpoint_index,
        "kpoint_coordinates": kpoint_coordinates,
        "spin_index": spin_index,
        "gamma_only": gamma_only,
        "scale_factor": scale_factor,
        "number_of_plane_waves": number_of_plane_waves,
        "max_plane_waves": max_plane_waves,
        "number_of_polarizations": number_of_polarizations,
        "number_of_bands": number_of_bands,
        "reciprocal_vectors": reciprocal_vectors,
        "miller_indices": miller_indices,
        "coefficients": coefficients,
    }
