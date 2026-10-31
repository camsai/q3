"""Independent reader/validator for BerkeleyGW mean-field files (WFN, RHO, vxc.dat).

This module is a deliberately independent implementation: it was written from
the definition of the BerkeleyGW mean-field file format -- the records that
Quantum ESPRESSO's ``pw2bgw.x`` converter writes and BerkeleyGW reads --
WITHOUT reading Q3's own C++ writer for these files. Its purpose is to be a
cross-check that does not share any code path, assumption, or bug with the
writer it validates in ``tests/validation/test_berkeleygw_export.py``. Do not
"fix" a disagreement between this reader and the writer by copying the
writer's interpretation here; re-derive from the format's definition instead.

Authority. The record order, contents and units follow the specification
BerkeleyGW publishes: J. Deslippe et al., Comput. Phys. Commun. 183, 1269
(2012), Appendix 11.1, "Specification of file formats" (arXiv:1111.4429),
cited below as "the BerkeleyGW specification". What it leaves open -- Fortran
record framing, integer and real kinds, the widths of the character fields
and of vxc.dat's columns, the count repeated before each G-vector or data
block, and the conventions of the symmetry records -- is taken from the
converter that writes these files, Quantum ESPRESSO's ``pw2bgw.x``. Citations
of it name the subroutine (``write_wfng`` for WFN, ``write_rhog`` for RHO,
``write_vxc_g``/``write_vxc_r`` for vxc.dat, in ``PP/src/pw2bgw.f90`` of
Quantum ESPRESSO 7.5, release tag ``qe-7.5``) and the record within it, never a
line number: the line numbers first written here were read off an unpinned
``develop`` checkout and moved with it. See PROVENANCE.md section 5.1.

Framing (all three binary/text conventions below are load-bearing):

- WFN and RHO are Fortran ``form='unformatted', access='sequential'`` files.
  Every record is ``[int32 length][payload][int32 length]``, both length
  markers giving the payload size in bytes. This reader validates the two
  markers agree for every single record and raises ``BerkeleyGWFormatError``
  naming the record if they do not, or if the file ends before a record is
  complete -- a desync here is exactly the symptom a writer framing bug would
  produce, so it must be loud, never silently swallowed or misaligned.
- All reals are float64 (``<f8``), all integers int32 (``<i4``), little-endian
  -- matching ``tests/utils/qe/binary_readers.py``'s existing convention for
  Q3/QE's own Fortran-unformatted files, and standard for x86_64 Linux HPC
  builds (a build/platform convention that the converter's source cannot
  itself confirm; it is the one fact in this module taken from the platform
  rather than from the format's definition).
- Arrays are stored column-major (Fortran order): for an array of shape
  (3, n) the FIRST index varies fastest, so element [j, i] sits at flat
  position j + 3 * i. This reader reshapes with ``order="F"`` throughout;
  never ``order="C"`` (which would silently transpose square matrices).

Every record-parsing function below names the subroutine and record it was
derived from, so a disagreement can be traced back to source.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

# ── Public exception type ─────────────────────────────────────────────


class BerkeleyGWFormatError(ValueError):
    """A BerkeleyGW mean-field file violates this reader's format contract.

    Raised for: Fortran record framing desync (leading/trailing length
    markers disagree), truncation (file ends mid-record), a record whose
    payload size does not match what the already-parsed header implies, an
    unrecognized ``stitle`` flavor tag, or a spin-polarized (``ns > 1``) body
    that this reader deliberately does not attempt to parse (see
    ``_read_wfn_body``'s docstring).
    """


# ── Fortran unformatted sequential record framing ─────────────────────

_INT32 = np.dtype("<i4")
_FLOAT64 = np.dtype("<f8")
_COMPLEX128 = np.dtype("<c16")
_LENGTH_MARKER_SIZE = 4


class _FortranSequentialReader:
    """Reads Fortran ``unformatted``/``sequential`` records one at a time.

    Every record is ``[int32 length][payload][int32 length]``. Each call to
    ``read_raw_record`` validates that the trailing length marker matches the
    leading one and that the file did not end early, raising
    ``BerkeleyGWFormatError`` (naming both a running record index and the
    caller-supplied human label) otherwise.
    """

    def __init__(self, file_handle, source_path):
        self._file_handle = file_handle
        self.source_path = source_path
        self._record_count = 0

    def read_raw_record(self, label):
        self._record_count += 1
        leading_bytes = self._file_handle.read(_LENGTH_MARKER_SIZE)
        if len(leading_bytes) < _LENGTH_MARKER_SIZE:
            raise BerkeleyGWFormatError(
                f"{self.source_path}: unexpected end of file before the leading "
                f"length marker of record #{self._record_count} ({label}) -- "
                f"read {len(leading_bytes)} of {_LENGTH_MARKER_SIZE} bytes."
            )
        payload_length = int(np.frombuffer(leading_bytes, dtype=_INT32, count=1)[0])
        if payload_length < 0:
            raise BerkeleyGWFormatError(
                f"{self.source_path}: record #{self._record_count} ({label}) "
                f"has a negative leading length ({payload_length}) -- corrupted file."
            )
        payload = self._file_handle.read(payload_length)
        if len(payload) != payload_length:
            raise BerkeleyGWFormatError(
                f"{self.source_path}: truncated file -- record #{self._record_count} "
                f"({label}) declared a {payload_length}-byte payload but only "
                f"{len(payload)} byte(s) remained in the file."
            )
        trailing_bytes = self._file_handle.read(_LENGTH_MARKER_SIZE)
        if len(trailing_bytes) < _LENGTH_MARKER_SIZE:
            raise BerkeleyGWFormatError(
                f"{self.source_path}: truncated file -- record #{self._record_count} "
                f"({label}) is missing its trailing length marker entirely."
            )
        trailing_length = int(np.frombuffer(trailing_bytes, dtype=_INT32, count=1)[0])
        if trailing_length != payload_length:
            raise BerkeleyGWFormatError(
                f"{self.source_path}: Fortran record framing corrupted at record "
                f"#{self._record_count} ({label}): leading length = {payload_length} "
                f"bytes, trailing length = {trailing_length} bytes. The reader and "
                f"writer disagree on this record's size -- this is a writer/framing "
                f"bug, not a parsing ambiguity."
            )
        return payload


def _as_array(payload, dtype, expected_count, label, source_path):
    """Interprets a raw record payload as ``expected_count`` values of ``dtype``.

    Raises ``BerkeleyGWFormatError`` (rather than a bare numpy error) if the
    payload's byte length does not exactly match, so a header field that
    implies the wrong array size is diagnosed as clearly as a framing error.
    """
    dtype = np.dtype(dtype)
    expected_bytes = dtype.itemsize * expected_count
    if len(payload) != expected_bytes:
        raise BerkeleyGWFormatError(
            f"{source_path}: {label} payload is {len(payload)} byte(s), expected "
            f"{expected_bytes} byte(s) for {expected_count} x {dtype.itemsize}-byte "
            f"element(s) of dtype {dtype}."
        )
    return np.frombuffer(payload, dtype=dtype, count=expected_count)


def _reshape_fortran(flat_array, shape):
    """Reshapes a flat array using Fortran (column-major) order.

    The first axis varies fastest -- e.g. for a (3,3) matrix, flat elements
    0,1,2 become column 0 (rows 0,1,2), the column-major order in which the
    converter writes every 3x3 block. Using ``order="C"`` here would
    silently transpose any non-symmetric matrix.
    """
    return np.asarray(flat_array).reshape(shape, order="F")


def _read_scalar_int32(reader, label):
    payload = reader.read_raw_record(label)
    array = _as_array(payload, _INT32, 1, label, reader.source_path)
    return int(array[0])


# ── Shared header records (byte-identical layout in WFN and RHO) ──────


def _read_title_block(reader, expected_prefix, label="record 1 (stitle, sdate, stime)"):
    """See ``_read_title_block_for``; ``expected_prefix`` may also be a tuple of them."""
    prefixes = (expected_prefix,) if isinstance(expected_prefix, str) else tuple(expected_prefix)
    return _read_title_block_for(reader, prefixes, label)


def _read_title_block_for(reader, expected_prefixes, label):
    """Record 1: 32-byte stitle + 32-byte sdate + 32-byte stime = 96 bytes.

    The converter (``write_wfng`` for WFN, ``write_rhog`` for RHO) builds
    these as space-padded fixed-width character fields and writes them as one
    record. ``stitle`` is ``"<expected_prefix>-Real"`` or
    ``"<expected_prefix>-Complex"`` (e.g. "WFN-Real", "RHO-Complex").
    """
    payload = reader.read_raw_record(label)
    if len(payload) != 96:
        raise BerkeleyGWFormatError(
            f"{reader.source_path}: {label} is {len(payload)} byte(s), expected 96 "
            f"(32-byte stitle + 32-byte sdate + 32-byte stime)."
        )
    try:
        title = payload[0:32].decode("ascii").rstrip()
        creation_date = payload[32:64].decode("ascii").rstrip()
        creation_time = payload[64:96].decode("ascii").rstrip()
    except UnicodeDecodeError as exc:
        raise BerkeleyGWFormatError(f"{reader.source_path}: {label} is not ASCII: {exc}") from exc
    if title in {f"{prefix}-Real" for prefix in expected_prefixes}:
        is_complex_flavor = False
    elif title in {f"{prefix}-Complex" for prefix in expected_prefixes}:
        is_complex_flavor = True
    else:
        expected = " or ".join(
            f"'{prefix}-Real' or '{prefix}-Complex'" for prefix in expected_prefixes
        )
        raise BerkeleyGWFormatError(
            f"{reader.source_path}: {label} stitle is {title!r}, expected {expected}."
        )
    return {
        "title": title,
        "creation_date": creation_date,
        "creation_time": creation_time,
        "is_complex_flavor": is_complex_flavor,
    }


def _read_lattice_and_symmetry_records(
    reader, number_of_symmetry_operations, number_of_atoms, section_label
):
    """Records 4-8: geometry, symmetry, and atoms -- identical in WFN and RHO.

    ``write_wfng`` and ``write_rhog`` compute these five records the same
    way, so WFN and RHO carry byte-identical records 4-8 (verified by
    comparing the two subroutines).
    """
    # Record 4: celvol, alat, avec (3x3), adot (3x3) -- 20 float64 (record 4 of both
    # writers). avec holds the lattice vectors a_i in units of alat, one per column, and
    # adot is the real-space metric tensor: adot[i, j] = alat^2 * (a_i . a_j), i.e.
    # alat^2 * avec^T @ avec.
    label = f"{section_label} record 4 (celvol, alat, avec, adot)"
    payload = reader.read_raw_record(label)
    values = _as_array(payload, _FLOAT64, 20, label, reader.source_path)
    cell_volume_bohr3 = float(values[0])
    lattice_parameter_bohr = float(values[1])
    lattice_vectors_alat_units = _reshape_fortran(values[2:11], (3, 3)).copy()
    real_space_metric_tensor_bohr2 = _reshape_fortran(values[11:20], (3, 3)).copy()

    # Record 5: recvol, blat, bvec (3x3), bdot (3x3) -- 20 float64 (record 5 of both
    # writers). recvol = (2*pi)^3 / celvol, the Brillouin-zone volume; blat = 2*pi/alat,
    # the unit of the reciprocal lattice vectors b_i in bvec; bdot is the reciprocal
    # metric tensor, bdot[i, j] = blat^2 * (b_i . b_j) = blat^2 * bvec^T @ bvec, built
    # the same way as adot.
    label = f"{section_label} record 5 (recvol, blat, bvec, bdot)"
    payload = reader.read_raw_record(label)
    values = _as_array(payload, _FLOAT64, 20, label, reader.source_path)
    reciprocal_cell_volume_bohr3 = float(values[0])
    reciprocal_lattice_parameter_inverse_bohr = float(values[1])
    reciprocal_lattice_vectors_tpiba_units = _reshape_fortran(values[2:11], (3, 3)).copy()
    reciprocal_space_metric_tensor_inverse_bohr2 = _reshape_fortran(values[11:20], (3, 3)).copy()

    # Record 6: mtrx, one 3x3 int32 rotation matrix per symmetry operation, crystal
    # basis (record 6 of both writers).
    label = f"{section_label} record 6 (symmetry rotation matrices)"
    payload = reader.read_raw_record(label)
    flat = _as_array(payload, _INT32, 9 * number_of_symmetry_operations, label, reader.source_path)
    symmetry_rotation_matrices = _reshape_fortran(
        flat, (3, 3, number_of_symmetry_operations)
    ).copy()

    # Record 7: tnp, three float64 per symmetry operation (record 7 of both writers).
    # NOTE: the BerkeleyGW specification gives these in units of the lattice vectors
    # times 2*pi, not as bare fractional translations, and the converter's values are
    # more specific still: it stores 2*pi times the fractional
    # translation t carried through the inverse of the operation's crystal-basis
    # rotation matrix S, t' = (S^-1)^T t, with each component of t' first reduced
    # modulo 1 into (-0.5, 0.5] -- the CLOSED end at +0.5, not at -0.5 -- and only then
    # scaled by 2*pi (the symmetry set-up shared by ``write_wfng`` and ``write_rhog``).
    # This matters if this array's *values* are ever cross-checked -- the 2*pi factor
    # is baked in on disk.
    label = f"{section_label} record 7 (symmetry fractional translations x 2*pi)"
    payload = reader.read_raw_record(label)
    flat = _as_array(
        payload, _FLOAT64, 3 * number_of_symmetry_operations, label, reader.source_path
    )
    symmetry_fractional_translations_two_pi = _reshape_fortran(
        flat, (3, number_of_symmetry_operations)
    ).copy()

    # Record 8: (apos(3,i), Z_i) for i=1..nat, INTERLEAVED per atom: 3 float64 then 1
    # int32, repeated (record 8 of both writers). Positions are Cartesian, in alat
    # units (the convention QE keeps atomic positions in). A structured dtype with no
    # alignment padding reproduces this interleaving exactly (numpy does not pad
    # list-of-tuples dtypes unless align=True is requested, which it is not here).
    label = f"{section_label} record 8 (atom positions interleaved with atomic numbers)"
    payload = reader.read_raw_record(label)
    atom_dtype = np.dtype([("position", _FLOAT64, (3,)), ("atomic_number", _INT32)])
    atoms = _as_array(payload, atom_dtype, number_of_atoms, label, reader.source_path)
    atom_positions_alat_units = np.ascontiguousarray(atoms["position"].T).astype(_FLOAT64)
    atomic_numbers = atoms["atomic_number"].copy()

    return {
        "cell_volume_bohr3": cell_volume_bohr3,
        "lattice_parameter_bohr": lattice_parameter_bohr,
        "lattice_vectors_alat_units": lattice_vectors_alat_units,
        "real_space_metric_tensor_bohr2": real_space_metric_tensor_bohr2,
        "reciprocal_cell_volume_bohr3": reciprocal_cell_volume_bohr3,
        "reciprocal_lattice_parameter_inverse_bohr": reciprocal_lattice_parameter_inverse_bohr,
        "reciprocal_lattice_vectors_tpiba_units": reciprocal_lattice_vectors_tpiba_units,
        "reciprocal_space_metric_tensor_inverse_bohr2": reciprocal_space_metric_tensor_inverse_bohr2,
        "symmetry_rotation_matrices": symmetry_rotation_matrices,
        "symmetry_fractional_translations_two_pi": symmetry_fractional_translations_two_pi,
        "atom_positions_alat_units": atom_positions_alat_units,
        "atomic_numbers": atomic_numbers,
    }


# ── WFN ─────────────────────────────────────────────────────────────


def read_berkeleygw_wavefunction_file(path):
    """Reads a BerkeleyGW ``WFN`` mean-field file (18 header records + body).

    Header layout follows the BerkeleyGW specification's header records as
    ``write_wfng`` writes all 18 of them; body layout follows the same
    subroutine's body (per-k-point G-vectors, then per-band
    coefficients).

    Body-parsing scope: this reader supports ``number_of_spin_components ==
    1`` bodies only (any flavor, real or complex). ``write_wfng`` chooses
    between two body layouts: whenever the file is real-flavored or
    spin-polarized, it writes a fundamentally different, spin-interleaved
    layout (one record per band holding ALL spins' coefficients
    concatenated) rather than the simple one-record-per-(k,spin,band) layout
    this reader implements, which it uses only for the complex flavor with
    one spin component. For ``ns > 1`` that means the interleaved layout
    every time. Rather than guess
    at that interleaving, a spin-polarized body raises
    ``BerkeleyGWFormatError`` with an explicit explanation. The HEADER is
    parsed fully generally for any ``ns`` -- only body (G-vector/coefficient)
    parsing is restricted.

    Returns a dict with every header field (spelled-out keys, see module
    source for the exact list) plus:
      - "kpoint_gvectors": list of length number_of_kpoints, each an
        int32 ndarray of shape (3, ngk[k]) -- that k-point's Miller indices.
      - "wavefunction_coefficients": list of length number_of_kpoints, each
        an ndarray of shape (number_of_bands, ngk[k]), dtype complex128 or
        float64 matching the file's flavor.
    """
    path = Path(path)
    with open(path, "rb") as file_handle:
        reader = _FortranSequentialReader(file_handle, path)
        header = {}
        header.update(_read_title_block(reader, "WFN"))

        # Record 2: ns, ng, ntran, cell_symmetry, nat, ecutrho, nk, nbands, ngkmax,
        # ecutwfc -- MIXED TYPES in one record: int32 x5, float64, int32 x3, float64
        # (``write_wfng`` record 2; ns is the number of spin components in the file and
        # nk the k-points per spin channel, the converter's total k-point-state count
        # divided by ns -- both as its ``set_spin`` subroutine arranges them).
        label = "record 2 (ns, ng, ntran, cell_symmetry, nat, ecutrho, nk, nbands, ngkmax, ecutwfc)"
        payload = reader.read_raw_record(label)
        record_2_dtype = np.dtype(
            [
                ("number_of_spin_components", _INT32),
                ("number_of_g_vectors", _INT32),
                ("number_of_symmetry_operations", _INT32),
                ("cell_symmetry_class", _INT32),
                ("number_of_atoms", _INT32),
                ("charge_density_cutoff_rydberg", _FLOAT64),
                ("number_of_kpoints", _INT32),
                ("number_of_bands", _INT32),
                ("maximum_g_vectors_per_kpoint", _INT32),
                ("wavefunction_cutoff_rydberg", _FLOAT64),
            ]
        )
        record_2 = _as_array(payload, record_2_dtype, 1, label, path)[0]
        number_of_spin_components = int(record_2["number_of_spin_components"])
        number_of_g_vectors = int(record_2["number_of_g_vectors"])
        number_of_symmetry_operations = int(record_2["number_of_symmetry_operations"])
        number_of_atoms = int(record_2["number_of_atoms"])
        number_of_kpoints = int(record_2["number_of_kpoints"])
        number_of_bands = int(record_2["number_of_bands"])
        maximum_g_vectors_per_kpoint = int(record_2["maximum_g_vectors_per_kpoint"])
        header.update(
            {
                "number_of_spin_components": number_of_spin_components,
                "number_of_g_vectors": number_of_g_vectors,
                "number_of_symmetry_operations": number_of_symmetry_operations,
                "cell_symmetry_class": int(record_2["cell_symmetry_class"]),
                "number_of_atoms": number_of_atoms,
                "charge_density_cutoff_rydberg": float(record_2["charge_density_cutoff_rydberg"]),
                "number_of_kpoints": number_of_kpoints,
                "number_of_bands": number_of_bands,
                "maximum_g_vectors_per_kpoint": maximum_g_vectors_per_kpoint,
                "wavefunction_cutoff_rydberg": float(record_2["wavefunction_cutoff_rydberg"]),
            }
        )

        # Record 3: FFTgrid (3 int32), kgrid (3 int32), kshift (3 float64)
        # (``write_wfng`` record 3; kshift is half the Monkhorst-Pack shift flag along
        # each axis, so 0 or 0.5).
        label = "record 3 (FFTgrid, kgrid, kshift)"
        payload = reader.read_raw_record(label)
        record_3_dtype = np.dtype(
            [
                ("fft_grid_dimensions", _INT32, (3,)),
                ("kpoint_grid_dimensions", _INT32, (3,)),
                ("kpoint_grid_shifts", _FLOAT64, (3,)),
            ]
        )
        record_3 = _as_array(payload, record_3_dtype, 1, label, path)[0]
        header["fft_grid_dimensions"] = np.array(record_3["fft_grid_dimensions"], dtype=_INT32)
        header["kpoint_grid_dimensions"] = np.array(
            record_3["kpoint_grid_dimensions"], dtype=_INT32
        )
        header["kpoint_grid_shifts"] = np.array(record_3["kpoint_grid_shifts"], dtype=_FLOAT64)

        header.update(
            _read_lattice_and_symmetry_records(
                reader, number_of_symmetry_operations, number_of_atoms, "WFN"
            )
        )

        # Record 9: ngk, nk int32 -- plane waves per k-point (``write_wfng`` record 9).
        label = "record 9 (ngk)"
        payload = reader.read_raw_record(label)
        number_of_gvectors_per_kpoint = _as_array(
            payload, _INT32, number_of_kpoints, label, path
        ).copy()
        header["number_of_gvectors_per_kpoint"] = number_of_gvectors_per_kpoint

        # Record 10: kw, nk float64 weights, normalized so sum = 1 (``write_wfng`` record
        # 10 writes QE's k-point weight times the number of spin components over 2, the
        # spin count being the one ``set_spin`` sets for the non-spinor case; QE's weights
        # sum to 2, as Q3's results.json documents its own do (docs/io_reference.md), so
        # /2 recovers 1).
        label = "record 10 (kw)"
        payload = reader.read_raw_record(label)
        header["kpoint_weights"] = _as_array(
            payload, _FLOAT64, number_of_kpoints, label, path
        ).copy()

        # Record 11: kpt, shape (3, nk) float64, CRYSTAL (fractional) coordinates --
        # QE holds k-points in Cartesian coordinates, so ``write_wfng`` converts them to
        # crystal coordinates just before its header records and back just after.
        label = "record 11 (kpt)"
        payload = reader.read_raw_record(label)
        flat = _as_array(payload, _FLOAT64, 3 * number_of_kpoints, label, path)
        header["kpoints_crystal_coordinates"] = _reshape_fortran(
            flat, (3, number_of_kpoints)
        ).copy()

        # Records 12-13: ifmin/ifmax, nk*ns int32 each (``write_wfng`` records 12-13).
        label = "record 12 (ifmin)"
        payload = reader.read_raw_record(label)
        header["lowest_occupied_band_index"] = _as_array(
            payload, _INT32, number_of_kpoints * number_of_spin_components, label, path
        ).copy()
        label = "record 13 (ifmax)"
        payload = reader.read_raw_record(label)
        header["highest_occupied_band_index"] = _as_array(
            payload, _INT32, number_of_kpoints * number_of_spin_components, label, path
        ).copy()

        # Records 14-15: energies/occupations, shape (nbands, nk*ns) float64 (``write_wfng``
        # records 14-15). Energies are in Rydberg: the converter copies QE's eigenvalues
        # into this record with no rescaling, and converts them to eV explicitly wherever
        # else it needs eV -- confirming that the eigenvalues themselves, hence this
        # record, are Rydberg, not eV.
        # NOTE: for ns > 1 the nk*ns axis is ordered spin-block-major (QE's global k-point
        # list is nk points of spin 1 followed by nk points of spin 2, and the converter
        # keeps that order) -- NOT interleaved per k-point.
        label = "record 14 (energies)"
        payload = reader.read_raw_record(label)
        flat = _as_array(
            payload,
            _FLOAT64,
            number_of_bands * number_of_kpoints * number_of_spin_components,
            label,
            path,
        )
        header["band_energies_rydberg"] = _reshape_fortran(
            flat, (number_of_bands, number_of_kpoints * number_of_spin_components)
        ).copy()
        label = "record 15 (occupations)"
        payload = reader.read_raw_record(label)
        flat = _as_array(
            payload,
            _FLOAT64,
            number_of_bands * number_of_kpoints * number_of_spin_components,
            label,
            path,
        )
        header["band_occupations"] = _reshape_fortran(
            flat, (number_of_bands, number_of_kpoints * number_of_spin_components)
        ).copy()

        # Record 16: nrecord, always 1 (the converter sets it to 1 once and never
        # recomputes it from an array size before any write).
        nrecord = _read_scalar_int32(reader, "record 16 (nrecord)")
        if nrecord != 1:
            raise BerkeleyGWFormatError(
                f"{path}: record 16 (nrecord) = {nrecord}, expected 1 -- this reader "
                f"(and pw2bgw.x itself, which always writes nrecord=1) does "
                f"not support multi-part chunked G-vector records."
            )

        # Record 17: ng (repeated) -- cross-check against record 2's value.
        ng_repeated = _read_scalar_int32(reader, "record 17 (ng)")
        if ng_repeated != number_of_g_vectors:
            raise BerkeleyGWFormatError(
                f"{path}: record 17 (ng) = {ng_repeated}, but record 2 already "
                f"declared number_of_g_vectors = {number_of_g_vectors}."
            )

        # Record 18: gvec, shape (3, ng) int32 Miller indices -- the charge-density G
        # set (``write_wfng`` record 18).
        label = "record 18 (gvec)"
        payload = reader.read_raw_record(label)
        flat = _as_array(payload, _INT32, 3 * number_of_g_vectors, label, path)
        header["charge_density_gvectors"] = _reshape_fortran(flat, (3, number_of_g_vectors)).copy()

        kpoint_gvectors, wavefunction_coefficients = _read_wfn_body(reader, header)

    header["kpoint_gvectors"] = kpoint_gvectors
    header["wavefunction_coefficients"] = wavefunction_coefficients
    header["source_path"] = path
    return header


def _read_wfn_body(reader, header):
    """Reads the WFN body: per k-point G-vectors, then per-band coefficients.

    See ``read_berkeleygw_wavefunction_file``'s docstring for the
    ``number_of_spin_components > 1`` scope restriction this enforces.
    """
    path = reader.source_path
    number_of_kpoints = header["number_of_kpoints"]
    number_of_spin_components = header["number_of_spin_components"]
    number_of_bands = header["number_of_bands"]
    number_of_gvectors_per_kpoint = header["number_of_gvectors_per_kpoint"]
    coefficient_dtype = _COMPLEX128 if header["is_complex_flavor"] else _FLOAT64

    if number_of_spin_components != 1:
        raise BerkeleyGWFormatError(
            f"{path}: number_of_spin_components = {number_of_spin_components}; this "
            f"reader only supports parsing the WFN BODY (G-vector/coefficient records) "
            f"for spin-unpolarized (ns=1) files -- see "
            f"read_berkeleygw_wavefunction_file's docstring for why ns>1 uses a "
            f"structurally different record layout in pw2bgw.x's output that is not "
            f"implemented here. The header above was parsed fully and is valid."
        )

    kpoint_gvectors = []
    wavefunction_coefficients = []
    for kpoint_index in range(number_of_kpoints):
        expected_gvector_count = int(number_of_gvectors_per_kpoint[kpoint_index])

        # [nrecord=1] [ngk] [gvec_k, shape (3, ngk)] -- ``write_wfng``'s body, written
        # once per k-point (for the first spin channel only, which is the single spin
        # channel that exists when ns=1).
        nrecord = _read_scalar_int32(reader, f"k-point {kpoint_index} gvec block nrecord")
        if nrecord != 1:
            raise BerkeleyGWFormatError(
                f"{path}: k-point {kpoint_index} gvec block nrecord = {nrecord}, expected 1."
            )
        label = f"k-point {kpoint_index} gvec block ngk"
        actual_gvector_count = _read_scalar_int32(reader, label)
        if actual_gvector_count != expected_gvector_count:
            raise BerkeleyGWFormatError(
                f"{path}: k-point {kpoint_index} gvec block declares ngk = "
                f"{actual_gvector_count}, but header record 9 declared "
                f"ngk[{kpoint_index}] = {expected_gvector_count}."
            )
        label = f"k-point {kpoint_index} gvec block payload"
        payload = reader.read_raw_record(label)
        flat = _as_array(payload, _INT32, 3 * expected_gvector_count, label, path)
        kpoint_gvectors.append(_reshape_fortran(flat, (3, expected_gvector_count)).copy())

        # Per band: [nrecord=1] [ngk] [ngk coefficients] -- ``write_wfng``'s
        # one-record-per-band layout (complex flavor AND ns=1, guaranteed here since we
        # already rejected ns>1 above; WFN-Real with ns=1 also reduces to this exact
        # layout, since each band's record then holds a single spin -- see docstring).
        band_coefficients = np.empty(
            (number_of_bands, expected_gvector_count), dtype=coefficient_dtype
        )
        for band_index in range(number_of_bands):
            nrecord = _read_scalar_int32(
                reader, f"k-point {kpoint_index} band {band_index} nrecord"
            )
            if nrecord != 1:
                raise BerkeleyGWFormatError(
                    f"{path}: k-point {kpoint_index} band {band_index} nrecord = "
                    f"{nrecord}, expected 1."
                )
            label = f"k-point {kpoint_index} band {band_index} ngk"
            actual_gvector_count = _read_scalar_int32(reader, label)
            if actual_gvector_count != expected_gvector_count:
                raise BerkeleyGWFormatError(
                    f"{path}: k-point {kpoint_index} band {band_index} declares ngk = "
                    f"{actual_gvector_count}, expected {expected_gvector_count}."
                )
            label = f"k-point {kpoint_index} band {band_index} coefficients"
            payload = reader.read_raw_record(label)
            band_coefficients[band_index, :] = _as_array(
                payload, coefficient_dtype, expected_gvector_count, label, path
            )
        wavefunction_coefficients.append(band_coefficients)

    return kpoint_gvectors, wavefunction_coefficients


# ── RHO ─────────────────────────────────────────────────────────────


def _read_field_file(path, expected_prefixes, field_key):
    """Reads a field file -- RHO, VXC or VSC: 8 header records, the G-set, one field per spin.

    The three share RHO's layout (the BerkeleyGW specification's RHO and VXC
    files; BerkeleyGW 4.0's ParaBands reads VSC with the same routines, in
    ``MeanField/ParaBands/parabands.f90``), and differ only in the title and
    in what the values mean. ``expected_prefixes`` lists the titles accepted,
    first the one named in errors; the values land under ``field_key``.
    """
    path = Path(path)
    with open(path, "rb") as file_handle:
        reader = _FortranSequentialReader(file_handle, path)
        header = {}
        header.update(_read_title_block(reader, expected_prefixes))

        # Record 2: ns, ng, ntran, cell_symmetry, nat, ecutrho -- all that's shared with
        # WFN's record 2; RHO omits nk/nbands/ngkmax/ecutwfc entirely (``write_rhog``
        # record 2).
        label = "record 2 (ns, ng, ntran, cell_symmetry, nat, ecutrho)"
        payload = reader.read_raw_record(label)
        record_2_dtype = np.dtype(
            [
                ("number_of_spin_components", _INT32),
                ("number_of_g_vectors", _INT32),
                ("number_of_symmetry_operations", _INT32),
                ("cell_symmetry_class", _INT32),
                ("number_of_atoms", _INT32),
                ("charge_density_cutoff_rydberg", _FLOAT64),
            ]
        )
        record_2 = _as_array(payload, record_2_dtype, 1, label, path)[0]
        number_of_spin_components = int(record_2["number_of_spin_components"])
        number_of_g_vectors = int(record_2["number_of_g_vectors"])
        number_of_symmetry_operations = int(record_2["number_of_symmetry_operations"])
        number_of_atoms = int(record_2["number_of_atoms"])
        header.update(
            {
                "number_of_spin_components": number_of_spin_components,
                "number_of_g_vectors": number_of_g_vectors,
                "number_of_symmetry_operations": number_of_symmetry_operations,
                "cell_symmetry_class": int(record_2["cell_symmetry_class"]),
                "number_of_atoms": number_of_atoms,
                "charge_density_cutoff_rydberg": float(record_2["charge_density_cutoff_rydberg"]),
            }
        )

        # Record 3: FFTgrid (3 int32) ONLY -- no kgrid/kshift (``write_rhog`` record 3).
        label = "record 3 (FFTgrid)"
        payload = reader.read_raw_record(label)
        header["fft_grid_dimensions"] = _as_array(payload, _INT32, 3, label, path).copy()

        header.update(
            _read_lattice_and_symmetry_records(
                reader, number_of_symmetry_operations, number_of_atoms, expected_prefixes[0]
            )
        )

        # Body record group 1: [nrecord=1] [ng] [gvec, shape (3, ng)] (``write_rhog``'s
        # body).
        nrecord = _read_scalar_int32(reader, "body gvec block nrecord")
        if nrecord != 1:
            raise BerkeleyGWFormatError(f"{path}: body gvec block nrecord = {nrecord}, expected 1.")
        ng_repeated = _read_scalar_int32(reader, "body gvec block ng")
        if ng_repeated != number_of_g_vectors:
            raise BerkeleyGWFormatError(
                f"{path}: body gvec block ng = {ng_repeated}, but header record 2 "
                f"already declared number_of_g_vectors = {number_of_g_vectors}."
            )
        label = "body gvec block payload"
        payload = reader.read_raw_record(label)
        flat = _as_array(payload, _INT32, 3 * number_of_g_vectors, label, path)
        header["charge_density_gvectors"] = _reshape_fortran(flat, (3, number_of_g_vectors)).copy()

        # Body record group 2: [nrecord=1] [ng] [rho, shape (ng, ns)] (``write_rhog``'s
        # body). rho(G) is multiplied by the cell volume before writing -- see this
        # function's docstring.
        nrecord = _read_scalar_int32(reader, "body rho block nrecord")
        if nrecord != 1:
            raise BerkeleyGWFormatError(f"{path}: body rho block nrecord = {nrecord}, expected 1.")
        ng_repeated = _read_scalar_int32(reader, "body rho block ng")
        if ng_repeated != number_of_g_vectors:
            raise BerkeleyGWFormatError(
                f"{path}: body rho block ng = {ng_repeated}, but header record 2 "
                f"already declared number_of_g_vectors = {number_of_g_vectors}."
            )
        coefficient_dtype = _COMPLEX128 if header["is_complex_flavor"] else _FLOAT64
        label = "body rho block payload"
        payload = reader.read_raw_record(label)
        flat = _as_array(
            payload, coefficient_dtype, number_of_g_vectors * number_of_spin_components, label, path
        )
        header[field_key] = _reshape_fortran(
            flat, (number_of_g_vectors, number_of_spin_components)
        ).copy()

    header["source_path"] = path
    return header


def read_berkeleygw_charge_density_file(path):
    """Reads a BerkeleyGW ``RHO`` mean-field file (8 header records + body).

    Header layout follows the BerkeleyGW specification as ``write_rhog``
    writes its records 1-8 -- identical
    to WFN records 1, 4-8, but record 2 omits the k-point fields and record 3
    is FFT grid only (no kgrid/kshift).

    Returns a dict with every header field plus:
      - "charge_density_gvectors": int32 ndarray, shape (3, number_of_g_vectors).
      - "charge_density_g_times_cell_volume": complex128 (or float64, if
        real flavor) ndarray, shape (number_of_g_vectors,
        number_of_spin_components). NOTE the name: ``write_rhog`` multiplies
        rho(G) by the cell volume before writing it
        -- this is the raw on-disk BerkeleyGW convention, not the bare
        density. See ``read_berkeleygw_wavefunction_file``'s docstring for
        the general conventions (framing, dtypes, column-major order).
    """
    return _read_field_file(path, ("RHO",), "charge_density_g_times_cell_volume")


def read_berkeleygw_potential_file(path):
    """Reads a BerkeleyGW ``VXC`` or ``VSC`` file: a potential on the G-set, per spin.

    VXC is the exchange-correlation potential (the BerkeleyGW
    specification's VXC file), VSC the local Kohn-Sham potential ParaBands
    diagonalizes with; both in Rydberg, with G = 0 the cell average.

    Returns ``read_berkeleygw_charge_density_file``'s header fields plus
    "potential_g", shape (number_of_g_vectors, number_of_spin_components).
    """
    return _read_field_file(path, ("VXC", "VSC"), "potential_g")


# ── vxc.dat (ASCII) ───────────────────────────────────────────────────

# Field widths of vxc.dat's three line kinds, as ``write_vxc_g`` and ``write_vxc_r``
# both write them. Header line: kx ky kz in three 13-wide fields with 9 decimals,
# then ndiag noffdiag in two 8-wide integer fields.
_VXC_HEADER_LINE_WIDTHS = (13, 13, 13, 8, 8)
# Diagonal line: spin band in two 8-wide integer fields, then Re(Vxc) Im(Vxc) in two
# 15-wide fields with 9 decimals.
_VXC_DIAGONAL_LINE_WIDTHS = (8, 8, 15, 15)
# Off-diagonal line: spin band_bra band_ket in three 8-wide integer fields, then
# Re(Vxc) Im(Vxc) as on a diagonal line.
_VXC_OFFDIAGONAL_LINE_WIDTHS = (8, 8, 8, 15, 15)


def _split_fixed_width(line, widths, label, source_path):
    """Slices a fixed-width Fortran-formatted line into its field strings.

    Fortran FORMAT edit descriptors (e.g. ``F13.9``) right-justify each field
    in an exact column width with NO separating whitespace guaranteed
    between fields -- a value that fills its entire field abuts the next
    field directly. Splitting on whitespace would silently misparse that
    case, so this always slices by the declared column widths instead.
    """
    total_width = sum(widths)
    stripped_line = line.rstrip("\n")
    if len(stripped_line) < total_width:
        raise BerkeleyGWFormatError(
            f"{source_path}: {label} is {len(stripped_line)} character(s) wide, "
            f"expected {total_width} for fields of width {widths} -- truncated file "
            f"or corrupted line: {stripped_line!r}"
        )
    fields = []
    position = 0
    for width in widths:
        fields.append(stripped_line[position : position + width])
        position += width
    return fields


def read_berkeleygw_exchange_correlation_file(path):
    """Reads a BerkeleyGW ``vxc.dat`` file (ASCII exchange-correlation matrix elements).

    Format follows the BerkeleyGW specification's vxc.dat section, with
    the column widths ``write_vxc_g``/``write_vxc_r`` use (both write
    byte-identical output): one header line per k-point
    (``kx ky kz ndiag noffdiag``, crystal coordinates), then ``ndiag``
    diagonal-matrix-element lines, then ``noffdiag`` off-diagonal lines.
    ``ndiag``/``noffdiag`` are already multiplied by ``number_of_spin_components``
    on disk (the header line carries ns times the number of diagonal
    elements and ns times the square of the number of off-diagonal bands),
    so they are exactly this reader's per-k-point line counts, not a "per
    spin" count needing further multiplication. Matrix elements are in eV
    (converted from Rydberg where each one is computed).

    Returns a dict: {"number_of_kpoints": int, "kpoints": [...]} where each
    entry of "kpoints" is a dict:
      - "kpoint_crystal_coordinates": float64 ndarray, shape (3,).
      - "diagonal_entry_count" / "offdiagonal_entry_count": int (as read from
        the header line -- already nspin-scaled, see above).
      - "diagonal_entries": list of dicts with "spin_index", "band_index",
        "vxc_ev" (Python complex).
      - "offdiagonal_entries": list of dicts with "spin_index",
        "band_index_bra", "band_index_ket", "vxc_ev" (Python complex). As
        ``write_vxc_g`` computes it, the matrix element is
        <bra|Vxc|ket> = sum_G conj(c_bra(G)) (Vxc psi_ket)(G).
    """
    path = Path(path)
    kpoint_blocks = []
    with open(path, encoding="ascii") as file_handle:
        lines = file_handle.readlines()

    line_index = 0
    kpoint_index = 0
    while line_index < len(lines):
        label = f"vxc.dat k-point {kpoint_index} header line"
        kx_text, ky_text, kz_text, ndiag_text, noffdiag_text = _split_fixed_width(
            lines[line_index], _VXC_HEADER_LINE_WIDTHS, label, path
        )
        line_index += 1
        kpoint_crystal_coordinates = np.array(
            [float(text) for text in (kx_text, ky_text, kz_text)], dtype=_FLOAT64
        )
        diagonal_entry_count = int(ndiag_text)
        offdiagonal_entry_count = int(noffdiag_text)

        diagonal_entries = []
        for _ in range(diagonal_entry_count):
            if line_index >= len(lines):
                raise BerkeleyGWFormatError(
                    f"{path}: k-point {kpoint_index} declared {diagonal_entry_count} "
                    f"diagonal entries but the file ended early."
                )
            label = f"vxc.dat k-point {kpoint_index} diagonal entry"
            spin_text, band_text, real_text, imag_text = _split_fixed_width(
                lines[line_index], _VXC_DIAGONAL_LINE_WIDTHS, label, path
            )
            line_index += 1
            diagonal_entries.append(
                {
                    "spin_index": int(spin_text),
                    "band_index": int(band_text),
                    "vxc_ev": complex(float(real_text), float(imag_text)),
                }
            )

        offdiagonal_entries = []
        for _ in range(offdiagonal_entry_count):
            if line_index >= len(lines):
                raise BerkeleyGWFormatError(
                    f"{path}: k-point {kpoint_index} declared {offdiagonal_entry_count} "
                    f"off-diagonal entries but the file ended early."
                )
            label = f"vxc.dat k-point {kpoint_index} off-diagonal entry"
            spin_text, band_bra_text, band_ket_text, real_text, imag_text = _split_fixed_width(
                lines[line_index], _VXC_OFFDIAGONAL_LINE_WIDTHS, label, path
            )
            line_index += 1
            offdiagonal_entries.append(
                {
                    "spin_index": int(spin_text),
                    "band_index_bra": int(band_bra_text),
                    "band_index_ket": int(band_ket_text),
                    "vxc_ev": complex(float(real_text), float(imag_text)),
                }
            )

        kpoint_blocks.append(
            {
                "kpoint_crystal_coordinates": kpoint_crystal_coordinates,
                "diagonal_entry_count": diagonal_entry_count,
                "offdiagonal_entry_count": offdiagonal_entry_count,
                "diagonal_entries": diagonal_entries,
                "offdiagonal_entries": offdiagonal_entries,
            }
        )
        kpoint_index += 1

    return {"number_of_kpoints": len(kpoint_blocks), "kpoints": kpoint_blocks, "source_path": path}


# ── VKB (Kleinman-Bylander projectors) ────────────────────────────────


def _read_block(reader, expected_count, dtype, values_per_item, label):
    """One ``[nrecord=1][count][values]`` block; returns the values, flat."""
    nrecord = _read_scalar_int32(reader, f"{label} nrecord")
    if nrecord != 1:
        raise BerkeleyGWFormatError(
            f"{reader.source_path}: {label} nrecord = {nrecord}, expected 1."
        )
    count = _read_scalar_int32(reader, f"{label} count")
    if count != expected_count:
        raise BerkeleyGWFormatError(
            f"{reader.source_path}: {label} count = {count}, expected {expected_count}."
        )
    payload = reader.read_raw_record(f"{label} payload")
    return _as_array(payload, dtype, values_per_item * count, label, reader.source_path)


def read_berkeleygw_kleinman_bylander_file(path):
    """Reads a BerkeleyGW ``VKB`` file: the nonlocal pseudopotential at every k-point.

    The BerkeleyGW specification does not cover VKB; this follows the reader
    BerkeleyGW 4.0 ships, ``pp_read_header`` and ``pp_read_beta`` in
    ``MeanField/ParaBands/pseudopot_vkb.f90``. Records 1 and 3-11 are WFN's;
    record 2 replaces WFN's nbands with nsp, nkb, nhm; records 12-14 are the
    species of each atom, the projector channels per species and the D
    coefficients; then the charge-density G-set and, per k-point, its
    G-vectors and every projector channel per spin, atoms grouped by species.

    Returns the header fields plus:
      - "atom_species": int ndarray (nat,), 1-based.
      - "projectors_per_species": int ndarray (nsp,).
      - "nonlocal_coefficients": float64 ndarray (nhm, nhm, nat, ns), Rydberg.
      - "kpoint_gvectors": list over k-points of int32 (3, ngk).
      - "projectors": list over k-points of complex128 (ns, nat, nhm, ngk),
        each atom's channels at their own index, zero beyond its count --
        already put back from the file's species-grouped order.
    """
    path = Path(path)
    with open(path, "rb") as file_handle:
        reader = _FortranSequentialReader(file_handle, path)
        header = {}
        header.update(_read_title_block(reader, "VKB"))

        label = "record 2 (ns, ng, ntran, cell_symmetry, nat, ecutrho, nk, nsp, nkb, nhm, ngkmax, ecutwfc)"
        payload = reader.read_raw_record(label)
        record_2_dtype = np.dtype(
            [
                ("number_of_spin_components", _INT32),
                ("number_of_g_vectors", _INT32),
                ("number_of_symmetry_operations", _INT32),
                ("cell_symmetry_class", _INT32),
                ("number_of_atoms", _INT32),
                ("charge_density_cutoff_rydberg", _FLOAT64),
                ("number_of_kpoints", _INT32),
                ("number_of_species", _INT32),
                ("number_of_projectors", _INT32),
                ("maximum_projectors_per_atom", _INT32),
                ("maximum_g_vectors_per_kpoint", _INT32),
                ("wavefunction_cutoff_rydberg", _FLOAT64),
            ]
        )
        record_2 = _as_array(payload, record_2_dtype, 1, label, path)[0]
        for name in record_2_dtype.names:
            value = record_2[name]
            header[name] = float(value) if value.dtype == _FLOAT64 else int(value)
        number_of_spin_components = header["number_of_spin_components"]
        number_of_atoms = header["number_of_atoms"]
        number_of_kpoints = header["number_of_kpoints"]
        number_of_species = header["number_of_species"]
        maximum_projectors_per_atom = header["maximum_projectors_per_atom"]

        label = "record 3 (FFTgrid, kgrid, kshift)"
        payload = reader.read_raw_record(label)
        record_3_dtype = np.dtype(
            [
                ("fft_grid_dimensions", _INT32, (3,)),
                ("kpoint_grid_dimensions", _INT32, (3,)),
                ("kpoint_grid_shifts", _FLOAT64, (3,)),
            ]
        )
        record_3 = _as_array(payload, record_3_dtype, 1, label, path)[0]
        for name in record_3_dtype.names:
            header[name] = np.array(record_3[name])

        header.update(
            _read_lattice_and_symmetry_records(
                reader, header["number_of_symmetry_operations"], number_of_atoms, "VKB"
            )
        )

        payload = reader.read_raw_record("record 9 (ngk)")
        header["number_of_gvectors_per_kpoint"] = _as_array(
            payload, _INT32, number_of_kpoints, "record 9 (ngk)", path
        ).copy()
        payload = reader.read_raw_record("record 10 (kw)")
        header["kpoint_weights"] = _as_array(
            payload, _FLOAT64, number_of_kpoints, "record 10 (kw)", path
        ).copy()
        payload = reader.read_raw_record("record 11 (kpt)")
        header["kpoints_crystal_coordinates"] = _reshape_fortran(
            _as_array(payload, _FLOAT64, 3 * number_of_kpoints, "record 11 (kpt)", path),
            (3, number_of_kpoints),
        ).copy()

        payload = reader.read_raw_record("record 12 (ityp)")
        header["atom_species"] = _as_array(
            payload, _INT32, number_of_atoms, "record 12 (ityp)", path
        ).copy()
        payload = reader.read_raw_record("record 13 (nh)")
        header["projectors_per_species"] = _as_array(
            payload, _INT32, number_of_species, "record 13 (nh)", path
        ).copy()
        shape = (
            maximum_projectors_per_atom,
            maximum_projectors_per_atom,
            number_of_atoms,
            number_of_spin_components,
        )
        payload = reader.read_raw_record("record 14 (deeq)")
        header["nonlocal_coefficients"] = _reshape_fortran(
            _as_array(payload, _FLOAT64, int(np.prod(shape)), "record 14 (deeq)", path), shape
        ).copy()

        header["charge_density_gvectors"] = _reshape_fortran(
            _read_block(reader, header["number_of_g_vectors"], _INT32, 3, "G-set"),
            (3, header["number_of_g_vectors"]),
        ).copy()

        # Atoms in the order the file stores their projectors: grouped by species.
        atoms_by_species = [
            atom_index
            for species in range(1, number_of_species + 1)
            for atom_index in range(number_of_atoms)
            if header["atom_species"][atom_index] == species
        ]
        header["kpoint_gvectors"] = []
        header["projectors"] = []
        for kpoint_index in range(number_of_kpoints):
            number_of_plane_waves = int(header["number_of_gvectors_per_kpoint"][kpoint_index])
            header["kpoint_gvectors"].append(
                _reshape_fortran(
                    _read_block(
                        reader,
                        number_of_plane_waves,
                        _INT32,
                        3,
                        f"k-point {kpoint_index + 1} G-vectors",
                    ),
                    (3, number_of_plane_waves),
                ).copy()
            )
            projectors = np.zeros(
                (
                    number_of_spin_components,
                    number_of_atoms,
                    maximum_projectors_per_atom,
                    number_of_plane_waves,
                ),
                dtype=_COMPLEX128,
            )
            for spin_index in range(number_of_spin_components):
                for atom_index in atoms_by_species:
                    species = header["atom_species"][atom_index]
                    for channel in range(header["projectors_per_species"][species - 1]):
                        projectors[spin_index, atom_index, channel] = _read_block(
                            reader,
                            number_of_plane_waves,
                            _COMPLEX128,
                            1,
                            f"k-point {kpoint_index + 1} projector",
                        )
            header["projectors"].append(projectors)

        trailing = file_handle.read(1)
        if trailing:
            raise BerkeleyGWFormatError(f"{path}: data after the last k-point's projectors.")

    header["source_path"] = path
    return header


def nonlocal_potential_matrix(kleinman_bylander_file, kpoint_index, spin_index=0):
    """V_NL(G, G') = Σ_atom Σ_ij β_i(G) D_ij conj(β_j(G')) at one k-point, Rydberg.

    The contraction ParaBands builds (``ham_build_column_nl_col`` in
    ``MeanField/ParaBands/hamiltonian.f90``). It does not depend on the
    phase, sign or order a code chooses for its projector channels, so two
    codes' VKB files of one system are compared through it.
    """
    projectors = kleinman_bylander_file["projectors"][kpoint_index][spin_index]
    coefficients = kleinman_bylander_file["nonlocal_coefficients"]
    number_of_plane_waves = projectors.shape[-1]
    matrix = np.zeros((number_of_plane_waves, number_of_plane_waves), dtype=_COMPLEX128)
    for atom_index in range(projectors.shape[0]):
        species = kleinman_bylander_file["atom_species"][atom_index]
        count = kleinman_bylander_file["projectors_per_species"][species - 1]
        beta = projectors[atom_index, :count]  # (count, ngk)
        d_matrix = coefficients[:count, :count, atom_index, spin_index]
        matrix += beta.T @ d_matrix @ beta.conj()
    return matrix
