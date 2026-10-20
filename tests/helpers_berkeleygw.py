"""Independent reader/validator for BerkeleyGW mean-field files (WFN, RHO, vxc.dat).

This module is a deliberately independent implementation: it was written from
the BerkeleyGW mean-field file format's format sources -- Quantum ESPRESSO's
``PP/src/pw2bgw.f90`` converter (the reference implementation BerkeleyGW
itself is built against) -- WITHOUT reading Q3's own C++ writer for these
files. Its purpose is to be a cross-check that does not share any code path,
assumption, or bug with the writer it validates in
``tests/validation/test_berkeleygw_export.py``. Do not "fix" a disagreement
between this reader and the writer by copying the writer's interpretation
here; re-derive from pw2bgw.f90 instead.

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
  builds (this is a build/platform convention pw2bgw.f90's source text cannot
  itself confirm; it is the one fact in this module taken on the format
  contract's authority rather than derived from the .f90 source).
- Fortran arrays are column-major: for a nested implied-do
  ``((a(j,i), j=1,3), i=1,n)`` the FIRST index (``j``) varies fastest. This
  reader reshapes with ``order="F"`` throughout; never ``order="C"`` (which
  would silently transpose square matrices).

Every record-parsing function below cites the pw2bgw.f90 line numbers it was
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
    0,1,2 become column 0 (rows 0,1,2), matching pw2bgw's
    ``((a(j,i), j=1,3), i=1,3)`` implied-do. Using ``order="C"`` here would
    silently transpose any non-symmetric matrix.
    """
    return np.asarray(flat_array).reshape(shape, order="F")


def _read_scalar_int32(reader, label):
    payload = reader.read_raw_record(label)
    array = _as_array(payload, _INT32, 1, label, reader.source_path)
    return int(array[0])


# ── Shared header records (byte-identical layout in WFN and RHO) ──────


def _read_title_block(reader, expected_prefix, label="record 1 (stitle, sdate, stime)"):
    """Record 1: 32-byte stitle + 32-byte sdate + 32-byte stime = 96 bytes.

    pw2bgw.f90:730-737 (WFN) / :1574-1581 (RHO) construct these as
    space-padded fixed-width CHARACTER fields; :969 (WFN) / :1715 (RHO) write
    them as one record. ``stitle`` is ``"<expected_prefix>-Real"`` or
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
    if title == f"{expected_prefix}-Real":
        is_complex_flavor = False
    elif title == f"{expected_prefix}-Complex":
        is_complex_flavor = True
    else:
        raise BerkeleyGWFormatError(
            f"{reader.source_path}: {label} stitle is {title!r}, expected "
            f"'{expected_prefix}-Real' or '{expected_prefix}-Complex'."
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

    pw2bgw.f90:981-987 (WFN) / :1720-1726 (RHO) write byte-identical formulas
    for these five records (verified by direct comparison of both
    subroutines' source).
    """
    # Record 4: celvol, alat, avec(3,3), adot(3,3) -- 20 float64 (pw2bgw.f90:981-982 /
    # :1720-1721). adot = alat^2 * avec^T @ avec (metric tensor), per the accumulation
    # loop at pw2bgw.f90:778-790 / :1645-1657 (adot(j,i) = sum_k at(k,j)*at(k,i)*alat^2).
    label = f"{section_label} record 4 (celvol, alat, avec, adot)"
    payload = reader.read_raw_record(label)
    values = _as_array(payload, _FLOAT64, 20, label, reader.source_path)
    cell_volume_bohr3 = float(values[0])
    lattice_parameter_bohr = float(values[1])
    lattice_vectors_alat_units = _reshape_fortran(values[2:11], (3, 3)).copy()
    real_space_metric_tensor_bohr2 = _reshape_fortran(values[11:20], (3, 3)).copy()

    # Record 5: recvol, blat, bvec(3,3), bdot(3,3) -- 20 float64 (pw2bgw.f90:983-984 /
    # :1722-1723). recvol = 8*pi^3/omega (pw2bgw.f90:776 / :1643); blat = tpiba = 2*pi/alat
    # (external cell_base module -- pw2bgw.f90 only imports it, see module docstring in
    # read_berkeleygw_wavefunction_file). bdot = tpiba^2 * bvec^T @ bvec, same pattern as
    # adot (pw2bgw.f90:792-804 / :1659-1671).
    label = f"{section_label} record 5 (recvol, blat, bvec, bdot)"
    payload = reader.read_raw_record(label)
    values = _as_array(payload, _FLOAT64, 20, label, reader.source_path)
    reciprocal_cell_volume_bohr3 = float(values[0])
    reciprocal_lattice_parameter_inverse_bohr = float(values[1])
    reciprocal_lattice_vectors_tpiba_units = _reshape_fortran(values[2:11], (3, 3)).copy()
    reciprocal_space_metric_tensor_inverse_bohr2 = _reshape_fortran(values[11:20], (3, 3)).copy()

    # Record 6: mtrx(3,3,ntran) int32 rotation matrices, crystal basis
    # (pw2bgw.f90:985 / :1724).
    label = f"{section_label} record 6 (symmetry rotation matrices)"
    payload = reader.read_raw_record(label)
    flat = _as_array(payload, _INT32, 9 * number_of_symmetry_operations, label, reader.source_path)
    symmetry_rotation_matrices = _reshape_fortran(
        flat, (3, 3, number_of_symmetry_operations)
    ).copy()

    # Record 7: tnp(3,ntran) float64 (pw2bgw.f90:986 / :1725). NOTE (correction to the
    # format contract doc, which calls these bare "fractional translations"): pw2bgw
    # actually stores 2*pi times a rotation-inverse-transformed fractional
    # translation, wrapped so that the CLOSED end is at +0.5, not at -0.5:
    # the interval is (-0.5, 0.5] (pw2bgw.f90:827-851: r2 = inverse(s(:,:,i)); t2 = r2^T @
    # ft(:,i), wrapped; translation(:,i) = t2 * tpi). This matters if this array's
    # *values* are ever cross-checked -- the 2*pi factor is baked in on disk.
    label = f"{section_label} record 7 (symmetry fractional translations x 2*pi)"
    payload = reader.read_raw_record(label)
    flat = _as_array(
        payload, _FLOAT64, 3 * number_of_symmetry_operations, label, reader.source_path
    )
    symmetry_fractional_translations_two_pi = _reshape_fortran(
        flat, (3, number_of_symmetry_operations)
    ).copy()

    # Record 8: (apos(3,i), Z_i) for i=1..nat, INTERLEAVED per atom: 3 float64 then 1
    # int32, repeated (pw2bgw.f90:987 / :1726). Positions are Cartesian, in alat units
    # (QE's "tau" convention). A structured dtype with no alignment padding reproduces
    # this interleaving exactly (numpy does not pad list-of-tuples dtypes unless
    # align=True is requested, which it is not here).
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

    Header layout follows pw2bgw.f90:969-999 (``write_wfng``); body layout
    follows pw2bgw.f90:1026-1291 (per-k-point G-vectors, then per-band
    coefficients).

    Body-parsing scope: this reader supports ``number_of_spin_components ==
    1`` bodies only (any flavor, real or complex). For ``ns > 1``,
    ``write_wfng``'s ``proc_wf`` branch (pw2bgw.f90:712-716 defines
    ``proc_wf = (real_or_complex == 1) .OR. (nspin > 1)``) writes a
    fundamentally different, spin-interleaved record layout
    (pw2bgw.f90:1250-1289: one record per band holding ALL spins'
    coefficients concatenated, gathered via ``wfng_dist``) rather than the
    simple one-record-per-(k,spin,band) layout this reader implements
    (pw2bgw.f90:1227-1238, the ``proc_wf == False`` branch, which is only
    taken for ``real_or_complex == 2 .AND. nspin == 1``). Rather than guess
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
        # (pw2bgw.f90:972-973; "ns" is written as nsf, "nk" as nk_g/ns -- see
        # set_spin(), pw2bgw.f90:5224-5244).
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

        # Record 3: FFTgrid(3) int32, kgrid(3) int32, kshift(3) float64
        # (pw2bgw.f90:974-979; kshift = 0.5*k1 etc. at :979).
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

        # Record 9: ngk(nk) int32 -- plane waves per k-point (pw2bgw.f90:988).
        label = "record 9 (ngk)"
        payload = reader.read_raw_record(label)
        number_of_gvectors_per_kpoint = _as_array(
            payload, _INT32, number_of_kpoints, label, path
        ).copy()
        header["number_of_gvectors_per_kpoint"] = number_of_gvectors_per_kpoint

        # Record 10: kw(nk) float64 weights, normalized so sum = 1 (pw2bgw.f90:990-991:
        # wk(ik)*nst/2, where nst==nspin for the non-spinor case -- set_spin(),
        # pw2bgw.f90:5224-5244 -- and Q3's own results.json documents its k-point
        # weights as normalized to sum to 2.0 (docs/io_reference.md), so /2 recovers 1).
        label = "record 10 (kw)"
        payload = reader.read_raw_record(label)
        header["kpoint_weights"] = _as_array(
            payload, _FLOAT64, number_of_kpoints, label, path
        ).copy()

        # Record 11: kpt(3,nk) float64, CRYSTAL (fractional) coordinates --
        # pw2bgw.f90:964 calls cryst_to_cart(..., at, -1) immediately before the write
        # block and cryst_to_cart(..., bg, +1) afterward (:1002) to restore Cartesian.
        label = "record 11 (kpt)"
        payload = reader.read_raw_record(label)
        flat = _as_array(payload, _FLOAT64, 3 * number_of_kpoints, label, path)
        header["kpoints_crystal_coordinates"] = _reshape_fortran(
            flat, (3, number_of_kpoints)
        ).copy()

        # Records 12-13: ifmin/ifmax(nk*ns) int32 (pw2bgw.f90:993-994).
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

        # Records 14-15: energies/occupations(nbands, nk*ns) float64 (pw2bgw.f90:995-996).
        # Energies are in Rydberg: et_g is copied from QE's `et` with no rescaling
        # (pw2bgw.f90:855-865), and elsewhere in this same file `et(...)  * rytoev` is
        # used whenever an eV value is needed (e.g. :4258, :5071) -- confirming `et`
        # itself (hence et_g, hence this record) is Rydberg, not eV.
        # NOTE: for ns > 1 the nk*ns axis is ordered spin-block-major (QE's global k
        # list is nk points of spin-1 followed by nk points of spin-2, per the kmap
        # construction at pw2bgw.f90:758-763) -- NOT interleaved per k-point.
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

        # Record 16: nrecord, always 1 (pw2bgw.f90:740 sets nrecord=1 once; grep of the
        # whole file shows it is never recomputed from array size before any WRITE).
        nrecord = _read_scalar_int32(reader, "record 16 (nrecord)")
        if nrecord != 1:
            raise BerkeleyGWFormatError(
                f"{path}: record 16 (nrecord) = {nrecord}, expected 1 -- this reader "
                f"(and pw2bgw.f90 itself, which hardcodes nrecord=1 throughout) does "
                f"not support multi-part chunked G-vector records."
            )

        # Record 17: ng (repeated) -- cross-check against record 2's value.
        ng_repeated = _read_scalar_int32(reader, "record 17 (ng)")
        if ng_repeated != number_of_g_vectors:
            raise BerkeleyGWFormatError(
                f"{path}: record 17 (ng) = {ng_repeated}, but record 2 already "
                f"declared number_of_g_vectors = {number_of_g_vectors}."
            )

        # Record 18: gvec(3,ng) int32 Miller indices -- the charge-density G set
        # (pw2bgw.f90:999).
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
            f"structurally different record layout in pw2bgw.f90 that is not "
            f"implemented here. The header above was parsed fully and is valid."
        )

    kpoint_gvectors = []
    wavefunction_coefficients = []
    for kpoint_index in range(number_of_kpoints):
        expected_gvector_count = int(number_of_gvectors_per_kpoint[kpoint_index])

        # [nrecord=1] [ngk(ik)] [gvec_k(3, ngk(ik))] -- pw2bgw.f90:1062-1069, written
        # once per k-point (guarded by "is == 1" in the source, i.e. only for the
        # single spin channel that exists when ns=1).
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

        # Per band: [nrecord=1] [ngk(ik)] [coefficients(ngk(ik))] -- pw2bgw.f90:1227-1238
        # (the proc_wf == False branch: real_or_complex==2 AND nspin==1, guaranteed here
        # since we already rejected ns>1 above; WFN-Real with ns=1 also reduces to this
        # exact layout since nst=1 there too -- see docstring).
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


def read_berkeleygw_charge_density_file(path):
    """Reads a BerkeleyGW ``RHO`` mean-field file (8 header records + body).

    Header layout follows pw2bgw.f90:1715-1726 (``write_rhog``) -- identical
    to WFN records 1, 4-8, but record 2 omits the k-point fields and record 3
    is FFT grid only (no kgrid/kshift).

    Returns a dict with every header field plus:
      - "charge_density_gvectors": int32 ndarray, shape (3, number_of_g_vectors).
      - "charge_density_g_times_cell_volume": complex128 (or float64, if
        real flavor) ndarray, shape (number_of_g_vectors,
        number_of_spin_components). NOTE the name: pw2bgw.f90:1706-1710
        multiplies rho(G) by the cell volume (``omega``) before writing it
        -- this is the raw on-disk BerkeleyGW convention, not the bare
        density. See ``read_berkeleygw_wavefunction_file``'s docstring for
        the general conventions (framing, dtypes, column-major order).
    """
    path = Path(path)
    with open(path, "rb") as file_handle:
        reader = _FortranSequentialReader(file_handle, path)
        header = {}
        header.update(_read_title_block(reader, "RHO"))

        # Record 2: ns, ng, ntran, cell_symmetry, nat, ecutrho -- all that's shared with
        # WFN's record 2; RHO omits nk/nbands/ngkmax/ecutwfc entirely (pw2bgw.f90:1718).
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

        # Record 3: FFTgrid(3) int32 ONLY -- no kgrid/kshift (pw2bgw.f90:1719).
        label = "record 3 (FFTgrid)"
        payload = reader.read_raw_record(label)
        header["fft_grid_dimensions"] = _as_array(payload, _INT32, 3, label, path).copy()

        header.update(
            _read_lattice_and_symmetry_records(
                reader, number_of_symmetry_operations, number_of_atoms, "RHO"
            )
        )

        # Body record group 1: [nrecord=1] [ng] [gvec(3,ng)] (pw2bgw.f90:1727-1729).
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

        # Body record group 2: [nrecord=1] [ng] [rho(ng,ns)] (pw2bgw.f90:1730-1738).
        # rho(G) is multiplied by the cell volume before writing (pw2bgw.f90:1706-1710)
        # -- see this function's docstring.
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
        header["charge_density_g_times_cell_volume"] = _reshape_fortran(
            flat, (number_of_g_vectors, number_of_spin_components)
        ).copy()

    header["source_path"] = path
    return header


# ── vxc.dat (ASCII) ───────────────────────────────────────────────────

# FORMAT (3F13.9, 2I8): kx ky kz ndiag noffdiag -- pw2bgw.f90:3516 (write_vxc_g) /
# :2931 (write_vxc_r, byte-identical format statements in both subroutines).
_VXC_HEADER_LINE_WIDTHS = (13, 13, 13, 8, 8)
# FORMAT (2I8, 2F15.9): spin band Re(Vxc) Im(Vxc) -- pw2bgw.f90:3517 / :2932.
_VXC_DIAGONAL_LINE_WIDTHS = (8, 8, 15, 15)
# FORMAT (3I8, 2F15.9): spin band_bra band_ket Re(Vxc) Im(Vxc) -- pw2bgw.f90:3518 / :2933.
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

    Format follows pw2bgw.f90's ``write_vxc_g``/``write_vxc_r`` (both write
    byte-identical output): one header line per k-point
    (``kx ky kz ndiag noffdiag``, crystal coordinates), then ``ndiag``
    diagonal-matrix-element lines, then ``noffdiag`` off-diagonal lines.
    ``ndiag``/``noffdiag`` are already multiplied by ``number_of_spin_components``
    on disk (pw2bgw.f90:3483-3484 / :2894: ``nspin * ndiag``,
    ``nspin * noffdiag**2``), so they are exactly this reader's per-k-point
    line counts, not a "per spin" count needing further multiplication.
    Matrix elements are in eV (``rytoev``-scaled at the point of computation,
    e.g. pw2bgw.f90:3340, :3439).

    Returns a dict: {"number_of_kpoints": int, "kpoints": [...]} where each
    entry of "kpoints" is a dict:
      - "kpoint_crystal_coordinates": float64 ndarray, shape (3,).
      - "diagonal_entry_count" / "offdiagonal_entry_count": int (as read from
        the header line -- already nspin-scaled, see above).
      - "diagonal_entries": list of dicts with "spin_index", "band_index",
        "vxc_ev" (Python complex).
      - "offdiagonal_entries": list of dicts with "spin_index",
        "band_index_bra" (pw2bgw's "ib2"), "band_index_ket" (pw2bgw's "ib"),
        "vxc_ev" (Python complex). Derived from pw2bgw.f90:3417-3442: the
        matrix element is ``conj(psi_ib2) . (Vxc psi_ib)``, i.e. <ib2|Vxc|ib>.
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
            [float(kx_text), float(ky_text), float(kz_text)], dtype=_FLOAT64
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
