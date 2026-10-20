"""Independent readers and metrics for Wannier90-format interface files.

This module is a deliberately independent implementation, written from the
Wannier90 user guide (v3.1.0, 5 March 2020 — ``doc/user_guide/wannier-pp.tex``
§"seedname.nnkp/mmn/amn/eig file") WITHOUT reading Wannier90's Fortran
sources or Q3's own C++ writers for these files. Its purpose is to be a
cross-check that shares no code path, assumption, or bug with the writer it
validates. Do not "fix" a disagreement between this reader and a Q3 writer by
copying the writer's interpretation here; re-derive from the user guide
instead.

Framing (the conventions below are load-bearing):

- **All indices in these files are 1-based** (Fortran origin). Every reader
  here converts to 0-based on the way in and validates the declared range,
  so an off-by-one in a writer surfaces as a loud ``WannierFormatError``
  rather than a silently shifted matrix.
- **``.mmn`` matrix element order: the first index m varies fastest.** A
  block's ``num_bands**2`` value lines therefore reshape as
  ``order="F"`` into ``M[m, n]``; ``order="C"`` would silently transpose
  every block, which for a near-Hermitian overlap is exactly the kind of bug
  that survives casual inspection.
- **``.eig`` carries no header.** Its dimensions are inferred from the
  index ranges actually present, and the reader requires the (band, k-point)
  index set to be exactly the full rectangle — a missing or duplicated line
  is an error, not a hole to be filled with zeros.
- **Energies are eV in ``.eig``** (user guide, §"seedname.eig file"), while
  Q3 works in Rydberg internally. Conversion is the caller's job; nothing
  here converts silently.
- **``.nnkp`` lattice blocks are Å (real) and Å⁻¹ (reciprocal)**, and its
  ``kpoints`` block is in crystallographic (fractional) coordinates.

Metric conventions are documented on each function; the gauge-invariant
spread follows Marzari & Vanderbilt, PRB 56, 12847 (1997). The band distance
is the package's own (``q3.workflows.band_distance``): the automatic workflow
accepts or rejects a model on it and this suite checks the workflow, so there
is one definition, and it is re-exported from here for the tests that read it
beside the readers.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import numpy as np
from pydantic import BaseModel, ConfigDict
from q3.workflows.band_distance import BandDistance as BandDistance
from q3.workflows.band_distance import band_distance as band_distance

from helpers import QE_BIN_DIR

# ── Locating the shared reference build ───────────────────────────────


def locate_reference_binary_directory() -> Path | None:
    """Directory holding the QE/Wannier90 reference binaries, or None.

    One rule for every live-QE test: ``tests/helpers.py`` honours
    ``$Q3_QE_BIN_DIR`` and otherwise walks the repository's ancestors for
    ``reference/dft/q-e/build_libxc/bin`` (which is what makes it work from a
    git worktree).
    """
    return QE_BIN_DIR if QE_BIN_DIR.is_dir() else None


# ── What a parsed file is held in ─────────────────────────────────────


class ParsedFile(BaseModel):
    """The contents of one parsed file: frozen, its numpy arrays held as they are."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)


# ── Public exception type ─────────────────────────────────────────────


class WannierFormatError(ValueError):
    """A Wannier90-format file violates this reader's format contract.

    Raised for: a malformed or missing header, a declared count that does not
    match the number of records actually present, a truncated file, an index
    outside its declared range, a duplicated or missing (band, k-point)
    entry, a non-numeric field, or an unterminated ``.nnkp`` block.
    """


# ── Small parsing utilities ───────────────────────────────────────────


def _significant_lines(path: Path) -> list[tuple[int, str]]:
    """Return ``(line_number, text)`` for every non-blank line, 1-based.

    Line numbers are preserved through filtering so error messages can point
    at the real line in the file rather than an index into a filtered list.
    """
    path = Path(path)
    if not path.is_file():
        raise WannierFormatError(f"{path}: file does not exist")
    numbered = []
    for line_number, raw in enumerate(path.read_text().splitlines(), start=1):
        text = raw.strip()
        if text:
            numbered.append((line_number, text))
    return numbered


def _parse_integers(path: Path, line_number: int, text: str, count: int) -> list[int]:
    fields = text.split()
    if len(fields) < count:
        raise WannierFormatError(
            f"{path}:{line_number}: expected {count} integers, found {len(fields)}: {text!r}"
        )
    try:
        return [int(field) for field in fields[:count]]
    except ValueError as error:
        raise WannierFormatError(
            f"{path}:{line_number}: expected {count} integers, got {text!r}"
        ) from error


def _parse_complex(path: Path, line_number: int, text: str) -> complex:
    fields = text.split()
    if len(fields) < 2:
        raise WannierFormatError(
            f"{path}:{line_number}: expected a real and an imaginary part, got {text!r}"
        )
    try:
        return complex(float(fields[0]), float(fields[1]))
    except ValueError as error:
        raise WannierFormatError(
            f"{path}:{line_number}: expected two real numbers, got {text!r}"
        ) from error


def _check_index(path: Path, line_number: int, name: str, value: int, upper: int) -> int:
    """Validate a 1-based index against ``1 <= value <= upper``; return 0-based."""
    if not 1 <= value <= upper:
        raise WannierFormatError(
            f"{path}:{line_number}: {name} index {value} outside declared range 1..{upper}"
        )
    return value - 1


# ── seedname.mmn ──────────────────────────────────────────────────────


class OverlapFile(ParsedFile):
    """Contents of a ``seedname.mmn`` file.

    ``overlaps[kpoint_index, neighbor_index]`` is the ``M[m, n]`` block for
    that (k, b) pair, 0-based in both k and neighbor position.
    ``neighbor_kpoint[k, n]`` is the 0-based index of the k-point that is the
    periodic image of k+b, and ``periodic_image_shift[k, n]`` is the integer
    reciprocal-lattice vector carrying that image to the actual k+b.
    """

    comment: str
    number_of_bands: int
    number_of_kpoints: int
    number_of_neighbors: int
    overlaps: np.ndarray  # complex, (num_kpts, nntot, num_bands, num_bands)
    neighbor_kpoint: np.ndarray  # int, (num_kpts, nntot)
    periodic_image_shift: np.ndarray  # int, (num_kpts, nntot, 3)


def read_overlap_file(path: str | Path) -> OverlapFile:
    """Read ``seedname.mmn``.

    Layout (user guide §"seedname.mmn file"): a comment line; then
    ``num_bands num_kpts nntot``; then ``num_kpts * nntot`` blocks, each a
    5-integer header (k, the image k-point of k+b, and the 3 G-vector
    components in reciprocal lattice units) followed by ``num_bands**2``
    lines of ``Re Im`` with the first index m varying fastest.

    The block headers' own k ordinals are validated to sweep k-points in
    order, each with exactly ``nntot`` neighbours — a writer that emitted
    blocks in a different order, or dropped one, fails here rather than
    producing a silently mis-associated overlap.
    """
    path = Path(path)
    lines = _significant_lines(path)
    if len(lines) < 2:
        raise WannierFormatError(f"{path}: file is too short to contain a header")

    comment = lines[0][1]
    header_line_number, header_text = lines[1]
    number_of_bands, number_of_kpoints, number_of_neighbors = _parse_integers(
        path, header_line_number, header_text, 3
    )
    for name, value in (
        ("num_bands", number_of_bands),
        ("num_kpts", number_of_kpoints),
        ("nntot", number_of_neighbors),
    ):
        if value <= 0:
            raise WannierFormatError(
                f"{path}:{header_line_number}: {name} must be positive, got {value}"
            )

    block_size = 1 + number_of_bands * number_of_bands
    expected = 2 + number_of_kpoints * number_of_neighbors * block_size
    if len(lines) != expected:
        raise WannierFormatError(
            f"{path}: header declares {number_of_kpoints} k-points x "
            f"{number_of_neighbors} neighbours x ({number_of_bands}^2 + 1) lines "
            f"= {expected} significant lines, found {len(lines)}"
        )

    overlaps = np.zeros(
        (number_of_kpoints, number_of_neighbors, number_of_bands, number_of_bands),
        dtype=complex,
    )
    neighbor_kpoint = np.zeros((number_of_kpoints, number_of_neighbors), dtype=int)
    periodic_image_shift = np.zeros((number_of_kpoints, number_of_neighbors, 3), dtype=int)

    cursor = 2
    for kpoint_index in range(number_of_kpoints):
        for neighbor_index in range(number_of_neighbors):
            line_number, text = lines[cursor]
            values = _parse_integers(path, line_number, text, 5)
            block_kpoint = _check_index(path, line_number, "k-point", values[0], number_of_kpoints)
            if block_kpoint != kpoint_index:
                raise WannierFormatError(
                    f"{path}:{line_number}: blocks must sweep k-points in order with "
                    f"{number_of_neighbors} neighbours each; expected k-point "
                    f"{kpoint_index + 1}, found {values[0]}"
                )
            neighbor_kpoint[kpoint_index, neighbor_index] = _check_index(
                path, line_number, "neighbour k-point", values[1], number_of_kpoints
            )
            periodic_image_shift[kpoint_index, neighbor_index] = values[2:5]
            cursor += 1

            flat = np.empty(number_of_bands * number_of_bands, dtype=complex)
            for element_index in range(number_of_bands * number_of_bands):
                element_line_number, element_text = lines[cursor]
                flat[element_index] = _parse_complex(path, element_line_number, element_text)
                cursor += 1
            # m fastest -> Fortran order (see this module's docstring).
            overlaps[kpoint_index, neighbor_index] = flat.reshape(
                (number_of_bands, number_of_bands), order="F"
            )

    return OverlapFile(
        comment=comment,
        number_of_bands=number_of_bands,
        number_of_kpoints=number_of_kpoints,
        number_of_neighbors=number_of_neighbors,
        overlaps=overlaps,
        neighbor_kpoint=neighbor_kpoint,
        periodic_image_shift=periodic_image_shift,
    )


# ── seedname.amn ──────────────────────────────────────────────────────


class ProjectionFile(ParsedFile):
    """Contents of a ``seedname.amn`` file.

    ``projections[kpoint_index, band_index, wannier_index]`` = A_mn(k).
    """

    comment: str
    number_of_bands: int
    number_of_kpoints: int
    number_of_wannier_functions: int
    projections: np.ndarray  # complex, (num_kpts, num_bands, num_wann)


def read_projection_file(path: str | Path) -> ProjectionFile:
    """Read ``seedname.amn``.

    Layout (user guide §"seedname.amn file"): a comment line; then
    ``num_bands num_kpts num_wann``; then ``num_bands * num_wann * num_kpts``
    lines of ``m n k Re Im``.

    Unlike ``.mmn``, every element carries its own explicit (m, n, k) triple,
    so this reader does not assume an ordering — it fills by index and then
    requires every slot to have been written exactly once. A writer emitting
    a duplicate triple (and thus leaving another slot empty) fails here.
    """
    path = Path(path)
    lines = _significant_lines(path)
    if len(lines) < 2:
        raise WannierFormatError(f"{path}: file is too short to contain a header")

    comment = lines[0][1]
    header_line_number, header_text = lines[1]
    number_of_bands, number_of_kpoints, number_of_wannier_functions = _parse_integers(
        path, header_line_number, header_text, 3
    )
    for name, value in (
        ("num_bands", number_of_bands),
        ("num_kpts", number_of_kpoints),
        ("num_wann", number_of_wannier_functions),
    ):
        if value <= 0:
            raise WannierFormatError(
                f"{path}:{header_line_number}: {name} must be positive, got {value}"
            )

    expected = 2 + number_of_bands * number_of_wannier_functions * number_of_kpoints
    if len(lines) != expected:
        raise WannierFormatError(
            f"{path}: header declares {number_of_bands} x {number_of_wannier_functions} x "
            f"{number_of_kpoints} = {expected - 2} element lines, found {len(lines) - 2}"
        )

    projections = np.zeros(
        (number_of_kpoints, number_of_bands, number_of_wannier_functions), dtype=complex
    )
    written = np.zeros(projections.shape, dtype=bool)

    for line_number, text in lines[2:]:
        fields = text.split()
        if len(fields) < 5:
            raise WannierFormatError(f"{path}:{line_number}: expected 'm n k Re Im', got {text!r}")
        indices = _parse_integers(path, line_number, text, 3)
        band_index = _check_index(path, line_number, "band", indices[0], number_of_bands)
        wannier_index = _check_index(
            path, line_number, "Wannier", indices[1], number_of_wannier_functions
        )
        kpoint_index = _check_index(path, line_number, "k-point", indices[2], number_of_kpoints)
        try:
            value = complex(float(fields[3]), float(fields[4]))
        except ValueError as error:
            raise WannierFormatError(
                f"{path}:{line_number}: expected two real numbers after the indices, got {text!r}"
            ) from error

        if written[kpoint_index, band_index, wannier_index]:
            raise WannierFormatError(
                f"{path}:{line_number}: duplicate entry for "
                f"(band {indices[0]}, Wannier {indices[1]}, k-point {indices[2]})"
            )
        projections[kpoint_index, band_index, wannier_index] = value
        written[kpoint_index, band_index, wannier_index] = True

    if not written.all():
        missing = int((~written).sum())
        raise WannierFormatError(
            f"{path}: {missing} (band, Wannier, k-point) combinations were never written"
        )

    return ProjectionFile(
        comment=comment,
        number_of_bands=number_of_bands,
        number_of_kpoints=number_of_kpoints,
        number_of_wannier_functions=number_of_wannier_functions,
        projections=projections,
    )


# ── seedname.eig ──────────────────────────────────────────────────────


class EigenvalueFile(ParsedFile):
    """Contents of a ``seedname.eig`` file.

    ``eigenvalues[kpoint_index, band_index]`` in **eV** (the file's own unit;
    nothing here converts to Rydberg).
    """

    number_of_bands: int
    number_of_kpoints: int
    eigenvalues: np.ndarray  # float, (num_kpts, num_bands)


def read_eigenvalue_file(path: str | Path) -> EigenvalueFile:
    """Read ``seedname.eig``.

    Layout (user guide §"seedname.eig file"): every line is
    ``band_index kpoint_index eigenvalue``, with no header at all. Dimensions
    are therefore inferred from the maximum indices present, and the index
    set is then required to be exactly the full rectangle — see this module's
    docstring on why a hole is an error rather than a zero.
    """
    path = Path(path)
    lines = _significant_lines(path)
    if not lines:
        raise WannierFormatError(f"{path}: file contains no data lines")

    entries: list[tuple[int, int, int, float]] = []
    maximum_band = 0
    maximum_kpoint = 0
    for line_number, text in lines:
        fields = text.split()
        if len(fields) < 3:
            raise WannierFormatError(
                f"{path}:{line_number}: expected 'band kpoint eigenvalue', got {text!r}"
            )
        indices = _parse_integers(path, line_number, text, 2)
        try:
            eigenvalue = float(fields[2])
        except ValueError as error:
            raise WannierFormatError(
                f"{path}:{line_number}: eigenvalue is not a real number: {text!r}"
            ) from error
        if indices[0] < 1 or indices[1] < 1:
            raise WannierFormatError(
                f"{path}:{line_number}: band and k-point indices are 1-based, got {text!r}"
            )
        maximum_band = max(maximum_band, indices[0])
        maximum_kpoint = max(maximum_kpoint, indices[1])
        entries.append((line_number, indices[0], indices[1], eigenvalue))

    eigenvalues = np.zeros((maximum_kpoint, maximum_band), dtype=float)
    written = np.zeros(eigenvalues.shape, dtype=bool)
    for line_number, band, kpoint, eigenvalue in entries:
        if written[kpoint - 1, band - 1]:
            raise WannierFormatError(
                f"{path}:{line_number}: duplicate entry for band {band}, k-point {kpoint}"
            )
        eigenvalues[kpoint - 1, band - 1] = eigenvalue
        written[kpoint - 1, band - 1] = True

    if not written.all():
        missing = int((~written).sum())
        raise WannierFormatError(
            f"{path}: index range implies {maximum_kpoint} k-points x {maximum_band} bands, "
            f"but {missing} (band, k-point) combinations are absent"
        )

    return EigenvalueFile(
        number_of_bands=maximum_band,
        number_of_kpoints=maximum_kpoint,
        eigenvalues=eigenvalues,
    )


# ── seedname.nnkp ─────────────────────────────────────────────────────


class NeighborListFile(ParsedFile):
    """Contents of the blocks of a ``seedname.nnkp`` this harness needs.

    ``real_lattice`` rows are A1, A2, A3 in Å; ``reciprocal_lattice`` rows are
    B1, B2, B3 in Å⁻¹; ``kpoints`` are fractional. ``neighbor_kpoint`` and
    ``periodic_image_shift`` mirror ``OverlapFile``'s meaning.
    """

    real_lattice: np.ndarray  # float, (3, 3)
    reciprocal_lattice: np.ndarray  # float, (3, 3)
    kpoints: np.ndarray  # float, (num_kpts, 3)
    number_of_neighbors: int
    neighbor_kpoint: np.ndarray  # int, (num_kpts, nntot)
    periodic_image_shift: np.ndarray  # int, (num_kpts, nntot, 3)


def _extract_block(path: Path, lines: list[tuple[int, str]], name: str) -> list[tuple[int, str]]:
    """Return the body lines of ``begin <name>`` ... ``end <name>``."""
    start = None
    for position, (line_number, text) in enumerate(lines):
        if text.lower().split() == ["begin", name]:
            start = position
            break
    if start is None:
        raise WannierFormatError(f"{path}: no 'begin {name}' block found")
    for position in range(start + 1, len(lines)):
        if lines[position][1].lower().split() == ["end", name]:
            return lines[start + 1 : position]
    raise WannierFormatError(
        f"{path}:{lines[start][0]}: 'begin {name}' block is never closed by 'end {name}'"
    )


def _read_three_by_three(path: Path, body: list[tuple[int, str]], name: str) -> np.ndarray:
    if len(body) != 3:
        raise WannierFormatError(
            f"{path}: '{name}' block must hold exactly 3 rows, found {len(body)}"
        )
    matrix = np.zeros((3, 3), dtype=float)
    for row, (line_number, text) in enumerate(body):
        fields = text.split()
        if len(fields) < 3:
            raise WannierFormatError(
                f"{path}:{line_number}: '{name}' row needs 3 components, got {text!r}"
            )
        try:
            matrix[row] = [float(field) for field in fields[:3]]
        except ValueError as error:
            raise WannierFormatError(
                f"{path}:{line_number}: '{name}' row is not numeric: {text!r}"
            ) from error
    return matrix


def read_neighbor_list_file(path: str | Path) -> NeighborListFile:
    """Read the ``real_lattice``, ``recip_lattice``, ``kpoints`` and ``nnkpts``
    blocks of ``seedname.nnkp``.

    Layout (user guide §"seedname.nnkp file"): ``begin <name>`` /
    ``end <name>`` delimited blocks. The ``kpoints`` and ``nnkpts`` blocks
    each open with their own count, which this reader cross-checks against
    the number of rows actually present — the ``.nnkp`` gate in the plan
    compares shells and b-ordering, so a count/row mismatch has to be an
    error rather than a silent truncation.
    """
    path = Path(path)
    lines = _significant_lines(path)

    real_lattice = _read_three_by_three(
        path, _extract_block(path, lines, "real_lattice"), "real_lattice"
    )
    reciprocal_lattice = _read_three_by_three(
        path, _extract_block(path, lines, "recip_lattice"), "recip_lattice"
    )

    kpoint_body = _extract_block(path, lines, "kpoints")
    if not kpoint_body:
        raise WannierFormatError(f"{path}: 'kpoints' block is empty")
    count_line_number, count_text = kpoint_body[0]
    number_of_kpoints = _parse_integers(path, count_line_number, count_text, 1)[0]
    if number_of_kpoints <= 0:
        raise WannierFormatError(
            f"{path}:{count_line_number}: k-point count must be positive, got {number_of_kpoints}"
        )
    if len(kpoint_body) - 1 != number_of_kpoints:
        raise WannierFormatError(
            f"{path}:{count_line_number}: 'kpoints' block declares {number_of_kpoints} "
            f"k-points but holds {len(kpoint_body) - 1} rows"
        )
    kpoints = np.zeros((number_of_kpoints, 3), dtype=float)
    for row, (line_number, text) in enumerate(kpoint_body[1:]):
        fields = text.split()
        if len(fields) < 3:
            raise WannierFormatError(
                f"{path}:{line_number}: k-point needs 3 fractional components, got {text!r}"
            )
        try:
            kpoints[row] = [float(field) for field in fields[:3]]
        except ValueError as error:
            raise WannierFormatError(
                f"{path}:{line_number}: k-point row is not numeric: {text!r}"
            ) from error

    neighbor_body = _extract_block(path, lines, "nnkpts")
    if not neighbor_body:
        raise WannierFormatError(f"{path}: 'nnkpts' block is empty")
    count_line_number, count_text = neighbor_body[0]
    number_of_neighbors = _parse_integers(path, count_line_number, count_text, 1)[0]
    if number_of_neighbors <= 0:
        raise WannierFormatError(
            f"{path}:{count_line_number}: nntot must be positive, got {number_of_neighbors}"
        )
    expected_rows = number_of_kpoints * number_of_neighbors
    if len(neighbor_body) - 1 != expected_rows:
        raise WannierFormatError(
            f"{path}:{count_line_number}: 'nnkpts' block declares nntot={number_of_neighbors} "
            f"for {number_of_kpoints} k-points ({expected_rows} rows) but holds "
            f"{len(neighbor_body) - 1}"
        )

    neighbor_kpoint = np.zeros((number_of_kpoints, number_of_neighbors), dtype=int)
    periodic_image_shift = np.zeros((number_of_kpoints, number_of_neighbors, 3), dtype=int)
    for row, (line_number, text) in enumerate(neighbor_body[1:]):
        values = _parse_integers(path, line_number, text, 5)
        kpoint_index = _check_index(path, line_number, "k-point", values[0], number_of_kpoints)
        neighbor_index = row % number_of_neighbors
        if kpoint_index != row // number_of_neighbors:
            raise WannierFormatError(
                f"{path}:{line_number}: 'nnkpts' rows must sweep k-points in order with "
                f"{number_of_neighbors} neighbours each; expected k-point "
                f"{row // number_of_neighbors + 1}, found {values[0]}"
            )
        neighbor_kpoint[kpoint_index, neighbor_index] = _check_index(
            path, line_number, "neighbour k-point", values[1], number_of_kpoints
        )
        periodic_image_shift[kpoint_index, neighbor_index] = values[2:5]

    return NeighborListFile(
        real_lattice=real_lattice,
        reciprocal_lattice=reciprocal_lattice,
        kpoints=kpoints,
        number_of_neighbors=number_of_neighbors,
        neighbor_kpoint=neighbor_kpoint,
        periodic_image_shift=periodic_image_shift,
    )


# ── b-vectors and finite-difference weights ───────────────────────────


def compute_b_vectors(neighbor_list: NeighborListFile) -> np.ndarray:
    """Cartesian b-vectors in Å⁻¹, shape ``(num_kpts, nntot, 3)``.

    b = (k_neighbour + G) − k, with the fractional difference carried into
    Cartesian by the reciprocal lattice. Computed per k-point rather than
    once, so a ``.nnkp`` whose neighbour table is inconsistent across
    k-points shows up as a spread in |b| rather than being averaged away.
    """
    number_of_kpoints, number_of_neighbors = neighbor_list.neighbor_kpoint.shape
    b_vectors = np.zeros((number_of_kpoints, number_of_neighbors, 3), dtype=float)
    for kpoint_index in range(number_of_kpoints):
        origin = neighbor_list.kpoints[kpoint_index]
        for neighbor_index in range(number_of_neighbors):
            target = (
                neighbor_list.kpoints[neighbor_list.neighbor_kpoint[kpoint_index, neighbor_index]]
                + neighbor_list.periodic_image_shift[kpoint_index, neighbor_index]
            )
            b_vectors[kpoint_index, neighbor_index] = (
                target - origin
            ) @ neighbor_list.reciprocal_lattice
    return b_vectors


def solve_finite_difference_weights(b_vectors: np.ndarray, tolerance: float = 1e-8) -> np.ndarray:
    """Per-neighbour weights w_b (Å²) satisfying the B1 completeness condition.

    Solves Σ_b w_b b_α b_β = δ_αβ (Marzari & Vanderbilt 1997, App. B) in the
    least-squares sense over the 6 independent components of the symmetric
    tensor, using the b-vectors of the FIRST k-point (on a uniform mesh every
    k-point carries the same shell of b-vectors). Weights are shared by all
    b of equal length, which this solve produces naturally because equal-|b|
    vectors contribute identical rows.

    Raises ``WannierFormatError`` if the residual exceeds ``tolerance`` — i.e.
    the neighbour set cannot satisfy completeness at all, which is a defect in
    the file being read, not a numerical accident.
    """
    if b_vectors.ndim != 3 or b_vectors.shape[2] != 3:
        raise ValueError("b_vectors must have shape (num_kpts, nntot, 3)")
    shell = b_vectors[0]
    # Rows: the 6 independent components xx, yy, zz, xy, xz, yz.
    design = np.stack(
        [
            shell[:, 0] * shell[:, 0],
            shell[:, 1] * shell[:, 1],
            shell[:, 2] * shell[:, 2],
            shell[:, 0] * shell[:, 1],
            shell[:, 0] * shell[:, 2],
            shell[:, 1] * shell[:, 2],
        ],
        axis=0,
    )
    target = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
    weights, *_ = np.linalg.lstsq(design, target, rcond=None)
    residual = float(np.linalg.norm(design @ weights - target))
    if residual > tolerance:
        raise WannierFormatError(
            f"b-vector set does not satisfy the B1 completeness condition "
            f"(residual {residual:.3e} > {tolerance:.1e}); the neighbour list is unusable "
            f"for finite differences"
        )
    return weights


# ── Metrics ───────────────────────────────────────────────────────────


def gauge_invariant_spread(
    overlaps: np.ndarray, weights: np.ndarray, number_of_wannier_functions: int
) -> float:
    """Ω_I in Å², from raw overlaps — no gauge, no minimisation involved.

    Ω_I = (1/N_k) Σ_k Σ_b w_b [ N_wann − Σ_mn |M_mn^(k,b)|² ]
    (Marzari & Vanderbilt 1997). This is the quantity that is invariant under
    any per-k unitary, which is exactly why the plan compares it across codes
    rather than comparing M elements directly: two correct codes will
    generally produce different M and identical Ω_I.

    ``overlaps`` is ``(num_kpts, nntot, num_bands, num_bands)``; for an
    isolated manifold ``num_bands == num_wann``.
    """
    if overlaps.ndim != 4:
        raise ValueError("overlaps must have shape (num_kpts, nntot, num_bands, num_bands)")
    number_of_kpoints, number_of_neighbors = overlaps.shape[:2]
    if weights.shape != (number_of_neighbors,):
        raise ValueError(f"weights must have shape ({number_of_neighbors},), got {weights.shape}")
    squared_magnitudes = np.abs(overlaps) ** 2
    per_pair = squared_magnitudes.sum(axis=(2, 3))  # (num_kpts, nntot)
    deficit = number_of_wannier_functions - per_pair
    return float((weights[None, :] * deficit).sum() / number_of_kpoints)


def projectabilities(projections: np.ndarray) -> np.ndarray:
    """p_nk = Σ_m |A_mn(k)|², shape ``(num_kpts, num_bands)``.

    ``projections`` is ``(num_kpts, num_bands, num_wann)``; the sum runs over
    the Wannier/trial index, leaving one number per Bloch state. Meaningful in
    [0, 1] only when the trial orbitals are orthonormal in the relevant metric
    (the S metric for ultrasoft/PAW) — this function does not and cannot check
    that; it is the caller's contract.
    """
    if projections.ndim != 3:
        raise ValueError("projections must have shape (num_kpts, num_bands, num_wann)")
    return (np.abs(projections) ** 2).sum(axis=2)


# ── Wannier90 output (.wout) ─────────────────────────────────────────────
#
# Written from the layout of the "Initial State" / "Final State" blocks in the
# Wannier90 user guide's description of the output file (v3.1, §8; unchanged in
# v4.0), independently of any code that produces it. Centres are in Angstrom
# and spreads in Angstrom squared, which is what Wannier90 prints; nothing here
# converts them, so a caller comparing against Q3's bohr must do that itself
# and say so.


class Wannier90State(ParsedFile):
    """One centre-and-spread block of a .wout, plus the Omega decomposition
    that follows the Final State block (absent for the Initial State)."""

    centers_angstrom: np.ndarray  # [wannier][3]
    spreads_angstrom_squared: np.ndarray  # [wannier]
    total_spread_angstrom_squared: float
    omega_invariant: float | None = None
    omega_diagonal: float | None = None
    omega_off_diagonal: float | None = None
    omega_total: float | None = None


_CENTRE_LINE = re.compile(
    r"WF centre and spread\s+(\d+)\s+\(\s*([-+0-9.Ee]+),\s*([-+0-9.Ee]+),\s*([-+0-9.Ee]+)\s*\)\s+([-+0-9.Ee]+)"
)
_SUM_LINE = re.compile(r"Sum of centres and spreads\s+\([^)]*\)\s+([-+0-9.Ee]+)")
_OMEGA_LINE = re.compile(r"Omega\s+(I|D|OD|Total)\s+=\s+([-+0-9.Ee]+)")


def read_wannier90_output(path: str | Path, state: str = "Final State") -> Wannier90State:
    """Parse the named state block ("Initial State" or "Final State") of a .wout.

    Raises WannierFormatError if the block is absent -- a run that crashed
    before its Final State has an Initial State and nothing else, and that
    distinction is the whole reason the block name is an argument.
    """
    path = Path(path)
    text = path.read_text()
    start = text.find(state)
    if start < 0:
        raise WannierFormatError(f"{path}: no '{state}' block -- did the run reach it?")
    block = text[start:]
    end = block.find("\n\n", block.find("Sum of centres and spreads"))
    body = block if end < 0 else block[: end + 1]

    centres = []
    spreads = []
    for match in _CENTRE_LINE.finditer(body):
        centres.append([float(match.group(2)), float(match.group(3)), float(match.group(4))])
        spreads.append(float(match.group(5)))
    if not centres:
        raise WannierFormatError(f"{path}: '{state}' block has no 'WF centre and spread' lines")
    total_match = _SUM_LINE.search(body)
    if total_match is None:
        raise WannierFormatError(f"{path}: '{state}' block has no 'Sum of centres and spreads'")

    omegas: dict[str, float] = {}
    if state == "Final State":
        # The decomposition follows the block; scan a bounded window after it.
        for match in _OMEGA_LINE.finditer(block[: len(body) + 1200]):
            omegas[match.group(1)] = float(match.group(2))

    return Wannier90State(
        centers_angstrom=np.array(centres),
        spreads_angstrom_squared=np.array(spreads),
        total_spread_angstrom_squared=float(total_match.group(1)),
        omega_invariant=omegas.get("I"),
        omega_diagonal=omegas.get("D"),
        omega_off_diagonal=omegas.get("OD"),
        omega_total=omegas.get("Total"),
    )


# ── Wannier90 as an oracle ────────────────────────────────────────────────
#
# Not the 3.1.0 inside the QE tree: that build's minimiser segfaults at its
# first iteration on any input, and only its `-pp` mode has ever worked here.
# The 4.0.1 tree under reference/ (LGPL-2.1, provenance in its reference.json)
# is the one with projectability disentanglement and guiding centres. It lives
# in an ANCESTOR of the repository, like the QE build, and is never on a CI
# runner, so every test that needs it skips rather than fails when it is absent.


def locate_wannier90_binary() -> Path | None:
    """`$Q3_WANNIER90_BIN`, else `reference/wannier90-4.0.1/build/wannier90.x` under any ancestor."""
    override = os.environ.get("Q3_WANNIER90_BIN")
    if override:
        candidate = Path(override).expanduser()
        return candidate if candidate.is_file() else None
    # tests/helpers_wannier.py -> the repository root, then every ancestor above it.
    repository_root = Path(__file__).resolve().parents[1]
    for ancestor in [repository_root, *repository_root.parents]:
        candidate = ancestor / "reference" / "wannier90-4.0.1" / "build" / "wannier90.x"
        if candidate.is_file():
            return candidate
    return None


def write_wannier90_input(
    path: str | Path,
    *,
    real_lattice_angstrom: np.ndarray,
    atoms_fractional: list[tuple[str, tuple[float, float, float]]],
    kpoints_fractional: list[tuple[float, float, float]],
    mp_grid: tuple[int, int, int],
    number_of_bands: int,
    guiding_centers_angstrom: np.ndarray,
    fermi_energy_ev: float | None = None,
    frozen_window_above_fermi_ev: float | None = None,
    projectability_minimum: float | None = None,
    projectability_maximum: float | None = None,
) -> Path:
    """A `.win` that fronts Q3's exported matrices for a full Wannier90 run.

    The projections block carries one `c=` (Cartesian, Angstrom) entry per
    Wannier function and exists only to give `guiding_centres` its centres;
    the `.amn` is Q3's, so the angular labels never matter and `s` is used
    throughout. The k-points are listed in the order Q3 enumerated them, which
    is what makes Q3's `.eig`/`.amn`/`.mmn` rows line up. Written from the
    Wannier90 user guide's description of the input file (v3.1 §2, v4.0
    unchanged), independently of any writer in the C++.
    """
    path = Path(path)
    number_of_wannier_functions = int(len(guiding_centers_angstrom))
    lines = [f"num_wann = {number_of_wannier_functions}", f"num_bands = {number_of_bands}", ""]
    lines.append("begin projections")
    for center in guiding_centers_angstrom:
        lines.append(f"c={center[0]:.8f},{center[1]:.8f},{center[2]:.8f}:s")
    lines += ["end projections", "", "begin unit_cell_cart", "ang"]
    for row in real_lattice_angstrom:
        lines.append(f"  {row[0]:.10f} {row[1]:.10f} {row[2]:.10f}")
    lines += ["end unit_cell_cart", "", "begin atoms_frac"]
    for symbol, fractional in atoms_fractional:
        lines.append(f"{symbol} {fractional[0]:.8f} {fractional[1]:.8f} {fractional[2]:.8f}")
    lines += [
        "end atoms_frac",
        "",
        f"mp_grid = {mp_grid[0]} {mp_grid[1]} {mp_grid[2]}",
        "",
        "begin kpoints",
    ]
    for kpoint in kpoints_fractional:
        lines.append(f"  {kpoint[0]:.10f} {kpoint[1]:.10f} {kpoint[2]:.10f}")
    lines += ["end kpoints", ""]
    if number_of_bands > number_of_wannier_functions:
        lines += [
            "dis_froz_proj = .true.",
            f"dis_proj_min = {0.01 if projectability_minimum is None else projectability_minimum}",
            f"dis_proj_max = {0.95 if projectability_maximum is None else projectability_maximum}",
            "dis_num_iter = 4000",
            "dis_mix_ratio = 0.5",
            "dis_conv_tol = 1.0e-10",
            "dis_conv_window = 5",
        ]
        if fermi_energy_ev is not None:
            window = 0.0 if frozen_window_above_fermi_ev is None else frozen_window_above_fermi_ev
            lines.append(f"dis_froz_max = {fermi_energy_ev + window:.8f}")
        lines.append("")
    lines += [
        "num_iter = 1000",
        "conv_window = 5",
        "conv_tol = 1.0e-10",
        "guiding_centres = .true.",
        "",
    ]
    path.write_text("\n".join(lines))
    return path
