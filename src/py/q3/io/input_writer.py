"""q3.io.input_writer — Q3 YAML input file construction utilities.

Provides helpers to:

* Validate and normalise calculation type in an existing YAML file
  (``ensure_calculation_type``).
* Write a fresh SCF input YAML from structured settings
  (``write_scf_input``).
* Write an NSCF input YAML with an explicit k-point list
  (``write_nscf_input``).
* Write the SCF input for a supercell of an existing input's cell
  (``write_supercell_scf_input``).
* Write a material document to the ESSE JSON file the input refers to
  (``write_material_json``), so a structure held in memory can be run without
  first being saved by hand.

The YAML format mirrors the examples in ``examples/`` and is documented in
``src/io/input/parser/yaml.hpp``.
"""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path

import numpy as np
import yaml

from q3.errors import Q3InputError
from q3.io.input_reader import make_file_references_absolute

# ---------------------------------------------------------------------------
# Recognised aliases per calculation type
# ---------------------------------------------------------------------------

_CALCULATION_ALIASES: dict[str, tuple[str, ...]] = {
    "self_consistent_field": ("scf",),
    "non_self_consistent_field": ("nscf", "bands", "band_structure"),
    "relaxation": ("relax",),
    "density_of_states": ("dos",),
    "verification": ("verify", "restart_stability"),
}


def _canonical_calculation(raw: str) -> str:
    """Return the canonical calculation type for *raw*, or raise Q3InputError."""
    raw_lower = raw.strip().lower()
    for canonical, aliases in _CALCULATION_ALIASES.items():
        if raw_lower == canonical or raw_lower in aliases:
            return canonical
    known = ", ".join([c] + list(a) for c, a in _CALCULATION_ALIASES.items())
    raise Q3InputError(f"Unknown calculation type '{raw}'.  Known types: {known}")


# ---------------------------------------------------------------------------
# ensure_calculation_type
# ---------------------------------------------------------------------------


def ensure_calculation_type(
    input_path: str | Path,
    expected: str,
    aliases: tuple[str, ...] = (),
) -> Path:
    """Validate that *input_path* specifies the expected calculation type.

    Reads the YAML, checks ``control.calculation``, and raises
    ``Q3InputError`` if it does not match *expected* (or *aliases*).

    Parameters
    ----------
    input_path:
        Path to the YAML input file.
    expected:
        The canonical calculation type (e.g. ``"self_consistent_field"``).
    aliases:
        Additional aliases accepted for *expected* (e.g. ``("scf",)``).

    Returns
    -------
    Path
        The resolved path to the input file (unchanged).

    Raises
    ------
    Q3InputError
        If the calculation type does not match.
    """
    input_path = Path(input_path).expanduser().resolve()

    with open(input_path) as file_handle:
        config = yaml.safe_load(file_handle)

    raw = config.get("control", {}).get("calculation", "scf")
    canonical = _canonical_calculation(raw)

    allowed = {expected} | set(aliases)
    if canonical not in {_canonical_calculation(a) for a in allowed}:
        raise Q3InputError(
            f"Input file '{input_path}' specifies calculation='{raw}' "
            f"(canonical: '{canonical}'), but expected '{expected}'."
        )

    return input_path


# ---------------------------------------------------------------------------
# write_scf_input
# ---------------------------------------------------------------------------


def write_material_json(material, output_path: str | Path) -> Path:
    """Write a material document to *output_path* as ESSE JSON.

    Q3 reads its structure from a JSON file named by the manifest's ``material``
    key, so a structure that exists only in memory -- built by a workflow, or
    handed over as a ``mat3ra.made`` ``Material`` -- has to be written down
    before it can be run. This is that step, and it is deliberately the only
    thing this function does.

    *material* may be:

    * a ``dict`` already in ESSE form, as
      :func:`q3.structure.ase_atoms.ase_atoms_to_material_json` returns;
    * any object exposing ``to_dict()`` or ``to_json()``, which is how
      ``mat3ra.made.Material`` and its relatives present themselves. Accepting
      the protocol rather than the class keeps ``mat3ra.made`` out of this
      package's dependencies, where it is not otherwise needed;
    * a JSON string.

    Parameters
    ----------
    material:
        The structure, in one of the forms above.
    output_path:
        Where to write the JSON.

    Returns
    -------
    Path
        The path written, for passing straight to ``write_scf_input``.

    Raises
    ------
    Q3InputError
        If *material* is none of those forms, or does not carry the ``lattice``
        and ``basis`` keys Q3's reader requires. Both are checked here because
        the alternative is a parser error naming a missing key in a file the
        caller never opened.
    """
    if isinstance(material, str):
        try:
            document = json.loads(material)
        except json.JSONDecodeError as error:
            raise Q3InputError(
                f"write_material_json: the string given is not valid JSON ({error})."
            ) from error
    elif isinstance(material, dict):
        document = material
    elif hasattr(material, "to_dict"):
        document = material.to_dict()
    elif hasattr(material, "to_json"):
        document = json.loads(material.to_json())
    else:
        raise Q3InputError(
            "write_material_json: expected an ESSE dict, a JSON string, or an object with "
            f"to_dict()/to_json() (as mat3ra.made.Material has); got {type(material).__name__}."
        )

    missing = [key for key in ("lattice", "basis") if key not in document]
    if missing:
        raise Q3InputError(
            "write_material_json: the material document is missing "
            + ", ".join(f"'{key}'" for key in missing)
            + ". Q3's reader requires both 'lattice' and 'basis'; present keys: "
            + ", ".join(sorted(document))
            or "none"
        )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(document, indent=4) + "\n")
    return output_path


def write_scf_input(
    output_path: str | Path,
    *,
    material_path: str | Path,
    pseudopotentials: dict[str, str],
    pseudopotential_subtype: str = "norm-conserving",
    wavefunction_cutoff: float = 48.0,
    density_cutoff: float | None = None,
    kgrid_dimensions: tuple[int, int, int] = (4, 4, 4),
    kgrid_shifts: tuple[int, int, int] = (0, 0, 0),
    number_of_bands: int = 0,
    convergence_threshold: float = 1.0e-8,
    maximum_iterations: int | None = None,
    mixing_mode: str = "broyden",
    mixing_beta: float = 0.7,
    mixing_ndim: int = 8,
    exchange_correlation_functional: str | None = None,
    outdir: str = "./output",
    prefix: str = "q3",
    smearing: str = "none",
    smearing_width: float = 0.0,
) -> Path:
    """Write a Q3 YAML input file for an SCF calculation.

    Parameters
    ----------
    output_path:
        Where to write the YAML file.
    material_path:
        Path to the material JSON (ESSE format).
    pseudopotentials:
        Mapping of element symbol → UPF file path.
    pseudopotential_subtype:
        ``"norm-conserving"`` or ``"ultrasoft"`` or ``"paw"``.
    wavefunction_cutoff:
        Plane-wave kinetic energy cutoff in Ry.
    density_cutoff:
        Charge density cutoff in Ry.  Defaults to 4× wfc cutoff.
    kgrid_dimensions:
        Monkhorst–Pack grid dimensions ``(n1, n2, n3)``.
    kgrid_shifts:
        MP grid shifts ``(s1, s2, s3)`` (0 or 1).
    number_of_bands:
        Number of bands to compute.  ``0`` → auto.
    convergence_threshold:
        SCF convergence threshold in Ry.
    maximum_iterations:
        The most SCF iterations the run may take.  ``None`` leaves the key out,
        so the engine's default applies.
    mixing_mode:
        Density mixing algorithm: ``"broyden"`` or ``"anderson"``.
    mixing_beta:
        Mixing parameter β.
    mixing_ndim:
        Mixing history size.
    exchange_correlation_functional:
        XC functional label (``"pbe"``, ``"lda"``, …). ``None`` leaves it out,
        and the engine runs the functional the pseudopotentials state, as it
        does for any input that names none.
    outdir:
        Output directory written into the YAML.
    prefix:
        File prefix written into the YAML.
    smearing:
        Smearing scheme: ``"none"``, ``"gaussian"``, ``"fermi-dirac"``, etc.
    smearing_width:
        Smearing width in Ry.

    Returns
    -------
    Path
        Resolved path to the written file.
    """
    output_path = Path(output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    effective_density_cutoff = density_cutoff if density_cutoff else 4.0 * wavefunction_cutoff

    config: dict = {
        "material": str(material_path),
        "model": {
            "type": "dft",
            "subtype": "gga",
        },
        "method": {
            "type": "pseudopotential",
            "subtype": pseudopotential_subtype,
            "cutoffs": {
                "wavefunction": wavefunction_cutoff,
                "density": effective_density_cutoff,
            },
            "pseudopotentials": pseudopotentials,
            "scf": {
                "convergence_threshold": convergence_threshold,
                "number_of_bands": number_of_bands,
                "mixing": {
                    "mode": mixing_mode,
                    "beta": mixing_beta,
                    "ndim": mixing_ndim,
                },
                "kgrid": {
                    "dimensions": list(kgrid_dimensions),
                    "shifts": list(kgrid_shifts),
                },
                "smearing": {
                    "type": smearing,
                    "degauss": smearing_width,
                },
            },
        },
        "control": {
            "calculation": "scf",
            "outdir": outdir,
            "prefix": prefix,
        },
    }
    if exchange_correlation_functional is not None:
        config["model"]["functional"] = exchange_correlation_functional

    if maximum_iterations is not None:
        config["method"]["scf"]["max_iterations"] = maximum_iterations

    with open(output_path, "w") as file_handle:
        yaml.dump(config, file_handle, default_flow_style=False, sort_keys=False)

    return output_path


# ---------------------------------------------------------------------------
# write_nscf_input
# ---------------------------------------------------------------------------


def write_nscf_input(
    output_path: str | Path,
    *,
    scf_input_path: str | Path,
    kpoints_list: list[tuple[float, float, float, float]],
    number_of_bands: int | None = None,
    prefix_override: str | None = None,
    compute_projections: bool = False,
    uniform_grid: bool = False,
) -> Path:
    """Write a Q3 YAML input file for an NSCF calculation.

    Copies all settings from *scf_input_path* and overrides:

    * ``control.calculation`` → ``nscf``
    * ``control.kpoints_list`` → *kpoints_list*
    * ``method.scf.number_of_bands`` → *number_of_bands* (if provided)
    * ``control.compute_projections`` → *compute_projections* (if ``True``)
    * ``control.band_path`` and ``control.band_path_points_per_segment`` are
      dropped (if *uniform_grid*)

    The output ``prefix`` remains the same as the SCF run so that the
    NSCF run can find the ``effective-potential.q3cp`` checkpoint written
    by the SCF run.

    Parameters
    ----------
    output_path:
        Where to write the NSCF YAML file.
    scf_input_path:
        Path to the SCF input YAML (used as template).
    kpoints_list:
        Explicit k-points as a list of ``(kx, ky, kz, weight)`` tuples in
        crystal (fractional) coordinates.
    number_of_bands:
        Override for number of bands.  When ``None``, the SCF value is kept.
    prefix_override:
        Override for the prefix.  When ``None``, the SCF prefix is kept.
    compute_projections:
        When ``True``, sets ``control.compute_projections: true`` so the NSCF
        run also computes ortho-atomic/Löwdin orbital projections (used by
        ``q3.workflows.pdos``).  Omitted from the YAML when ``False``.
    uniform_grid:
        When ``True``, *kpoints_list* is a uniform grid rather than a band
        path, so the template's band path does not describe it.  The engine
        reads ``control.band_path_points_per_segment`` as the length of each
        segment of *kpoints_list*: it would stop seeding a k-point's
        wavefunctions from its neighbour at every multiple of it, and re-solve
        the points it takes for segment edges.  Both band-path keys are
        dropped, so a band-structure manifest can serve as the template of a
        PDOS or interpolation grid.

    Returns
    -------
    Path
        Resolved path to the written NSCF input file.
    """
    output_path = Path(output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    scf_input_path = Path(scf_input_path).expanduser().resolve()

    with open(scf_input_path) as file_handle:
        config = copy.deepcopy(yaml.safe_load(file_handle))

    # Override calculation type and k-points.
    control = config.setdefault("control", {})
    control["calculation"] = "nscf"
    control["kpoints_list"] = [list(kpt) for kpt in kpoints_list]

    if prefix_override is not None:
        control["prefix"] = prefix_override

    if compute_projections:
        control["compute_projections"] = True

    if uniform_grid:
        control.pop("band_path", None)
        control.pop("band_path_points_per_segment", None)

    if number_of_bands is not None:
        config.setdefault("method", {}).setdefault("scf", {})["number_of_bands"] = number_of_bands

    # Remove kgrid from the NSCF input to make it clear it's unused
    # (the C++ parser ignores it when kpoints_list is present, but removing
    # it avoids confusion when reading the generated file).
    method_scf = config.get("method", {}).get("scf", {})
    method_scf.pop("kgrid", None)

    with open(output_path, "w") as file_handle:
        yaml.dump(config, file_handle, default_flow_style=False, sort_keys=False)

    return output_path


# ---------------------------------------------------------------------------
# write_supercell_scf_input
# ---------------------------------------------------------------------------


# TODO: the k-grid rescaling is mat3ra-wode's job -- its PointsGridDataProvider keeps a
#   grid by KPPRA across cells -- but wode's calculate_dimensions is not implemented in
#   Python yet (2026.10.8).  Call it here once it is; the Q3 YAML handling stays.
def write_supercell_scf_input(
    output_path: str | Path,
    *,
    scf_input_path: str | Path,
    supercell_matrix,
) -> Path:
    """Write the SCF input for a supercell of the cell *scf_input_path* describes.

    Copies every setting and changes the two that do not carry over to a
    bigger cell:

    * ``method.scf.number_of_bands`` is dropped, so the engine counts bands for
      the supercell's own electrons (occupied + 4, or Quantum ESPRESSO's rule
      when smearing).  A unit-cell count inherited by a 16-atom supercell would
      leave most of its valence states out of an SCF that still converges.
    * The k-grid of a diagonal supercell is divided by the diagonal, rounding
      up and never below 1: the supercell's zone is that much smaller, and where
      the division is exact the two meshes are the same points -- silicon's
      2x2x2 shifted mesh on the 2-atom cell is a 1x1x1 shifted mesh on the
      2x2x2 supercell.  A non-diagonal supercell keeps the template's grid,
      which oversamples (slower, not wrong).

    File references are made absolute, since the copy is written elsewhere.
    The material stays the template's: the caller supplies each supercell.

    Parameters
    ----------
    output_path:
        Where to write the supercell YAML file.
    scf_input_path:
        Path to the unit-cell SCF input (used as template, not modified).
    supercell_matrix:
        3x3 integer matrix taking the unit cell to the supercell.

    Returns
    -------
    Path
        Resolved path to the written supercell input file.
    """
    output_path = Path(output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    scf_input_path = Path(scf_input_path).expanduser().resolve()

    with open(scf_input_path) as file_handle:
        config = yaml.safe_load(file_handle)
    make_file_references_absolute(config, scf_input_path.parent)

    scf_section = config.setdefault("method", {}).setdefault("scf", {})
    scf_section.pop("number_of_bands", None)

    supercell_matrix = np.asarray(supercell_matrix, dtype=int)
    diagonal = np.diagonal(supercell_matrix)
    kgrid = scf_section.get("kgrid", {})
    if "dimensions" in kgrid and np.array_equal(supercell_matrix, np.diag(diagonal)):
        kgrid["dimensions"] = [
            max(1, math.ceil(int(dimension) / int(divisor)))
            for dimension, divisor in zip(kgrid["dimensions"], diagonal, strict=True)
        ]

    with open(output_path, "w") as file_handle:
        yaml.dump(config, file_handle, default_flow_style=False, sort_keys=False)

    return output_path
