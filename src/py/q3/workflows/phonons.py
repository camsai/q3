"""q3.workflows.phonons — Finite-difference phonon workflows.

Two entry points, both evaluating forces with ``Q3ASECalculator`` and both
reporting the ESSE ``phonon_dispersions`` property:

``run_gamma_phonons_workflow``
    The Gamma-point (q=0) slice.  6N displacements of the ORIGINAL cell
    (2 per atom per Cartesian axis), driven through ASE's ``Phonons`` class,
    with no supercell and no symmetry reduction of the displacement set.

``run_phonopy_workflow``
    The supercell path.  Phonopy generates the symmetry-reduced displaced
    supercells, Q3 evaluates the forces on each, and Phonopy turns the force
    sets into force constants and answers at any q: a dispersion along a
    high-symmetry path, the density of states and the harmonic
    thermodynamics on a q-mesh.  For silicon in a 2x2x2 supercell that is TWO
    displaced SCF runs against the Gamma path's twelve, and it resolves the
    whole Brillouin zone rather than its centre.

The acoustic sum rule is imposed by the library that owns the force
constants -- ASE's ``Phonons.read(acoustic=True)`` on the Gamma path,
Phonopy's ``symmetrize_force_constants()`` on the supercell path -- and is not
re-derived here.  Phonopy is used strictly as a library, through its public
Python API.

Only the orchestration lives here.  The supercell input is
``q3.io.write_supercell_scf_input``, the band path ``q3.workflows.kpath``, and
the results' ESSE shapes -- including the representation chosen where ESSE has
no schema (thermal properties, force constants) -- are ``q3.properties.phonons``.

Example::

    from q3.workflows import run_gamma_phonons_workflow, run_phonopy_workflow

    gamma = run_gamma_phonons_workflow(
        "examples/total_energy--semiconductor--silicon/nc.yml",
        q3_binary="./build/q3",
    )
    print(gamma["frequencies_thz"])

    full = run_phonopy_workflow(
        "examples/total_energy--semiconductor--silicon/nc.yml",
        supercell_matrix=(2, 2, 2),
        band_path=["G", "X", "W", "K", "G", "L"],
        q3_binary="./build/q3",
    )
    print(full["phonon_dispersions"]["yDataSeries"][-1])  # the highest branch, THz
    print(full["zero_point_energy"])
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import yaml
from ase import Atoms
from ase.phonons import Phonons

from q3.calculators.ase_calculator import Q3ASECalculator
from q3.io.input_writer import write_supercell_scf_input
from q3.properties.phonons import (
    force_constants_file,
    harmonic_thermal_properties,
    phonon_density_of_states,
    phonon_dispersions,
    phonon_frequencies_ev_to_thz_and_cm1,
    zero_point_energy,
)
from q3.structure.ase_atoms import material_json_to_ase_atoms
from q3.workflows.kpath import band_path_from_input, interpolate_kpath

try:
    from phonopy import Phonopy
    from phonopy.structure.atoms import PhonopyAtoms
except ImportError:  # the optional `phonons` extra; run_phonopy_workflow says so
    Phonopy = PhonopyAtoms = None


def run_gamma_phonons_workflow(
    scf_input_path: str | Path,
    *,
    displacement_angstrom: float = 0.01,
    apply_acoustic_sum_rule: bool = True,
    mode: str = "auto",
    q3_binary: str | Path | None = None,
    workdir: str | Path | None = None,
    cache_name: str = "phonon_cache",
    capture_output: bool = True,
) -> dict[str, Any]:
    """Compute Gamma-point phonon frequencies and eigenvectors.

    Uses central-difference forces (``+-displacement_angstrom`` per atom per
    Cartesian axis) to build the real-space force-constant matrix, applies
    the acoustic sum rule (translational invariance: rigidly translating the
    whole crystal must cost zero energy), mass-weights it into the dynamical
    matrix, and diagonalizes at q=(0,0,0).

    The cell is fixed throughout (only atomic positions are perturbed), so
    unlike ``run_vc_relax_workflow`` this is safe to run in the faster
    ``mode="bindings"`` path (the default "auto" picks it automatically when
    ``q3_core`` is importable) -- ``Q3ASECalculator``'s bindings-mode cell
    guard never triggers here since the cell never changes.

    Parameters
    ----------
    scf_input_path:
        Path to a Q3 YAML input file for the SCF calculation.
    displacement_angstrom:
        Finite-difference displacement magnitude in Angstrom.
    apply_acoustic_sum_rule:
        When ``True`` (default), corrects each atom's self force-constant
        block to enforce exact translational invariance (the three acoustic
        modes at Gamma are otherwise only approximately zero, limited by
        finite-difference/SCF-convergence noise).
    mode, q3_binary, workdir, capture_output:
        Forwarded to ``Q3ASECalculator``. ``workdir`` also hosts the ASE
        ``Phonons`` displacement cache (see ``cache_name``); passing the
        same ``workdir``/``cache_name`` on a second call resumes from
        whichever per-displacement calculations already completed.
    cache_name:
        Subdirectory (under ``workdir``) for the ASE ``Phonons`` JSON cache
        of per-displacement forces.

    Returns
    -------
    dict
        Contains:
        - ``"frequencies_ev"``, ``"frequencies_thz"``, ``"frequencies_cm1"``
          — Gamma-point frequencies, ascending order, 3*N_atoms entries.
          A negative value denotes an imaginary (unstable) mode, matching
          ASE's convention.
        - ``"eigenvectors"`` — mass-weighted-normalized Cartesian
          displacement pattern per mode, shape ``(3*N_atoms, N_atoms, 3)``,
          units 1/sqrt(amu) (not renormalized to 1).
        - ``"phonon_dispersions"`` — the same frequencies as the ESSE
          ``phonon_dispersions`` property at its one q-point: ``xDataArray``
          is ``[[0, 0, 0]]`` and ``yDataSeries`` holds one series per branch,
          in cm^-1.
    """
    scf_input_path = Path(scf_input_path).expanduser().resolve()

    with open(scf_input_path) as file_handle:
        config = yaml.safe_load(file_handle)

    material_path = (scf_input_path.parent / config.get("material", "")).resolve()
    atoms = material_json_to_ase_atoms(material_path)

    calculator = Q3ASECalculator(
        scf_input_path,
        mode=mode,
        q3_binary=q3_binary,
        workdir=workdir,
        capture_output=capture_output,
    )
    atoms.calc = calculator

    # Match Q3ASECalculator's own default workdir exactly, so the phonon
    # cache sits alongside the actual per-displacement Q3 outputs rather
    # than in an unrelated sibling directory.
    effective_workdir = Path(workdir) if workdir is not None else (Path.cwd() / "q3_ase_workdir")
    cache_path = effective_workdir / cache_name
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    phonons = Phonons(
        atoms,
        calculator,
        supercell=(1, 1, 1),
        delta=displacement_angstrom,
        name=str(cache_path),
    )
    phonons.run()
    phonons.read(acoustic=apply_acoustic_sum_rule, symmetrize=3)

    frequencies_ev, eigenvectors = phonons.band_structure([[0.0, 0.0, 0.0]], modes=True)
    frequencies_ev = frequencies_ev[0]
    eigenvectors = eigenvectors[0]

    frequencies_thz, frequencies_cm1 = phonon_frequencies_ev_to_thz_and_cm1(frequencies_ev)

    return {
        "frequencies_ev": frequencies_ev.tolist(),
        "frequencies_thz": frequencies_thz.tolist(),
        "frequencies_cm1": frequencies_cm1.tolist(),
        "eigenvectors": eigenvectors.tolist(),
        "phonon_dispersions": phonon_dispersions(
            [[0.0, 0.0, 0.0]], [frequencies_cm1], units="cm-1"
        ),
    }


def run_phonopy_workflow(
    scf_input_path: str | Path,
    *,
    supercell_matrix: Any = (2, 2, 2),
    displacement_angstrom: float = 0.01,
    central_differences: bool = True,
    band_path: list[str] | None = None,
    lattice_type: str | None = None,
    points_per_segment: int | None = None,
    mesh: Any = (20, 20, 20),
    temperature_min_kelvin: float = 0.0,
    temperature_max_kelvin: float = 1000.0,
    temperature_step_kelvin: float = 10.0,
    apply_acoustic_sum_rule: bool = True,
    q3_binary: str | Path | None = None,
    workdir: str | Path | None = None,
    capture_output: bool = True,
) -> dict[str, Any]:
    """Compute phonon dispersions, the density of states and thermodynamics through Phonopy.

    Parameters
    ----------
    scf_input_path:
        A Q3 YAML input describing the UNIT cell.  The supercell input is
        derived from it by ``q3.io.write_supercell_scf_input``; the file
        itself is not modified.
    supercell_matrix:
        Three integers (a diagonal supercell) or a 3x3 integer matrix.  The
        force constants reach only as far as the supercell does, which makes
        this the convergence parameter of the whole method: too small a
        supercell shows up as a dispersion that is wrong away from Gamma, not
        as a failure.
    displacement_angstrom:
        Finite-difference displacement magnitude; Phonopy chooses directions.
    central_differences:
        When ``True`` (default) every displacement is run at ``+u`` and ``-u``,
        doubling the SCF count, and it is not optional in practice.  Phonopy's
        own default runs one sign wherever site symmetry makes the two
        equivalent, which holds for the exact energy surface but not for a
        k-sampled one: silicon on the examples' 2x2x2 shifted mesh gives
        -0.00887 and +0.00150 Ry/bohr for ``+u`` and ``-u``, an antisymmetric
        half that is the physics and a symmetric -0.00368 that is sampling
        error (it vanishes on a Gamma-centred mesh).  A forward difference
        keeps all of it and reports a Gamma-optical mode of 4.9 THz instead of
        14.4, silently.  Set ``False`` only with a mesh converged to earn it.
    band_path, lattice_type, points_per_segment:
        The high-symmetry path, resolved as ``run_band_structure_workflow``
        resolves it -- including the input file's own ``control.band_path``
        and the material's ``lattice.type``.
    mesh:
        q-point mesh for the density of states and the thermodynamics.  It
        interpolates force constants that already exist, so it costs no SCF.
    temperature_min_kelvin, temperature_max_kelvin, temperature_step_kelvin:
        Temperature grid for the thermal properties.
    apply_acoustic_sum_rule:
        Apply Phonopy's ``symmetrize_force_constants()`` (default ``True``).
    q3_binary, workdir, capture_output:
        Forwarded to ``Q3ASECalculator``.  *workdir* also receives the
        supercell input and ``phonopy_params.yaml``.

    Returns
    -------
    dict
        Built by ``q3.properties.phonons``.  ESSE properties where a schema
        exists: ``"phonon_dispersions"`` (q-points along the path in the
        material's own crystal coordinates, one series per branch, THz),
        ``"phonon_dos"`` (THz, states/THz per unit cell) and
        ``"zero_point_energy"`` (kJ/mol of unit cells, Phonopy's unit).
        ``"thermal_properties"`` holds ``"free_energy"`` (kJ/mol),
        ``"entropy"`` and ``"heat_capacity"`` (J/K/mol), per mole of unit
        cells: ESSE has no temperature-dependent property for any of them, so
        each is ESSE's abstract two-dimensional plot, temperature (K) in
        ``xDataArray`` and the values in ``yDataSeries[0]``.
        ``"force_constants"`` is the ESSE ``file_content`` property naming
        ``Phonopy.save()``'s ``phonopy_params.yaml`` (supercell,
        displacements, forces and force constants, for ``phonopy.load``):
        ESSE has no force-constants schema either.
    """
    if Phonopy is None:
        raise ImportError(
            "Phonopy is required for run_phonopy_workflow.  Install it with: pip install q3[phonons]"
        )

    scf_input_path = Path(scf_input_path).expanduser().resolve()
    with open(scf_input_path) as file_handle:
        config = yaml.safe_load(file_handle)
    band_path, lattice_type, points_per_segment, _ = band_path_from_input(
        scf_input_path, config, band_path, lattice_type, points_per_segment
    )
    unit_cell = material_json_to_ase_atoms((scf_input_path.parent / config["material"]).resolve())

    supercell_matrix = np.array(supercell_matrix, dtype=int)
    if supercell_matrix.shape == (3,):
        supercell_matrix = np.diag(supercell_matrix)

    # An identity primitive matrix, not Phonopy's 'auto': 'auto' re-expresses
    # the cell in spglib's standardized primitive basis, which for FCC is a
    # different choice of vectors from Q3's ibrav=2 one, and a q-point is
    # fractional coordinates OF a basis. The path comes from q3.workflows.kpath
    # in the material's own basis, as for run_band_structure_workflow.
    phonon = Phonopy(
        PhonopyAtoms(
            symbols=unit_cell.get_chemical_symbols(),
            cell=np.array(unit_cell.get_cell()),
            scaled_positions=unit_cell.get_scaled_positions(),
        ),
        supercell_matrix=supercell_matrix,
        primitive_matrix=np.eye(3),
    )
    phonon.generate_displacements(
        distance=displacement_angstrom,
        is_plusminus=True if central_differences else "auto",
    )

    workdir = Path(workdir or Path.cwd() / "q3_ase_workdir").expanduser().resolve()
    calculator = Q3ASECalculator(
        write_supercell_scf_input(
            workdir / f"{scf_input_path.stem}_supercell.yml",
            scf_input_path=scf_input_path,
            supercell_matrix=supercell_matrix,
        ),
        # A supercell is by construction not the cell the input names, which
        # the bindings path cannot solve (see Q3ASECalculator.calculate()).
        mode="subprocess",
        q3_binary=q3_binary,
        workdir=workdir,
        capture_output=capture_output,
    )
    force_sets = []
    for displaced in phonon.supercells_with_displacements:
        atoms = Atoms(
            symbols=displaced.symbols,
            cell=displaced.cell,
            scaled_positions=displaced.scaled_positions,
            pbc=True,
        )
        atoms.calc = calculator
        force_sets.append(atoms.get_forces())
    phonon.forces = force_sets
    phonon.produce_force_constants()
    if apply_acoustic_sum_rule:
        phonon.symmetrize_force_constants(show_drift=False)

    qpoints, _ = interpolate_kpath(band_path, lattice_type, points_per_segment=points_per_segment)
    phonon.run_band_structure([[list(qpoint) for qpoint in qpoints]])
    phonon.run_mesh(list(mesh), is_gamma_center=True)
    phonon.run_total_dos()
    phonon.run_thermal_properties(
        t_min=temperature_min_kelvin,
        t_max=temperature_max_kelvin,
        t_step=temperature_step_kelvin,
    )
    thermal_properties = phonon.thermal_properties

    return {
        "phonon_dispersions": phonon_dispersions(qpoints, phonon.band_structure.frequencies[0]),
        "phonon_dos": phonon_density_of_states(
            phonon.total_dos.frequency_points, phonon.total_dos.dos
        ),
        "zero_point_energy": zero_point_energy(thermal_properties.zero_point_energy),
        "thermal_properties": harmonic_thermal_properties(
            thermal_properties.temperatures,
            thermal_properties.free_energy,
            thermal_properties.entropy,
            thermal_properties.heat_capacity,
        ),
        "force_constants": force_constants_file(
            phonon.save(str(workdir / "phonopy_params.yaml"), settings={"force_constants": True})
        ),
    }
