"""
Validation tests for the Gamma-point phonon workflow in Q3
(q3.workflows.phonons.run_gamma_phonons_workflow).

While building this workflow, Si's Gamma-optical phonon (which must be
exactly triply degenerate -- diamond structure has Oh symmetry at Gamma,
which is why bulk Si shows one sharp Raman line, not three) came out split
into three different frequencies, with individual atomic forces showing
components that must be exactly zero by symmetry. This was root-caused to
TWO compounding issues, both now fixed:

1. symmetrize_vector() (layer_1_structure/symmetry/symmetry.hpp) applied
   the crystal rotation matrix un-transposed to a covariant quantity
   (forces), instead of its transpose -- invisible for the full
   48-operation cubic point group (every previously-validated system in
   this codebase), but not for the 8-operation subgroup a single-atom
   displacement reduces to. FIXED in
   feature/2026-07-26-0-fix-reduced-symmetry-forces; see
   tests/unit/layer_1_structure/symmetry/test_symmetry.cpp for a direct,
   minimal regression test, and TODOs.md's resolved entry for the full
   writeup. This fix alone eliminates the symmetry-forbidden force
   components entirely (verified both by that unit test and by an
   improved, correctly-signed QE comparison for individual displacements).

2. SymmetryAnalyzer::reduce_kpoints_internal() (the default automatic
   Monkhorst-Pack + IBZ reduction path) built k-point equivalence classes
   from compute_bravais_lattice_rotations() -- the Bravais LATTICE's
   holohedral point group, which only looks at lattice vectors and ignores
   the atomic basis entirely. For any crystal whose basis reduces the true
   symmetry below the lattice's (every finite-difference-displaced
   structure this workflow generates), that group is too large: it treats
   k-points as equivalent under rotations that are not actual symmetries of
   the decorated crystal, corrupting the k-point sum for both forces and
   stress. The chosen representative's contribution was otherwise scaled
   by the (inflated) star weight, and the reconstruction also used the
   real-space rotation convention instead of the reciprocal one -- a
   crystallographic rotation matrix in a generally oblique fractional
   basis (e.g. FCC's rhombohedral primitive cell) is not its own
   inverse-transpose. FIXED in feature/2026-07-27-0-ibz-vector-reconstruction
   by reusing get_symmetry_operations() (the same basis-aware, spglib-backed
   symmetry detection already used for density symmetrization) and its
   reciprocal_rotation field. Verified two ways: (a) reduce_kpoints_internal
   now agrees with reduce_kpoints (the independent spglib
   spg_get_ir_reciprocal_mesh path) and with a fully unreduced explicit
   k-point list to displayed precision, on both force and stress, for the
   exact displaced-Si case documented in TODOs.md; (b) the previously-xfail
   tests below now pass outright.
"""

import subprocess

import numpy as np
import pytest
from ase import units as ase_units
from q3.structure.ase_atoms import material_json_to_ase_atoms
from q3.workflows.phonons import run_gamma_phonons_workflow

from helpers import (
    EXAMPLES_DIRECTORY,
    QE_PW,
    extract_quantum_espresso_forces,
    generate_qe_input,
    requires_qe,
)

RY_BOHR_TO_EV_ANGSTROM = 25.71104

SI_EXAMPLE_DIR = EXAMPLES_DIRECTORY / "total_energy--semiconductor--silicon"


@pytest.fixture(scope="module")
def silicon_gamma_phonon_result(q3_command, tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp("gamma_phonons")
    return run_gamma_phonons_workflow(
        SI_EXAMPLE_DIR / "nc.yml",
        displacement_angstrom=0.01,
        q3_binary=q3_command[0],
        workdir=tmp_path / "phonon_workdir",
    )


# ── 1. Acoustic sum rule (regression; expected to pass) ───────────────────────


def test_silicon_acoustic_modes_vanish_after_asr(silicon_gamma_phonon_result):
    """The three lowest (acoustic) Gamma modes must be ~0 after the acoustic
    sum rule is applied -- this is a mathematical consequence of enforcing
    translational invariance on the force-constant matrix, so it holds
    regardless of whether the underlying per-displacement forces are
    otherwise physically correct (see module docstring).
    """
    frequencies_thz = np.array(silicon_gamma_phonon_result["frequencies_thz"])
    acoustic_modes = np.sort(np.abs(frequencies_thz))[:3]
    assert np.all(acoustic_modes < 0.3), (
        f"Acoustic modes should be < 0.3 THz after ASR, got {acoustic_modes} THz"
    )


# ── 2. Optical mode degeneracy ─────────────────────────────────────────────────


def test_silicon_optical_modes_triply_degenerate(silicon_gamma_phonon_result):
    frequencies_thz = np.array(silicon_gamma_phonon_result["frequencies_thz"])
    optical_modes = np.sort(frequencies_thz)[3:]
    spread = optical_modes.max() - optical_modes.min()
    assert spread < 0.1, (
        f"Gamma-optical modes should be triply degenerate (<0.1 THz spread), "
        f"got {optical_modes} THz (spread {spread:.4f} THz)"
    )


# ── 3. QE cross-check (tier-2) ─────────────────────────────────────────────────


def _qe_gamma_phonon_frequencies_thz(
    material_name,
    ecutwfc,
    ecutrho,
    kgrid,
    kshift,
    pp_filename,
    displacement_angstrom,
    output_directory,
):
    """Independently compute Gamma-point phonon frequencies via QE, by
    scripting the identical +-displacement finite-difference calculation
    QE's own way (forces only via pw.x, no ph.x/DFPT), and assembling +
    diagonalizing the dynamical matrix here.

    Mirrors q3.workflows.phonons.run_gamma_phonons_workflow's algorithm
    exactly (central differences -> ASR-corrected force constants ->
    mass-weighted dynamical matrix -> diagonalize), so that a discrepancy
    between this function's result and Q3's reflects a difference in the
    underlying per-displacement FORCES, not a difference in the phonon math.
    """
    material_path = EXAMPLES_DIRECTORY / "data" / "materials" / f"{material_name}.json"
    atoms = material_json_to_ase_atoms(material_path)
    cell = np.array(atoms.get_cell())
    equilibrium_fractional = np.array(atoms.get_scaled_positions())
    number_of_atoms = len(atoms)

    def displaced_fractional(atom_index, axis, sign):
        cartesian = equilibrium_fractional[atom_index] @ cell
        cartesian = cartesian.copy()
        cartesian[axis] += sign * displacement_angstrom
        new_fractional = cartesian @ np.linalg.inv(cell)
        return tuple(new_fractional.tolist())

    def forces_for(coordinate_overrides, case_name):
        qe_input = generate_qe_input(
            case_name=case_name,
            material_name=material_name,
            pp_type="nc",
            pp_filename=pp_filename,
            ecutwfc=ecutwfc,
            ecutrho=ecutrho,
            kgrid=kgrid,
            kshift=kshift,
            output_directory=str(output_directory),
            coordinate_overrides=coordinate_overrides,
        )
        # run_qe_full doesn't capture forces; re-run pw.x directly here since
        # tprnfor isn't part of generate_qe_input's default namelist set.
        with open(qe_input) as file_handle:
            content = file_handle.read()
        content = content.replace("&CONTROL", "&CONTROL\n  tprnfor = .true.", 1)
        with open(qe_input, "w") as file_handle:
            file_handle.write(content)

        result = subprocess.run(
            [QE_PW, "-in", qe_input], capture_output=True, text=True, timeout=300
        )
        return np.array(extract_quantum_espresso_forces(result.stdout))

    phi = np.zeros((3 * number_of_atoms, 3 * number_of_atoms))
    for atom_index in range(number_of_atoms):
        for axis in range(3):
            overrides_plus = {atom_index: displaced_fractional(atom_index, axis, +1)}
            overrides_minus = {atom_index: displaced_fractional(atom_index, axis, -1)}
            forces_plus = forces_for(overrides_plus, f"qe_phonon_{atom_index}{axis}p")
            forces_minus = forces_for(overrides_minus, f"qe_phonon_{atom_index}{axis}m")
            force_constant_row = -(forces_plus - forces_minus) / (2.0 * displacement_angstrom)
            phi[3 * atom_index + axis, :] = force_constant_row.flatten() * RY_BOHR_TO_EV_ANGSTROM

    phi = (phi + phi.T) / 2.0
    for atom_index in range(number_of_atoms):
        block_sum = np.zeros((3, 3))
        for other_index in range(number_of_atoms):
            if other_index != atom_index:
                block_sum += phi[
                    3 * atom_index : 3 * atom_index + 3, 3 * other_index : 3 * other_index + 3
                ]
        phi[3 * atom_index : 3 * atom_index + 3, 3 * atom_index : 3 * atom_index + 3] = -block_sum

    masses = np.array(atoms.get_masses())
    dynamical_matrix = np.zeros_like(phi)
    for atom_index in range(number_of_atoms):
        for other_index in range(number_of_atoms):
            block = phi[3 * atom_index : 3 * atom_index + 3, 3 * other_index : 3 * other_index + 3]
            dynamical_matrix[
                3 * atom_index : 3 * atom_index + 3, 3 * other_index : 3 * other_index + 3
            ] = block / np.sqrt(masses[atom_index] * masses[other_index])

    eigenvalues = np.linalg.eigvalsh(dynamical_matrix)
    omega_squared_si = eigenvalues * ase_units._e / (ase_units._amu * (1e-10) ** 2)
    omega_si = np.sign(omega_squared_si) * np.sqrt(np.abs(omega_squared_si))
    frequencies_thz = omega_si / (2.0 * np.pi) / 1e12
    return np.sort(frequencies_thz)


@requires_qe
def test_silicon_gamma_phonons_match_qe(silicon_gamma_phonon_result, tmp_path):
    q3_frequencies_thz = np.sort(np.array(silicon_gamma_phonon_result["frequencies_thz"]))
    qe_frequencies_thz = _qe_gamma_phonon_frequencies_thz(
        material_name="silicon",
        ecutwfc=48.0,
        ecutrho=192.0,
        kgrid=[2, 2, 2],
        kshift=[1, 1, 1],
        pp_filename="Si.upf",
        displacement_angstrom=0.01,
        output_directory=tmp_path,
    )
    max_difference_thz = np.abs(q3_frequencies_thz - qe_frequencies_thz).max()
    assert max_difference_thz < 0.2, (
        f"Q3 Gamma frequencies {q3_frequencies_thz} THz differ from QE's "
        f"{qe_frequencies_thz} THz by up to {max_difference_thz:.4f} THz (tolerance 0.2 THz)"
    )
