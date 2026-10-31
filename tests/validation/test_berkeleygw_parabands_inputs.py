"""BerkeleyGW's VXC, VSC and VKB files from a Q3 NSCF run.

VSC (the local Kohn-Sham potential) and VKB (the nonlocal pseudopotential at
every k-point) are what BerkeleyGW's ParaBands needs to extend a WFN with more
bands; VXC is what sigma reads under `dont_use_vxcdat`. Three levels, each
needing more of the reference software than the last:

  1. Q3 alone: the files are written, read back with the independent reader in
     helpers_berkeleygw.py, and agree with the WFN and RHO they go with --
     ParaBands stops on any difference in k-points, weights, plane-wave counts
     or G-vectors.
  2. Against pw2bgw.x (requires_qe): VXC and VSC field by field, and the
     nonlocal potential V_NL(G, G') = Σ β D β† built from each VKB. V_NL, not
     the projectors themselves, because the phase, sign and order of each
     projector channel are a code's own choice and V_NL depends on none of them.
  3. Through ParaBands (requires_parabands): the Hamiltonian it builds from
     Q3's VSC and VKB has the eigenvalues pw.x finds for the same crystal --
     the end-to-end statement that the files describe the right H.

The dataset carries a nonlinear core correction on purpose: pw2bgw evaluates
its VXC from the valence density alone (`vxc_zero_rho_core`, default .TRUE.),
which with NLCC differs from the Vxc the Hamiltonian applies by about 0.02 Ry.

Measured (silicon, 25 Ry, 2x2x2 grid listed point by point): VXC and VSC 7e-7
Ry from pw2bgw's, V_NL 5e-16 Ry, ParaBands eigenvalues 0.004 meV from pw.x's
over all eight bands.
"""

import os
import subprocess

import numpy as np
import pytest
import yaml

from helpers import (
    BERKELEYGW_PARABANDS,
    DATA_DIRECTORY,
    QE_PW,
    QE_PW2BGW,
    requires_parabands,
    requires_pw2bgw,
    requires_qe,
    run_q3,
)
from helpers_berkeleygw import (
    nonlocal_potential_matrix,
    read_berkeleygw_charge_density_file,
    read_berkeleygw_kleinman_bylander_file,
    read_berkeleygw_potential_file,
    read_berkeleygw_wavefunction_file,
)

PSEUDOPOTENTIAL_DIRECTORY = DATA_DIRECTORY / "pseudopotentials" / "nc"
SILICON_MATERIAL = DATA_DIRECTORY / "materials" / "silicon.json"
# silicon.json's FCC primitive cell, a = 3.866976 A, as QE's ibrav = 2 conventional
# lattice parameter in bohr.
CONVENTIONAL_LATTICE_PARAMETER_BOHR = 3.866976 * 2**0.5 / 0.529177210903
WAVEFUNCTION_CUTOFF_RY = 25.0
NUMBER_OF_BANDS = 8
# The full 2x2x2 grid, every point with the same weight: BerkeleyGW's kgrid is then
# recovered from the list itself.
KPOINTS_CRYSTAL = [[x, y, z] for x in (0.0, 0.5) for y in (0.0, 0.5) for z in (0.0, 0.5)]

RY_TO_EV = 13.605693122994
POTENTIAL_TOLERANCE_RY = 1.0e-5
NONLOCAL_POTENTIAL_TOLERANCE_RY = 1.0e-12
EIGENVALUE_TOLERANCE_EV = 2.0e-5
# Single-threaded, so the reference codes do not take every core of the machine.
SINGLE_THREAD_ENVIRONMENT = {**os.environ, "OMP_NUM_THREADS": "1"}


def _q3_manifest(control, kgrid=None):
    scf = {"convergence_threshold": 1.0e-12, "number_of_bands": NUMBER_OF_BANDS}
    if kgrid is not None:
        scf["kgrid"] = {"dimensions": kgrid, "shifts": [0, 0, 0]}
    return {
        "material": str(SILICON_MATERIAL),
        "model": {"type": "dft", "subtype": "gga", "functional": "pbe"},
        "method": {
            "type": "pseudopotential",
            "subtype": "norm-conserving",
            "cutoffs": {
                "wavefunction": WAVEFUNCTION_CUTOFF_RY,
                "density": 4 * WAVEFUNCTION_CUTOFF_RY,
            },
            "pseudopotentials": {"Si": str(PSEUDOPOTENTIAL_DIRECTORY / "Si.upf")},
            "scf": scf,
        },
        "control": {"outdir": "./output", "prefix": "si", **control},
    }


def _run_q3_manifest(q3_command, manifest, path):
    path.write_text(yaml.safe_dump(manifest, sort_keys=False))
    result = run_q3(q3_command, path, working_directory=str(path.parent))
    assert result.returncode == 0, (
        f"q3 {path.name} exited {result.returncode}:\n{result.stdout[-3000:]}\n{result.stderr[-2000:]}"
    )


@pytest.fixture(scope="module")
def q3_export_directory(q3_command, tmp_path_factory):
    """A Q3 SCF, then the NSCF that exports WFN, RHO, VXC, VSC and VKB."""
    directory = tmp_path_factory.mktemp("q3_parabands_inputs")
    _run_q3_manifest(
        q3_command, _q3_manifest({"calculation": "scf"}, kgrid=[4, 4, 4]), directory / "scf.yml"
    )
    nscf_control = {
        "calculation": "nscf",
        "mean_field_export": "berkeleygw",
        "mean_field_export_kleinman_bylander_projectors": True,
        "kpoints_list": [[*point, 1.0] for point in KPOINTS_CRYSTAL],
    }
    _run_q3_manifest(q3_command, _q3_manifest(nscf_control), directory / "nscf.yml")
    return directory / "output"


def _qe_deck(calculation, kpoints_card):
    return f"""&control
  calculation = '{calculation}', prefix = 'si', outdir = './output'
  pseudo_dir = '{PSEUDOPOTENTIAL_DIRECTORY}'
/
&system
  ibrav = 2, celldm(1) = {CONVENTIONAL_LATTICE_PARAMETER_BOHR:.10f}, nat = 2, ntyp = 1
  ecutwfc = {WAVEFUNCTION_CUTOFF_RY}, ecutrho = {4 * WAVEFUNCTION_CUTOFF_RY}, nbnd = {NUMBER_OF_BANDS}
/
&electrons
  conv_thr = 1.0d-12, diago_full_acc = .true.
/
ATOMIC_SPECIES
  Si 28.0855 Si.upf
ATOMIC_POSITIONS crystal
  Si 0.875 0.875 0.875
  Si 0.125 0.125 0.125
{kpoints_card}"""


def _run_reference(command, directory, label):
    result = subprocess.run(
        command,
        cwd=directory,
        capture_output=True,
        text=True,
        timeout=600,
        env=SINGLE_THREAD_ENVIRONMENT,
    )
    assert result.returncode == 0, f"{label} exited {result.returncode}:\n{result.stdout[-3000:]}"
    return result


@pytest.fixture(scope="module")
def pw2bgw_directory(tmp_path_factory):
    """pw.x SCF and bands on the same crystal and k-points, then pw2bgw.x."""
    directory = tmp_path_factory.mktemp("pw2bgw_parabands_inputs")
    (directory / "scf.in").write_text(_qe_deck("scf", "K_POINTS automatic\n 4 4 4 0 0 0\n"))
    kpoints_card = "K_POINTS crystal\n8\n" + "".join(
        f" {x} {y} {z} 1.0\n" for x, y, z in KPOINTS_CRYSTAL
    )
    (directory / "bands.in").write_text(_qe_deck("bands", kpoints_card))
    (directory / "pp.in").write_text(
        "&input_pw2bgw\n"
        "  prefix = 'si', outdir = './output', real_or_complex = 2\n"
        "  wfng_flag = .true., wfng_file = 'WFN', wfng_kgrid = .true.\n"
        "  wfng_nk1 = 2, wfng_nk2 = 2, wfng_nk3 = 2\n"
        "  vxcg_flag = .true., vxcg_file = 'VXC'\n"
        "  vscg_flag = .true., vscg_file = 'VSC'\n"
        "  vkbg_flag = .true., vkbg_file = 'VKB'\n"
        "/\n"
    )
    _run_reference([QE_PW, "-in", "scf.in"], directory, "pw.x scf")
    _run_reference([QE_PW, "-in", "bands.in"], directory, "pw.x bands")
    _run_reference([QE_PW2BGW, "-in", "pp.in"], directory, "pw2bgw.x")
    return directory / "output"


# ── 1. Q3 alone ───────────────────────────────────────────────────────


def test_parabands_inputs_agree_with_their_wfn(q3_export_directory):
    wavefunction = read_berkeleygw_wavefunction_file(q3_export_directory / "WFN")
    charge_density = read_berkeleygw_charge_density_file(q3_export_directory / "RHO")
    projectors = read_berkeleygw_kleinman_bylander_file(q3_export_directory / "VKB")

    for name in ("VXC", "VSC"):
        potential = read_berkeleygw_potential_file(q3_export_directory / name)
        np.testing.assert_array_equal(
            potential["charge_density_gvectors"], charge_density["charge_density_gvectors"]
        )
        assert np.max(np.abs(potential["potential_g"][0].imag)) < 1.0e-12, (
            f"{name}(G = 0) is the cell average of a real potential"
        )

    # Everything ParaBands compares between WFN and VKB before it starts.
    np.testing.assert_array_equal(
        projectors["number_of_gvectors_per_kpoint"], wavefunction["number_of_gvectors_per_kpoint"]
    )
    np.testing.assert_array_equal(projectors["kpoint_weights"], wavefunction["kpoint_weights"])
    np.testing.assert_array_equal(
        projectors["kpoints_crystal_coordinates"], wavefunction["kpoints_crystal_coordinates"]
    )
    np.testing.assert_array_equal(
        projectors["kpoint_grid_dimensions"], wavefunction["kpoint_grid_dimensions"]
    )
    np.testing.assert_array_equal(
        projectors["kpoint_grid_shifts"], wavefunction["kpoint_grid_shifts"]
    )
    np.testing.assert_array_equal(projectors["kpoint_grid_dimensions"], [2, 2, 2])
    for kpoint_index in range(wavefunction["number_of_kpoints"]):
        np.testing.assert_array_equal(
            projectors["kpoint_gvectors"][kpoint_index],
            wavefunction["kpoint_gvectors"][kpoint_index],
        )
        matrix = nonlocal_potential_matrix(projectors, kpoint_index)
        assert np.max(np.abs(matrix - matrix.conj().T)) < 1.0e-14, "V_NL must be Hermitian"

    # Silicon: one species, two atoms; the dataset's two projectors for each of
    # l = 0, 1, 2, expanded over m: 2 x (1 + 3 + 5) = 18 channels per atom.
    assert projectors["number_of_species"] == 1
    assert list(projectors["projectors_per_species"]) == [18]
    assert projectors["number_of_projectors"] == 36


# ── 2. Against pw2bgw.x ───────────────────────────────────────────────


def _potential_by_miller_index(path):
    potential = read_berkeleygw_potential_file(path)
    return {
        tuple(miller): value
        for miller, value in zip(
            potential["charge_density_gvectors"].T, potential["potential_g"][:, 0]
        )
    }


@requires_qe
@requires_pw2bgw
@pytest.mark.parametrize("name", ["VXC", "VSC"])
def test_potential_matches_pw2bgw(q3_export_directory, pw2bgw_directory, name):
    q3_potential = _potential_by_miller_index(q3_export_directory / name)
    reference_potential = _potential_by_miller_index(pw2bgw_directory / name)
    assert set(q3_potential) == set(reference_potential)
    largest_difference = max(abs(q3_potential[g] - reference_potential[g]) for g in q3_potential)
    assert largest_difference < POTENTIAL_TOLERANCE_RY, (
        f"{name}: largest |Q3 - pw2bgw| over the G-set is {largest_difference:.2e} Ry"
    )


@requires_qe
@requires_pw2bgw
def test_nonlocal_potential_matches_pw2bgw(q3_export_directory, pw2bgw_directory):
    q3_projectors = read_berkeleygw_kleinman_bylander_file(q3_export_directory / "VKB")
    reference_projectors = read_berkeleygw_kleinman_bylander_file(pw2bgw_directory / "VKB")
    assert q3_projectors["number_of_projectors"] == reference_projectors["number_of_projectors"]
    for kpoint_index in range(q3_projectors["number_of_kpoints"]):
        q3_gvectors = [tuple(g) for g in q3_projectors["kpoint_gvectors"][kpoint_index].T]
        reference_index = {
            tuple(g): index
            for index, g in enumerate(reference_projectors["kpoint_gvectors"][kpoint_index].T)
        }
        order = [reference_index[g] for g in q3_gvectors]
        reference_matrix = nonlocal_potential_matrix(reference_projectors, kpoint_index)[
            np.ix_(order, order)
        ]
        q3_matrix = nonlocal_potential_matrix(q3_projectors, kpoint_index)
        largest_difference = np.max(np.abs(q3_matrix - reference_matrix))
        assert largest_difference < NONLOCAL_POTENTIAL_TOLERANCE_RY, (
            f"k-point {kpoint_index + 1}: largest |V_NL Q3 - pw2bgw| is {largest_difference:.2e} Ry"
        )


# ── 3. Through ParaBands ──────────────────────────────────────────────


@requires_qe
@requires_pw2bgw
@requires_parabands
def test_parabands_hamiltonian_reproduces_pw_eigenvalues(
    q3_export_directory, pw2bgw_directory, tmp_path
):
    h5py = pytest.importorskip("h5py")
    for name in ("WFN", "VSC", "VKB"):
        (tmp_path / name).symlink_to(q3_export_directory / name)
    (tmp_path / "parabands.inp").write_text(
        "input_wfn_file WFN\noutput_wfn_file WFN_parabands.h5\n"
        "vsc_file VSC\nvkb_file VKB\nnumber_bands 40\n"
    )
    _run_reference([str(BERKELEYGW_PARABANDS)], tmp_path, "parabands.cplx.x")

    with h5py.File(tmp_path / "WFN_parabands.h5", "r") as handle:
        parabands_energies_ry = handle["mf_header/kpoints/el"][0, :, :NUMBER_OF_BANDS]
    reference_energies_ry = read_berkeleygw_wavefunction_file(pw2bgw_directory / "WFN")[
        "band_energies_rydberg"
    ].T
    largest_difference_ev = np.max(np.abs(parabands_energies_ry - reference_energies_ry)) * RY_TO_EV
    assert largest_difference_ev < EIGENVALUE_TOLERANCE_EV, (
        f"ParaBands on Q3's VSC + VKB is {largest_difference_ev * 1000:.4f} meV from pw.x"
    )
