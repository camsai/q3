"""
Validation tests: compare Q3 total energies against Quantum ESPRESSO (LibXC build).

These tests generate QE input files from Q3's material JSON files (ensuring
identical lattice parameters, pseudopotentials, k-grids, and cutoffs), run
both codes, and verify that total energies agree within a tight tolerance.

Run:
    .venv/bin/pytest tests/validation/test_qe_energy_comparison.py -v \
        --q3-bin ./build/q3
"""

import subprocess

import pytest

from helpers import (
    EXAMPLES_DIRECTORY,
    QE_PW,
    generate_qe_input,
    requires_qe,
    run_q3,
    run_qe,
    run_qe_full,
)

# ─── Constants ────────────────────────────────────────────────────────────────

# Total energy tolerance: 0.05 mRy (50 μRy) for NC, 0.1 mRy for US.
TOLERANCE_NC_MILLI_RYDBERG = 0.05
TOLERANCE_US_MILLI_RYDBERG = 0.1

# PAW: looser than NC/US, matching the tolerance the one-center engine's own
# A2.6 gate carries. Diamond agrees to 1.98 meV/atom, Si to 4.68, Cu to 9.22
# -- comfortably inside 1 mRy/atom, and the margin is deliberate: unlike NC/US
# convergence (a pure basis-set question), PAW compares two independently
# implemented one-center Hxc engines against each other, where the honest
# floor is a few tenths of a mRy per atom, not the few-μRy floor a shared-basis
# comparison reaches.
TOLERANCE_PAW_MILLI_RYDBERG_PER_ATOM = 1.0

# Roadmap Window 1 smearing-campaign gate (matched degauss): ΔE < 1e-6 Ry.
TOLERANCE_SMEARING_MILLI_RYDBERG = 1e-3


# ─── Helpers ──────────────────────────────────────────────────────────────────


def parse_q3_total_energy(stdout):
    """Parse Q3 stdout to find the converged total energy."""
    for line in stdout.split("\n"):
        if "total energy" in line and "=" in line and "Ry" in line:
            try:
                return float(line.split("=")[1].split("Ry")[0])
            except ValueError:
                continue
    raise RuntimeError("Q3 did not output total energy")


def parse_q3_total_magnetization(stdout):
    """Parse Q3 stdout for the converged total magnetization (Bohr mag/cell)."""
    for line in stdout.split("\n"):
        stripped = line.strip()
        if stripped.startswith("total magnetization"):
            return float(stripped.split("=")[1].split("Bohr")[0])
    raise RuntimeError("Q3 did not output total magnetization (nspin=1 run?)")


def parse_q3_dispersion_energy(stdout):
    """Parse Q3 stdout to find the "Dispersion (D3-BJ)" energy line."""
    for line in stdout.split("\n"):
        if "Dispersion (D3-BJ)" in line and "=" in line:
            return float(line.split("=")[1].strip())
    raise RuntimeError("Q3 did not output a Dispersion (D3-BJ) energy line")


def parse_qe_dispersion_energy(stdout):
    """Parse QE stdout to find the "DFT-D3 Dispersion" energy line (in Ry)."""
    for line in stdout.split("\n"):
        if "DFT-D3 Dispersion" in line and "=" in line:
            return float(line.split("=")[1].split("Ry")[0])
    raise RuntimeError("QE did not output a DFT-D3 Dispersion energy line")


# ─── Test Cases ───────────────────────────────────────────────────────────────


SILICON_CASES = [
    pytest.param(
        "si_nc_2x2x2",
        "total_energy--semiconductor--silicon",
        "nc.yml",
        "nc",
        "Si.upf",
        48.0,
        192.0,
        [2, 2, 2],
        [1, 1, 1],
        8,
        None,
        0.0,
        TOLERANCE_NC_MILLI_RYDBERG,
        None,
        id="Si-NC-2x2x2",
    ),
    pytest.param(
        "si_us_2x2x2",
        "total_energy--semiconductor--silicon",
        "us.yml",
        "us",
        "Si_PBE_USPP.UPF",
        40.0,
        200.0,
        [2, 2, 2],
        [1, 1, 1],
        None,
        None,
        0.0,
        TOLERANCE_US_MILLI_RYDBERG,
        None,
        id="Si-US-2x2x2",
    ),
]

COPPER_CASES = [
    pytest.param(
        "cu_nc_4x4x4",
        "total_energy--metal--copper",
        "nc.yml",
        "nc",
        "Cu.upf",
        120.0,
        480.0,
        [4, 4, 4],
        [1, 1, 1],
        20,
        "gauss",
        0.02,
        TOLERANCE_NC_MILLI_RYDBERG,
        None,
        id="Cu-NC-4x4x4",
    ),
    pytest.param(
        "cu_us_4x4x4",
        "total_energy--metal--copper",
        "us.yml",
        "us",
        "cu_pbe_v1.2.uspp.F.UPF",
        40.0,
        200.0,
        [4, 4, 4],
        [1, 1, 1],
        20,
        "gauss",
        0.02,
        TOLERANCE_US_MILLI_RYDBERG,
        None,
        id="Cu-US-4x4x4",
    ),
]

# Roadmap Window 1: "MP + FD + Marzari-Vanderbilt smearing" — one matched-degauss
# campaign per scheme, Cu FCC, ΔE < 1e-6 Ry.
SMEARING_CASES = [
    pytest.param(
        "cu_nc_mp",
        "total_energy--metal--copper",
        "nc_mp.yml",
        "nc",
        "Cu.upf",
        120.0,
        480.0,
        [4, 4, 4],
        [1, 1, 1],
        20,
        "mp",
        0.02,
        TOLERANCE_SMEARING_MILLI_RYDBERG,
        None,
        id="Cu-NC-MethfesselPaxton",
    ),
    pytest.param(
        "cu_nc_fd",
        "total_energy--metal--copper",
        "nc_fd.yml",
        "nc",
        "Cu.upf",
        120.0,
        480.0,
        [4, 4, 4],
        [1, 1, 1],
        20,
        "fd",
        0.02,
        TOLERANCE_SMEARING_MILLI_RYDBERG,
        None,
        id="Cu-NC-FermiDirac",
    ),
    pytest.param(
        "cu_nc_mv",
        "total_energy--metal--copper",
        "nc_mv.yml",
        "nc",
        "Cu.upf",
        120.0,
        480.0,
        [4, 4, 4],
        [1, 1, 1],
        20,
        "mv",
        0.02,
        TOLERANCE_SMEARING_MILLI_RYDBERG,
        None,
        id="Cu-NC-MarzariVanderbilt",
    ),
]

# Roadmap: DFT-D3(BJ) dispersion, "dispersion term Delta < 1e-8 Ry on identical
# geometry". vdw_corr='dft-d3', dftd3_version=4, dftd3_threebody=.false. in QE.
TOLERANCE_D3_MILLI_RYDBERG = 1e-5

DISPERSION_CASES = [
    pytest.param(
        "si_nc_d3",
        "total_energy--semiconductor--silicon",
        "nc_d3.yml",
        "nc",
        "Si.upf",
        48.0,
        192.0,
        [2, 2, 2],
        [1, 1, 1],
        8,
        None,
        0.0,
        TOLERANCE_D3_MILLI_RYDBERG,
        "d3bj",
        id="Si-NC-D3BJ",
    ),
    pytest.param(
        "cu_nc_d3",
        "total_energy--metal--copper",
        "nc_d3.yml",
        "nc",
        "Cu.upf",
        120.0,
        480.0,
        [4, 4, 4],
        [1, 1, 1],
        20,
        "gauss",
        0.02,
        TOLERANCE_D3_MILLI_RYDBERG,
        "d3bj",
        id="Cu-NC-D3BJ",
    ),
]


@requires_qe
@pytest.mark.parametrize(
    "case_name,example_dir,yml_file,pp_type,pp_filename,"
    "ecutwfc,ecutrho,kgrid,kshift,"
    "number_of_bands,smearing,degauss,"
    "tolerance_mry,dispersion",
    SILICON_CASES + COPPER_CASES + SMEARING_CASES,
)
def test_qe_energy_comparison(
    q3_command,
    case_name,
    example_dir,
    yml_file,
    pp_type,
    pp_filename,
    ecutwfc,
    ecutrho,
    kgrid,
    kshift,
    number_of_bands,
    smearing,
    degauss,
    tolerance_mry,
    dispersion,
    tmp_path,
):
    """Compare Q3 and QE total energies with identical parameters."""
    # Run Q3
    example_path = EXAMPLES_DIRECTORY / example_dir
    yml_path = example_path / yml_file
    q3_result = run_q3(q3_command, yml_path, working_directory=example_path)
    assert q3_result.returncode == 0, f"Q3 failed: {q3_result.stderr}\n{q3_result.stdout}"
    q3_energy = parse_q3_total_energy(q3_result.stdout)

    # Run QE with matching parameters
    qe_input = generate_qe_input(
        case_name=case_name,
        material_name=example_dir.split("--")[-1],
        pp_type=pp_type,
        pp_filename=pp_filename,
        ecutwfc=ecutwfc,
        ecutrho=ecutrho,
        kgrid=kgrid,
        kshift=kshift,
        number_of_bands=number_of_bands,
        smearing=smearing,
        degauss=degauss,
        dispersion=dispersion,
        output_directory=str(tmp_path),
    )
    qe_energy = run_qe(qe_input)

    # Compare
    delta_mry = abs(q3_energy - qe_energy) * 1000
    print(
        f"\n  Q3 = {q3_energy:.9f} Ry, QE = {qe_energy:.9f} Ry, "
        f"|Δ| = {delta_mry:.4f} mRy (tol = {tolerance_mry:.4f} mRy)"
    )
    assert delta_mry < tolerance_mry, (
        f"Q3 energy {q3_energy:.9f} Ry differs from QE {qe_energy:.9f} Ry "
        f"by {delta_mry:.4f} mRy (tolerance: {tolerance_mry} mRy)"
    )


# ─── PAW cases ──────────────────────────────────────────────────────────────
#
# A separate parametrization rather than folding into SILICON_CASES/
# COPPER_CASES above: PAW needs QE's diagonalization forced to 'cg' (every
# PAW dataset tried aborts the default Davidson solver in cegterg with
# "S matrix not positive definite" -- confirmed across 30/120 through 80/960
# Ry, so it is not a cutoff or convergence issue, it is the solver), and its
# tolerance is naturally per-atom (mixing a 1-atom Cu cell with a 2-atom Si
# cell into one absolute-mRy tolerance would be either too loose for Cu or
# too tight for Si).

PAW_CASES = [
    pytest.param(
        "si_paw",
        "total_energy--semiconductor--silicon",
        "paw.yml",
        "Si.upf",
        40.0,
        320.0,
        [2, 2, 2],
        [1, 1, 1],
        None,
        None,
        0.0,
        1.0e-8,
        2,
        id="Si-PAW",
    ),
    pytest.param(
        "cu_paw",
        "total_energy--metal--copper",
        "paw.yml",
        "Cu.upf",
        40.0,
        320.0,
        [4, 4, 4],
        [1, 1, 1],
        20,
        "gauss",
        0.02,
        1.0e-12,
        1,
        id="Cu-PAW",
    ),
    pytest.param(
        "c_paw",
        "total_energy--insulator--diamond",
        "paw.yml",
        "C.upf",
        60.0,
        480.0,
        [4, 4, 4],
        [0, 0, 0],
        None,
        None,
        0.0,
        1.0e-8,
        2,
        id="C-PAW",
    ),
]


@requires_qe
@pytest.mark.parametrize(
    "case_name,example_dir,yml_file,pp_filename,"
    "ecutwfc,ecutrho,kgrid,kshift,"
    "number_of_bands,smearing,degauss,"
    "convergence_threshold,number_of_atoms",
    PAW_CASES,
)
def test_qe_paw_energy_comparison(
    q3_command,
    case_name,
    example_dir,
    yml_file,
    pp_filename,
    ecutwfc,
    ecutrho,
    kgrid,
    kshift,
    number_of_bands,
    smearing,
    degauss,
    convergence_threshold,
    number_of_atoms,
    tmp_path,
):
    """Compare Q3 and QE total energies for a PAW dataset, per atom."""
    example_path = EXAMPLES_DIRECTORY / example_dir
    yml_path = example_path / yml_file
    q3_result = run_q3(q3_command, yml_path, working_directory=example_path)
    assert q3_result.returncode == 0, f"Q3 failed: {q3_result.stderr}\n{q3_result.stdout}"
    q3_energy = parse_q3_total_energy(q3_result.stdout)

    qe_input = generate_qe_input(
        case_name=case_name,
        material_name=example_dir.split("--")[-1],
        pp_type="paw",
        pp_filename=pp_filename,
        ecutwfc=ecutwfc,
        ecutrho=ecutrho,
        kgrid=kgrid,
        kshift=kshift,
        number_of_bands=number_of_bands,
        smearing=smearing,
        degauss=degauss,
        convergence_threshold=convergence_threshold,
        diagonalization="cg",
        output_directory=str(tmp_path),
    )
    qe_energy = run_qe(qe_input)

    delta_mry_per_atom = abs(q3_energy - qe_energy) * 1000 / number_of_atoms
    print(
        f"\n  Q3 = {q3_energy:.9f} Ry, QE = {qe_energy:.9f} Ry, "
        f"|Δ| = {delta_mry_per_atom:.4f} mRy/atom "
        f"(tol = {TOLERANCE_PAW_MILLI_RYDBERG_PER_ATOM:.4f} mRy/atom)"
    )
    assert delta_mry_per_atom < TOLERANCE_PAW_MILLI_RYDBERG_PER_ATOM, (
        f"Q3 energy {q3_energy:.9f} Ry differs from QE {qe_energy:.9f} Ry by "
        f"{delta_mry_per_atom:.4f} mRy/atom "
        f"(tolerance: {TOLERANCE_PAW_MILLI_RYDBERG_PER_ATOM} mRy/atom)"
    )


@requires_qe
@pytest.mark.parametrize(
    "case_name,example_dir,yml_file,pp_type,pp_filename,"
    "ecutwfc,ecutrho,kgrid,kshift,"
    "number_of_bands,smearing,degauss,"
    "tolerance_mry,dispersion",
    DISPERSION_CASES,
)
def test_qe_dispersion_energy_comparison(
    q3_command,
    case_name,
    example_dir,
    yml_file,
    pp_type,
    pp_filename,
    ecutwfc,
    ecutrho,
    kgrid,
    kshift,
    number_of_bands,
    smearing,
    degauss,
    tolerance_mry,
    dispersion,
    tmp_path,
):
    """Compare Q3 and QE isolated DFT-D3(BJ) dispersion energy terms.

    Roadmap gate: "dispersion term Delta < 1e-8 Ry on identical geometry".
    A full-total-energy comparison is not appropriate here: enabling
    vdw_corr='dft-d3' forces QE's plain-name 'pbe' input_dft (see
    generate_qe_input), which introduces small XC-evaluation baseline
    differences unrelated to dispersion correctness. Isolating the
    dispersion term itself avoids that contamination.
    """
    # Run Q3
    example_path = EXAMPLES_DIRECTORY / example_dir
    yml_path = example_path / yml_file
    q3_result = run_q3(q3_command, yml_path, working_directory=example_path)
    assert q3_result.returncode == 0, f"Q3 failed: {q3_result.stderr}\n{q3_result.stdout}"
    q3_dispersion_energy = parse_q3_dispersion_energy(q3_result.stdout)

    # Run QE with matching parameters
    qe_input = generate_qe_input(
        case_name=case_name,
        material_name=example_dir.split("--")[-1],
        pp_type=pp_type,
        pp_filename=pp_filename,
        ecutwfc=ecutwfc,
        ecutrho=ecutrho,
        kgrid=kgrid,
        kshift=kshift,
        number_of_bands=number_of_bands,
        smearing=smearing,
        degauss=degauss,
        dispersion=dispersion,
        output_directory=str(tmp_path),
    )
    qe_result = subprocess.run(
        [QE_PW, "-in", qe_input],
        capture_output=True,
        text=True,
        timeout=600,
    )
    qe_dispersion_energy = parse_qe_dispersion_energy(qe_result.stdout)

    # Compare the isolated dispersion terms
    delta_mry = abs(q3_dispersion_energy - qe_dispersion_energy) * 1000
    print(
        f"\n  Q3 dispersion = {q3_dispersion_energy:.9f} Ry, "
        f"QE dispersion = {qe_dispersion_energy:.9f} Ry, "
        f"|Δ| = {delta_mry:.6f} mRy (tol = {tolerance_mry:.6f} mRy)"
    )
    assert delta_mry < tolerance_mry, (
        f"Q3 dispersion energy {q3_dispersion_energy:.9f} Ry differs from QE "
        f"{qe_dispersion_energy:.9f} Ry by {delta_mry:.6f} mRy "
        f"(tolerance: {tolerance_mry} mRy)"
    )


# ─── LSDA (spin-polarized) cases: Window 1 Fe-FM / Ni-FM / Cr-AFM ────────────
#
# Roadmap-pinned protocol (Window 1):
# PseudoDojo NC v0.4 stringent, 12^3 k-grid, Gaussian degauss=0.02 Ry.
# Criteria: ΔE < 1 mRy (all three); moment Δ < 0.02 μB (Fe, Ni — Cr's AFM
# configuration has zero net moment by symmetry in both codes, so only ΔE
# is a meaningful gate for it).

TOLERANCE_LSDA_MILLI_RYDBERG = 1.0
TOLERANCE_MOMENT_MU_B = 0.02

TOLERANCE_CHARGED_CELL_MILLI_RYDBERG = 0.05

LSDA_CASES = [
    pytest.param(
        "fe_nc_lsda",
        "total_energy--metal--magnetic--iron",
        "nc.yml",
        "nc",
        "Fe.upf",
        90.0,
        360.0,
        [12, 12, 12],
        [1, 1, 1],
        {"Fe": 0.6},
        0.02,
        TOLERANCE_MOMENT_MU_B,
        id="Fe-BCC-FM-NC-12x12x12",
    ),
    pytest.param(
        "ni_nc_lsda",
        "total_energy--metal--magnetic--nickel",
        "nc.yml",
        "nc",
        "Ni.upf",
        60.0,
        240.0,
        [12, 12, 12],
        [1, 1, 1],
        {"Ni": 0.3},
        0.02,
        TOLERANCE_MOMENT_MU_B,
        id="Ni-FCC-FM-NC-12x12x12",
    ),
    pytest.param(
        "cr_afm_nc_lsda",
        "total_energy--metal--magnetic--chromium",
        "nc.yml",
        "nc",
        "Cr.upf",
        90.0,
        360.0,
        [12, 12, 12],
        [1, 1, 1],
        {"Cr1": 0.5, "Cr2": -0.5},
        0.02,
        None,  # AFM: net moment is trivially ~0 in both codes, not a useful gate
        id="Cr-BCC-AFM-NC-12x12x12",
    ),
]


@requires_qe
@pytest.mark.parametrize(
    "case_name,example_dir,yml_file,pp_type,pp_filename,"
    "ecutwfc,ecutrho,kgrid,kshift,"
    "starting_magnetization,degauss,moment_tolerance_mu_b",
    LSDA_CASES,
)
def test_qe_lsda_energy_and_moment_comparison(
    q3_command,
    case_name,
    example_dir,
    yml_file,
    pp_type,
    pp_filename,
    ecutwfc,
    ecutrho,
    kgrid,
    kshift,
    starting_magnetization,
    degauss,
    moment_tolerance_mu_b,
    tmp_path,
):
    """Compare Q3 and QE LSDA total energy (and moment, where meaningful).

    Window 1 roadmap validation: Fe-FM, Ni-FM, Cr-AFM, against the protocol
    pinned in the maintainers' roadmap (cutoffs and k-grids are recorded per
    dataset in examples/data/pseudopotentials/metadata.yml).
    """
    example_path = EXAMPLES_DIRECTORY / example_dir
    yml_path = example_path / yml_file
    q3_result = run_q3(q3_command, yml_path, working_directory=example_path)
    assert q3_result.returncode == 0, f"Q3 failed: {q3_result.stderr}\n{q3_result.stdout}"
    q3_energy = parse_q3_total_energy(q3_result.stdout)
    q3_moment = parse_q3_total_magnetization(q3_result.stdout)

    qe_input = generate_qe_input(
        case_name=case_name,
        material_name=example_dir.split("--")[-1],
        pp_type=pp_type,
        pp_filename=pp_filename,
        ecutwfc=ecutwfc,
        ecutrho=ecutrho,
        kgrid=kgrid,
        kshift=kshift,
        smearing="gaussian",
        degauss=degauss,
        number_of_spin_components=2,
        starting_magnetization=starting_magnetization,
        mixing_beta=0.1,
        mixing_mode="plain",
        output_directory=str(tmp_path),
    )
    qe_energy, qe_moment = run_qe_full(qe_input)

    delta_mry = abs(q3_energy - qe_energy) * 1000
    print(
        f"\n  Q3 = {q3_energy:.9f} Ry (moment {q3_moment:.4f} μB), "
        f"QE = {qe_energy:.9f} Ry (moment {qe_moment:.4f} μB), "
        f"|ΔE| = {delta_mry:.4f} mRy (tol = {TOLERANCE_LSDA_MILLI_RYDBERG:.4f} mRy)"
    )
    assert delta_mry < TOLERANCE_LSDA_MILLI_RYDBERG, (
        f"Q3 energy {q3_energy:.9f} Ry differs from QE {qe_energy:.9f} Ry "
        f"by {delta_mry:.4f} mRy (tolerance: {TOLERANCE_LSDA_MILLI_RYDBERG} mRy)"
    )

    if moment_tolerance_mu_b is not None:
        delta_moment = abs(q3_moment - qe_moment)
        assert delta_moment < moment_tolerance_mu_b, (
            f"Q3 total magnetization {q3_moment:.4f} μB differs from QE "
            f"{qe_moment:.4f} μB by {delta_moment:.4f} μB "
            f"(tolerance: {moment_tolerance_mu_b} μB)"
        )


@requires_qe
def test_qe_charged_cell_energy_comparison(q3_command, tmp_path):
    """Compare Q3 and QE total energy for a net-charged (non-neutral) cell.

    total_charge follows QE's tot_charge convention (positive = electrons
    removed). Si FCC (8 valence electrons neutral) with total_charge=1.0
    becomes a 7-electron cationic cell -- an odd electron count that cannot
    be represented by fixed occupations at a single k-point, hence the
    Gaussian smearing in nc_charged.yml (mirrored here for QE).

    This is intentionally an energy-only check: the Ewald ion-ion sum and
    the Hartree G=0 skip are both keyed only to the ionic charge (see
    ewald_summation.hpp), so total_charge support needs no compensating-
    background term -- only the electron count changes. No isolated-system
    (Makov-Payne) correction is applied on either side, so both energies
    include the same periodic-image self-interaction artifact and should
    still agree tightly.
    """
    example_path = EXAMPLES_DIRECTORY / "total_energy--semiconductor--silicon"
    yml_path = example_path / "nc_charged.yml"
    q3_result = run_q3(q3_command, yml_path, working_directory=example_path)
    assert q3_result.returncode == 0, f"Q3 failed: {q3_result.stderr}\n{q3_result.stdout}"
    q3_energy = parse_q3_total_energy(q3_result.stdout)

    qe_input = generate_qe_input(
        case_name="si_nc_charged",
        material_name="silicon",
        pp_type="nc",
        pp_filename="Si.upf",
        ecutwfc=48.0,
        ecutrho=192.0,
        kgrid=[2, 2, 2],
        kshift=[1, 1, 1],
        number_of_bands=8,
        smearing="gaussian",
        degauss=0.01,
        tot_charge=1.0,
        output_directory=str(tmp_path),
    )
    qe_energy = run_qe(qe_input)

    delta_mry = abs(q3_energy - qe_energy) * 1000
    print(
        f"\n  Q3 = {q3_energy:.9f} Ry, QE = {qe_energy:.9f} Ry, "
        f"|Δ| = {delta_mry:.4f} mRy (tol = {TOLERANCE_CHARGED_CELL_MILLI_RYDBERG:.4f} mRy)"
    )
    assert delta_mry < TOLERANCE_CHARGED_CELL_MILLI_RYDBERG, (
        f"Q3 charged-cell energy {q3_energy:.9f} Ry differs from QE {qe_energy:.9f} Ry "
        f"by {delta_mry:.4f} mRy (tolerance: {TOLERANCE_CHARGED_CELL_MILLI_RYDBERG} mRy)"
    )
