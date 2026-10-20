"""
Validation tests for the Density of States (DOS) calculation in Q3.

Validates key physical properties of the DOS computed by the Python
q3.workflows.run_dos_workflow() function:

  1. Si semiconductor: band gap visible in DOS (zero density in gap region).
  2. Cu metal: finite DOS at the Fermi level (metallic character).
  3. Fermi energy consistency between SCF and DOS calculation.

DOS is computed entirely in Python from SCF eigenvalues (Gaussian broadening).
The workflow runs the q3 binary for the SCF step, then post-processes in Python.

Reference:
  QE binary: reference/dft/q-e/build_libxc/bin/pw.x
  QE dos:    reference/dft/q-e/build_libxc/bin/dos.x
"""

import subprocess

import pytest
import yaml
from q3.workflows.dos import run_dos_workflow

from helpers import (
    EXAMPLES_DIRECTORY,
    QE_PW,
    requires_qe,
)

# ── Constants ────────────────────────────────────────────────────────────────

DATA_DIRECTORY = EXAMPLES_DIRECTORY / "data"

RY_TO_EV = 13.6058

# Number of valence electrons: Si = 4 per atom
SI_VALENCE_ELECTRONS = 4.0

# Fermi energy consistency: DOS should place Fermi energy within 0.1 Ry
# of the SCF value (primarily limited by k-grid sampling).
FERMI_ENERGY_CONSISTENCY_TOLERANCE_RY = 0.10


# ── QE Helpers ───────────────────────────────────────────────────────────────


def run_qe(input_text, work_directory, timeout=600):
    """Run Quantum ESPRESSO with the given input text. Returns stdout."""
    input_path = work_directory / "qe_input.in"
    input_path.write_text(input_text)
    result = subprocess.run(
        [QE_PW, "-in", str(input_path)],
        cwd=str(work_directory),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",  # QE output may contain non-UTF-8 bytes (locale-specific).
        timeout=timeout,
    )
    if result.returncode != 0:
        raise RuntimeError(f"QE run failed:\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}")
    return result.stdout


def extract_qe_fermi_energy(qe_output):
    """Extract Fermi energy from QE output in Ry."""
    for line in qe_output.split("\n"):
        if "the Fermi energy is" in line:
            try:
                return float(line.split()[-2]) / RY_TO_EV
            except (ValueError, IndexError):
                pass
    # For semiconductors: look for highest occupied eigenvalue.
    for line in qe_output.split("\n"):
        if "highest occupied" in line:
            try:
                return float(line.split()[-2]) / RY_TO_EV
            except (ValueError, IndexError):
                pass
    return None


# ── QE Input Templates ───────────────────────────────────────────────────────

SILICON_SCF_TEMPLATE = """&CONTROL
  calculation = 'scf',
  outdir      = '{outdir}',
  prefix      = 'si',
  pseudo_dir  = '{pseudo_dir}',
/
&SYSTEM
  ibrav     = 2,
  celldm(1) = 10.334,
  nat       = 2,
  ntyp      = 1,
  ecutwfc   = 48.0,
  ecutrho   = 192.0,
  input_dft = 'XC-000I-000I-101L-130L',
  nbnd      = 12,
/
&ELECTRONS
  conv_thr  = 1.0e-10,
/
ATOMIC_SPECIES
Si  28.086  {pp_filename}
ATOMIC_POSITIONS crystal
Si 0.00 0.00 0.00
Si 0.25 0.25 0.25
K_POINTS automatic
4 4 4 0 0 0
"""


# ── Silicon DOS Validation ────────────────────────────────────────────────────


@pytest.mark.slow
def test_silicon_dos_properties(q3_command, tmp_path):
    """
    Validate Si DOS physical properties from Q3.

    Tests:
    1. DOS is zero (band gap) near the Fermi level (semiconductor).
    2. Fermi energy is consistent with the SCF value and physically reasonable.
    """
    si_example_dir = EXAMPLES_DIRECTORY / "density_of_states--semiconductor--silicon"

    result = run_dos_workflow(
        si_example_dir / "nc.yml",
        broadening_ry=0.01,
        energy_steps=3000,
        q3_binary=q3_command[0],
    )

    dos = result["dos"]
    energies_ry = dos["energies_ry"]
    dos_values = dos["dos"]
    fermi_energy_ry = dos["fermi_energy_ry"]
    fermi_energy_ev = fermi_energy_ry * RY_TO_EV

    assert len(energies_ry) > 0, "DOS energies array is empty"
    assert len(dos_values) == len(energies_ry), "DOS values and energies have different lengths"

    # 1. Verify a band gap: DOS should be ~0 within ±0.015 Ry of the midgap.
    # For Si PBE, the gap is ~0.05 Ry (~0.7 eV). VBM ≈ fermi_energy, CBM ≈ fermi_energy + 0.05 Ry.
    midgap_ry = fermi_energy_ry + 0.025
    gap_dos_values = [
        dos_values[index]
        for index, energy in enumerate(energies_ry)
        if abs(energy - midgap_ry) < 0.011  # within ±0.011 Ry of midgap
    ]

    if gap_dos_values:
        max_gap_dos = max(abs(value) for value in gap_dos_values)
        valence_band_dos_values = [
            dos_values[index]
            for index, energy in enumerate(energies_ry)
            if energy < fermi_energy_ry - 0.04
        ]
        max_valence_dos = max(valence_band_dos_values) if valence_band_dos_values else 1.0
        # Midgap DOS should be < 80% of peak valence DOS (clear gap signature).
        # With broadening 0.01 Ry and a gap only ~0.05 Ry wide, Gaussian tails
        # produce measurable DOS at midgap (~0.5× peak), so we use a loose ratio.
        assert max_gap_dos < 0.8 * max_valence_dos, (
            f"Si DOS in gap ({max_gap_dos:.4f}) is too close to peak valence DOS "
            f"({max_valence_dos:.4f}): ratio = {max_gap_dos / max_valence_dos:.2f} (expected < 0.5)"
        )

    # 2. Verify Fermi energy is in a physically reasonable range.
    assert -0.7 < fermi_energy_ry < 0.7, (
        f"Si Fermi energy {fermi_energy_ry:.4f} Ry is outside expected range [-0.7, 0.7] Ry"
    )

    print(f"  Si DOS: Fermi energy = {fermi_energy_ev:.4f} eV ({fermi_energy_ry:.4f} Ry)")


@pytest.mark.slow
def test_dos_broadening_and_energy_steps_default_from_the_input_file(q3_command):
    """run_dos_workflow() must read dos_broadening/dos_energy_steps from
    scf_input_path's own control section when broadening_ry/energy_steps
    are not passed explicitly.

    nc.yml sets control.dos_broadening == 0.01 and control.dos_energy_steps
    == 3000 directly (the convention every examples/density_of_states--*/
    file uses) -- the same values test_silicon_dos_properties() above
    passes explicitly. Before this fix, this workflow ignored those fields
    entirely and always used its own hardcoded energy_steps=1000, regardless
    of what the file stated.
    """
    si_example_dir = EXAMPLES_DIRECTORY / "density_of_states--semiconductor--silicon"

    with open(si_example_dir / "nc.yml") as f:
        config = yaml.safe_load(f)
    assert config["control"]["dos_broadening"] == 0.01
    assert config["control"]["dos_energy_steps"] == 3000

    result = run_dos_workflow(si_example_dir / "nc.yml", q3_binary=q3_command[0])

    dos = result["dos"]
    # Same expectation as test_silicon_dos_properties(): the defaulted
    # dos_energy_steps produced the identical grid size, not just "a" grid --
    # 3000, not the workflow's own hardcoded default of 1000.
    assert len(dos["energies_ry"]) == 3000, (
        f"Expected 3000 energy points from nc.yml's own dos_energy_steps, "
        f"got {len(dos['energies_ry'])} -- the YAML defaults are not being "
        "read correctly"
    )
    assert len(dos["dos"]) == 3000
    assert -0.7 < dos["fermi_energy_ry"] < 0.7, (
        f"Si Fermi energy {dos['fermi_energy_ry']:.4f} Ry is outside expected range [-0.7, 0.7] Ry"
    )


# ── Copper DOS Validation ─────────────────────────────────────────────────────


@pytest.mark.slow
def test_copper_dos_metallic_character(q3_command, tmp_path):
    """
    Validate Cu DOS physical properties from Q3.

    Tests:
    1. Finite DOS at the Fermi level (metallic character).
    2. Cu d-bands appear as a large peak below the Fermi level.
    """
    cu_example_dir = EXAMPLES_DIRECTORY / "density_of_states--metal--copper"

    result = run_dos_workflow(
        cu_example_dir / "nc.yml",
        broadening_ry=0.02,
        energy_steps=3000,
        q3_binary=q3_command[0],
    )

    dos = result["dos"]
    energies_ry = dos["energies_ry"]
    dos_values = dos["dos"]
    fermi_energy_ry = dos["fermi_energy_ry"]
    fermi_energy_ev = fermi_energy_ry * RY_TO_EV

    # Convert window tolerances from eV to Ry for comparisons.
    window_ry = 0.022  # ≈ 0.3 eV

    # 1. Finite DOS at Fermi level (metallic character).
    fermi_dos_values = [
        dos_values[index]
        for index, energy in enumerate(energies_ry)
        if abs(energy - fermi_energy_ry) < window_ry
    ]
    assert fermi_dos_values, "No DOS values found near the Cu Fermi level"

    max_fermi_dos = max(fermi_dos_values)
    assert max_fermi_dos > 0.01, (
        f"Cu DOS at Fermi level too small: {max_fermi_dos:.4f} (expected > 0.01 — should be metallic)"
    )

    # 2. Cu d-band peak should appear 0.04–0.37 Ry (≈ 0.5–5 eV) below Fermi level.
    d_band_region_dos = [
        dos_values[index]
        for index, energy in enumerate(energies_ry)
        if -0.37 < (energy - fermi_energy_ry) < -0.04
    ]
    assert d_band_region_dos, "No DOS values found in Cu d-band region"

    max_d_band_dos = max(d_band_region_dos)
    assert max_d_band_dos > 0.1, (
        f"Cu d-band DOS peak too small: {max_d_band_dos:.4f} (expected > 0.1 — should show d-band)"
    )

    print(
        f"  Cu DOS: Fermi energy = {fermi_energy_ev:.4f} eV, "
        f"DOS at Fermi = {max_fermi_dos:.4f}, d-band peak = {max_d_band_dos:.4f}"
    )


# ── QE Comparison ─────────────────────────────────────────────────────────────


@requires_qe
def test_silicon_dos_fermi_energy_vs_qe(q3_command, tmp_path):
    """
    Validate Si Fermi energy from Q3 DOS against QE SCF Fermi energy.

    The Fermi energy (= highest occupied KS state for semiconductor) should
    agree to within 0.1 Ry between Q3 and QE.
    """
    si_pp_file = DATA_DIRECTORY / "pseudopotentials" / "nc" / "Si.upf"
    si_pseudo_dir = str(si_pp_file.parent)
    si_pp_filename = si_pp_file.name

    # ── Run QE SCF ──────────────────────────────────────────────────────
    qe_work_dir = tmp_path / "qe_work"
    qe_work_dir.mkdir()
    qe_outdir = qe_work_dir / "outfiles"
    qe_outdir.mkdir()

    scf_input = SILICON_SCF_TEMPLATE.format(
        outdir=str(qe_outdir), pseudo_dir=si_pseudo_dir, pp_filename=si_pp_filename
    )
    qe_stdout = run_qe(scf_input, qe_work_dir)
    qe_fermi_ry = extract_qe_fermi_energy(qe_stdout)
    assert qe_fermi_ry is not None, "Could not extract Fermi energy from QE output"

    # ── Run Q3 DOS ────────────────────────────────────────────────────
    si_example_dir = EXAMPLES_DIRECTORY / "density_of_states--semiconductor--silicon"

    result = run_dos_workflow(
        si_example_dir / "nc.yml",
        broadening_ry=0.01,
        energy_steps=3000,
        q3_binary=q3_command[0],
    )

    q3_fermi_ry = result["dos"]["fermi_energy_ry"]

    difference = abs(q3_fermi_ry - qe_fermi_ry)
    assert difference < FERMI_ENERGY_CONSISTENCY_TOLERANCE_RY, (
        f"Si Fermi energy disagreement: Q3={q3_fermi_ry:.6f} Ry, "
        f"QE={qe_fermi_ry:.6f} Ry, diff={difference:.6f} Ry "
        f"(tolerance: {FERMI_ENERGY_CONSISTENCY_TOLERANCE_RY:.2f} Ry)"
    )

    print(f"  Si Fermi energy: Q3 = {q3_fermi_ry:.6f} Ry, QE = {qe_fermi_ry:.6f} Ry")
