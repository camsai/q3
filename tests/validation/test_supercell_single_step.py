"""
Supercell single-step validation tests for q3.

Runs each 3×3×3 supercell example for exactly 1 SCF iteration and validates
that the resulting energy falls within an expected range. This catches
regressions in initialization, Hamiltonian construction, and the first
diagonalization without the cost of full SCF convergence.

The energy ranges account for non-determinism in wavefunction initialization
(random seed in Davidson). QE LibXC reference values are documented for
cross-code comparison but not actively tested, since 1-step energies depend
on wavefunction initialization which differs between codes.
"""

import pytest

from helpers import (
    EXAMPLES_DIRECTORY,
    apply_config_overlay,
    run_q3_and_load_results,
)

# ── Reference energy ranges (Ry) after 1 SCF iteration ──────────────
#
# Each system specifies [min_energy, max_energy] that the single-step
# total energy must fall within. Ranges are chosen to be wide enough
# for Davidson random-seed variation but narrow enough to catch real
# regressions (typically ~1% of total energy or less).
#
# QE values are documented for cross-code comparison but not asserted.
#
# Generated: 2026-06-24, Q3 commit b3f1721

SUPERCELL_REFERENCE_DATA = {
    "si_nc_3x3x3": {
        "example_path": EXAMPLES_DIRECTORY
        / "total_energy--semiconductor--silicon"
        / "nc_3x3x3.yml",
        "min_energy": -457.0,
        "max_energy": -454.0,
        "qe_energy": -455.70234790,
        "prefix": "si_nc_3x3x3",
    },
    "si_us_3x3x3": {
        "example_path": EXAMPLES_DIRECTORY
        / "total_energy--semiconductor--silicon"
        / "us_3x3x3.yml",
        "min_energy": -425.0,
        "max_energy": -422.0,
        "qe_energy": -423.43497692,
        "prefix": "si_us_3x3x3",
    },
    "cu_nc_3x3x3": {
        "example_path": EXAMPLES_DIRECTORY / "total_energy--metal--copper" / "nc_3x3x3.yml",
        "min_energy": -10441.0,
        "max_energy": -10437.0,
        "qe_energy": -10439.16006813,
        "prefix": "cu_nc_3x3x3",
    },
    "cu_us_3x3x3": {
        "example_path": EXAMPLES_DIRECTORY / "total_energy--metal--copper" / "us_3x3x3.yml",
        # Range accommodates Davidson random-seed variance (Kerker + ndim=32 config).
        # Observed values: ~-10887 Ry (Q3), -10883.89 Ry (QE reference).
        # Wide range needed because metallic systems have high initialization sensitivity.
        "min_energy": -10892.0,
        "max_energy": -10875.0,
        "qe_energy": -10883.89241692,
        "prefix": "cu_us_3x3x3",
    },
}

SINGLE_STEP_OVERLAY = {
    "method": {
        "scf": {
            "max_iterations": 1,
        },
    },
}


# ── Tests ────────────────────────────────────────────────────────────


@pytest.mark.slow
@pytest.mark.parametrize(
    "system_key",
    list(SUPERCELL_REFERENCE_DATA.keys()),
    ids=list(SUPERCELL_REFERENCE_DATA.keys()),
)
def test_supercell_single_step(q3_command, system_key, tmp_path):
    """Run a supercell example for 1 SCF iteration and validate the energy."""
    reference = SUPERCELL_REFERENCE_DATA[system_key]

    merged_yaml = apply_config_overlay(
        reference["example_path"],
        SINGLE_STEP_OVERLAY,
        tmp_path,
    )

    results, _ = run_q3_and_load_results(
        q3_command,
        merged_yaml,
        reference["prefix"],
        working_directory=tmp_path,
        timeout=300,
    )

    computed_energy = results["total_energy_ry"]
    min_energy = reference["min_energy"]
    max_energy = reference["max_energy"]

    assert min_energy <= computed_energy <= max_energy, (
        f"Single-step energy for {system_key}: {computed_energy:.6f} Ry "
        f"is outside the expected range [{min_energy:.6f}, {max_energy:.6f}] Ry\n"
        f"QE reference (for comparison): {reference['qe_energy']} Ry"
    )
