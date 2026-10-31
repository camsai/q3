"""
Tier-2 (slow) full SCF convergence tests for 3x3x3 supercells.

These tests run the full SCF loop to convergence for large supercells,
verifying robustness, eigensolver stability, and final energies.
They are marked 'slow' and only run when '--tier2' is passed.
"""

import pytest

from helpers import (
    EXAMPLES_DIRECTORY,
    apply_config_overlay,
    run_q3_and_load_results,
)

FULL_SCF_OVERLAY = {
    "method": {
        "scf": {
            "max_iterations": 100,
        },
    },
}

# ── Converged Reference Energies (Ry) ─────────────────────────────────
#
# Generated: 2026-06-23
# Note: Silicon examples use 130 bands (108 occupied + 20% empty) to
# ensure Davidson subspace stability at the Gamma point.

SUPERCELL_FULL_SCF_REFERENCE_DATA = {
    "si_nc_3x3x3": {
        "example_path": EXAMPLES_DIRECTORY
        / "total_energy--semiconductor--silicon"
        / "nc_3x3x3.yml",
        "q3_energy": -455.773125931,
        "tolerance": 1e-4,
        "prefix": "si_nc_3x3x3",
    },
    "si_us_3x3x3": {
        "example_path": EXAMPLES_DIRECTORY
        / "total_energy--semiconductor--silicon"
        / "us_3x3x3.yml",
        "q3_energy": -423.494514158,
        "tolerance": 1e-4,
        "prefix": "si_us_3x3x3",
    },
    "cu_nc_3x3x3": {
        "example_path": EXAMPLES_DIRECTORY / "total_energy--metal--copper" / "nc_3x3x3.yml",
        "q3_energy": -10438.575344,
        "tolerance": 1e-3,  # Metals with smearing can have slightly more variance
        "prefix": "cu_nc_3x3x3",
    },
    "cu_us_3x3x3": {
        "example_path": EXAMPLES_DIRECTORY / "total_energy--metal--copper" / "us_3x3x3.yml",
        "q3_energy": -10902.864098,
        "tolerance": 1e-3,
        "prefix": "cu_us_3x3x3",
    },
}


# ── Tests ────────────────────────────────────────────────────────────


@pytest.mark.slow
@pytest.mark.parametrize(
    "system_key",
    list(SUPERCELL_FULL_SCF_REFERENCE_DATA.keys()),
    ids=list(SUPERCELL_FULL_SCF_REFERENCE_DATA.keys()),
)
def test_supercell_full_scf(q3_command, system_key, tmp_path):
    """Run a supercell example to full SCF convergence and validate."""
    reference = SUPERCELL_FULL_SCF_REFERENCE_DATA[system_key]

    # Run the full example YAML with overriden max_iterations to allow convergence
    merged_yaml = apply_config_overlay(
        reference["example_path"],
        FULL_SCF_OVERLAY,
        tmp_path,
    )

    results, _ = run_q3_and_load_results(
        q3_command,
        merged_yaml,
        reference["prefix"],
        working_directory=tmp_path,
        timeout=1800,  # Allow up to 30 minutes for slow supercells
    )

    assert results.runtime_log.converged is True, f"SCF failed to converge for {system_key}"

    computed_energy = results.properties.total_energy.value
    expected_energy = reference["q3_energy"]

    difference = abs(computed_energy - expected_energy)
    assert difference < reference["tolerance"], (
        f"Full SCF energy for {system_key}: {computed_energy:.6f} Ry "
        f"differs from reference {expected_energy:.6f} Ry "
        f"by {difference:.6f} Ry (tolerance: {reference['tolerance']} Ry)"
    )
