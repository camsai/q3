"""
Validation test for the Makov-Payne isolated-system correction.

Overlays control.isolated_system_correction: makov-payne onto the existing
charged-cell example (examples/total_energy--semiconductor--silicon/nc_charged.yml)
and checks the correction against a QE reference obtained independently for this
exact system (FCC Si, celldm(1)=10.3344 bohr, tot_charge=+1.0, PBE, NC
pseudopotential): QE prints 1st order = 0.27948401 Ry, 2nd order = -0.91595649 Ry
(see TODOs.md's Makov-Payne write-up).
"""

from helpers import EXAMPLES_DIRECTORY, apply_config_overlay, run_q3_and_load_results

SILICON_EXAMPLE_DIRECTORY = EXAMPLES_DIRECTORY / "total_energy--semiconductor--silicon"
NC_CHARGED_INPUT = SILICON_EXAMPLE_DIRECTORY / "nc_charged.yml"

# QE reference values for this exact system (see module docstring). The 1st-order
# (monopole) term depends only on the net charge and lattice geometry -- no cross-code
# SCF-density or ionic-gauge ambiguity -- so Q3 reproduces it to 8 significant figures.
# The 2nd-order (dipole+quadrupole) term additionally depends on the converged
# electronic density's own moments and on the (QE-unwrapped) ionic-position gauge, so
# small cross-code differences are physically expected there; Q3 agrees with QE to
# ~0.08% (0.0007 Ry) for this system.
QE_FIRST_ORDER_RYDBERG = 0.27948401
QE_SECOND_ORDER_RYDBERG = -0.91595649
FIRST_ORDER_TOLERANCE_RYDBERG = 1.0e-5
SECOND_ORDER_TOLERANCE_RYDBERG = 1.0e-3


def test_makov_payne_correction_matches_qe_reference(q3_command, tmp_path):
    overlay = {"control": {"isolated_system_correction": "makov-payne"}}
    input_path = apply_config_overlay(NC_CHARGED_INPUT, overlay, tmp_path)
    results, _ = run_q3_and_load_results(
        q3_command, input_path, "si_nc_charged", working_directory=str(tmp_path)
    )

    correction = results.get("isolated_system_correction")
    assert correction is not None, (
        "results.json is missing the isolated_system_correction object -- "
        "control.isolated_system_correction: makov-payne should have populated it"
    )
    assert correction["method"] == "makov-payne"

    first_order = float(correction["first_order_ry"])
    second_order = float(correction["second_order_ry"])
    corrected_total = float(correction["corrected_total_energy_ry"])
    total_energy = float(results["total_energy_ry"])

    assert abs(first_order - QE_FIRST_ORDER_RYDBERG) < FIRST_ORDER_TOLERANCE_RYDBERG, (
        f"1st order (monopole) term = {first_order:.8f} Ry, "
        f"expected {QE_FIRST_ORDER_RYDBERG:.8f} Ry (QE reference)"
    )
    assert abs(second_order - QE_SECOND_ORDER_RYDBERG) < SECOND_ORDER_TOLERANCE_RYDBERG, (
        f"2nd order (dipole+quadrupole) term = {second_order:.8f} Ry, "
        f"expected {QE_SECOND_ORDER_RYDBERG:.8f} Ry (QE reference, ~1e-3 Ry tolerance "
        f"for expected cross-code SCF/ionic-gauge differences)"
    )
    assert abs(float(correction["total_ry"]) - (first_order + second_order)) < 1.0e-12
    assert abs(corrected_total - (total_energy + first_order + second_order)) < 1.0e-9, (
        "isolated_system_correction.corrected_total_energy_ry must equal "
        "total_energy_ry + first_order + second_order"
    )


def test_makov_payne_not_applied_without_opt_in(q3_command, tmp_path):
    """Without control.isolated_system_correction, no correction object is written."""
    input_path = apply_config_overlay(NC_CHARGED_INPUT, {}, tmp_path)
    results, _ = run_q3_and_load_results(
        q3_command, input_path, "si_nc_charged", working_directory=str(tmp_path)
    )
    assert "isolated_system_correction" not in results
