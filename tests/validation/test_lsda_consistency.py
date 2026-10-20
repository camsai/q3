"""
LSDA internal-consistency tests (machine-validated, no reference code needed).

For a non-magnetic system, a collinear spin-polarized (LSDA) calculation must
converge to the SAME total energy as the unpolarized calculation, regardless of
the starting magnetization: at zero moment the spin-resolved functional reduces
exactly to the unpolarized one.

These tests guard the LSDA accounting fixed in 2026-07: per-spin occupied band
counts, per-channel k-point weights, the magnetization channel in the SCF
convergence metric, and (total, magnetization)-representation density mixing.

Run:
    .venv/bin/pytest tests/validation/test_lsda_consistency.py -v --q3-bin ./build/q3
"""

import pytest

from helpers import (
    EXAMPLES_DIRECTORY,
    apply_config_overlay,
    run_overlay_and_read_energy,
    run_q3_and_load_results,
    total_pressure_kbar,
)

SI_NC_INPUT = EXAMPLES_DIRECTORY / "total_energy--semiconductor--silicon" / "nc.yml"
SI_NC_PREFIX = "si_nc"

LSDA_CONSISTENCY_TOLERANCE_RYDBERG = 1e-6
"""
Zero-moment LSDA vs unpolarized agreement bound.

The two paths differ only by floating-point noise (different loop structure and
the LibXC polarized-branch evaluation at equal spins); measured agreement is
~1e-9 Ry on Si. 1e-6 Ry leaves margin while still catching any real accounting
error (the historical bugs produced 5 mRy - 9 Ry discrepancies).
"""

LSDA_STRESS_CONSISTENCY_TOLERANCE_KBAR = 0.1
"""
Zero-moment LSDA vs unpolarized total-pressure agreement bound.

Guards the spin-polarized GGA gradient stress term (ExchangeCorrelationStress-
TensorCalculator::compute_gga_gradient_stress_polarized): at zero moment, ρ↑=ρ↓
and the three-sigma polarized functional derivative must reduce exactly to the
unpolarized one, so the two total-pressure values should agree to numerical
noise, same as the energy check above.
"""


def lsda_overlay(starting_magnetization):
    return {
        "method": {
            "scf": {
                "number_of_spin_components": 2,
                "starting_magnetization": {"Si": starting_magnetization},
            }
        }
    }


@pytest.fixture(scope="module")
def unpolarized_energy(q3_command, tmp_path_factory):
    run_directory = tmp_path_factory.mktemp("lsda_ref_unpolarized")
    return run_overlay_and_read_energy(q3_command, SI_NC_INPUT, None, SI_NC_PREFIX, run_directory)


@pytest.fixture(scope="module")
def unpolarized_pressure(q3_command, tmp_path_factory):
    run_directory = tmp_path_factory.mktemp("lsda_ref_unpolarized_stress")
    merged_input = apply_config_overlay(SI_NC_INPUT, None, run_directory)
    results, _ = run_q3_and_load_results(
        q3_command, merged_input, SI_NC_PREFIX, working_directory=run_directory
    )
    return total_pressure_kbar(results)


@pytest.fixture(scope="module")
def unpolarized_kpoint_weights(q3_command, tmp_path_factory):
    run_directory = tmp_path_factory.mktemp("lsda_ref_unpolarized_kpoint_weights")
    merged_input = apply_config_overlay(SI_NC_INPUT, None, run_directory)
    results, _ = run_q3_and_load_results(
        q3_command, merged_input, SI_NC_PREFIX, working_directory=run_directory
    )
    return results["kpoint_weights"]


@pytest.mark.parametrize("starting_magnetization", [0.0, 0.3])
def test_lsda_matches_unpolarized_for_nonmagnetic_system(
    q3_command, unpolarized_energy, tmp_path, starting_magnetization
):
    """Si LSDA total energy equals the unpolarized result for any starting moment."""
    lsda_energy = run_overlay_and_read_energy(
        q3_command, SI_NC_INPUT, lsda_overlay(starting_magnetization), SI_NC_PREFIX, tmp_path
    )
    difference = abs(lsda_energy - unpolarized_energy)
    assert difference < LSDA_CONSISTENCY_TOLERANCE_RYDBERG, (
        f"LSDA/unpolarized consistency violated (m0={starting_magnetization}):\n"
        f"  E(unpolarized) = {unpolarized_energy:.12f} Ry\n"
        f"  E(LSDA)        = {lsda_energy:.12f} Ry\n"
        f"  |diff|         = {difference:.2e} Ry"
        f"  (threshold: {LSDA_CONSISTENCY_TOLERANCE_RYDBERG:.0e} Ry)"
    )


@pytest.mark.parametrize("starting_magnetization", [0.0, 0.3])
def test_lsda_stress_matches_unpolarized_for_nonmagnetic_system(
    q3_command, unpolarized_pressure, tmp_path, starting_magnetization
):
    """Si LSDA total pressure (GGA gradient stress included) equals the unpolarized result."""
    merged_input = apply_config_overlay(SI_NC_INPUT, lsda_overlay(starting_magnetization), tmp_path)
    results, _ = run_q3_and_load_results(
        q3_command, merged_input, SI_NC_PREFIX, working_directory=tmp_path
    )
    lsda_pressure = total_pressure_kbar(results)
    difference = abs(lsda_pressure - unpolarized_pressure)
    assert difference < LSDA_STRESS_CONSISTENCY_TOLERANCE_KBAR, (
        f"LSDA/unpolarized stress consistency violated (m0={starting_magnetization}):\n"
        f"  P(unpolarized) = {unpolarized_pressure:.4f} kbar\n"
        f"  P(LSDA)        = {lsda_pressure:.4f} kbar\n"
        f"  |diff|         = {difference:.2e} kbar"
        f"  (threshold: {LSDA_STRESS_CONSISTENCY_TOLERANCE_KBAR:.2f} kbar)"
    )


def test_lsda_kpoint_weights_repeat_across_spin_channels(
    q3_command, unpolarized_kpoint_weights, tmp_path
):
    """LSDA results.json kpoint_weights are the (halved) unpolarized weights, tiled per
    spin channel.

    Regression test for a k-point STATE vs PHYSICAL-k-point index bug in FlavorResult
    reporting: kpoint_weights[state_index] indexed the physical-only KPointsGrid (size
    N) directly with the spin-doubled state index (size 2N for LSDA), reading
    out-of-bounds memory for the second spin channel -- observed in practice as
    subnormal garbage (e.g. 5e-324) and weights summing to 1.0 instead of 2.0. Fixed by
    mapping state_index -> state_index % N before indexing kpoint_grid(), matching the
    convention already used correctly in the charge-density accumulator and Fermi-energy
    bisection.

    Each physical weight is halved for LSDA (build_global_kpoint_grid_from_input's
    spin-degeneracy scaling: the two spin channels of one physical k-point together
    reconstruct its unpolarized weight), so the expected pattern is the halved
    unpolarized weights repeated once per spin channel, not the raw weights repeated.
    """
    merged_input = apply_config_overlay(SI_NC_INPUT, lsda_overlay(0.3), tmp_path)
    results, _ = run_q3_and_load_results(
        q3_command, merged_input, SI_NC_PREFIX, working_directory=tmp_path
    )
    weights = results["kpoint_weights"]
    halved_unpolarized_weights = [w * 0.5 for w in unpolarized_kpoint_weights]

    assert len(weights) == 2 * len(unpolarized_kpoint_weights)
    assert weights == pytest.approx(halved_unpolarized_weights * 2)
    assert sum(weights) == pytest.approx(2.0)
